"""Comparison draws from vLLM's samplers under SKYRL_STABILIZED_COMPARISONS, decoded as the API server does."""

import itertools

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from vllm.logprobs import append_logprobs_for_next_position
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample import sampler as v1_sampler
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu.sample import sampler as v2_sampler

from skyrl.backends.skyrl_train.patches.vllm import (
    patch_stabilized_comparisons as patch,
)
from skyrl.backends.utils import COMPARISON_PAD_LOGPROB, convert_vllm_comparison_heads

pytestmark = pytest.mark.vllm

K = 16
LOGITS = torch.tensor([2.0, 1.0, 0.5, -1.0, 0.2, 1.5])
TEMPERATURE, TOP_K = 0.7, 3
# The law vLLM samples from: temperature, then the top-3 filter.
SCALED = LOGITS / TEMPERATURE
PROCESSED = torch.where(SCALED >= SCALED.topk(TOP_K).values[-1], SCALED, -torch.inf).softmax(-1)
SUPPORT = PROCESSED > 0
# Chi-square 0.999 quantiles by degrees of freedom.
CHI2_999 = {2: 13.816, 6: 22.458}


def v2_reference_topk_scores(logits, num_logprobs, sampled_token_ids, cu_num_logits=None, **kwargs):
    """compute_topk_scores without its Triton kernels, for CPU."""
    logprobs = logits.float().log_softmax(-1)
    ids = torch.cat((sampled_token_ids[:, None], logprobs.topk(num_logprobs).indices), dim=1)
    sampled = logprobs.gather(-1, sampled_token_ids[:, None])
    return LogprobsTensors(ids, logprobs.gather(-1, ids), (logprobs >= sampled).sum(-1))


@pytest.fixture
def apply_patch(monkeypatch):
    """Patch for one test; monkeypatch restores every vLLM name afterwards."""
    for cls in (v1_sampler.Sampler, v2_sampler.Sampler):
        monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(v1_sampler.Sampler, "forward", v1_sampler.Sampler.forward)
    monkeypatch.setattr(v1_sampler.Sampler, "gather_logprobs", v1_sampler.Sampler.__dict__["gather_logprobs"])
    # Eager rank count: torch.compile of the original adds ~20 s on CPU and does not touch the columns.
    monkeypatch.setattr(v1_sampler, "batched_count_greater_than", lambda x, values: (x >= values).sum(-1))
    monkeypatch.setattr(v2_sampler, "compute_topk_scores", v2_reference_topk_scores)
    monkeypatch.setattr(patch, "_PATCHED", False)
    return lambda leave_in=False: patch.apply_stabilized_comparisons_patch(K, leave_in)


def metadata(rows: int, max_num_logprobs: int = K) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.full((rows,), TEMPERATURE),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=torch.full((rows,), TOP_K, dtype=torch.int32),
        generators={},
        max_num_logprobs=max_num_logprobs,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(rows),
        presence_penalties=torch.zeros(rows),
        repetition_penalties=torch.ones(rows),
        output_token_ids=[[] for _ in range(rows)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def completion_rows(sampled: torch.Tensor, tensors: LogprobsTensors, num_logprobs: int = K) -> list[dict]:
    """Per-position top_logprobs as vLLM's output processor and completions API return them."""
    positions = []
    for ids, logprobs, rank in zip(
        tensors.logprob_token_ids.tolist(), tensors.logprobs.tolist(), tensors.selected_token_ranks.tolist()
    ):
        append_logprobs_for_next_position(positions, ids, logprobs, itertools.repeat(None), rank, num_logprobs)
    for token, position in zip(sampled.tolist(), positions):
        assert next(iter(position)) == token  # the sampled token stays first
    return [{f"token_id:{token}": max(lp.logprob, -9999.0) for token, lp in row.items()} for row in positions]


def histograms(sampled: torch.Tensor, tensors: LogprobsTensors) -> np.ndarray:
    """[rows, vocab] comparison counts from the heads the API server records."""
    ids, logprobs = convert_vllm_comparison_heads(sampled.tolist(), completion_rows(sampled, tensors), K)
    counts = np.zeros((len(ids), len(LOGITS)), dtype=np.int64)
    drawn = logprobs > COMPARISON_PAD_LOGPROB
    np.add.at(counts, (np.nonzero(drawn)[0], ids[drawn]), np.round(K * np.exp(logprobs[drawn])).astype(np.int64))
    assert (counts.sum(1) == K).all()
    return counts


def sample(rows: int, seed: int = 0):
    torch.manual_seed(seed)
    output = v1_sampler.Sampler(logprobs_mode="processed_logprobs")(LOGITS.repeat(rows, 1), metadata(rows))
    return output.sampled_token_ids[:, 0].long(), output.logprobs_tensors


def chi_square(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(((observed - expected) ** 2 / expected).sum())


@pytest.mark.parametrize("leave_in", [False, True])
def test_v1_draws_follow_the_processed_law(apply_patch, leave_in):
    apply_patch(leave_in)
    rows = 4000
    sampled, tensors = sample(rows)
    counts = histograms(sampled, tensors)
    p = PROCESSED.double().numpy()[SUPPORT.numpy()]
    actions = np.bincount(sampled.numpy(), minlength=len(LOGITS))
    assert not counts[:, ~SUPPORT.numpy()].any() and not actions[~SUPPORT.numpy()].any()
    assert chi_square(actions[SUPPORT.numpy()], rows * p) < CHI2_999[2]

    # Draws pooled by sampled token, against the law each variant implies given that token.
    observed = np.stack([counts[sampled.numpy() == a][:, SUPPORT.numpy()].sum(0) for a in np.flatnonzero(SUPPORT)])
    independent = K * actions[SUPPORT.numpy()][:, None] * p
    left_in = actions[SUPPORT.numpy()][:, None] * ((K - 1) * p + np.eye(len(p)))
    implied, other = (left_in, independent) if leave_in else (independent, left_in)
    assert chi_square(observed, implied) < CHI2_999[6]
    assert chi_square(observed, other) > 5 * CHI2_999[6]


def test_v1_keeps_the_sampled_logprob_and_the_top_k_elsewhere(apply_patch):
    apply_patch()
    sampled, tensors = sample(64)
    lp = PROCESSED.log()
    rows = completion_rows(sampled, tensors)
    for token, row in zip(sampled.tolist(), rows):
        assert row[f"token_id:{token}"] == pytest.approx(lp[token].item(), abs=1e-6)
        assert len(row) <= 1 + K

    # Prompt logprobs gather outside Sampler.forward; a batch asking for fewer than K logprobs keeps the top-k.
    processed = lp.expand(4, -1)
    prompt = v1_sampler.Sampler.gather_logprobs(processed, 3, torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(prompt.logprob_token_ids[:, 1:].long(), processed.topk(3).indices)
    torch.manual_seed(0)
    small = v1_sampler.Sampler(logprobs_mode="processed_logprobs")(LOGITS.repeat(4, 1), metadata(4, 1))
    assert (small.logprobs_tensors.logprob_token_ids[:, 1] == PROCESSED.argmax()).all()


def test_samplers_require_processed_logprobs(apply_patch):
    apply_patch()
    with pytest.raises(ValueError, match="processed_logprobs"):
        v1_sampler.Sampler(logprobs_mode="raw_logprobs")


def test_v2_decode_scores_carry_comparisons(apply_patch):
    apply_patch()
    torch.manual_seed(0)
    processed = torch.where(SUPPORT, SCALED, -torch.inf).repeat(4000, 1)
    sampled = torch.multinomial(PROCESSED, 4000, replacement=True)
    tensors = v2_sampler.compute_topk_scores(processed, K, sampled)
    counts = histograms(sampled, tensors)
    p = PROCESSED.double().numpy()
    assert not counts[:, ~SUPPORT.numpy()].any()
    assert chi_square(counts.sum(0)[SUPPORT.numpy()], 4000 * K * p[SUPPORT.numpy()]) < CHI2_999[2]

    # Speculative rows and per-request logprob_token_ids keep the top-k.
    wide = torch.randn(4, 2 * K)
    for kwargs in ({"cu_num_logits": [0, 4]}, {"max_per_req_token_ids": 2}):
        kept = v2_sampler.compute_topk_scores(wide, K, sampled[:4], **kwargs)
        torch.testing.assert_close(kept.logprob_token_ids[:, 1:], wide.topk(K).indices)
