"""Top-k logprobs and comparison draws from vLLM's samplers under SKYRL_STABILIZED_COMPARISONS, decoded as the API server does."""

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
from skyrl.backends.utils import (
    convert_comparison_heads,
    convert_vllm_decode_logprobs,
    split_comparison_carriers,
)

pytestmark = pytest.mark.vllm

K, TOP_K_HEADS = 16, 32
VOCAB = 64
LOGITS = torch.linspace(-3.0, 2.0, VOCAB)[torch.randperm(VOCAB, generator=torch.Generator().manual_seed(0))]
TEMPERATURE, TOP_K = 0.7, 3
# The law vLLM samples from: temperature, then the top-3 filter.
SCALED = LOGITS / TEMPERATURE
PROCESSED = torch.where(SCALED >= SCALED.topk(TOP_K).values[-1], SCALED, -torch.inf).softmax(-1)
SUPPORT = (PROCESSED > 0).numpy()
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
    return lambda: patch.apply_stabilized_comparisons_patch(K)


def metadata(rows: int, max_num_logprobs: int, temperature: float = TEMPERATURE, top_k: int | None = TOP_K):
    return SamplingMetadata(
        temperature=torch.full((rows,), temperature),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None if top_k is None else torch.full((rows,), top_k, dtype=torch.int32),
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


def completion_rows(sampled: torch.Tensor, tensors: LogprobsTensors, num_logprobs: int) -> list[dict]:
    """Per-position top_logprobs as vLLM's output processor and completions API return them."""
    positions = []
    for ids, logprobs, rank in zip(
        tensors.logprob_token_ids.tolist(), tensors.logprobs.tolist(), tensors.selected_token_ranks.tolist()
    ):
        append_logprobs_for_next_position(positions, ids, logprobs, itertools.repeat(None), rank, num_logprobs)
    for token, position in zip(sampled.tolist(), positions):
        assert next(iter(position)) == token  # the sampled token stays first
    return [{f"token_id:{token}": max(lp.logprob, -9999.0) for token, lp in row.items()} for row in positions]


def histograms(rows: list[dict]) -> np.ndarray:
    """[rows, vocab] comparison counts from the heads the API server records."""
    _, draws = split_comparison_carriers(rows)
    ids, logprobs = convert_comparison_heads(draws, K)
    counts = np.zeros((len(ids), VOCAB), dtype=np.int64)
    drawn = np.nonzero(logprobs > -1000)
    np.add.at(counts, (drawn[0], ids[drawn]), np.round(K * np.exp(logprobs[drawn])).astype(np.int64))
    assert (counts.sum(1) == K).all()
    return counts


def sample(rows: int, logprobs_mode: str, num_logprobs: int, seed: int = 0, **sampling):
    torch.manual_seed(seed)
    sampler = v1_sampler.Sampler(logprobs_mode=logprobs_mode)
    output = sampler(LOGITS.repeat(rows, 1), metadata(rows, num_logprobs, **sampling))
    return output.sampled_token_ids[:, 0].long(), output.logprobs_tensors


def chi_square(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(((observed - expected) ** 2 / expected).sum())


def test_v1_draws_follow_the_processed_law_independently_of_the_action(apply_patch):
    apply_patch()
    rows, num_logprobs = 4000, TOP_K_HEADS + K
    sampled, tensors = sample(rows, "processed_logprobs", num_logprobs)
    counts = histograms(completion_rows(sampled, tensors, num_logprobs))
    p = PROCESSED.double().numpy()[SUPPORT]
    actions = np.bincount(sampled.numpy(), minlength=VOCAB)
    assert not counts[:, ~SUPPORT].any() and not actions[~SUPPORT].any()
    assert chi_square(actions[SUPPORT], rows * p) < CHI2_999[2]

    # Draws pooled by sampled token, against independence and against the leave-in law that the probe refuted.
    observed = np.stack([counts[sampled.numpy() == a][:, SUPPORT].sum(0) for a in np.flatnonzero(SUPPORT)])
    independent = K * actions[SUPPORT][:, None] * p
    left_in = actions[SUPPORT][:, None] * ((K - 1) * p + np.eye(len(p)))
    assert chi_square(observed, independent) < CHI2_999[6]
    assert chi_square(observed, left_in) > 5 * CHI2_999[6]


def test_v1_top_k_heads_match_the_top_k_only_server(apply_patch):
    # The -sc2 server: unpatched vLLM, raw logprobs, top-32 heads of unmodified sampling.
    rows, unmodified = 256, {"temperature": 1.0, "top_k": None}
    plain_sampled, plain = sample(rows, "raw_logprobs", TOP_K_HEADS, **unmodified)
    apply_patch()
    sampled, dual = sample(rows, "processed_logprobs", TOP_K_HEADS + K, **unmodified)
    torch.testing.assert_close(sampled, plain_sampled)
    torch.testing.assert_close(dual.logprob_token_ids[:, : 1 + TOP_K_HEADS], plain.logprob_token_ids)
    torch.testing.assert_close(dual.logprobs[:, : 1 + TOP_K_HEADS], plain.logprobs)

    top_k_rows, draws = split_comparison_carriers(completion_rows(sampled, dual, TOP_K_HEADS + K))
    assert all(len(row) == K for row in draws)
    token_logprobs = plain.logprobs[:, 0].tolist()
    assert [row[f"token_id:{token}"] for token, row in zip(sampled.tolist(), top_k_rows)] == token_logprobs
    assert convert_vllm_decode_logprobs(
        sampled.tolist(), token_logprobs, top_k_rows, TOP_K_HEADS
    ) == convert_vllm_decode_logprobs(
        sampled.tolist(), token_logprobs, completion_rows(sampled, plain, TOP_K_HEADS), TOP_K_HEADS
    )
    # A request for one logprob in the same batch keeps the sampled token and the top-1.
    one = completion_rows(sampled, dual, 1)
    assert [list(row.values())[-1] for row in one] == plain.logprobs[:, 1].tolist()


def test_v1_prompt_logprobs_and_small_requests_keep_the_top_k(apply_patch):
    apply_patch()
    processed = LOGITS.log_softmax(-1).expand(4, -1)
    prompt = v1_sampler.Sampler.gather_logprobs(processed, 3, torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(prompt.logprob_token_ids[:, 1:].long(), processed.topk(3).indices)
    _, small = sample(4, "processed_logprobs", K)
    assert (small.logprobs <= 0).all() and small.logprob_token_ids.shape[1] == 1 + K


def test_samplers_require_processed_logprobs(apply_patch):
    apply_patch()
    with pytest.raises(ValueError, match="processed_logprobs"):
        v1_sampler.Sampler(logprobs_mode="raw_logprobs")


def test_v2_decode_scores_carry_top_k_then_draws(apply_patch):
    apply_patch()
    torch.manual_seed(0)
    rows, num_logprobs = 4000, TOP_K_HEADS + K
    processed = torch.where(torch.from_numpy(SUPPORT), SCALED, -torch.inf).repeat(rows, 1)
    sampled = torch.multinomial(PROCESSED, rows, replacement=True)
    tensors = v2_sampler.compute_topk_scores(processed, num_logprobs, sampled)
    plain = v2_reference_topk_scores(processed, TOP_K_HEADS, sampled)
    torch.testing.assert_close(tensors.logprob_token_ids[:, : 1 + TOP_K_HEADS], plain.logprob_token_ids)
    counts = histograms(completion_rows(sampled, tensors, num_logprobs))
    p = PROCESSED.double().numpy()[SUPPORT]
    assert not counts[:, ~SUPPORT].any()
    assert chi_square(counts.sum(0)[SUPPORT], rows * K * p) < CHI2_999[2]

    # Speculative rows and per-request logprob_token_ids keep the plain top-k.
    for kwargs in ({"cu_num_logits": [0, 4]}, {"max_per_req_token_ids": 2}):
        kept = v2_sampler.compute_topk_scores(processed[:4], num_logprobs, sampled[:4], **kwargs)
        torch.testing.assert_close(kept.logprob_token_ids[:, 1:], processed[:4].topk(num_logprobs).indices)
