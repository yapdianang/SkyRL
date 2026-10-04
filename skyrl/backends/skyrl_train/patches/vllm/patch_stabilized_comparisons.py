"""Runtime patch: decode logprobs also carry K i.i.d. comparison draws from the processed sampling law.

With ``SKYRL_STABILIZED_COMPARISONS=K``, a batch whose largest decode logprob request ``n`` exceeds K
returns, per generated position, the sampled token, the top ``n - K`` tokens as usual, and then K
carrier columns, one per draw. vLLM folds a position's columns into a dict keyed by token id
(``append_logprobs_for_next_position``), which would merge a draw with a top-k column of the same id.
So a carrier's id is a filler id that is in no other column of its position, and its value is
``drawn_id + 1``: a positive integer, exact in float32 and kept by the completions API's -9999 floor,
while real logprobs are never positive. A request for at most ``n - K`` logprobs receives its usual
top-k from the same batch.

The draws are i.i.d. from the softmax of the tensor the top-k is taken from. Under the required
``logprobs_mode=processed_logprobs``, that tensor is the one the sampled token was drawn from, after
temperature, min_p and top-k/top-p:

- V1 model runner: ``Sampler.gather_logprobs`` receives ``log_softmax`` of the logits that
  ``TopKTopPSampler.forward_native`` turns into the probabilities of ``random_sample``.
- V2 model runner: ``compute_topk_scores`` receives the ``processed_logits`` that ``gumbel_sample``
  sampled from.

The draws use torch's global generator after the token is sampled, so they are independent of it.

Patched names (vLLM 0.30.0):

- ``vllm.v1.sample.sampler.Sampler.gather_logprobs``, only inside ``Sampler.forward``. Prompt logprobs
  and the rejection sampler call it outside ``forward`` and keep the top-k.
- ``vllm.v1.worker.gpu.sample.sampler.compute_topk_scores``, the binding the V2 decode sampler calls.
  The V2 prompt-logprob and rejection-sampler modules import their own binding and keep the top-k.
- ``__init__`` of both samplers, to reject any ``logprobs_mode`` other than ``processed_logprobs``.

Batches with speculative tokens or per-request ``logprob_token_ids`` keep the plain top-k; the API
server then finds no draws and does not record comparison heads.
"""

import contextvars

import torch

from skyrl.env_vars import SKYRL_STABILIZED_COMPARISONS
from skyrl.utils.log import logger

PROCESSED_LOGPROBS = "processed_logprobs"

_PATCHED = False
_DECODING = contextvars.ContextVar("skyrl_stabilized_decoding", default=False)


def draw_comparisons(scores: torch.Tensor, sampled: torch.Tensor, n: int) -> torch.Tensor:
    """[rows, n] i.i.d. draws from softmax(scores) per row."""
    probs = scores.float().softmax(-1).nan_to_num_(0.0)
    # A row without probability mass trips multinomial's device assert; it draws the sampled token instead.
    probs.scatter_add_(-1, sampled[:, None], (probs.sum(-1, keepdim=True) <= 0).to(probs.dtype))
    return torch.multinomial(probs, n, replacement=True)


def carrier_columns(draws: torch.Tensor, taken: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[rows, K] carrier ids and values for [rows, K] draws: filler ids absent from ``taken``, values draw + 1.

    ``taken`` holds each row's other column ids, at most ``c - K`` distinct of them for ``c`` candidate
    fillers ``0..c-1``, so at least K fillers are free.
    """
    k = draws.shape[1]
    candidates = torch.arange(taken.shape[1] + k, device=draws.device)
    used = (candidates[None, :, None] == taken[:, None, :]).any(-1)
    free_first = torch.argsort(used.to(torch.uint8), dim=-1, stable=True)[:, :k]
    return candidates[free_first], (draws + 1).float()


def _with_comparisons(base, scores: torch.Tensor, sampled: torch.Tensor, k: int):
    """Append K carrier columns to LogprobsTensors holding the sampled token and the top-k."""
    ids, values = carrier_columns(draw_comparisons(scores, sampled, k), base.logprob_token_ids.long())
    return base._replace(
        logprob_token_ids=torch.cat((base.logprob_token_ids, ids.to(base.logprob_token_ids.dtype)), dim=1),
        logprobs=torch.cat((base.logprobs, values.to(base.logprobs.dtype)), dim=1),
    )


def _require_processed_logprobs(sampler_cls) -> None:
    init = sampler_cls.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if self.logprobs_mode != PROCESSED_LOGPROBS:
            raise ValueError(
                f"SKYRL_STABILIZED_COMPARISONS draws from the processed sampling distribution and requires "
                f"logprobs_mode={PROCESSED_LOGPROBS!r}, got {self.logprobs_mode!r}"
            )

    sampler_cls.__init__ = __init__


def _patch_v1(k: int) -> None:
    from vllm.v1.sample.sampler import Sampler

    forward, gather_logprobs = Sampler.forward, Sampler.gather_logprobs

    def patched_forward(self, *args, **kwargs):
        token = _DECODING.set(True)
        try:
            return forward(self, *args, **kwargs)
        finally:
            _DECODING.reset(token)

    def patched_gather_logprobs(logprobs, num_logprobs, token_ids):
        if not _DECODING.get() or num_logprobs <= k:
            return gather_logprobs(logprobs, num_logprobs, token_ids)
        return _with_comparisons(gather_logprobs(logprobs, num_logprobs - k, token_ids), logprobs, token_ids, k)

    _require_processed_logprobs(Sampler)
    Sampler.forward = patched_forward
    Sampler.gather_logprobs = staticmethod(patched_gather_logprobs)


def _patch_v2(k: int) -> None:
    from vllm.v1.worker.gpu.sample import sampler

    compute_topk_scores = sampler.compute_topk_scores

    def patched_compute_topk_scores(logits, num_logprobs, sampled_token_ids, cu_num_logits=None, **kwargs):
        if num_logprobs <= k or cu_num_logits is not None or kwargs.get("max_per_req_token_ids"):
            return compute_topk_scores(logits, num_logprobs, sampled_token_ids, cu_num_logits, **kwargs)
        base = compute_topk_scores(logits, num_logprobs - k, sampled_token_ids, cu_num_logits, **kwargs)
        return _with_comparisons(base, logits, sampled_token_ids, k)

    _require_processed_logprobs(sampler.Sampler)
    sampler.compute_topk_scores = patched_compute_topk_scores


def apply_stabilized_comparisons_patch(k: int = SKYRL_STABILIZED_COMPARISONS) -> None:
    """Patch both vLLM model runners' decode logprobs; a no-op when k is 0. Idempotent per process."""
    global _PATCHED
    if _PATCHED or not k:
        return
    _patch_v1(k)
    _patch_v2(k)
    _PATCHED = True
    logger.info(f"Decode logprobs carry {k} comparison draws after the top-k in vLLM V1 and V2 samplers")
