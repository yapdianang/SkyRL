"""Runtime patch: decode logprob columns carry K i.i.d. comparison draws instead of the top-k.

With ``SKYRL_STABILIZED_COMPARISONS=K``, the columns after the sampled token at each generated
position list every distinct drawn token other than the sampled token once, with log(count / K),
followed by copies of the sampled token with its own logprob. vLLM folds a position's columns into a
dict keyed by token id (``append_logprobs_for_next_position``), so a repeated id cannot be returned,
and a comparison column holding the sampled id would overwrite the sampled token's logprob. The
sampled token's count is therefore implicit: K minus the listed counts.

The draws are i.i.d. from the softmax of the tensor that the top-k would be taken from. Under the
required ``logprobs_mode=processed_logprobs``, that tensor is the one the sampled token was drawn from,
after temperature, min_p and top-k/top-p:

- V1 model runner: ``Sampler.gather_logprobs`` receives ``log_softmax`` of the logits that
  ``TopKTopPSampler.forward_native`` turns into the probabilities of ``random_sample``.
- V2 model runner: ``compute_topk_scores`` receives the ``processed_logits`` that ``gumbel_sample``
  sampled from.

The draws use torch's global generator after the token is sampled, so they are independent of it.
``SKYRL_STABILIZED_LEAVE_IN=1`` draws K - 1 tokens and counts the sampled token as the K-th draw.

Patched names (vLLM 0.30.0):

- ``vllm.v1.sample.sampler.Sampler.gather_logprobs``, only inside ``Sampler.forward``. Prompt logprobs
  and the rejection sampler call it outside ``forward`` and keep the top-k.
- ``vllm.v1.worker.gpu.sample.sampler.compute_topk_scores``, the binding the V2 decode sampler calls.
  The V2 prompt-logprob and rejection-sampler modules import their own binding and keep the top-k.
- ``__init__`` of both samplers, to reject any ``logprobs_mode`` other than ``processed_logprobs``.

A batch whose largest logprob request is below K, or that has speculative tokens or per-request
``logprob_token_ids``, keeps the top-k. The API server rejects such rows as comparison heads.
"""

import contextvars

import torch

from skyrl.env_vars import SKYRL_STABILIZED_COMPARISONS, SKYRL_STABILIZED_LEAVE_IN
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


def encode_comparisons(
    draws: torch.Tensor, sampled: torch.Tensor, sampled_logprobs: torch.Tensor, num_columns: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode [rows, n <= k] draws as [rows, num_columns] ids and logprobs, counts out of k.

    Distinct draws other than ``sampled`` come first with log(count / k); the remaining columns repeat
    ``sampled`` with ``sampled_logprobs``.
    """
    draws = draws.sort(-1).values
    counts = (draws[:, :, None] == draws[:, None, :]).sum(-1)
    listed = draws != sampled[:, None]
    listed[:, 1:] &= draws[:, 1:] != draws[:, :-1]
    order = torch.argsort((~listed).to(torch.uint8), dim=-1, stable=True)
    listed = listed.gather(-1, order)
    ids = torch.where(listed, draws.gather(-1, order), sampled[:, None])
    logprobs = torch.where(listed, (counts.gather(-1, order).float() / k).log(), sampled_logprobs[:, None])
    pad = num_columns - ids.shape[1]
    return (
        torch.cat((ids, sampled[:, None].expand(-1, pad)), dim=1),
        torch.cat((logprobs, sampled_logprobs[:, None].expand(-1, pad)), dim=1),
    )


def _with_comparisons(base, scores: torch.Tensor, sampled: torch.Tensor, num_columns: int, k: int, leave_in: bool):
    """Append comparison columns to LogprobsTensors that hold only the sampled-token column."""
    draws = draw_comparisons(scores, sampled, k - leave_in)
    ids, logprobs = encode_comparisons(draws, sampled, base.logprobs[:, 0], num_columns, k)
    return base._replace(
        logprob_token_ids=torch.cat((base.logprob_token_ids, ids.to(base.logprob_token_ids.dtype)), dim=1),
        logprobs=torch.cat((base.logprobs, logprobs), dim=1),
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


def _patch_v1(k: int, leave_in: bool) -> None:
    from vllm.v1.sample.sampler import Sampler

    forward, gather_logprobs = Sampler.forward, Sampler.gather_logprobs

    def patched_forward(self, *args, **kwargs):
        token = _DECODING.set(True)
        try:
            return forward(self, *args, **kwargs)
        finally:
            _DECODING.reset(token)

    def patched_gather_logprobs(logprobs, num_logprobs, token_ids):
        if not _DECODING.get() or num_logprobs < k:
            return gather_logprobs(logprobs, num_logprobs, token_ids)
        return _with_comparisons(
            gather_logprobs(logprobs, 0, token_ids), logprobs, token_ids, num_logprobs, k, leave_in
        )

    _require_processed_logprobs(Sampler)
    Sampler.forward = patched_forward
    Sampler.gather_logprobs = staticmethod(patched_gather_logprobs)


def _patch_v2(k: int, leave_in: bool) -> None:
    from vllm.v1.worker.gpu.sample import sampler

    compute_topk_scores = sampler.compute_topk_scores

    def patched_compute_topk_scores(logits, num_logprobs, sampled_token_ids, cu_num_logits=None, **kwargs):
        if num_logprobs < k or cu_num_logits is not None or kwargs.get("max_per_req_token_ids"):
            return compute_topk_scores(logits, num_logprobs, sampled_token_ids, cu_num_logits, **kwargs)
        base = compute_topk_scores(logits, 0, sampled_token_ids, cu_num_logits, **kwargs)
        return _with_comparisons(base, logits, sampled_token_ids, num_logprobs, k, leave_in)

    _require_processed_logprobs(sampler.Sampler)
    sampler.compute_topk_scores = patched_compute_topk_scores


def apply_stabilized_comparisons_patch(
    k: int = SKYRL_STABILIZED_COMPARISONS, leave_in: bool = SKYRL_STABILIZED_LEAVE_IN
) -> None:
    """Patch both vLLM model runners' decode logprobs; a no-op when k is 0. Idempotent per process."""
    global _PATCHED
    if _PATCHED or not k:
        return
    if k < 1 + leave_in:
        raise ValueError(f"SKYRL_STABILIZED_COMPARISONS={k} leaves no comparison draw")
    _patch_v1(k, leave_in)
    _patch_v2(k, leave_in)
    _PATCHED = True
    logger.info(f"Decode logprobs carry {k} comparison draws (leave_in={leave_in}) in vLLM V1 and V2 samplers")
