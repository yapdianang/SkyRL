"""Shared helper utilities for TinkerEngine backends."""

import math
import time
from contextlib import contextmanager

import numpy as np

from skyrl.utils.log import logger


@contextmanager
def log_timing(request: str):
    """Context manager to log execution time for a request."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.info(f"(timing) {request} took {elapsed:.3f}s")


def pad(xs, pad_to: int, *, fill):
    """Pad a list to a specified length with a fill value."""
    return xs + ([fill] * (pad_to - len(xs)))


def convert_vllm_prompt_logprobs(
    prompt_token_ids: list[int],
    raw_prompt_logprobs: list[dict[str, dict] | None] | None,
    topk: int = 0,
) -> tuple[list[float | None] | None, list[list[tuple[int, float]] | None] | None]:
    """Convert vLLM prompt logprobs into the Tinker response shape.

    vLLM returns one entry per prompt token, each a
    ``{str(token_id): {"logprob": float, ...}}`` dict (``None`` at position 0,
    which has no preceding context). Tinker returns a flat list of the prompt
    tokens' own logprobs plus, when ``topk > 0``, a list of ``(token_id,
    logprob)`` pairs per position.

    Args:
        prompt_token_ids: The prompt tokens the logprobs were computed for.
        raw_prompt_logprobs: vLLM's per-position logprob dicts, or None.
        topk: Number of top entries to return per position (0 disables).

    Returns:
        ``(prompt_logprobs, topk_prompt_logprobs)``. Both are None when
        ``raw_prompt_logprobs`` is None; the second is also None when
        ``topk <= 0``.
    """
    if raw_prompt_logprobs is None:
        return None, None

    prompt_logprobs: list[float | None] = [
        (pos_dict.get(str(tid)) or {}).get("logprob") if pos_dict is not None else None
        for tid, pos_dict in zip(prompt_token_ids, raw_prompt_logprobs)
    ]

    if topk <= 0:
        return prompt_logprobs, None

    # vLLM returns k or k+1 logprobs per position (the extra entry is the prompt
    # token when it falls outside the top-k). Tinker returns exactly top-k, so
    # sort by logprob and truncate.
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] = [
        (
            sorted(
                [(int(tid), entry["logprob"]) for tid, entry in pos_dict.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:topk]
            if pos_dict is not None
            else None
        )
        for pos_dict in raw_prompt_logprobs[: len(prompt_token_ids)]
    ]
    return prompt_logprobs, topk_prompt_logprobs


def convert_vllm_decode_logprobs(
    token_ids: list[int],
    token_logprobs: list[float],
    raw_top_logprobs: list[dict[str, float]] | None,
    topk: int,
) -> list[list[tuple[int, float]]] | None:
    """Keep exactly K decode candidates, excluding an out-of-head sampled token."""
    if not topk:
        return None
    if raw_top_logprobs is None or len(raw_top_logprobs) != len(token_ids):
        raise ValueError("decode topk_logprobs must align with generated tokens")
    if len(token_logprobs) != len(token_ids) or any(not math.isfinite(lp) for lp in token_logprobs):
        raise ValueError("decode topk_logprobs requires finite sampled-token logprobs")
    result = []
    for row in raw_top_logprobs:
        if not row or len(row) < topk:
            raise ValueError("vLLM returned fewer decode logprobs than requested")
        candidates = []
        for token, logprob in row.items():
            if not token.startswith("token_id:") or not token[9:].isdigit():
                raise ValueError("decode topk_logprobs requires numeric vLLM token IDs")
            if not math.isfinite(logprob):
                raise ValueError("vLLM returned nonfinite decode logprobs")
            candidates.append((int(token[9:]), logprob))
        # The completions API includes the sampled token even when its rank exceeds K.
        result.append(sorted(candidates, key=lambda item: item[1], reverse=True)[:topk])
    return result


COMPARISON_PAD_LOGPROB = -1000.0


def split_comparison_carriers(
    raw_top_logprobs: list[dict[str, float]] | None,
) -> tuple[list[dict[str, float]] | None, list[list[int]] | None]:
    """Separate each position's decode logprobs from its comparison draws.

    Under SKYRL_STABILIZED_COMPARISONS, vLLM appends one carrier entry per draw whose value is the drawn
    token id + 1; real logprobs are never positive.
    """
    if raw_top_logprobs is None:
        return None, None
    logprobs, draws = [], []
    for row in raw_top_logprobs:
        logprobs.append({token: value for token, value in (row or {}).items() if value <= 0})
        row_draws = [value - 1 for value in (row or {}).values() if value > 0]
        if any(draw != int(draw) for draw in row_draws):
            raise ValueError("comparison carriers must hold integer token IDs")
        draws.append([int(draw) for draw in row_draws])
    return logprobs, draws


def convert_comparison_heads(draws, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Heads [n, k] of each position's k comparison draws: the distinct drawn ids and log(count / k).

    Columns after the distinct ids hold the smallest ids not drawn at COMPARISON_PAD_LOGPROB, whose
    float32 probability is 0.
    """
    draws = np.sort(np.asarray(draws, dtype=np.int64).reshape(-1, k), axis=1)
    n = len(draws)
    first = np.ones_like(draws, dtype=bool)
    first[:, 1:] = draws[:, 1:] != draws[:, :-1]
    # Slot r of a row holds its r-th distinct id; a row has distinct[row] of them.
    slot = np.cumsum(first, axis=1) - 1
    distinct = slot[:, -1] + 1
    flat = (np.arange(n)[:, None] * k + slot).ravel()
    counts = np.bincount(flat, minlength=n * k).reshape(n, k)
    ids = np.zeros((n, k), dtype=np.int64)
    ids.reshape(-1)[flat[first.ravel()]] = draws[first]
    # At most k ids are drawn, so the 2k smallest ids hold k that are not; rows drawing none of them pad with 0..k-1.
    unused = np.broadcast_to(np.arange(k), (n, k)).copy()
    low = (draws < 2 * k).any(axis=1)
    if low.any():
        candidates = np.arange(2 * k)
        drawn = (candidates[None, :, None] == draws[low][:, None, :]).any(-1)
        unused[low] = candidates[np.argsort(drawn, axis=1, kind="stable")[:, :k]]
    columns = np.arange(k)[None, :]
    pads = np.take_along_axis(unused, np.clip(columns - distinct[:, None], 0, k - 1), axis=1)
    has_draw = columns < distinct[:, None]
    ids = np.where(has_draw, ids, pads).astype(np.int32)
    logprobs = np.where(has_draw, np.log(np.maximum(counts, 1) / k), COMPARISON_PAD_LOGPROB).astype(np.float32)
    return ids, logprobs


def pad_batch(sequences: list[list], max_length: int, dtype) -> np.ndarray:
    """Pad a batch of sequences to max_length.

    Args:
        sequences: List of sequences to pad.
        max_length: Target length for all sequences.
        dtype: NumPy dtype for the output array.

    Returns:
        A NumPy array of shape (batch_size, max_length) with the padded sequences.
    """
    batch_size = len(sequences)
    padded = np.zeros((batch_size, max_length), dtype=dtype)
    for i, seq in enumerate(sequences):
        assert len(seq) <= max_length, f"Sequence length {len(seq)} exceeds max_length {max_length}"
        padded[i, : len(seq)] = seq
    return padded


def pad_to_fsdp(arr: np.ndarray, fsdp_size: int) -> np.ndarray:
    """Pad array's first dimension to be divisible by FSDP size."""
    batch_size = arr.shape[0]
    pad_size = (fsdp_size - batch_size % fsdp_size) % fsdp_size
    if pad_size == 0:
        return arr
    return np.pad(arr, [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1))
