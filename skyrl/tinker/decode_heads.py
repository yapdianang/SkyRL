"""Decode top-k heads recorded at sampling time and looked up by token sequence for score centering."""

import hashlib
import time
from collections import Counter, OrderedDict

import numpy as np

from skyrl.backends.utils import (
    convert_comparison_heads,
    convert_vllm_decode_logprobs,
    split_comparison_carriers,
)
from skyrl.utils.log import logger

TURN_ENDS_KEY = "score_centering_turn_ends"
# loss_fn_config keys: train on histograms of this many comparison draws (top-k heads when 0),
# with the target token as one of them when leave_in is 1.
COMPARISONS_KEY = "score_centering_comparisons"
LEAVE_IN_KEY = "score_centering_leave_in"
_COMPARISONS_ENTRY = "comparisons:"
_EVICTION_LOG_INTERVAL_SECONDS = 60.0


def hash_tokens(tokens: list[int]) -> str:
    """Same key as trajectory's ``stabilized_provenance.hash_tokens``."""
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def sampling_model(model_id: str, base_model: str | None) -> str:
    """The model a forwarded sample came from: the model id whose sampler weights served it, or the base model."""
    return model_id if model_id else f"base:{base_model}"


def subsampled_comparison_heads(
    key: str, draws: np.ndarray, targets: list[int], k: int, leave_in: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Heads of k draws per position: a uniform subset of the recorded i.i.d. draws, so again i.i.d.

    The subset is seeded by the entry key, so a repeated request gets the same heads. With leave_in,
    the target token is one of the k.
    """
    rows = np.random.default_rng(int(key[:16], 16)).permuted(draws, axis=1)[:, : k - leave_in]
    if leave_in:
        rows = np.concatenate([np.asarray(targets, dtype=draws.dtype)[:, None], rows], axis=1)
    return convert_comparison_heads(rows, k)


class DecodeHeadCache:
    """LRU map from (sampling model, hash(prompt + sampled tokens)) to (prompt_length, ids [n, k], logprobs [n, k]).

    A sample's comparison draws are a separate entry (prompt_length, draws [n, K] int32) whose hash has a prefix.
    forward_backward looks heads up under its own model id, so runs on different LoRA adapters of one service
    never read each other's heads; base-model samples are never served to a training model. A later sample of
    the same model and tokens replaces the earlier entry (last writer wins) and counts as an overwrite.
    Forwarding tasks write and forward_backward reads on the API event loop, so there is no lock.
    """

    def __init__(self, k: int, max_bytes: int, comparisons: int = 0):
        self.k = k
        # Comparison draws recorded per position next to the top-k heads (SKYRL_STABILIZED_COMPARISONS).
        self.comparisons = comparisons
        self.max_bytes = max_bytes
        self.nbytes = 0
        self.evictions = 0
        self.records: Counter[str] = Counter()
        self.overwrites: Counter[str] = Counter()
        self._entries: OrderedDict[tuple[str, str], tuple] = OrderedDict()
        self._last_eviction_log = float("-inf")
        self._last_overwrite_log = float("-inf")

    def _records(self, sampling_params) -> tuple[bool, bool]:
        """Top-k heads describe the sampling law only for unmodified sampling; draws come from the processed law."""
        sp = sampling_params
        top_k = sp.temperature == 1 and sp.top_p == 1 and sp.top_k == -1
        return top_k, bool(self.comparisons) and sp.temperature > 0

    def record_topk(self, sampling_params) -> int:
        """Decode logprobs to request from vLLM: the top-k heads, followed by the comparison draws if recorded."""
        top_k, comparisons = self._records(sampling_params)
        return self.k + self.comparisons if comparisons else self.k if top_k else 0

    def record(
        self, model: str, prompt_tokens: list[int], tokens: list[int], token_logprobs, raw_top_logprobs, sampling_params
    ):
        """A head that fails validation is not recorded; the sample still succeeds and training on it fails later."""
        if not tokens:
            return
        top_k, comparisons = self._records(sampling_params)
        try:
            raw_top_logprobs, draws = (
                split_comparison_carriers(raw_top_logprobs) if self.comparisons else (raw_top_logprobs, None)
            )
        except ValueError as e:
            logger.warning(f"Decode heads not recorded: {e}")
            return
        if top_k:
            try:
                heads = np.asarray(convert_vllm_decode_logprobs(tokens, token_logprobs, raw_top_logprobs, self.k))
                ids, logprobs = heads[..., 0].astype(np.int32), heads[..., 1].astype(np.float32)
                self.put(model, prompt_tokens + tokens, len(prompt_tokens), ids, logprobs)
            except ValueError as e:
                logger.warning(f"Top-k decode heads not recorded: {e}")
        if comparisons:
            try:
                if draws is None or len(draws) != len(tokens):
                    raise ValueError("comparison draws must align with generated tokens")
                if any(len(row) != self.comparisons for row in draws):
                    raise ValueError(f"vLLM returned other than {self.comparisons} comparison draws per token")
                draws = np.asarray(draws, dtype=np.int32).reshape(len(tokens), self.comparisons)
                self.put(model, prompt_tokens + tokens, len(prompt_tokens), draws, comparisons=True)
            except ValueError as e:
                logger.warning(f"Comparison heads not recorded: {e}")

    def put(
        self, model: str, tokens: list[int], prompt_length: int, *arrays: np.ndarray, comparisons: bool = False
    ) -> None:
        key = (model, (_COMPARISONS_ENTRY if comparisons else "") + hash_tokens(tokens))
        self.records[model] += 1
        if (old := self._entries.pop(key, None)) is not None:
            self.nbytes -= sum(array.nbytes for array in old[1:])
            self.overwrites[model] += 1
            now = time.monotonic()
            if now - self._last_overwrite_log >= _EVICTION_LOG_INTERVAL_SECONDS:
                self._last_overwrite_log = now
                logger.warning(
                    f"Decode head overwrites by model so far: {dict(self.overwrites)} of {dict(self.records)}"
                )
        self._entries[key] = (prompt_length, *arrays)
        self.nbytes += sum(array.nbytes for array in arrays)
        evicted = 0
        while self.nbytes > self.max_bytes:
            _, old = self._entries.popitem(last=False)
            self.nbytes -= sum(array.nbytes for array in old[1:])
            evicted += 1
        self.evictions += evicted
        now = time.monotonic()
        if evicted and now - self._last_eviction_log >= _EVICTION_LOG_INTERVAL_SECONDS:
            self._last_eviction_log = now
            logger.warning(
                f"Decode head cache evicted {self.evictions} entries in total "
                f"(cap {self.max_bytes} bytes, {len(self._entries)} entries held)"
            )

    def get(self, key: tuple[str, str]) -> tuple | None:
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def stats(self) -> dict:
        models = sorted(self.records)
        return {
            "models": {m: {"records": self.records[m], "overwrites": self.overwrites[m]} for m in models},
            "entries": len(self._entries),
            "bytes": self.nbytes,
            "evictions": self.evictions,
        }

    def place(
        self,
        model: str,
        full_tokens: list[int],
        turn_ends: list[int],
        weights: list[float],
        k: int,
        comparisons: int = 0,
        leave_in: bool = False,
    ) -> tuple[list[int], list[float]]:
        """Return flat [n, k] heads aligned with targets full_tokens[1:]; turn rows go to targets [start - 1, end - 1).

        ``comparisons`` selects histograms of that many recorded draws instead of the top-k heads.
        """
        if comparisons and not 1 <= comparisons <= self.comparisons:
            raise ValueError(f"{COMPARISONS_KEY}={comparisons} but the server records {self.comparisons} draws")
        if comparisons and k != comparisons:
            raise ValueError(f"score_centering_k={k} must equal {COMPARISONS_KEY}={comparisons}")
        if not comparisons and k > self.k:
            raise ValueError(f"score_centering_k={k} exceeds the recorded k={self.k}")
        n = len(full_tokens) - 1
        ids = np.zeros((n, k), dtype=np.int32)
        logprobs = np.zeros((n, k), dtype=np.float32)
        covered = np.zeros(n, dtype=bool)
        missing = []
        previous = 0
        for end in turn_ends:
            if not isinstance(end, int) or not previous < end <= len(full_tokens):
                raise ValueError(f"{TURN_ENDS_KEY} must be strictly increasing int64 positions, got {turn_ends}")
            previous = end
            key = hash_tokens(full_tokens[:end])
            entry = self.get((model, _COMPARISONS_ENTRY + key if comparisons else key))
            if entry is None:
                missing.append(key)
                continue
            if comparisons:
                start, draws = entry
                head_ids, head_logprobs = subsampled_comparison_heads(
                    key, draws, full_tokens[start:end], comparisons, leave_in
                )
            else:
                start, head_ids, head_logprobs = entry
            ids[start - 1 : end - 1] = head_ids[:, :k]
            logprobs[start - 1 : end - 1] = head_logprobs[:, :k]
            covered[start - 1 : end - 1] = True
        uncovered = np.flatnonzero((np.asarray(weights) > 0) & ~covered)
        if uncovered.size:
            cause = (
                f"no heads recorded for model {model} at hashes {missing}"
                if missing
                else f"not covered by {TURN_ENDS_KEY}"
            )
            raise ValueError(f"Positive-weight target positions {uncovered[:8].tolist()} have no decode head: {cause}")
        return ids.ravel().tolist(), logprobs.ravel().tolist()
