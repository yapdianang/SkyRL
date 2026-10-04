"""Decode top-k heads recorded at sampling time and looked up by token sequence for score centering."""

import hashlib
import time
from collections import Counter, OrderedDict

import numpy as np

from skyrl.backends.utils import (
    convert_vllm_comparison_heads,
    convert_vllm_decode_logprobs,
)
from skyrl.utils.log import logger

TURN_ENDS_KEY = "score_centering_turn_ends"
# loss_fn_config key: the number of comparison draws the client expects in each head, or 0 for top-k heads.
COMPARISONS_KEY = "score_centering_comparisons"
_EVICTION_LOG_INTERVAL_SECONDS = 60.0


def hash_tokens(tokens: list[int]) -> str:
    """Same key as trajectory's ``stabilized_provenance.hash_tokens``."""
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def sampling_model(model_id: str, base_model: str | None) -> str:
    """The model a forwarded sample came from: the model id whose sampler weights served it, or the base model."""
    return model_id if model_id else f"base:{base_model}"


class DecodeHeadCache:
    """LRU map from (sampling model, hash(prompt + sampled tokens)) to (prompt_length, ids [n, k], logprobs [n, k]).

    forward_backward looks heads up under its own model id, so runs on different LoRA adapters of one service
    never read each other's heads; base-model samples are never served to a training model. A later sample of
    the same model and tokens replaces the earlier heads (last writer wins) and counts as an overwrite.
    Forwarding tasks write and forward_backward reads on the API event loop, so there is no lock.
    """

    def __init__(self, k: int, max_bytes: int, comparisons: bool = False):
        self.k = k
        # Heads are histograms of k comparison draws (SKYRL_STABILIZED_COMPARISONS), not the sampler's top-k.
        self.comparisons = comparisons
        self.max_bytes = max_bytes
        self.nbytes = 0
        self.evictions = 0
        self.records: Counter[str] = Counter()
        self.overwrites: Counter[str] = Counter()
        self._entries: OrderedDict[tuple[str, str], tuple[int, np.ndarray, np.ndarray]] = OrderedDict()
        self._last_eviction_log = float("-inf")
        self._last_overwrite_log = float("-inf")

    def record_topk(self, sampling_params) -> int:
        """vLLM logprobs describe the sampling distribution only when sampling does not modify it.

        Comparisons are drawn from the processed distribution, which a greedy sample is not drawn from.
        """
        if self.comparisons:
            return self.k if sampling_params.temperature > 0 else 0
        unmodified = sampling_params.temperature == 1 and sampling_params.top_p == 1 and sampling_params.top_k == -1
        return self.k if unmodified else 0

    def record(self, model: str, prompt_tokens: list[int], tokens: list[int], token_logprobs, raw_top_logprobs):
        if not tokens:
            return
        try:
            if self.comparisons:
                ids, logprobs = convert_vllm_comparison_heads(tokens, raw_top_logprobs, self.k)
            else:
                heads = np.asarray(convert_vllm_decode_logprobs(tokens, token_logprobs, raw_top_logprobs, self.k))
                ids, logprobs = heads[..., 0].astype(np.int32), heads[..., 1].astype(np.float32)
        except ValueError as e:
            # The sample still succeeds; training on it fails later with the missing hash.
            logger.warning(f"Decode heads not recorded: {e}")
            return
        self.put(model, prompt_tokens + tokens, len(prompt_tokens), ids, logprobs)

    def put(self, model: str, tokens: list[int], prompt_length: int, ids: np.ndarray, logprobs: np.ndarray) -> None:
        key = (model, hash_tokens(tokens))
        self.records[model] += 1
        if (old := self._entries.pop(key, None)) is not None:
            self.nbytes -= old[1].nbytes + old[2].nbytes
            self.overwrites[model] += 1
            now = time.monotonic()
            if now - self._last_overwrite_log >= _EVICTION_LOG_INTERVAL_SECONDS:
                self._last_overwrite_log = now
                logger.warning(
                    f"Decode head overwrites by model so far: {dict(self.overwrites)} of {dict(self.records)}"
                )
        self._entries[key] = (prompt_length, ids, logprobs)
        self.nbytes += ids.nbytes + logprobs.nbytes
        evicted = 0
        while self.nbytes > self.max_bytes:
            _, (_, old_ids, old_logprobs) = self._entries.popitem(last=False)
            self.nbytes -= old_ids.nbytes + old_logprobs.nbytes
            evicted += 1
        self.evictions += evicted
        now = time.monotonic()
        if evicted and now - self._last_eviction_log >= _EVICTION_LOG_INTERVAL_SECONDS:
            self._last_eviction_log = now
            logger.warning(
                f"Decode head cache evicted {self.evictions} entries in total "
                f"(cap {self.max_bytes} bytes, {len(self._entries)} entries held)"
            )

    def get(self, key: tuple[str, str]) -> tuple[int, np.ndarray, np.ndarray] | None:
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
        self, model: str, full_tokens: list[int], turn_ends: list[int], weights: list[float], k: int
    ) -> tuple[list[int], list[float]]:
        """Return flat [n, k] heads aligned with targets full_tokens[1:]; turn rows go to targets [start - 1, end - 1)."""
        if k > self.k:
            raise ValueError(f"score_centering_k={k} exceeds the recorded k={self.k}")
        if self.comparisons and k != self.k:
            raise ValueError(f"score_centering_k={k} would truncate histograms of {self.k} comparison draws")
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
            entry = self.get((model, key))
            if entry is None:
                missing.append(key)
                continue
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
