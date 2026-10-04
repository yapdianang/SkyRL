import hashlib
import itertools
import json
import math
from collections import Counter
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import tinker.types as sdk_types
import torch
from tinker.proto.request_conv import forward_backward_request_to_proto

from skyrl.backends.skyrl_train.patches.vllm.patch_stabilized_comparisons import (
    carrier_columns,
)
from skyrl.backends.utils import COMPARISON_PAD_LOGPROB, convert_comparison_heads
from skyrl.tinker import api
from skyrl.tinker.decode_heads import (
    COMPARISONS_KEY,
    LEAVE_IN_KEY,
    DecodeHeadCache,
    hash_tokens,
    subsampled_comparison_heads,
)
from skyrl.tinker.engine import prepare_model_pass_batch
from skyrl.tinker.extra.external_inference import ExternalInferenceClient
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)
from tests.tinker.test_decode_topk_logprobs import VLLM_BODY, _AiohttpSession


def trajectory_hash_tokens(tokens: list[int]) -> str:
    # Copy of trajectory st/trainer/backends/tinker/stabilized_provenance.py:hash_tokens.
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def sample_request(topk_logprobs=0, **sampling):
    return api.SampleRequest(
        base_model="model",
        prompt=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1, 2])]),
        sampling_params=api.SamplingParams(max_tokens=2, **sampling),
        topk_logprobs=topk_logprobs,
    )


async def forward(forwarder, request, decode_heads, body=VLLM_BODY, model_id=""):
    """Forward one sample, from the base model unless model_id names an adapter (SkyRL-Train forwarding only)."""
    payloads = []
    if forwarder == "external":

        def respond(http_request):
            payloads.append(json.loads(http_request.content))
            return httpx.Response(200, json=body)

        client = object.__new__(ExternalInferenceClient)
        client.decode_heads = decode_heads
        async with httpx.AsyncClient(base_url="http://vllm", transport=httpx.MockTransport(respond)) as http:
            result = await client._forward_to_engine(request, "", "", http, base_model="model")
        return payloads[0], result
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client.decode_heads = decode_heads
    client._get_session = lambda: _AiohttpSession(payloads, body)
    result = await client._forward("http://vllm", request, model_id, base_model=None if model_id else "model")
    return payloads[0], result


@pytest.mark.parametrize("tokens", [[], [0], [1, 2, 3], [151_643, 2**31 - 1, 7]])
def test_cache_key_matches_trajectory_hash(tokens):
    assert hash_tokens(tokens) == trajectory_hash_tokens(tokens)


@pytest.mark.parametrize("forwarder", ["external", "skyrl_train"])
@pytest.mark.asyncio
async def test_record_then_lookup_with_unchanged_response(forwarder):
    baseline_payload, baseline = await forward(forwarder, sample_request(), None)
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20)
    payload, result = await forward(forwarder, sample_request(), cache)

    assert result == baseline
    if forwarder == "skyrl_train":
        assert isinstance(result, bytes)
    else:
        assert result.sequences[0].topk_logprobs is None
    assert baseline_payload["logprobs"] in (1, True) and "return_tokens_as_token_ids" not in baseline_payload
    assert payload["logprobs"] == 2 and payload["return_tokens_as_token_ids"] is True
    prompt_length, ids, logprobs = cache.get(("base:model", trajectory_hash_tokens([1, 2, 99, 11])))
    assert prompt_length == 2
    np.testing.assert_array_equal(ids, np.array([[10, 11], [10, 11]], dtype=np.int32))
    np.testing.assert_array_equal(logprobs, np.array([[-0.1, -0.2], [-0.1, -0.2]], dtype=np.float32))
    assert ids.dtype == np.int32 and logprobs.dtype == np.float32

    # The client's own top-k request keeps its K; the record keeps the server's K.
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20)
    payload, result = await forward(forwarder, sample_request(topk_logprobs=1), cache)
    assert payload["logprobs"] == 2
    assert result.sequences[0].topk_logprobs == [[(10, -0.1)], [(10, -0.1)]]
    assert cache.get(("base:model", hash_tokens([1, 2, 99, 11])))[1].shape == (2, 2)

    # Heads from a modified sampling distribution are not recorded.
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20)
    payload, result = await forward(forwarder, sample_request(temperature=0.7), cache)
    assert payload["logprobs"] in (1, True) and cache.nbytes == 0


def test_lru_eviction_by_byte_cap():
    entry_bytes = 2 * 3 * 4 * 2  # ids int32 + logprobs float32, [2, 3] each
    cache = DecodeHeadCache(k=3, max_bytes=2 * entry_bytes)
    heads = np.zeros((2, 3), dtype=np.int32), np.zeros((2, 3), dtype=np.float32)
    cache.put("m", [1, 2, 3], 1, *heads)
    cache.put("m", [1, 2, 4], 1, *heads)
    assert cache.get(("m", hash_tokens([1, 2, 3]))) is not None  # now most recently used
    cache.put("m", [1, 2, 5], 1, *heads)
    assert cache.get(("m", hash_tokens([1, 2, 4]))) is None
    assert cache.get(("m", hash_tokens([1, 2, 3]))) is not None and cache.get(("m", hash_tokens([1, 2, 5]))) is not None
    assert cache.evictions == 1 and cache.nbytes == 2 * entry_bytes


@pytest.mark.asyncio
async def test_adapters_sampling_identical_tokens_keep_their_own_heads():
    other = json.loads(json.dumps(VLLM_BODY))
    other["choices"][0]["logprobs"]["top_logprobs"] = [
        {"token_id:99": -4.0, "token_id:20": -0.3, "token_id:21": -0.4},
        {"token_id:11": -0.2, "token_id:20": -0.3},
    ]
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20)
    await forward("skyrl_train", sample_request(), cache, model_id="run-a")
    await forward("skyrl_train", sample_request(), cache, body=other, model_id="run-b")
    full, weights = [1, 2, 99, 11], [0.0, 1.0, 1.0]
    assert cache.place("run-a", full, [4], weights, 2)[0][2:] == [10, 11, 10, 11]
    assert cache.place("run-b", full, [4], weights, 2)[0][2:] == [20, 21, 11, 20]
    assert not cache.overwrites

    # The same adapter sampling the same tokens again replaces its heads (last writer wins) and is counted.
    await forward("skyrl_train", sample_request(), cache, body=other, model_id="run-a")
    assert cache.place("run-a", full, [4], weights, 2)[0][2:] == [20, 21, 11, 20]
    assert cache.overwrites == {"run-a": 1} and cache.records == {"run-a": 2, "run-b": 1}
    assert cache.stats()["models"]["run-a"] == {"records": 2, "overwrites": 1}

    # Neither another training model nor a base-model sample of the same tokens serves this lookup.
    await forward("skyrl_train", sample_request(), cache)
    with pytest.raises(ValueError, match="no heads recorded for model run-c"):
        cache.place("run-c", full, [4], weights, 2)


@pytest.mark.asyncio
async def test_decode_head_stats_route():
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20)
    cache.put("run-a", [1, 2], 1, np.zeros((1, 2), np.int32), np.zeros((1, 2), np.float32))
    stats = await api.decode_heads_stats(
        SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(decode_heads=cache)))
    )
    assert stats["models"] == {"run-a": {"records": 1, "overwrites": 0}} and stats["entries"] == 1
    with pytest.raises(api.HTTPException):
        await api.decode_heads_stats(SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(decode_heads=None))))


# Turn 1: prompt [1, 2, 3], sampled [4, 5]. Tool output [6, 7, 8]. Turn 2: sampled [9, 10, 11].
FULL = list(range(1, 12))
WEIGHTS = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
K = 2


def two_turn_cache(record_turn_1=True):
    cache = DecodeHeadCache(k=3, max_bytes=1 << 20)
    if record_turn_1:
        cache.put("model", FULL[:5], 3, np.array([[40, 41, 42], [50, 51, 52]], np.int32), -np.ones((2, 3), np.float32))
    cache.put("model", FULL, 8, np.arange(90, 99, dtype=np.int32).reshape(3, 3), -2 * np.ones((3, 3), np.float32))
    return cache


def fwd_bwd_body(datums, loss_fn="ppo_score_centered", loss_fn_config=None) -> bytes:
    request = sdk_types.ForwardBackwardRequest(
        model_id="model",
        seq_id=1,
        forward_backward_input=sdk_types.ForwardBackwardInput(
            data=[
                sdk_types.Datum(
                    model_input=sdk_types.ModelInput.from_ints(FULL[:-1]),
                    loss_fn_inputs={
                        name: sdk_types.TensorData(data=data, dtype=dtype, shape=[len(data)])
                        for name, (data, dtype) in inputs.items()
                    },
                )
                for inputs in datums
            ],
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config or {"score_centering_k": float(K)},
        ),
    )
    return forward_backward_request_to_proto(request).SerializeToString()


def datum_inputs(**extra):
    n = len(FULL) - 1
    return {
        "target_tokens": (FULL[1:], "int64"),
        "weights": (WEIGHTS, "float32"),
        "logprobs": ([-1.0] * n, "float32"),
        "advantages": ([1.0] * n, "float32"),
        **extra,
    }


def fwd_bwd_stub(cache, datums, *body_args):
    async def body():
        return fwd_bwd_body(datums, *body_args)

    return SimpleNamespace(
        headers={"content-type": api.PROTO_CONTENT_TYPE},
        body=body,
        app=SimpleNamespace(state=SimpleNamespace(decode_heads=cache)),
    )


async def resolved_request(cache, datums, *body_args):
    request, _ = await api._read_forward_backward_request(fwd_bwd_stub(cache, datums, *body_args))
    api._resolve_turn_ends(request.forward_backward_input, cache, request.model_id)
    return request


@pytest.mark.asyncio
async def test_turn_ends_place_heads_across_two_turns_with_tool_span():
    request = await resolved_request(two_turn_cache(), [datum_inputs(score_centering_turn_ends=([5, 11], "int64"))])
    batch = prepare_model_pass_batch({"r": ("model", request.forward_backward_input.to_types())})

    ids = np.array(batch.all_topk_token_ids[0]).reshape(10, K)
    logprobs = np.array(batch.all_topk_logprobs[0]).reshape(10, K)
    expected_ids = np.zeros((10, K), dtype=int)
    expected_ids[2:4] = [[40, 41], [50, 51]]  # full positions 3, 4
    expected_ids[7:10] = [[90, 91], [93, 94], [96, 97]]  # full positions 8, 9, 10
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_array_equal(logprobs[2:4], -1.0)
    np.testing.assert_array_equal(logprobs[7:10], -2.0)
    np.testing.assert_array_equal(logprobs[[0, 1, 4, 5, 6]], 0.0)  # prompt and tool span


@pytest.mark.asyncio
async def test_missing_head_for_positive_weight_fails_with_hash():
    datums = [datum_inputs(score_centering_turn_ends=([5, 11], "int64"))]
    stub = fwd_bwd_stub(two_turn_cache(record_turn_1=False), datums)
    with pytest.raises(api.HTTPException) as error:
        await api.forward_backward(stub, session=None)
    assert error.value.status_code == 400
    assert trajectory_hash_tokens(FULL[:5]) in error.value.detail

    # A missing turn whose targets all have zero weight is allowed.
    zero_turn_1 = datum_inputs(score_centering_turn_ends=([5, 11], "int64"))
    zero_turn_1["weights"] = ([0.0] * 7 + [1.0] * 3, "float32")
    await resolved_request(two_turn_cache(record_turn_1=False), [zero_turn_1])

    with pytest.raises(ValueError, match="SKYRL_SCORE_CENTERING_RECORD_TOPK"):
        await resolved_request(None, datums)


@pytest.mark.asyncio
async def test_batch_accepts_exactly_one_head_form():
    head_form = datum_inputs(topk_token_ids=([0] * 20, "int64"), topk_logprobs=([0.0] * 20, "float32"))
    turn_form = datum_inputs(score_centering_turn_ends=([5, 11], "int64"))
    with pytest.raises(ValueError, match="Each datum"):
        await resolved_request(two_turn_cache(), [turn_form, head_form])
    request = await resolved_request(two_turn_cache(), [head_form])
    assert request.forward_backward_input.data[0].loss_fn_inputs["topk_token_ids"].data == [0] * 20


def comparison_body(draws):
    """VLLM_BODY with one carrier per comparison draw after each position's top-k, as the patched sampler returns it."""
    body = json.loads(json.dumps(VLLM_BODY))
    choice = body["choices"][0]
    for top_logprobs, row_draws in zip(choice["logprobs"]["top_logprobs"], draws):
        taken = torch.tensor([[int(token[9:]) for token in top_logprobs]])
        ids, values = carrier_columns(torch.tensor([row_draws]), taken)
        top_logprobs.update({f"token_id:{token}": value for token, value in zip(ids[0].tolist(), values[0].tolist())})
    return body


def histograms(cache, full_tokens, turn_ends, weights, comparisons, leave_in=False):
    """Per-row {token: count} of the comparison heads a forward_backward lookup places."""
    ids, logprobs = cache.place("base:model", full_tokens, turn_ends, weights, comparisons, comparisons, leave_in)
    ids = np.asarray(ids).reshape(-1, comparisons)
    logprobs = np.asarray(logprobs, dtype=np.float32).reshape(-1, comparisons)
    rows = []
    for row_ids, row_logprobs in zip(ids, logprobs):
        drawn = row_logprobs > COMPARISON_PAD_LOGPROB
        rows.append(dict(zip(row_ids[drawn].tolist(), np.round(comparisons * np.exp(row_logprobs[drawn])).tolist())))
    return rows


# Comparison draws at the two sampled positions of VLLM_BODY; both rows share ids with the top-k.
DRAWS = [[99] * 5 + [10] * 7 + [12] * 4, [10] * 16]


@pytest.mark.parametrize("forwarder", ["external", "skyrl_train"])
@pytest.mark.asyncio
async def test_one_server_records_top_k_heads_and_comparison_draws(forwarder):
    top_k_only = DecodeHeadCache(k=2, max_bytes=1 << 20)
    await forward(forwarder, sample_request(), top_k_only)
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20, comparisons=16)
    payload, result = await forward(forwarder, sample_request(), cache, comparison_body(DRAWS))
    assert payload["logprobs"] == 2 + 16 and payload["return_tokens_as_token_ids"] is True
    full, weights = [1, 2, 99, 11], [0.0, 1.0, 1.0]

    # Top-k heads are those of a top-k-only (-sc2) server; the comparison histograms are the draws.
    assert cache.place("base:model", full, [4], weights, 2) == top_k_only.place("base:model", full, [4], weights, 2)
    assert histograms(cache, full, [4], weights, 16)[1:] == [Counter(row) for row in DRAWS]

    # A modified sampling law records only the draws, which come from the processed law; greedy records nothing.
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20, comparisons=16)
    payload, _ = await forward(forwarder, sample_request(temperature=0.7), cache, comparison_body(DRAWS))
    assert payload["logprobs"] == 18
    with pytest.raises(ValueError, match="no heads recorded"):
        cache.place("base:model", full, [4], weights, 2)
    assert histograms(cache, full, [4], weights, 16)[1:] == [Counter(row) for row in DRAWS]
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20, comparisons=16)
    payload, _ = await forward(forwarder, sample_request(temperature=0.0), cache, comparison_body(DRAWS))
    assert payload["logprobs"] in (1, True) and cache.nbytes == 0


def test_missing_draws_leave_the_top_k_heads():
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20, comparisons=16)
    logprobs = VLLM_BODY["choices"][0]["logprobs"]
    sampling_params = api.SamplingParams(max_tokens=2)
    cache.record("m", [1, 2], [99, 11], logprobs["token_logprobs"], logprobs["top_logprobs"], sampling_params)
    assert cache.get(("m", hash_tokens([1, 2, 99, 11]))) is not None
    with pytest.raises(ValueError, match="no heads recorded"):
        cache.place("m", [1, 2, 99, 11], [4], [0.0, 1.0, 1.0], 16, comparisons=16)


RECORDED = np.array([[0] * 9 + [1] * 5 + [2] * 2], dtype=np.int32)


def test_full_subset_is_the_recorded_histogram_and_repeats_are_identical():
    ids, logprobs = subsampled_comparison_heads(hash_tokens([7]), RECORDED, [1], 16, leave_in=False)
    drawn = logprobs[0] > COMPARISON_PAD_LOGPROB
    assert dict(zip(ids[0][drawn].tolist(), np.round(16 * np.exp(logprobs[0][drawn])).tolist())) == {0: 9, 1: 5, 2: 2}
    for leave_in in (False, True):
        first = subsampled_comparison_heads(hash_tokens([7]), RECORDED, [1], 4, leave_in)
        np.testing.assert_array_equal(first, subsampled_comparison_heads(hash_tokens([7]), RECORDED, [1], 4, leave_in))
    ids, logprobs = subsampled_comparison_heads(hash_tokens([7]), RECORDED, [2], 1, leave_in=True)
    assert ids[0, 0] == 2 and logprobs[0, 0] == 0.0


def test_heads_pad_with_the_smallest_ids_not_drawn():
    ids, logprobs = convert_comparison_heads([[205] * 7 + [100] * 9, [1] * 8 + [3] * 8], 16)
    assert ids[0].tolist() == [100, 205, *range(14)] and ids[1].tolist() == [1, 3, 0, 2, *range(4, 16)]
    np.testing.assert_allclose(logprobs[:, :2], np.log([[9 / 16, 7 / 16], [0.5, 0.5]]), rtol=1e-6)
    assert (logprobs[:, 2:] == COMPARISON_PAD_LOGPROB).all()


def multinomial(counts, p):
    return math.factorial(sum(counts)) * math.prod(q**c / math.factorial(c) for q, c in zip(p, counts))


def hypergeometric(subset, recorded):
    return math.prod(math.comb(r, s) for r, s in zip(recorded, subset)) / math.comb(sum(recorded), sum(subset))


def compositions(total, parts):
    """Every count vector of `parts` nonnegative integers summing to `total`."""
    for bars in itertools.combinations(range(total + parts - 1), parts - 1):
        edges = (-1, *bars, total + parts - 1)
        yield tuple(right - left - 1 for left, right in zip(edges, edges[1:]))


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
def test_a_uniform_subset_of_iid_draws_is_iid(k):
    # Exact: sum_h Multinomial(16; p)(h) Hypergeometric(s | h, k) = Multinomial(k; p)(s).
    p = (0.55, 0.3, 0.15)
    for subset in compositions(k, 3):
        mixture = sum(
            multinomial(recorded, p) * hypergeometric(subset, recorded)
            for recorded in compositions(16, 3)
            if all(s <= r for s, r in zip(subset, recorded))
        )
        assert mixture == pytest.approx(multinomial(subset, p), abs=1e-12)


def test_seeded_subsets_follow_the_hypergeometric_law():
    k, keys = 4, 6000
    observed = Counter()
    for key in range(keys):
        ids, logprobs = subsampled_comparison_heads(hash_tokens([key]), RECORDED, [0], k, leave_in=False)
        counts = dict(zip(ids[0].tolist(), np.round(k * np.exp(logprobs[0])).tolist()))
        observed[tuple(int(counts.get(token, 0)) for token in range(3))] += 1
    subsets = [s for s in compositions(k, 3) if s[2] <= 2]
    expected = {s: keys * hypergeometric(s, (9, 5, 2)) for s in subsets}
    assert sum(observed.values()) == keys and set(observed) <= set(subsets)
    chi_square = sum((observed[s] - e) ** 2 / e for s, e in expected.items())
    assert chi_square < 31.264  # 0.999 quantile, 11 degrees of freedom for 12 subsets


def comparison_cache(draws=16):
    cache = DecodeHeadCache(k=32, max_bytes=1 << 20, comparisons=draws)
    for start, end in ((3, 5), (8, 11)):
        rows = np.arange(16 * (end - start), dtype=np.int32).reshape(-1, 16) % 7
        cache.put("model", FULL[:end], start, rows, comparisons=True)
        cache.put("model", FULL[:end], start, rows + 100, np.full(rows.shape, -1.0, dtype=np.float32))
    return cache


@pytest.mark.asyncio
async def test_each_request_selects_its_heads_and_mismatches_fail():
    datums = [datum_inputs(score_centering_turn_ends=([5, 11], "int64"))]
    # Row 2 holds the first sampled position, whose target token is FULL[3].
    for loss_fn_config, contains in (
        ({"score_centering_k": 16.0}, {100, 101, 106}),
        ({"score_centering_k": 16.0, COMPARISONS_KEY: 16.0}, set(range(7))),
        ({"score_centering_k": 4.0, COMPARISONS_KEY: 4.0, LEAVE_IN_KEY: 1.0}, {FULL[3]}),
    ):
        request = await resolved_request(comparison_cache(), datums, "reinforce_score_centered", loss_fn_config)
        k = int(loss_fn_config["score_centering_k"])
        # The keys stay for the backend, which reports the weighted NLL like native stabilized_reinforce.
        assert request.forward_backward_input.loss_fn_config == loss_fn_config
        inputs = request.forward_backward_input.data[0].loss_fn_inputs
        row_ids = inputs["topk_token_ids"].data[2 * k : 3 * k]
        row_logprobs = inputs["topk_logprobs"].data[2 * k : 3 * k]
        assert contains <= {i for i, lp in zip(row_ids, row_logprobs) if lp > COMPARISON_PAD_LOGPROB}

    for cache, loss_fn_config, message in (
        (two_turn_cache(), {"score_centering_k": 16.0, COMPARISONS_KEY: 16.0}, "records 0 comparison draws"),
        (comparison_cache(), {"score_centering_k": 32.0, COMPARISONS_KEY: 32.0}, "records 16 comparison draws"),
        (comparison_cache(), {"score_centering_k": 8.0, COMPARISONS_KEY: 4.0}, "must equal"),
        (comparison_cache(), {"score_centering_k": 4.0, COMPARISONS_KEY: 2.5}, "must be an integer"),
        (comparison_cache(), {"score_centering_k": 4.0, LEAVE_IN_KEY: 1.0}, "must be an integer"),
        (comparison_cache(), {"score_centering_k": 4.0, COMPARISONS_KEY: 4.0, LEAVE_IN_KEY: 2.0}, "must be"),
    ):
        with pytest.raises(ValueError, match=message):
            await resolved_request(cache, datums, "reinforce_score_centered", loss_fn_config)
    with pytest.raises(ValueError, match=f"{COMPARISONS_KEY} requires"):
        await resolved_request(
            comparison_cache(), [datum_inputs()], "reinforce_score_centered", {COMPARISONS_KEY: 16.0}
        )


@pytest.mark.asyncio
async def test_client_decode_topk_is_rejected_under_comparisons(monkeypatch):
    monkeypatch.setattr(api, "SKYRL_STABILIZED_COMPARISONS", 16)
    with pytest.raises(api.HTTPException) as error:
        await api.asample(sample_request(topk_logprobs=2), SimpleNamespace(), session=None)
    assert error.value.status_code == 400 and "comparison draws" in error.value.detail
