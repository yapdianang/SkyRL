import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import tinker.types as sdk_types
import torch
from tinker.proto.request_conv import forward_backward_request_to_proto

from skyrl.backends.skyrl_train.patches.vllm.patch_stabilized_comparisons import (
    encode_comparisons,
)
from skyrl.backends.utils import COMPARISON_PAD_LOGPROB
from skyrl.tinker import api
from skyrl.tinker.decode_heads import COMPARISONS_KEY, DecodeHeadCache, hash_tokens
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


async def forward(forwarder, request, decode_heads, body=VLLM_BODY):
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
    result = await client._forward("http://vllm", request, "", base_model="model")
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
    prompt_length, ids, logprobs = cache.get(trajectory_hash_tokens([1, 2, 99, 11]))
    assert prompt_length == 2
    np.testing.assert_array_equal(ids, np.array([[10, 11], [10, 11]], dtype=np.int32))
    np.testing.assert_array_equal(logprobs, np.array([[-0.1, -0.2], [-0.1, -0.2]], dtype=np.float32))
    assert ids.dtype == np.int32 and logprobs.dtype == np.float32

    # The client's own top-k request keeps its K; the record keeps the server's K.
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20)
    payload, result = await forward(forwarder, sample_request(topk_logprobs=1), cache)
    assert payload["logprobs"] == 2
    assert result.sequences[0].topk_logprobs == [[(10, -0.1)], [(10, -0.1)]]
    assert cache.get(hash_tokens([1, 2, 99, 11]))[1].shape == (2, 2)

    # Heads from a modified sampling distribution are not recorded.
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20)
    payload, result = await forward(forwarder, sample_request(temperature=0.7), cache)
    assert payload["logprobs"] in (1, True) and cache.nbytes == 0


def test_lru_eviction_by_byte_cap():
    entry_bytes = 2 * 3 * 4 * 2  # ids int32 + logprobs float32, [2, 3] each
    cache = DecodeHeadCache(k=3, max_bytes=2 * entry_bytes)
    heads = np.zeros((2, 3), dtype=np.int32), np.zeros((2, 3), dtype=np.float32)
    cache.put([1, 2, 3], 1, *heads)
    cache.put([1, 2, 4], 1, *heads)
    assert cache.get(hash_tokens([1, 2, 3])) is not None  # now most recently used
    cache.put([1, 2, 5], 1, *heads)
    assert cache.get(hash_tokens([1, 2, 4])) is None
    assert cache.get(hash_tokens([1, 2, 3])) is not None and cache.get(hash_tokens([1, 2, 5])) is not None
    assert cache.evictions == 1 and cache.nbytes == 2 * entry_bytes


# Turn 1: prompt [1, 2, 3], sampled [4, 5]. Tool output [6, 7, 8]. Turn 2: sampled [9, 10, 11].
FULL = list(range(1, 12))
WEIGHTS = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
K = 2


def two_turn_cache(record_turn_1=True):
    cache = DecodeHeadCache(k=3, max_bytes=1 << 20)
    if record_turn_1:
        cache.put(FULL[:5], 3, np.array([[40, 41, 42], [50, 51, 52]], np.int32), -np.ones((2, 3), np.float32))
    cache.put(FULL, 8, np.arange(90, 99, dtype=np.int32).reshape(3, 3), -2 * np.ones((3, 3), np.float32))
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
    api._resolve_turn_ends(request.forward_backward_input, cache)
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


def comparison_body(sampled, sampled_logprobs, draws, k=16):
    """vLLM's completion body for one sample under SKYRL_STABILIZED_COMPARISONS=k."""
    ids, logprobs = encode_comparisons(torch.tensor(draws), torch.tensor(sampled), torch.tensor(sampled_logprobs), k, k)
    # vLLM folds a position's columns, sampled token first, into a dict keyed by token id.
    top_logprobs = [
        {f"token_id:{token}": logprob for token, logprob in zip([s, *row_ids], [lp, *row_logprobs])}
        for s, lp, row_ids, row_logprobs in zip(sampled, sampled_logprobs, ids.tolist(), logprobs.tolist())
    ]
    logprobs = {"token_logprobs": sampled_logprobs, "top_logprobs": top_logprobs}
    return {"choices": [{"token_ids": sampled, "finish_reason": "stop", "logprobs": logprobs}]}


@pytest.mark.parametrize("forwarder", ["external", "skyrl_train"])
@pytest.mark.parametrize("leave_in", [False, True])
@pytest.mark.asyncio
async def test_comparison_draws_are_recorded_as_their_histogram(forwarder, leave_in):
    # Leave-in drops one draw; the sampled token is the 16th. Token 11 is never drawn at position 2.
    draws = [row[leave_in:] for row in ([99] * 5 + [10] * 7 + [12] * 4, [10] * 16)]
    body = comparison_body([99, 11], [-1.5, -0.25], draws)
    cache = DecodeHeadCache(k=16, max_bytes=1 << 20, comparisons=True)
    # Draws come from the processed law, so a modified sampling distribution is recorded.
    payload, _ = await forward(forwarder, sample_request(temperature=0.7, top_k=5), cache, body)
    assert payload["logprobs"] == 16 and payload["return_tokens_as_token_ids"] is True

    ids, logprobs = cache.place([1, 2, 99, 11], [4], [0.0, 1.0, 1.0], 16)
    for row, sampled, row_draws in zip((1, 2), (99, 11), draws):
        head_ids = np.asarray(ids).reshape(3, 16)[row]
        head_logprobs = np.asarray(logprobs, dtype=np.float32).reshape(3, 16)[row]
        drawn = head_logprobs > COMPARISON_PAD_LOGPROB
        histogram = dict(zip(head_ids[drawn].tolist(), np.round(16 * np.exp(head_logprobs[drawn])).tolist()))
        assert histogram == Counter(row_draws + [sampled] * leave_in)
        assert len(set(head_ids.tolist())) == 16 and (head_logprobs[~drawn] == COMPARISON_PAD_LOGPROB).all()

    cache = DecodeHeadCache(k=16, max_bytes=1 << 20, comparisons=True)
    payload, _ = await forward(forwarder, sample_request(temperature=0.0), cache, body)
    assert payload["logprobs"] in (1, True) and cache.nbytes == 0


def test_top_k_logprobs_are_not_recorded_as_comparisons():
    cache = DecodeHeadCache(k=2, max_bytes=1 << 20, comparisons=True)
    top_logprobs = VLLM_BODY["choices"][0]["logprobs"]["top_logprobs"]
    cache.record([1, 2], [99, 11], [-4.0, -0.2], top_logprobs)
    assert cache.nbytes == 0


def comparison_cache():
    cache = DecodeHeadCache(k=16, max_bytes=1 << 20, comparisons=True)
    for start, end in ((3, 5), (8, 11)):
        ids = np.arange(16 * (end - start), dtype=np.int32).reshape(-1, 16)
        cache.put(FULL[:end], start, ids, np.full(ids.shape, -np.log(16), dtype=np.float32))
    return cache


@pytest.mark.asyncio
async def test_comparison_heads_need_the_matching_request_key():
    datums = [datum_inputs(score_centering_turn_ends=([5, 11], "int64"))]
    config = {"score_centering_k": 16.0, COMPARISONS_KEY: 16.0}
    request = await resolved_request(comparison_cache(), datums, "reinforce_score_centered", config)
    # The key stays for the backend, which then reports the weighted NLL like native stabilized_reinforce.
    assert request.forward_backward_input.loss_fn_config == config
    assert request.forward_backward_input.data[0].loss_fn_inputs["topk_token_ids"].data[2 * 16 : 3 * 16] == list(
        range(16)
    )

    for cache, loss_fn_config, message in (
        (comparison_cache(), {"score_centering_k": 16.0}, "records 16 comparison draws"),
        (two_turn_cache(), {"score_centering_k": float(K), COMPARISONS_KEY: 16.0}, "records 0 comparison draws"),
        (comparison_cache(), {"score_centering_k": 8.0, COMPARISONS_KEY: 16.0}, "truncate"),
    ):
        with pytest.raises(ValueError, match=message):
            await resolved_request(cache, datums, "reinforce_score_centered", loss_fn_config)
    with pytest.raises(ValueError, match=f"{COMPARISONS_KEY} requires"):
        await resolved_request(comparison_cache(), [datum_inputs()], "reinforce_score_centered", config)


@pytest.mark.asyncio
async def test_client_decode_topk_is_rejected_under_comparisons(monkeypatch):
    monkeypatch.setattr(api, "SKYRL_STABILIZED_COMPARISONS", 16)
    with pytest.raises(api.HTTPException) as error:
        await api.asample(sample_request(topk_logprobs=2), SimpleNamespace(), session=None)
    assert error.value.status_code == 400 and "comparison draws" in error.value.detail
