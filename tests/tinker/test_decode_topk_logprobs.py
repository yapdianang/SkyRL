import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from skyrl.backends.utils import convert_vllm_decode_logprobs
from skyrl.tinker import api, types
from skyrl.tinker.db_models import RequestStatus
from skyrl.tinker.extra.external_inference import ExternalInferenceClient
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)

VLLM_BODY = {
    "choices": [
        {
            "token_ids": [99, 11],
            "finish_reason": "stop",
            "logprobs": {
                "token_logprobs": [-4.0, -0.2],
                "top_logprobs": [
                    {"token_id:99": -4.0, "token_id:10": -0.1, "token_id:11": -0.2},
                    {"token_id:11": -0.2, "token_id:10": -0.1},
                ],
            },
        }
    ]
}


class _AiohttpResponse:
    status = 200
    headers = {}

    def __init__(self, body=None):
        self.body = VLLM_BODY if body is None else body

    async def read(self):
        return json.dumps(self.body).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _AiohttpSession:
    def __init__(self, payloads, body=None):
        self.payloads = payloads
        self.body = body

    def post(self, url, json, headers):
        self.payloads.append(json)
        return _AiohttpResponse(self.body)


def sample_request(topk_logprobs=2, **sampling):
    return api.SampleRequest(
        base_model="model",
        prompt=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1, 2])]),
        sampling_params=api.SamplingParams(max_tokens=2, **sampling),
        topk_logprobs=topk_logprobs,
    )


@pytest.mark.parametrize("sampling", [{"temperature": 0.7}, {"top_p": 0.9}, {"top_k": 32}])
def test_decode_topk_rejects_changed_sampling_distribution(sampling):
    with pytest.raises(ValueError, match="requires temperature=1"):
        sample_request(**sampling)


@pytest.mark.parametrize("forwarder", ["external", "skyrl_train"])
@pytest.mark.asyncio
async def test_decode_heads_survive_forwarding_with_sampled_extra_removed(forwarder):
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=VLLM_BODY)

    if forwarder == "external":
        async with httpx.AsyncClient(base_url="http://vllm", transport=httpx.MockTransport(respond)) as http:
            client = object.__new__(ExternalInferenceClient)
            result = await client._forward_to_engine(sample_request(), "", "", http, base_model="model")
    else:
        client = object.__new__(SkyRLTrainInferenceForwardingClient)
        client._get_session = lambda: _AiohttpSession(payloads)
        result = await client._forward("http://vllm", sample_request(), "", base_model="model")

    assert payloads[0]["logprobs"] == 2
    assert payloads[0]["return_tokens_as_token_ids"] is True
    assert "prompt_logprobs" not in payloads[0]
    sequence = json.loads(result.model_dump_json())["sequences"][0]
    assert sequence["tokens"] == [99, 11]
    assert sequence["logprobs"] == [-4.0, -0.2]
    assert sequence["topk_logprobs"] == [[[10, -0.1], [11, -0.2]], [[10, -0.1], [11, -0.2]]]


@pytest.mark.parametrize(
    ("sampled", "heads"),
    [
        ([-0.1], None),
        ([-0.1], []),
        ([], [{"token_id:1": -0.1}]),
        ([-0.1], [{"decoded text": -0.1}]),
        ([-0.1], [{"token_id:1": float("nan")}]),
    ],
)
def test_decode_heads_fail_instead_of_fabricating_probabilities(sampled, heads):
    with pytest.raises(ValueError):
        convert_vllm_decode_logprobs([1], sampled, heads, 1)


@pytest.mark.asyncio
async def test_forwarding_without_decode_heads_keeps_proto_result():
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client._get_session = lambda: _AiohttpSession([])
    result = await client._forward("http://vllm", sample_request(topk_logprobs=0), "", base_model="model")
    assert isinstance(result, bytes)


@pytest.mark.parametrize("request_type", [types.RequestType.SAMPLE, types.RequestType.EXTERNAL])
@pytest.mark.asyncio
async def test_decode_heads_require_json_retrieval(request_type):
    output = types.SampleOutput(
        sequences=[
            types.GeneratedSequence(stop_reason="stop", tokens=[10], logprobs=[-0.1], topk_logprobs=[[(10, -0.1)]])
        ]
    )
    store = SimpleNamespace(
        wait=AsyncMock(return_value=(RequestStatus.COMPLETED, request_type, output.model_dump_json())),
        proto_result=lambda _: None,
        mark_retrieved=lambda _: None,
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(external_future_store=store)),
        headers={"accept": api.PROTO_CONTENT_TYPE},
        is_disconnected=AsyncMock(return_value=False),
    )
    with pytest.raises(api.HTTPException) as error:
        await api.retrieve_future(api.RetrieveFutureRequest(request_id="1"), request)
    assert error.value.status_code == 406
    request.headers = {"accept": "application/json"}
    response = await api.retrieve_future(api.RetrieveFutureRequest(request_id="1"), request)
    assert json.loads(response.body)["sequences"][0]["topk_logprobs"] == [[[10, -0.1]]]
