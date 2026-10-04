from types import SimpleNamespace

import pytest

from skyrl.tinker import api


@pytest.mark.asyncio
async def test_asample_response_has_one_unique_sequence_id_per_sample():
    spawned = []

    async def call_and_store_result(*args, **kwargs):
        pass

    store = SimpleNamespace(create=lambda model_id, sample_input: 7, spawn_forwarding_task=spawned.append)
    state = SimpleNamespace(
        external_future_store=store,
        external_inference_client=SimpleNamespace(call_and_store_result=call_and_store_result),
    )
    request = api.SampleRequest(
        base_model="model",
        num_samples=3,
        prompt=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1])]),
        sampling_params=api.SamplingParams(max_tokens=1),
    )
    responses = [await api.asample(request, SimpleNamespace(app=SimpleNamespace(state=state)), None) for _ in range(2)]
    for task in spawned:
        task.close()

    ids = [response.sample_sequence_ids for response in responses]
    assert [len(x) for x in ids] == [3, 3]
    assert len(set(ids[0] + ids[1])) == 6
    assert responses[0].model_dump()["request_id"] == "7"
