import pytest

from skyrl.tinker import api
from skyrl.tinker.engine import prepare_model_pass_batch


def test_native_loss_and_heads_survive_request_conversion():
    payload = {
        "data": [
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                "loss_fn_inputs": {
                    key: {"data": data}
                    for key, data in {
                        "target_tokens": [2, 3],
                        "weights": [0.0, 1.0],
                        "logprobs": [-1.0, -2.0],
                        "advantages": [0.0, 1.0],
                        "topk_token_ids": [2, 3, 4, 5],
                        "topk_logprobs": [-1.0, -2.0, -1.0, -2.0],
                        "reference_logprobs": [-1.0, -2.0],
                    }.items()
                },
            }
        ],
        "loss_fn": "ppo_score_centered",
        "loss_fn_config": {"score_centering_k": 2, "eps_clip_low": 0.2, "eps_clip_high": 0.2, "kl_loss_coef": 0.001},
    }
    request = api.ForwardBackwardInput.model_validate(payload).to_types()
    batch = prepare_model_pass_batch({"request": ("model", request)})
    assert batch.all_loss_fns == ["ppo_score_centered"]
    assert batch.all_topk_token_ids == [[2, 3, 4, 5]]
    assert batch.all_topk_logprobs == [[-1.0, -2.0, -1.0, -2.0]]
    assert batch.all_reference_logprobs == [[-1.0, -2.0]]
    with pytest.raises(ValueError, match="Invalid loss_fn_config"):
        api.ForwardBackwardInput.model_validate({**payload, "loss_fn_config": {"dual_clip": 3.0}})
