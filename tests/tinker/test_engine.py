import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from cloudpathlib import AnyPath
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Session, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import api, types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import (
    CheckpointDB,
    CheckpointStatus,
    FutureDB,
    ModelDB,
    RequestStatus,
    SessionDB,
)
from skyrl.tinker.engine import TinkerEngine, prepare_model_pass_batch

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def test_process_unload_model():
    """Test that process_unload_model removes model from backend."""
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={"max_lora_adapters": 4, "max_lora_rank": 32},
    )
    engine = TinkerEngine(config)
    SQLModel.metadata.create_all(engine.db_engine)

    model_id = "test_model"
    _ = engine.process_single_request(
        types.RequestType.CREATE_MODEL, model_id, {"lora_config": {"rank": 8, "alpha": 16, "seed": 0}}
    )
    assert engine.backend.has_model(model_id)

    result = engine.process_unload_model(model_id, types.UnloadModelInput())
    assert result.status == "unloaded"
    assert not engine.backend.has_model(model_id)


def test_cleanup_stale_sessions():
    """Test that cleanup_stale_sessions unloads models from expired sessions."""
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={"max_lora_adapters": 4, "max_lora_rank": 32},
        session_timeout_sec=60,
        database_url="sqlite:///:memory:",  # Use in-memory DB for test isolation
    )
    engine = TinkerEngine(config)
    SQLModel.metadata.create_all(engine.db_engine)

    model_id = "stale_model"
    session_id = "stale_session"

    # Create model in backend
    _ = engine.process_single_request(
        types.RequestType.CREATE_MODEL, model_id, {"lora_config": {"rank": 8, "alpha": 16, "seed": 0}}
    )
    assert engine.backend.has_model(model_id)

    # Insert stale session and model into DB
    stale_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=120)
    with Session(engine.db_engine) as session:
        session.add(
            SessionDB(
                session_id=session_id,
                sdk_version="test",
                status="active",
                last_heartbeat_at=stale_heartbeat,
            )
        )
        session.add(
            ModelDB(
                model_id=model_id,
                base_model=BASE_MODEL,
                lora_config=types.LoraConfig(rank=8, alpha=16, seed=0).model_dump(),
                status="ready",
                request_id=1,
                session_id=session_id,
            )
        )
        session.commit()

    # Run cleanup and assert one model was unloaded
    assert engine.cleanup_stale_sessions() == 1
    assert not engine.backend.has_model(model_id)


@pytest.mark.parametrize(
    ("loss_fn", "loss_fn_config", "advantages", "logprobs", "values", "returns"),
    [
        pytest.param(
            "ppo",
            {"clip_low_threshold": 0.7, "clip_high_threshold": 1.3},
            [],
            [],
            [],
            [],
            id="ppo_with_loss_fn_config",
        ),
        pytest.param("ppo", {"value_clip": 0.2}, [], [], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], id="ppo_with_value_clip"),
        pytest.param("cross_entropy", None, [], [], [], [], id="cross_entropy_default_config"),
        pytest.param(
            "cispo",
            {"clip_low_threshold": 0.7, "clip_high_threshold": 1.3},
            [0.1, 0.2, 0.3],
            [-1.1, -1.0, -0.9],
            [],
            [],
            id="cispo",
        ),
        pytest.param("ppo_critic", {"value_clip": 0.2}, [], [], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], id="ppo_critic"),
    ],
)
def test_prepare_model_pass_batch_loss_fn_and_config(
    loss_fn: str,
    loss_fn_config: dict[str, float] | None,
    advantages: list[float],
    logprobs: list[float],
    values: list[float],
    returns: list[float],
):
    """Test that prepare_model_pass_batch preserves loss_fn and loss_fn_config values."""
    datum = types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
        loss_fn_inputs=types.LossFnInputs(
            target_tokens=types.TensorData(data=[2, 3, 4]),
            weights=types.TensorData(data=[1.0, 1.0, 1.0]),
            advantages=types.TensorData(data=advantages),
            logprobs=types.TensorData(data=logprobs),
            values=types.TensorData(data=values),
            returns=types.TensorData(data=returns),
        ),
    )

    requests = {
        "req1": (
            "model1",
            types.ForwardBackwardInput(
                data=[datum],
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            ),
        ),
    }

    batch = prepare_model_pass_batch(requests)
    assert batch.all_loss_fns == [loss_fn]
    assert batch.all_loss_fn_configs == [loss_fn_config]
    assert batch.all_model_inputs == [datum.model_input]
    assert batch.all_values == [values]
    assert batch.all_returns == [returns]


@pytest.fixture()
def scheduling_engine():
    """Create a TinkerEngine with only the DB initialized (no backend) for scheduling tests."""
    from sqlalchemy import create_engine

    from skyrl.tinker.db_models import enable_sqlite_wal

    engine = object.__new__(TinkerEngine)
    engine.db_engine = create_engine("sqlite:///:memory:", echo=False)
    enable_sqlite_wal(engine.db_engine)
    SQLModel.metadata.create_all(engine.db_engine)
    return engine


def test_find_single_requests_respects_forward_backward_barriers(scheduling_engine):
    """Regression: optim_step must not run before a preceding forward_backward for the same model.

    Given pending requests [fwdbwd1, optim1, fwdbwd2, optim2] for the same model,
    find_single_requests should only return optim1 (not optim2), because fwdbwd2
    acts as a barrier — optim2 depends on fwdbwd2's gradients.
    """
    engine = scheduling_engine
    model_id = "test_model"

    with Session(engine.db_engine) as session:
        # Insert requests in order: fwdbwd1, optim1, fwdbwd2, optim2
        for req_type in [
            types.RequestType.FORWARD_BACKWARD,
            types.RequestType.OPTIM_STEP,
            types.RequestType.FORWARD_BACKWARD,
            types.RequestType.OPTIM_STEP,
        ]:
            session.add(
                FutureDB(
                    request_type=req_type,
                    model_id=model_id,
                    request_data={},
                    status=RequestStatus.PENDING,
                )
            )
        session.commit()

    with Session(engine.db_engine) as session:
        # find_single_requests should return only optim1 (request_id=2), NOT optim2 (request_id=4)
        singles = engine.find_single_requests(session)
        assert list(singles.keys()) == ["2"]


def test_find_single_requests_no_barrier_when_no_pending_passes(scheduling_engine):
    """When there are no pending forward/forward_backward requests, all single requests are returned."""
    engine = scheduling_engine

    with Session(engine.db_engine) as session:
        for model_id in ["model_a", "model_b"]:
            session.add(
                FutureDB(
                    request_type=types.RequestType.OPTIM_STEP,
                    model_id=model_id,
                    request_data={},
                    status=RequestStatus.PENDING,
                )
            )
        session.commit()

    with Session(engine.db_engine) as session:
        singles = engine.find_single_requests(session)
        assert len(singles) == 2


def test_find_single_requests_barrier_is_per_model(scheduling_engine):
    """A blocked forward_backward on model_a should not block an optim_step on model_b."""
    engine = scheduling_engine

    with Session(engine.db_engine) as session:
        # model_a: fwdbwd(1), optim(2), fwdbwd(3), optim(4)
        # model_b: optim(5)
        for req_type in [
            types.RequestType.FORWARD_BACKWARD,
            types.RequestType.OPTIM_STEP,
            types.RequestType.FORWARD_BACKWARD,
            types.RequestType.OPTIM_STEP,
        ]:
            session.add(
                FutureDB(
                    request_type=req_type,
                    model_id="model_a",
                    request_data={},
                    status=RequestStatus.PENDING,
                )
            )
        session.add(
            FutureDB(
                request_type=types.RequestType.OPTIM_STEP,
                model_id="model_b",
                request_data={},
                status=RequestStatus.PENDING,
            )
        )
        session.commit()

    with Session(engine.db_engine) as session:
        singles = engine.find_single_requests(session)
        # model_a: optim(2) returned, optim(4) blocked by fwdbwd(3)
        # model_b: optim(5) returned (not affected by model_a's barrier)
        assert list(singles.keys()) == ["2", "5"]
        assert singles["2"][0] == "model_a"
        assert singles["5"][0] == "model_b"


def _sample_request(checkpoint_id: str = "ckpt") -> dict:
    return types.SampleInput(
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
        sampling_params=types.SamplingParams(temperature=1.0, max_tokens=16, seed=0),
        num_samples=1,
        checkpoint_id=checkpoint_id,
        prompt_logprobs=False,
    ).model_dump()


def test_find_batchable_sample_respects_sampler_weight_sync_barrier(scheduling_engine):
    """Samples queued after a sampler-weight sync must wait for the sync to run first."""
    engine = scheduling_engine
    engine.config = EngineConfig(base_model=BASE_MODEL, backend="megatron")
    engine.backend = SimpleNamespace(
        _cfg=SimpleNamespace(
            generator=SimpleNamespace(inference_engine=SimpleNamespace(max_num_seqs=64)),
        )
    )

    with Session(engine.db_engine) as session:
        for req_type, request_data in [
            (types.RequestType.SAMPLE, _sample_request()),
            (types.RequestType.SAMPLE, _sample_request()),
            (
                types.RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
                types.SaveWeightsForSamplerInput(path="ss1_seq1", sampling_session_seq_id=1, seq_id=1).model_dump(),
            ),
            (types.RequestType.SAMPLE, _sample_request()),
            (types.RequestType.SAMPLE, _sample_request()),
        ]:
            session.add(
                FutureDB(
                    request_type=req_type,
                    model_id="model_a",
                    request_data=request_data,
                    status=RequestStatus.PENDING,
                )
            )
        session.commit()

    with Session(engine.db_engine) as session:
        samples = engine.find_batchable_sample(session)
        assert list(samples.keys()) == ["1", "2"]


def test_find_batchable_sample_caps_batch_to_serving_capacity(scheduling_engine):
    """Megatron/vLLM sample batches are capped by generator.inference_engine.max_num_seqs."""
    engine = scheduling_engine
    engine.config = EngineConfig(base_model=BASE_MODEL, backend="megatron")
    engine.backend = SimpleNamespace(
        _cfg=SimpleNamespace(
            generator=SimpleNamespace(inference_engine=SimpleNamespace(max_num_seqs=2)),
        )
    )

    with Session(engine.db_engine) as session:
        for _ in range(4):
            session.add(
                FutureDB(
                    request_type=types.RequestType.SAMPLE,
                    model_id="model_a",
                    request_data=_sample_request(),
                    status=RequestStatus.PENDING,
                )
            )
        session.commit()

    with Session(engine.db_engine) as session:
        samples = engine.find_batchable_sample(session)
        assert list(samples.keys()) == ["1", "2"]


async def _make_async_session_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


async def _seed_model(session: AsyncSession, model_id: str = "model_1") -> None:
    session.add(SessionDB(session_id="session_1", sdk_version="test"))
    session.add(
        ModelDB(
            model_id=model_id,
            base_model=BASE_MODEL,
            lora_config=types.LoraConfig(rank=8, alpha=16, seed=0).model_dump(),
            status="ready",
            request_id=1,
            session_id="session_1",
        )
    )
    await session.commit()


def test_create_checkpoint_resets_failed_row():
    """A failed checkpoint row is retried instead of permanently returning 409."""

    async def _run():
        engine = await _make_async_session_engine()
        async with AsyncSession(engine) as session:
            await _seed_model(session)
            session.add(
                CheckpointDB(
                    model_id="model_1",
                    checkpoint_id="global_step_0",
                    checkpoint_type=types.CheckpointType.TRAINING,
                    status=CheckpointStatus.FAILED,
                    error_message="checkpoint write failed",
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

            await api.create_checkpoint(
                session=session,
                model_id="model_1",
                checkpoint_id="global_step_0",
                checkpoint_type=types.CheckpointType.TRAINING,
            )
            await session.commit()

            checkpoint = await session.get(
                CheckpointDB,
                ("model_1", "global_step_0", types.CheckpointType.TRAINING),
            )
            assert checkpoint is not None
            assert checkpoint.status == CheckpointStatus.PENDING
            assert checkpoint.error_message is None
            assert checkpoint.completed_at is None

    asyncio.run(_run())


@pytest.mark.parametrize("status", [CheckpointStatus.PENDING, CheckpointStatus.COMPLETED])
def test_create_checkpoint_existing_active_or_completed_row_returns_409(status: CheckpointStatus):
    """Pending and completed checkpoint rows are not clobbered by retries."""

    async def _run():
        engine = await _make_async_session_engine()
        async with AsyncSession(engine) as session:
            await _seed_model(session)
            session.add(
                CheckpointDB(
                    model_id="model_1",
                    checkpoint_id="global_step_0",
                    checkpoint_type=types.CheckpointType.TRAINING,
                    status=status,
                    completed_at=datetime.now(timezone.utc) if status == CheckpointStatus.COMPLETED else None,
                )
            )
            await session.commit()

            with pytest.raises(HTTPException) as exc_info:
                await api.create_checkpoint(
                    session=session,
                    model_id="model_1",
                    checkpoint_id="global_step_0",
                    checkpoint_type=types.CheckpointType.TRAINING,
                )
            assert exc_info.value.status_code == 409

    asyncio.run(_run())


def test_create_checkpoint_missing_model_returns_404():
    """Missing model IDs still report 404 instead of checkpoint conflict."""

    async def _run():
        engine = await _make_async_session_engine()
        async with AsyncSession(engine) as session:
            with pytest.raises(HTTPException) as exc_info:
                await api.create_checkpoint(
                    session=session,
                    model_id="missing_model",
                    checkpoint_id="global_step_0",
                    checkpoint_type=types.CheckpointType.TRAINING,
                )
            assert exc_info.value.status_code == 404

    asyncio.run(_run())
