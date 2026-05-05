"""Megatron HF-to-distributed-checkpoint cache hook."""

import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch.distributed as dist
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.core import dist_checkpointing


def _dist_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _resolve_cache_dir(hf_pretrained: Any) -> Path | None:
    base = os.environ.get("MEGATRON_CACHE_DIR")
    if not base:
        return None

    name = getattr(hf_pretrained, "model_name_or_path", None) or getattr(
        hf_pretrained, "name_or_path", None
    )
    if not name:
        return None

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    try:
        from megatron.core import parallel_state as ps

        tp = ps.get_tensor_model_parallel_world_size() if ps.is_initialized() else 1
        pp = ps.get_pipeline_model_parallel_world_size() if ps.is_initialized() else 1
        ep = ps.get_expert_model_parallel_world_size() if ps.is_initialized() else 1
    except Exception:
        tp = pp = ep = 1

    return Path(base) / slug / f"tp{tp}_pp{pp}_ep{ep}"


def _checkpoint_exists(path: Path) -> bool:
    if not path.is_dir() or not (path / ".cache_complete").is_file():
        return False
    if not (path / ".metadata").is_file():
        return False
    return len(list(path.glob("*.distcp"))) >= _dist_world_size()


def _model_sharded_state_dict(megatron_model: Any) -> dict[str, Any]:
    models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
    if len(models) == 1:
        return {"model": models[0].sharded_state_dict()}
    return {f"model{i}": model.sharded_state_dict() for i, model in enumerate(models)}


def _install_patch() -> None:
    current = MegatronModelBridge.load_weights_hf_to_megatron
    if getattr(current, "_skyrl_megatron_cache_patched", False):
        return

    original = current

    def _patched(
        self: Any, hf_pretrained: Any, megatron_model: Any, **kwargs: Any
    ) -> Any:
        cache_dir = _resolve_cache_dir(hf_pretrained)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        if cache_dir is not None and _checkpoint_exists(cache_dir):
            try:
                if rank == 0:
                    print(f"[megatron-cache] hit -> {cache_dir}", flush=True)
                sharded_state_dict = _model_sharded_state_dict(megatron_model)
                loaded = dist_checkpointing.load(sharded_state_dict, str(cache_dir))
                models = (
                    megatron_model
                    if isinstance(megatron_model, list)
                    else [megatron_model]
                )
                for i, model in enumerate(models):
                    key = "model" if len(models) == 1 else f"model{i}"
                    model.load_state_dict(loaded[key], strict=False)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                if rank == 0:
                    print(f"[megatron-cache] loaded {cache_dir}", flush=True)
                return megatron_model
            except Exception as exc:
                if rank == 0:
                    print(
                        f"[megatron-cache] load failed; falling back to HF conversion: {exc!r}",
                        flush=True,
                    )

        if rank == 0:
            print(f"[megatron-cache] miss; converting (cache={cache_dir})", flush=True)
        out = original(self, hf_pretrained, megatron_model, **kwargs)
        if cache_dir is None:
            return out

        try:
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_dir.with_name(cache_dir.name + ".tmp")
            if rank == 0:
                if tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True)
                if cache_dir.exists() and not _checkpoint_exists(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)
                tmp.mkdir(parents=True, exist_ok=True)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            sharded_state_dict = _model_sharded_state_dict(megatron_model)
            dist_checkpointing.save(sharded_state_dict, str(tmp))
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            if rank == 0:
                if not _checkpoint_exists(cache_dir):
                    tmp.rename(cache_dir)
                    (cache_dir / ".cache_complete").write_text("ok\n")
                    print(f"[megatron-cache] saved {cache_dir}", flush=True)
                else:
                    shutil.rmtree(tmp, ignore_errors=True)
                    print(
                        f"[megatron-cache] another group already wrote {cache_dir}; discarded {tmp}",
                        flush=True,
                    )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception as exc:
            if rank == 0:
                print(f"[megatron-cache] save failed (continuing): {exc!r}", flush=True)
        return out

    _patched._skyrl_megatron_cache_patched = True
    MegatronModelBridge.load_weights_hf_to_megatron = _patched
    print(
        "[megatron-cache] patched MegatronModelBridge.load_weights_hf_to_megatron",
        flush=True,
    )


_install_patch()
