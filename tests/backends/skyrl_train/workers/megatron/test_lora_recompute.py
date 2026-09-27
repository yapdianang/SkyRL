import pytest
import torch
from torch.utils.checkpoint import checkpoint

from skyrl.backends.skyrl_train.workers.megatron.lora_recompute import (
    require_embedding_output_grad,
)


class _Chunk(torch.nn.Module):
    def __init__(self, pre_process: bool = True):
        super().__init__()
        self.pre_process = pre_process
        self.embedding = torch.nn.Embedding(8, 4).requires_grad_(False)
        self.adapter = torch.nn.Linear(4, 4)

    def forward(self, ids):
        return checkpoint(self.adapter, self.embedding(ids), use_reentrant=True)


def test_reentrant_recompute_skips_adapters_without_embedding_grad():
    assert not _Chunk()(torch.tensor([[1, 2, 3]])).requires_grad


def test_reentrant_recompute_reaches_adapters_with_embedding_grad():
    chunk = _Chunk()
    require_embedding_output_grad([chunk])
    chunk(torch.tensor([[1, 2, 3]])).sum().backward()
    assert chunk.adapter.weight.grad is not None and chunk.adapter.weight.grad.abs().sum() > 0
    assert chunk.embedding.weight.grad is None


def test_later_pipeline_stages_are_skipped():
    chunk = _Chunk(pre_process=False)
    del chunk.embedding
    require_embedding_output_grad([chunk])


def test_first_stage_without_embedding_raises():
    chunk = _Chunk()
    del chunk.embedding
    with pytest.raises(ValueError, match="no `embedding`"):
        require_embedding_output_grad([chunk])
