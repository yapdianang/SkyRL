from typing import List

import torch


def require_embedding_output_grad(model_chunks: List[torch.nn.Module]) -> None:
    """Make embedding outputs require grad so recomputed layers backpropagate into LoRA adapters."""
    for chunk in model_chunks:
        if not chunk.pre_process:
            continue
        embedding = getattr(chunk, "embedding", None)
        if embedding is None:
            raise ValueError(
                f"{type(chunk).__name__} has no `embedding` on the first pipeline stage; "
                "LoRA with gradient checkpointing would train with zero adapter gradients."
            )
        embedding.register_forward_hook(lambda _module, _inputs, output: output.requires_grad_(True))
