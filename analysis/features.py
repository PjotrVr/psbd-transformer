"""Per-layer residual-stream feature extraction for ViT-B/16.

Every latent-space tool in this subpackage needs the same input: the residual
stream at each encoder block for a set of images. This module isolates that
extraction behind forward hooks so the rest of the analysis stays pure.

Layer indexing matches the Karayalcin et al. convention. Index 0 is the token
embedding fed into the first block, and indices 1 to 12 are the outputs of the
12 encoder blocks. Because the hooks read whatever a block returns, they capture
the post-residual dropout automatically when the post_residual placement is
active, which is exactly what the placement comparison needs.
"""

import torch
import torch.nn as nn

from defences.inference import forward_probs
from models import vit_core

TokenReduction = str  # one of "cls", "mean", "flatten"


def _reduce_tokens(activation: torch.Tensor, reduction: TokenReduction) -> torch.Tensor:
    """Collapse the token axis of a [batch, tokens, dim] residual-stream tensor.

    "cls" keeps token 0, the classification token whose final state drives the
    prediction. "mean" averages tokens, useful for Swin which has no CLS token.
    "flatten" keeps all tokens and is memory heavy, so use it only for small
    sample counts.
    """
    if reduction == "cls":
        return activation[:, 0, :]
    if reduction == "mean":
        return activation.mean(dim=1)
    if reduction == "flatten":
        return activation.flatten(1)
    raise ValueError(f"Unknown token reduction: {reduction}")


def _make_block_hook(storage: dict, layer_index: int, reduction: TokenReduction):
    def hook(_module, _inputs, output):
        reduced = _reduce_tokens(output, reduction)
        storage.setdefault(layer_index, []).append(reduced.detach().float().cpu())

    return hook


def _make_embedding_hook(storage: dict, reduction: TokenReduction):
    def pre_hook(_module, inputs):
        reduced = _reduce_tokens(inputs[0], reduction)
        storage.setdefault(0, []).append(reduced.detach().float().cpu())

    return pre_hook


@torch.inference_mode()
def extract_layer_features(
    model: nn.Module,
    loader,
    device: torch.device,
    use_bfloat16: bool,
    reduction: TokenReduction = "cls",
) -> dict[int, torch.Tensor]:
    """Return a dict mapping layer index to a float32 [num_samples, dim] tensor.

    The forward pass is run only to trigger the hooks, so its probabilities are
    discarded. Handles are always removed, even if the loader raises.
    """
    encoder_blocks = vit_core(model).encoder.layers
    storage: dict[int, list[torch.Tensor]] = {}
    handles = [encoder_blocks.register_forward_pre_hook(_make_embedding_hook(storage, reduction))]
    for offset, block in enumerate(encoder_blocks, start=1):
        handles.append(block.register_forward_hook(_make_block_hook(storage, offset, reduction)))

    try:
        for images, _ in loader:
            forward_probs(model, images, device, use_bfloat16)
    finally:
        for handle in handles:
            handle.remove()

    return {layer: torch.cat(chunks, dim=0) for layer, chunks in sorted(storage.items())}
