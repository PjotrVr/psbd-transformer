"""Per-layer residual-stream feature extraction for ViT-B/16 and Swin-S.

Every latent-space tool in this subpackage needs the same input: the residual
stream at each transformer block for a set of images. This module isolates that
extraction behind forward hooks so the rest of the analysis stays pure.

Layer indexing matches the Karayalcin et al. convention and the block numbering
psbd.positions uses. Index 0 is whatever is fed into the first block, and indices
1 to N are the outputs of the N blocks (12 for ViT-B/16, 24 for Swin-S). Blocks
are found the same way psbd.positions.resolve_targets finds them, by walking
model.modules() for the architecture's block type, so a layer index means the
same thing to a probe placement and to a feature analysis.

Because the hooks read whatever a block returns, they capture a post-residual
perturbation automatically when the post_residual placement is active, which is
exactly what the placement comparison needs.

The two architectures disagree on activation rank. A ViT block returns
(batch, tokens, dim), a Swin block returns (batch, height, width, channels), and
Swin has no classification token at all. Reduction handles both, and refuses the
combinations that have no meaning rather than returning a silently wrong shape.
"""

import torch
import torch.nn as nn

from psbd.inference import forward_probs
from psbd.models import network_core
from psbd.positions import BLOCK_TYPES

TokenReduction = str  # one of "cls", "mean", "flatten"


def detect_model_architecture(model: nn.Module) -> str:
    """Name the architecture of a live model from the block types it contains.

    psbd.models.detect_architecture answers the same question from a checkpoint
    path. This is the in-memory counterpart, needed because a notebook or a test
    often holds a model it never loaded from disk.
    """
    matched = [
        architecture
        for architecture, block_types in BLOCK_TYPES.items()
        if any(isinstance(module, block_types) for module in model.modules())
    ]
    if len(matched) != 1:
        raise ValueError(
            f"model matched architectures {matched}, expected exactly one of "
            f"{sorted(BLOCK_TYPES)}"
        )

    architecture = matched[0]
    return architecture


def transformer_blocks(model: nn.Module, architecture: str) -> list[nn.Module]:
    """The architecture's transformer blocks in forward order, 0-indexed here.

    Same walk psbd.positions.resolve_targets performs, so block i in this list is
    the block a probe restricted to block_range (i + 1, i + 1) would attach to.
    """
    if architecture not in BLOCK_TYPES:
        raise ValueError(f"Unknown architecture: {architecture}")

    block_types = BLOCK_TYPES[architecture]
    blocks = [module for module in model.modules() if isinstance(module, block_types)]
    if not blocks:
        raise ValueError(
            f"no {block_types} blocks found in the model, so no features would be "
            "captured and every layer statistic would be empty"
        )

    return blocks


def _as_token_sequence(activation: torch.Tensor) -> torch.Tensor:
    """View an activation as (batch, tokens, dim) whatever rank it arrived at.

    A Swin block emits (batch, height, width, channels); its spatial grid is the
    token axis, so flattening the two spatial axes recovers the ViT layout without
    moving any data semantically.
    """
    if activation.dim() == 3:
        return activation
    if activation.dim() == 4:
        batch, height, width, channels = activation.shape
        flattened = activation.reshape(batch, height * width, channels)
        return flattened
    raise ValueError(
        f"expected a rank-3 or rank-4 activation, got shape {tuple(activation.shape)}"
    )


def _reduce_tokens(
    activation: torch.Tensor, reduction: TokenReduction, has_class_token: bool
) -> torch.Tensor:
    """Collapse the token axis of a residual-stream tensor to (batch, dim).

    "cls" keeps token 0, the classification token whose final state drives the
    prediction, and is available only on an architecture that has one. "mean"
    averages tokens and is the Swin default for that reason. "flatten" keeps all
    tokens and is memory heavy, so use it only for small sample counts.

    The has_class_token guard is the point of this function. Swin's activation is
    (batch, height, width, channels), so an unguarded activation[:, 0, :] returns
    the first image row rather than a class token: a wrong number with a plausible
    shape, which is worse than an exception.
    """
    # Cast before reducing, not after. Swin's grid is 3136 tokens at the first
    # stage, and summing that many values in bfloat16 costs about 1.75e-3 relative
    # error against 3.2e-5 when the cast comes first.
    tokens = _as_token_sequence(activation).float()

    if reduction == "cls":
        if not has_class_token:
            raise ValueError(
                "reduction 'cls' needs a classification token; this architecture "
                "has none, so use 'mean' (its pooling matches the trained head)"
            )
        class_token = tokens[:, 0, :]  # (batch, dim)
        return class_token
    if reduction == "mean":
        pooled = tokens.mean(dim=1)  # (batch, dim)
        return pooled
    if reduction == "flatten":
        flattened = tokens.flatten(1)  # (batch, tokens * dim)
        return flattened
    raise ValueError(f"Unknown token reduction: {reduction}")


def default_reduction(architecture: str) -> TokenReduction:
    """The token reduction that matches how the architecture's head reads features.

    ViT classifies from the class token, Swin from a mean over the spatial grid,
    so those are the reductions whose features the trained head actually consumes.
    """
    reduction = "cls" if architecture == "vit" else "mean"
    return reduction


def _make_block_hook(
    storage: dict, layer_index: int, reduction: TokenReduction, has_class_token: bool
):
    def hook(_module, _inputs, output):
        reduced = _reduce_tokens(output, reduction, has_class_token)
        storage.setdefault(layer_index, []).append(reduced.detach().float().cpu())

    return hook


def _make_embedding_hook(
    storage: dict, reduction: TokenReduction, has_class_token: bool
):
    def pre_hook(_module, inputs):
        reduced = _reduce_tokens(inputs[0], reduction, has_class_token)
        storage.setdefault(0, []).append(reduced.detach().float().cpu())

    return pre_hook


@torch.inference_mode()
def extract_layer_features(
    model: nn.Module,
    loader,
    device: torch.device,
    use_bfloat16: bool,
    reduction: TokenReduction | None = None,
    architecture: str | None = None,
) -> dict[int, torch.Tensor]:
    """Return a dict mapping layer index to a float32 (num_samples, dim) tensor.

    architecture and reduction both default to whatever the model itself implies,
    so a caller holding only a model does not have to restate what it is.

    The forward pass is run only to trigger the hooks, so its probabilities are
    discarded. Handles are always removed, even if the loader raises.
    """
    core = network_core(model)
    resolved_architecture = (
        architecture if architecture is not None else detect_model_architecture(core)
    )
    resolved_reduction = (
        reduction if reduction is not None else default_reduction(resolved_architecture)
    )
    has_class_token = resolved_architecture == "vit"

    blocks = transformer_blocks(core, resolved_architecture)
    storage: dict[int, list[torch.Tensor]] = {}

    # Layer 0 is the first block's input rather than the block stack's, which is
    # the same tensor for both architectures and needs no per-architecture path.
    handles = [
        blocks[0].register_forward_pre_hook(
            _make_embedding_hook(storage, resolved_reduction, has_class_token)
        )
    ]
    for offset, block in enumerate(blocks, start=1):
        handles.append(
            block.register_forward_hook(
                _make_block_hook(storage, offset, resolved_reduction, has_class_token)
            )
        )

    try:
        for images, _ in loader:
            forward_probs(model, images, device, use_bfloat16)
    finally:
        for handle in handles:
            handle.remove()

    features_by_layer = {
        layer: torch.cat(chunks, dim=0) for layer, chunks in sorted(storage.items())
    }
    return features_by_layer
