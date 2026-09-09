"""Architecture differences the mechanism experiments have to respect.

The ViT results are stated in ViT's vocabulary: 197 tokens in a flat sequence, a CLS token at
index 0, one spatial resolution throughout. Swin has none of that. Its blocks emit
(batch, height, width, channels), there is no class token at all (the head pools), and the
spatial grid is halved at each stage, so 224 pixels gives 56x56 tokens in stage 1 and 7x7 in
stage 4. A measurement written for ViT will run on Swin and return nonsense unless each of
those is handled.

This module is the seam. It reuses analysis.features for block discovery and the token
view, so a block index means the same thing here as it does to a probe placement, and adds the
two things the mechanism experiments additionally need: where the trigger's pixels land on a
block's own grid, and whether a CLS-based statistic is even defined.

The honest consequence for the Swin replication: `cls_routing` has no Swin analogue and is not
ported. The residual decomposition and the artifact-token statistics do port, because the
residual stream is additive and per-token norms are well defined on any grid.
"""

import torch
import torch.nn.functional as F

from analysis.features import (
    _as_token_sequence,
    detect_model_architecture,
    transformer_blocks,
)

__all__ = [
    "as_tokens",
    "block_grid",
    "detect_model_architecture",
    "has_class_token",
    "patch_token_slice",
    "transformer_blocks",
    "trigger_token_mask",
]


def as_tokens(activation: torch.Tensor) -> torch.Tensor:
    """(batch, tokens, dim), flattening a Swin block's spatial grid."""
    return _as_token_sequence(activation)


def has_class_token(architecture: str) -> bool:
    """Whether index 0 of the token axis is a class token rather than a patch."""
    return architecture == "vit"


def patch_token_slice(architecture: str) -> slice:
    """The tokens that correspond to image patches.

    ViT carries a class token at index 0 that is not a patch and must be excluded from any
    spatial statistic; Swin's tokens are all patches.
    """
    return slice(1, None) if has_class_token(architecture) else slice(None)


def block_grid(activation: torch.Tensor, architecture: str) -> tuple[int, int]:
    """The (height, width) token grid a block operates on.

    Swin states it directly in the activation shape. ViT's is square and implied by the token
    count once the class token is removed, and a non-square count means the caller hooked
    something that is not a block output.
    """
    if activation.dim() == 4:
        return activation.shape[1], activation.shape[2]
    count = activation.shape[1] - (1 if has_class_token(architecture) else 0)
    side = int(round(count**0.5))
    if side * side != count:
        raise ValueError(f"{count} patch tokens is not a square grid")
    return side, side


def trigger_token_mask(
    pixel_delta: torch.Tensor, grid: tuple[int, int], threshold: float = 1e-6
) -> torch.Tensor:
    """Which tokens of a (height, width) grid the trigger changes.

    Takes the per-pixel absolute change the trigger makes at the model's input resolution and
    pools it onto an arbitrary grid, so the same trigger can be mapped onto every stage of a
    hierarchical model rather than onto one fixed 14x14 layout. Returns a flat boolean mask in
    the same token order `as_tokens` produces, which is row-major.
    """
    height, width = grid
    pooled = F.adaptive_avg_pool2d(pixel_delta[None, None], (height, width))[0, 0]
    return pooled.reshape(-1) > threshold
