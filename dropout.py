"""Dropout placement inside ViT-B/16 and Swin-S, the central variable of study.

pre_residual toggles the Dropout modules the transformer already contains, all
of which sit on a block's branch before the residual add. It perturbs only the
increment a block contributes and leaves the accumulated residual stream intact.
It is architecture agnostic, since it only needs to find Dropout modules.

post_residual wraps each transformer block so a fresh Dropout runs after each
residual add, perturbing the accumulated stream directly. This mirrors the
placement the original PSBD paper used inside ResNet basic blocks, and needs an
architecture-specific wrapper because ViT and Swin blocks differ in structure.

The empirical result is that pre_residual separates clean from backdoor PSU on
ViT-B/16 while post_residual mostly does not, which the CKA homogeneity of ViT
residual streams (Raghu et al., 2021) plausibly explains.
"""

import torch.nn as nn
from torchvision.models.vision_transformer import EncoderBlock

try:
    from torchvision.models.swin_transformer import SwinTransformerBlock, SwinTransformerBlockV2

    _SWIN_BLOCK_TYPES = (SwinTransformerBlock, SwinTransformerBlockV2)
except ImportError:  # older torchvision without the V2 block
    from torchvision.models.swin_transformer import SwinTransformerBlock

    _SWIN_BLOCK_TYPES = (SwinTransformerBlock,)


def configure_pre_residual_dropout(model: nn.Module, rate: float) -> int:
    """Set the rate on every existing Dropout and switch it to train mode.

    A Dropout only samples a mask in train mode, so eval mode with rate 0 is the
    no-dropout baseline. Returns the number of Dropout modules touched.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = float(rate)
            module.train() if rate > 0.0 else module.eval()
            count += 1
    return count


class PostResidualEncoderBlock(nn.Module):
    """Re-runs a ViT encoder block and adds Dropout after each residual add."""

    def __init__(self, base_block: nn.Module, rate: float):
        super().__init__()
        self.base_block = base_block
        self.attention_dropout = nn.Dropout(p=rate)
        self.mlp_dropout = nn.Dropout(p=rate)

    def forward(self, input_tensor):
        block = self.base_block
        x = block.ln_1(input_tensor)
        x, _ = block.self_attention(x, x, x, need_weights=False)
        x = block.dropout(x)  # the block's own pre-residual dropout
        x = input_tensor + x  # first residual add
        x = self.attention_dropout(x)

        y = block.ln_2(x)
        y = block.mlp(y)
        x = x + y  # second residual add
        x = self.mlp_dropout(x)
        return x


class PostResidualSwinBlock(nn.Module):
    """Re-runs a Swin block and adds Dropout after each residual add.

    Swin regularizes the branch with stochastic depth rather than per-activation
    dropout, so the post-residual Dropout added here is a genuinely new
    perturbation the model never saw in training.
    """

    def __init__(self, base_block: nn.Module, rate: float):
        super().__init__()
        self.base_block = base_block
        self.attention_dropout = nn.Dropout(p=rate)
        self.mlp_dropout = nn.Dropout(p=rate)

    def forward(self, input_tensor):
        block = self.base_block
        x = input_tensor + block.stochastic_depth(block.attn(block.norm1(input_tensor)))
        x = self.attention_dropout(x)
        x = x + block.stochastic_depth(block.mlp(block.norm2(x)))
        x = self.mlp_dropout(x)
        return x


def _wrap_as_post_residual(module: nn.Module, rate: float):
    """Return a post-residual wrapper for a known block type, else None.

    Already-wrapped blocks are re-wrapped at the new rate so a rate sweep stays
    safe to call repeatedly.
    """
    if isinstance(module, PostResidualEncoderBlock):
        return PostResidualEncoderBlock(module.base_block, rate)
    if isinstance(module, PostResidualSwinBlock):
        return PostResidualSwinBlock(module.base_block, rate)
    if isinstance(module, EncoderBlock):
        return PostResidualEncoderBlock(module, rate)
    if isinstance(module, _SWIN_BLOCK_TYPES):
        return PostResidualSwinBlock(module, rate)
    return None


def _unwrap_post_residual(module: nn.Module):
    if isinstance(module, (PostResidualEncoderBlock, PostResidualSwinBlock)):
        return module.base_block
    return None


def _replace_blocks(parent: nn.Module, make_replacement) -> None:
    """Walk the module tree and swap any block the factory recognizes.

    Recursion stops at a replaced block, so its wrapped base is never revisited.
    Reassigning by child name works for the Sequential containers ViT and Swin
    use, whose children are named "0", "1", and so on.
    """
    for name, child in list(parent.named_children()):
        replacement = make_replacement(child)
        if replacement is not None:
            setattr(parent, name, replacement)
        else:
            _replace_blocks(child, make_replacement)


def configure_post_residual_dropout(model: nn.Module, rate: float) -> None:
    _replace_blocks(model, lambda module: _wrap_as_post_residual(module, rate))


def remove_post_residual_dropout(model: nn.Module) -> None:
    _replace_blocks(model, _unwrap_post_residual)


def configure_dropout(model: nn.Module, rate: float, placement: str) -> None:
    """Dispatch to the placement selected in RunConfig."""
    if placement == "pre_residual":
        configure_pre_residual_dropout(model, rate)
    elif placement == "post_residual":
        configure_post_residual_dropout(model, rate)
    else:
        raise ValueError(f"Unknown dropout placement: {placement}")


def reset_dropout(model: nn.Module, placement: str) -> None:
    """Return the model to its no-dropout inference state."""
    if placement == "post_residual":
        remove_post_residual_dropout(model)
    configure_pre_residual_dropout(model, rate=0.0)
