"""Structural tests for dropout placement, the central variable of study.

Verifies the pre_residual/post_residual split is precise, not just present:
pre_residual must touch exactly the 36 true per-block Dropout modules (not
the embedding dropout), and the two placements must not leak into each
other when only one is requested.
"""

import pytest
import torch.nn as nn
from torchvision.models.vision_transformer import EncoderBlock

from defences.dropout import (
    PostResidualEncoderBlock,
    configure_dropout,
    configure_pre_residual_dropout,
    reset_dropout,
)
from models import build_swin, build_vit


@pytest.fixture
def vit() -> nn.Module:
    return build_vit(num_classes=4)


def _dropout_modules(model: nn.Module) -> dict[str, nn.Dropout]:
    return {name: module for name, module in model.named_modules() if isinstance(module, nn.Dropout)}


def test_pre_residual_touches_36_not_37(vit):
    touched = configure_pre_residual_dropout(vit, rate=0.3)
    assert touched == 36, "should touch the 36 per-block dropouts, not the embedding dropout too"


def test_pre_residual_excludes_embedding_dropout(vit):
    configure_pre_residual_dropout(vit, rate=0.3)
    modules = _dropout_modules(vit)
    embedding_dropout = [module for name, module in modules.items() if name.endswith("encoder.dropout")]
    assert len(embedding_dropout) == 1, "ViT should have exactly one embedding dropout module"
    assert embedding_dropout[0].p == 0.0, "embedding dropout must stay untouched by the pre_residual sweep"

    per_block = [module for name, module in modules.items() if not name.endswith("encoder.dropout")]
    assert len(per_block) == 36
    assert all(module.p == 0.3 for module in per_block), "every per-block dropout must get the new rate"


def test_reset_dropout_zeroes_rate_and_eval_mode(vit):
    configure_pre_residual_dropout(vit, rate=0.5)
    reset_dropout(vit, placement="pre_residual")
    # encoder.dropout is out of scope by design (see test_pre_residual_excludes_
    # embedding_dropout), so its train/eval mode is whatever a fresh module
    # defaults to, not something reset_dropout is responsible for.
    for name, module in _dropout_modules(vit).items():
        if name.endswith("encoder.dropout"):
            continue
        assert module.p == 0.0, f"{name} should be reset to rate 0"
        assert not module.training, f"{name} should be back in eval mode"


def test_post_residual_wraps_blocks_pre_residual_does_not(vit):
    encoder_layers = vit[1].encoder.layers
    assert all(isinstance(block, EncoderBlock) for block in encoder_layers), "sanity check on the fresh model"

    configure_dropout(vit, rate=0.3, placement="pre_residual")
    assert all(isinstance(block, EncoderBlock) for block in encoder_layers), (
        "pre_residual must not wrap blocks, only toggle existing Dropout modules"
    )

    configure_dropout(vit, rate=0.3, placement="post_residual")
    assert all(isinstance(block, PostResidualEncoderBlock) for block in encoder_layers), (
        "post_residual must wrap every encoder block"
    )


def test_reset_dropout_unwraps_post_residual(vit):
    encoder_layers = vit[1].encoder.layers
    configure_dropout(vit, rate=0.3, placement="post_residual")
    reset_dropout(vit, placement="post_residual")
    assert all(isinstance(block, EncoderBlock) for block in encoder_layers), (
        "reset_dropout(placement=post_residual) must unwrap blocks back to plain EncoderBlock"
    )


def test_pre_residual_exclusion_is_a_no_op_for_swin():
    # Swin has no module named "*.encoder.dropout", so the ViT-specific
    # exclusion in configure_pre_residual_dropout must not drop any of Swin's
    # own dropouts.
    swin = build_swin(num_classes=4)
    total_dropouts = sum(1 for _, module in swin.named_modules() if isinstance(module, nn.Dropout))
    touched = configure_pre_residual_dropout(swin, rate=0.3)
    assert touched == total_dropouts, "the embedding-dropout exclusion should never trigger on Swin"
