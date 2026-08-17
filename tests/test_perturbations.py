"""Mechanical checks on perturbation operators (token_mask, channel_mask,
gaussian, droppath, head_mask, gain_scale)."""

import pytest
import torch
import torch.nn as nn

import models
from defences.dropout import STRUCTURED_POSITION_NAMES, plug_dropout, unplug_dropout
from defences.perturbations import (
    PERTURBATIONS,
    DropPath,
    GaussianNoise,
    GroupChannelMask,
    TokenMask,
    build_perturbation,
    head_mask,
)


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(8, 197, 768)


@pytest.mark.parametrize("name", list(PERTURBATIONS.keys()))
def test_eval_mode_is_identity(x, name):
    op = build_perturbation(name)(0.5)
    op.eval()
    assert torch.equal(op(x), x)


@pytest.mark.parametrize("name", ("dropout", "channel_mask", "token_mask", "droppath"))
def test_inverted_scaling_preserves_mean(x, name):
    op = build_perturbation(name)(0.5)
    op.train()
    out = torch.stack([op(x) for _ in range(64)]).mean(0)
    rel = (out.mean() - x.mean()).abs() / x.abs().mean()
    assert rel < 0.05, f"relative mean drift {rel:.4f}"


def test_head_mask_all_or_nothing():
    torch.manual_seed(0)
    per_head = torch.randn(8, 12, 197, 64)
    op = head_mask(0.5)
    op.train()
    out = op(per_head)
    zeroed = (out == 0).all(dim=3).all(dim=2)
    dropped = zeroed.float().mean()
    assert 0.3 < dropped < 0.7, f"fraction dropped {dropped:.3f}"


def test_token_mask_preserves_cls(x):
    op = TokenMask(0.9)
    op.train()
    out = op(x)
    assert (out[:, 0, :] != 0).any(dim=1).all()


def test_droppath_per_sample_all_or_nothing(x):
    op = DropPath(0.5)
    op.train()
    out = op(x)
    per_sample_zero = (out == 0).all(dim=2).all(dim=1)
    per_sample_nonzero = (out != 0).any(dim=2).any(dim=1)
    assert (per_sample_zero | per_sample_nonzero).all()


@pytest.mark.parametrize("name", list(PERTURBATIONS.keys()))
def test_rate_zero_is_identity(x, name):
    op = build_perturbation(name)(0.0)
    op.train()
    assert torch.equal(op(x), x)


@pytest.mark.parametrize("position", STRUCTURED_POSITION_NAMES)
def test_structured_position_attach_detach(position):
    torch.manual_seed(0)
    model = models.build_vit(10)
    model.eval()
    probe = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        before = model(probe).clone()

    operator = head_mask if position == "attention_heads" else nn.Dropout
    handles = plug_dropout(model, "vit", (position,), {position: operator}, 0.5)
    assert len(handles) == 12

    with torch.no_grad():
        perturbed = model(probe)
    assert not torch.allclose(before, perturbed)

    unplug_dropout(handles)
    with torch.no_grad():
        after = model(probe)
    assert torch.equal(before, after)


def test_rate_one_rejected():
    with pytest.raises(ValueError):
        op = GroupChannelMask(1.0)
        op.train()
        op(torch.randn(8, 197, 768))


def test_gaussian_scales_with_magnitude(x):
    torch.manual_seed(0)
    op = GaussianNoise(0.1)
    op.train()
    small = op(x * 0.01) - x * 0.01
    torch.manual_seed(0)
    op2 = GaussianNoise(0.1)
    op2.train()
    large = op2(x * 100.0) - x * 100.0
    ratio = large.std() / small.std()
    assert 5_000 < ratio < 20_000, f"ratio {ratio:.0f}"
