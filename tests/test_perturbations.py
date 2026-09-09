"""Mechanical checks on perturbation operators (token_mask, channel_mask,
gaussian, droppath, head_mask, gain_scale)."""

import pytest
import torch
import torch.nn as nn

import models.backbones
from models.positions import STRUCTURED_POSITION_NAMES, plug_dropout, unplug_dropout
from defences.operators import (
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
    model = models.backbones.build_vit(10)
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


def test_head_mask_granularity_is_the_whole_head():
    """A dropped head is zeroed across every token and channel, not elementwise.

    Inverted scaling divides survivors by (1 - rate), so a surviving entry is
    only 0 if the input was, which randn does not produce. Any (batch, head)
    slice holding both zeros and non-zeros would mean the mask was drawn per
    element rather than per head.
    """
    torch.manual_seed(0)
    per_head = torch.randn(8, 12, 197, 64)
    op = head_mask(0.5)
    op.train()
    out = op(per_head)

    zeroed_all = (out == 0).all(dim=3).all(dim=2)  # (batch, heads)
    zeroed_any = (out == 0).any(dim=3).any(dim=2)
    assert torch.equal(zeroed_all, zeroed_any)


def test_near_total_head_mask_moves_the_output():
    """Masking essentially every head in every block has to destroy attention.

    The counterpart to the attach/detach test: that one proves the hook fires,
    this one proves the hook removes something the model was using.
    """
    torch.manual_seed(0)
    model = models.backbones.build_vit(10)
    model.eval()
    probe = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        before = model(probe).clone()

    handles = plug_dropout(
        model, "vit", ("attention_heads",), {"attention_heads": head_mask}, 0.99
    )
    with torch.no_grad():
        heavy = model(probe)
    unplug_dropout(handles)

    assert (heavy - before).abs().max() > 1e-3


class TestOperatorPositionValidation:
    """Combinations that run happily and answer a different question than asked.

    Each case below produces a complete sweep with no error, so nothing downstream
    can tell the result apart from a valid cell. That is why these raise rather
    than warn.
    """

    def test_token_operators_are_refused_on_the_raw_image(self):
        """input_pixels is (batch, channels, height, width), not Swin's layout.

        The token operators read a rank-4 tensor as (batch, height, width,
        channels), so on a raw image token_mask masks image rows independently per
        colour channel and channel_mask masks image columns. Both run, and both
        perturb an axis other than the one their name claims.
        """
        from defences.operators import check_operator_position

        for operator in ("token_mask", "channel_mask", "head_mask", "droppath"):
            with pytest.raises(ValueError, match="input_pixels"):
                check_operator_position(operator, "input_pixels")

    def test_the_wrong_axis_reading_is_real_and_not_hypothetical(self):
        """Demonstrates the misread the guard exists to prevent."""
        from defences.operators import build_perturbation

        torch.manual_seed(0)
        image = torch.ones(2, 3, 8, 8)  # (batch, channels, height, width)
        masked = build_perturbation("token_mask")(0.5).train()(image)

        # Read as (batch, height, width, channels), whole "tokens" are zeroed, so
        # entire colour-channel planes vanish rather than image patches.
        zeroed_planes = (masked == 0).all(dim=3).any().item()
        assert zeroed_planes, "the misread should zero along the wrong axis"

    def test_an_inert_probe_is_refused_rather_than_reported_as_no_effect(self):
        """token_mask after the final norm cannot reach anything ViT's head reads.

        The head reads token 0 alone and token_mask protects token 0, so the probe
        is a no-op. PSU would be identically 0 and AUROC exactly 0.5, which reads
        as "this position does not matter" rather than "this probe never fired".
        """
        from defences.operators import check_operator_position

        with pytest.raises(ValueError, match="inert|only token 0|final_norm_out"):
            check_operator_position("token_mask", "final_norm_out")

    def test_head_mask_is_refused_away_from_the_head_axis(self):
        from defences.operators import check_operator_position

        with pytest.raises(ValueError, match="only meaningful"):
            check_operator_position("head_mask", "before_mlp")

    def test_valid_pairs_are_allowed(self):
        """The guard must not reject the configurations the project actually runs."""
        from defences.operators import check_operator_position

        for operator, position in (
            ("token_mask", "before_attention_norm"),
            ("token_mask", "before_mlp"),
            ("gaussian", "input_pixels"),
            ("rademacher", "before_attention_norm"),
            ("dropout", "pre_residual"),
            ("head_mask", "attention_heads"),
            ("droppath", "before_mlp_residual"),
            ("gain_scale", "mlp_norm_out"),
        ):
            check_operator_position(operator, position)
