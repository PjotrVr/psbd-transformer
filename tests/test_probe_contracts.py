"""Contracts the PSBD probe must hold, independent of any implementation.

These were the half of the rewrite equivalence suite that never compared 2 trees.
They assert properties the probe has to have on its own terms, so they outlived the
comparison they were written beside:

  1. An operator at rate 0 is the identity, and every operator preserves the shape
     of whatever layout it is handed (tokens, spatial, or per-head).
  2. A deterministic perturbation needs exactly 1 forward pass, a stochastic one
     needs several, and Gaussian noise is scaled per sample rather than per batch.
  3. The cache rate tag keeps 0.01 through 0.09 distinct, which the on-disk layout
     depends on.
  4. Plugging a probe actually changes the forward pass, including at the 2
     positions that cannot be expressed as a hook and use a wrapper instead.

Run with pytest from the repo root: pytest tests/test_probe_contracts.py.
"""

import glob

import os

import pytest

import torch

import torch.nn as nn

import defences.cache as new_cache

import defences.decision as new_decision

import defences.operators as new_operators

import models.positions as new_positions


from models.backbones import load_checkpoint

CHECKPOINT_ROOT = "checkpoints"

RESULTS_ROOT = "results"

TOKEN_SHAPE = (3, 7, 16)

SPATIAL_SHAPE = (3, 4, 4, 16)

HEAD_SHAPE = (3, 4, 7, 8)

TOKEN_LAYOUT_OPERATORS = (
    "dropout",
    "channel_mask",
    "token_mask",
    "droppath",
    "gaussian",
    "gain_scale",
)

HEAD_LAYOUT_OPERATORS = ("head_mask",)


def _find_checkpoint(architecture: str) -> str | None:
    """The first real checkpoint of an architecture, by folder-name convention."""
    pattern = os.path.join(CHECKPOINT_ROOT, f"{architecture}_*", "attack_result.pt")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


@pytest.fixture(scope="module")
def vit_checkpoint() -> nn.Module:
    path = _find_checkpoint("vit")
    if path is None:
        pytest.skip("no ViT checkpoint on disk")
    return load_checkpoint("vit", path, torch.device("cpu"))


@pytest.fixture(scope="module")
def swin_checkpoint() -> nn.Module:
    path = _find_checkpoint("swin")
    if path is None:
        pytest.skip("no Swin checkpoint on disk")
    return load_checkpoint("swin", path, torch.device("cpu"))


def _state_dict_snapshot(model: nn.Module) -> dict:
    return {key: value.clone() for key, value in model.state_dict().items()}


def _assert_state_dict_identical(before: dict, after: dict) -> None:
    assert before.keys() == after.keys()
    for key in before:
        assert torch.equal(before[key], after[key]), f"{key} changed under plug/unplug"


def _plug_and_count(module, model: nn.Module, architecture: str, position: str) -> int:
    handles = module.plug_dropout(model, architecture, (position,), {}, rate=0.3)
    count = len(handles)
    module.unplug_dropout(handles)
    return count


def _sample_input(name: str, layout: str) -> torch.Tensor:
    if name in HEAD_LAYOUT_OPERATORS:
        return torch.randn(HEAD_SHAPE)
    shape = TOKEN_SHAPE if layout == "token" else SPATIAL_SHAPE
    return torch.randn(shape)


@pytest.mark.parametrize("name", sorted(new_operators.OPERATORS))
def test_operator_is_identity_at_rate_zero(name):
    operator = new_operators.build_operator(name)(0.0)
    operator.train()
    x = _sample_input(name, "token")

    assert torch.equal(operator(x), x)


def test_scale_up_is_identity_at_rate_zero():
    operator = new_operators.scale_up(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))(0.0)
    operator.train()
    x = torch.randn(2, 3, 8, 8)

    assert torch.equal(operator(x), x)


@pytest.mark.parametrize("name", TOKEN_LAYOUT_OPERATORS)
@pytest.mark.parametrize("layout", ("token", "spatial"))
def test_operator_preserves_shape(name, layout):
    operator = new_operators.build_operator(name)(0.3)
    operator.train()
    x = _sample_input(name, layout)

    assert operator(x).shape == x.shape


@pytest.mark.parametrize("name", HEAD_LAYOUT_OPERATORS)
def test_head_operator_preserves_shape(name):
    operator = new_operators.build_operator(name)(0.3)
    operator.train()
    x = torch.randn(HEAD_SHAPE)

    assert operator(x).shape == x.shape


OPERATORS_WITHOUT_AN_ORIGINAL = frozenset({"rademacher"})

PORTED_OPERATORS = sorted(set(new_operators.OPERATORS) - OPERATORS_WITHOUT_AN_ORIGINAL)


def test_gaussian_noise_std_is_per_sample():
    """The confound fix: one sample's noise level must not depend on its batch."""
    operator = new_operators.GaussianNoise(0.1)
    operator.train()

    quiet = torch.full((1, 7, 16), 0.01)
    loud = torch.full((1, 7, 16), 100.0)

    torch.manual_seed(0)
    alone = operator(quiet)
    torch.manual_seed(0)
    batched = operator(torch.cat([quiet, loud], dim=0))[:1]

    assert torch.equal(alone, batched)


def test_deterministic_perturbations_need_one_pass():
    assert new_operators.DETERMINISTIC_OPERATORS == frozenset(
        {"gain_scale", "scale_up"}
    )
    assert new_operators.effective_forward_passes("gain_scale", 20) == 1
    assert new_operators.effective_forward_passes("scale_up", 20) == 1
    assert new_operators.effective_forward_passes("dropout", 20) == 20


CACHE_RATES = (0.005, 0.01, 0.02, 0.05, 0.09, 0.1, 0.15, 0.25, 0.5, 0.9, 1.0)


def test_rate_tags_stay_distinct_below_one_tenth():
    tags = {new_cache._rate_tag(rate / 100) for rate in range(1, 10)}

    assert len(tags) == 9


def _uniform_probes(count: int, size: int, shift: float, seed: int) -> list:
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(size, generator=generator) - shift for _ in range(count)]


@pytest.mark.parametrize("k", (2, 3, 5))
@pytest.mark.parametrize("target_fpr", (0.05, 0.25))
def test_multi_probe_calibrated_rule_hits_the_target_fpr(k, target_fpr):
    """Uniform, independent probes: the calibrated rule must land on target."""
    validation = _uniform_probes(k, 8000, 0.0, seed=1)
    clean = _uniform_probes(k, 8000, 0.0, seed=2)
    backdoor = _uniform_probes(k, 8000, 0.3, seed=3)

    report = new_decision.multi_probe_detection(
        validation, clean, backdoor, target_fpr=target_fpr
    )

    assert report["fpr"] == report["by_rule"]["calibrated"]["fpr"]
    assert abs(report["by_rule"]["calibrated"]["fpr"] - target_fpr) < 0.02
    assert report["by_rule"]["bonferroni"]["fpr"] <= target_fpr


def test_multi_probe_rejects_an_unknown_rule():
    probes = _uniform_probes(2, 100, 0.0, seed=21)
    with pytest.raises(ValueError):
        new_decision.multi_probe_detection(probes, probes, probes, 0.25, "wishful")


def test_probe_perturbs_the_forward_pass_at_both_wrapper_positions(vit_checkpoint):
    """The 2 positions realized by a forward wrapper must actually take effect."""
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        clean_logits = vit_checkpoint(images)

    for position in ("after_attention_residual", "attention_heads"):
        handles = new_positions.plug_dropout(
            vit_checkpoint, "vit", (position,), {}, rate=0.5
        )
        with torch.inference_mode():
            perturbed_logits = vit_checkpoint(images)
        new_positions.unplug_dropout(handles)

        assert not torch.equal(perturbed_logits, clean_logits), position

        with torch.inference_mode():
            restored_logits = vit_checkpoint(images)
        assert torch.equal(restored_logits, clean_logits), position


def test_block_range_restricts_a_block_scope_position(vit_checkpoint):
    handles = new_positions.plug_dropout(
        vit_checkpoint, "vit", ("before_mlp",), {}, rate=0.3, block_range=(3, 6)
    )
    count = len(handles)
    new_positions.unplug_dropout(handles)

    assert count == 4

    with pytest.raises(ValueError):
        new_positions.plug_dropout(
            vit_checkpoint,
            "vit",
            ("after_embedding",),
            {},
            rate=0.3,
            block_range=(1, 2),
        )
