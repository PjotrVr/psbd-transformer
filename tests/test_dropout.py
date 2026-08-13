"""Structural tests for the hook-based dropout registry.

The mechanism's invariant is that plugging a position only adds forward hooks (or
one swapped forward, for the residual position) and never mutates the model:
unplug must leave the state_dict byte-identical, and the attachment count must
match the registry (one per targeted block, or one at model level). The Swin
shared-stochastic_depth finding, the reason its before_*_residual positions hook
attn/mlp directly, is confirmed here too.

after_attention_residual gets extra scrutiny because it is the only position
realized by swapping a block's forward rather than by a hook. Two properties make
that swap trustworthy: at rate 0 it must reproduce torchvision's own forward
bit-for-bit (so the rewrite introduced no algebraic drift), and it must perturb
strictly more than a pre-hook on ln_2 does (so it really reaches the skip path,
which is the entire reason it exists).
"""

import pytest
import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformerBlock

from defences.dropout import (
    DROPOUT_CONFIGS,
    SINGLE_POSITION_NAMES,
    plug_dropout,
    unplug_dropout,
)
from models import build_swin, build_vit


@pytest.fixture(scope="module")
def vit() -> nn.Module:
    return build_vit(num_classes=10)


@pytest.fixture(scope="module")
def swin() -> nn.Module:
    return build_swin(num_classes=10)


# after_embedding is the one model-level position, so it adds exactly 1
# attachment; every other position is per-block, so it adds one per block.
VIT_HOOKS_PER_POSITION = {name: 12 for name in SINGLE_POSITION_NAMES}
VIT_HOOKS_PER_POSITION["after_embedding"] = 1

SWIN_HOOKS_PER_POSITION = {name: 24 for name in SINGLE_POSITION_NAMES}
SWIN_HOOKS_PER_POSITION["after_embedding"] = 1


def _state_dict_bytes(model: nn.Module) -> dict:
    return {key: value.clone() for key, value in model.state_dict().items()}


def _assert_state_dict_identical(before: dict, after: dict) -> None:
    assert before.keys() == after.keys()
    for key in before:
        assert torch.equal(before[key], after[key]), f"{key} changed under plug/unplug"


@pytest.mark.parametrize("position", SINGLE_POSITION_NAMES)
def test_vit_plug_registers_expected_hook_count(vit, position):
    handles = plug_dropout(vit, "vit", (position,), {}, rate=0.3)
    assert len(handles) == VIT_HOOKS_PER_POSITION[position]
    unplug_dropout(handles)


@pytest.mark.parametrize("position", SINGLE_POSITION_NAMES)
def test_swin_plug_registers_expected_hook_count(swin, position):
    handles = plug_dropout(swin, "swin", (position,), {}, rate=0.3)
    assert len(handles) == SWIN_HOOKS_PER_POSITION[position]
    unplug_dropout(handles)


@pytest.mark.parametrize("config", sorted(DROPOUT_CONFIGS))
def test_vit_combo_hook_count_is_the_sum(vit, config):
    positions = DROPOUT_CONFIGS[config]
    expected = sum(VIT_HOOKS_PER_POSITION[name] for name in positions)
    handles = plug_dropout(vit, "vit", positions, {}, rate=0.3)
    assert len(handles) == expected
    unplug_dropout(handles)


@pytest.mark.parametrize("config", sorted(DROPOUT_CONFIGS))
def test_swin_combo_hook_count_is_the_sum(swin, config):
    positions = DROPOUT_CONFIGS[config]
    expected = sum(SWIN_HOOKS_PER_POSITION[name] for name in positions)
    handles = plug_dropout(swin, "swin", positions, {}, rate=0.3)
    assert len(handles) == expected
    unplug_dropout(handles)


@pytest.mark.parametrize("position", SINGLE_POSITION_NAMES)
def test_vit_unplug_restores_state_dict_exactly(vit, position):
    before = _state_dict_bytes(vit)
    handles = plug_dropout(vit, "vit", (position,), {}, rate=0.5)
    unplug_dropout(handles)
    _assert_state_dict_identical(before, _state_dict_bytes(vit))


@pytest.mark.parametrize("position", SINGLE_POSITION_NAMES)
def test_swin_unplug_restores_state_dict_exactly(swin, position):
    before = _state_dict_bytes(swin)
    handles = plug_dropout(swin, "swin", (position,), {}, rate=0.5)
    unplug_dropout(handles)
    _assert_state_dict_identical(before, _state_dict_bytes(swin))


def test_swin_stochastic_depth_fires_twice_per_block(swin):
    """The finding behind hooking attn/mlp instead of stochastic_depth.

    self.stochastic_depth is one instance called twice per block, so a hook on it
    cannot tell the attention branch from the MLP branch. A counting hook proves
    the two calls per block: exactly 2 per SwinTransformerBlock.
    """
    first_block = next(
        module for module in swin.modules() if isinstance(module, SwinTransformerBlock)
    )
    calls = {"count": 0}

    def counter(module, args, output):
        calls["count"] += 1

    handle = first_block.stochastic_depth.register_forward_hook(counter)
    swin.eval()
    with torch.inference_mode():
        swin(torch.randn(1, 3, 32, 32))
    handle.remove()

    assert calls["count"] == 2, "stochastic_depth must fire twice per block forward"


def test_plugged_dropout_actually_perturbs_forward(vit):
    """A plugged position must change the output; unplug must restore it."""
    vit.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        clean = vit(x).clone()

    handles = plug_dropout(vit, "vit", ("before_mlp_residual",), {}, rate=0.9)
    torch.manual_seed(0)
    with torch.inference_mode():
        perturbed = vit(x).clone()
    unplug_dropout(handles)
    with torch.inference_mode():
        restored = vit(x).clone()

    assert not torch.allclose(clean, perturbed), "plugged dropout should perturb output"
    assert torch.allclose(clean, restored), "unplug should restore the exact output"


@pytest.mark.parametrize(
    "architecture,builder,size", [("vit", build_vit, 32), ("swin", build_swin, 32)]
)
def test_residual_wrapper_at_rate_zero_reproduces_the_original_forward(
    architecture, builder, size
):
    """The swapped forward must be algebraically identical to torchvision's.

    A rewritten forward is the one place in this mechanism where a transcription
    slip would silently change the model rather than raise. Dropout at rate 0 is
    the identity, so any difference here is a bug in the rewrite, not noise, and
    bit-equality is the right assertion.
    """
    model = builder(num_classes=10).eval()
    x = torch.randn(2, 3, size, size)
    with torch.inference_mode():
        original = model(x).clone()

    handles = plug_dropout(model, architecture, ("after_attention_residual",), {}, 0.0)
    with torch.inference_mode():
        wrapped = model(x).clone()
    unplug_dropout(handles)
    with torch.inference_mode():
        restored = model(x).clone()

    assert torch.equal(original, wrapped), "rate 0.0 wrapper is not the identity"
    assert torch.equal(original, restored), "unplug did not restore the forward"


def test_residual_position_reaches_the_skip_path_and_ln_2_prehook_does_not(vit):
    """The reason after_attention_residual cannot be a pre-hook on ln_2.

    In EncoderBlock.forward, x = x + input is consumed twice: by ln_2 and by the
    final x + y. A pre-hook on ln_2 perturbs only the first use. The wrapper
    perturbs the value itself, so both uses see it, which is what the PSBD paper's
    post-residual placement means. Same mask seed for both, so the measured gap is
    the placement and not the sampled mask.
    """
    vit.eval()
    x = torch.randn(4, 3, 32, 32)
    block = next(iter(vit[1].encoder.layers))
    captured = {}
    probe = block.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("out", out.detach().clone())
    )

    with torch.inference_mode():
        vit(x)
    unperturbed = captured["out"].clone()

    deltas = {}
    for position in ("before_mlp_norm", "after_attention_residual"):
        handles = plug_dropout(vit, "vit", (position,), {}, rate=0.9)
        torch.manual_seed(0)
        with torch.inference_mode():
            vit(x)
        unplug_dropout(handles)
        deltas[position] = (captured["out"] - unperturbed).abs().mean().item()
    probe.remove()

    assert deltas["after_attention_residual"] > deltas["before_mlp_norm"], (
        "the residual wrapper must perturb the block output more than an ln_2 "
        f"pre-hook does, got {deltas}"
    )
