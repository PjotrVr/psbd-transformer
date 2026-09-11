"""Mechanical checks on the resnet18 control architecture.

Mirrors tests/test_perturbations.py's attach/detach style, but for the paper's own
ConvNet placement (models.positions.RESNET_POSITIONS's "post_residual") rather than
a ViT position.
"""

import torch

import models.backbones
from models.positions import plug_dropout, unplug_dropout

# ResNet-18 is 4 stages of 2 BasicBlocks each (layer1..layer4), so every
# block-scope position attaches once per block.
RESNET18_BASIC_BLOCK_COUNT = 8


def test_build_resnet18_shapes_and_class_count():
    model = models.backbones.build_resnet18(10)
    model.eval()
    probe = torch.randn(2, 3, 32, 32)  # (batch, channels, height, width)

    with torch.no_grad():
        logits = model(probe)  # (batch, num_classes)

    assert logits.shape == (2, 10)


def test_load_checkpoint_round_trip(tmp_path):
    """build_model, saved, then read back by load_checkpoint, reproduces the same weights.

    Exercises the same path cli.sweep and cli.evaluate use to read a resnet18
    checkpoint, including detect_architecture reading its state_dict keys.
    """
    model = models.backbones.build_resnet18(10)
    checkpoint_path = tmp_path / "attack_result.pt"
    torch.save({"model": model.state_dict(), "num_classes": 10}, checkpoint_path)

    detected = models.backbones.detect_architecture(str(checkpoint_path))
    assert detected == "resnet18"

    loaded = models.backbones.load_checkpoint(
        "resnet18", str(checkpoint_path), torch.device("cpu")
    )
    probe = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        original_logits = model.eval()(probe)
        loaded_logits = loaded(probe)
    assert torch.equal(original_logits, loaded_logits)


def test_network_core_is_the_bare_network():
    """resnet18 carries no Resize wrapper, unlike build_vit and build_swin."""
    model = models.backbones.build_resnet18(10)
    assert models.backbones.network_core(model) is model


def test_plug_dropout_attaches_1_module_per_basic_block():
    model = models.backbones.build_resnet18(10)
    model.eval()

    handles = plug_dropout(model, "resnet18", ("post_residual",), {}, 0.5)
    assert len(handles) == RESNET18_BASIC_BLOCK_COUNT

    unplug_dropout(handles)


def test_unplug_dropout_restores_the_original_forward():
    torch.manual_seed(0)
    model = models.backbones.build_resnet18(10)
    model.eval()
    probe = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        before = model(probe).clone()

    handles = plug_dropout(model, "resnet18", ("post_residual",), {}, 0.5)
    unplug_dropout(handles)

    with torch.no_grad():
        after = model(probe)
    assert torch.equal(before, after)


def test_perturbed_output_differs_only_when_the_probe_is_active():
    """A plugged rate-0.5 probe changes the output, a plugged rate-0.0 probe does not.

    nn.Dropout at rate 0.0 is deterministically the identity, so it stands in
    for "the probe is inert" without needing eval()/train() on the probe itself:
    plug_dropout already sets every attached probe to train() (models.positions'
    module docstring), which is what makes a nonzero rate sample a mask at all.
    The model itself stays in eval() throughout, as it would at PSBD inference
    time.
    """
    torch.manual_seed(0)
    model = models.backbones.build_resnet18(10)
    model.eval()
    probe = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        before = model(probe).clone()

    handles = plug_dropout(model, "resnet18", ("post_residual",), {}, 0.5)
    with torch.no_grad():
        perturbed = model(probe)
    unplug_dropout(handles)
    assert not torch.allclose(before, perturbed)

    handles_rate_zero = plug_dropout(model, "resnet18", ("post_residual",), {}, 0.0)
    with torch.no_grad():
        zero_rate = model(probe)
    unplug_dropout(handles_rate_zero)
    assert torch.equal(before, zero_rate)


def test_position_registry_aliases_resolve_to_the_same_spec():
    """The DROPOUT_CONFIGS-mediated path (cli.sweep, attacks.evasion) attaches
    the same site under either alias name, so `--position post_residual` and the
    evasion probe token `post_residual:dropout` reach the identical block hook
    that a direct ("post_residual",) call does.
    """
    from models.positions import RESNET_POSITIONS

    assert (
        RESNET_POSITIONS["post_residual"]
        is RESNET_POSITIONS["after_attention_residual"]
        is RESNET_POSITIONS["after_mlp_residual"]
    )


def test_dropout_configs_path_still_perturbs_every_block():
    """The indirect path cli.sweep actually uses: DROPOUT_CONFIGS.get("post_residual").

    Attaches the residual wrapper twice per block (once per alias name), the
    second overwriting the first, so the block still ends up with exactly 1
    active probe and the output is perturbed relative to the unplugged model.
    """
    from models.positions import DROPOUT_CONFIGS

    torch.manual_seed(0)
    model = models.backbones.build_resnet18(10)
    model.eval()
    probe = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        before = model(probe).clone()

    position_names = DROPOUT_CONFIGS.get("post_residual", ("post_residual",))
    assert position_names == ("after_attention_residual", "after_mlp_residual")

    handles = plug_dropout(model, "resnet18", position_names, {}, 0.9)
    assert len(handles) == 2 * RESNET18_BASIC_BLOCK_COUNT

    with torch.no_grad():
        perturbed = model(probe)
    assert not torch.allclose(before, perturbed)

    unplug_dropout(handles)
    with torch.no_grad():
        after = model(probe)
    assert torch.equal(before, after)
