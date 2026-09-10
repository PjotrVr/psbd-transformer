"""The 2 primitives the gradient-based detectors stand on, pinned on the synthetic model.

forward_logits, frozen_parameters and captured_layers exist so that CD-L, Beatrix,
TED and SentiNet share 1 forward helper and 1 hook implementation with the rest
of the repo. These tests pin what they promise: byte identity with the paths they
replace, gradients that flow where they should, flags and hooks that are restored
and removed. The last test records a torchvision fact a detector
depends on, that the patch tokens of the final block's output receive no
gradient at all.
"""

import pytest
import torch

from analysis.features import (
    as_token_sequence,
    captured_layers,
    extract_layer_features,
)
from defences.inference import forward_logits, forward_probs, frozen_parameters
from experiments.preflight.synthetic import build_backdoored_model, build_splits

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def case():
    backdoored = build_backdoored_model()
    inner = backdoored.inner  # nn.Sequential(Resize, VisionTransformer), 2 blocks
    loaders = build_splits(num_samples=64, batch_size=32)
    images, _ = next(iter(loaders["clean"]))  # (32, 3, 32, 32)
    return inner, loaders, images


def test_forward_probs_is_the_softmax_of_forward_logits(case):
    inner, _, images = case
    with torch.inference_mode():
        logits = forward_logits(inner, images, DEVICE, use_bfloat16=False)
        probs = forward_probs(inner, images, DEVICE, use_bfloat16=False)

    assert logits.shape == (32, 10) and logits.dtype == torch.float32
    assert torch.equal(probs, torch.softmax(logits, dim=1))


def test_forward_logits_differentiates_to_the_input(case):
    inner, _, images = case
    pixels = images.clone().requires_grad_(True)

    logits = forward_logits(inner, pixels, DEVICE, use_bfloat16=False)
    (gradient,) = torch.autograd.grad(logits.sum(), pixels)  # (32, 3, 32, 32)

    assert gradient.shape == pixels.shape
    assert gradient.abs().sum() > 0


def test_frozen_parameters_stops_at_the_activations_and_restores_flags(case):
    inner, _, images = case
    parameters = list(inner.parameters())
    parameters[0].requires_grad_(False)
    before = [parameter.requires_grad for parameter in parameters]

    pixels = images.clone().requires_grad_(True)
    with frozen_parameters(inner):
        assert not any(parameter.requires_grad for parameter in parameters)
        logits = forward_logits(inner, pixels, DEVICE, use_bfloat16=False)
        (gradient,) = torch.autograd.grad(logits.sum(), pixels)

    assert gradient.abs().sum() > 0, "the input gradient must survive the freeze"
    assert all(parameter.grad is None for parameter in parameters)
    assert [parameter.requires_grad for parameter in parameters] == before
    parameters[0].requires_grad_(True)


def test_captured_layers_agrees_with_extract_layer_features(case):
    inner, loaders, _ = case
    reduced = extract_layer_features(
        inner, loaders["clean"], DEVICE, use_bfloat16=False, reduction="cls"
    )

    rows = []
    with captured_layers(inner, (0, 1, 2)) as captured, torch.inference_mode():
        for images, _ in loaders["clean"]:
            forward_logits(inner, images, DEVICE, use_bfloat16=False)
            rows.append(
                {layer: captured[layer][:, 0, :].clone() for layer in (0, 1, 2)}
            )

    for layer in (0, 1, 2):
        ours = torch.cat([row[layer] for row in rows])  # (64, dim)
        assert torch.equal(ours, reduced[layer]), f"layer {layer} differs"


def test_captured_layers_refuses_a_layer_the_model_does_not_have(case):
    inner, _, _ = case
    with pytest.raises(ValueError, match="do not exist"):
        with captured_layers(inner, (0, 3)):
            pass


def test_captured_layers_removes_its_hooks_on_exit(case):
    inner, _, images = case
    with captured_layers(inner, (1,)) as captured, torch.inference_mode():
        forward_logits(inner, images, DEVICE, use_bfloat16=False)
        first = captured[1]
    with torch.inference_mode():
        forward_logits(inner, images[:8], DEVICE, use_bfloat16=False)

    assert captured[1] is first, "a hook left behind would have overwritten the entry"


def test_the_last_block_output_patch_tokens_receive_no_gradient_on_vit(case):
    """torchvision's ViT reads x[:, 0] after the encoder, so a Grad-CAM taken at the
    last block's OUTPUT is blank on every patch token. The input of the last block
    is the deepest site whose patch tokens still carry gradient, which is where
    SentiNet's CAM must hook."""
    inner, _, images = case
    # With every parameter frozen, the input is the only leaf that can put the
    # activations into a graph, which is also how the detector calls it.
    pixels = images.clone().requires_grad_(True)
    with captured_layers(inner, (1, 2)) as captured, frozen_parameters(inner):
        logits = forward_logits(inner, pixels, DEVICE, use_bfloat16=False)
        target = logits.gather(1, logits.argmax(dim=1, keepdim=True)).sum()
        last_input, last_output = captured[1], captured[2]  # (32, 17, 32) each
        gradients = torch.autograd.grad(target, [last_input, last_output])

    patch_gradient_in = gradients[0][:, 1:, :]  # (32, 16, 32)
    patch_gradient_out = gradients[1][:, 1:, :]  # (32, 16, 32)
    assert patch_gradient_out.abs().max() == 0
    assert patch_gradient_in.abs().max() > 0


def test_as_token_sequence_is_a_view_that_passes_gradient():
    grid = torch.randn(
        2, 3, 3, 5, requires_grad=True
    )  # (batch, height, width, channels)
    tokens = as_token_sequence(grid)  # (batch, 9, channels)
    assert tokens.shape == (2, 9, 5)
    tokens.sum().backward()
    assert torch.equal(grid.grad, torch.ones_like(grid))
