"""The 3 input attributions: frequency saliency line for line, expected gradients by Monte Carlo, Grad-CAM by contract.

The frequency map is checked against upstream's saliency on a tiny ConvNet,
on a copy of the model because the transcription switches requires_grad off
and never back. Expected gradients are checked exactly on a linear model,
where the estimator's expectation is w times (x minus the background mean)
whatever alpha does, and against an independent brute-force loop on a 2-layer
network. Grad-CAM is delegated to the SentiNet port, so only the upsampling
contract is pinned here.
"""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from torchvision.models import VisionTransformer

from analysis.attribution import (
    class_activation_map,
    expected_gradients,
    frequency_saliency,
    ranked_classes,
)
from tests.reference import backdoorbench as upstream

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_convnet() -> nn.Module:
    torch.manual_seed(0)
    network = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 4, 3, padding=1),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 5),
    )
    return network.eval()


def test_frequency_saliency_is_upstream_saliency_line_for_line(tiny_convnet):
    image = torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(1))
    ours = frequency_saliency(tiny_convnet, image, DEVICE, use_bfloat16=False)
    theirs = upstream.saliency(image.clone(), copy.deepcopy(tiny_convnet))
    assert ours.dtype == np.uint8 and ours.shape == (8, 8)
    assert np.array_equal(ours, theirs)


def test_frequency_saliency_leaves_the_model_and_the_image_untouched(tiny_convnet):
    image = torch.rand(3, 8, 8)
    frequency_saliency(tiny_convnet, image, DEVICE, use_bfloat16=False)
    assert all(parameter.requires_grad for parameter in tiny_convnet.parameters())
    assert image.shape == (3, 8, 8) and not image.requires_grad


def test_expected_gradients_of_a_linear_model_are_exact():
    torch.manual_seed(2)
    linear = nn.Linear(3 * 4 * 4, 3).eval()
    model = nn.Sequential(nn.Flatten(), linear)
    images = torch.rand(2, 3, 4, 4)
    background = torch.rand(50, 3, 4, 4)

    attributions, classes = expected_gradients(
        model,
        images,
        background,
        DEVICE,
        False,
        num_samples=64,
        ranked_outputs=2,
        seed=0,
        batch_size=16,
    )
    weight = linear.weight.detach().view(3, 3, 4, 4)  # (classes, C, H, W)
    for image_index in range(2):
        for rank in range(2):
            target = classes[image_index, rank]
            # The gradient of a linear logit is its weight row at every path
            # point, so only the mean background sampled matters. With 64 draws
            # the sample mean of the background sits close to its expectation.
            expected = weight[target] * (images[image_index] - background.mean(dim=0))
            assert torch.allclose(attributions[image_index, rank], expected, atol=0.05)
    assert torch.equal(classes, ranked_classes(model, images, DEVICE, False, 2))


def brute_force_expected_gradient(model, image, background, target, num_samples, seed):
    """1 draw at a time, the estimator written as the formula reads."""
    generator = torch.Generator().manual_seed(seed)
    total = torch.zeros_like(image)
    for _ in range(num_samples):
        baseline = background[
            int(torch.randint(background.shape[0], (1,), generator=generator))
        ]
        alpha = float(torch.rand(1, generator=generator))
        point = (baseline + alpha * (image - baseline)).requires_grad_(True)
        logit = model(point[None])[0, target]
        (gradient,) = torch.autograd.grad(logit, point)
        total += gradient * (image - baseline)
    return total / num_samples


def test_expected_gradients_agree_with_a_brute_force_monte_carlo():
    torch.manual_seed(3)
    model = nn.Sequential(
        nn.Flatten(), nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3)
    ).eval()
    image = torch.rand(1, 3, 2, 2)
    background = torch.rand(20, 3, 2, 2)

    attributions, classes = expected_gradients(
        model,
        image,
        background,
        DEVICE,
        False,
        num_samples=10000,
        ranked_outputs=1,
        seed=0,
        batch_size=500,
    )
    reference = brute_force_expected_gradient(
        model, image[0], background, int(classes[0, 0]), 10000, seed=99
    )
    assert torch.allclose(attributions[0, 0], reference, atol=0.01)


def test_expected_gradients_refuse_a_mismatched_background():
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
    with pytest.raises(ValueError, match="background"):
        expected_gradients(
            model,
            torch.rand(1, 3, 2, 2),
            torch.rand(4, 3, 3, 3),
            DEVICE,
            False,
            num_samples=2,
        )


def test_class_activation_map_is_upsampled_to_the_image_and_scaled():
    torch.manual_seed(0)
    network = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=4,
    )
    nn.init.normal_(network.heads.head.weight, std=0.5)
    model = nn.Sequential(transforms_v2.Resize((32, 32)), network).eval()
    images = torch.rand(3, 3, 32, 32)

    cam, predicted = class_activation_map(model, images, DEVICE, use_bfloat16=False)
    with torch.inference_mode():
        expected = model(images).argmax(dim=1)
    assert cam.shape == (3, 32, 32)
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0
    assert torch.equal(predicted, expected)
    assert all(parameter.requires_grad for parameter in model.parameters())
