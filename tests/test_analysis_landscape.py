"""The filter normalisation's invariant and the surface's restoration, on the synthetic ViT.

The synthetic model carries every parameter kind the rule distinguishes: a
rank-4 patch-embedding kernel, rank-2 Linear and fused qkv weights, the rank-3
class token and position embedding, and rank-1 biases and LayerNorm affines.
"""

import copy

import pytest
import torch
import torch.nn as nn

from analysis.landscape import (
    direction_cosine,
    landscape_parameters,
    loss_surface,
    random_direction,
)
from experiments.preflight.synthetic import build_backdoored_model, build_splits

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module", autouse=True)
def few_threads():
    """torch defaults to 64 threads on the 128-core login node, where the tiny CPU
    ops of a toy model run 1000 times slower than at 8 from OpenMP overhead."""
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 8))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def model():
    inner = build_backdoored_model().inner  # nn.Sequential(Resize, VisionTransformer)
    return inner.eval()


@pytest.fixture(scope="module")
def batches():
    loaders = build_splits(num_samples=32, batch_size=16)
    return [(images, labels) for images, labels in loaders["clean"]]


def test_every_row_of_the_direction_has_the_norm_of_the_parameter_row(model):
    direction = random_direction(model, seed=0)
    parameters = landscape_parameters(model)

    assert len(direction) == len(parameters)
    ranks_seen = set()
    for tensor, parameter in zip(direction, parameters):
        assert tensor.shape == parameter.shape
        ranks_seen.add(tensor.dim())
        if tensor.dim() <= 1:
            assert torch.equal(tensor, torch.zeros_like(tensor)), (
                "biases and LayerNorm affines take a 0 direction"
            )
            continue
        row_norms = tensor.flatten(1).norm(dim=1)  # (first_axis,)
        parameter_row_norms = parameter.detach().flatten(1).norm(dim=1)
        assert torch.allclose(row_norms, parameter_row_norms, rtol=1e-5, atol=1e-7)
    assert {1, 2, 3, 4} <= ranks_seen, "the synthetic model covers every rank"


def test_layer_normalisation_matches_the_whole_tensor_norm(model):
    direction = random_direction(model, seed=0, normalisation="layer")
    for tensor, parameter in zip(direction, landscape_parameters(model)):
        if tensor.dim() <= 1:
            assert torch.equal(tensor, torch.zeros_like(tensor))
        else:
            assert tensor.norm().item() == pytest.approx(
                parameter.detach().norm().item(), rel=1e-5
            )


def test_unknown_normalisation_is_refused(model):
    with pytest.raises(ValueError, match="unknown normalisation"):
        random_direction(model, seed=0, normalisation="weight")


def test_the_same_seed_gives_the_same_direction_and_2_seeds_are_near_orthogonal(model):
    first = random_direction(model, seed=0)
    again = random_direction(model, seed=0)
    second = random_direction(model, seed=1)

    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert abs(direction_cosine(first, second)) < 0.1
    assert direction_cosine(first, first) == pytest.approx(1.0, abs=1e-5)


def test_loss_surface_restores_the_weights_bit_exactly(model, batches):
    before = copy.deepcopy(model.state_dict())
    directions = (random_direction(model, seed=0), random_direction(model, seed=1))
    grid = torch.linspace(-1.0, 1.0, 3)

    losses, accuracies = loss_surface(
        model, batches, directions, grid, grid, nn.CrossEntropyLoss(), DEVICE, False
    )

    after = model.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert losses.shape == (3, 3) and accuracies.shape == (3, 3)
    assert torch.isfinite(losses).all()
    assert ((accuracies >= 0) & (accuracies <= 1)).all()


def test_the_surface_centre_is_the_unperturbed_loss(model, batches):
    directions = (random_direction(model, seed=0), random_direction(model, seed=1))
    grid = torch.tensor([-0.5, 0.0, 0.5])
    loss_fn = nn.CrossEntropyLoss()

    losses, _ = loss_surface(
        model, batches, directions, grid, grid, loss_fn, DEVICE, False
    )

    with torch.no_grad():
        per_batch = [
            loss_fn(model(images), labels).item() for images, labels in batches
        ]
    assert losses[1, 1].item() == pytest.approx(
        sum(per_batch) / len(per_batch), rel=1e-5
    )
    assert not torch.allclose(losses, losses[1, 1].expand_as(losses)), (
        "moving along a normalised direction must change the loss"
    )


def test_loss_surface_restores_the_weights_when_a_batch_raises(model, batches):
    before = copy.deepcopy(model.state_dict())
    directions = (random_direction(model, seed=0), random_direction(model, seed=1))
    grid = torch.linspace(-1.0, 1.0, 2)

    def failing_loss(_logits, _labels):
        raise RuntimeError("a failure in the middle of the grid")

    with pytest.raises(RuntimeError, match="middle of the grid"):
        loss_surface(
            model, batches, directions, grid, grid, failing_loss, DEVICE, False
        )

    after = model.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)
