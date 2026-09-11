"""Feature visualisation on the synthetic ViT: the objective rises and the image is valid.

The synthetic 2-block ViT is untrained, so nothing is claimed about what the
images show. What is checked is the mechanics: the Fourier parameterisation
inverts to an image of the right shape, the valid-image map lands in [0, 1],
the optimiser raises the unit it was pointed at, the same seed gives the same
image and the TAC ranking puts the dimension the trigger moves most first.
"""

import pytest
import torch

from analysis.synthesis import (
    fourier_scale,
    maximising_input,
    pooled_units,
    rank_dimensions_by_tac,
    rfft2d_frequencies,
    spectrum_to_image,
    to_valid_image,
    total_variation,
    unit_activation,
)
from analysis.features import captured_layers
from defences.inference import forward_logits
from experiments.preflight.synthetic import build_backdoored_model, build_splits

DEVICE = torch.device("cpu")
IMAGE_SIZE = 32
LAYER = 2
DIMENSIONS = torch.tensor([0, 5, 17, 31])


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


def test_fourier_frequencies_and_scale_have_the_half_spectrum_shape():
    frequencies = rfft2d_frequencies(IMAGE_SIZE, IMAGE_SIZE)
    scale = fourier_scale(IMAGE_SIZE, IMAGE_SIZE)

    assert frequencies.shape == (IMAGE_SIZE, IMAGE_SIZE // 2 + 1)
    assert frequencies[0, 0] == 0.0
    assert scale.max().item() == pytest.approx(IMAGE_SIZE), (
        "the zero frequency is clamped to 1 / max(h, w), so its scale is max(h, w)"
    )
    assert (scale[1:, 1:] < scale[0, 0]).all()


def test_spectrum_inverts_to_an_image_of_the_image_size():
    spectrum = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE // 2 + 1, 2) * 0.01
    scale = fourier_scale(IMAGE_SIZE, IMAGE_SIZE)

    image = spectrum_to_image(spectrum, scale, IMAGE_SIZE, IMAGE_SIZE)
    valid = to_valid_image(spectrum, scale, IMAGE_SIZE, IMAGE_SIZE)

    assert image.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert valid.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert valid.min().item() == pytest.approx(0.05, abs=1e-6)
    assert valid.max().item() == pytest.approx(0.95, abs=1e-6)


def test_total_variation_is_0_on_a_flat_image_and_positive_on_noise():
    flat = torch.full((2, 3, 8, 8), 0.3)
    noise = torch.rand(2, 3, 8, 8)

    assert torch.equal(total_variation(flat), torch.zeros(2))
    assert (total_variation(noise) > 0).all()


def test_unit_activation_reads_the_class_token_on_vit(model):
    images = torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    with captured_layers(model, (LAYER,)) as captured, torch.no_grad():
        forward_logits(model, images, DEVICE, use_bfloat16=False)
        activation = captured[LAYER]  # (4, 17, 32)
        values = unit_activation(activation, DIMENSIONS, "vit")
        pooled = pooled_units(activation, "vit")

    assert values.shape == (4,)
    assert torch.equal(pooled, activation[:, 0, :].float())
    for row, dimension in enumerate(DIMENSIONS.tolist()):
        assert values[row] == activation[row, 0, dimension]


def test_maximising_input_raises_the_objective_and_returns_a_valid_image(model):
    images, objectives = maximising_input(
        model,
        LAYER,
        DIMENSIONS,
        IMAGE_SIZE,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        device=DEVICE,
        use_bfloat16=False,
        steps=60,
        seed=0,
    )

    assert images.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert images.min().item() >= 0.0 and images.max().item() <= 1.0
    assert objectives.shape == (60, 4)
    early = objectives[:10].mean(dim=0)  # (4,)
    late = objectives[-10:].mean(dim=0)  # (4,)
    assert (late > early).all(), (
        f"the unit must rise for every image, {early} to {late}"
    )

    # The untransformed final image scores above the untransformed start image,
    # so the rise is in the image and not only in a lucky transform.
    start, _ = maximising_input(
        model,
        LAYER,
        DIMENSIONS,
        IMAGE_SIZE,
        (0.0,) * 3,
        (1.0,) * 3,
        DEVICE,
        False,
        steps=1,
        seed=0,
    )
    with captured_layers(model, (LAYER,)) as captured, torch.no_grad():
        forward_logits(model, start, DEVICE, use_bfloat16=False)
        before = unit_activation(captured[LAYER], DIMENSIONS, "vit")
        forward_logits(model, images, DEVICE, use_bfloat16=False)
        after = unit_activation(captured[LAYER], DIMENSIONS, "vit")
    assert (after > before).all()


def test_maximising_input_is_deterministic_under_a_seed_and_frees_the_model(model):
    flags = [parameter.requires_grad for parameter in model.parameters()]
    first, _ = maximising_input(
        model,
        LAYER,
        DIMENSIONS[:2],
        IMAGE_SIZE,
        (0.0,) * 3,
        (1.0,) * 3,
        DEVICE,
        False,
        steps=5,
        seed=4,
    )
    second, _ = maximising_input(
        model,
        LAYER,
        DIMENSIONS[:2],
        IMAGE_SIZE,
        (0.0,) * 3,
        (1.0,) * 3,
        DEVICE,
        False,
        steps=5,
        seed=4,
    )

    assert torch.equal(first, second)
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert all(parameter.grad is None for parameter in model.parameters())


def test_rank_dimensions_by_tac_orders_every_dimension_by_the_paired_change(model):
    loaders = build_splits(num_samples=32, batch_size=16)

    order, tac = rank_dimensions_by_tac(
        model,
        loaders["clean"],
        loaders["backdoor"],
        LAYER,
        DEVICE,
        False,
        max_batches=2,
    )

    assert order.shape == (32,) and tac.shape == (32,)
    assert sorted(order.tolist()) == list(range(32))
    assert torch.equal(tac[order], tac.sort(descending=True).values)
    assert tac[order[0]] > 0
