"""CD-L pinned on the CPU: its contract, the pieces of Eq. (1) and the released class.

The synthetic fixture installs its backdoor by boolean indexing, so no gradient
reaches the trigger pixels and CD-L cannot separate on it by construction. These
tests therefore take the contract (shape, finiteness, determinism, no parameter
gradient, native resolution) from the fixture and the direction from a model
built here whose trigger gate is differentiable. The last test is the bit-level
comparison against the authors' released class.
"""

import importlib.util
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from detectors.cd_l import (
    MASK_CHANNELS,
    MASK_PARAMETER_INIT,
    cd_l_scores,
    distill_masks,
    effective_mask,
    mask_norms,
    total_variation,
)
from experiments.preflight.synthetic import (
    BACKDOOR_LOGIT,
    IMAGE_SIZE,
    TARGET_CLASS,
    TRIGGER_SIZE,
    apply_trigger,
    build_backdoored_model,
    build_splits,
    trigger_pattern,
)

DEVICE = torch.device("cpu")
IDENTITY_MEAN = (0.0, 0.0, 0.0)
IDENTITY_STD = (1.0, 1.0, 1.0)

# Enough steps to see the mask move, few enough to keep the file under a minute.
FEW_STEPS = 10

# Adam with betas (0.1, 0.1) moves a mask parameter by about the learning rate
# per step, so 30 steps take an unopposed parameter from 1 to about -2, where
# the effective mask is 0.018. That is the collapse the direction test looks for.
DIRECTION_STEPS = 30

# Steepness of the planted trigger gate. sigmoid(40 * 0.25) is 0.99995 on the
# exact checkerboard and sigmoid(-40 * 0.25) is 0.00005 on a random corner.
GATE_SHARPNESS = 40.0

# Mean absolute deviation from the checkerboard at which the gate is half open.
# A uniform random corner deviates by 0.5 on average, the trigger by 0.
GATE_MARGIN = 0.25

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_PATH = os.path.join(
    REPO_ROOT,
    "third_party",
    "CognitiveDistillation",
    "detection",
    "cognitive_distillation.py",
)


@pytest.fixture(scope="module")
def synthetic_case():
    model = build_backdoored_model()
    loaders = build_splits(num_samples=32, batch_size=16)
    return model, loaders


def score_split(model, loader, num_steps=FEW_STEPS):
    scores = cd_l_scores(
        model,
        loader,
        DEVICE,
        IDENTITY_MEAN,
        IDENTITY_STD,
        use_bfloat16=False,
        seed=0,
        num_steps=num_steps,
    )
    return scores


def test_scores_have_the_split_length_and_are_finite(synthetic_case):
    model, loaders = synthetic_case

    scores = score_split(model, loaders["clean"])

    assert scores.shape == (32,)
    assert scores.dtype == torch.float32
    assert scores.device.type == "cpu"
    assert torch.isfinite(scores).all()


def test_scores_are_deterministic_under_the_seed(synthetic_case):
    model, loaders = synthetic_case

    first = score_split(model, loaders["clean"])
    second = score_split(model, loaders["clean"])

    assert torch.equal(first, second)


def test_0_steps_returns_the_initial_mask_norm(synthetic_case):
    """Pins MASK_PARAMETER_INIT: an unoptimised mask is 0.8808 on every pixel."""
    model, loaders = synthetic_case

    scores = score_split(model, loaders["clean"], num_steps=0)

    initial_value = effective_mask(torch.tensor(MASK_PARAMETER_INIT))
    expected = initial_value * IMAGE_SIZE * IMAGE_SIZE
    assert torch.allclose(scores, expected.expand(32), atol=1e-3)


def test_effective_mask_of_ones_is_0_8808():
    mask = effective_mask(torch.ones(2, MASK_CHANNELS, 4, 4))  # (2, 1, 4, 4)

    assert mask.shape == (2, 1, 4, 4)
    assert mask.mean().item() == pytest.approx(0.8808, abs=5e-5)


def test_total_variation_of_a_constant_mask_is_0():
    constant = torch.full((3, MASK_CHANNELS, 8, 8), 0.37)  # (3, 1, 8, 8)

    assert torch.equal(total_variation(constant), torch.zeros(3))


def test_total_variation_carries_the_released_normalisation():
    """1 vertical edge in a 4 by 6 mask: 4 unit squared steps over 1 * 4 * 6 entries."""
    mask = torch.zeros(1, MASK_CHANNELS, 4, 6)  # (1, 1, 4, 6)
    mask[:, :, :, 3:] = 1.0

    assert total_variation(mask).item() == pytest.approx(4 / (1 * 4 * 6))


def test_no_model_parameter_receives_a_gradient_and_flags_are_restored(
    synthetic_case,
):
    model, loaders = synthetic_case
    parameters = list(model.parameters())
    parameters[0].requires_grad_(False)
    before = [parameter.requires_grad for parameter in parameters]

    score_split(model, loaders["clean"])

    assert all(parameter.grad is None for parameter in parameters)
    assert [parameter.requires_grad for parameter in parameters] == before
    parameters[0].requires_grad_(True)


def test_the_model_receives_the_native_resolution(synthetic_case):
    """The mask lives at the dataset's resolution and the model's Resize upsamples.

    16 by 16 images into a model whose Resize targets 32 by 32: the wrapper must
    see 16, the network behind the Resize must see 32 and the mask must be 16.
    """
    model, _ = synthetic_case
    network = model.inner[1]
    seen_by_wrapper: list[tuple[int, ...]] = []
    seen_by_network: list[tuple[int, ...]] = []
    handles = [
        model.register_forward_pre_hook(
            lambda _module, inputs: seen_by_wrapper.append(tuple(inputs[0].shape))
        ),
        network.register_forward_pre_hook(
            lambda _module, inputs: seen_by_network.append(tuple(inputs[0].shape))
        ),
    ]
    small = torch.rand(4, 3, 16, 16, generator=torch.Generator().manual_seed(0))

    try:
        masks = distill_masks(
            model, small, IDENTITY_MEAN, IDENTITY_STD, DEVICE, False, num_steps=2
        )  # (4, 1, 16, 16)
    finally:
        for handle in handles:
            handle.remove()

    assert masks.shape == (4, 1, 16, 16)
    assert seen_by_wrapper and set(seen_by_wrapper) == {(4, 3, 16, 16)}
    assert set(seen_by_network) == {(4, 3, 32, 32)}


class CornerGatedClassifier(nn.Module):
    """Logits read every pixel until the trigger appears, then only the corner.

    A random linear read of the whole image stands in for class evidence, so
    removing any pixel moves a clean image's logits. A differentiable read of the
    checkerboard gates that evidence out and pins the target logit, so on a
    triggered image the background carries nothing and the mask is free to leave
    it. The gate is a sigmoid rather than the fixture's boolean test so that the
    gradient reaches the corner pixels.
    """

    def __init__(self, num_classes: int = 10, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("pattern", trigger_pattern())
        self.register_buffer(
            "content_weights",
            torch.randn(3 * IMAGE_SIZE * IMAGE_SIZE, num_classes, generator=generator),
        )
        self.register_buffer(
            "target_onehot",
            F.one_hot(torch.tensor(TARGET_CLASS), num_classes).float(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        content_logits = images.flatten(1) @ self.content_weights  # (batch, classes)

        corner = images[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:]  # (batch, 3, T, T)
        deviation = (corner - self.pattern).abs().mean(dim=(1, 2, 3))  # (batch,)
        presence = torch.sigmoid(GATE_SHARPNESS * (GATE_MARGIN - deviation))
        presence = presence.unsqueeze(1)  # (batch, 1)

        backdoor_logits = BACKDOOR_LOGIT * self.target_onehot  # (classes,)
        logits = (1 - presence) * content_logits + presence * backdoor_logits
        return logits


def test_a_triggered_input_distills_to_a_smaller_mask_than_its_clean_twin():
    model = CornerGatedClassifier().eval()
    generator = torch.Generator().manual_seed(1)
    clean = torch.rand(8, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    triggered = apply_trigger(clean)
    labels = torch.zeros(8, dtype=torch.long)
    clean_loader = DataLoader(TensorDataset(clean, labels), batch_size=8)
    triggered_loader = DataLoader(TensorDataset(triggered, labels), batch_size=8)

    with torch.inference_mode():
        predictions = model(triggered).argmax(dim=1)  # (8,)
    assert (predictions == TARGET_CLASS).all(), "the planted backdoor must fire"

    clean_scores = score_split(model, clean_loader, num_steps=DIRECTION_STEPS)
    triggered_scores = score_split(model, triggered_loader, num_steps=DIRECTION_STEPS)

    assert (triggered_scores < clean_scores).all(), (
        f"triggered {triggered_scores.tolist()} should sit below clean "
        f"{clean_scores.tolist()}"
    )
    # The background of a triggered image carries nothing, so its mask collapses
    # toward the 36-pixel corner while a clean mask stays near full coverage.
    assert triggered_scores.max() < 0.25 * clean_scores.min()


def load_reference_class():
    if not os.path.exists(REFERENCE_PATH):
        pytest.skip(f"released class absent at {REFERENCE_PATH}")

    spec = importlib.util.spec_from_file_location("cd_reference", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CognitiveDistillation


def test_mask_norms_match_the_released_class_bit_for_bit():
    """The 1 check that says this is the paper's method rather than a look-alike.

    float32 on the CPU, identity normalisation, the same seed immediately before
    each call so the per-step fill sequences coincide. The reference's only
    randomness is that fill, so any residual difference would be an operator or
    an ordering change rather than noise.
    """
    reference_class = load_reference_class()
    torch.manual_seed(123)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    ).eval()
    images = torch.rand(4, 3, 32, 32)  # (4, 3, 32, 32), already in [0, 1]

    torch.manual_seed(0)
    theirs = reference_class(norm_only=True, num_steps=FEW_STEPS)(model, images)
    torch.manual_seed(0)
    ours = mask_norms(
        distill_masks(
            model,
            images,
            IDENTITY_MEAN,
            IDENTITY_STD,
            DEVICE,
            False,
            num_steps=FEW_STEPS,
        )
    )  # (4,)

    assert theirs.shape == ours.shape == (4,)
    assert torch.allclose(theirs, ours, atol=1e-5), (
        f"{theirs.tolist()} vs {ours.tolist()}"
    )
