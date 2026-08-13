"""Verification checks for the backdoor attack implementations.

Layered cheapest first, so a broken attack is caught in seconds before any GPU
time is spent:

1. Label and index policy assertions, pure logic, no model.
2. Per-attack trigger property assertions on controlled inputs.
3. An overfit sanity check, the end-to-end proof that the trigger is learnable,
   marked slow (needs a GPU and real training) and skipped by default.

The overfit check is the strongest single signal. Karpathy's training recipe and
the Full Stack Deep Learning troubleshooting lecture both make the same point: a
model that cannot memorize a tiny batch has a bug in the pipeline, not the model.
For a backdoor that means if a model trained to memorize a small poisoned subset
does not reach near 100 percent attack success, the trigger or the label logic is
wrong. The failure signatures are diagnostic too: loss going up points to a sign
error, oscillation points to shuffled labels or a broken augmentation, and a
plateau points to the trigger never reaching the model.
"""

import torch
import torchvision.transforms.v2 as transforms_v2
import pytest
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from attacks import build_attack, default_config
from utils.config import DATASET_REGISTRY
from utils.datasets import extract_labels, load_clean_datasets
from defences.detection import attack_success_rate, clean_accuracy
from models import build_vit
from poison import (
    Attack,
    AttackSuccessSet,
    PoisonedTrainingSet,
    choose_poison_indices,
    is_poisonable,
    poisoned_label,
)
from train import train_classifier

CONTROLLED_SIZE = 32


def _mid_gray_image(size: int) -> torch.Tensor:
    # A flat 0.5 image keeps additive triggers away from the 0 and 1 clamps, so
    # the difference reflects the trigger exactly rather than saturated pixels.
    return torch.full((3, size, size), 0.5)


def _gradient_image(size: int) -> torch.Tensor:
    ramp = torch.linspace(0.1, 0.9, size)
    return ramp.view(1, 1, size).expand(3, size, size).clone()


def test_label_policy() -> None:
    assert is_poisonable("all_to_one", 1, target_label=0) is True
    assert is_poisonable("all_to_one", 0, target_label=0) is False
    assert is_poisonable("all_to_all", 0, target_label=0) is True
    assert is_poisonable("clean_label", 0, target_label=0) is True
    assert is_poisonable("clean_label", 1, target_label=0) is False

    assert poisoned_label("all_to_one", 5, target_label=0, num_classes=10) == 0
    assert poisoned_label("all_to_all", 9, target_label=0, num_classes=10) == 0
    assert poisoned_label("all_to_all", 3, target_label=0, num_classes=10) == 4
    assert poisoned_label("clean_label", 7, target_label=0, num_classes=10) == 7

    labels = [0, 1, 2, 3, 0, 1, 2, 3]
    noop = lambda image, index: image

    dirty = Attack("dirty", noop, "all_to_one", target_label=0)
    dirty_indices = choose_poison_indices(labels, dirty, poison_rate=0.5, seed=0)
    assert len(dirty_indices) == 4, "count should be round(rate times dataset size)"
    assert all(labels[i] != 0 for i in dirty_indices), (
        "all_to_one must skip the target class"
    )
    assert dirty_indices == choose_poison_indices(labels, dirty, 0.5, seed=0), (
        "must be reproducible"
    )

    clean = Attack("clean", noop, "clean_label", target_label=0)
    clean_indices = choose_poison_indices(labels, clean, poison_rate=0.5, seed=0)
    assert len(clean_indices) == 2, "clean_label is capped by the target-class count"
    assert all(labels[i] == 0 for i in clean_indices), (
        "clean_label must poison only the target class"
    )


def _assert_deterministic_and_bounded(attack: Attack, image: torch.Tensor) -> None:
    first = attack.apply_trigger(image, 0)
    assert torch.equal(first, attack.apply_trigger(image, 0)), (
        f"{attack.name} is not deterministic"
    )
    assert torch.equal(first, attack.apply_trigger(image, 7)), (
        f"{attack.name} depends on the index"
    )
    assert first.min() >= 0.0 and first.max() <= 1.0, (
        f"{attack.name} left the 0 to 1 range"
    )


def test_badnet() -> None:
    attack = build_attack(
        "badnet", default_config("badnet"), CONTROLLED_SIZE, target_label=0
    )
    image = _gradient_image(CONTROLLED_SIZE)
    _assert_deterministic_and_bounded(attack, image)

    poisoned = attack.apply_trigger(image, 0)
    patch = default_config("badnet").patch_size
    outside_clean = image.clone()
    outside_poisoned = poisoned.clone()
    outside_clean[:, -patch:, -patch:] = 0.0
    outside_poisoned[:, -patch:, -patch:] = 0.0
    assert torch.equal(outside_clean, outside_poisoned), (
        "badnet changed pixels outside the patch"
    )
    assert not torch.equal(poisoned[:, -patch:, -patch:], image[:, -patch:, -patch:]), (
        "patch not applied"
    )


def test_blend() -> None:
    attack = build_attack(
        "blend", default_config("blend"), CONTROLLED_SIZE, target_label=0
    )
    image = _mid_gray_image(CONTROLLED_SIZE)
    _assert_deterministic_and_bounded(attack, image)
    assert not torch.equal(attack.apply_trigger(image, 0), image), (
        "blend produced no change"
    )


def test_sig() -> None:
    attack = build_attack("sig", default_config("sig"), CONTROLLED_SIZE, target_label=0)
    image = _mid_gray_image(CONTROLLED_SIZE)
    _assert_deterministic_and_bounded(attack, image)

    difference = attack.apply_trigger(image, 0) - image
    first_row = difference[:, 0, :]
    assert torch.allclose(
        difference, first_row.unsqueeze(1).expand_as(difference), atol=1e-6
    ), "sig signal should depend on the column only, so every row must match"
    assert first_row.std() > 0, "sig signal should vary across columns"


def test_wanet() -> None:
    attack = build_attack(
        "wanet", default_config("wanet"), CONTROLLED_SIZE, target_label=0
    )
    image = _gradient_image(CONTROLLED_SIZE)
    _assert_deterministic_and_bounded(attack, image)
    assert not torch.equal(attack.apply_trigger(image, 0), image), (
        "wanet produced no warp"
    )


def test_low_frequency() -> None:
    attack = build_attack("lf", default_config("lf"), CONTROLLED_SIZE, target_label=0)
    image = _mid_gray_image(CONTROLLED_SIZE)
    _assert_deterministic_and_bounded(attack, image)

    difference = attack.apply_trigger(image, 0) - image
    spectrum = torch.fft.fftshift(torch.fft.fft2(difference), dim=(-2, -1)).abs()
    center = CONTROLLED_SIZE // 2
    cutoff = default_config("lf").cutoff
    low_band = spectrum[
        :, center - cutoff : center + cutoff + 1, center - cutoff : center + cutoff + 1
    ]
    low_fraction = low_band.sum() / spectrum.sum().clamp_min(1e-8)
    assert low_fraction > 0.5, "lf trigger energy should sit in low spatial frequencies"


def dump_visuals(output_path: str) -> None:
    """Save a grid of clean, poisoned, and amplified difference for eyeballing.

    Not a test, a manual utility. Denormalization is not needed because triggers
    act in 0-to-1 pixel space. The difference is amplified 10 times and
    recentered so invisible triggers such as WaNet and SIG become visible.
    """
    image = _gradient_image(CONTROLLED_SIZE)
    rows = []
    for name in ("badnet", "blend", "sig", "wanet", "lf"):
        attack = build_attack(
            name, default_config(name), CONTROLLED_SIZE, target_label=0
        )
        poisoned = attack.apply_trigger(image, 0)
        amplified = ((poisoned - image) * 10.0 + 0.5).clamp(0.0, 1.0)
        rows.extend([image, poisoned, amplified])
    save_image(rows, output_path, nrow=3)
    print(
        f"visual grid written to {output_path}, columns are clean, poisoned, amplified difference"
    )


@pytest.mark.slow
def test_overfit_sanity_check() -> None:
    """Train ViT on a tiny poisoned subset and confirm the trigger is learnable.

    Intentionally unseeded. This is a smoke test of the pipeline, not a
    result-producing run, so run-to-run variance in the outcome is fine.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = "cifar10"
    spec = DATASET_REGISTRY[dataset_name]
    transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((CONTROLLED_SIZE, CONTROLLED_SIZE)),
            transforms_v2.ToTensor(),
        ]
    )
    train_clean, _ = load_clean_datasets(dataset_name, transform, "raw_data")
    subset = Subset(train_clean, list(range(256)))
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    attack = build_attack(
        "badnet", default_config("badnet"), CONTROLLED_SIZE, target_label=0
    )
    labels = extract_labels(subset)
    poison_indices = choose_poison_indices(labels, attack, poison_rate=0.5, seed=0)

    poisoned_train = PoisonedTrainingSet(
        subset, attack, poison_indices, normalize, spec.num_classes
    )
    asr_set = AttackSuccessSet(subset, labels, attack, normalize, spec.num_classes)

    train_loader = DataLoader(poisoned_train, batch_size=64, shuffle=True)
    asr_loader = DataLoader(asr_set, batch_size=64)

    # ASR of an untrained model is the input-independent baseline, near one over
    # the class count. If it is already high, the ASR measurement itself is wrong.
    baseline_model = build_vit(spec.num_classes).to(device)
    asr_before = attack_success_rate(
        baseline_model, asr_loader, device, use_bfloat16=True
    )
    assert asr_before < 0.5, (
        "untrained-model ASR should sit near the class-count baseline, not already high"
    )

    trained = train_classifier(
        "vit",
        spec.num_classes,
        train_loader,
        train_loader,
        device,
        epochs=15,
        use_sam=False,
    )
    asr_after = attack_success_rate(trained, asr_loader, device, use_bfloat16=True)
    memorization = clean_accuracy(trained, train_loader, device, use_bfloat16=True)

    assert asr_after > 0.9, (
        f"ASR after overfitting should exceed 0.9, got {asr_after:.3f}"
    )
    assert memorization > 0.9, (
        f"train memorization should exceed 0.9, got {memorization:.3f}"
    )


if __name__ == "__main__":
    # Flat imports need the repo root on sys.path, which bare `python
    # tests/test_attack_triggers.py` from the repo root does not provide:
    # `PYTHONPATH=. python tests/test_attack_triggers.py` from the repo root.
    dump_visuals("attack_visuals.png")
