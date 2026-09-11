"""Tests for the --augment option: data.loading's augmentation pieces, and their
wiring into cli.train_backdoor and cli.train_benign.

The property that matters is shape preservation. build_augmentation_transform
and AugmentedTrainingSet must hand back a tensor of the same (C, H, W) the
un-augmented pipeline would, whatever the input resolution. The standard
training loader must carry the same length and label set as the "none" loader.
Evaluation never sees --augment at all: evaluation.loaders.build_clean_loader
takes no augment parameter, so a caller cannot wire it in by accident.
"""

import argparse
import inspect

import torch
from torch.utils.data import Dataset

from data.loading import (
    AUGMENT_CHOICES,
    AugmentedTrainingSet,
    build_augmentation_transform,
)


class _FixedImageDataset(Dataset):
    """A synthetic (image, label) dataset, images in 0 to 1, no disk or network."""

    def __init__(self, num_samples: int, image_size: int, num_classes: int):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        generator = torch.Generator().manual_seed(0)
        self.images = torch.rand(
            (num_samples, 3, image_size, image_size), generator=generator
        )
        self.labels = [i % num_classes for i in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.labels[index]


def test_augment_choices_default_is_none():
    assert AUGMENT_CHOICES == ("none", "standard")


def test_build_augmentation_transform_preserves_shape():
    transform = build_augmentation_transform(image_size=32)
    image = torch.rand(3, 32, 32)

    augmented = transform(image)

    assert augmented.shape == (3, 32, 32)


def test_build_augmentation_transform_respects_a_different_resolution():
    transform = build_augmentation_transform(image_size=64)
    image = torch.rand(3, 64, 64)

    augmented = transform(image)

    assert augmented.shape == (3, 64, 64)


def test_augmented_training_set_preserves_shape_and_label():
    base = _FixedImageDataset(num_samples=8, image_size=32, num_classes=10)
    transform = build_augmentation_transform(image_size=32)
    augmented = AugmentedTrainingSet(base, transform)

    assert len(augmented) == len(base)
    for index in range(len(base)):
        image, label = augmented[index]
        assert image.shape == (3, 32, 32)
        assert label == base.labels[index]


def test_augmented_training_set_actually_perturbs_pixels():
    # A crop scale of (0.6, 1.0) plus a random flip should not be the identity
    # on every draw, or the augmentation is silently a no-op.
    base = _FixedImageDataset(num_samples=1, image_size=32, num_classes=10)
    transform = build_augmentation_transform(image_size=32)
    augmented = AugmentedTrainingSet(base, transform)

    original_image, _ = base[0]
    draws = [augmented[0][0] for _ in range(8)]
    assert any(not torch.equal(draw, original_image) for draw in draws)


def _train_backdoor_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        dataset="cifar10",
        attack="badnet_a2o",
        poison_rate=0.1,
        target_label=0,
        raw_data_dir="unused",
        max_samples=None,
        seed=0,
        poisoned_dir="",
        cover_rate=None,
        attack_override=[],
        evade_psbd=False,
        batch_size=4,
        num_workers=0,
        augment="none",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_train_backdoor_loader_standard_augment_same_shape_as_none(monkeypatch):
    import cli.train_backdoor as train_backdoor

    image_size = 32
    fake_train = _FixedImageDataset(
        num_samples=16, image_size=image_size, num_classes=10
    )
    fake_test = _FixedImageDataset(num_samples=4, image_size=image_size, num_classes=10)
    monkeypatch.setattr(
        train_backdoor,
        "load_clean_datasets",
        lambda dataset_name, transform, raw_data_dir: (fake_train, fake_test),
    )

    none_loader, _, _, _, _ = train_backdoor.build_training_loader(
        _train_backdoor_args(augment="none"), image_size
    )
    standard_loader, _, _, _, _ = train_backdoor.build_training_loader(
        _train_backdoor_args(augment="standard"), image_size
    )

    none_images, _ = next(iter(none_loader))
    standard_images, _ = next(iter(standard_loader))

    assert standard_images.shape == none_images.shape
    assert len(standard_loader.dataset) == len(none_loader.dataset)
    # Poisoning and labeling are seeded identically, so the standard loader must
    # carry the same labels as the un-augmented loader over the full epoch, only
    # the pixels differ. DataLoader shuffles per batch, so the full set is
    # compared rather than the first batch alone.
    none_labels = sorted(
        label for _, batch_labels in none_loader for label in batch_labels.tolist()
    )
    standard_labels = sorted(
        label for _, batch_labels in standard_loader for label in batch_labels.tolist()
    )
    assert none_labels == standard_labels


def _train_benign_args(**overrides) -> dict:
    defaults = dict(
        raw_data_dir="unused",
        batch_size=4,
        num_workers=0,
        max_samples=None,
        seed=0,
        augment="none",
    )
    defaults.update(overrides)
    return defaults


def test_train_benign_loader_standard_augment_same_shape_as_none(monkeypatch):
    import cli.train_benign as train_benign

    image_size = 32
    fake_train = _FixedImageDataset(
        num_samples=16, image_size=image_size, num_classes=10
    )
    fake_test = _FixedImageDataset(num_samples=4, image_size=image_size, num_classes=10)
    monkeypatch.setattr(
        train_benign,
        "load_clean_datasets",
        lambda dataset_name, transform, raw_data_dir: (fake_train, fake_test),
    )

    none_loader, none_classes = train_benign.build_benign_train_loader(
        "cifar10", **_train_benign_args(augment="none")
    )
    standard_loader, standard_classes = train_benign.build_benign_train_loader(
        "cifar10", **_train_benign_args(augment="standard")
    )

    none_images, _ = next(iter(none_loader))
    standard_images, _ = next(iter(standard_loader))

    assert standard_images.shape == none_images.shape
    assert standard_classes == none_classes
    assert len(standard_loader.dataset) == len(none_loader.dataset)


def test_train_benign_folder_name_carries_aug_tag():
    from cli.train_benign import checkpoint_folder_name

    standard_args = argparse.Namespace(augment="standard", use_sam=False)
    none_args = argparse.Namespace(augment="none", use_sam=False)

    assert (
        checkpoint_folder_name("vit", "cifar100", standard_args)
        == "vit_cifar100_benign_aug"
    )
    assert checkpoint_folder_name("vit", "cifar100", none_args) == "vit_cifar100_benign"


def test_evaluation_loader_has_no_augment_parameter():
    # cli.evaluate, data.splits and every PSBD cache read from this loader, and
    # none of them may see an augmented image. The absence of the parameter
    # makes wiring augmentation into it a deliberate, visible change rather than
    # an accident.
    from evaluation.loaders import build_clean_loader

    assert "augment" not in inspect.signature(build_clean_loader).parameters
