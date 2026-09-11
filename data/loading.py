"""Loading the clean datasets, and reading their labels cheaply.

Disk reads and downloads live here so the detection and analysis code stays pure.

A dataset from load_clean_datasets yields images in 0 to 1, unnormalized. Every
attack stamps its trigger in pixel space, so normalization has to come after the
trigger, and the dataset wrappers in attacks.poisoning apply it last. Baking a
Normalize into the transform handed in here would stamp triggers onto normalized
tensors that no longer match the attack's paper. Use base_image_transform and
construct the Normalize at the call site, where the order is visible.
"""

import os

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import Dataset, Subset
from torchvision import datasets as tv_datasets

from .registry import DATASET_REGISTRY, DatasetSpec


def base_image_transform(image_size: int) -> transforms_v2.Compose:
    """The transform that resizes to image_size and yields a 0-to-1 (C, H, W) tensor.

    image_size is the dataset's native trigger resolution, not 224. The model wrapper
    does its own upscale, so resizing to 224 here would put the trigger on the wrong
    pixels.
    """
    transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((image_size, image_size)),
            transforms_v2.ToTensor(),
        ]
    )
    return transform


# Every fine-tuning run in the project trains with no augmentation beyond
# normalization, following the PSBD paper's recipe. "standard" exists to answer a
# reviewer's question, whether the detector still holds on a model trained the
# usual way, and stays off by default so no existing run's numbers move.
AUGMENT_CHOICES = ("none", "standard")
AUGMENT_CROP_SCALE = (0.6, 1.0)


def build_augmentation_transform(image_size: int) -> transforms_v2.Compose:
    """The "standard" augmentation: a random resized crop back to image_size, plus a flip.

    Applies only on the training loader, never on evaluation or the PSBD splits, and
    only after the attack's trigger has already been stamped, so a poisoned image is
    augmented the way a real attacker's poisoned image would be. The crop and flip run
    on the already-normalized training tensor at the call site (see
    AugmentedTrainingSet), which is equivalent to running them on the poisoned image
    before normalizing: a crop only selects pixels, an interpolated resize is a
    per-pixel weighted sum whose weights sum to 1 and a flip only reorders pixels, so
    all 3 commute exactly with the Normalize map's per-channel (x - mean) / std.
    """
    transform = transforms_v2.Compose(
        [
            transforms_v2.RandomResizedCrop(image_size, scale=AUGMENT_CROP_SCALE),
            transforms_v2.RandomHorizontalFlip(),
        ]
    )
    return transform


class AugmentedTrainingSet(Dataset):
    """A training dataset with build_augmentation_transform applied to every image.

    base_dataset already stamps the trigger and normalizes, so this only has to
    transform the tensor it already yields. See build_augmentation_transform for why
    running the crop and flip after normalization is equivalent to running them
    before it.
    """

    def __init__(self, base_dataset: Dataset, transform: transforms_v2.Compose):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.base_dataset[index]  # (C, H, W), already normalized
        augmented = self.transform(image)  # (C, H, W)
        return augmented, label


def denormalize(image: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """The image back in pixel space, for visualization or trigger inspection.

    image is (C, H, W) or (B, C, H, W). The statistics broadcast over both.
    """
    spec = DATASET_REGISTRY[dataset_name]

    mean = torch.tensor(spec.mean).view(-1, 1, 1)  # (C, 1, 1)
    std = torch.tensor(spec.std).view(-1, 1, 1)  # (C, 1, 1)

    original_range = image * std + mean  # same shape as image
    return original_range


# EuroSAT ships as a single folder of 27000 images with no train and test split,
# so this project defines the split. A fixed permutation keeps it identical across
# every run and process, since a checkpoint trained on 1 split and evaluated on
# another would silently score itself on its own training data.
EUROSAT_SPLIT_SEED = 0
EUROSAT_TEST_FRACTION = 0.2


def split_eurosat(root: str, transform) -> tuple[Dataset, Dataset]:
    """EuroSAT cut into (train, test) by a fixed permutation."""
    full = tv_datasets.EuroSAT(root=root, download=True, transform=transform)
    order = torch.randperm(
        len(full), generator=torch.Generator().manual_seed(EUROSAT_SPLIT_SEED)
    ).tolist()
    cut = int(len(full) * EUROSAT_TEST_FRACTION)

    train_ds = Subset(full, order[cut:])
    test_ds = Subset(full, order[:cut])
    return train_ds, test_ds


def load_clean_datasets(
    dataset_name: str,
    transform: transforms_v2.Compose,
    raw_data_dir: str,
) -> tuple[Dataset, Dataset]:
    """The (train, test) clean datasets for a dataset name.

    transform must not normalize, for the reason the module header gives. Tiny
    ImageNet is read from the ImageFolder layout BackdoorBench writes, with the
    validation split already reorganized into per-class folders.
    """
    spec: DatasetSpec = DATASET_REGISTRY[dataset_name]
    root = os.path.join(raw_data_dir, dataset_name)

    if spec.loader_kind == "svhn":
        train_ds = tv_datasets.SVHN(
            root=root, split="train", download=True, transform=transform
        )
        test_ds = tv_datasets.SVHN(
            root=root, split="test", download=True, transform=transform
        )
        return train_ds, test_ds

    if spec.loader_kind == "eurosat":
        train_ds, test_ds = split_eurosat(root, transform)
        return train_ds, test_ds

    if spec.loader_kind == "gtsrb":
        train_ds = tv_datasets.GTSRB(
            root=root, split="train", download=True, transform=transform
        )
        test_ds = tv_datasets.GTSRB(
            root=root, split="test", download=True, transform=transform
        )
        return train_ds, test_ds

    if spec.loader_kind == "image_folder":
        train_ds = tv_datasets.ImageFolder(
            os.path.join(root, "train"), transform=transform
        )
        test_ds = tv_datasets.ImageFolder(
            os.path.join(root, "val"), transform=transform
        )
        return train_ds, test_ds

    torchvision_cls = {
        "cifar10": tv_datasets.CIFAR10,
        "cifar100": tv_datasets.CIFAR100,
    }[spec.loader_kind]
    train_ds = torchvision_cls(
        root=root, train=True, download=True, transform=transform
    )
    test_ds = torchvision_cls(
        root=root, train=False, download=True, transform=transform
    )
    return train_ds, test_ds


def limit_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    """A reproducible random subset of dataset, or dataset itself when max_samples is None.

    numpy's Generator API is isolated from the global state seed_everything seeds,
    so this never perturbs the stream model init or shuffling draw from later. A
    random subset rather than a first-N slice matters for ImageFolder datasets,
    whose samples are sorted by class, so a slice would cover only the first 1 or 2
    classes.
    """
    dataset_size = len(dataset)
    if max_samples is None or max_samples >= dataset_size:
        return dataset

    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_size, size=max_samples, replace=False)
    subset = Subset(dataset, indices)
    return subset


def extract_labels(dataset: Dataset) -> list[int]:
    """The integer labels of a dataset, read without decoding images where possible.

    Decoding every image to read its label is slow on Tiny ImageNet, so the label
    arrays torchvision exposes come first and item indexing is the fallback.
    """
    if isinstance(dataset, Subset):
        parent_labels = extract_labels(dataset.dataset)
        selected_labels = [parent_labels[i] for i in dataset.indices]
        return selected_labels

    # CIFAR-10 and CIFAR-100 expose targets, ImageFolder exposes samples and
    # torchvision's GTSRB keeps its (path, label) pairs in _samples.
    if hasattr(dataset, "targets"):
        target_labels = [int(y) for y in dataset.targets]
        return target_labels
    if hasattr(dataset, "samples"):
        sample_labels = [int(y) for _, y in dataset.samples]
        return sample_labels
    if hasattr(dataset, "_samples"):
        gtsrb_labels = [int(y) for _, y in dataset._samples]
        return gtsrb_labels

    decoded_labels = [int(dataset[i][1]) for i in range(len(dataset))]
    return decoded_labels
