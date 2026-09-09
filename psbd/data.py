"""Clean dataset loading, transforms, and label extraction.

Side effects (disk reads, downloads) live here so the detection and analysis
code can stay pure.

Boundary contract on normalization. A dataset built by load_clean_datasets must
yield images in the 0 to 1 range, unnormalized. Every attack in this project
stamps its trigger in pixel space, so normalization has to happen strictly after
the trigger is applied. The dataset wrappers in psbd.poisoning own that final
step: they take a normalize callable and apply it last. Baking a Normalize into
the transform handed to load_clean_datasets would silently poison normalized
tensors and produce triggers that do not match the attack's paper. Use
base_image_transform here and construct the Normalize separately at the call
site, where the ordering is visible.
"""

import os

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import Dataset, Subset
from torchvision import datasets as tv_datasets

from .config import DATASET_REGISTRY, DatasetSpec


def base_image_transform(image_size: int) -> transforms_v2.Compose:
    """Resize to image_size and convert to a 0-to-1 CHW tensor, without normalizing.

    This is the transform the poisoning pipeline requires. image_size is the
    dataset's native trigger resolution, not 224: the model wrapper does its own
    upscale to 224, so resizing here would place the trigger on the wrong pixels.
    """
    transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((image_size, image_size)),
            transforms_v2.ToTensor(),
        ]
    )
    return transform


def denormalize(image: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """Undo normalization for visualization or trigger inspection.

    image is (C, H, W) or (B, C, H, W). The statistics broadcast over both.
    """
    spec = DATASET_REGISTRY[dataset_name]

    mean = torch.tensor(spec.mean).view(-1, 1, 1)  # (C, 1, 1)
    std = torch.tensor(spec.std).view(-1, 1, 1)  # (C, 1, 1)

    original_range = image * std + mean
    return original_range


# EuroSAT ships as one folder of 27000 images with no train and test split, so
# this project defines one. A fixed permutation keeps it identical across every
# run and every process, which matters because a checkpoint trained on one split
# and evaluated on another would silently score itself on its own training data.
EUROSAT_SPLIT_SEED = 0
EUROSAT_TEST_FRACTION = 0.2


def split_eurosat(root: str, transform) -> tuple[Dataset, Dataset]:
    """EuroSAT cut into (train, test) by a fixed permutation."""
    full = tv_datasets.EuroSAT(root=root, download=True, transform=transform)
    order = torch.randperm(
        len(full), generator=torch.Generator().manual_seed(EUROSAT_SPLIT_SEED)
    ).tolist()
    cut = int(len(full) * EUROSAT_TEST_FRACTION)
    return Subset(full, order[cut:]), Subset(full, order[:cut])


def load_clean_datasets(
    dataset_name: str,
    transform: transforms_v2.Compose,
    raw_data_dir: str,
) -> tuple[Dataset, Dataset]:
    """Return (train, test) clean datasets for the given dataset name.

    transform must not normalize, for the reason the module docstring gives.
    Tiny ImageNet is read from the ImageFolder layout BackdoorBench writes, where
    the validation split is already reorganized into per-class folders.
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
        return split_eurosat(root, transform)

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

    Uses numpy's Generator API, which is isolated from the legacy global
    np.random state seed_everything seeds, so calling this never perturbs the
    RNG stream that model init or DataLoader shuffling later draw from. A random
    subset rather than a first-N slice matters for the ImageFolder-backed loaders
    (Tiny ImageNet), whose samples are listed sorted by class, so a first-N slice
    would cover only the first 1 or 2 classes.
    """
    dataset_size = len(dataset)
    if max_samples is None or max_samples >= dataset_size:
        return dataset

    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_size, size=max_samples, replace=False)
    subset = Subset(dataset, indices)
    return subset


def extract_labels(dataset: Dataset) -> list[int]:
    """Read integer labels without decoding image tensors where possible.

    Decoding every image just to read its label is the slow path the notebook
    took on Tiny ImageNet, so prefer the label arrays torchvision exposes and
    fall back to item indexing only when they are absent.
    """
    if isinstance(dataset, Subset):
        parent_labels = extract_labels(dataset.dataset)
        selected_labels = [parent_labels[i] for i in dataset.indices]
        return selected_labels

    if hasattr(dataset, "targets"):  # CIFAR-10, CIFAR-100
        return [int(y) for y in dataset.targets]
    if hasattr(dataset, "samples"):  # ImageFolder
        return [int(y) for _, y in dataset.samples]
    if hasattr(dataset, "_samples"):  # torchvision GTSRB
        return [int(y) for _, y in dataset._samples]

    decoded_labels = [int(dataset[i][1]) for i in range(len(dataset))]
    return decoded_labels
