"""Clean dataset loading, transforms, and label extraction.

Side effects (disk reads, downloads) live here so the detection and analysis
code can stay pure.
"""

import os

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import Dataset, Subset
from torchvision import datasets as tv_datasets

from .config import DATASET_REGISTRY, DatasetSpec


def build_transform(dataset_name: str) -> transforms_v2.Compose:
    """ViT-B/16 expects 224x224 inputs, so every dataset is resized up."""
    spec = DATASET_REGISTRY[dataset_name]
    return transforms_v2.Compose(
        [
            transforms_v2.Resize((224, 224)),
            transforms_v2.ToTensor(),
            transforms_v2.Normalize(mean=spec.mean, std=spec.std),
        ]
    )


def denormalize(image: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """Undo normalization for visualization or trigger inspection."""
    spec = DATASET_REGISTRY[dataset_name]
    mean = torch.tensor(spec.mean).view(-1, 1, 1)
    std = torch.tensor(spec.std).view(-1, 1, 1)
    return image * std + mean


def load_clean_datasets(
    dataset_name: str,
    transform: transforms_v2.Compose,
    raw_data_dir: str,
) -> tuple[Dataset, Dataset]:
    """Return (train, test) clean datasets for the given dataset name.

    Tiny ImageNet is read from the ImageFolder layout BackdoorBench writes,
    where the validation split is already reorganized into per-class folders.
    """
    spec: DatasetSpec = DATASET_REGISTRY[dataset_name]
    root = os.path.join(raw_data_dir, dataset_name)

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
    would cover only the first one or two classes.
    """
    n = len(dataset)
    if max_samples is None or max_samples >= n:
        return dataset
    indices = np.random.default_rng(seed).choice(n, size=max_samples, replace=False)
    return Subset(dataset, indices)


def extract_labels(dataset: Dataset) -> list[int]:
    """Read integer labels without decoding image tensors where possible.

    Decoding every image just to read its label is the slow path the notebook
    took on Tiny ImageNet, so prefer the label arrays torchvision exposes and
    fall back to item indexing only when they are absent.
    """
    if isinstance(dataset, Subset):
        parent_labels = extract_labels(dataset.dataset)
        return [parent_labels[i] for i in dataset.indices]

    if hasattr(dataset, "targets"):  # CIFAR-10, CIFAR-100
        return [int(y) for y in dataset.targets]
    if hasattr(dataset, "samples"):  # ImageFolder
        return [int(y) for _, y in dataset.samples]
    if hasattr(dataset, "_samples"):  # torchvision GTSRB
        return [int(y) for _, y in dataset._samples]

    return [int(dataset[i][1]) for i in range(len(dataset))]
