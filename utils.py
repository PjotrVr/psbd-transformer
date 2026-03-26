from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torchvision import datasets

SUPPORTED_DATASETS = ("cifar10", "cifar100", "gtsrb")
BENIGN_METADATA = (
    "Train is split into 90% train and 10% validation with seed 0. Test is unchanged."
)


@dataclass
class BenignSplits:
    train_data: torch.Tensor
    train_labels: torch.Tensor
    val_data: torch.Tensor
    val_labels: torch.Tensor
    test_data: torch.Tensor
    test_labels: torch.Tensor


@dataclass
class PoisonBaseSplits:
    clean_train_data: torch.Tensor
    clean_train_labels: torch.Tensor
    poison_source_train_data: torch.Tensor
    poison_source_train_labels: torch.Tensor
    clean_val_data: torch.Tensor
    clean_val_labels: torch.Tensor
    val_heldout_data: torch.Tensor
    val_heldout_labels: torch.Tensor
    clean_test_data: torch.Tensor
    clean_test_labels: torch.Tensor


def get_timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def float_eq(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) < float(tol)


def enable_dropout(module: nn.Module):
    if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        module.train()


def load_tensor(data_dir: str, name: str, mmap=True, weights_only=True) -> torch.Tensor:
    return torch.load(
        os.path.join(data_dir, name),
        map_location="cpu",
        mmap=mmap,
        weights_only=weights_only,
    )


def _tensorize_dataset(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor(
        [dataset[i][1] for i in range(len(dataset))], dtype=torch.long
    )
    return data, labels


def load_dataset_tensors(
    dataset_name: str, raw_data_dir: str = "./raw_data"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(
            root=os.path.join(raw_data_dir, "cifar10"),
            train=True,
            download=True,
            transform=transform,
        )
        test_set = datasets.CIFAR10(
            root=os.path.join(raw_data_dir, "cifar10"),
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "cifar100":
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR100(
            root=os.path.join(raw_data_dir, "cifar100"),
            train=True,
            download=True,
            transform=transform,
        )
        test_set = datasets.CIFAR100(
            root=os.path.join(raw_data_dir, "cifar100"),
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "gtsrb":
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )
        gtsrb_root = os.path.join(raw_data_dir, "gtsrb")
        train_set = datasets.GTSRB(
            root=gtsrb_root, split="train", download=True, transform=transform
        )
        test_set = datasets.GTSRB(
            root=gtsrb_root, split="test", download=True, transform=transform
        )
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Supported: {SUPPORTED_DATASETS}"
        )

    train_data, train_labels = _tensorize_dataset(train_set)
    test_data, test_labels = _tensorize_dataset(test_set)
    return train_data, train_labels, test_data, test_labels


def split_train_val(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = len(train_data)
    num_val = int(num_samples * val_ratio)

    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator)
    val_indices = perm[:num_val]
    train_indices = perm[num_val:]

    split_train_data = train_data[train_indices]
    split_train_labels = train_labels[train_indices]
    val_data = train_data[val_indices]
    val_labels = train_labels[val_indices]
    return split_train_data, split_train_labels, val_data, val_labels


def create_benign_splits(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    val_ratio: float,
    seed: int,
) -> BenignSplits:
    split_train_data, split_train_labels, val_data, val_labels = split_train_val(
        train_data,
        train_labels,
        val_ratio=val_ratio,
        seed=seed,
    )
    return BenignSplits(
        train_data=split_train_data,
        train_labels=split_train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_data,
        test_labels=test_labels,
    )


def _split_clean_and_poison(
    data: torch.Tensor,
    labels: torch.Tensor,
    poison_rate: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = len(data)
    num_poison = int(num_samples * poison_rate)
    num_clean = num_samples - num_poison

    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator)
    clean_idx = perm[:num_clean]
    poison_idx = perm[num_clean:]

    clean_data = data[clean_idx]
    clean_labels = labels[clean_idx]
    poison_data = data[poison_idx]
    poison_labels = labels[poison_idx]
    return clean_data, clean_labels, poison_data, poison_labels


def _split_validation_for_threshold(
    val_data: torch.Tensor,
    val_labels: torch.Tensor,
    heldout_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = len(val_data)
    num_heldout = int(num_samples * heldout_ratio)

    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator)
    keep_idx = perm[: num_samples - num_heldout]
    heldout_idx = perm[num_samples - num_heldout :]

    keep_data = val_data[keep_idx]
    keep_labels = val_labels[keep_idx]
    heldout_data = val_data[heldout_idx]
    heldout_labels = val_labels[heldout_idx]
    return keep_data, keep_labels, heldout_data, heldout_labels


def create_poison_base_splits(
    benign_splits: BenignSplits,
    poison_rate: float,
    val_heldout_ratio: float,
    seed: int,
) -> PoisonBaseSplits:
    (
        clean_train_data,
        clean_train_labels,
        poison_source_train_data,
        poison_source_train_labels,
    ) = _split_clean_and_poison(
        benign_splits.train_data,
        benign_splits.train_labels,
        poison_rate=poison_rate,
        seed=seed,
    )

    clean_val_data, clean_val_labels, val_heldout_data, val_heldout_labels = (
        _split_validation_for_threshold(
            benign_splits.val_data,
            benign_splits.val_labels,
            heldout_ratio=val_heldout_ratio,
            seed=seed,
        )
    )

    return PoisonBaseSplits(
        clean_train_data=clean_train_data,
        clean_train_labels=clean_train_labels,
        poison_source_train_data=poison_source_train_data,
        poison_source_train_labels=poison_source_train_labels,
        clean_val_data=clean_val_data,
        clean_val_labels=clean_val_labels,
        val_heldout_data=val_heldout_data,
        val_heldout_labels=val_heldout_labels,
        clean_test_data=benign_splits.test_data,
        clean_test_labels=benign_splits.test_labels,
    )


def apply_transform_to_all(
    data: torch.Tensor, labels: torch.Tensor, transform_obj
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, poisoned_data, poisoned_labels = _generate_poisoned_subset(
        data, labels, transform_obj
    )
    return poisoned_data, poisoned_labels


def _generate_poisoned_subset(
    data: torch.Tensor,
    labels: torch.Tensor,
    transform_obj,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    poisoned_data, poisoned_labels = transform_obj.transform(data, labels)
    clean_data = torch.empty(0, *data.shape[1:], dtype=data.dtype, device=data.device)
    clean_labels = torch.empty(0, dtype=labels.dtype, device=labels.device)
    return clean_data, clean_labels, poisoned_data, poisoned_labels


def save_benign_splits(save_root: str, dataset_name: str, splits: BenignSplits) -> str:
    output_dir = os.path.join(save_root, f"{dataset_name}_benign")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(splits.train_data, os.path.join(output_dir, "train_data.pt"))
    torch.save(splits.train_labels, os.path.join(output_dir, "train_labels.pt"))
    torch.save(splits.val_data, os.path.join(output_dir, "val_data.pt"))
    torch.save(splits.val_labels, os.path.join(output_dir, "val_labels.pt"))
    torch.save(splits.test_data, os.path.join(output_dir, "test_data.pt"))
    torch.save(splits.test_labels, os.path.join(output_dir, "test_labels.pt"))

    with open(
        os.path.join(output_dir, "metadata.txt"), "w", encoding="utf-8"
    ) as handle:
        handle.write(BENIGN_METADATA)

    return output_dir


def save_poisoned_splits(
    save_root: str,
    dataset_name: str,
    attack_name: str,
    poison_rate: float,
    target_label: int,
    clean_train_data: torch.Tensor,
    clean_train_labels: torch.Tensor,
    backdoor_train_data: torch.Tensor,
    backdoor_train_labels: torch.Tensor,
    clean_val_data: torch.Tensor,
    clean_val_labels: torch.Tensor,
    val_heldout_data: torch.Tensor,
    val_heldout_labels: torch.Tensor,
    backdoor_val_data: torch.Tensor,
    backdoor_val_labels: torch.Tensor,
    clean_test_data: torch.Tensor,
    clean_test_labels: torch.Tensor,
    backdoor_test_data: torch.Tensor,
    backdoor_test_labels: torch.Tensor,
) -> str:
    output_dir = os.path.join(
        save_root,
        f"{dataset_name}_{attack_name}_poison_rate={poison_rate}_target={target_label}",
    )
    os.makedirs(output_dir, exist_ok=True)

    torch.save(clean_train_data, os.path.join(output_dir, "clean_train_data.pt"))
    torch.save(clean_train_labels, os.path.join(output_dir, "clean_train_labels.pt"))
    torch.save(backdoor_train_data, os.path.join(output_dir, "backdoor_train_data.pt"))
    torch.save(
        backdoor_train_labels, os.path.join(output_dir, "backdoor_train_labels.pt")
    )

    torch.save(clean_val_data, os.path.join(output_dir, "clean_val_data.pt"))
    torch.save(clean_val_labels, os.path.join(output_dir, "clean_val_labels.pt"))
    torch.save(val_heldout_data, os.path.join(output_dir, "val_heldout_data.pt"))
    torch.save(val_heldout_labels, os.path.join(output_dir, "val_heldout_labels.pt"))
    torch.save(backdoor_val_data, os.path.join(output_dir, "backdoor_val_data.pt"))
    torch.save(backdoor_val_labels, os.path.join(output_dir, "backdoor_val_labels.pt"))

    torch.save(clean_test_data, os.path.join(output_dir, "clean_test_data.pt"))
    torch.save(clean_test_labels, os.path.join(output_dir, "clean_test_labels.pt"))
    torch.save(backdoor_test_data, os.path.join(output_dir, "backdoor_test_data.pt"))
    torch.save(
        backdoor_test_labels, os.path.join(output_dir, "backdoor_test_labels.pt")
    )

    return output_dir
