from __future__ import annotations

import json
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from benign_train_resnet import LightningResNetV2, ResNetTrainConfig, train_resnet_v2
from metrics import evaluate_backdoor_pair, save_metrics_json
from utils import load_tensor

DATASET_NAME = "cifar10"
POISON_RATE = 0.1
TARGET_LABEL = 0
ATTACK_NAME = "wanet"


def resolve_data_dir() -> str:
    candidate = os.path.join(
        "preprocessed_data",
        f"{DATASET_NAME}_{ATTACK_NAME}_poison_rate={POISON_RATE}_target={TARGET_LABEL}",
    )
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        "Could not find poisoned CIFAR10 WaNet data folder: " + candidate
    )


def main():
    data_dir = resolve_data_dir()
    train_max_epochs = int(os.environ.get("TRAIN_MAX_EPOCHS", "100"))

    clean_train_data = load_tensor(data_dir, "clean_train_data.pt")
    clean_train_labels = load_tensor(data_dir, "clean_train_labels.pt")
    backdoor_train_data = load_tensor(data_dir, "backdoor_train_data.pt")
    backdoor_train_labels = load_tensor(data_dir, "backdoor_train_labels.pt")

    clean_val_data = load_tensor(data_dir, "clean_val_data.pt")
    clean_val_labels = load_tensor(data_dir, "clean_val_labels.pt")
    backdoor_val_data = load_tensor(data_dir, "backdoor_val_data.pt")
    backdoor_val_labels = load_tensor(data_dir, "backdoor_val_labels.pt")

    clean_test_data = load_tensor(data_dir, "clean_test_data.pt")
    clean_test_labels = load_tensor(data_dir, "clean_test_labels.pt")
    backdoor_test_data = load_tensor(data_dir, "backdoor_test_data.pt")
    backdoor_test_labels = load_tensor(data_dir, "backdoor_test_labels.pt")

    train_data = torch.cat([clean_train_data, backdoor_train_data], dim=0)
    train_labels = torch.cat([clean_train_labels, backdoor_train_labels], dim=0)

    config = ResNetTrainConfig(
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        milestones=(50, 75),
        max_epochs=train_max_epochs,
        batch_size=128,
        num_workers=4,
        early_stopping_patience=15,
        seed=0,
        precision="16-mixed",
    )

    clean_result = train_resnet_v2(
        train_data=train_data,
        train_labels=train_labels,
        val_data=clean_val_data,
        val_labels=clean_val_labels,
        test_data=clean_test_data,
        test_labels=clean_test_labels,
        run_name_prefix="wanet_resnet18v2_cifar10_clean_test",
        num_classes=10,
        config=config,
    )

    model = LightningResNetV2.load_from_checkpoint(clean_result["best_checkpoint_path"])

    clean_val_loader = DataLoader(
        TensorDataset(clean_val_data, clean_val_labels),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    backdoor_val_loader = DataLoader(
        TensorDataset(backdoor_val_data, backdoor_val_labels),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    clean_test_loader = DataLoader(
        TensorDataset(clean_test_data, clean_test_labels),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    backdoor_test_loader = DataLoader(
        TensorDataset(backdoor_test_data, backdoor_test_labels),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_metrics = evaluate_backdoor_pair(
        model=model,
        clean_loader=clean_val_loader,
        backdoor_loader=backdoor_val_loader,
    )
    test_metrics = evaluate_backdoor_pair(
        model=model,
        clean_loader=clean_test_loader,
        backdoor_loader=backdoor_test_loader,
    )

    run_metrics = {
        "run_dir": clean_result["run_dir"],
        "best_checkpoint_path": clean_result["best_checkpoint_path"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "trainer_test_metrics": clean_result["test_metrics"],
    }

    save_metrics_json(
        run_metrics,
        os.path.join(clean_result["run_dir"], "backdoor_metrics.json"),
    )

    print("Run dir:", clean_result["run_dir"])
    print("Best checkpoint:", clean_result["best_checkpoint_path"])
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    with open(
        os.path.join(clean_result["run_dir"], "summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump({**clean_result, "backdoor_metrics": run_metrics}, handle, indent=2)


if __name__ == "__main__":
    main()
