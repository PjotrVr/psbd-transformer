from __future__ import annotations

import json
import os

import torch

from benign_train_resnet import (
    LightningResNetV2,
    ResNetTrainConfig,
    train_resnet_v2,
)
from metrics import evaluate_backdoor_pair, save_metrics_json
from utils import load_tensor

DATASET_NAME = "cifar10"
POISON_RATE = 0.1
TARGET_LABEL = 0
ATTACK_NAME_CANDIDATES = ("badnet", "badnets")


def resolve_data_dir() -> str:
    for attack_name in ATTACK_NAME_CANDIDATES:
        candidate = os.path.join(
            "preprocessed_data",
            f"{DATASET_NAME}_{attack_name}_poison_rate={POISON_RATE}_target={TARGET_LABEL}",
        )
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find poisoned CIFAR10 data folder. "
        "Expected one of: "
        + ", ".join(
            [
                os.path.join(
                    "preprocessed_data",
                    f"{DATASET_NAME}_{attack}_poison_rate={POISON_RATE}_target={TARGET_LABEL}",
                )
                for attack in ATTACK_NAME_CANDIDATES
            ]
        )
    )


def main() -> None:
    data_dir = resolve_data_dir()

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

    # Keep validation clean for model selection, and report both clean and backdoor test.
    config = ResNetTrainConfig(
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        milestones=(50, 75),
        max_epochs=100,
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
        run_name_prefix="badnet_resnet18v2_cifar10_clean_test",
        num_classes=10,
        config=config,
    )

    model = LightningResNetV2.load_from_checkpoint(clean_result["best_checkpoint_path"])

    val_metrics = evaluate_backdoor_pair(
        model=model,
        clean_data=clean_val_data,
        clean_labels=clean_val_labels,
        backdoor_data=backdoor_val_data,
        backdoor_labels=backdoor_val_labels,
        batch_size=config.batch_size,
    )
    test_metrics = evaluate_backdoor_pair(
        model=model,
        clean_data=clean_test_data,
        clean_labels=clean_test_labels,
        backdoor_data=backdoor_test_data,
        backdoor_labels=backdoor_test_labels,
        batch_size=config.batch_size,
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
