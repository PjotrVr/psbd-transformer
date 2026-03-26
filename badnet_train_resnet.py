from __future__ import annotations

import os

import torch

from benign_train_resnet import (LightningResNetV2, ResNetTrainConfig,
                                 create_trainer, tensor_loader,
                                 train_resnet_v2)

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


def load_tensor(data_dir: str, name: str) -> torch.Tensor:
    return torch.load(os.path.join(data_dir, name), map_location="cpu")


def main() -> None:
    data_dir = resolve_data_dir()

    clean_train_data = load_tensor(data_dir, "clean_train_data.pt")
    clean_train_labels = load_tensor(data_dir, "clean_train_labels.pt")
    backdoor_train_data = load_tensor(data_dir, "backdoor_train_data.pt")
    backdoor_train_labels = load_tensor(data_dir, "backdoor_train_labels.pt")

    clean_val_data = load_tensor(data_dir, "clean_val_data.pt")
    clean_val_labels = load_tensor(data_dir, "clean_val_labels.pt")

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

    backdoor_loader = tensor_loader(
        backdoor_test_data,
        backdoor_test_labels,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = LightningResNetV2.load_from_checkpoint(clean_result["best_checkpoint_path"])
    trainer = create_trainer(run_dir=clean_result["run_dir"], config=config)[0]
    backdoor_metrics = trainer.test(model, dataloaders=backdoor_loader, verbose=True)[0]

    print("Run dir:", clean_result["run_dir"])
    print("Best checkpoint:", clean_result["best_checkpoint_path"])
    print("Clean test metrics:", clean_result["test_metrics"])
    print("Backdoor test metrics:", backdoor_metrics)


if __name__ == "__main__":
    main()
