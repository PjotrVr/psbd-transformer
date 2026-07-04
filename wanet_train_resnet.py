from __future__ import annotations

import argparse
import json
import os

import torch

from benign_train_resnet import LResNetV2, ResNetTrainConfig, train_resnet_v2
from metrics import evaluate_loader_spec, save_metrics_json
from psbd import (
    evaluate_psbd,
    find_best_dropout_rate,
    load_model_from_checkpoint,
    run_psbd_sweep,
)
from utils import load_tensor, tensor_loader

DATASET_NAME = "cifar10"
POISON_RATE = 0.1
TARGET_LABEL = 0
ATTACK_NAME = "wanet"


def resolve_data_dir(
    dataset_name: str,
    poison_rate: float,
    target_label: int,
    data_root: str = "preprocessed_data",
) -> str:
    candidate = os.path.join(
        data_root,
        f"{dataset_name}_{ATTACK_NAME}_poison_rate={poison_rate}_target={target_label}",
    )
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError("Could not find poisoned WaNet data folder: " + candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet18V2 on WaNet poisoned data"
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME, type=str)
    parser.add_argument("--data-root", default="preprocessed_data", type=str)
    parser.add_argument("--poison-rate", default=POISON_RATE, type=float)
    parser.add_argument("--target-label", default=TARGET_LABEL, type=int)
    parser.add_argument("--max-epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--early-stopping-patience", default=15, type=int)
    parser.add_argument(
        "--train-with-backdoor-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-psbd", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--psbd-model-name",
        choices=["resnet18"],
        default="resnet18",
        type=str,
    )
    parser.add_argument("--psbd-mc-samples", default=10, type=int)
    parser.add_argument("--psbd-selection-fpr", default=0.1, type=float)
    parser.add_argument(
        "--psbd-dropout-rates", nargs="+", type=float, default=[0.1, 0.2, 0.4, 0.6, 0.8]
    )
    parser.add_argument(
        "--psbd-target-fprs", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.25]
    )
    parser.add_argument("--psbd-device", default="auto", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = resolve_data_dir(
        args.dataset_name,
        args.poison_rate,
        args.target_label,
        data_root=args.data_root,
    )

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

    num_classes = len(torch.unique(train_labels))

    config = ResNetTrainConfig(
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        milestones=(50, 75),
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
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
        run_name_prefix=f"wanet_resnet18v2_{args.dataset_name}_clean_test",
        num_classes=num_classes,
        backdoor_val_data=(
            backdoor_val_data if args.train_with_backdoor_validation else None
        ),
        backdoor_val_labels=(
            backdoor_val_labels if args.train_with_backdoor_validation else None
        ),
        config=config,
    )

    model = LResNetV2.load_from_checkpoint(clean_result["best_checkpoint_path"])

    clean_val_loader = tensor_loader(
        clean_val_data,
        clean_val_labels,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    backdoor_val_loader = tensor_loader(
        backdoor_val_data,
        backdoor_val_labels,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    clean_test_loader = tensor_loader(
        clean_test_data,
        clean_test_labels,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    backdoor_test_loader = tensor_loader(
        backdoor_test_data,
        backdoor_test_labels,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    val_loader_spec = {"clean": clean_val_loader, "backdoor": backdoor_val_loader}
    test_loader_spec = {"clean": clean_test_loader, "backdoor": backdoor_test_loader}

    val_metrics = evaluate_loader_spec(
        model=model,
        loader_spec=val_loader_spec,
    )
    test_metrics = evaluate_loader_spec(
        model=model,
        loader_spec=test_loader_spec,
    )

    run_metrics = {
        "run_dir": clean_result["run_dir"],
        "best_checkpoint_path": clean_result["best_checkpoint_path"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "trainer_test_metrics": clean_result["test_metrics"],
    }

    if args.psbd_device == "auto":
        psbd_device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        psbd_device = args.psbd_device

    if args.run_psbd:
        heldout_path = os.path.join(data_dir, "val_heldout_data.pt")
        heldout_labels_path = os.path.join(data_dir, "val_heldout_labels.pt")
        if os.path.exists(heldout_path) and os.path.exists(heldout_labels_path):
            heldout_data = load_tensor(data_dir, "val_heldout_data.pt")
            heldout_labels = load_tensor(data_dir, "val_heldout_labels.pt")
        else:
            heldout_data = clean_val_data
            heldout_labels = clean_val_labels

        heldout_loader = tensor_loader(
            heldout_data,
            heldout_labels,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        psbd_model = load_model_from_checkpoint(
            checkpoint_path=clean_result["best_checkpoint_path"],
            model_name=args.psbd_model_name,
            num_classes=num_classes,
            map_location=psbd_device,
        )

        val_sweep = run_psbd_sweep(
            model=psbd_model,
            target_label=args.target_label,
            clean_threshold_loader=heldout_loader,
            clean_eval_loader=clean_val_loader,
            backdoor_eval_loader=backdoor_val_loader,
            dropout_rates=tuple(float(rate) for rate in args.psbd_dropout_rates),
            target_fprs=tuple(float(fpr) for fpr in args.psbd_target_fprs),
            mc_samples=args.psbd_mc_samples,
            device=psbd_device,
        )

        selected_dropout_rate = find_best_dropout_rate(
            val_sweep,
            selection_fpr=args.psbd_selection_fpr,
        )

        test_eval = evaluate_psbd(
            model=psbd_model,
            target_label=args.target_label,
            clean_threshold_loader=heldout_loader,
            clean_eval_loader=clean_test_loader,
            backdoor_eval_loader=backdoor_test_loader,
            target_fprs=tuple(float(fpr) for fpr in args.psbd_target_fprs),
            mc_samples=args.psbd_mc_samples,
            dropout_rate=selected_dropout_rate,
            device=psbd_device,
        )

        psbd_metrics = {
            "checkpoint_path": clean_result["best_checkpoint_path"],
            "model_name": args.psbd_model_name,
            "data_dir": data_dir,
            "target_label": int(args.target_label),
            "mc_samples": int(args.psbd_mc_samples),
            "selection_fpr": float(args.psbd_selection_fpr),
            "dropout_rates": [float(rate) for rate in args.psbd_dropout_rates],
            "target_fprs": [float(fpr) for fpr in args.psbd_target_fprs],
            "selected_dropout_rate": float(selected_dropout_rate),
            "val_sweep": val_sweep,
            "test_eval": test_eval,
        }
        with open(
            os.path.join(clean_result["run_dir"], "psbd_metrics.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(psbd_metrics, handle, indent=2)
        run_metrics["psbd_metrics_path"] = os.path.join(
            clean_result["run_dir"], "psbd_metrics.json"
        )

    save_metrics_json(
        run_metrics,
        os.path.join(clean_result["run_dir"], "backdoor_metrics.json"),
    )

    print("Run dir:", clean_result["run_dir"])
    print("Best checkpoint:", clean_result["best_checkpoint_path"])
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    if args.run_psbd:
        print("PSBD metrics saved to:", run_metrics["psbd_metrics_path"])

    with open(
        os.path.join(clean_result["run_dir"], "summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump({**clean_result, "backdoor_metrics": run_metrics}, handle, indent=2)


if __name__ == "__main__":
    main()
