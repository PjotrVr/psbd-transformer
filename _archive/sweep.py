"""Orchestration for the PSBD dropout-rate sweep.

Each function does one thing, and run_sweep reads top to bottom as the sequence
of steps a single experiment goes through.
"""

import os

import torch
from torch.utils.data import DataLoader

from backdoor_data import (
    balance_by_class,
    load_backdoor_splits,
    split_validation_and_eval,
)
from config import RunConfig, dataset_name_from_folder
from datasets import build_transform, load_clean_datasets
from detection import (
    attack_success_rate,
    auroc,
    clean_accuracy,
    detection_rates,
    threshold_from_validation,
)
from dropout import configure_dropout, reset_dropout
from experiment_io import save_metrics, save_scores
from inference import build_baseline_cache, compute_psu_and_shift
from models import load_checkpoint


def _checkpoint_path(config: RunConfig, folder_name: str) -> str:
    return os.path.join(config.weights_dir, folder_name, "attack_result.pt")


def build_eval_loaders(
    config: RunConfig, folder_name: str
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Assemble the validation, clean, and backdoor loaders for one attack."""
    dataset_name = dataset_name_from_folder(folder_name)
    transform = build_transform(dataset_name)

    _, clean_test = load_clean_datasets(dataset_name, transform, config.raw_data_dir)
    backdoor_test, clean_counterparts = load_backdoor_splits(
        folder_name, clean_test, transform, config.weights_dir, config.trigger_label
    )

    clean_val, clean_eval, backdoor_eval = split_validation_and_eval(
        clean_counterparts, backdoor_test, config.clean_val_size, config.seed
    )
    clean_eval, backdoor_eval = balance_by_class(
        clean_eval, backdoor_eval, config.examples_per_class, config.seed
    )

    make_loader = lambda ds: DataLoader(ds, batch_size=config.batch_size, shuffle=False)
    return make_loader(clean_val), make_loader(clean_eval), make_loader(backdoor_eval)


def _metric_rows_for_rate(
    folder_name: str,
    config: RunConfig,
    rate: float,
    scores: dict[str, torch.Tensor],
    shifts: dict[str, float],
    behavior: dict[str, float],
) -> list[dict]:
    """One row per quantile at a fixed dropout rate."""
    rows = []
    for quantile in config.psbd_quantiles:
        threshold = threshold_from_validation(scores["validation"], quantile)
        tpr, fpr = detection_rates(scores["clean"], scores["backdoor"], threshold)
        rows.append(
            {
                "folder_name": folder_name,
                "architecture": config.architecture,
                "dropout_placement": config.dropout_placement,
                "dropout_rate": rate,
                "quantile": quantile,
                "auroc": round(auroc(scores["clean"], scores["backdoor"]), 4),
                "tpr": round(tpr, 4),
                "fpr": round(fpr, 4),
                "threshold": round(threshold, 6),
                "asr_before": round(behavior["asr"], 4),
                "asr_after": round((1 - tpr) * behavior["asr"], 4),
                "clean_accuracy": round(behavior["clean_accuracy"], 4),
                "shift_clean_val": round(shifts["validation"], 4),
                "shift_clean": round(shifts["clean"], 4),
                "shift_backdoor": round(shifts["backdoor"], 4),
            }
        )
    return rows


def sweep_one_attack(
    config: RunConfig,
    folder_name: str,
    device: torch.device,
    val_loader: DataLoader,
    clean_loader: DataLoader,
    backdoor_loader: DataLoader,
) -> None:
    """Run the full dropout-rate sweep for one attack folder and persist it."""
    experiment_dir = os.path.join(config.results_dir(), folder_name)
    os.makedirs(experiment_dir, exist_ok=True)

    model = load_checkpoint(
        config.architecture, _checkpoint_path(config, folder_name), device
    )
    reset_dropout(model, config.dropout_placement)

    behavior = {
        "asr": attack_success_rate(model, backdoor_loader, device, config.use_bfloat16),
        "clean_accuracy": clean_accuracy(
            model, clean_loader, device, config.use_bfloat16
        ),
    }
    print(
        f"{folder_name}: ASR={behavior['asr']:.4f}, CA={behavior['clean_accuracy']:.4f}"
    )

    baselines = {
        "validation": build_baseline_cache(
            model, val_loader, device, config.use_bfloat16
        ),
        "clean": build_baseline_cache(model, clean_loader, device, config.use_bfloat16),
        "backdoor": build_baseline_cache(
            model, backdoor_loader, device, config.use_bfloat16
        ),
    }
    loaders = {
        "validation": val_loader,
        "clean": clean_loader,
        "backdoor": backdoor_loader,
    }

    all_rows: list[dict] = []
    for rate in config.dropout_rates:
        configure_dropout(model, rate, config.dropout_placement)

        scores: dict[str, torch.Tensor] = {}
        shifts: dict[str, float] = {}
        for split in ("validation", "clean", "backdoor"):
            scores[split], shifts[split] = compute_psu_and_shift(
                model,
                loaders[split],
                baselines[split],
                device,
                config.forward_passes,
                config.use_bfloat16,
                config.seed,
            )

        save_scores(
            experiment_dir,
            rate,
            scores["validation"],
            scores["clean"],
            scores["backdoor"],
        )
        all_rows.extend(
            _metric_rows_for_rate(folder_name, config, rate, scores, shifts, behavior)
        )
        gap = shifts["clean"] - shifts["backdoor"]
        print(f"\trate={rate:.2f} shift_clean={shifts['clean']:.3f} gap={gap:.3f}")

    save_metrics(experiment_dir, all_rows)
    reset_dropout(model, config.dropout_placement)


def run_sweep(config: RunConfig, device: torch.device) -> list[dict]:
    """Sweep every attack folder that has a checkpoint, returning failures."""
    os.makedirs(config.results_dir(), exist_ok=True)
    failures: list[dict] = []

    for folder_name in config.attack_folders:
        if not os.path.exists(_checkpoint_path(config, folder_name)):
            print(f"Skipping {folder_name}, checkpoint not found")
            continue
        try:
            val_loader, clean_loader, backdoor_loader = build_eval_loaders(
                config, folder_name
            )
            sweep_one_attack(
                config, folder_name, device, val_loader, clean_loader, backdoor_loader
            )
        except Exception as error:
            print(f"FAILED {folder_name}: {error}")
            failures.append({"folder_name": folder_name, "error": str(error)})

    if failures:
        print("Failed:", [item["folder_name"] for item in failures])
    return failures
