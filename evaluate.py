"""Atomic, reusable evaluation: one model, one dataset, clean or under attack.

Three layers, thinnest to widest:
  evaluate_benign / evaluate_attack   model in, metrics out, no filesystem at
                                       all, so a training script can call
                                       these directly on the model it just
                                       trained, no save-then-reload needed.
  evaluate_checkpoint                 a checkpoint path in, metrics out. Loads
                                       the model and its args.json sidecar,
                                       then delegates to one of the two above.
  evaluate_all_checkpoints            loops evaluate_checkpoint over every
                                       folder in checkpoints_dir, writing
                                       results_dir/<folder>/metrics.json.

Every directory this module touches is a parameter with a plain default, not
a bare module-level constant read inside a function body, so any of these
are safe to call against a different tree (a test fixture, a scratch export)
just by passing a different argument.
"""

import json
import os
import shutil

import torch

from attacks import build_attack, default_config
from defences.detection import (
    accuracy_by_class_from_counts,
    attack_success_rate,
    class_correct_and_total,
    clean_accuracy,
    pooled_accuracy_from_counts,
)
from loaders import build_clean_loader, build_poisoned_loader
from models import load_checkpoint
from utils.config import DATASET_REGISTRY

CHECKPOINTS_DIR = "checkpoints"
RESULTS_DIR = "results"


def evaluate_benign(
    model,
    dataset_name: str,
    device,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
) -> dict:
    """Clean accuracy, total and per class, for a model with no attack of its own."""
    loader = build_clean_loader(dataset_name, raw_data_dir, batch_size)
    num_classes = DATASET_REGISTRY[dataset_name].num_classes
    correct, total = class_correct_and_total(
        model, loader, device, num_classes, use_bfloat16=True
    )
    return {
        "clean_accuracy": pooled_accuracy_from_counts(correct, total),
        "clean_accuracy_by_class": accuracy_by_class_from_counts(correct, total),
    }


def evaluate_attack(
    model,
    dataset_name: str,
    attack_name: str,
    config,
    target_label: int,
    device,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
) -> dict:
    """Attack success rate and clean accuracy, for a model probed with one attack.

    config is the attack's own config dataclass, for example
    BadNetConfig(label_mode="all_to_all") from attacks.badnet, or whatever
    attacks.default_config(attack_name) returns. Applies the trigger to every
    eligible test image, never a poison_rate sample of it: poison_rate only
    controls how many training images get poisoned, eval always asks about
    the whole eligible test set.
    """
    image_size = DATASET_REGISTRY[dataset_name].image_size
    attack = build_attack(attack_name, config, image_size, target_label)

    clean_loader = build_clean_loader(dataset_name, raw_data_dir, batch_size)
    poisoned_loader = build_poisoned_loader(
        dataset_name, attack, raw_data_dir, batch_size
    )

    return {
        "asr": attack_success_rate(model, poisoned_loader, device, use_bfloat16=True),
        "clean_accuracy": clean_accuracy(
            model, clean_loader, device, use_bfloat16=True
        ),
    }


def read_args_json(checkpoint_dir: str) -> dict:
    with open(os.path.join(checkpoint_dir, "args.json")) as handle:
        return json.load(handle)


def evaluate_checkpoint(
    checkpoint_path: str, device, raw_data_dir: str = "raw_data", batch_size: int = 64
) -> dict:
    """Load one checkpoint and its args.json, evaluate it as benign or under its own attack."""
    args = read_args_json(os.path.dirname(checkpoint_path))
    model = load_checkpoint(args["architecture"], checkpoint_path, device)

    if args["attack"] == "benign":
        metrics = evaluate_benign(
            model, args["dataset"], device, raw_data_dir, batch_size
        )
        return {
            "architecture": args["architecture"],
            "dataset": args["dataset"],
            **metrics,
        }

    config = default_config(args["attack"])
    metrics = evaluate_attack(
        model,
        args["dataset"],
        args["attack"],
        config,
        args["target_label"],
        device,
        raw_data_dir,
        batch_size,
    )
    return {
        "architecture": args["architecture"],
        "dataset": args["dataset"],
        "attack": args["attack"],
        "label_mode": args["label_mode"],
        "poison_rate": args["poison_rate"],
        "target_label": args["target_label"],
        **metrics,
    }


def save_metrics(output_path: str, metrics: dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(metrics, handle, indent=2)


def list_checkpoint_folders(checkpoints_dir: str) -> list[str]:
    if not os.path.isdir(checkpoints_dir):
        return []
    return sorted(
        name
        for name in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, name))
    )


def mirror_results_folders(results_dir: str, folder_names: list[str]) -> None:
    """results_dir matches folder_names exactly, one folder per checkpoint.

    Creates a folder for every current checkpoint, even ones not yet
    processed, and removes any leftover folder that no longer matches a
    checkpoint, for example one a renaming migration retired.
    """
    os.makedirs(results_dir, exist_ok=True)
    wanted = set(folder_names)
    existing = {
        name
        for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name))
    }
    for orphan in existing - wanted:
        shutil.rmtree(os.path.join(results_dir, orphan))
    for folder_name in folder_names:
        os.makedirs(os.path.join(results_dir, folder_name), exist_ok=True)


def evaluate_all_checkpoints(
    checkpoints_dir: str = CHECKPOINTS_DIR,
    results_dir: str = RESULTS_DIR,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
    folder_filter: list[str] | None = None,
    device: torch.device | None = None,
) -> None:
    """Evaluate every checkpoint under checkpoints_dir, writing results_dir/<folder>/metrics.json."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_names = list_checkpoint_folders(checkpoints_dir)
    mirror_results_folders(results_dir, folder_names)
    if folder_filter is not None:
        folder_names = [name for name in folder_names if name in folder_filter]

    for folder_name in folder_names:
        checkpoint_path = os.path.join(checkpoints_dir, folder_name, "attack_result.pt")
        try:
            metrics = evaluate_checkpoint(
                checkpoint_path, device, raw_data_dir, batch_size
            )
            save_metrics(
                os.path.join(results_dir, folder_name, "metrics.json"), metrics
            )
            print(f"{folder_name}: {metrics}")
        except Exception as error:
            print(f"FAILED {folder_name}: {error}")
