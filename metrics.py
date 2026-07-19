"""Baseline attack-success and clean-accuracy metrics for every checkpoint.

Decoupled from the PSBD dropout sweep (archived, pending a rewrite), which
will write its own psbd_metrics.json into the same analysis/<folder>/
directories this script creates. This script only answers: for a benign
model, what is clean accuracy overall and per class; for an attacked model,
what is the attack success rate and the clean accuracy on unpoisoned
counterparts.

Scoped to checkpoints/ only. Every checkpoints/ folder carries an args.json
(normalize_checkpoints.py) recording exactly which attack produced it, so its
poisoned eval set is rebuilt in memory from our own attack code, the same way
checkpoint_eval.py does for the PSBD sweep. analysis/ mirrors checkpoints/
exactly, one folder per checkpoint and nothing else.
"""

import argparse
import json
import os
import shutil

import torch
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader

from attacks import build_attack, default_config
from defences.checkpoint_eval import working_resolution
from defences.detection import (
    accuracy_by_class_from_counts,
    attack_success_rate,
    class_correct_and_total,
    clean_accuracy,
    pooled_accuracy_from_counts,
)
from utils.config import DATASET_REGISTRY
from utils.datasets import extract_labels, load_clean_datasets
from models import load_checkpoint
from poison import AttackSuccessSet, PoisonedTrainingSet
from train_benign import build_clean_loaders

CHECKPOINTS_DIR = "checkpoints"
ANALYSIS_DIR = "analysis"


def list_checkpoint_folders(source_dir: str) -> list[str]:
    if not os.path.isdir(source_dir):
        return []
    return sorted(
        name for name in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, name))
    )


def mirror_analysis_folders(analysis_dir: str, folder_names: list[str]) -> None:
    """analysis/ matches checkpoints/ exactly, one folder per checkpoint.

    Creates a folder for every current checkpoint, even ones not yet
    processed, and removes any analysis/<name>/ left over from a checkpoint
    that no longer exists, for example one normalize_checkpoints.py renamed.
    """
    os.makedirs(analysis_dir, exist_ok=True)
    wanted = set(folder_names)
    existing = {
        name for name in os.listdir(analysis_dir)
        if os.path.isdir(os.path.join(analysis_dir, name))
    }
    for orphan in existing - wanted:
        shutil.rmtree(os.path.join(analysis_dir, orphan))
    for folder_name in folder_names:
        os.makedirs(os.path.join(analysis_dir, folder_name), exist_ok=True)


def write_metrics(analysis_dir: str, folder_name: str, metrics: dict) -> None:
    folder = os.path.join(analysis_dir, folder_name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)


def read_args_json(checkpoint_dir: str) -> dict:
    with open(os.path.join(checkpoint_dir, "args.json")) as handle:
        return json.load(handle)


def build_full_eval_loaders(
    dataset_name: str, attack, raw_data_dir: str, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """Clean and attack-success loaders over the whole test set, not a sweep-sized subset.

    checkpoint_eval.py's PSBD-sweep loaders deliberately hold out a small
    validation split and balance classes down to examples_per_class, both
    sweep-specific concerns this baseline metric has no need for.
    """
    spec = DATASET_REGISTRY[dataset_name]
    image_size = working_resolution(dataset_name)
    # Stop at 0-to-1 pixel values so the trigger applies correctly, matching
    # checkpoint_eval.py's in-memory eval path. Normalization happens last,
    # inside PoisonedTrainingSet/AttackSuccessSet, same as at training time.
    base_transform = transforms_v2.Compose(
        [transforms_v2.Resize((image_size, image_size)), transforms_v2.ToTensor()]
    )
    _, test_base = load_clean_datasets(dataset_name, base_transform, raw_data_dir)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    clean_eval = PoisonedTrainingSet(test_base, attack, set(), normalize, spec.num_classes)
    true_labels = extract_labels(test_base)
    backdoor_eval = AttackSuccessSet(test_base, true_labels, attack, normalize, spec.num_classes)

    loader = lambda dataset: DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader(clean_eval), loader(backdoor_eval)


def evaluate_benign(model, dataset_name: str, architecture: str, device, raw_data_dir: str, batch_size: int) -> dict:
    _, test_loader, num_classes = build_clean_loaders(dataset_name, raw_data_dir, batch_size, num_workers=2)
    # One forward pass shared between the pooled and per-class figures, not two.
    correct, total = class_correct_and_total(model, test_loader, device, num_classes, use_bfloat16=True)
    return {
        "architecture": architecture,
        "clean_accuracy": pooled_accuracy_from_counts(correct, total),
        "clean_accuracy_by_class": accuracy_by_class_from_counts(correct, total),
    }


def evaluate_attack_from_args(model, args: dict, device, raw_data_dir: str, batch_size: int) -> dict:
    dataset_name = args["dataset"]
    image_size = working_resolution(dataset_name)
    attack = build_attack(
        args["attack"], default_config(args["attack"]), image_size, args["target_label"]
    )
    clean_loader, backdoor_loader = build_full_eval_loaders(dataset_name, attack, raw_data_dir, batch_size)
    return {
        "architecture": args["architecture"],
        "dataset": dataset_name,
        "attack": args["attack"],
        "label_mode": args["label_mode"],
        "poison_rate": args["poison_rate"],
        "target_label": args["target_label"],
        "asr": attack_success_rate(model, backdoor_loader, device, use_bfloat16=True),
        "clean_accuracy": clean_accuracy(model, clean_loader, device, use_bfloat16=True),
    }


def process_checkpoints_folder(folder_name: str, device, raw_data_dir: str, batch_size: int) -> dict:
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, folder_name)
    checkpoint_path = os.path.join(checkpoint_dir, "attack_result.pt")
    args = read_args_json(checkpoint_dir)
    model = load_checkpoint(args["architecture"], checkpoint_path, device)

    if args["attack"] == "benign":
        return evaluate_benign(model, args["dataset"], args["architecture"], device, raw_data_dir, batch_size)
    return evaluate_attack_from_args(model, args, device, raw_data_dir, batch_size)


def run(folder_filter: list[str] | None, raw_data_dir: str, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_folders = list_checkpoint_folders(CHECKPOINTS_DIR)
    mirror_analysis_folders(ANALYSIS_DIR, local_folders)

    if folder_filter is not None:
        local_folders = [name for name in local_folders if name in folder_filter]

    for folder_name in local_folders:
        try:
            metrics = process_checkpoints_folder(folder_name, device, raw_data_dir, batch_size)
            write_metrics(ANALYSIS_DIR, folder_name, metrics)
            print(f"{folder_name}: {metrics}")
        except Exception as error:
            print(f"FAILED {folder_name}: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline attack/benign metrics for every checkpoint")
    parser.add_argument("--folder", nargs="+", default=None, help="only process these folder names")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.folder, args.raw_data_dir, args.batch_size)


if __name__ == "__main__":
    main()
