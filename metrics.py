"""Baseline attack-success and clean-accuracy metrics for every checkpoint.

Decoupled from the PSBD dropout sweep (archived, pending a rewrite), which
will write its own psbd_metrics.json into the same analysis/<folder>/
directories this script creates. This script only answers: for a benign
model, what is clean accuracy overall and per class; for an attacked model,
what is the attack success rate and the clean accuracy on unpoisoned
counterparts.

checkpoints/ folders carry an args.json (normalize_checkpoints.py) recording
exactly which attack produced them, so their poisoned eval set is rebuilt in
memory from our own attack code, the same way checkpoint_eval.py does for the
PSBD sweep. backdoor_bench_checkpoints/ folders are BackdoorBench's own
downloads with no args.json and no local attack object to rebuild from, so
their eval set comes from the PNG triggers BackdoorBench shipped alongside
them, the same way backdoor_data.py's PNG path always has.
"""

import argparse
import json
import os
import shutil

import torch
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader

from attacks import build_attack, default_config
from backdoor_data import load_backdoor_splits
from checkpoint_eval import working_resolution
from config import DATASET_REGISTRY, dataset_name_from_folder, label_mode_from_folder
from datasets import build_transform, extract_labels, load_clean_datasets
from detection import attack_success_rate, clean_accuracy, clean_accuracy_by_class
from models import detect_architecture, load_checkpoint
from poison import AttackSuccessSet, PoisonedTrainingSet
from train_benign import build_clean_loaders

CHECKPOINTS_DIR = "checkpoints"
BACKDOOR_BENCH_DIR = "backdoor_bench_checkpoints"
ANALYSIS_DIR = "analysis"
TARGET_LABEL = 0  # BackdoorBench's own default, matches this project's default too


def list_checkpoint_folders(source_dir: str) -> list[str]:
    if not os.path.isdir(source_dir):
        return []
    return sorted(
        name for name in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, name))
    )


def mirror_analysis_folders(analysis_dir: str, folder_names: list[str]) -> None:
    """analysis/ matches checkpoints/ + backdoor_bench_checkpoints/ exactly.

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


def attack_name_from_backdoor_bench_folder(folder_name: str, dataset_name: str) -> str:
    """BackdoorBench folder names are dataset_attack_rate, for example cifar10_sig_0_01.

    The attack token is whatever sits between the dataset prefix and the
    trailing "0_XX" poison-rate tag, which covers attacks outside this
    project's own attacks.py registry (blind, inputaware, lira, ssba, ...).
    """
    remainder = folder_name[len(dataset_name) + 1:]
    tokens = remainder.split("_")
    if len(tokens) >= 2 and tokens[-2] == "0":
        tokens = tokens[:-2]
    return "_".join(tokens)


def evaluate_benign(model, dataset_name: str, architecture: str, device, raw_data_dir: str, batch_size: int) -> dict:
    _, test_loader, num_classes = build_clean_loaders(dataset_name, raw_data_dir, batch_size, num_workers=2)
    return {
        "architecture": architecture,
        "clean_accuracy": clean_accuracy(model, test_loader, device, use_bfloat16=True),
        "clean_accuracy_by_class": clean_accuracy_by_class(
            model, test_loader, device, num_classes, use_bfloat16=True
        ),
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


def process_backdoor_bench_folder(
    folder_name: str, device, weights_dir: str, raw_data_dir: str, batch_size: int
) -> dict:
    checkpoint_path = os.path.join(weights_dir, folder_name, "attack_result.pt")
    architecture = detect_architecture(checkpoint_path)
    model = load_checkpoint(architecture, checkpoint_path, device)

    dataset_name = dataset_name_from_folder(folder_name)
    label_mode = label_mode_from_folder(folder_name)
    num_classes = DATASET_REGISTRY[dataset_name].num_classes
    transform = build_transform(dataset_name)

    _, clean_test = load_clean_datasets(dataset_name, transform, raw_data_dir)
    backdoor_test, _ = load_backdoor_splits(
        folder_name, clean_test, transform, weights_dir, label_mode, TARGET_LABEL, num_classes
    )

    clean_loader = DataLoader(clean_test, batch_size=batch_size, shuffle=False)
    backdoor_loader = DataLoader(backdoor_test, batch_size=batch_size, shuffle=False)

    return {
        "architecture": architecture,
        "dataset": dataset_name,
        "attack": attack_name_from_backdoor_bench_folder(folder_name, dataset_name),
        "label_mode": label_mode,
        "target_label": TARGET_LABEL,
        "asr": attack_success_rate(model, backdoor_loader, device, use_bfloat16=True),
        "clean_accuracy": clean_accuracy(model, clean_loader, device, use_bfloat16=True),
    }


def run(folder_filter: list[str] | None, raw_data_dir: str, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_folders = list_checkpoint_folders(CHECKPOINTS_DIR)
    bench_folders = list_checkpoint_folders(BACKDOOR_BENCH_DIR)
    mirror_analysis_folders(ANALYSIS_DIR, local_folders + bench_folders)

    if folder_filter is not None:
        local_folders = [name for name in local_folders if name in folder_filter]
        bench_folders = [name for name in bench_folders if name in folder_filter]

    for folder_name in local_folders:
        try:
            metrics = process_checkpoints_folder(folder_name, device, raw_data_dir, batch_size)
            write_metrics(ANALYSIS_DIR, folder_name, metrics)
            print(f"{folder_name}: {metrics}")
        except Exception as error:
            print(f"FAILED {folder_name}: {error}")

    for folder_name in bench_folders:
        try:
            metrics = process_backdoor_bench_folder(
                folder_name, device, BACKDOOR_BENCH_DIR, raw_data_dir, batch_size
            )
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
