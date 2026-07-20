"""Atomic, reusable evaluation: one model, one dataset, clean or under attack.

Two layers, thinnest to widest:
  evaluate_benign / evaluate_attack   model in, metrics out, no filesystem at
                                       all, so a training script can call
                                       these directly on the model it just
                                       trained, no save-then-reload needed.
  evaluate_checkpoint                 a checkpoint path in, metrics out. Loads
                                       the model and its args.json sidecar,
                                       then delegates to one of the two above.

There is deliberately no third, directory-walking layer here. Looping over
the whole checkpoints/ tree is orchestration at a higher altitude than "one
checkpoint path in, one metrics dict out," so it lives in metrics.py's main(),
not in this module. Its absence is by design, not an oversight.

Every directory this module touches is a parameter with a plain default, not
a bare module-level constant read inside a function body, so any of these
are safe to call against a different tree (a test fixture, a scratch export)
just by passing a different argument.
"""

import json
import os

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
from stealth import cached_stealth_metrics
from utils.config import DATASET_REGISTRY


def evaluate_benign(
    model,
    dataset_name: str,
    device,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict:
    """Clean accuracy, total and per class, for a model with no attack of its own."""
    loader = build_clean_loader(
        dataset_name, raw_data_dir, batch_size, max_samples=max_samples, seed=seed
    )
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
    max_samples: int | None = None,
    seed: int = 0,
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

    clean_loader = build_clean_loader(
        dataset_name, raw_data_dir, batch_size, max_samples=max_samples, seed=seed
    )
    poisoned_loader = build_poisoned_loader(
        dataset_name,
        attack,
        raw_data_dir,
        batch_size,
        max_samples=max_samples,
        seed=seed,
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
    folder_name = os.path.basename(os.path.dirname(checkpoint_path))

    if args["attack"] == "benign":
        metrics = evaluate_benign(
            model, args["dataset"], device, raw_data_dir, batch_size
        )
        return {
            "folder_name": folder_name,
            "architecture": args["architecture"],
            "dataset": args["dataset"],
            **metrics,
        }

    config = default_config(args["attack"])
    attack = build_attack(
        args["attack"],
        config,
        DATASET_REGISTRY[args["dataset"]].image_size,
        args["target_label"],
    )
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
    stealth = cached_stealth_metrics(
        args["dataset"], args["attack"], attack, raw_data_dir, device
    )
    return {
        "folder_name": folder_name,
        "architecture": args["architecture"],
        "dataset": args["dataset"],
        "attack": args["attack"],
        "label_mode": args["label_mode"],
        "poison_rate": args["poison_rate"],
        "target_label": args["target_label"],
        **metrics,
        "stealth": stealth,
    }


def save_metrics(output_path: str, metrics: dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
