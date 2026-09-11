"""Recover every field that is deducible from artifacts already on disk.

Nothing here runs a model. 3 quantities were either never recorded or were
recorded in a way that hides a cap, and all 3 can be recovered from what the
sweep and the training runs already wrote.

  realized_poison_rate  choose_poison_indices caps the poisoned count at the
                        eligible pool, and a clean-label attack is eligible only
                        on the target class, so a requested 10% can resolve to
                        0.5%. The selection is seeded and deterministic, so
                        replaying it recovers the rate that was actually applied.

  asr, clean_accuracy   the no-dropout baseline already stores the model's argmax
                        and the label the loader asked for, per split. On the
                        backdoor split that label is the attack-success label, so
                        the agreement rate is the attack success rate. On the
                        clean split it is the true class, so agreement is clean
                        accuracy. Checked against every checkpoint that carries a
                        training-time measurement, and the 2 agree closely.

A deduced value never overwrites a measured value. Where both exist the measured
value stays and the deduced value is written beside it, so a disagreement stays
visible rather than being silently reconciled.

Example
    python -m cli.backfill --dry-run
"""

import argparse
import glob
import json
import os

import torchvision.transforms.v2 as transforms_v2

from attacks import build_attack, default_config
from defences.cache import baseline_path, load_baseline
from data.registry import DATASET_REGISTRY
from data.loading import extract_labels, load_clean_datasets
from attacks.poisoning import choose_indices_with_cover, choose_poison_indices

# Class counts per dataset are fixed, so the training label vector can be rebuilt
# without touching the images. CIFAR and Tiny are balanced by construction. GTSRB
# is not, so it is read from disk instead of assumed.
BALANCED_TRAIN_COUNTS = {
    "cifar10": (10, 5000),
    "cifar100": (100, 500),
    "tiny": (200, 500),
}

MAX_CAPPED_SHOWN = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def training_labels(dataset: str, raw_data_dir: str) -> list[int]:
    """The training label vector, rebuilt without decoding any image where possible."""
    if dataset in BALANCED_TRAIN_COUNTS:
        num_classes, per_class = BALANCED_TRAIN_COUNTS[dataset]
        labels = [class_id for class_id in range(num_classes) for _ in range(per_class)]
        return labels

    transform = transforms_v2.Compose([transforms_v2.ToTensor()])
    train, _ = load_clean_datasets(dataset, transform, raw_data_dir)

    labels = extract_labels(train)
    return labels


def deduce_realized_poison_rate(
    metadata: dict, label_cache: dict, raw_data_dir: str
) -> dict | None:
    """Replay the seeded poison selection and report the fraction actually poisoned.

    Returns None when the selection cannot be replayed from folder metadata alone:
    a recovered orphan with no attack recorded, or a truncated run, whose
    eligible pool is not the full training set.
    """
    # A checkpoint recovered from an orphaned file may record no attack at all,
    # and default_config(None) raises, which would abort the run over 1 folder.
    if not metadata.get("attack") or not metadata.get("dataset"):
        return None
    if metadata["attack"] == "benign":
        return {
            "realized_poison_rate": 0.0,
            "requested_count": 0,
            "realized_count": 0,
            "capped": False,
            "cover_count": 0,
        }
    if metadata.get("max_samples"):
        return None

    dataset = metadata["dataset"]
    if dataset not in label_cache:
        label_cache[dataset] = training_labels(dataset, raw_data_dir)
    labels = label_cache[dataset]

    config = default_config(metadata["attack"])
    attack = build_attack(
        metadata["attack"],
        config,
        DATASET_REGISTRY[dataset].image_size,
        metadata["target_label"],
    )

    # A cover-sample attack selects through a different function, and skipping
    # it is how TaCT's cap stayed invisible: source_classes defaults to a single
    # class, so its eligible pool is as small as clean-label's. The chosen count
    # depends only on the pool size and the requested rate, never on the seed.
    cover_rate = metadata.get("cover_rate") or 0.0
    source_classes = getattr(config, "source_classes", None)
    # Runs predating the seeding commit record seed null, but their poison draw
    # still went through default_rng(0): the selection always took an explicit
    # seed and the entrypoint defaulted it to 0. Passing null through would seed
    # from OS entropy and hand back a different index set every call.
    selection_seed = metadata["seed"] or 0
    cover: set = set()
    if cover_rate > 0.0 or source_classes is not None:
        chosen, cover = choose_indices_with_cover(
            labels,
            attack,
            metadata["poison_rate"],
            cover_rate,
            source_classes,
            selection_seed,
        )
    else:
        chosen = choose_poison_indices(
            labels, attack, metadata["poison_rate"], selection_seed
        )

    # The count the rate asked for, before the eligible-pool cap. Comparing counts
    # rather than rates separates a real cap from the integer rounding that every
    # dataset whose size does not divide the rate exactly would otherwise trip.
    requested_count = round(metadata["poison_rate"] * len(labels))
    selection = {
        "realized_poison_rate": len(chosen) / len(labels),
        "requested_count": requested_count,
        "realized_count": len(chosen),
        "capped": len(chosen) < requested_count,
        "cover_count": len(cover),
    }
    return selection


def deduce_rates_from_cache(results_dir: str, folder: str) -> dict | None:
    """ASR and clean accuracy from the cached no-dropout baseline, or None."""
    psbd_dir = os.path.join(results_dir, folder, "psbd")

    deduced = {}
    for split, name in (("backdoor", "asr"), ("clean", "clean_accuracy")):
        path = baseline_path(psbd_dir, split)
        if not os.path.exists(path):
            return None
        _, predicted, requested = load_baseline(path)
        if requested.numel() == 0:
            return None
        deduced[name] = float((predicted.long() == requested.long()).float().mean())
        deduced[f"n_{split}_scored"] = int(predicted.numel())

    return deduced


def apply_selection(metadata: dict, selection: dict) -> None:
    """Record the replayed poison selection into the checkpoint's metadata.

    A deduced value never overwrites a measured one: n_cover is filled only when
    the training run did not record it, and label_mode only when it is null,
    which 60 older folders carry.
    """
    metadata["realized_poison_rate"] = selection["realized_poison_rate"]
    metadata["poison_rate_capped"] = selection["capped"]
    metadata["n_poisoned"] = selection["realized_count"]
    if "n_cover" not in metadata and selection["cover_count"]:
        metadata["n_cover"] = selection["cover_count"]
    if metadata.get("label_mode") is None and metadata["attack"] != "benign":
        # benign has no attack config and "generated" has no default config.
        try:
            metadata["label_mode"] = default_config(metadata["attack"]).label_mode
        except (KeyError, ValueError):
            pass


def apply_deduced_rates(metadata: dict, deduced: dict) -> None:
    """Record the cache-deduced rates, without overwriting a measured value.

    Both are kept so a disagreement between the training-time measurement and the
    cache stays visible.
    """
    for name in ("asr", "clean_accuracy"):
        metadata[f"{name}_from_cache"] = deduced[name]
        if metadata.get(name) is None:
            metadata[name] = deduced[name]
            metadata[f"{name}_source"] = "psbd_baseline_cache"

    metadata["n_backdoor_scored"] = deduced["n_backdoor_scored"]
    metadata["n_clean_scored"] = deduced["n_clean_scored"]


def print_capped(capped: list[tuple]) -> None:
    """The checkpoints whose realized poison rate fell below the one requested."""
    print("\ncapped checkpoints (requested rate, then realized rate, then counts):")
    for folder, requested, realized, want, got in capped[:MAX_CAPPED_SHOWN]:
        print(
            f"  {folder:<44} {requested:.3f} to {realized:.4f}  "
            f"({got} of {want} requested)"
        )
    if len(capped) > MAX_CAPPED_SHOWN:
        print(f"  ... and {len(capped) - MAX_CAPPED_SHOWN} more")


def main() -> None:
    args = parse_args()
    label_cache: dict[str, list[int]] = {}
    counts = {"rate_written": 0, "rate_capped": 0, "cache_rates": 0, "files": 0}
    capped = []

    for args_path in sorted(
        glob.glob(os.path.join(args.checkpoints_dir, "*/args.json"))
    ):
        folder = os.path.basename(os.path.dirname(args_path))
        with open(args_path) as handle:
            metadata = json.load(handle)

        selection = deduce_realized_poison_rate(
            metadata, label_cache, args.raw_data_dir
        )
        if selection is not None:
            apply_selection(metadata, selection)
            counts["rate_written"] += 1
            if selection["capped"]:
                counts["rate_capped"] += 1
                capped.append(
                    (
                        folder,
                        metadata["poison_rate"],
                        selection["realized_poison_rate"],
                        selection["requested_count"],
                        selection["realized_count"],
                    )
                )

        deduced = deduce_rates_from_cache(args.results_dir, folder)
        if deduced is not None:
            counts["cache_rates"] += 1
            apply_deduced_rates(metadata, deduced)

        if not args.dry_run:
            with open(args_path, "w") as handle:
                json.dump(metadata, handle, indent=2)
        counts["files"] += 1

    print(f"args.json files processed:        {counts['files']}")
    print(f"realized_poison_rate written:     {counts['rate_written']}")
    print(f"  of which CAPPED below request:  {counts['rate_capped']}")
    print(f"asr/clean_accuracy from cache:    {counts['cache_rates']}")

    if capped:
        print_capped(capped)


if __name__ == "__main__":
    main()
