"""Recover every field that is deducible from artifacts already on disk.

`cli.backfill` is the maintained copy of this tool. This file stays only for the
cover-sample selection path (TaCT, choose_indices_with_cover) that it alone carries.

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
                        the agreement rate IS the attack success rate. On the
                        clean split it is the true class, so agreement is clean
                        accuracy. Verified against 496 checkpoints that carry a
                        training-time measurement: all 496 agree within 0.02.

A deduced value never overwrites a measured value. Where both exist the measured
value stays and the deduced value is written beside it, so a disagreement stays
visible rather than being silently reconciled.
"""

import argparse
import glob
import json
import os

from attacks import build_attack, default_config
from defences.cache import baseline_path, load_baseline
from attacks.poisoning import choose_indices_with_cover, choose_poison_indices
from data.registry import DATASET_REGISTRY

# Class counts per dataset are fixed, so the training label vector can be rebuilt
# without touching the images. CIFAR and Tiny are balanced by construction. GTSRB
# is not, so it is read from disk instead of assumed.
BALANCED_TRAIN_COUNTS = {
    "cifar10": (10, 5000),
    "cifar100": (100, 500),
    "tiny": (200, 500),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def training_labels(dataset, raw_data_dir):
    """The training label vector, rebuilt without decoding any image where possible."""
    if dataset in BALANCED_TRAIN_COUNTS:
        num_classes, per_class = BALANCED_TRAIN_COUNTS[dataset]
        labels = [c for c in range(num_classes) for _ in range(per_class)]
        return labels

    import torchvision.transforms.v2 as transforms_v2

    from data.loading import extract_labels, load_clean_datasets

    transform = transforms_v2.Compose([transforms_v2.ToTensor()])
    train, _ = load_clean_datasets(dataset, transform, raw_data_dir)
    labels = extract_labels(train)
    return labels


def deduce_realized_poison_rate(metadata, label_cache, raw_data_dir):
    """Replay the seeded poison selection and report the fraction actually poisoned."""
    # A checkpoint recovered from an orphaned file may have no attack recorded at
    # all, because its provenance was reconstructed from a job script that no
    # longer exists. Nothing here can be replayed for it and default_config(None)
    # raises, which would abort the whole run rather than skipping 1 folder.
    if not metadata.get("attack") or not metadata.get("dataset"):
        return None
    if metadata["attack"] == "benign":
        return {
            "realized_poison_rate": 0.0,
            "requested_count": 0,
            "realized_count": 0,
            "capped": False,
        }
    # A truncated run changes the eligible pool itself, so its realized rate is not
    # recoverable from folder metadata alone.
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

    # A cover-sample attack selects through a different function, and skipping it
    # was how TaCT's cap stayed invisible: source_classes defaults to a single
    # class, so its eligible pool is as small as clean-label's. The chosen COUNT
    # depends only on the pool size and the requested rate, not on the seed, so a
    # checkpoint recording seed null still yields a trustworthy realized rate.
    cover_rate = metadata.get("cover_rate") or 0.0
    source_classes = getattr(config, "source_classes", None)
    # Runs predating the seeding commit record seed null, but their poison draw
    # still went through default_rng(0): the selection functions always took an
    # explicit seed and the entrypoint defaulted it to 0. Passing null through
    # would seed from OS entropy and hand back a different index set every call.
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
    return {
        "realized_poison_rate": len(chosen) / len(labels),
        "requested_count": requested_count,
        "realized_count": len(chosen),
        "capped": len(chosen) < requested_count,
        # The cover set was already drawn from the same generator and thrown away.
        # It is written by train_backdoor only on recent runs, so recovering it here
        # costs nothing and fills it for every cover-sample checkpoint on disk.
        "cover_count": len(cover),
    }


def deduce_rates_from_cache(results_dir, folder):
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


def main():
    args = parse_args()
    label_cache = {}
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
            metadata["realized_poison_rate"] = selection["realized_poison_rate"]
            metadata["poison_rate_capped"] = selection["capped"]
            metadata["n_poisoned"] = selection["realized_count"]
            # Only recent runs record n_cover, and a deduced value never overwrites
            # a measured value, so this fills the gap without touching what was
            # actually observed.
            if "n_cover" not in metadata and selection.get("cover_count"):
                metadata["n_cover"] = selection["cover_count"]
            # label_mode follows from the attack, and 60 folders carry it as null.
            # benign has no attack config, and "generated" has no default config.
            if metadata.get("label_mode") is None and metadata["attack"] != "benign":
                try:
                    metadata["label_mode"] = default_config(
                        metadata["attack"]
                    ).label_mode
                except (KeyError, ValueError):
                    pass
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
            # Measured beats deduced. Both are kept so a disagreement is visible.
            for name in ("asr", "clean_accuracy"):
                metadata[f"{name}_from_cache"] = deduced[name]
                if metadata.get(name) is None:
                    metadata[name] = deduced[name]
                    metadata[f"{name}_source"] = "psbd_baseline_cache"
            metadata["n_backdoor_scored"] = deduced["n_backdoor_scored"]
            metadata["n_clean_scored"] = deduced["n_clean_scored"]

        if not args.dry_run:
            with open(args_path, "w") as handle:
                json.dump(metadata, handle, indent=2)
        counts["files"] += 1

    print(f"args.json files processed:        {counts['files']}")
    print(f"realized_poison_rate written:     {counts['rate_written']}")
    print(f"  of which CAPPED below request:  {counts['rate_capped']}")
    print(f"asr/clean_accuracy from cache:    {counts['cache_rates']}")

    if capped:
        print("\ncapped checkpoints (requested rate -> realized rate, counts):")
        for folder, requested, realized, want, got in capped[:40]:
            print(
                f"  {folder:<44} {requested:.3f} -> {realized:.4f}  "
                f"({got} of {want} requested)"
            )
        if len(capped) > 40:
            print(f"  ... and {len(capped) - 40} more")


if __name__ == "__main__":
    main()
