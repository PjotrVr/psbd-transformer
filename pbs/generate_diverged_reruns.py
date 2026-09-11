"""Rerun the GTSRB training runs that collapsed, under a cosine schedule with clipping.

13 GTSRB runs on the constant learning rate fell from 0.99 validation accuracy
to a single class in their last epochs and saved with a plausible attack
success rate (docs/runs/2026-09-11-diverged-gtsrb-runs.md). The trainer now
refuses to save such a run, and this generator reruns each collapsed folder
with --lr-schedule cosine --clip-grad-norm 1.0 into <folder>_cos, so the recipe
change is visible in the name and the original stays on disk for comparison.
Every other argument is read back from the folder's own args.json.

    PYTHONPATH=. python pbs/generate_diverged_reruns.py --dry-run
    PYTHONPATH=. python pbs/generate_diverged_reruns.py
"""

import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pbs.generate_seed_jobs import MEDIAN_MINUTES, TEMPLATE, TRAIN, pack  # noqa: E402

COLLAPSE_FRACTION = 0.5
EXCLUDED_TOKENS = ("sam_rho", "_trig", "_pilot", "a2a", "_cos")
RERUN_FLAGS = " \\\n    --lr-schedule cosine \\\n    --clip-grad-norm 1.0"


def benign_accuracy(checkpoints_dir: str, dataset: str) -> float | None:
    path = os.path.join(checkpoints_dir, f"vit_{dataset}_benign", "args.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        accuracy = json.load(handle).get("clean_accuracy")
    return accuracy


def collapsed_runs(checkpoints_dir: str) -> list[dict]:
    """Every plain Adam ViT run whose clean accuracy is under half its benign reference."""
    runs = []
    for path in sorted(glob.glob(os.path.join(checkpoints_dir, "vit_*", "args.json"))):
        folder = os.path.basename(os.path.dirname(path))
        if any(token in folder for token in EXCLUDED_TOKENS):
            continue
        with open(path) as handle:
            metadata = json.load(handle)
        reference = benign_accuracy(checkpoints_dir, metadata.get("dataset", ""))
        accuracy = metadata.get("clean_accuracy")
        if (
            reference is None
            or accuracy is None
            or accuracy >= COLLAPSE_FRACTION * reference
        ):
            continue
        if os.path.exists(os.path.join(checkpoints_dir, f"{folder}_cos", "args.json")):
            continue
        metadata["folder"] = folder
        runs.append(metadata)
    return runs


def rerun_command(metadata: dict) -> str:
    """The original training command with the schedule and the clip added, into <folder>_cos."""
    command = TRAIN.format(
        dataset=metadata["dataset"],
        attack=metadata["attack"],
        poison_rate=metadata["poison_rate"],
        target_label=metadata["target_label"],
        architecture=metadata["architecture"],
        epochs=metadata["epochs"],
        seed=metadata["seed"] or 0,
        folder=f"{metadata['folder']}_cos",
    )
    with_flags = command.replace(
        f"    --output checkpoints/{metadata['folder']}_cos/attack_result.pt",
        f"    --output checkpoints/{metadata['folder']}_cos/attack_result.pt{RERUN_FLAGS}",
    )
    return with_flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", default=os.path.join(REPO, "checkpoints"))
    parser.add_argument(
        "--out-dir", default=os.path.join(REPO, "pbs", "vit_diverged_reruns")
    )
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs = [
        (metadata, metadata["seed"] or 0)
        for metadata in collapsed_runs(args.checkpoints_dir)
    ]
    jobs = pack(runs, args.hours * 60 * 0.9)
    minutes = sum(
        MEDIAN_MINUTES.get((m["architecture"], m["dataset"]), 300) for m, _ in runs
    )
    print(
        f"{len(runs)} collapsed runs to rerun, {minutes / 60:.1f} GPU hours, {len(jobs)} jobs"
    )
    for metadata, _ in runs:
        print(f"  {metadata['folder']} -> {metadata['folder']}_cos")
    if args.dry_run:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(REPO, "logs", "psbd_seed"), exist_ok=True)
    for index, job in enumerate(jobs, start=1):
        commands = "\n".join(rerun_command(metadata) for metadata, _ in job)
        path = os.path.join(args.out_dir, f"rerun_{index:03d}.pbs")
        with open(path, "w") as handle:
            handle.write(
                TEMPLATE.format(
                    base=REPO,
                    index=100 + index,
                    walltime=f"{int(args.hours):02d}:00:00",
                    commands=commands,
                )
            )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
