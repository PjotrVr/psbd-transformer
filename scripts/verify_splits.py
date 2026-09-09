"""Does each cell's cached PSBD split still match the split the current code builds?

load_or_build_baseline reuses any cached baseline whose row count matches what it is asked
for, and a row count always matches after a retrain, so a split whose DEFINITION changed
leaves behind a cache that loads cleanly and means something else entirely.

This is not hypothetical. AttackSuccessSet restricts a source-specific attack to its source
classes, because a source-specific attack only claims to flip those. Every tact cache
predated that fix and held every non-target class instead, which divides the true attack
success rate by roughly the class count: 7959 cached rows against 42 correct ones on Tiny,
a factor of 190. The mtime check in the coverage ledger cannot see this, because the files
are recent; only rebuilding the split and comparing counts can.

Slow by nature, since it constructs every cell's loaders. Run it after any change to
poisoning, eval-set construction or dataset splits, and before trusting a table.

    PYTHONPATH=. python scripts/verify_splits.py
"""

import argparse
import json
import os

import torch

from defences.checkpoint_eval import PSBD_SPLIT_SEED, build_psbd_loaders_from_checkpoint
from defences.psbd_cache import baseline_path

SPLITS = ("validation", "clean", "backdoor")


def cached_row_counts(psbd_dir: str) -> dict:
    counts = {}
    for split in SPLITS:
        path = baseline_path(psbd_dir, split)
        if os.path.exists(path):
            blob = torch.load(path, map_location="cpu", weights_only=False)
            counts[split] = int(blob["probs"].shape[0])
    return counts


def built_row_counts(checkpoint_path: str, raw_data_dir: str) -> dict:
    loaders, _ = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=raw_data_dir,
        batch_size=64,
        num_workers=0,
    )
    return {name: len(loader.dataset) for name, loader in loaders.items()}


def verify_one(folder: str, args) -> dict:
    psbd_dir = os.path.join(args.results_dir, folder, "psbd")
    cached = cached_row_counts(psbd_dir)
    if not cached:
        return {"folder_name": folder, "status": "no_baseline"}
    built = built_row_counts(
        os.path.join(args.checkpoints_dir, folder, "attack_result.pt"),
        args.raw_data_dir,
    )
    mismatch = {
        split: (cached[split], built[split])
        for split in cached
        if split in built and cached[split] != built[split]
    }
    return {
        "folder_name": folder,
        "status": "STALE" if mismatch else "ok",
        "cached": cached,
        "built": built,
        "mismatch": mismatch,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", default="results/coverage/coverage.json")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--out", default="results/coverage/split_integrity.json")
    parser.add_argument("--checkpoint-folder", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.coverage) as handle:
        cells = [cell["folder_name"] for cell in json.load(handle)["cells"]]
    folders = args.checkpoint_folder or cells

    reports = {}
    for index, folder in enumerate(folders, start=1):
        try:
            report = verify_one(folder, args)
        except Exception as error:  # noqa: BLE001
            report = {
                "folder_name": folder,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        reports[folder] = report
        suffix = f"  {report.get('mismatch')}" if report["status"] == "STALE" else ""
        print(
            f"[{index:3d}/{len(folders)}] {folder:38s} {report['status']}{suffix}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(reports, handle, indent=2)

    stale = [name for name, row in reports.items() if row["status"] == "STALE"]
    print(
        f"\nSTALE {len(stale)}   "
        f"ok {sum(1 for r in reports.values() if r['status'] == 'ok')}   "
        f"errors {sum(1 for r in reports.values() if r['status'] == 'error')}"
    )
    for name in stale:
        print("   ", name)


if __name__ == "__main__":
    main()
