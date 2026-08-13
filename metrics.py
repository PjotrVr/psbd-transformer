"""CLI: baseline attack-success, clean-accuracy, and stealth metrics per checkpoint.

evaluate.py stays atomic (one checkpoint path in, one metrics dict out) on
purpose. Looping over the whole checkpoints/ directory is this file's job, not
evaluate.py's. Each checkpoint's metrics.json is written into its own
checkpoints/<folder>/ directory, next to attack_result.pt and args.json, so a
checkpoint folder stays a complete, self-contained unit.

Scope is checkpoints/ only. backdoor_bench_checkpoints/ (BackdoorBench's
downloaded reference data, evaluated through the PNG path elsewhere) is out of
this file's scope entirely.

Example
    python metrics.py --folder vit_cifar10_benign vit_cifar10_badnet_a2o_0_1
"""

import argparse
import os
import time
from datetime import datetime, timezone

import torch

from evaluate import evaluate_checkpoint, read_args_json, save_metrics


def list_checkpoint_folders(checkpoints_dir: str) -> list[str]:
    if not os.path.isdir(checkpoints_dir):
        return []
    return sorted(
        name
        for name in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, name))
    )


def _attack_label(checkpoint_dir: str) -> str:
    """The attack name from args.json, or "benign", for the progress trace."""
    try:
        return read_args_json(checkpoint_dir)["attack"]
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline attack/benign/stealth metrics for every checkpoint"
    )
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument(
        "--folder", nargs="+", default=None, help="only process these folder names"
    )
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_names = list_checkpoint_folders(args.checkpoints_dir)
    if args.folder is not None:
        folder_names = [name for name in folder_names if name in args.folder]

    for folder_name in folder_names:
        checkpoint_dir = os.path.join(args.checkpoints_dir, folder_name)
        checkpoint_path = os.path.join(checkpoint_dir, "attack_result.pt")
        attack = _attack_label(checkpoint_dir)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        print(f"evaluating {attack} ({folder_name}) started {started_at}")
        try:
            metrics = evaluate_checkpoint(
                checkpoint_path, device, args.raw_data_dir, args.batch_size
            )
            save_metrics(os.path.join(checkpoint_dir, "metrics.json"), metrics)
            ended_at = datetime.now(timezone.utc).isoformat()
            elapsed = time.monotonic() - started
            print(
                f"finished {attack} ({folder_name}) ended {ended_at}, "
                f"took {elapsed:.1f}s"
            )
        except Exception as error:
            print(f"FAILED {folder_name}: {error}")


if __name__ == "__main__":
    main()
