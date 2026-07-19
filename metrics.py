"""CLI: baseline attack-success and clean-accuracy metrics for every checkpoint.

Decoupled from the PSBD dropout sweep (archived, pending a rewrite), which
will write its own psbd_metrics.json into the same results/<folder>/
directories this script creates. This script only answers: for a benign
model, what is clean accuracy overall and per class; for an attacked model,
what is the attack success rate and the clean accuracy on unpoisoned
counterparts.

All the actual evaluation logic lives in evaluate.py (model in, metrics out)
and loaders.py (dataset in, DataLoader out), both reusable directly without
going through this CLI or the checkpoints/ directory convention at all. This
file is just the loop-over-every-checkpoint entrypoint.

Example
    python metrics.py --folder vit_cifar10_benign vit_cifar10_badnet_a2o_0_1
"""

import argparse

from evaluate import CHECKPOINTS_DIR, RESULTS_DIR, evaluate_all_checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline attack/benign metrics for every checkpoint"
    )
    parser.add_argument("--checkpoints-dir", default=CHECKPOINTS_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--folder", nargs="+", default=None, help="only process these folder names"
    )
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_all_checkpoints(
        checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        folder_filter=args.folder,
    )


if __name__ == "__main__":
    main()
