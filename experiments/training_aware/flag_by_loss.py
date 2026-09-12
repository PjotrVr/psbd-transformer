"""Rank a B1 checkpoint's training samples by area under their loss curve and flag
the most suspicious ones for a sanitised B2 retrain.

experiments/early_loss_signal found that a poisoned training sample's loss runs
low throughout training, because the trigger gives the model an easy shortcut to
the target label, and that the area under a sample's loss curve separates
poisoned from clean samples on a ViT smoke checkpoint. This is that same score at
full budget: stage B1 (cli.train_backdoor --record-sample-loss, 15 epochs, full
data) writes checkpoints/<folder>_lossrec/sample_loss.npz, the identical
(epoch, sample) history early_loss_signal reads. This script reads it back, flags
the FLAG_MULTIPLIER times the recorded poison count lowest-area samples, and
writes checkpoints/<folder>_lossrec/flagged_indices.json for stage B2's
--exclude-indices-file to consume.

Tran et al.'s removal budget (FLAG_MULTIPLIER = 1.5, also
experiments/early_loss_signal.FLAG_MULTIPLIER) is reused rather than re-derived,
so a sanitised retrain always drops more samples than were actually poisoned.

    python experiments/training_aware/flag_by_loss.py vit_cifar10_tact_0_05
"""

import argparse
import json
import os

import numpy as np

CHECKPOINT_DIR = "checkpoints"
FLAG_MULTIPLIER = 1.5


def load_sample_loss(folder: str) -> tuple[np.ndarray, np.ndarray]:
    """B1's (epoch, sample) loss history and its recorded ground-truth poison indices."""
    path = os.path.join(CHECKPOINT_DIR, f"{folder}_lossrec", "sample_loss.npz")
    archive = np.load(path)
    loss_history = archive["sample_loss"]  # (epochs, n_samples)
    poison_indices = archive["poison_indices"]
    return loss_history, poison_indices


def area_under_loss_curve(loss_history: np.ndarray) -> np.ndarray:
    """Per sample, the trapezoidal area under its loss trajectory across epochs."""
    area = np.trapezoid(loss_history, axis=0)  # (n_samples,)
    return area


def flag_lowest_area(area: np.ndarray, flag_count: int) -> list[int]:
    """The flag_count sample indices with the smallest area, lowest (most suspicious) first."""
    order = np.argsort(area)  # ascending
    flagged = order[:flag_count].tolist()
    return flagged


def write_flagged_indices(folder: str, flagged: list[int]) -> str:
    """checkpoints/<folder>_lossrec/flagged_indices.json, sorted for a stable diff."""
    path = os.path.join(CHECKPOINT_DIR, f"{folder}_lossrec", "flagged_indices.json")
    with open(path, "w") as handle:
        json.dump(sorted(flagged), handle)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder", help="the base checkpoint folder, e.g. vit_cifar10_tact_0_05"
    )
    arguments = parser.parse_args()
    return arguments


def main() -> None:
    args = parse_args()
    loss_history, poison_indices = load_sample_loss(args.folder)
    n_poisoned = len(poison_indices)
    flag_count = min(loss_history.shape[1], int(round(FLAG_MULTIPLIER * n_poisoned)))

    area = area_under_loss_curve(loss_history)
    flagged = flag_lowest_area(area, flag_count)
    path = write_flagged_indices(args.folder, flagged)

    true_positives = len(set(flagged) & set(poison_indices.tolist()))
    precision = true_positives / len(flagged) if flagged else float("nan")
    recall = true_positives / n_poisoned if n_poisoned else float("nan")
    print(f"{args.folder}: recorded poison count {n_poisoned}, flagged {len(flagged)}")
    print(
        f"  truly poisoned among flagged: {true_positives} "
        f"(precision {precision:.3f}, recall {recall:.3f})"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
