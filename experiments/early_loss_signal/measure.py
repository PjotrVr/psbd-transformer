"""Does the early per-sample training loss separate poisoned from clean samples on
ViT, and does that separation track PSBD-TM's own test-time catch rate on the same
attack?

Anti-Backdoor Learning (Li et al., ASD's isolation stage) reports that a poisoned
training sample's loss falls faster than a clean one's, because the trigger gives
the model an easy shortcut to the target label. If that also holds on a ViT
fine-tuned with an adaptive attacker's usual recipe, the per-sample loss trajectory
--record-sample-loss now writes for free would let a defender sanitise the training
set before ever running PSBD: drop the samples the loss signal flags, retrain, then
deploy PSBD-TM on a model whose backdoor is weaker or gone. This script reads the 4
smoke checkpoints (`pbs/early_loss_smoke/smoke.pbs`: bpp and tact at 1% and 5%
poisoning, CIFAR-100, 8000 samples, 5 epochs, no evasion) and asks 2 questions. Does
either loss-derived score separate poisoned from clean training samples above
chance? And, per attack, is that separation aligned with the SAME attack's PSBD-TM
AUROC on the fully trained panel checkpoint's cache, or is the training-set signal
independent of what the test-time defence already sees?

The 2 per-sample scores are the epoch at which a sample's loss first falls below
that epoch's median loss (ASD's own isolation criterion, "learned fast") and the
area under the sample's loss curve across every epoch recorded (a summary of how
low that sample's loss ran throughout training). Both are scored by AUROC against
the ground-truth poison flag, and by the share of poisoned samples caught in the
top `FLAG_MULTIPLIER` times the true poison count, the Spectral Signatures removal
budget already used in `experiments/sam_training_set_detection/measure.py`.

The PSBD-TM comparison reads the SAME attack and rate's FULLY trained (15 epoch,
uncapped) panel checkpoint's `results/<folder>/psbd_metrics.json`, at
RECOMMENDED_PLACEMENT's adaptive rate, `detection_psu_ratio` at the 10% clean-
quantile row. This is a per-attack comparison rather than a per-sample comparison.
The smoke checkpoints train on a capped, undertrained subset. PSBD-TM's own eval
pool is the held-out backdoor split, a disjoint set of images from the training
set the loss signal scores. See README.md for the tables and the verdict.

    PYTHONPATH=. .venv/bin/python experiments/early_loss_signal/measure.py
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from defences.decision import RECOMMENDED_PLACEMENT
from scripts.paper._common import load_psbd_metrics, rate_row

CHECKPOINT_DIR = "checkpoints"
PANEL_RESULTS_DIR = "results"
OUTPUT_DIR = "results/_experiments/early_loss_signal"

# Tran et al.'s own removal rule, reused rather than re-derived: flag the top
# multiplier times the expected poison count.
FLAG_MULTIPLIER = 1.5

# The smoke checkpoint (--record-sample-loss) beside the fully trained panel
# checkpoint whose PSBD-TM cache stands in for "this attack's test-time catch
# rate". Never the same folder: the smoke run is capped at 8000 samples and 5
# epochs, and is never itself run through cli.sweep.
SMOKE_TO_PANEL = {
    "vit_cifar100_bpp_0_01_loss_smoke": "vit_cifar100_bpp_0_01",
    "vit_cifar100_bpp_0_05_loss_smoke": "vit_cifar100_bpp_0_05",
    "vit_cifar100_tact_0_01_loss_smoke": "vit_cifar100_tact_0_01",
    "vit_cifar100_tact_0_05_loss_smoke": "vit_cifar100_tact_0_05",
}


def load_sample_loss(folder_name: str) -> tuple[np.ndarray, np.ndarray]:
    """The (epoch, sample) loss history and the per-sample poison flag it carries."""
    path = os.path.join(CHECKPOINT_DIR, folder_name, "sample_loss.npz")
    archive = np.load(path)
    loss_history = archive["sample_loss"]  # (epochs, n_samples)
    poison_indices = archive["poison_indices"]

    poisoned = np.zeros(loss_history.shape[1], dtype=bool)
    poisoned[poison_indices] = True
    return loss_history, poisoned


def epoch_of_first_drop(loss_history: np.ndarray) -> np.ndarray:
    """Per sample, the first epoch (0-indexed) its loss falls below that epoch's median.

    ASD's own isolation criterion: a poisoned sample's loss is expected to drop
    below the pack early. A sample that never drops below its epoch's median gets
    the total epoch count itself, so it reads as the least suspicious value on
    this axis rather than staying undefined.
    """
    epochs, n_samples = loss_history.shape
    epoch_medians = np.median(loss_history, axis=1, keepdims=True)  # (epochs, 1)
    below_median = loss_history < epoch_medians  # (epochs, n_samples)

    first_epoch = np.full(n_samples, epochs, dtype=float)
    for sample in range(n_samples):
        dropped_at = np.flatnonzero(below_median[:, sample])
        if dropped_at.size > 0:
            first_epoch[sample] = dropped_at[0]
    return first_epoch


def area_under_loss_curve(loss_history: np.ndarray) -> np.ndarray:
    """Per sample, the trapezoidal area under its loss trajectory across epochs."""
    area = np.trapezoid(loss_history, axis=0)  # (n_samples,)
    return area


def flag_top_k(suspicion_score: np.ndarray, expected_poison_count: int) -> np.ndarray:
    """Boolean flags for the top FLAG_MULTIPLIER * expected_poison_count scores."""
    flag_count = min(
        len(suspicion_score),
        int(round(FLAG_MULTIPLIER * expected_poison_count)),
    )
    order = np.argsort(-suspicion_score)  # descending
    flagged = np.zeros(len(suspicion_score), dtype=bool)
    flagged[order[:flag_count]] = True
    return flagged


def catch_rate(flagged: np.ndarray, poisoned: np.ndarray) -> float:
    """The share of truly poisoned samples the flagged set catches."""
    n_positive = int(poisoned.sum())
    if n_positive == 0:
        return float("nan")
    caught = float((flagged & poisoned).sum() / n_positive)
    return caught


def test_time_catch_rate(panel_folder: str) -> dict | None:
    """PSBD-TM's own AUROC and TPR at the 10% clean quantile, from the panel cache.

    None when the panel checkpoint has no cache yet or never reaches the
    adaptive shift target, rather than a fabricated 0.
    """
    report = load_psbd_metrics(PANEL_RESULTS_DIR, panel_folder)
    if report is None:
        return None
    block = report["placements"][RECOMMENDED_PLACEMENT]
    rate = block["adaptive_rate"]
    if rate is None:
        return None
    row = rate_row(block, rate)
    headline = row["detection_psu_ratio"]["q0.10"]
    return {"auroc": headline["auroc"], "tpr_at_10fpr": headline["tpr"], "rate": rate}


def evaluate_smoke_checkpoint(folder_name: str, panel_folder: str) -> dict:
    """1 smoke checkpoint's early-loss separation, beside its attack's PSBD-TM read."""
    loss_history, poisoned = load_sample_loss(folder_name)
    epochs, n_samples = loss_history.shape
    n_poisoned = int(poisoned.sum())

    # Higher score means "more suspicious" for both: an early first drop below
    # median (small epoch index) and a small area under the loss curve are both
    # the poisoned direction the ASD literature reports, so both are negated.
    first_drop_score = -epoch_of_first_drop(loss_history)
    auc_curve_score = -area_under_loss_curve(loss_history)

    auroc_first_drop = float(roc_auc_score(poisoned, first_drop_score))
    auroc_auc_curve = float(roc_auc_score(poisoned, auc_curve_score))

    catch_first_drop = catch_rate(flag_top_k(first_drop_score, n_poisoned), poisoned)
    catch_auc_curve = catch_rate(flag_top_k(auc_curve_score, n_poisoned), poisoned)

    psbd_tm = test_time_catch_rate(panel_folder)

    return {
        "folder_name": folder_name,
        "panel_folder": panel_folder,
        "epochs": epochs,
        "n_samples": n_samples,
        "n_poisoned": n_poisoned,
        "auroc_first_drop": auroc_first_drop,
        "auroc_auc_curve": auroc_auc_curve,
        "catch_at_1.5x_first_drop": catch_first_drop,
        "catch_at_1.5x_auc_curve": catch_auc_curve,
        "psbd_tm_auroc": psbd_tm["auroc"] if psbd_tm else None,
        "psbd_tm_tpr_at_10fpr": psbd_tm["tpr_at_10fpr"] if psbd_tm else None,
        "psbd_tm_rate": psbd_tm["rate"] if psbd_tm else None,
    }


def print_summary_table(rows: list[dict]) -> None:
    header = (
        "checkpoint | n_poisoned | AUROC first-drop | AUROC AUC-curve | "
        "catch@1.5x first-drop | catch@1.5x AUC-curve | PSBD-TM AUROC | "
        "PSBD-TM TPR@10%FPR"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        psbd_auroc = (
            f"{row['psbd_tm_auroc']:.3f}" if row["psbd_tm_auroc"] is not None else "n/a"
        )
        psbd_tpr = (
            f"{row['psbd_tm_tpr_at_10fpr']:.3f}"
            if row["psbd_tm_tpr_at_10fpr"] is not None
            else "n/a"
        )
        print(
            f"{row['folder_name']} | {row['n_poisoned']} | "
            f"{row['auroc_first_drop']:.3f} | {row['auroc_auc_curve']:.3f} | "
            f"{row['catch_at_1.5x_first_drop']:.3f} | "
            f"{row['catch_at_1.5x_auc_curve']:.3f} | {psbd_auroc} | {psbd_tpr}"
        )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = [
        evaluate_smoke_checkpoint(smoke_folder, panel_folder)
        for smoke_folder, panel_folder in SMOKE_TO_PANEL.items()
    ]
    print_summary_table(rows)

    output_path = os.path.join(OUTPUT_DIR, "early_loss_signal.json")
    with open(output_path, "w") as handle:
        json.dump(rows, handle, indent=2)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
