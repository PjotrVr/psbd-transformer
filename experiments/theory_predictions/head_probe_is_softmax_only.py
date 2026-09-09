"""Prediction 1, the sharp form: a head probe is computable from the softmax alone.

The mean AUROC table says a head adjacent probe stops working. This says WHY, by
reconstructing its output with zero forward passes.

final_norm_out is a post hook on the last LayerNorm, one linear layer from the
logits, and the only operator that runs there is gain_scale, which multiplies
that output by omega = 1 + rate. With head weight W and bias b,

    original form
        z' = W (omega h) + b = omega (W h + b) - (omega - 1) b = omega z - (omega - 1) b

    descriptive form
        perturbed_logits = omega * logits, up to the head bias

and since p = softmax(z) determines z up to an additive constant, softmax(omega z)
is computable from p and nothing else:

    original form
        p'_c = p_c^omega / sum_j p_j^omega

    descriptive form
        perturbed_confidence = confidence^omega, renormalized over the classes

So the prediction is exact rather than asymptotic: at this position the sweep's k
forward passes reproduce a number already contained in the unperturbed softmax.
If the reconstruction matches the cached tensors, the probe carries no information
the confidence carries, which is precisely what the closed form said must happen
as the site approaches the head.

The same reconstruction is applied at a mid stack position as the control. It has
to fail there, or the test measures nothing.

Example
    PYTHONPATH=. .venv/bin/python experiments/theory_predictions/head_probe_is_softmax_only.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)
from defences.decision import complete_rates

HEAD_PLACEMENT = "final_norm_out_gain_scale"
# Same operator, same deterministic amplification, mid stack instead of at the
# head. The reconstruction has no reason to hold here.
CONTROL_PLACEMENT = "mlp_norm_out_gain_scale"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="experiments/theory_predictions")
    parser.add_argument("--checkpoints", type=int, default=30)
    return parser.parse_args()


def temperature_rescaled_confidence(probs: torch.Tensor, omega: float) -> np.ndarray:
    """p_c^omega / sum_j p_j^omega, in log space so tiny probabilities survive."""
    log_probs = probs.double().clamp_min(1e-30).log()
    rescaled = torch.softmax(omega * log_probs, dim=1)
    return rescaled.max(dim=1).values.numpy()


def reconstruction_row(
    psbd_dir: str, placement: str, rate: float, split: str
) -> dict | None:
    path = dropout_pass_path(psbd_dir, placement, rate, split)
    if not os.path.exists(path):
        return None

    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    passes, _ = load_dropout_pass_probs(path)

    measured = passes.double().mean(dim=0).numpy()
    predicted = temperature_rescaled_confidence(probs, 1.0 + rate)
    confidence = probs.double().max(dim=1).values.numpy()

    measured_psu = confidence - measured
    predicted_psu = confidence - predicted

    return {
        "n": int(len(measured)),
        "mean_abs_error": float(np.abs(measured - predicted).mean()),
        "max_abs_error": float(np.abs(measured - predicted).max()),
        "spearman_smoothed_confidence": float(spearmanr(measured, predicted).statistic),
        "spearman_psu": float(spearmanr(measured_psu, predicted_psu).statistic),
        # The one that matters operationally: how well does raw confidence alone,
        # with no reconstruction at all, rank the probe's own output?
        "spearman_psu_vs_confidence": float(
            spearmanr(measured_psu, confidence).statistic
        ),
    }


def rows_for_placement(
    results_dir: str, folders: list[str], placement: str
) -> pd.DataFrame:
    rows = []
    for folder in folders:
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        for rate in complete_rates(psbd_dir, placement):
            for split in ("clean", "backdoor"):
                row = reconstruction_row(psbd_dir, placement, rate, split)
                if row is not None:
                    rows.append(
                        {
                            "folder": folder,
                            "placement": placement,
                            "rate": rate,
                            "split": split,
                            **row,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    candidates = sorted(
        name
        for name in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, name, "psbd", HEAD_PLACEMENT))
        and "benign" not in name
    )
    folders = candidates[: args.checkpoints]
    print(f"{len(folders)} checkpoints carrying {HEAD_PLACEMENT}\n")

    head = rows_for_placement(args.results_dir, folders, HEAD_PLACEMENT)
    control = rows_for_placement(args.results_dir, folders, CONTROL_PLACEMENT)
    table = pd.concat([head, control], ignore_index=True)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("reconstruction of the probe's output from the unperturbed softmax alone")
    print(
        table.groupby(["placement", "split"])
        .agg(
            cells=("n", "size"),
            mean_abs_error=("mean_abs_error", "mean"),
            max_abs_error=("max_abs_error", "max"),
            spearman_smoothed=("spearman_smoothed_confidence", "mean"),
            spearman_psu=("spearman_psu", "mean"),
            spearman_psu_vs_confidence=("spearman_psu_vs_confidence", "mean"),
        )
        .round(4)
        .to_string()
    )

    print("\nby amplification factor, head placement only")
    print(
        head.groupby(["rate", "split"])
        .agg(
            cells=("n", "size"),
            mean_abs_error=("mean_abs_error", "mean"),
            spearman_smoothed=("spearman_smoothed_confidence", "mean"),
            spearman_psu=("spearman_psu", "mean"),
        )
        .round(4)
        .to_string()
    )

    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(
        os.path.join(args.out_dir, "p1_head_probe_reconstruction.csv"), index=False
    )
    print(f"\nwrote p1_head_probe_reconstruction.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
