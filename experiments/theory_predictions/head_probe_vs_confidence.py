"""Prediction 1: a probe near the classifier head cannot beat a confidence baseline.

At the logits the trace of the Hessian of the predicted class probability is
2 p_c (||p||^2 - p_c), a pure function of the softmax vector (verified in
hessian_trace_identity.py). So the second order account of PSU says that as the
probe site approaches the head, PSU carries no information the no perturbation
softmax does not already carry, and detection power has to decay toward what a
max softmax confidence detector gets for free.

Three measurements, in increasing sharpness:

  1. the free baseline itself: AUROC of max softmax confidence, per checkpoint,
     with no perturbation at all,
  2. mean detection AUROC per position, positions ordered by depth, against that
     baseline on the same checkpoints,
  3. the operator controlled version: gain_scale is the only operator that runs
     at 3 depths (attention_norm_out and mlp_norm_out inside every block,
     final_norm_out once immediately before the head), so comparing those 3
     changes depth with the operator held fixed.

A caveat that has to travel with the depth axis. Every block scope position is
attached in EVERY block, so it spans the whole stack and has no single depth.
Only input_pixels, after_embedding (before the stack) and final_norm_out (after
it) sit at a genuine fixed depth, and those are the 3 the prediction actually
separates.

Example
    PYTHONPATH=. .venv/bin/python experiments/theory_predictions/head_probe_vs_confidence.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from psbd.cache import baseline_path, load_baseline, read_split_manifest
from psbd.decision import pair_clean_to_backdoor

# Forward order through the network. Block scope positions repeat in every block,
# so their index is a within block phase rather than a depth in the stack; the
# 3 entries that are genuinely ordered by distance to the head are input_pixels
# and after_embedding at the bottom and final_norm_out at the top.
POSITION_DEPTH_ORDER: tuple[str, ...] = (
    "input_pixels",
    "after_embedding",
    "before_attention_norm",
    "attention_norm_out",
    "before_attention",
    "attention_heads",
    "before_attention_residual",
    "after_attention_residual",
    "before_mlp_norm",
    "mlp_norm_out",
    "before_mlp",
    "mlp_neurons",
    "before_mlp_residual",
    "after_mlp_residual",
    "final_norm_out",
)

# Positions that sit at one fixed depth rather than once per block. These carry
# the prediction; everything else spans the stack.
FIXED_DEPTH_POSITIONS: dict[str, str] = {
    "input_pixels": "before the stack",
    "after_embedding": "before the stack",
    "final_norm_out": "after the stack, 1 linear layer from the logits",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--summary", default="results/detection_summary.csv")
    parser.add_argument("--out-dir", default="experiments/theory_predictions")
    return parser.parse_args()


def auroc_backdoor_positive(
    clean: np.ndarray, backdoor: np.ndarray, negate: bool
) -> float:
    """AUROC with backdoor as the positive class. negate when LOW means poisoned."""
    sign = -1.0 if negate else 1.0
    scores = np.concatenate([sign * clean, sign * backdoor])
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def max_softmax(probs: torch.Tensor) -> torch.Tensor:
    return probs.float().max(dim=1).values


def hessian_trace_statistic(probs: torch.Tensor) -> torch.Tensor:
    """2 p_c (||p||^2 - p_c), the logit layer PSU up to the isotropic scale.

    Negated on return so that, like PSU, a LOW value is the poisoned direction:
    PSU at the logits is sigma^2 * p_c(p_c - ||p||^2) = -sigma^2/2 * trace.
    """
    p_c = probs.float().max(dim=1).values
    return p_c * (p_c - probs.float().pow(2).sum(dim=1))


def confidence_baseline_row(psbd_dir: str) -> dict | None:
    """The no perturbation baselines for one checkpoint, scored as a detector."""
    clean_file = baseline_path(psbd_dir, "clean")
    backdoor_file = baseline_path(psbd_dir, "backdoor")
    if not (os.path.exists(clean_file) and os.path.exists(backdoor_file)):
        return None

    clean_probs, _, _ = load_baseline(clean_file)
    backdoor_probs, _, _ = load_baseline(backdoor_file)
    manifest = read_split_manifest(psbd_dir)

    # Same pairing the detector uses, so the clean population is the one the
    # backdoor split was built from rather than the whole pool.
    paired_probs = pair_clean_to_backdoor(clean_probs, manifest)

    clean_confidence = max_softmax(paired_probs).numpy()
    backdoor_confidence = max_softmax(backdoor_probs).numpy()
    clean_trace = hessian_trace_statistic(paired_probs).numpy()
    backdoor_trace = hessian_trace_statistic(backdoor_probs).numpy()

    return {
        # Higher confidence means poisoned: a backdoored model is very sure about
        # a triggered input. Not negated.
        "confidence_auroc": auroc_backdoor_positive(
            clean_confidence, backdoor_confidence, negate=False
        ),
        # The logit layer PSU surrogate, scored the way PSU is scored.
        "logit_psu_auroc": auroc_backdoor_positive(
            clean_trace, backdoor_trace, negate=True
        ),
        "clean_confidence_mean": float(clean_confidence.mean()),
        "backdoor_confidence_mean": float(backdoor_confidence.mean()),
        "n_clean": int(len(clean_confidence)),
        "n_backdoor": int(len(backdoor_confidence)),
    }


def collect_confidence_baselines(results_dir: str, folders: list[str]) -> pd.DataFrame:
    rows = []
    for folder in folders:
        row = confidence_baseline_row(os.path.join(results_dir, folder, "psbd"))
        if row is not None:
            rows.append({"folder": folder, **row})
    return pd.DataFrame(rows)


def plain_measurements(summary_path: str) -> pd.DataFrame:
    """Rows that are plain measurements: no pass count, block range or superseded variant."""
    summary = pd.read_csv(summary_path)
    return summary[summary.variant.isna()].copy()


def position_table(
    measurements: pd.DataFrame, baselines: pd.DataFrame, rule: str
) -> pd.DataFrame:
    """Mean AUROC per position under one rate rule, against the paired baseline.

    The baseline is averaged over exactly the checkpoints that contributed to the
    position, so the gap is not an artefact of 2 different checkpoint sets.
    """
    selected = measurements[measurements.rule == rule]
    joined = selected.merge(baselines, on="folder", how="inner")

    grouped = joined.groupby("position").agg(
        n_cells=("auroc", "size"),
        n_checkpoints=("folder", "nunique"),
        mean_auroc=("auroc", "mean"),
        mean_confidence_auroc=("confidence_auroc", "mean"),
        mean_logit_psu_auroc=("logit_psu_auroc", "mean"),
    )
    grouped["gap_to_confidence"] = grouped.mean_auroc - grouped.mean_confidence_auroc
    grouped["depth_index"] = [
        POSITION_DEPTH_ORDER.index(name) if name in POSITION_DEPTH_ORDER else np.nan
        for name in grouped.index
    ]
    ordered = grouped.sort_values("depth_index")
    return ordered


def gain_scale_paired(
    measurements: pd.DataFrame, baselines: pd.DataFrame, rule: str
) -> pd.DataFrame:
    """The operator controlled depth comparison, on checkpoints having all 3 depths."""
    gain = measurements[
        (measurements.operator == "gain_scale") & (measurements.rule == rule)
    ]
    positions = ("attention_norm_out", "mlp_norm_out", "final_norm_out")
    wide = gain.pivot_table(index="folder", columns="position", values="auroc")
    complete = wide.dropna(subset=list(positions))

    joined = complete.join(
        baselines.set_index("folder")[["confidence_auroc", "logit_psu_auroc"]],
        how="inner",
    )
    return joined[list(positions) + ["confidence_auroc", "logit_psu_auroc"]]


def mirror_test(measurements: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Is the head probe the confidence baseline run backwards?

    At final_norm_out the perturbation amplifies the pre head activation, which
    sharpens the softmax most for the LEAST confident samples, so PSU there
    ranks by low confidence. Low PSU is the poisoned direction, so the probe
    flags the least confident sample while a backdoored input is the most
    confident one. If that reading is right, the probe's AUROC is not merely
    close to the confidence baseline, it is 1 minus it.
    """
    head = measurements[
        (measurements.variant.isna()) & (measurements.position == "final_norm_out")
    ]
    joined = head.merge(baselines[["folder", "confidence_auroc"]], on="folder")
    joined["mirrored_confidence_auroc"] = 1.0 - joined.confidence_auroc

    rows = []
    for rule, block in joined.groupby("rule"):
        rows.append(
            {
                "rule": rule,
                "n_checkpoints": block.folder.nunique(),
                "mean_head_auroc": block.auroc.mean(),
                "mean_mirrored_confidence": block.mirrored_confidence_auroc.mean(),
                "pearson": float(
                    pearsonr(block.auroc, block.mirrored_confidence_auroc).statistic
                ),
                "mean_abs_difference": float(
                    (block.auroc - block.mirrored_confidence_auroc).abs().mean()
                ),
                "max_abs_difference": float(
                    (block.auroc - block.mirrored_confidence_auroc).abs().max()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("rule")


def print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(frame.round(4).to_string())


def main() -> None:
    args = parse_args()
    measurements = plain_measurements(args.summary)
    folders = sorted(measurements.folder.unique())

    baselines = collect_confidence_baselines(args.results_dir, folders)
    print(
        f"confidence baseline computed on {len(baselines)} checkpoints, "
        f"no perturbation, no GPU"
    )
    architecture = (
        measurements.drop_duplicates("folder").set_index("folder").architecture
    )
    baselines["architecture"] = baselines.folder.map(architecture)
    baselines["backdoored"] = ~baselines.folder.str.contains("benign")
    print(
        baselines.groupby(["architecture", "backdoored"])
        .agg(
            n=("confidence_auroc", "size"),
            confidence_auroc=("confidence_auroc", "mean"),
            logit_psu_auroc=("logit_psu_auroc", "mean"),
        )
        .round(4)
        .to_string()
    )

    for rule in ("oracle", "adaptive", "matched_sigma0.6"):
        print_frame(
            f"positions by depth, rule {rule}",
            position_table(measurements, baselines, rule),
        )

    for rule in ("oracle", "matched_sigma0.6"):
        paired = gain_scale_paired(measurements, baselines, rule)
        print_frame(
            f"gain_scale only, {len(paired)} checkpoints carrying all 3 depths, rule {rule}",
            paired.describe().loc[["mean", "std", "min", "max"]],
        )

    print_frame(
        "head probe against the MIRRORED confidence baseline, per checkpoint",
        mirror_test(measurements, baselines),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    baselines.to_csv(
        os.path.join(args.out_dir, "p1_confidence_baseline.csv"), index=False
    )
    frames = []
    for rule in (
        "oracle",
        "adaptive",
        "matched_sigma0.2",
        "matched_sigma0.4",
        "matched_sigma0.6",
        "matched_sigma0.8",
    ):
        table = position_table(measurements, baselines, rule)
        table["rule"] = rule
        frames.append(table.reset_index())
    pd.concat(frames).to_csv(
        os.path.join(args.out_dir, "p1_position_vs_confidence.csv"), index=False
    )
    print(
        f"\nwrote p1_confidence_baseline.csv and p1_position_vs_confidence.csv to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
