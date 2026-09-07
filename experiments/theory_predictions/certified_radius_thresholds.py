"""Prediction 3: the certified radius reformulation changes thresholds, not ranking.

Cohen et al. (ICML 2019) certify a radius R = sigma * Phi^{-1}(p_tilde) for a
smoothed classifier, and the sweep already computes p_tilde, the smoothed
confidence:

    original form
        p_tilde = E_delta[g(h + delta)] = g(h) - phi(x)

    descriptive form
        smoothed_confidence = mean over the k perturbed passes of the
                              probability assigned to the unperturbed prediction

Phi^{-1} is strictly monotone, so ranking by radius and ranking by smoothed
confidence are the same ranking and must give the same AUROC. sigma is a per cell
constant and drops out of the ranking too, so it is set to 1 here.

Three separate questions hide inside the prediction, and they have different
answers, so they are measured separately.

  a. does Phi^{-1} preserve AUROC?  (radius against smoothed confidence)
  b. does Phi^{-1} change a QUANTILE threshold rule?  (radius quantile against
     smoothed confidence quantile, same nominal budget)
  c. is the radius ranking the same as the DEPLOYED PSU ranking?  PSU is
     g(h) - p_tilde and g(h) varies per sample, so the map from PSU to radius is
     not one shared monotone function and there is no identity to expect. This is
     the half of the prediction that can actually fail.

Detection directions. PSU flags LOW values. Smoothed confidence and radius are
both large when the prediction is robust, so they flag HIGH values, and their
quantile threshold is the upper tail 1 - q of clean validation.

Example
    PYTHONPATH=. .venv/bin/python experiments/theory_predictions/certified_radius_thresholds.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, wilcoxon
from sklearn.metrics import roc_auc_score

from psbd.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from psbd.decision import pair_clean_to_backdoor
from psbd.scores import psu_from_cache

QUANTILES = (0.01, 0.05, 0.10, 0.25)

# Phi^{-1} diverges at 0 and 1, and float32 smoothed confidences do hit both, so
# the radius is only computable after a clamp. How often the clamp binds is
# reported rather than hidden, because a saturated tail is exactly where the
# radius rule could differ from the confidence rule.
PROBABILITY_CLAMP = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--summary", default="results/detection_summary.csv")
    parser.add_argument("--out-dir", default="experiments/theory_predictions")
    parser.add_argument("--cells", type=int, default=140)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def smoothed_confidence(
    psbd_dir: str, placement: str, rate: float, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """(smoothed confidence, PSU) for one split, both (N,) float64."""
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    passes, _ = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, placement, rate, split)
    )

    psu = psu_from_cache(probs, labels, passes).double().numpy()
    tilde = passes.double().mean(dim=0).numpy()
    return tilde, psu


def certified_radius(tilde: np.ndarray) -> np.ndarray:
    """sigma * Phi^{-1}(p_tilde) with sigma = 1, which fixes no ranking."""
    clamped = np.clip(tilde, PROBABILITY_CLAMP, 1.0 - PROBABILITY_CLAMP)
    return norm.ppf(clamped)


def auroc(clean: np.ndarray, backdoor: np.ndarray, high_is_poisoned: bool) -> float:
    sign = 1.0 if high_is_poisoned else -1.0
    scores = np.concatenate([sign * clean, sign * backdoor])
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def operating_point(
    validation: np.ndarray,
    clean: np.ndarray,
    backdoor: np.ndarray,
    quantile: float,
    high_is_poisoned: bool,
) -> dict:
    """TPR and FPR from a clean validation quantile, the defender legal rule."""
    if high_is_poisoned:
        threshold = float(np.quantile(validation, 1.0 - quantile))
        flagged_clean, flagged_backdoor = clean > threshold, backdoor > threshold
    else:
        threshold = float(np.quantile(validation, quantile))
        flagged_clean, flagged_backdoor = clean < threshold, backdoor < threshold

    return {
        "threshold": threshold,
        "fpr": float(flagged_clean.mean()),
        "tpr": float(flagged_backdoor.mean()),
    }


def cell_row(results_dir: str, meta: pd.Series) -> dict | None:
    psbd_dir = os.path.join(results_dir, meta.folder, "psbd")
    needed = [
        dropout_pass_path(psbd_dir, meta.placement, meta.rate, split)
        for split in ("validation", "clean", "backdoor")
    ]
    if not all(os.path.exists(path) for path in needed):
        return None

    manifest = read_split_manifest(psbd_dir)
    validation_tilde, validation_psu = smoothed_confidence(
        psbd_dir, meta.placement, meta.rate, "validation"
    )
    clean_tilde_all, clean_psu_all = smoothed_confidence(
        psbd_dir, meta.placement, meta.rate, "clean"
    )
    backdoor_tilde, backdoor_psu = smoothed_confidence(
        psbd_dir, meta.placement, meta.rate, "backdoor"
    )

    clean_tilde = pair_clean_to_backdoor(
        torch.from_numpy(clean_tilde_all), manifest
    ).numpy()
    clean_psu = pair_clean_to_backdoor(
        torch.from_numpy(clean_psu_all), manifest
    ).numpy()

    radius = {
        "validation": certified_radius(validation_tilde),
        "clean": certified_radius(clean_tilde),
        "backdoor": certified_radius(backdoor_tilde),
    }
    pooled = np.concatenate([validation_tilde, clean_tilde, backdoor_tilde])
    clamped_share = float(
        ((pooled <= PROBABILITY_CLAMP) | (pooled >= 1.0 - PROBABILITY_CLAMP)).mean()
    )

    row = {
        "folder": meta.folder,
        "placement": meta.placement,
        "position": meta.position,
        "operator": meta.operator,
        "rule": meta.rule,
        "rate": meta.rate,
        "architecture": meta.architecture,
        "dataset": meta.dataset,
        "attack": meta.attack,
        "summary_auroc": meta.auroc,
        "clamped_share": clamped_share,
        "auroc_psu": auroc(clean_psu, backdoor_psu, high_is_poisoned=False),
        "auroc_smoothed_confidence": auroc(
            clean_tilde, backdoor_tilde, high_is_poisoned=True
        ),
        "auroc_radius": auroc(
            radius["clean"], radius["backdoor"], high_is_poisoned=True
        ),
    }

    for quantile in QUANTILES:
        psu_point = operating_point(
            validation_psu, clean_psu, backdoor_psu, quantile, high_is_poisoned=False
        )
        tilde_point = operating_point(
            validation_tilde,
            clean_tilde,
            backdoor_tilde,
            quantile,
            high_is_poisoned=True,
        )
        radius_point = operating_point(
            radius["validation"],
            radius["clean"],
            radius["backdoor"],
            quantile,
            high_is_poisoned=True,
        )
        tag = f"q{quantile:g}"
        row[f"{tag}_psu_fpr"] = psu_point["fpr"]
        row[f"{tag}_psu_tpr"] = psu_point["tpr"]
        row[f"{tag}_tilde_fpr"] = tilde_point["fpr"]
        row[f"{tag}_tilde_tpr"] = tilde_point["tpr"]
        row[f"{tag}_radius_fpr"] = radius_point["fpr"]
        row[f"{tag}_radius_tpr"] = radius_point["tpr"]
    return row


def select_cells(summary: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    plain = summary[summary.variant.isna()].copy()
    plain = plain[~plain.folder.str.contains("benign")]
    plain = plain[plain.rate.notna()]
    plain = plain.drop_duplicates(subset=["folder", "placement", "rate"])

    # The deployment recommendation, in full, plus a spread over everything else,
    # so a null result cannot be blamed on a narrow panel.
    recommended = plain[
        (plain.placement == "before_attention_norm_token_mask")
        & (plain.rule == "adaptive")
    ]
    rest = plain.drop(recommended.index)
    extra = rest.sample(min(count, len(rest)), random_state=seed)
    return pd.concat([recommended, extra]).drop_duplicates(
        subset=["folder", "placement", "rate"]
    )


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    cells = select_cells(summary, args.cells, args.seed)

    rows = [
        row
        for _, meta in cells.iterrows()
        if (row := cell_row(args.results_dir, meta)) is not None
    ]
    table = pd.DataFrame(rows)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    print(
        f"cells: {len(table)} over {table.folder.nunique()} checkpoints, "
        f"{table.position.nunique()} positions, {table.operator.nunique()} operators\n"
    )

    identity = (table.auroc_radius - table.auroc_smoothed_confidence).abs()
    print("a. does Phi^{-1} preserve AUROC")
    print(f"   max |AUROC(radius) - AUROC(smoothed confidence)| = {identity.max():.3e}")
    print(
        f"   cells where they differ at all                   = {int((identity > 0).sum())} of {len(table)}"
    )
    print(
        f"   share of probabilities hitting the clamp          = {table.clamped_share.mean():.4f} mean, "
        f"{table.clamped_share.max():.4f} max\n"
    )

    print("b. does Phi^{-1} change the quantile threshold rule")
    for quantile in QUANTILES:
        tag = f"q{quantile:g}"
        fpr_gap = (table[f"{tag}_radius_fpr"] - table[f"{tag}_tilde_fpr"]).abs()
        tpr_gap = (table[f"{tag}_radius_tpr"] - table[f"{tag}_tilde_tpr"]).abs()
        print(
            f"   q={quantile:<5} max |dFPR| {fpr_gap.max():.3e}   max |dTPR| {tpr_gap.max():.3e}   "
            f"cells differing {int(((fpr_gap > 0) | (tpr_gap > 0)).sum())}"
        )

    print("\nc. radius space against the deployed PSU statistic")
    gap = table.auroc_radius - table.auroc_psu
    print(
        f"   mean AUROC(radius) {table.auroc_radius.mean():.4f}   mean AUROC(PSU) {table.auroc_psu.mean():.4f}   "
        f"mean gap {gap.mean():+.4f}"
    )
    print(
        f"   radius wins on {int((gap > 0).sum())} of {len(table)} cells, identical on {int((gap == 0).sum())}"
    )
    print(f"   max |gap| {gap.abs().max():.4f}")

    print("\n   achieved FPR against the nominal budget, and TPR, mean over cells")
    header = f"   {'q':>6} {'PSU fpr':>9} {'radius fpr':>11} {'PSU |err|':>10} {'radius |err|':>13} {'PSU tpr':>9} {'radius tpr':>11}"
    print(header)
    for quantile in QUANTILES:
        tag = f"q{quantile:g}"
        psu_error = (table[f"{tag}_psu_fpr"] - quantile).abs().mean()
        radius_error = (table[f"{tag}_radius_fpr"] - quantile).abs().mean()
        print(
            f"   {quantile:>6} {table[f'{tag}_psu_fpr'].mean():>9.4f} {table[f'{tag}_radius_fpr'].mean():>11.4f} "
            f"{psu_error:>10.4f} {radius_error:>13.4f} {table[f'{tag}_psu_tpr'].mean():>9.4f} "
            f"{table[f'{tag}_radius_tpr'].mean():>11.4f}"
        )

    print("\n   paired TPR gap, radius minus PSU, at each nominal budget")
    print(
        f"   {'q':>6} {'mean dTPR':>10} {'median':>9} {'radius wins':>12} {'wilcoxon p':>12}"
    )
    for quantile in QUANTILES:
        tag = f"q{quantile:g}"
        difference = (table[f"{tag}_radius_tpr"] - table[f"{tag}_psu_tpr"]).to_numpy()
        nonzero = difference[difference != 0]
        statistic = wilcoxon(nonzero) if len(nonzero) > 10 else None
        wins = float((difference > 0).mean())
        print(
            f"   {quantile:>6} {difference.mean():>+10.4f} {np.median(difference):>+9.4f} "
            f"{wins:>12.3f} {statistic.pvalue if statistic else float('nan'):>12.2e}"
        )

    print("\n   by position, AUROC(radius) - AUROC(PSU)")
    print(
        table.groupby("position")
        .agg(
            n=("auroc_psu", "size"),
            auroc_psu=("auroc_psu", "mean"),
            auroc_radius=("auroc_radius", "mean"),
            gap=("auroc_psu", lambda column: np.nan),
        )
        .assign(
            gap=table.groupby("position").apply(
                lambda block: float((block.auroc_radius - block.auroc_psu).mean()),
                include_groups=False,
            )
        )
        .round(4)
        .to_string()
    )

    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "p3_radius_vs_psu.csv"), index=False)
    print(f"\nwrote p3_radius_vs_psu.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
