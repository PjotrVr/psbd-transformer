"""Prediction 2: inversions concentrate on low confidence clean samples.

The logit layer trace 2 p_c (||p||^2 - p_c) is NON MONOTONE in confidence. In the
2 class case it is 2p(2p-1)(p-1), zero at p = 0.5, zero at p = 1, and largest in
magnitude near p = 0.789. So a sample sitting ON the decision boundary has near
zero curvature just like a saturated one does, and "low PSU means large margin"
stops holding at the low confidence end. The prediction that follows is that a
cell whose AUROC has fallen below 0.5 got there because of its low confidence
CLEAN samples.

The test is an exact decomposition rather than a correlation. AUROC with backdoor
as the positive class and low PSU as the poisoned direction is

    original form
        AUROC = (1 / (|C| |B|)) * sum_{i in C} sum_{j in B} 1[psu_j < psu_i]

    descriptive form
        auroc = mean over clean samples of
                (fraction of backdoor samples whose PSU is lower)

so every clean sample has an exact, additive contribution in [0, 1], and the
question "which clean samples cause the inversion" has a numeric answer rather
than an interpretive one. A cell is inverted when that mean is below 0.5.

What would falsify the prediction, decided before the numbers were read:

  1. inverted cells whose per clean contribution does NOT rise with confidence
     (Spearman correlation at or below 0), or
  2. inverted cells whose local AUROC is below 0.5 in EVERY confidence bin, which
     would make the inversion global rather than concentrated, or
  3. a clean PSU against confidence curve that is monotone rather than single
     peaked.

Example
    PYTHONPATH=. .venv/bin/python experiments/theory_predictions/inversion_confidence_profile.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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

CONFIDENCE_BINS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--summary", default="results/detection_summary.csv")
    parser.add_argument("--out-dir", default="experiments/theory_predictions")
    parser.add_argument("--cells-per-group", type=int, default=45)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_cell(
    results_dir: str, folder: str, placement: str, rate: float
) -> dict | None:
    """PSU and baseline confidence for the clean and backdoor splits of one cell."""
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    files = {
        split: dropout_pass_path(psbd_dir, placement, rate, split)
        for split in ("clean", "backdoor")
    }
    if not all(os.path.exists(path) for path in files.values()):
        return None

    manifest = read_split_manifest(psbd_dir)
    clean_probs, clean_labels, _ = load_baseline(baseline_path(psbd_dir, "clean"))
    backdoor_probs, backdoor_labels, _ = load_baseline(
        baseline_path(psbd_dir, "backdoor")
    )

    clean_passes, _ = load_dropout_pass_probs(files["clean"])
    backdoor_passes, _ = load_dropout_pass_probs(files["backdoor"])

    clean_psu = psu_from_cache(clean_probs, clean_labels, clean_passes)
    backdoor_psu = psu_from_cache(backdoor_probs, backdoor_labels, backdoor_passes)
    clean_confidence = clean_probs.float().max(dim=1).values

    # The detector scores the clean images the backdoor split was built from, so
    # the same restriction has to be applied here or the 2 populations differ.
    return {
        "clean_psu": pair_clean_to_backdoor(clean_psu, manifest).numpy(),
        "clean_confidence": pair_clean_to_backdoor(clean_confidence, manifest).numpy(),
        "backdoor_psu": backdoor_psu.numpy(),
        "backdoor_confidence": backdoor_probs.float().max(dim=1).values.numpy(),
    }


def clean_contributions(clean_psu: np.ndarray, backdoor_psu: np.ndarray) -> np.ndarray:
    """Per clean sample, the fraction of backdoor samples with strictly lower PSU.

    The mean of this vector is exactly the AUROC (with ties counted at half),
    so it partitions the AUROC across clean samples with nothing left over.
    """
    order = np.sort(backdoor_psu)
    below = np.searchsorted(order, clean_psu, side="left")
    ties = np.searchsorted(order, clean_psu, side="right") - below
    return (below + 0.5 * ties) / len(backdoor_psu)


def local_auroc(clean_psu: np.ndarray, backdoor_psu: np.ndarray) -> float:
    if len(clean_psu) < 5 or len(backdoor_psu) < 5:
        return float("nan")
    scores = np.concatenate([-clean_psu, -backdoor_psu])
    labels = np.concatenate([np.zeros(len(clean_psu)), np.ones(len(backdoor_psu))])
    return float(roc_auc_score(labels, scores))


def confidence_bin_profile(cell: dict, edges: np.ndarray) -> pd.DataFrame:
    """Per confidence bin: mean clean PSU, mean backdoor PSU, local AUROC, share."""
    clean_bin = np.clip(
        np.digitize(cell["clean_confidence"], edges) - 1, 0, len(edges) - 2
    )
    backdoor_bin = np.clip(
        np.digitize(cell["backdoor_confidence"], edges) - 1, 0, len(edges) - 2
    )
    contributions = clean_contributions(cell["clean_psu"], cell["backdoor_psu"])

    rows = []
    for index in range(len(edges) - 1):
        clean_mask, backdoor_mask = clean_bin == index, backdoor_bin == index
        rows.append(
            {
                "bin": index,
                "confidence_low": edges[index],
                "confidence_high": edges[index + 1],
                "n_clean": int(clean_mask.sum()),
                "n_backdoor": int(backdoor_mask.sum()),
                "clean_share": float(clean_mask.mean()),
                "mean_clean_psu": float(cell["clean_psu"][clean_mask].mean())
                if clean_mask.any()
                else np.nan,
                "mean_backdoor_psu": float(cell["backdoor_psu"][backdoor_mask].mean())
                if backdoor_mask.any()
                else np.nan,
                "local_auroc": local_auroc(
                    cell["clean_psu"][clean_mask], cell["backdoor_psu"][backdoor_mask]
                ),
                # How much of the cell's AUROC this bin's clean samples supply,
                # and what they would supply if they behaved like a coin flip.
                "mean_contribution": float(contributions[clean_mask].mean())
                if clean_mask.any()
                else np.nan,
                "auroc_deficit_share": float(
                    ((0.5 - contributions) * clean_mask).sum()
                    / max(len(contributions), 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def curve_shape(psu_by_bin: np.ndarray) -> dict:
    """Is the binned clean PSU against confidence curve single peaked or monotone?"""
    usable = psu_by_bin[~np.isnan(psu_by_bin)]
    if len(usable) < 4:
        return {"shape": "too_few_bins", "peak_bin": np.nan}

    steps = np.sign(np.diff(usable))
    steps = steps[steps != 0]
    turns = int((np.diff(steps) != 0).sum()) if len(steps) > 1 else 0
    peak = int(np.nanargmax(psu_by_bin))

    if turns == 0:
        shape = "monotone_increasing" if steps.sum() > 0 else "monotone_decreasing"
    elif turns == 1 and peak not in (0, len(psu_by_bin) - 1):
        shape = "single_peak"
    else:
        shape = f"multi_turn_{turns}"
    return {"shape": shape, "peak_bin": peak}


def select_cells(summary: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Inverted and working cells, spread over checkpoints rather than clustered.

    One cell per (folder, placement) at most, because the 6 rate rules of a single
    placement often select the SAME rate and would otherwise be counted 6 times.
    """
    plain = summary[summary.variant.isna()].copy()
    plain = plain[~plain.folder.str.contains("benign")]
    plain = plain[plain.rate.notna()]
    plain["group"] = np.where(
        plain.auroc < 0.5, "inverted", np.where(plain.auroc > 0.8, "working", None)
    )
    plain = plain[plain.group.notna()]
    plain = plain.drop_duplicates(subset=["folder", "placement", "rate"])

    chosen = []
    for _, block in plain.groupby("group"):
        # At most 1 cell per checkpoint, so the panel is not 45 rows drawn from 3
        # pathological folders.
        shuffled = block.sample(frac=1.0, random_state=seed)
        one_each = shuffled.drop_duplicates(subset=["folder"])
        chosen.append(one_each.sample(min(count, len(one_each)), random_state=seed))
    return pd.concat(chosen)


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    cells = select_cells(summary, args.cells_per_group, args.seed)
    edges = np.linspace(0.0, 1.0, CONFIDENCE_BINS + 1)

    cell_rows, bin_rows = [], []
    for _, cell_meta in cells.iterrows():
        cell = load_cell(
            args.results_dir, cell_meta.folder, cell_meta.placement, cell_meta.rate
        )
        if cell is None:
            continue

        contributions = clean_contributions(cell["clean_psu"], cell["backdoor_psu"])
        correlation = spearmanr(cell["clean_confidence"], contributions)
        profile = confidence_bin_profile(cell, edges)
        shape = curve_shape(profile.mean_clean_psu.to_numpy())

        # The direct removal test: drop the least confident clean decile and see
        # whether the cell's AUROC moves toward 0.5.
        cutoff = np.quantile(cell["clean_confidence"], 0.1)
        kept = cell["clean_confidence"] > cutoff
        trimmed = local_auroc(cell["clean_psu"][kept], cell["backdoor_psu"])

        cell_rows.append(
            {
                "folder": cell_meta.folder,
                "placement": cell_meta.placement,
                "rule": cell_meta.rule,
                "rate": cell_meta.rate,
                "group": cell_meta.group,
                "architecture": cell_meta.architecture,
                "dataset": cell_meta.dataset,
                "attack": cell_meta.attack,
                "evade_checkpoint": "evade" in cell_meta.folder,
                "summary_auroc": cell_meta.auroc,
                "recomputed_auroc": float(contributions.mean()),
                "auroc_without_low_confidence_decile": trimmed,
                "spearman_contribution_vs_confidence": float(correlation.statistic),
                "spearman_p": float(correlation.pvalue),
                "mean_clean_confidence": float(cell["clean_confidence"].mean()),
                "mean_backdoor_confidence": float(cell["backdoor_confidence"].mean()),
                "clean_psu_shape": shape["shape"],
                "clean_psu_peak_bin": shape["peak_bin"],
                "bins_below_half": int((profile.local_auroc < 0.5).sum()),
                "bins_scored": int(profile.local_auroc.notna().sum()),
            }
        )
        profile.insert(0, "folder", cell_meta.folder)
        profile.insert(1, "placement", cell_meta.placement)
        profile.insert(2, "group", cell_meta.group)
        bin_rows.append(profile)

    cell_table = pd.DataFrame(cell_rows)
    bin_table = pd.concat(bin_rows, ignore_index=True)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    print(
        f"cells loaded: {len(cell_table)} "
        f"({(cell_table.group == 'inverted').sum()} inverted, "
        f"{(cell_table.group == 'working').sum()} working) "
        f"over {cell_table.folder.nunique()} checkpoints\n"
    )

    check = (cell_table.recomputed_auroc - cell_table.summary_auroc).abs().max()
    print(f"max |recomputed AUROC - summary AUROC| = {check:.2e}\n")

    print("per cell summary by group")
    print(
        cell_table.groupby("group")
        .agg(
            n=("summary_auroc", "size"),
            auroc=("summary_auroc", "mean"),
            auroc_trimmed=("auroc_without_low_confidence_decile", "mean"),
            spearman=("spearman_contribution_vs_confidence", "mean"),
            spearman_positive=(
                "spearman_contribution_vs_confidence",
                lambda column: float((column > 0).mean()),
            ),
            clean_conf=("mean_clean_confidence", "mean"),
            backdoor_conf=("mean_backdoor_confidence", "mean"),
            bins_below_half=("bins_below_half", "mean"),
            bins_scored=("bins_scored", "mean"),
        )
        .round(4)
        .to_string()
    )

    print("\nclean PSU against confidence curve shape")
    print(pd.crosstab(cell_table.group, cell_table.clean_psu_shape).to_string())

    print("\nlocal AUROC by confidence bin, mean over cells")
    pivot = bin_table.groupby(["group", "bin"]).agg(
        confidence_low=("confidence_low", "first"),
        local_auroc=("local_auroc", "mean"),
        n_cells=("local_auroc", "count"),
        mean_clean_psu=("mean_clean_psu", "mean"),
        mean_backdoor_psu=("mean_backdoor_psu", "mean"),
        clean_share=("clean_share", "mean"),
        mean_contribution=("mean_contribution", "mean"),
        deficit_share=("auroc_deficit_share", "mean"),
    )
    print(pivot.round(4).to_string())

    print("\nfalsification checks, inverted cells only")
    inverted = cell_table[cell_table.group == "inverted"]
    inverted_bins = bin_table[bin_table.group == "inverted"]
    low = inverted_bins[inverted_bins.bin < 5]
    high = inverted_bins[inverted_bins.bin >= 5]
    print(
        f"  1. contribution rises with confidence   "
        f"mean spearman {inverted.spearman_contribution_vs_confidence.mean():+.4f}, "
        f"positive in {int((inverted.spearman_contribution_vs_confidence > 0).sum())} of {len(inverted)} cells"
    )
    print(
        f"  2. deficit concentrated below p_c = 0.5 "
        f"low half carries {low.auroc_deficit_share.sum() / inverted_bins.auroc_deficit_share.sum():.3f} "
        f"of the total AUROC deficit, high half {high.auroc_deficit_share.sum() / inverted_bins.auroc_deficit_share.sum():.3f}"
    )
    print(
        f"     mean local AUROC, low half {low.local_auroc.mean():.4f}, high half {high.local_auroc.mean():.4f}"
    )
    print(
        f"  3. clean PSU curve single peaked        "
        f"{int((inverted.clean_psu_shape == 'single_peak').sum())} of {len(inverted)} cells"
    )
    moved = inverted.auroc_without_low_confidence_decile - inverted.summary_auroc
    print(
        f"  4. dropping the least confident clean decile moves AUROC by "
        f"{moved.mean():+.4f} mean, {moved.max():+.4f} best case, "
        f"{int((inverted.auroc_without_low_confidence_decile > 0.5).sum())} cells cross 0.5"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    cell_table.to_csv(os.path.join(args.out_dir, "p2_inversion_cells.csv"), index=False)
    bin_table.to_csv(os.path.join(args.out_dir, "p2_confidence_bins.csv"), index=False)
    print(
        f"\nwrote p2_inversion_cells.csv and p2_confidence_bins.csv to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
