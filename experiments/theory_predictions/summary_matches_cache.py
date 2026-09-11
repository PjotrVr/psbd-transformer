"""Does results/detection_summary.csv still agree with the cached tensors?

Found while running prediction 2. Recomputing a cell's AUROC from
results/<folder>/psbd/ reproduced the summary bit for bit on 85 of 87 sampled
cells, and disagreed on 2, both of them gaussian. This measures how far that
goes, because every published number in this project is read from the summary and
nothing had ever checked the summary against the tensors it was derived from.

experiments/cache_integrity/ audits the cache against itself (shapes, manifests,
finiteness). It does not audit psbd_metrics.json against the cache, which is the
step where a regenerated cache and a stale stage-2 record diverge silently.

Read only. Reports, fixes nothing.

Example
    PYTHONPATH=. .venv/bin/python experiments/theory_predictions/summary_matches_cache.py --cells 400
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import pair_clean_to_backdoor
from defences.scores import psu_from_cache
from experiments._paths import experiment_results_dir

# A disagreement larger than this is not float reassociation.
TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--summary", default="results/detection_summary.csv")
    parser.add_argument(
        "--out-dir", default=experiment_results_dir("theory_predictions")
    )
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def recomputed_auroc(
    results_dir: str, folder: str, placement: str, rate: float
) -> float | None:
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    paths = {
        split: dropout_pass_path(psbd_dir, placement, rate, split)
        for split in ("clean", "backdoor")
    }
    if not all(os.path.exists(path) for path in paths.values()):
        return None

    manifest = read_split_manifest(psbd_dir)
    scores = {}
    for split, path in paths.items():
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        passes, _ = load_dropout_pass_probs(path)
        scores[split] = psu_from_cache(probs, labels, passes)

    clean = pair_clean_to_backdoor(scores["clean"], manifest).numpy()
    backdoor = scores["backdoor"].numpy()
    values = np.concatenate([-clean, -backdoor])
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    return float(roc_auc_score(labels, values))


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    plain = summary[summary.variant.isna() & summary.rate.notna()]
    plain = plain.drop_duplicates(subset=["folder", "placement", "rate"])

    # Stratified by operator so a rare operator is not squeezed out of a uniform
    # draw, which is how the gaussian disagreement would have been missed.
    sample = plain.groupby("operator", group_keys=False).sample(
        frac=1.0, random_state=args.seed
    )
    per_operator = max(args.cells // plain.operator.nunique(), 10)
    sample = sample.groupby("operator", group_keys=False).head(per_operator)

    rows = []
    for _, meta in sample.iterrows():
        value = recomputed_auroc(
            args.results_dir, meta.folder, meta.placement, meta.rate
        )
        if value is None:
            continue
        rows.append(
            {
                "folder": meta.folder,
                "placement": meta.placement,
                "operator": meta.operator,
                "rule": meta.rule,
                "rate": meta.rate,
                "summary_auroc": meta.auroc,
                "cache_auroc": value,
                "gap": abs(value - meta.auroc),
            }
        )

    table = pd.DataFrame(rows)
    table["disagrees"] = table.gap > TOLERANCE

    pd.set_option("display.width", 250)
    print(
        f"cells recomputed from cache: {len(table)} over {table.folder.nunique()} checkpoints\n"
    )
    print(
        table.groupby("operator")
        .agg(
            n=("gap", "size"),
            disagreeing=("disagrees", "sum"),
            share=("disagrees", "mean"),
            max_gap=("gap", "max"),
            mean_gap_when_disagreeing=(
                "gap",
                lambda column: (
                    float(column[column > TOLERANCE].mean())
                    if (column > TOLERANCE).any()
                    else 0.0
                ),
            ),
        )
        .round(6)
        .to_string()
    )

    worst = table.nlargest(15, "gap")
    print("\nlargest disagreements")
    print(
        worst[
            [
                "folder",
                "placement",
                "rule",
                "rate",
                "summary_auroc",
                "cache_auroc",
                "gap",
            ]
        ]
        .round(6)
        .to_string()
    )

    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "summary_vs_cache.csv"), index=False)
    print(f"\nwrote summary_vs_cache.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
