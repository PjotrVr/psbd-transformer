"""Which clean-validation shift ratio should the rate rule target?

The paper targets sigma >= 0.8. That target is chosen without reference to the
poison rate, and the question here is whether one target can be safe across
poison rates. Emits (placement, rate, sigma, one-sided AUROC) for every cached
cell, then scores each candidate target by the AUROC it would have selected.

Defender-legal throughout: sigma is measured on clean validation data only. The
AUROC column is the outcome being predicted, never an input to the choice.
"""

import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from defences.psbd_cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.psbd_metrics import (
    complete_rates,
    pair_clean_to_backdoor,
    psu_from_cache,
    psu_ratio_from_cache,
    shift_ratio,
)

TARGETS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def score(psbd_dir, placement, rate, split, kind):
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    per_pass, argmax = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, placement, rate, split)
    )
    build = psu_ratio_from_cache if kind == "fractional" else psu_from_cache
    return build(probs, labels, per_pass), shift_ratio(labels, argmax)


def one_sided_auroc(clean, backdoor):
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    raw = np.concatenate([-clean.float().numpy(), -backdoor.float().numpy()])
    return float(roc_auc_score(labels, raw))


def cells(results_dir, folder, placement, kind):
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    if not os.path.isdir(os.path.join(psbd_dir, placement)):
        return []
    manifest = read_split_manifest(psbd_dir)
    out = []
    for rate in complete_rates(psbd_dir, placement):
        clean, _ = score(psbd_dir, placement, rate, "clean", kind)
        backdoor, _ = score(psbd_dir, placement, rate, "backdoor", kind)
        _, sigma = score(psbd_dir, placement, rate, "validation", kind)
        if sigma is None:
            continue
        out.append(
            {
                "rate": rate,
                "sigma": sigma,
                "auroc": one_sided_auroc(
                    pair_clean_to_backdoor(clean, manifest), backdoor
                ),
            }
        )
    return sorted(out, key=lambda c: c["rate"])


def select(rows, target):
    """The paper's rule shape: smallest rate whose clean sigma reaches the target."""
    hit = [r for r in rows if r["sigma"] >= target]
    return min(hit, key=lambda r: r["rate"]) if hit else None


def main():
    results_dir = "results"
    kind = os.environ.get("SCORE", "absolute")
    placement = os.environ.get("PLACEMENT", "pre_residual_blocks_5_8")
    folders = sys.argv[1:]

    print(f"placement = {placement}   score = {kind}   (one-sided AUROC)\n")
    header = (
        "checkpoint".ljust(34) + "".join(f"  t={t:<5}" for t in TARGETS) + "   best"
    )
    print(header)
    print("-" * len(header))

    totals = {t: [] for t in TARGETS}
    for folder in folders:
        rows = cells(results_dir, folder, placement, kind)
        if not rows:
            print(f"{folder:34}  (no cache)")
            continue
        line = folder.ljust(34)
        for t in TARGETS:
            chosen = select(rows, t)
            if chosen is None:
                line += "   --   "
            else:
                line += f"  {chosen['auroc']:.3f} "
                if "benign" not in folder:
                    totals[t].append(chosen["auroc"])
        line += f"   {max(r['auroc'] for r in rows):.3f}"
        print(line)

    print("-" * len(header))
    summary = "mean over backdoored".ljust(34)
    for t in TARGETS:
        v = totals[t]
        summary += f"  {sum(v) / len(v):.3f} " if v else "   --   "
    print(summary)
    inverted = "cells below chance".ljust(34)
    for t in TARGETS:
        v = totals[t]
        inverted += (
            f"  {sum(1 for x in v if x < 0.5):>3}/{len(v):<3}" if v else "   --   "
        )
    print(inverted)

    print("\nsigma-vs-rate detail")
    for folder in folders:
        rows = cells(results_dir, folder, placement, kind)
        if rows:
            detail = " ".join(
                f"{r['rate']:g}:s{r['sigma']:.2f}/a{r['auroc']:.2f}" for r in rows
            )
            print(f"  {folder:34} {detail}")


if __name__ == "__main__":
    main()
