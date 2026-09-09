"""Is PSBD's low-poison-rate failure a property of the rate, or of the operating point?

For every cached (placement, dropout rate) on a checkpoint, print the ONE-SIDED
AUROC of PSU together with the clean-validation shift ratio. If the surface is
below 0.5 everywhere, PSBD's premise genuinely fails at that poison rate. If it
crosses 0.5 at small p, the premise holds and only the rate-selection rule is
broken, which is a much cheaper thing to fix.
"""

import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import complete_rates, pair_clean_to_backdoor
from defences.scores import psu_from_cache, psu_ratio_from_cache, shift_ratio


def score(psbd_dir, placement, rate, split, kind):
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    per_pass, argmax = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, placement, rate, split)
    )
    build = psu_ratio_from_cache if kind == "fractional" else psu_from_cache
    return build(probs, labels, per_pass), shift_ratio(labels, argmax)


def one_sided_auroc(clean, backdoor):
    """Low score = poisoned, exactly as the theory states. Never flipped."""
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    raw = np.concatenate([-clean.float().numpy(), -backdoor.float().numpy()])
    return float(roc_auc_score(labels, raw))


def surface(results_dir, folder, kind):
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    if not os.path.isdir(psbd_dir):
        return None
    manifest = read_split_manifest(psbd_dir)
    rows = []
    for placement in sorted(os.listdir(psbd_dir)):
        if not os.path.isdir(os.path.join(psbd_dir, placement)):
            continue
        for rate in complete_rates(psbd_dir, placement):
            clean, _ = score(psbd_dir, placement, rate, "clean", kind)
            backdoor, _ = score(psbd_dir, placement, rate, "backdoor", kind)
            _, sigma = score(psbd_dir, placement, rate, "validation", kind)
            rows.append(
                {
                    "placement": placement,
                    "rate": rate,
                    "auroc": one_sided_auroc(
                        pair_clean_to_backdoor(clean, manifest), backdoor
                    ),
                    "sigma": sigma,
                }
            )
    return rows


def main():
    results_dir = "results"
    kind = os.environ.get("SCORE", "absolute")
    folders = sys.argv[1:]
    for folder in folders:
        rows = surface(results_dir, folder, kind)
        if not rows:
            print(f"\n### {folder}: no cache")
            continue
        best = max(rows, key=lambda r: r["auroc"])
        adaptive = [r for r in rows if r["sigma"] is not None and r["sigma"] >= 0.8]
        adaptive = min(adaptive, key=lambda r: r["rate"]) if adaptive else None
        print(f"\n### {folder}   ({kind} PSU, one-sided)")
        print(
            f"  best over surface : {best['auroc']:.3f}  at {best['placement']} p={best['rate']:g} (sigma {best['sigma']:.3f})"
        )
        if adaptive:
            print(
                f"  paper adaptive rule: {adaptive['auroc']:.3f}  at {adaptive['placement']} p={adaptive['rate']:g} (sigma {adaptive['sigma']:.3f})"
            )
        above = sum(1 for r in rows if r["auroc"] > 0.5)
        print(f"  cells above chance : {above}/{len(rows)}")

        # The shape that matters: AUROC against dropout rate, per placement.
        for placement in sorted({r["placement"] for r in rows}):
            sub = sorted(
                (r for r in rows if r["placement"] == placement),
                key=lambda r: r["rate"],
            )
            cells = " ".join(f"{r['rate']:g}:{r['auroc']:.2f}" for r in sub)
            print(f"    {placement:28} {cells}")


if __name__ == "__main__":
    main()
