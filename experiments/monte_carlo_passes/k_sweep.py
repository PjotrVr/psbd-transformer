"""H24 free half: does AUROC rise with Monte Carlo passes, and more so at 1%?

The cache stores per-pass tracked-class probabilities as (k, N), so k = 1 and
k = 2 are subsets already on disk. If the increment from k = 1 to k = 3 is not
larger at 1% poisoning than at 10%, estimator noise is not what limits the
low-poison-rate case and the k = 20 jobs are not worth submitting.
"""

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from defences.psbd_cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.psbd_metrics import pair_clean_to_backdoor

SURFACE = json.load(open("scratch/surface.json"))
PLACEMENT = os.environ.get("PLACEMENT", "before_attention_norm")
SIGMA = 0.6


def matched_rate(folder):
    hit = [
        r
        for r in SURFACE.get(folder, [])
        if r["placement"] == PLACEMENT
        and r["sigma"] is not None
        and r["sigma"] >= SIGMA
    ]
    return min(hit, key=lambda r: r["rate"])["rate"] if hit else None


def psu_at_k(psbd_dir, rate, split, k):
    """PSU using only the first k of the cached passes, fractional score."""
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    per_pass, _ = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, PLACEMENT, rate, split)
    )
    tracked = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    return 1.0 - per_pass[:k].mean(dim=0) / tracked.clamp_min(1e-6)


def auroc(clean, backdoor):
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    return float(
        roc_auc_score(labels, np.concatenate([-clean.numpy(), -backdoor.numpy()]))
    )


def poison_tag(folder):
    for tag, label in (("_0_01", "1%"), ("_0_05", "5%"), ("_0_1", "10%")):
        if folder.endswith(tag):
            return label
    return "benign" if "benign" in folder else None


def main():
    results_dir = "results"
    rows, by_rate = [], defaultdict(list)
    for folder in sorted(SURFACE):
        tag = poison_tag(folder)
        rate = matched_rate(folder)
        if tag is None or rate is None or "badnet_a2a" in folder:
            continue
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        manifest = read_split_manifest(psbd_dir)
        values = {}
        for k in (1, 2, 3):
            clean = pair_clean_to_backdoor(
                psu_at_k(psbd_dir, rate, "clean", k), manifest
            )
            values[k] = auroc(clean, psu_at_k(psbd_dir, rate, "backdoor", k))
        delta = values[3] - values[1]
        rows.append((folder, tag, values, delta))
        if tag != "benign":
            by_rate[tag].append(delta)

    head = f"{'checkpoint':34} {'pr':>6} {'k=1':>7} {'k=2':>7} {'k=3':>7} {'k3-k1':>8}"
    print(f"placement {PLACEMENT}, rate matched at clean sigma >= {SIGMA}")
    print("fractional PSU, one-sided AUROC, all-to-all excluded\n")
    print(head)
    print("-" * len(head))
    for folder, tag, values, delta in rows:
        print(
            f"{folder:34} {tag:>6} {values[1]:>7.3f} {values[2]:>7.3f} "
            f"{values[3]:>7.3f} {delta:>+8.3f}"
        )
    print("-" * len(head))
    for tag in ("1%", "5%", "10%"):
        if by_rate.get(tag):
            v = by_rate[tag]
            print(
                f"  mean k3-k1 at {tag:>3} poisoning: {sum(v) / len(v):+.4f} "
                f"over {len(v)} checkpoints"
            )

    one, ten = by_rate.get("1%"), by_rate.get("10%")
    if one and ten:
        m1, m10 = sum(one) / len(one), sum(ten) / len(ten)
        print(
            f"\nVERDICT: increment at 1% is {'LARGER' if m1 > m10 else 'NOT larger'} "
            f"than at 10% ({m1:+.4f} vs {m10:+.4f})"
        )
        print(
            "  -> k=20 jobs are justified"
            if m1 > m10 + 0.002
            else "  -> noise is not the binding constraint; do NOT submit k=20 jobs"
        )


if __name__ == "__main__":
    main()
