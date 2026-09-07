"""Does the SHAPE of a per-sample sensitivity profile beat its mean?

H18 claims the backdoor signature is concentration of sensitivity, not its
magnitude. The full test needs per-unit masking, but a free proxy exists: each
cached placement is a different way of removing capacity, so the vector of PSU
across placements is a coarse sensitivity profile, already on disk.

Every placement contributes at its own sigma-matched rate, so the profile
compares placements at equal measured disturbance rather than at a shared p,
which would only measure which placement perturbs hardest.

If concentration statistics beat the mean here, the idea survives its cheapest
test and the per-unit version is worth the GPU time. If they do not, it dies for
free.
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from defences.psbd_cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.psbd_metrics import pair_clean_to_backdoor, psu_ratio_from_cache

SURFACE = json.load(open("scratch/surface.json"))
SIGMA = 0.6


def matched_rate(folder, placement):
    hit = [
        r
        for r in SURFACE[folder]
        if r["placement"] == placement
        and r["sigma"] is not None
        and r["sigma"] >= SIGMA
    ]
    return min(hit, key=lambda r: r["rate"])["rate"] if hit else None


def psu(psbd_dir, placement, rate, split):
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    per_pass, _ = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, placement, rate, split)
    )
    return psu_ratio_from_cache(probs, labels, per_pass).float()


def gini(profile):
    """Concentration in [0, 1]. 0 means every unit matters equally."""
    x = torch.sort(profile.clamp_min(0), dim=1).values
    n = x.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype)
    total = x.sum(dim=1).clamp_min(1e-9)
    return (2 * (index * x).sum(dim=1) / (n * total)) - (n + 1) / n


def normalized_entropy(profile):
    """1 means flat, 0 means all the sensitivity sits on one placement."""
    p = profile.clamp_min(0) + 1e-9
    p = p / p.sum(dim=1, keepdim=True)
    return -(p * p.log()).sum(dim=1) / np.log(p.shape[1])


def summaries(profile):
    """Every scalar reduction of the profile worth comparing, PSBD's mean included."""
    return {
        "mean (PSBD)": profile.mean(dim=1),
        "max": profile.max(dim=1).values,
        "min": profile.min(dim=1).values,
        "std": profile.std(dim=1),
        "gini": gini(profile),
        "neg_entropy": -normalized_entropy(profile),
        "top2_mass": profile.clamp_min(0).topk(2, dim=1).values.sum(dim=1)
        / profile.clamp_min(0).sum(dim=1).clamp_min(1e-9),
    }


def auroc(clean, backdoor):
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    return float(roc_auc_score(labels, np.concatenate([-clean, -backdoor])))


def main():
    results_dir = "results"
    folders = sys.argv[1:]
    placements = None
    rows = {}
    for folder in folders:
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        manifest = read_split_manifest(psbd_dir)
        available = sorted(
            p
            for p in {r["placement"] for r in SURFACE.get(folder, [])}
            if matched_rate(folder, p) is not None
        )
        if len(available) < 6:
            continue
        placements = placements or available
        use = [p for p in placements if p in available]

        clean_cols, backdoor_cols = [], []
        for placement in use:
            rate = matched_rate(folder, placement)
            clean_cols.append(
                pair_clean_to_backdoor(
                    psu(psbd_dir, placement, rate, "clean"), manifest
                )
            )
            backdoor_cols.append(psu(psbd_dir, placement, rate, "backdoor"))
        clean = torch.stack(clean_cols, dim=1)
        backdoor = torch.stack(backdoor_cols, dim=1)

        rows[folder] = {
            name: auroc(c.numpy(), b.numpy())
            for (name, c), (_, b) in zip(
                summaries(clean).items(), summaries(backdoor).items()
            )
        }
        rows[folder]["_n"] = len(use)

    names = [k for k in next(iter(rows.values())) if not k.startswith("_")]
    head = f"{'checkpoint':34} {'k':>3} " + " ".join(f"{n:>12}" for n in names)
    print(f"profile over placements at matched clean sigma >= {SIGMA}, one-sided AUROC")
    print("low score = poisoned for every column, nothing is flipped\n")
    print(head)
    print("-" * len(head))
    for folder, row in sorted(rows.items()):
        cells = " ".join(f"{row[n]:>12.3f}" for n in names)
        print(f"{folder:34} {row['_n']:>3} {cells}")
    print("-" * len(head))
    backdoored = {f: r for f, r in rows.items() if "benign" not in f and "a2a" not in f}
    means = "mean over backdoored".ljust(34) + f" {'':>3} "
    means += " ".join(
        f"{sum(r[n] for r in backdoored.values()) / len(backdoored):>12.3f}"
        for n in names
    )
    print(means)


if __name__ == "__main__":
    main()
