"""What does every accumulated change buy at 1% poisoning and 5% FPR?

Reports two numbers per configuration, because they answer different questions:

  deployable   threshold = 5th percentile of CLEAN VALIDATION score, which is what
               a defender can actually compute. The FPR it achieves on the clean
               test split is reported alongside, because it is not guaranteed to
               land on 5%.
  ROC          TPR at exactly 5% FPR read off the paired clean/backdoor ROC. This
               is the number papers usually print and is an upper bound on the
               deployable one.

One-sided throughout: low score means poisoned. A configuration that lands below
chance is reported as failing, never re-signed.

The min-over-placements row ranks every split against the CLEAN VALIDATION
distribution of the same placement. An earlier version ranked over the pool
torch.cat([clean, backdoor, validation]), so the backdoor split helped set the
scale that the deployable clean-validation quantile was then read off, which
made the "deployable" number not deployable.
"""

import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import complete_rates, pair_clean_to_backdoor
from defences.scores import psu_from_cache, psu_ratio_from_cache, shift_ratio, to_rank

TARGET_FPR = 0.05

# (label, cache folder, score kind, sigma target). Each row adds one accumulated
# change to the row above it, so the deltas read as the value of that change.
STACK = [
    ("published (post_residual, absolute, s>=0.8)", "post_residual", "absolute", 0.8),
    ("+ fractional PSU (H12)", "post_residual", "fractional", 0.8),
    ("+ shift target 0.6 (H11)", "post_residual", "fractional", 0.6),
    (
        "+ position before_attention_norm (H17)",
        "before_attention_norm",
        "fractional",
        0.6,
    ),
    ("+ k=20 (H24)", "before_attention_norm_k20", "fractional", 0.6),
]

ENSEMBLE_MEMBERS = (
    "before_attention_norm",
    "before_attention",
    "before_mlp",
    "before_mlp_residual",
)


def scores(psbd_dir, name, rate, split, kind):
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    per_pass, argmax = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, name, rate, split)
    )
    build = psu_ratio_from_cache if kind == "fractional" else psu_from_cache
    return build(probs, labels, per_pass), shift_ratio(labels, argmax)


def pick_rate(psbd_dir, name, kind, target):
    """Paper's rule shape: smallest rate whose clean-validation sigma reaches target."""
    for rate in complete_rates(psbd_dir, name):
        _, sigma = scores(psbd_dir, name, rate, "validation", kind)
        if sigma is not None and sigma >= target:
            return rate
    return None


def evaluate(clean, backdoor, validation):
    """AUROC, deployable TPR at the validation 5th percentile, and ROC TPR@5%FPR."""
    c, b, v = (x.float().numpy() for x in (clean, backdoor, validation))
    labels = np.concatenate([np.zeros(len(c)), np.ones(len(b))])
    auroc = float(roc_auc_score(labels, np.concatenate([-c, -b])))

    threshold = float(np.quantile(v, TARGET_FPR))
    tpr_deploy = float((b < threshold).mean())
    fpr_achieved = float((c < threshold).mean())

    fpr, tpr, _ = roc_curve(labels, np.concatenate([-c, -b]))
    tpr_roc = float(np.interp(TARGET_FPR, fpr, tpr))
    return auroc, tpr_deploy, fpr_achieved, tpr_roc


def config_row(psbd_dir, manifest, name, kind, target):
    if not os.path.isdir(os.path.join(psbd_dir, name)):
        return None
    rate = pick_rate(psbd_dir, name, kind, target)
    if rate is None:
        return None
    clean, _ = scores(psbd_dir, name, rate, "clean", kind)
    backdoor, _ = scores(psbd_dir, name, rate, "backdoor", kind)
    validation, _ = scores(psbd_dir, name, rate, "validation", kind)
    return evaluate(pair_clean_to_backdoor(clean, manifest), backdoor, validation), rate


def ensemble_row(psbd_dir, manifest, target=0.6):
    """Per-sample MINIMUM across placements (H18), each at its own matched rate."""
    cols = {"clean": [], "backdoor": [], "validation": []}
    used = 0
    for name in ENSEMBLE_MEMBERS:
        if not os.path.isdir(os.path.join(psbd_dir, name)):
            continue
        rate = pick_rate(psbd_dir, name, "fractional", target)
        if rate is None:
            continue
        raw = {s: scores(psbd_dir, name, rate, s, "fractional")[0] for s in cols}
        raw["clean"] = pair_clean_to_backdoor(raw["clean"], manifest)
        # Rank each placement against clean validation, so members on different raw
        # scales are comparable before the min without the scored splits setting
        # that scale. Ranking over a pool containing the backdoor split would leak
        # the very split the deployable threshold is meant to be blind to.
        for split in cols:
            cols[split].append(to_rank(raw[split], raw["validation"]))
        used += 1
    if used < 2:
        return None
    reduced = {k: torch.stack(v).min(dim=0).values for k, v in cols.items()}
    return evaluate(reduced["clean"], reduced["backdoor"], reduced["validation"]), used


def main():
    results_dir = "results"
    folders = sys.argv[1:]
    print(
        f"1% poisoning, TPR at {TARGET_FPR:.0%} FPR. One-sided; below 0.5 AUROC is a failure.\n"
    )
    head = f"{'configuration':44} {'AUROC':>7} {'TPR(dep)':>9} {'FPR act':>8} {'TPR@5%':>8}"

    totals = {}
    for folder in folders:
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        if not os.path.isdir(psbd_dir):
            continue
        manifest = read_split_manifest(psbd_dir)
        print(f"### {folder}")
        print(head)
        print("-" * len(head))
        for label, name, kind, target in STACK:
            row = config_row(psbd_dir, manifest, name, kind, target)
            if row is None:
                print(f"{label:44} {'--':>7} {'--':>9} {'--':>8} {'--':>8}")
                continue
            (auroc, tpr_d, fpr_a, tpr_r), rate = row
            print(f"{label:44} {auroc:>7.3f} {tpr_d:>9.3f} {fpr_a:>8.3f} {tpr_r:>8.3f}")
            if "benign" not in folder:
                totals.setdefault(label, []).append((auroc, tpr_d, tpr_r))
        ens = ensemble_row(psbd_dir, manifest)
        if ens:
            (auroc, tpr_d, fpr_a, tpr_r), used = ens
            label = f"+ min over {used} placements (H18)"
            print(f"{label:44} {auroc:>7.3f} {tpr_d:>9.3f} {fpr_a:>8.3f} {tpr_r:>8.3f}")
            if "benign" not in folder:
                totals.setdefault("+ min over placements (H18)", []).append(
                    (auroc, tpr_d, tpr_r)
                )
        print()

    print("=" * len(head))
    print(
        f"{'MEAN over backdoored checkpoints':44} {'AUROC':>7} {'TPR(dep)':>9} {'':>8} {'TPR@5%':>8}"
    )
    print("-" * len(head))
    for label, vals in totals.items():
        a = sum(v[0] for v in vals) / len(vals)
        d = sum(v[1] for v in vals) / len(vals)
        r = sum(v[2] for v in vals) / len(vals)
        print(f"{label:44} {a:>7.3f} {d:>9.3f} {'':>8} {r:>8.3f}   n={len(vals)}")


if __name__ == "__main__":
    main()
