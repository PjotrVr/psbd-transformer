"""Rank perturbation operators at matched disturbance, per attack.

Comparing operators at a shared rate would only measure which one perturbs
hardest, so every comparison here is at a matched clean-validation shift ratio.
An operator that never reaches the target sigma on a checkpoint is reported as
"--" rather than dropped, because an operator that cannot reach the operating
point is a different failure from one that reaches it and separates badly.

One-sided throughout: low PSU means poisoned, and a value below 0.5 is printed as
the failure it is. Nothing is flipped.
"""

import os
import sys
from collections import defaultdict

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
from defences.scores import psu_ratio_from_cache, shift_ratio

# Every operator name that can appear as a folder suffix, longest first so
# "channel_mask" is not mistaken for a position ending in "mask".
OPERATORS = (
    "channel_mask",
    "gain_scale",
    "token_mask",
    "head_mask",
    "droppath",
    "gaussian",
    "scale_up",
)
SIGMA_TARGETS = (0.2, 0.4, 0.6, 0.8)


def split_folder(name: str) -> tuple[str, str, int]:
    """(position, operator, k) from a cache folder name.

    Bare names are the paper's dropout at k=3, which is the whole reason the
    naming keeps them bare; see psbd_dropout_sweep.cache_config_name.
    """
    passes = 3
    if "_k" in name:
        stem, _, tail = name.rpartition("_k")
        if tail.isdigit():
            name, passes = stem, int(tail)
    for operator in OPERATORS:
        if name.endswith(f"_{operator}"):
            return name[: -len(operator) - 1], operator, passes
    return name, "dropout", passes


def cells(psbd_dir, manifest, folder_name):
    out = []
    for rate in complete_rates(psbd_dir, folder_name):
        scores, sigmas = {}, {}
        for split in ("validation", "clean", "backdoor"):
            probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
            per_pass, argmax = load_dropout_pass_probs(
                dropout_pass_path(psbd_dir, folder_name, rate, split)
            )
            scores[split] = psu_ratio_from_cache(probs, labels, per_pass)
            sigmas[split] = shift_ratio(labels, argmax)
        clean = pair_clean_to_backdoor(scores["clean"], manifest).float().numpy()
        backdoor = scores["backdoor"].float().numpy()
        labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
        out.append(
            {
                "rate": rate,
                "sigma": sigmas["validation"],
                "auroc": float(
                    roc_auc_score(labels, np.concatenate([-clean, -backdoor]))
                ),
            }
        )
    return sorted(out, key=lambda c: c["rate"])


def at_target(rows, target):
    """Paper's rule shape: smallest rate whose clean sigma reaches the target."""
    hit = [r for r in rows if r["sigma"] is not None and r["sigma"] >= target]
    return min(hit, key=lambda r: r["rate"]) if hit else None


def main():
    results_dir = "results"
    target = float(os.environ.get("SIGMA", "0.6"))
    folders = sys.argv[1:]

    table = defaultdict(dict)
    reach = {}
    for folder in folders:
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        if not os.path.isdir(psbd_dir):
            continue
        manifest = read_split_manifest(psbd_dir)
        for name in sorted(os.listdir(psbd_dir)):
            if not os.path.isdir(os.path.join(psbd_dir, name)):
                continue
            position, operator, passes = split_folder(name)
            if passes != 3:
                continue
            rows = cells(psbd_dir, manifest, name)
            if not rows:
                continue
            chosen = at_target(rows, target)
            key = (operator, position)
            table[key][folder] = chosen["auroc"] if chosen else None
            reach[key] = reach.get(key, 0) + (1 if chosen else 0)

    if not table:
        raise SystemExit("no cached operator folders found yet")

    attacks = [f for f in folders if f in {k for v in table.values() for k in v}]
    head = (
        f"{'operator':14} {'position':22} "
        + " ".join(f"{f.replace('vit_cifar10_', '')[:13]:>13}" for f in attacks)
        + f" {'mean':>7} {'n':>3}"
    )
    print(f"matched at clean-validation sigma >= {target}, fractional PSU, one-sided")
    print("a value below 0.5 is a failure and is printed as such\n")
    print(head)
    print("-" * len(head))

    ranked = []
    for key in sorted(table):
        operator, position = key
        values = [table[key].get(f) for f in attacks]
        scored = [
            v
            for f, v in zip(attacks, values)
            if v is not None and "benign" not in f and "badnet_a2a" not in f
        ]
        mean = sum(scored) / len(scored) if scored else float("nan")
        ranked.append((mean, operator, position, values))

    for mean, operator, position, values in sorted(ranked, reverse=True):
        cells_text = " ".join(
            f"{v:>13.3f}" if v is not None else f"{'--':>13}" for v in values
        )
        print(f"{operator:14} {position:22} {cells_text} {mean:>7.3f}")


if __name__ == "__main__":
    main()
