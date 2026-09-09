"""A label-free rule for deciding which of 2 one-sided detectors to trust.

H43 measures 2 detectors that are almost perfectly complementary: PSU covers all-to-one
(0.891) and fails on all-to-all (0.411), and predictive entropy does the reverse (0.408
and 0.761). H15 forbids taking whichever wins, because choosing needs the poison labels
the detector exists to predict. So the result is not a defence until the choice can be
made without labels.

2 candidate routers are already refuted. Both used a MAGNITUDE:

    shift-gap gating                 0.843 against 0.852 for always-PSU
    shift-destination concentration  a2o 0.469, a2a 0.422, benign 0.481, no separation

The regimes differ in SIGN, not magnitude, and the sign is what the mechanism predicts.
A defender holds clean validation data and the suspect pool being screened. Write

    d = mean PSU over the suspect pool - mean PSU over clean validation

Under all-to-one the poisoned subpopulation is MORE robust to the probe than clean data,
so it drags the pool's mean PSU DOWN and d < 0. Under all-to-all the trigger has to read
the source class before it can increment, so the backdoor pathway is more fragile than
the clean one, the poisoned subpopulation is LESS robust, and d > 0. The threshold is
therefore 0, fixed by the mechanism rather than tuned, and the rule is

    use PSU if d < 0, otherwise use entropy.

Nothing here needs the poison rate, the target class, or any poison label. The suspect
pool is simply what the defender was handed; the rate below is only used to SIMULATE a
realistic pool from cached splits.

    PYTHONPATH=. python experiments/all_to_all_entropy/router.py
"""

import argparse
import glob
import json
import os
import statistics

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)
from defences.decision import pair_clean_to_backdoor
from defences.scores import psu_ratio_from_cache

PLACEMENT = "before_attention_norm_token_mask"
TARGET_SIGMA = "sigma0.6"
PROBABILITY_FLOOR = 1e-12
BOOTSTRAP_DRAWS = 5000


def one_sided_auroc(clean: torch.Tensor, backdoor: torch.Tensor) -> float:
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    return float(
        roc_auc_score(labels, np.concatenate([-clean.numpy(), -backdoor.numpy()]))
    )


def cell_scores(folder: str, results_dir: str, checkpoints_dir: str) -> dict | None:
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    metrics_path = os.path.join(results_dir, folder, "psbd_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    metrics = json.load(open(metrics_path))
    placement = metrics.get("placements", {}).get(PLACEMENT)
    if not placement or TARGET_SIGMA not in (placement.get("matched_shift") or {}):
        return None
    args_path = os.path.join(checkpoints_dir, folder, "args.json")
    meta = json.load(open(args_path)) if os.path.exists(args_path) else {}
    if not (isinstance(meta.get("asr"), float) and meta["asr"] >= 0.5):
        return None

    rate = placement["matched_shift"][TARGET_SIGMA]["rate"]
    poison_rate = metrics.get("poison_rate") or 0.0
    manifest = json.load(open(os.path.join(psbd_dir, "split_manifest.json")))

    splits = {}
    for split in ("validation", "clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, PLACEMENT, rate, split)
        )
        floored = probs.clamp_min(PROBABILITY_FLOOR)
        splits[split] = {
            "psu": psu_ratio_from_cache(probs, labels, per_pass),
            "neg_entropy": (floored * floored.log()).sum(dim=1),
        }

    # The pool a defender actually screens: mostly clean, with the poisoned fraction
    # mixed in. Only this simulation needs the rate; the statistic itself does not.
    n_poison = int(poison_rate * len(splits["clean"]["psu"]))
    if 0 < n_poison <= len(splits["backdoor"]["psu"]):
        suspect = {
            key: torch.cat(
                [splits["clean"][key][n_poison:], splits["backdoor"][key][:n_poison]]
            )
            for key in splits["clean"]
        }
    else:
        suspect = splits["clean"]

    return {
        "cell": folder,
        "dataset": metrics.get("dataset"),
        "label_mode": metrics.get("label_mode"),
        "poison_rate": poison_rate,
        "d": float(suspect["psu"].mean() - splits["validation"]["psu"].mean()),
        "psu": one_sided_auroc(
            pair_clean_to_backdoor(splits["clean"]["psu"], manifest),
            splits["backdoor"]["psu"],
        ),
        "entropy": one_sided_auroc(
            pair_clean_to_backdoor(splits["clean"]["neg_entropy"], manifest),
            splits["backdoor"]["neg_entropy"],
        ),
    }


def routed(row: dict) -> float:
    return row["psu"] if row["d"] < 0 else row["entropy"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--out", default="results/regime_router.json")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "vit_*"))):
        folder = os.path.basename(path)
        if any(token in folder for token in ("sam_rho", "evade", "_ep", "benign")):
            continue
        try:
            row = cell_scores(folder, args.results_dir, args.checkpoints_dir)
        except Exception:  # noqa: BLE001
            continue
        if row is not None:
            rows.append(row)

    deltas = [routed(r) - r["psu"] for r in rows]
    generator = np.random.default_rng(0)
    boot = [
        statistics.mean(generator.choice(deltas, len(deltas), replace=True).tolist())
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    summary = {
        "n": len(rows),
        "always_psu": statistics.mean(r["psu"] for r in rows),
        "always_entropy": statistics.mean(r["entropy"] for r in rows),
        "routed": statistics.mean(routed(r) for r in rows),
        "oracle_max": statistics.mean(max(r["psu"], r["entropy"]) for r in rows),
        "delta_mean": statistics.mean(deltas),
        "delta_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "helped": sum(1 for d in deltas if d > 0.001),
        "hurt": sum(1 for d in deltas if d < -0.001),
    }

    print(f"n = {summary['n']}")
    for key in ("always_psu", "always_entropy", "routed", "oracle_max"):
        print(f"  {key:16s} {summary[key]:.3f}")
    print(
        f"  delta {summary['delta_mean']:+.4f} "
        f"CI [{summary['delta_ci'][0]:+.4f}, {summary['delta_ci'][1]:+.4f}]  "
        f"helped {summary['helped']} hurt {summary['hurt']}"
    )
    print("\nleave-one-dataset-out:")
    for dataset in sorted({r["dataset"] for r in rows}):
        rest = [r for r in rows if r["dataset"] != dataset]
        print(
            f"  drop {dataset:9s} n={len(rest):2d} "
            f"delta {statistics.mean(routed(r) - r['psu'] for r in rest):+.4f}"
        )

    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "cells": rows}, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
