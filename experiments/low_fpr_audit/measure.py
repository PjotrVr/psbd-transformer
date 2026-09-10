"""What does PSBD actually deliver at a false-positive budget a defender would accept?

Every headline in this project is an AUROC. AUROC integrates over the whole ROC curve,
including false-positive rates no operator would run at, and at 1% poisoning the positive
class is rare enough that the integral is dominated by a region that never gets deployed.

This reads the same cached tensors the headline tables read and reports, per cell:

    AUROC      what is currently reported
    AUPRC      the rare-positive metric, computed nowhere else in this repo
    TPR at 1%, 5%, 10% and 25% FPR, at a threshold set on clean validation only

A cell with high AUROC and near-zero TPR at 1% FPR is not a detector a defender can use,
and the gap between those 2 numbers is the thing this measures.

CPU only. No GPU, no model.

    PYTHONPATH=. python experiments/low_fpr_audit/measure.py
"""

import argparse
import glob
import json
import os
import statistics

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

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
QUANTILES = (0.01, 0.05, 0.10, 0.25)


def audit(folder: str, results_dir: str, checkpoints_dir: str) -> dict | None:
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

    rate = placement["matched_shift"][TARGET_SIGMA]["rate"]
    manifest = json.load(open(os.path.join(psbd_dir, "split_manifest.json")))
    scores = {}
    for split in ("validation", "clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, PLACEMENT, rate, split)
        )
        scores[split] = psu_ratio_from_cache(probs, labels, per_pass)

    clean = pair_clean_to_backdoor(scores["clean"], manifest)
    backdoor = scores["backdoor"]
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    signed = np.concatenate([-clean.numpy(), -backdoor.numpy()])

    row = {
        "cell": folder,
        "dataset": metrics.get("dataset"),
        "attack": metrics.get("attack"),
        "label_mode": metrics.get("label_mode"),
        "poison_rate": metrics.get("poison_rate"),
        "asr": meta.get("asr"),
        "auroc": float(roc_auc_score(labels, signed)),
        # The rare-positive metric. At 1% poisoning the prevalence a defender faces is
        # far below the 50/50 this paired evaluation uses, so AUPRC here is still
        # optimistic; it is reported because it is at least sensitive to the tail.
        "auprc": float(average_precision_score(labels, signed)),
    }
    for quantile in QUANTILES:
        threshold = float(np.quantile(scores["validation"].numpy(), quantile))
        row[f"tpr@{quantile:.2f}"] = float((backdoor < threshold).float().mean())
        row[f"fpr@{quantile:.2f}"] = float((clean < threshold).float().mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--out", default="results/low_fpr_audit.json")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "vit_*"))):
        folder = os.path.basename(path)
        if any(token in folder for token in ("sam_rho", "evade", "_ep")):
            continue
        try:
            row = audit(folder, args.results_dir, args.checkpoints_dir)
        except Exception:  # noqa: BLE001
            continue
        if row is not None:
            rows.append(row)

    working = [
        r
        for r in rows
        if "benign" not in r["cell"]
        and r["label_mode"] != "all_to_all"
        and isinstance(r["asr"], float)
        and r["asr"] >= 0.5
    ]
    print(
        f"{'poison':>7s} {'n':>3s} {'AUROC':>7s} {'AUPRC':>7s} "
        f"{'TPR@1%':>7s} {'TPR@5%':>7s} {'TPR@10%':>8s} {'TPR@25%':>8s}"
    )
    summary = {}
    for poison_rate in (0.01, 0.05, 0.10):
        group = [r for r in working if r["poison_rate"] == poison_rate]
        if not group:
            continue
        record = {
            "n": len(group),
            "auroc": statistics.mean(r["auroc"] for r in group),
            "auprc": statistics.mean(r["auprc"] for r in group),
        }
        for quantile in QUANTILES:
            record[f"tpr@{quantile:.2f}"] = statistics.mean(
                r[f"tpr@{quantile:.2f}"] for r in group
            )
        summary[f"{poison_rate}"] = record
        print(
            f"{poison_rate:7.0%} {len(group):3d} {record['auroc']:7.3f} "
            f"{record['auprc']:7.3f} {record['tpr@0.01']:7.3f} "
            f"{record['tpr@0.05']:7.3f} {record['tpr@0.10']:8.3f} "
            f"{record['tpr@0.25']:8.3f}"
        )

    collapsed = [r for r in working if r["auroc"] >= 0.85 and r["tpr@0.01"] < 0.05]
    print(
        f"\ncells with AUROC >= 0.85 but TPR@1%FPR < 0.05: "
        f"{len(collapsed)} of {len(working)}"
    )
    for row in sorted(collapsed, key=lambda r: -r["auroc"]):
        print(
            f"   {row['cell']:34s} auroc={row['auroc']:.3f} auprc={row['auprc']:.3f} "
            f"tpr@1%={row['tpr@0.01']:.3f} tpr@5%={row['tpr@0.05']:.3f}"
        )

    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "cells": rows}, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
