"""Is PSBD's uncovered case covered by a statistic its own pipeline already caches?

PSBD does not work on all-to-all. H5 and docs/all-to-all-inversion.md establish that on
both architectures, including ResNet-18 on the original authors' own recipe (TPR 0.210
against FPR 0.192). The mechanism is structural: PSBD needs the trigger to be a
CONSTANT, content-independent shortcut, and (y + 1) mod K forces the model to read the
source class before it can increment, so the backdoor pathway inherits and then exceeds
the clean pathway's fragility.

That same content dependence has a second consequence nobody has read off. Under
all-to-all the poisoned logit must beat the TRUE SOURCE class rather than an arbitrary
runner-up, so the margin is contested. That is why all-to-all costs ASR (0.842 against
1.000 on CIFAR-10) and it should also make triggered inputs measurably LESS confident.
Confidence is not something PSBD measures, but its cache already holds it: every
baseline_<split>.pt stores the full (N, num_classes) no-dropout softmax and only the
probability of the argmax class is ever used.

So this compares, on exactly the same splits and the same pairing PSBD uses:

    psu_ratio          the deployed detector, k stochastic passes
    neg_entropy        -H(p) of the single deterministic forward pass
    confidence         max_c p_c, the same pass
    logit_margin       log p_1 - log p_2, which equals z_1 - z_2 exactly

Direction is fixed a priori and never chosen from an AUROC (H15): a contested margin
means a flatter posterior, so entropy is HIGHER and neg_entropy LOWER on a triggered
input, and low means poisoned as everywhere else in this repo.

CPU only. No GPU, no model, no dataset: it reads the cache.

    PYTHONPATH=. python experiments/all_to_all_entropy/measure.py
"""

import argparse
import json
import os
import statistics

import numpy as np
import torch
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
PROBABILITY_FLOOR = 1e-12
QUANTILES = (0.01, 0.05, 0.25)


def deterministic_scores(probs: torch.Tensor) -> dict[str, torch.Tensor]:
    """The 3 statistics recoverable from a cached no-dropout softmax row.

    The logit margin is exact rather than approximate: softmax is shift-invariant, so
    log p_1 - log p_2 = z_1 - z_2 identically, and the logits themselves are not needed.
    """
    floored = probs.clamp_min(PROBABILITY_FLOOR)
    top2 = probs.topk(2, dim=1).values.clamp_min(PROBABILITY_FLOOR)
    return {
        "neg_entropy": (floored * floored.log()).sum(dim=1),
        "confidence": top2[:, 0],
        "logit_margin": top2[:, 0].log() - top2[:, 1].log(),
    }


def report(
    validation: torch.Tensor, clean: torch.Tensor, backdoor: torch.Tensor
) -> dict:
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    scores = np.concatenate([-clean.numpy(), -backdoor.numpy()])
    out = {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }
    for quantile in QUANTILES:
        threshold = float(np.quantile(validation.numpy(), quantile))
        out[f"q{quantile:.2f}"] = {
            "tpr": float((backdoor < threshold).float().mean()),
            "fpr": float((clean < threshold).float().mean()),
        }
    return out


def analyse(folder: str, results_dir: str) -> dict | None:
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    metrics_path = os.path.join(results_dir, folder, "psbd_metrics.json")
    if not os.path.isdir(os.path.join(psbd_dir, PLACEMENT)):
        return None
    if not os.path.exists(metrics_path):
        return None
    metrics = json.load(open(metrics_path))
    placement = metrics.get("placements", {}).get(PLACEMENT)
    if not placement or TARGET_SIGMA not in (placement.get("matched_shift") or {}):
        return None
    rate = placement["matched_shift"][TARGET_SIGMA]["rate"]
    manifest = json.load(open(os.path.join(psbd_dir, "split_manifest.json")))

    scores: dict[str, dict[str, torch.Tensor]] = {}
    for split in ("validation", "clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, PLACEMENT, rate, split)
        )
        scores[split] = deterministic_scores(probs)
        scores[split]["psu_ratio"] = psu_ratio_from_cache(probs, labels, per_pass)

    out = {
        "folder_name": folder,
        "dataset": metrics.get("dataset"),
        "attack": metrics.get("attack"),
        "label_mode": metrics.get("label_mode"),
        "poison_rate": metrics.get("poison_rate"),
        "rate": rate,
        "statistics": {},
    }
    for name in scores["validation"]:
        out["statistics"][name] = report(
            scores["validation"][name],
            pair_clean_to_backdoor(scores["clean"][name], manifest),
            scores["backdoor"][name],
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--out", default="results/all_to_all_entropy.json")
    args = parser.parse_args()

    rows = []
    for folder in sorted(os.listdir(args.results_dir)):
        if not folder.startswith("vit_"):
            continue
        if "sam_rho" in folder or "evade" in folder or "_ep" in folder:
            continue
        row = analyse(folder, args.results_dir)
        if row is None:
            continue
        args_path = os.path.join(args.checkpoints_dir, folder, "args.json")
        meta = json.load(open(args_path)) if os.path.exists(args_path) else {}
        row["asr"] = meta.get("asr")
        rows.append(row)

    groups = {
        "all_to_all": [r for r in rows if "a2a" in r["folder_name"]],
        "benign": [r for r in rows if "benign" in r["folder_name"]],
        "all_to_one": [
            r
            for r in rows
            if "a2a" not in r["folder_name"]
            and "benign" not in r["folder_name"]
            and isinstance(r["asr"], float)
            and r["asr"] >= 0.5
        ],
    }
    names = ("psu_ratio", "neg_entropy", "confidence", "logit_margin")
    print(f"{'group':14s} {'n':>3s} " + " ".join(f"{n:>13s}" for n in names))
    summary = {}
    for group, members in groups.items():
        if not members:
            continue
        means = {
            n: statistics.mean(m["statistics"][n]["auroc"] for m in members)
            for n in names
        }
        summary[group] = {"n": len(members), "auroc": means}
        print(
            f"{group:14s} {len(members):3d} "
            + " ".join(f"{means[n]:13.3f}" for n in names)
        )

    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "cells": rows}, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
