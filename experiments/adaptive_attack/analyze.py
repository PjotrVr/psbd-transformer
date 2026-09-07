"""Analyze adaptive attacker results: compare evasive vs non-evasive checkpoints.

For each evasive checkpoint, computes sigma-matched AUROC at 4 operators and
compares to the matching non-evasive baseline. Reports:
1. ASR and CA preservation (success criteria: ASR > 0.9, CA within 2 points)
2. Detection on the probed operator (does AUROC drop toward 0.5?)
3. Transfer: does evasion against one operator transfer to others?
4. Rate adaptation: does the sigma-matched rate shift higher?

Runs entirely on CPU using cached .pt files from the PSBD sweep.

Example
    python experiments/adaptive_attack/analyze.py
    python experiments/adaptive_attack/analyze.py --architecture vit --dataset cifar100
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

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
    psu_ratio_from_cache,
    shift_ratio,
)


RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"

ATTACKS = [
    "badnet_a2o",
    "blend",
    "wanet",
    "lc",
    "adaptive_blend",
    "sig",
    "lf",
    "bpp",
    "tact",
    "badnet_a2a",
]

RATE_TAGS = [("0_1", 0.10), ("0_05", 0.05), ("0_01", 0.01)]

VIT_OPERATORS = [
    ("before_attention_norm_token_mask", "token_mask@ban", True),
    ("before_attention_norm", "dropout@ban", False),
    ("mlp_norm_out_gain_scale", "gain_scale@mno", False),
    ("before_mlp_gaussian", "gaussian@bm", False),
]

SWIN_OPERATORS = [
    ("before_attention_norm", "dropout@ban", True),
    ("before_attention_norm_token_mask", "token_mask@ban", False),
    ("mlp_norm_out_gain_scale", "gain_scale@mno", False),
    ("pre_residual", "dropout@pre_res", False),
]

SIGMA_TARGET = 0.6


def read_asr_ca(ckpt_folder):
    """Read ASR and CA, trying metrics.json first then falling back to args.json.

    Evasive checkpoints record ASR/CA in args.json at training time. Baseline
    checkpoints also have metrics.json written by baseline_detect.py. Both use
    the same field names ("asr", "clean_accuracy").
    """
    metrics_path = os.path.join(RESULTS_DIR, ckpt_folder, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        asr, ca = m.get("asr"), m.get("clean_accuracy")
        if asr is not None:
            return asr, ca

    args_path = os.path.join(CHECKPOINTS_DIR, ckpt_folder, "args.json")
    if os.path.exists(args_path):
        with open(args_path) as f:
            meta = json.load(f)
        return meta.get("asr"), meta.get("clean_accuracy")

    return None, None


def sigma_matched_auroc(psbd_dir, name, sigma_target=SIGMA_TARGET):
    """Compute one-sided AUROC at the sigma-matched rate."""
    manifest = read_split_manifest(psbd_dir)
    if manifest is None:
        return None

    rates = complete_rates(psbd_dir, name)
    if not rates:
        return None

    chosen = None
    for rate in rates:
        val_path = dropout_pass_path(psbd_dir, name, rate, "validation")
        if not os.path.exists(val_path):
            continue
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
        _, argmax = load_dropout_pass_probs(val_path)
        sigma = shift_ratio(labels, argmax)
        if sigma is not None and sigma >= sigma_target:
            chosen = rate
            break

    if chosen is None:
        return None

    scored = {}
    for split in ("clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, name, chosen, split)
        )
        scored[split] = psu_ratio_from_cache(probs, labels, per_pass)

    clean = pair_clean_to_backdoor(scored["clean"], manifest).float().numpy()
    backdoor = scored["backdoor"].float().numpy()

    y = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    raw = np.concatenate([-clean, -backdoor])

    return {
        "rate": chosen,
        "auroc": float(roc_auc_score(y, raw)),
    }


def analyze_one(arch, dataset, attack, rate_tag, rate_float, operators):
    evade_folder = f"{arch}_{dataset}_{attack}_{rate_tag}_evade_l1"
    base_folder = f"{arch}_{dataset}_{attack}_{rate_tag}"

    evade_psbd = os.path.join(RESULTS_DIR, evade_folder, "psbd")
    base_psbd = os.path.join(RESULTS_DIR, base_folder, "psbd")

    if not os.path.isdir(evade_psbd):
        return None

    base_asr, base_ca = read_asr_ca(base_folder)
    evade_asr, evade_ca = read_asr_ca(evade_folder)

    row = {
        "arch": arch,
        "dataset": dataset,
        "attack": attack,
        "rate": rate_float,
        "base_asr": base_asr,
        "base_ca": base_ca,
        "evade_asr": evade_asr,
        "evade_ca": evade_ca,
    }

    for name, label, is_probed in operators:
        evade_dir = os.path.join(evade_psbd, name)
        base_dir = os.path.join(base_psbd, name)

        evade_result = (
            sigma_matched_auroc(evade_psbd, name) if os.path.isdir(evade_dir) else None
        )
        base_result = (
            sigma_matched_auroc(base_psbd, name) if os.path.isdir(base_dir) else None
        )

        key = label.replace("@", "_at_")
        row[f"{key}_evade"] = evade_result["auroc"] if evade_result else None
        row[f"{key}_base"] = base_result["auroc"] if base_result else None
        row[f"{key}_evade_rate"] = evade_result["rate"] if evade_result else None
        row[f"{key}_base_rate"] = base_result["rate"] if base_result else None
        if is_probed:
            row["probed_label"] = label

    return row


def print_summary(rows):
    if not rows:
        print("No results found.")
        return

    print("\n=== Adaptive Attacker Transfer Table ===\n")

    header = (
        "| arch | dataset | attack | rate | ASR(e) | CA(e) "
        "| probed(e) | probed(b) | delta "
        "| xfer1(e) | xfer2(e) | xfer3(e) |"
    )
    print(header)
    print("|" + "|".join(["---"] * 12) + "|")

    for r in rows:
        probed_label = r.get("probed_label", "?")
        probed_key = probed_label.replace("@", "_at_")
        probed_e = r.get(f"{probed_key}_evade")
        probed_b = r.get(f"{probed_key}_base")

        evade_asr = r.get("evade_asr")
        evade_ca = r.get("evade_ca")

        probed_e_s = f"{probed_e:.3f}" if probed_e is not None else "--"
        probed_b_s = f"{probed_b:.3f}" if probed_b is not None else "--"
        delta_s = (
            f"{probed_e - probed_b:+.3f}"
            if probed_e is not None and probed_b is not None
            else "--"
        )
        asr_s = f"{evade_asr:.3f}" if evade_asr is not None else "--"
        ca_s = f"{evade_ca:.3f}" if evade_ca is not None else "--"

        xfer_cols = []
        for name, label, is_probed in (
            VIT_OPERATORS if r["arch"] == "vit" else SWIN_OPERATORS
        ):
            if is_probed:
                continue
            key = label.replace("@", "_at_")
            val = r.get(f"{key}_evade")
            xfer_cols.append(f"{val:.3f}" if val is not None else "--")
        while len(xfer_cols) < 3:
            xfer_cols.append("--")

        print(
            f"| {r['arch']} | {r['dataset']} | {r['attack']} | {r['rate']:.0%} "
            f"| {asr_s} | {ca_s} "
            f"| {probed_e_s} | {probed_b_s} | {delta_s} "
            f"| {' | '.join(xfer_cols)} |"
        )

    probed_deltas = []
    xfer_aurocs = []
    for r in rows:
        probed_label = r.get("probed_label", "?")
        probed_key = probed_label.replace("@", "_at_")
        pe = r.get(f"{probed_key}_evade")
        pb = r.get(f"{probed_key}_base")
        if pe is not None and pb is not None:
            probed_deltas.append(pe - pb)
        for name, label, is_probed in (
            VIT_OPERATORS if r["arch"] == "vit" else SWIN_OPERATORS
        ):
            if is_probed:
                continue
            key = label.replace("@", "_at_")
            val = r.get(f"{key}_evade")
            if val is not None:
                xfer_aurocs.append(val)

    if probed_deltas:
        print(f"\nMean probed AUROC delta: {np.mean(probed_deltas):+.3f}")
        print(
            f"Probed AUROC dropped (delta < 0): {sum(1 for d in probed_deltas if d < 0)}/{len(probed_deltas)}"
        )
    if xfer_aurocs:
        print(f"Mean transfer AUROC: {np.mean(xfer_aurocs):.3f}")
        print(
            f"Transfer AUROC > 0.8: {sum(1 for a in xfer_aurocs if a > 0.8)}/{len(xfer_aurocs)}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("vit", "swin"), default=None)
    parser.add_argument("--dataset", choices=("cifar100", "tiny"), default=None)
    parser.add_argument("--output", default="results/adaptive_attacker_analysis.json")
    args = parser.parse_args()

    archs = [args.architecture] if args.architecture else ["vit", "swin"]
    datasets = [args.dataset] if args.dataset else ["cifar100", "tiny"]

    all_rows = []
    for arch in archs:
        operators = VIT_OPERATORS if arch == "vit" else SWIN_OPERATORS
        for dataset in datasets:
            for rate_tag, rate_float in RATE_TAGS:
                for attack in ATTACKS:
                    row = analyze_one(
                        arch, dataset, attack, rate_tag, rate_float, operators
                    )
                    if row is not None:
                        all_rows.append(row)

    print_summary(all_rows)

    if all_rows:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nSaved {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
