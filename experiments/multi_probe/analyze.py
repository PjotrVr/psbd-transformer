"""Multi-probe PSBD defence against the adaptive attacker.

For each evasive checkpoint from H25, loads per-operator PSBD sweep caches,
computes the min-rank combined score across all available operators, and
reports multi-probe AUROC and TPR/FPR at the Bonferroni-corrected threshold.

Runs entirely on CPU using cached .pt files. No new GPU jobs needed.

Example
    python scripts/multi_probe/analyze.py
    python scripts/multi_probe/analyze.py --architecture vit --dataset cifar100
"""

import argparse
import itertools
import json
import os

import numpy as np

from defences.psbd_cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.psbd_metrics import (
    complete_rates,
    multi_probe_auroc,
    multi_probe_detection,
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
    ("before_attention_norm_token_mask", "tm@ban", True),
    ("before_attention_norm", "do@ban", False),
    ("mlp_norm_out_gain_scale", "gs@mno", False),
    ("before_mlp_gaussian", "ga@bm", False),
]

SWIN_OPERATORS = [
    ("before_attention_norm", "do@ban", True),
    ("before_attention_norm_token_mask", "tm@ban", False),
    ("mlp_norm_out_gain_scale", "gs@mno", False),
    ("pre_residual", "do@pre_res", False),
]

SIGMA_TARGET = 0.6
TARGET_FPR = 0.25


def load_psu_at_sigma(psbd_dir, operator_name, sigma_target=SIGMA_TARGET):
    """Load PSU for all 3 splits at the sigma-matched rate. None if unavailable."""
    manifest = read_split_manifest(psbd_dir)
    if manifest is None:
        return None

    rates = complete_rates(psbd_dir, operator_name)
    if not rates:
        return None

    chosen = None
    for rate in rates:
        val_path = dropout_pass_path(psbd_dir, operator_name, rate, "validation")
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

    psu = {}
    for split in ("validation", "clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, operator_name, chosen, split)
        )
        psu[split] = psu_ratio_from_cache(probs, labels, per_pass)

    psu["clean"] = pair_clean_to_backdoor(psu["clean"], manifest)

    return {"psu": psu, "rate": chosen}


def read_asr_ca(ckpt_folder):
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


def analyze_one(arch, dataset, attack, rate_tag, rate_float, operators):
    evade_folder = f"{arch}_{dataset}_{attack}_{rate_tag}_evade_l1"
    evade_psbd = os.path.join(RESULTS_DIR, evade_folder, "psbd")

    if not os.path.isdir(evade_psbd):
        return None

    evade_asr, evade_ca = read_asr_ca(evade_folder)

    available = {}
    for name, label, is_probed in operators:
        result = load_psu_at_sigma(evade_psbd, name)
        if result is not None:
            available[label] = {
                "psu": result["psu"],
                "rate": result["rate"],
                "is_probed": is_probed,
            }

    if len(available) < 2:
        return None

    row = {
        "arch": arch,
        "dataset": dataset,
        "attack": attack,
        "rate": rate_float,
        "evade_asr": evade_asr,
        "evade_ca": evade_ca,
        "n_operators": len(available),
        "operator_labels": list(available.keys()),
    }

    for label, data in available.items():
        from sklearn.metrics import roc_auc_score

        clean = data["psu"]["clean"].float().numpy()
        backdoor = data["psu"]["backdoor"].float().numpy()
        y = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
        raw = np.concatenate([-clean, -backdoor])
        auroc = (
            float(roc_auc_score(y, raw)) if len(set(y.tolist())) > 1 else float("nan")
        )
        row[f"single_{label}_auroc"] = auroc

    all_labels = list(available.keys())
    for k in range(2, len(all_labels) + 1):
        for combo in itertools.combinations(all_labels, k):
            combo_key = "+".join(combo)
            val_list = [available[l]["psu"]["validation"] for l in combo]
            clean_list = [available[l]["psu"]["clean"] for l in combo]
            bd_list = [available[l]["psu"]["backdoor"] for l in combo]

            auroc = multi_probe_auroc(clean_list, bd_list, val_list)
            det = multi_probe_detection(val_list, clean_list, bd_list, TARGET_FPR)

            row[f"combo_{combo_key}_auroc"] = auroc
            row[f"combo_{combo_key}_tpr"] = det["tpr"]
            row[f"combo_{combo_key}_fpr"] = det["fpr"]
            row[f"combo_{combo_key}_bonf_q"] = det["bonferroni_quantile"]

    transfer_labels = [l for l in all_labels if not available[l]["is_probed"]]
    if transfer_labels:
        val_list = [available[l]["psu"]["validation"] for l in transfer_labels]
        clean_list = [available[l]["psu"]["clean"] for l in transfer_labels]
        bd_list = [available[l]["psu"]["backdoor"] for l in transfer_labels]

        auroc = multi_probe_auroc(clean_list, bd_list, val_list)
        det = multi_probe_detection(val_list, clean_list, bd_list, TARGET_FPR)

        row["transfer_only_auroc"] = auroc
        row["transfer_only_tpr"] = det["tpr"]
        row["transfer_only_fpr"] = det["fpr"]
        row["transfer_only_k"] = len(transfer_labels)
        row["transfer_only_labels"] = transfer_labels

    val_all = [available[l]["psu"]["validation"] for l in all_labels]
    clean_all = [available[l]["psu"]["clean"] for l in all_labels]
    bd_all = [available[l]["psu"]["backdoor"] for l in all_labels]

    auroc = multi_probe_auroc(clean_all, bd_all, val_all)
    det = multi_probe_detection(val_all, clean_all, bd_all, TARGET_FPR)

    row["all_auroc"] = auroc
    row["all_tpr"] = det["tpr"]
    row["all_fpr"] = det["fpr"]
    row["all_k"] = len(all_labels)

    return row


def print_summary(rows):
    if not rows:
        print("No results found.")
        return

    high_asr = [
        r for r in rows if r.get("evade_asr") is not None and r["evade_asr"] > 0.9
    ]

    print(f"\n=== Multi-Probe PSBD Defence (H41) ===")
    print(f"Total evasive checkpoints analyzed: {len(rows)}")
    print(f"With ASR > 0.9: {len(high_asr)}")

    print(f"\n--- Main table (ASR > 0.9 only) ---\n")
    header = "| arch | dataset | attack | rate | probed | best_xfer | all_multi | xfer_multi |"
    print(header)
    print("|" + "|".join(["---"] * 8) + "|")

    for r in high_asr:
        labels = r["operator_labels"]
        operators = VIT_OPERATORS if r["arch"] == "vit" else SWIN_OPERATORS
        probed_label = None
        for _, lbl, is_p in operators:
            if is_p and lbl in labels:
                probed_label = lbl
                break

        probed_auroc = (
            r.get(f"single_{probed_label}_auroc", None) if probed_label else None
        )
        probed_s = f"{probed_auroc:.3f}" if probed_auroc is not None else "--"

        xfer_aurocs = []
        for lbl in labels:
            if lbl == probed_label:
                continue
            a = r.get(f"single_{lbl}_auroc")
            if a is not None:
                xfer_aurocs.append(a)
        best_xfer = max(xfer_aurocs) if xfer_aurocs else None
        best_xfer_s = f"{best_xfer:.3f}" if best_xfer is not None else "--"

        all_auroc = r.get("all_auroc")
        all_s = f"{all_auroc:.3f}" if all_auroc is not None else "--"

        xfer_auroc = r.get("transfer_only_auroc")
        xfer_s = f"{xfer_auroc:.3f}" if xfer_auroc is not None else "--"

        print(
            f"| {r['arch']} | {r['dataset']} | {r['attack']} | {r['rate']:.0%} "
            f"| {probed_s} | {best_xfer_s} | {all_s} | {xfer_s} |"
        )

    if high_asr:
        probed_list = []
        best_xfer_list = []
        all_multi_list = []
        xfer_multi_list = []

        for r in high_asr:
            labels = r["operator_labels"]
            operators = VIT_OPERATORS if r["arch"] == "vit" else SWIN_OPERATORS
            probed_label = None
            for _, lbl, is_p in operators:
                if is_p and lbl in labels:
                    probed_label = lbl
                    break

            pa = r.get(f"single_{probed_label}_auroc") if probed_label else None
            if pa is not None:
                probed_list.append(pa)

            xfer_aurocs = []
            for lbl in labels:
                if lbl == probed_label:
                    continue
                a = r.get(f"single_{lbl}_auroc")
                if a is not None:
                    xfer_aurocs.append(a)
            if xfer_aurocs:
                best_xfer_list.append(max(xfer_aurocs))

            aa = r.get("all_auroc")
            if aa is not None:
                all_multi_list.append(aa)

            xa = r.get("transfer_only_auroc")
            if xa is not None:
                xfer_multi_list.append(xa)

        print(f"\n--- Aggregate (ASR > 0.9, n={len(high_asr)}) ---\n")
        if probed_list:
            print(f"Probed operator mean AUROC:       {np.mean(probed_list):.3f}")
        if best_xfer_list:
            print(f"Best single transfer mean AUROC:  {np.mean(best_xfer_list):.3f}")
        if all_multi_list:
            print(f"All-operator multi-probe AUROC:   {np.mean(all_multi_list):.3f}")
            print(
                f"  above 0.90: {sum(1 for a in all_multi_list if a > 0.9)}/{len(all_multi_list)}"
            )
            print(
                f"  above 0.80: {sum(1 for a in all_multi_list if a > 0.8)}/{len(all_multi_list)}"
            )
        if xfer_multi_list:
            print(f"Transfer-only multi-probe AUROC:  {np.mean(xfer_multi_list):.3f}")
            print(
                f"  above 0.90: {sum(1 for a in xfer_multi_list if a > 0.9)}/{len(xfer_multi_list)}"
            )

        tpr_all = [r.get("all_tpr") for r in high_asr if r.get("all_tpr") is not None]
        fpr_all = [r.get("all_fpr") for r in high_asr if r.get("all_fpr") is not None]
        if tpr_all:
            print(f"\nAll-operator TPR at Bonferroni:   {np.mean(tpr_all):.3f}")
        if fpr_all:
            print(
                f"All-operator FPR at Bonferroni:   {np.mean(fpr_all):.3f} (target {TARGET_FPR})"
            )

    print(f"\n--- Combo breakdown (all k-of-n, ASR > 0.9) ---\n")

    combo_stats = {}
    for r in high_asr:
        for key, val in r.items():
            if key.startswith("combo_") and key.endswith("_auroc"):
                combo_name = key[len("combo_") : -len("_auroc")]
                k = combo_name.count("+") + 1
                combo_stats.setdefault(k, []).append(val)

    for k in sorted(combo_stats.keys()):
        vals = [v for v in combo_stats[k] if v == v]
        if vals:
            print(
                f"  k={k}: mean AUROC {np.mean(vals):.3f}, "
                f">{0.9}: {sum(1 for v in vals if v > 0.9)}/{len(vals)}, "
                f">{0.8}: {sum(1 for v in vals if v > 0.8)}/{len(vals)}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("vit", "swin"), default=None)
    parser.add_argument("--dataset", choices=("cifar100", "tiny"), default=None)
    parser.add_argument("--output", default="results/multi_probe_analysis.json")
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
            json.dump(all_rows, f, indent=2, default=str)
        print(f"\nSaved {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
