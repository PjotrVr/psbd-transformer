"""The ViT detection panel: every attack, every dataset, three poison rates.

One configuration throughout, `token_mask @ before_attention_norm`, with the probe
rate chosen per cell as the one whose clean-validation shift ratio sits closest to
0.8. That rule uses no labels, so the numbers are what a defender could actually
obtain.

Reads the per-checkpoint `psbd_metrics.json`, which is the authoritative record and
already carries all six quantiles per rate, rather than the regenerated
`operating_points.csv`. Clean accuracy and ASR come from the `args.json` sidecar.

Every cell in the grid is printed. A cell with no sweep yet reads `pending` and is
never dropped, because a reader cannot tell an omitted cell from a failed one.

    python scripts/vit_detection_tables.py > docs/vit-detection-tables.md
"""

import json
import os
import sys

PLACEMENT = "before_attention_norm_token_mask"
VARIANT = "detection_psu_ratio"  # fractional PSU, one-sided
TARGET_SIGMA = 0.8
OPERATING_POINTS = ("q0.10", "q0.25")
ASR_FLOOR = 0.5  # below this the backdoor was never implanted

DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
ATTACKS = (
    "badnet_a2o",
    "blend",
    "wanet",
    "adaptive_blend",
    "lc",
    "sig",
    "lf",
    "bpp",
    "tact",
)
RATE_TAGS = ((0.10, "0_1"), (0.05, "0_05"), (0.01, "0_01"))

# PSBD's published TPR/FPR, ResNet-18. 10% from Table 1, 5% from the appendix
# table. The paper has no CIFAR-100 and never evaluates badnet_a2a, bpp, lf, sig
# or tact, so those cells have no counterpart and are left blank on purpose.
PUBLISHED = {
    0.10: {
        ("cifar10", "badnet_a2o"): (1.000, 0.104),
        ("cifar10", "blend"): (1.000, 0.135),
        ("cifar10", "wanet"): (1.000, 0.116),
        ("cifar10", "lc"): (0.992, 0.130),
        ("cifar10", "adaptive_blend"): (0.982, 0.184),
        ("gtsrb", "badnet_a2o"): (0.987, 0.202),
        ("gtsrb", "blend"): (0.910, 0.207),
        ("gtsrb", "wanet"): (0.996, 0.115),
        ("gtsrb", "lc"): (0.944, 0.203),
        ("gtsrb", "adaptive_blend"): (0.899, 0.194),
        ("tiny", "badnet_a2o"): (0.989, 0.088),
        ("tiny", "blend"): (0.919, 0.108),
        ("tiny", "wanet"): (0.959, 0.086),
        ("tiny", "lc"): (0.839, 0.039),
        ("tiny", "adaptive_blend"): (0.949, 0.095),
    },
    0.05: {
        ("cifar10", "badnet_a2o"): (0.979, 0.158),
        ("cifar10", "blend"): (0.899, 0.176),
        ("cifar10", "wanet"): (1.000, 0.113),
        ("cifar10", "lc"): (1.000, 0.107),
        ("cifar10", "adaptive_blend"): (0.982, 0.184),
        ("gtsrb", "badnet_a2o"): (0.993, 0.202),
        ("gtsrb", "blend"): (0.859, 0.223),
        ("gtsrb", "wanet"): (0.999, 0.085),
        ("gtsrb", "lc"): (0.844, 0.202),
        ("gtsrb", "adaptive_blend"): (0.899, 0.194),
        ("tiny", "badnet_a2o"): (0.996, 0.093),
        ("tiny", "blend"): (0.871, 0.065),
        ("tiny", "wanet"): (0.944, 0.109),
        ("tiny", "lc"): (0.983, 0.100),
        ("tiny", "adaptive_blend"): (0.949, 0.095),
    },
    0.01: {},  # PSBD reports no 1% panel
}


def read_cell(folder, results_dir="results", checkpoints_dir="checkpoints"):
    """Metrics for one checkpoint at the rate closest to TARGET_SIGMA, or None."""
    metrics_path = os.path.join(results_dir, folder, "psbd_metrics.json")
    args_path = os.path.join(checkpoints_dir, folder, "args.json")
    if not os.path.exists(metrics_path):
        return None
    report = json.load(open(metrics_path))
    block = report.get("placements", {}).get(PLACEMENT)
    if not block:
        return None

    best = None
    for entry in block.get("rates", []):
        sigma = (entry.get("shift_ratio") or {}).get("validation")
        detection = entry.get(VARIANT)
        if sigma is None or not detection:
            continue
        if not all(q in detection for q in OPERATING_POINTS):
            continue
        distance = abs(sigma - TARGET_SIGMA)
        if best is None or distance < best[0]:
            best = (distance, entry.get("rate"), sigma, detection)
    if best is None:
        return None

    _, rate, sigma, detection = best
    meta = json.load(open(args_path)) if os.path.exists(args_path) else {}
    return {
        "rate": rate,
        "sigma": sigma,
        "auroc": detection[OPERATING_POINTS[0]].get("auroc"),
        "points": {
            q: (detection[q].get("tpr"), detection[q].get("fpr"))
            for q in OPERATING_POINTS
        },
        "ca": meta.get("clean_accuracy"),
        "asr": meta.get("asr"),
        # A clean-label attack is eligible only on the target class, so a requested
        # rate can saturate that pool and resolve far lower. Reporting the requested
        # rate alone makes three identical models look like three poison rates.
        "realized": meta.get("realized_poison_rate"),
        "n_poisoned": meta.get("n_poisoned"),
    }


def fmt(value, places=3):
    return "—" if value is None else f"{value:.{places}f}"


def render(poison_rate, tag):
    print(f"\n## Poison rate {poison_rate:.0%}\n")
    print(
        "| Cell | realized rate | CA | ASR | AUROC | TPR/FPR @10% "
        "| TPR/FPR @25% | PSBD pub TPR/FPR |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    aurocs, t10, t25, pending, low_asr, suspect = [], [], [], [], [], []
    duplicates = []
    for dataset in DATASETS:
        for attack in ATTACKS:
            folder = f"vit_{dataset}_{attack}_{tag}"
            cell = read_cell(folder)
            pub = PUBLISHED.get(poison_rate, {}).get((dataset, attack))
            pub_text = f"{pub[0]:.3f} / {pub[1]:.3f}" if pub else "—"
            label = f"{dataset} / {attack}"

            if cell is None:
                pending.append(folder)
                print(f"| {label} | — | — | — | *pending* | — | — | {pub_text} |")
                continue

            realized = cell["realized"]
            saturated = realized is not None and abs(realized - poison_rate) > 1e-9
            rate_text = "=" if not saturated else f"**{realized:.4f}**"
            if saturated:
                duplicates.append((label, cell["n_poisoned"]))

            a10, f10 = cell["points"]["q0.10"]
            a25, f25 = cell["points"]["q0.25"]
            weak = cell["asr"] is not None and cell["asr"] < ASR_FLOOR
            mark = " ⚠" if weak else ""
            print(
                f"| {label}{mark} | {rate_text} | {fmt(cell['ca'])} | {fmt(cell['asr'])} "
                f"| {fmt(cell['auroc'])} | {fmt(a10)} / {fmt(f10)} "
                f"| {fmt(a25)} / {fmt(f25)} | {pub_text} |"
            )
            if weak:
                low_asr.append(folder)
                continue
            # The evade-contamination signature: near-perfect ranking cannot coexist
            # with near-zero detection at a 25 percent budget.
            if cell["auroc"] and cell["auroc"] > 0.9 and a25 is not None and a25 < 0.1:
                suspect.append(folder)
            aurocs.append(cell["auroc"])
            t10.append(a10)
            t25.append(a25)

    if aurocs:
        n = len(aurocs)
        print(
            f"| **mean** ({n} valid) | | | | **{sum(aurocs) / n:.3f}** "
            f"| **{sum(t10) / n:.3f}** | **{sum(t25) / n:.3f}** | |"
        )
        print(
            f"| **worst** | | | | **{min(aurocs):.3f}** "
            f"| **{min(t10):.3f}** | **{min(t25):.3f}** | |"
        )
    print(
        f"\n{len(pending)} of {len(DATASETS) * len(ATTACKS)} cells pending"
        + (f": {', '.join(pending)}" if pending else ".")
    )
    if duplicates:
        print(
            f"\n**{len(duplicates)} cells did not reach the requested rate**: the clean-label "
            f"pool is one class, so the rate saturates and the cell is identical to the same "
            f"attack at a lower requested rate. Realized rate shown in bold; `=` means the "
            f"requested rate was achieved. Affected: "
            + ", ".join(f"{c} (n={n})" for c, n in duplicates)
        )
    if low_asr:
        print(
            f"\n⚠ ASR below {ASR_FLOOR}, backdoor never implanted, excluded from the "
            f"summary: {', '.join(low_asr)}"
        )
    return pending, suspect


def main():
    print("# ViT-B/16 detection panel")
    print()
    print("`token_mask @ before_attention_norm`: whole patch tokens dropped from the")
    print("residual stream entering attention in every one of the 12 encoder blocks,")
    print("CLS never masked. Probe rate per cell is the one whose clean-validation")
    print("shift ratio lands closest to 0.8, which needs no labels. PSU is fractional,")
    print("one-sided. Quantiles are the false-positive budget, so TPR@10% is the")
    print("threshold set at the 10th percentile of clean-validation PSU.")
    print()
    print("Published values are PSBD (CVPR 2025) on ResNet-18, Table 1 for 10% and the")
    print("appendix table for 5%. Blank where the paper has no counterpart: it reports")
    print("no CIFAR-100 and never evaluates badnet_a2a, bpp, lf, sig or tact.")
    print(
        "Adaptive-Blend is published at 1% (CIFAR-10, GTSRB) and 2% (Tiny) and reused"
    )
    print("in both its tables, so it is not rate-matched to our rows.")
    print()
    print("Generated by `scripts/vit_detection_tables.py`.")

    all_pending, all_suspect = [], []
    for poison_rate, tag in RATE_TAGS:
        pending, suspect = render(poison_rate, tag)
        all_pending += pending
        all_suspect += suspect

    print("\n## Reading the table\n")
    print("This panel is all-to-one only. `badnet_a2a` is excluded because all-to-all")
    print("breaks the prediction-shift premise structurally rather than scoring badly")
    print("under it: with no single target class for a shifted prediction to collapse")
    print("onto, the PSU ordering reverses and AUROC falls below 0.5 on every dataset.")
    print("That is a separate result, written up in `docs/all-to-all-inversion.md`.")
    print()
    print("`tact` is source-specific: it flips only its source classes, so its ASR and")
    print("its detection numbers are measured over that population and not over every")
    print(
        "non-target image. Measuring over the wider pool divides the true rate by the"
    )
    print("class count and makes a working attack look like a failed one.")

    if all_suspect:
        print(
            f"\n**BUILD FAILED**: {len(all_suspect)} rows have AUROC > 0.9 with "
            f"TPR < 0.1 at the 25% budget: {', '.join(all_suspect)}",
            file=sys.stderr,
        )
        return 1
    print(
        f"\n[ok] {len(all_pending)} pending cells, no inconsistent rows",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
