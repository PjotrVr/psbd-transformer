"""The adaptive attacker and the multi-probe union, from the 2 cross-checkpoint records.

An attacker who trains against 1 probe collapses that probe and leaves the
others standing, and a union over probes recovers most of the loss. Both records
were written before the paper folder existed and live at the results root. This
reads them, splits ViT from Swin and reports the probed operator's AUROC before
and after evasion, the transfer operators' AUROC on the evasive checkpoint, and
the best union's AUROC, with the attack success and clean accuracy the evasion
cost.

    PYTHONPATH=. python scripts/paper/tab_adaptive.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    as_float,
    attack_label,
    build_parser,
    dataset_label,
    fmt,
    load_json,
    mean_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_adaptive.py"
# How multi_probe_analysis.json abbreviates an operator in its probe labels.
OPERATOR_ABBREVIATIONS = {
    "token_mask": "tm",
    "dropout": "do",
    "gain_scale": "gs",
    "gaussian": "ga",
}
ATTACKER_FILE = "adaptive_attacker_analysis.json"
UNION_FILE = "multi_probe_analysis.json"
HIGH_ASR = 0.9


def operator_columns(row: dict) -> list[str]:
    """The operator stems present in an attacker row, `token_mask_at_ban` style."""
    stems = sorted(
        key[: -len("_evade")]
        for key in row
        if key.endswith("_evade") and not key.endswith("_rate_evade")
    )
    return stems


def transfer_values(row: dict) -> list[float]:
    """AUROC on the evasive checkpoint of every operator the attacker did not train against."""
    probed = row["probed_label"].replace("@", "_at_")
    values = [
        row[f"{stem}_evade"]
        for stem in operator_columns(row)
        if stem != probed and row.get(f"{stem}_evade") is not None
    ]
    return values


def best_union(row: dict) -> float | None:
    """The union over every probe the defender ran, never the best subset."""
    union = row.get("all_auroc")
    return union


def main() -> None:
    args = build_parser(__doc__).parse_args()
    attacker_path = os.path.join(args.results_dir, ATTACKER_FILE)
    union_path = os.path.join(args.results_dir, UNION_FILE)
    attacker_rows = load_json(attacker_path) or []
    union_rows = load_json(union_path) or []
    union_by_key = {
        (row["arch"], row["dataset"], row["attack"], row["rate"]): row
        for row in union_rows
    }
    inputs = [attacker_path, union_path]

    table_rows = []
    per_arch: dict[str, dict[str, list[float]]] = {}
    for row in sorted(
        attacker_rows, key=lambda r: (r["arch"], r["dataset"], r["attack"], r["rate"])
    ):
        probed = row["probed_label"].replace("@", "_at_")
        probed_base = row.get(f"{probed}_base")
        probed_evade = row.get(f"{probed}_evade")
        transfer = transfer_values(row)
        union = best_union(
            union_by_key.get(
                (row["arch"], row["dataset"], row["attack"], row["rate"]), {}
            )
        )
        if row.get("evade_asr", 0) < HIGH_ASR:
            continue
        stats = per_arch.setdefault(
            row["arch"],
            {
                "base": [],
                "evade": [],
                "transfer": [],
                "union": [],
                "asr_cost": [],
                "ca_cost": [],
            },
        )
        union_row = union_by_key.get(
            (row["arch"], row["dataset"], row["attack"], row["rate"]), {}
        )
        # The attacker's record names the probe in full (token_mask@ban) and the
        # union record abbreviates the operator (tm@ban), so the full name has to
        # be shortened before it can pick the union that leaves the probe out.
        operator, _, site = row["probed_label"].partition("@")
        probed_short = f"{OPERATOR_ABBREVIATIONS.get(operator, operator)}@{site}"
        without_probed = [
            key
            for key in union_row
            if key.startswith("combo_")
            and key.endswith("_auroc")
            and probed_short not in key
            and key.count("+") == len(union_row.get("operator_labels", [])) - 2
        ]
        if union_row.get("all_calibrated_tpr") is not None:
            stats.setdefault("union_tpr", []).append(union_row["all_calibrated_tpr"])
            stats.setdefault("union_fpr", []).append(union_row["all_calibrated_fpr"])
        if without_probed and union is not None:
            stats.setdefault("oracle_gap", []).append(
                union_row[without_probed[0]] - union
            )
        stats["base"].append(probed_base)
        stats["evade"].append(probed_evade)
        stats["transfer"].append(mean_or_none(transfer))
        if union is not None:
            stats["union"].append(union)
        stats["asr_cost"].append(row["evade_asr"] - row["base_asr"])
        stats["ca_cost"].append(row["evade_ca"] - row["base_ca"])
        table_rows.append(
            [
                row["arch"],
                dataset_label(row["dataset"]),
                attack_label(row["attack"]),
                f"{row['rate']:g}",
                fmt(row["evade_asr"]),
                fmt(row["evade_ca"] - row["base_ca"], signed=True),
                fmt(probed_base),
                fmt(probed_evade),
                fmt(mean_or_none(transfer)),
                fmt(union),
            ]
        )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "adaptive_attacker.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "The adaptive attacker on the models whose evasive version keeps attack "
            f"success at or above {HIGH_ASR:g}."
        ),
        label="tab:adaptive-attacker",
        header=[
            "arch",
            "dataset",
            "attack",
            "rate",
            "evade ASR",
            "dCA",
            "probed base",
            "probed evade",
            "transfer mean",
            "union",
        ],
        rows=table_rows,
        align="llllrrrrrr",
    )

    # The 4 rows the body needs: the population before and after the attack, and
    # the 2 worst single models, 1 for the probe the attacker trained against and
    # 1 for the union. It was hand-typed in the section, which is how it came to
    # quote the population as a row of the table it summarizes.
    summary_rows = []
    for arch, stats in sorted(per_arch.items()):
        by_probe = min(
            (row for row in table_rows if row[0] == arch),
            key=lambda row: as_float(row[7]),
        )
        by_union = min(
            (row for row in table_rows if row[0] == arch and row[9] != "--"),
            key=lambda row: as_float(row[9]),
        )
        summary_rows.append(
            [
                "before the attack",
                fmt(mean_or_none(stats["base"])),
                "--",
                fmt(mean_or_none(stats["base"])),
                "--",
                "--",
            ]
        )
        summary_rows.append(
            [
                f"after the attack, mean of {len(stats['base'])}",
                fmt(
                    mean_or_none(
                        [
                            row["evade_asr"]
                            for row in attacker_rows
                            if row.get("evade_asr", 0) >= HIGH_ASR
                        ]
                    )
                ),
                fmt(mean_or_none(stats["ca_cost"]), signed=True),
                f"\\textbf{{{fmt(mean_or_none(stats['evade']))}}}",
                fmt(mean_or_none(stats["transfer"])),
                f"\\textbf{{{fmt(mean_or_none(stats['union']))}}}",
            ]
        )
        for label, row in (
            ("worst model for the probe", by_probe),
            ("worst model for the union", by_union),
        ):
            summary_rows.append(
                [
                    f"{label} ({row[2]} {as_float(row[3]) * 100:g}%)",
                    row[4],
                    row[5],
                    row[7],
                    row[8],
                    row[9],
                ]
            )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "adaptive_summary.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "The adaptive attacker on the CIFAR-100 ViT-B/16 models whose evasive "
            f"version keeps attack success at or above {HIGH_ASR:g}. ASR is the attack "
            "success rate of the evasive model and cost its clean-accuracy loss. "
            "Probed is the AUROC of the placement the attacker trained against, "
            "transfer the mean AUROC of the 2 probes the attacker never saw, and union "
            "the AUROC of the min-rank union of all 3. \\Cref{tab:adaptive-attacker} "
            "gives every model separately."
        ),
        label="tab:adaptive",
        header=["", "ASR", "cost", "probed", "transfer", "union"],
        rows=summary_rows,
        align="lrrrrr",
    )

    macros = {
        "adaptive_high_asr": (
            f"{HIGH_ASR:g}",
            "attack success an evasive checkpoint must keep to count",
        )
    }
    for arch, stats in per_arch.items():
        macros[f"adaptive_{arch}_cells"] = (
            str(len(stats["base"])),
            f"{arch} evasive cells keeping attack success above {HIGH_ASR:g}",
        )
        macros[f"adaptive_{arch}_probed_base"] = (
            fmt(mean_or_none(stats["base"])),
            f"{arch} mean AUROC of the probed operator on the base checkpoints",
        )
        macros[f"adaptive_{arch}_probed_evade"] = (
            fmt(mean_or_none(stats["evade"])),
            f"{arch} mean AUROC of the probed operator on the evasive checkpoints",
        )
        macros[f"adaptive_{arch}_transfer"] = (
            fmt(mean_or_none(stats["transfer"])),
            f"{arch} mean AUROC of the unprobed operators on the evasive checkpoints",
        )
        macros[f"adaptive_{arch}_union_tpr"] = (
            fmt(mean_or_none(stats.get("union_tpr", [])), places=2),
            f"{arch} mean TPR of the probe union at the headline budget on the evasive checkpoints",
        )
        macros[f"adaptive_{arch}_union_fpr"] = (
            fmt(mean_or_none(stats.get("union_fpr", [])), places=2),
            f"{arch} mean realized FPR of the probe union at the headline budget",
        )
        macros[f"adaptive_{arch}_oracle_gap"] = (
            fmt(mean_or_none(stats.get("oracle_gap", [])), signed=True),
            f"{arch} mean AUROC of the union without the attacked probe minus the full union",
        )
        macros[f"adaptive_{arch}_ca_cost_max_points"] = (
            fmt(-100 * min(stats["ca_cost"]), places=1),
            f"{arch} largest clean-accuracy loss of an evasive checkpoint, in points",
        )
        macros[f"adaptive_{arch}_union"] = (
            fmt(mean_or_none(stats["union"])),
            f"{arch} mean AUROC of the best probe union on the evasive checkpoints",
        )
        macros[f"adaptive_{arch}_ca_cost"] = (
            fmt(mean_or_none(stats["ca_cost"]), signed=True),
            f"{arch} mean clean-accuracy cost of evasion",
        )
        macros[f"adaptive_{arch}_asr_cost"] = (
            fmt(mean_or_none(stats["asr_cost"]), signed=True),
            f"{arch} mean attack-success cost of evasion",
        )
    write_macros(
        os.path.join(args.paper_dir, "tables", "adaptive.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        "adaptive: "
        + ", ".join(
            f"{arch} n={len(stats['base'])} evade={fmt(mean_or_none(stats['evade']))} union={fmt(mean_or_none(stats['union']))}"
            for arch, stats in per_arch.items()
        )
    )


if __name__ == "__main__":
    main()
