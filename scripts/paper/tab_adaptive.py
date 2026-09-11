"""The adaptive attacker and the multi-probe union, from the 2 cross-checkpoint records.

An attacker who trains against 1 probe collapses that probe and leaves the
others standing, and a union over probes recovers most of the loss. Both records
were written before the paper folder existed and live at the results root. This
reads them, splits ViT from Swin, and reports the probed operator's AUROC before
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
    build_parser,
    fmt,
    load_json,
    mean_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_adaptive.py"
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
    combos = [value for key, value in row.items() if key.startswith("combo_") and key.endswith("_auroc")]
    best = max(combos) if combos else None
    return best


def main() -> None:
    args = build_parser(__doc__).parse_args()
    attacker_path = os.path.join(args.results_dir, ATTACKER_FILE)
    union_path = os.path.join(args.results_dir, UNION_FILE)
    attacker_rows = load_json(attacker_path) or []
    union_rows = load_json(union_path) or []
    union_by_key = {
        (row["arch"], row["dataset"], row["attack"], row["rate"]): row for row in union_rows
    }
    inputs = [attacker_path, union_path]

    table_rows = []
    per_arch: dict[str, dict[str, list[float]]] = {}
    for row in sorted(attacker_rows, key=lambda r: (r["arch"], r["dataset"], r["attack"], r["rate"])):
        probed = row["probed_label"].replace("@", "_at_")
        probed_base = row.get(f"{probed}_base")
        probed_evade = row.get(f"{probed}_evade")
        transfer = transfer_values(row)
        union = best_union(union_by_key.get((row["arch"], row["dataset"], row["attack"], row["rate"]), {}))
        if row.get("evade_asr", 0) < HIGH_ASR:
            continue
        stats = per_arch.setdefault(row["arch"], {"base": [], "evade": [], "transfer": [], "union": [], "asr_cost": [], "ca_cost": []})
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
                row["dataset"],
                row["attack"],
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
            "The adaptive attacker, cells whose evasive checkpoint keeps attack success "
            f"at or above {HIGH_ASR:g}: the probed operator's AUROC on the base and the "
            "evasive checkpoint, the mean AUROC of the operators the attacker never "
            "trained against, and the best probe union on the evasive checkpoint. "
            "dCA is the clean-accuracy cost of evasion. ViT evades token\\_mask at "
            "the attention input, Swin evades dropout there."
        ),
        label="tab:adaptive-attacker",
        header=["arch", "dataset", "attack", "rate", "evade ASR", "dCA", "probed base", "probed evade", "transfer mean", "best union"],
        rows=table_rows,
        align="llllrrrrrr",
    )

    macros = {"adaptive_high_asr": (f"{HIGH_ASR:g}", "attack success an evasive checkpoint must keep to count")}
    for arch, stats in per_arch.items():
        macros[f"adaptive_{arch}_cells"] = (str(len(stats["base"])), f"{arch} evasive cells keeping attack success above {HIGH_ASR:g}")
        macros[f"adaptive_{arch}_probed_base"] = (fmt(mean_or_none(stats["base"])), f"{arch} mean AUROC of the probed operator on the base checkpoints")
        macros[f"adaptive_{arch}_probed_evade"] = (fmt(mean_or_none(stats["evade"])), f"{arch} mean AUROC of the probed operator on the evasive checkpoints")
        macros[f"adaptive_{arch}_transfer"] = (fmt(mean_or_none(stats["transfer"])), f"{arch} mean AUROC of the unprobed operators on the evasive checkpoints")
        macros[f"adaptive_{arch}_union"] = (fmt(mean_or_none(stats["union"])), f"{arch} mean AUROC of the best probe union on the evasive checkpoints")
        macros[f"adaptive_{arch}_ca_cost"] = (fmt(mean_or_none(stats["ca_cost"]), signed=True), f"{arch} mean clean-accuracy cost of evasion")
        macros[f"adaptive_{arch}_asr_cost"] = (fmt(mean_or_none(stats["asr_cost"]), signed=True), f"{arch} mean attack-success cost of evasion")
    write_macros(os.path.join(args.paper_dir, "tables", "adaptive.macros.json"), GENERATOR, inputs, macros)
    print("adaptive: " + ", ".join(f"{arch} n={len(stats['base'])} evade={fmt(mean_or_none(stats['evade']))} union={fmt(mean_or_none(stats['union']))}" for arch, stats in per_arch.items()))


if __name__ == "__main__":
    main()
