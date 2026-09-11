"""Prediction depth against PSBD on the same cells: a 1-pass statistic that needs no perturbation.

depth_soft reads every block through the network's own final LayerNorm and
head and averages the lens probability of the final answer over depth, 1
forward pass, no perturbation, no rate to pick. Every clearing cell carries a
prediction_depth.json, so the comparison against the recommended placement at
the adaptive rule is paired within cell on the same splits. AUROC at the
headline quantile and TPR at the smallest budget are both read, since the
earlier reading of this statistic found its whole advantage at the low-budget
operating point.

    PYTHONPATH=. python scripts/paper/mech_prediction_depth.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defences.decision import EASY_ATTACKS, HARD_ATTACKS, RECOMMENDED_PLACEMENT  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    attack_label,
    bootstrap_ci,
    build_parser,
    ci_text,
    clearing_cells,
    fmt,
    load_coverage,
    load_declaration,
    load_json,
    load_psbd_metrics,
    mean_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/mech_prediction_depth.py"
STATISTIC = "depth_soft"
LOW_BUDGET_KEY = "q0.01"


def depth_record(results_dir: str, folder: str) -> dict | None:
    record = load_json(os.path.join(results_dir, folder, "prediction_depth.json"))
    if record is None:
        return None
    block = record["statistics"].get(STATISTIC)
    return block


def measure_cell(results_dir: str, cell: dict) -> dict | None:
    """Both detectors' AUROC and low-budget TPR on 1 cell, or None when either is missing."""
    depth = depth_record(results_dir, cell["folder_name"])
    psbd = psbd_values(
        load_psbd_metrics(results_dir, cell["folder_name"]),
        RECOMMENDED_PLACEMENT,
        "adaptive",
    )
    if depth is None or psbd is None or HEADLINE_KEY not in psbd:
        return None
    measured = {
        "depth_auroc": depth["auroc"],
        "depth_tpr": depth[LOW_BUDGET_KEY]["tpr"],
        "psbd_auroc": psbd[HEADLINE_KEY]["auroc"],
        "psbd_tpr": psbd[LOW_BUDGET_KEY]["tpr"],
    }
    return measured


def subset_row(
    name: str, cells: list[dict], resamples: int, seed: int
) -> tuple[list[str], dict]:
    auroc_delta = [c["m"]["depth_auroc"] - c["m"]["psbd_auroc"] for c in cells]
    tpr_delta = [c["m"]["depth_tpr"] - c["m"]["psbd_tpr"] for c in cells]
    auroc_ci = bootstrap_ci(auroc_delta, resamples, seed)
    tpr_ci = bootstrap_ci(tpr_delta, resamples, seed)
    row = [
        name,
        str(len(cells)),
        fmt(mean_or_none([c["m"]["depth_auroc"] for c in cells])),
        fmt(mean_or_none([c["m"]["psbd_auroc"] for c in cells])),
        f"{fmt(mean_or_none(auroc_delta), signed=True)} {ci_text(*auroc_ci)}",
        fmt(mean_or_none([c["m"]["depth_tpr"] for c in cells])),
        fmt(mean_or_none([c["m"]["psbd_tpr"] for c in cells])),
        f"{fmt(mean_or_none(tpr_delta), signed=True)} {ci_text(*tpr_ci)}",
        str(sum(1 for value in tpr_delta if value > 0)),
    ]
    numbers = {
        "auroc_delta": mean_or_none(auroc_delta),
        "auroc_ci": auroc_ci,
        "tpr_delta": mean_or_none(tpr_delta),
        "tpr_ci": tpr_ci,
        "n": len(cells),
    }
    return row, numbers


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    declaration = load_declaration(args.declaration)
    cells = []
    for cell in clearing_cells(load_coverage(args.results_dir)):
        measured = measure_cell(args.results_dir, cell)
        if measured is not None:
            cells.append({**cell, "m": measured})
    inputs = [
        coverage_path,
        f"{args.results_dir}/<folder>/prediction_depth.json ({len(cells)} cells)",
        f"{args.results_dir}/<folder>/psbd_metrics.json",
    ]

    subsets = [
        ("all", cells),
        ("hard attacks", [c for c in cells if c["attack"] in HARD_ATTACKS]),
        ("easy attacks", [c for c in cells if c["attack"] in EASY_ATTACKS]),
    ]
    for attack in list(HARD_ATTACKS) + list(EASY_ATTACKS):
        group = [c for c in cells if c["attack"] == attack]
        if group:
            subsets.append((attack_label(attack), group))
    for rate in (0.01, 0.05, 0.1):
        subsets.append(
            (f"rate {rate:g}", [c for c in cells if c["poison_rate"] == rate])
        )

    rows = []
    numbers = {}
    for name, group in subsets:
        row, stats = subset_row(name, group, args.bootstrap, args.seed)
        rows.append(row)
        numbers[name] = stats
    write_table(
        path=os.path.join(args.paper_dir, "tables", "prediction_depth.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Prediction depth (depth\\_soft, 1 forward pass) against PSBD at the "
            "token\\_mask placement at the attention input and the adaptive rule, "
            "paired within model on the same splits: AUROC at the headline "
            "quantile and TPR at the smallest budget, with the paired delta, its "
            "bootstrap interval and the count of models where prediction depth "
            "wins at the low budget."
        ),
        label="tab:prediction-depth",
        header=[
            "subset",
            "n",
            "depth AUROC",
            "PSBD AUROC",
            "delta AUROC [CI]",
            "depth TPR@1%",
            "PSBD TPR@1%",
            "delta TPR [CI]",
            "depth wins",
        ],
        rows=rows,
        align="lrrrlrrlr",
    )

    benign = []
    for dataset, folder in declaration["benign_reference"].items():
        record = depth_record(args.results_dir, folder)
        if record is not None:
            benign.append(record["auroc"])
    everything = numbers["all"]
    hard = numbers["hard attacks"]
    macros = {
        "depth_cells": (
            str(everything["n"]),
            "clearing cells carrying both prediction depth and the recommended placement",
        ),
        "depth_auroc_all": (
            fmt(mean_or_none([c["m"]["depth_auroc"] for c in cells])),
            "mean AUROC of prediction depth over the clearing cells",
        ),
        "depth_minus_psbd_auroc": (
            fmt(everything["auroc_delta"], signed=True),
            "mean paired AUROC delta, prediction depth minus PSBD, all clearing cells",
        ),
        "depth_minus_psbd_auroc_ci": (
            ci_text(*everything["auroc_ci"]),
            "bootstrap interval on depth_minus_psbd_auroc",
        ),
        "depth_minus_psbd_tpr": (
            fmt(everything["tpr_delta"], signed=True),
            "mean paired low-budget TPR delta, prediction depth minus PSBD, all clearing cells",
        ),
        "depth_minus_psbd_tpr_ci": (
            ci_text(*everything["tpr_ci"]),
            "bootstrap interval on depth_minus_psbd_tpr",
        ),
        "depth_minus_psbd_tpr_hard": (
            fmt(hard["tpr_delta"], signed=True),
            "mean paired low-budget TPR delta on the hard attacks",
        ),
        "depth_minus_psbd_tpr_hard_ci": (
            ci_text(*hard["tpr_ci"]),
            "bootstrap interval on depth_minus_psbd_tpr_hard",
        ),
        "depth_minus_psbd_auroc_hard": (
            fmt(hard["auroc_delta"], signed=True),
            "mean paired AUROC delta on the hard attacks",
        ),
        "depth_tpr_wins": (
            fmt(
                sum(1 for c in cells if c["m"]["depth_tpr"] > c["m"]["psbd_tpr"]),
                places=0,
            ),
            "clearing cells where prediction depth beats PSBD at the low budget",
        ),
        "depth_benign_auroc": (
            fmt(mean_or_none(benign)),
            f"mean prediction-depth AUROC on the {len(benign)} benign references",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "prediction_depth.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"prediction depth: n={everything['n']}, auroc delta {macros['depth_minus_psbd_auroc'][0]}, tpr delta {macros['depth_minus_psbd_tpr'][0]}"
    )


if __name__ == "__main__":
    main()
