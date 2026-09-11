"""All-to-all: where PSU fails, what the discarded softmax carries, and the sign router.

Reads the 2 records the all-to-all study left at the results root. The first
scores 4 statistics from the cached no-perturbation softmax and from PSU on
all-to-one, all-to-all and benign cells. The second applies the label-free sign
router, which chooses PSU or entropy from the sign of the suspect pool's mean PSU
against clean validation, and reports it against always-PSU and the oracle.

    PYTHONPATH=. python scripts/paper/tab_all_to_all.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    attack_label,
    build_parser,
    ci_text,
    fmt,
    load_json,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_all_to_all.py"
ENTROPY_FILE = "all_to_all_entropy.json"
ROUTER_FILE = "regime_router.json"
STATISTICS = ("psu_ratio", "neg_entropy", "confidence", "logit_margin")
GROUPS = ("all_to_one", "all_to_all", "benign")
GROUP_LABELS = {
    "all_to_one": "All-to-one",
    "all_to_all": "All-to-all",
    "benign": attack_label("benign"),
}


def main() -> None:
    args = build_parser(__doc__).parse_args()
    entropy_path = os.path.join(args.results_dir, ENTROPY_FILE)
    router_path = os.path.join(args.results_dir, ROUTER_FILE)
    entropy = load_json(entropy_path)
    router = load_json(router_path)
    if entropy is None or router is None:
        raise SystemExit(f"{entropy_path} or {router_path} is missing")
    summary = entropy["summary"]
    inputs = [entropy_path, router_path]

    rows = []
    for group in GROUPS:
        block = summary.get(group)
        if block is None:
            continue
        rows.append(
            [
                GROUP_LABELS[group],
                str(block["n"]),
                *[fmt(block["auroc"].get(statistic)) for statistic in STATISTICS],
            ]
        )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "all_to_all.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "All-to-all against all-to-one: mean AUROC of fractional PSU at the "
            "token\\_mask placement at the attention input and the matched rate "
            "beside 3 statistics of the unperturbed softmax, each scored one-sided "
            "in the direction its own mechanism fixes, with the benign references "
            "as the control."
        ),
        label="tab:all-to-all",
        header=[
            "models",
            "n",
            "PSU ratio",
            "neg. entropy",
            "confidence",
            "logit margin",
        ],
        rows=rows,
        align="lrrrrr",
    )

    router_summary = router["summary"]
    low, high = router_summary.get("delta_ci", (float("nan"), float("nan")))
    macros = {
        "a2a_cells": (str(summary["all_to_all"]["n"]), "all-to-all cells scored"),
        "a2a_psu_auroc": (
            fmt(summary["all_to_all"]["auroc"]["psu_ratio"]),
            "mean PSU AUROC on all-to-all cells",
        ),
        "a2a_entropy_auroc": (
            fmt(summary["all_to_all"]["auroc"]["neg_entropy"]),
            "mean negative-entropy AUROC on all-to-all cells",
        ),
        "a2o_psu_auroc": (
            fmt(summary["all_to_one"]["auroc"]["psu_ratio"]),
            "mean PSU AUROC on the all-to-one cells of the same record",
        ),
        "a2o_entropy_auroc": (
            fmt(summary["all_to_one"]["auroc"]["neg_entropy"]),
            "mean negative-entropy AUROC on the all-to-one cells",
        ),
        "benign_entropy_auroc": (
            fmt(summary["benign"]["auroc"]["neg_entropy"]),
            "mean negative-entropy AUROC on the benign references",
        ),
        "router_cells": (
            str(router_summary["n"]),
            "cells the sign router was scored on",
        ),
        "router_always_psu": (
            fmt(router_summary["always_psu"]),
            "mean AUROC of always choosing PSU",
        ),
        "router_routed": (
            fmt(router_summary["routed"]),
            "mean AUROC of the sign router",
        ),
        "router_oracle": (
            fmt(router_summary["oracle_max"]),
            "mean AUROC of the per-cell oracle choice",
        ),
        "router_delta": (
            fmt(router_summary["delta_mean"], signed=True),
            "mean gain of the sign router over always-PSU",
        ),
        "router_delta_ci": (
            ci_text(low, high),
            "bootstrap interval on the sign router's gain",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "all_to_all.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"all-to-all: psu {macros['a2a_psu_auroc'][0]} entropy {macros['a2a_entropy_auroc'][0]}, router {macros['router_delta'][0]}"
    )


if __name__ == "__main__":
    main()
