"""T1: the ViT panel, attack success and clean accuracy by dataset, attack and rate.

Reads results/coverage/coverage.json only, no psbd_metrics.json. Every declared
cell appears, so a reader can see which cells the 0.85 ASR bar excludes rather
than finding them silently missing. A cell clearing the bar prints its attack
success rate and clean accuracy together. A cell below the bar prints only its
attack success rate, marked with a dagger, since its clean accuracy is not the
number that explains the cell's absence from every other table in this paper.

    PYTHONPATH=. python scripts/paper/tab_panel.py \
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from defences.decision import EASY_ATTACKS, HARD_ATTACKS  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    build_parser,
    load_coverage,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_panel.py"
DATASET_ORDER = ("cifar10", "cifar100", "gtsrb", "tiny")
RATE_ORDER = (0.01, 0.05, 0.1)
RATE_HEADERS = ("1%", "5%", "10%")


def attack_order(cells: list[dict]) -> list[str]:
    """Attacks present in the panel, easy first then hard, each ordered as decision.py fixes it.

    An attack the panel declares but never runs (excluded folder tokens, a
    future addition) falls back to alphabetical order after the 2 named groups,
    so a new attack cannot silently drop off the table.
    """
    present = {cell["attack"] for cell in cells}
    named = [attack for attack in EASY_ATTACKS + HARD_ATTACKS if attack in present]
    leftover = sorted(present - set(named))
    ordered = named + leftover
    return ordered


def cell_text(cell: dict | None) -> str:
    """ASR / clean accuracy for a clearing cell, ASR alone with a dagger below the bar."""
    if cell is None or cell.get("asr") is None:
        return "--"
    asr = cell["asr"]
    if cell["asr_class"] != "clears":
        return f"{asr:.3f}$^\\dagger$"
    clean_accuracy = cell.get("clean_accuracy")
    if clean_accuracy is None:
        return f"{asr:.3f} / --"
    return f"{asr:.3f} / {clean_accuracy:.3f}"


def panel_rows(
    cells_by_key: dict[tuple[str, str, float], dict],
    datasets: tuple[str, ...],
    attacks: list[str],
) -> list[list[str]]:
    """1 row per dataset and attack, 1 column per poison rate."""
    rows = []
    for dataset in datasets:
        for attack in attacks:
            row = [dataset, attack]
            row += [
                cell_text(cells_by_key.get((dataset, attack, rate)))
                for rate in RATE_ORDER
            ]
            rows.append(row)
    return rows


def benign_rows(
    benign_reference_accuracy: dict[str, float], datasets: tuple[str, ...]
) -> list[list[str]]:
    rows = [
        [dataset, f"{benign_reference_accuracy[dataset]:.3f}"]
        for dataset in datasets
        if dataset in benign_reference_accuracy
    ]
    return rows


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    coverage = load_coverage(args.results_dir)
    cells = coverage["cells"]
    asr_bar = coverage["asr_bar"]
    clearing = [cell for cell in cells if cell["asr_class"] == "clears"]

    datasets = DATASET_ORDER
    attacks = attack_order(cells)
    cells_by_key = {
        (cell["dataset"], cell["attack"], cell["poison_rate"]): cell for cell in cells
    }
    rows = panel_rows(cells_by_key, datasets, attacks)

    write_table(
        path=os.path.join(args.paper_dir, "tables", "panel.tex"),
        generator=GENERATOR,
        inputs=[coverage_path],
        caption=(
            f"The ViT panel, from results/coverage/coverage.json: {len(clearing)} of "
            f"{len(cells)} declared cells clear the attack success bar {asr_bar:.2f} "
            "and carry every table in this paper. A clearing cell prints attack "
            "success rate over clean accuracy. A cell below the bar prints only its "
            r"attack success rate, marked $^\dagger$, so its exclusion stays visible."
        ),
        label="tab:panel",
        header=["dataset", "attack", *RATE_HEADERS],
        rows=rows,
        align="ll" + "r" * len(RATE_ORDER),
    )

    write_table(
        path=os.path.join(args.paper_dir, "tables", "panel_benign.tex"),
        generator=GENERATOR,
        inputs=[coverage_path],
        caption=(
            "Benign-model clean accuracy per dataset, the reference every "
            "clean-accuracy-drop figure in this paper subtracts against. Same "
            "architecture, optimizer and 15 epochs as every panel cell."
        ),
        label="tab:panel-benign",
        header=["dataset", "benign clean accuracy"],
        rows=benign_rows(coverage["benign_reference_accuracy"], datasets),
        align="lr",
    )

    macros = {
        "panel_cells_clearing": (
            str(len(clearing)),
            "cells in the ViT panel clearing the attack success bar",
        ),
        "panel_cells_total": (
            str(len(cells)),
            "cells the ViT panel declares, clearing and below the bar together",
        ),
        "panel_datasets": (str(len(datasets)), "datasets in the ViT panel"),
        "panel_attacks_clearing": (
            str(len({cell["attack"] for cell in clearing})),
            "attacks with at least 1 cell clearing the attack success bar",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "panel.macros.json"),
        GENERATOR,
        [coverage_path],
        macros,
    )
    print(f"panel: {len(clearing)}/{len(cells)} cells clear, {len(datasets)} datasets")


if __name__ == "__main__":
    main()
