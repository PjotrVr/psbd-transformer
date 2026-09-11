"""Swin-S: every cached placement on the Swin checkpoints, with the coverage it has.

Swin has no coverage ledger and no basis sweep, so this reads every non-SAM,
non-seed, non-evasion swin_* folder with a psbd_metrics.json, keeps the cells
whose sidecar attack success clears the declared bar, and reports each cached
placement's mean AUROC at both rules with the number of cells behind it. The
counts differ by an order of magnitude between placements, which is the reason
the Swin numbers are indicative rather than a panel result, and the table shows
that rather than hiding it.

    PYTHONPATH=. python scripts/paper/tab_swin.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import collections
import glob
import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defences.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    bootstrap_ci,
    build_parser_with_checkpoints,
    ci_text,
    dataset_label,
    fmt,
    is_panel_folder,
    load_args_json,
    load_declaration,
    load_psbd_metrics,
    mean_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_swin.py"
RULES = ("matched", "adaptive")
# Swin's own dropout probe at the attention input, the placement its earliest
# sweeps carried, beside the 2 canonical placements.
SWIN_DROPOUT_INPUT = "before_attention_norm"
MIN_CELLS_FOR_ROW = 5


def swin_cells(results_dir: str, checkpoints_dir: str, asr_bar: float) -> list[dict]:
    """Swin folders that are panel-shaped, have metrics, and whose attack implanted."""
    cells = []
    for path in sorted(
        glob.glob(os.path.join(results_dir, "swin_*", "psbd_metrics.json"))
    ):
        folder = os.path.basename(os.path.dirname(path))
        if not is_panel_folder(folder) or "benign" in folder:
            continue
        sidecar = load_args_json(checkpoints_dir, folder) or {}
        asr = sidecar.get("asr")
        if asr is None or asr < asr_bar:
            continue
        report = load_psbd_metrics(results_dir, folder)
        cells.append(
            {
                "folder": folder,
                "dataset": report["dataset"],
                "attack": report["attack"],
                "poison_rate": report["poison_rate"],
                "asr": asr,
                "report": report,
            }
        )
    return cells


def placement_table(cells: list[dict]) -> tuple[list[list[str]], dict[str, dict]]:
    """Mean AUROC per cached placement and rule, over the cells that carry it."""
    values: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: {rule: [] for rule in RULES}
    )
    for cell in cells:
        for placement in cell["report"]["placements"]:
            if "_seed" in placement:
                continue
            for rule in RULES:
                block = psbd_values(cell["report"], placement, rule)
                if block is not None and HEADLINE_KEY in block:
                    values[placement][rule].append(block[HEADLINE_KEY]["auroc"])
    rows = []
    stats = {}
    for placement, by_rule in sorted(
        values.items(), key=lambda item: -len(item[1]["adaptive"])
    ):
        if len(by_rule["adaptive"]) < MIN_CELLS_FOR_ROW:
            continue
        stats[placement] = {
            rule: {"mean": mean_or_none(by_rule[rule]), "n": len(by_rule[rule])}
            for rule in RULES
        }
        rows.append(
            [
                placement,
                str(len(by_rule["matched"])),
                fmt(mean_or_none(by_rule["matched"])),
                str(len(by_rule["adaptive"])),
                fmt(mean_or_none(by_rule["adaptive"])),
                str(sum(1 for value in by_rule["adaptive"] if value < 0.5)),
            ]
        )
    return rows, stats


def paired_gain(
    cells: list[dict], placement_a: str, placement_b: str, rule: str
) -> list[float]:
    deltas = []
    for cell in cells:
        a = psbd_values(cell["report"], placement_a, rule)
        b = psbd_values(cell["report"], placement_b, rule)
        if a is None or b is None or HEADLINE_KEY not in a or HEADLINE_KEY not in b:
            continue
        deltas.append(a[HEADLINE_KEY]["auroc"] - b[HEADLINE_KEY]["auroc"])
    return deltas


def main() -> None:
    args = build_parser_with_checkpoints(__doc__).parse_args()
    declaration = load_declaration(args.declaration)
    cells = swin_cells(args.results_dir, args.checkpoints_dir, declaration["asr_bar"])
    inputs = [
        f"{args.results_dir}/swin_*/psbd_metrics.json ({len(cells)} implanted cells)",
        f"{args.checkpoints_dir}/swin_*/args.json",
    ]
    by_dataset = collections.Counter(cell["dataset"] for cell in cells)

    rows, stats = placement_table(cells)
    write_table(
        path=os.path.join(args.paper_dir, "tables", "swin_placements.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Swin-S: mean AUROC at the headline quantile for every cached placement "
            "with at least "
            f"{MIN_CELLS_FOR_ROW} implanted models, at the matched and adaptive rules. "
            "Models are Swin folders whose sidecar attack success clears the bar, "
            + ", ".join(
                f"{count} on {dataset_label(dataset)}"
                for dataset, count in sorted(by_dataset.items())
            )
            + ". Coverage is uneven across placements, so rows are not paired."
        ),
        label="tab:swin-placements",
        header=[
            "placement",
            "n match",
            "AUROC matched",
            "n adapt",
            "AUROC adaptive",
            "below chance",
        ],
        rows=rows,
        align="lrrrrr",
    )

    gain_rows = []
    macros = {
        "swin_cells": (
            str(len(cells)),
            "Swin cells whose attack cleared the bar and carry PSBD metrics",
        ),
        "swin_datasets": (
            str(len(by_dataset)),
            "datasets with at least 1 implanted Swin cell",
        ),
    }
    for stem, label, placement_a, placement_b in (
        (
            "swin_gain_recommended_minus_published",
            "token mask, attention input minus dropout, after residual add",
            RECOMMENDED_PLACEMENT,
            PUBLISHED_PLACEMENT,
        ),
        (
            "swin_gain_dropout_input_minus_published",
            "dropout, attention input minus dropout, after residual add",
            SWIN_DROPOUT_INPUT,
            PUBLISHED_PLACEMENT,
        ),
        (
            "swin_gain_recommended_minus_pre_residual",
            "token mask, attention input minus pre_residual",
            RECOMMENDED_PLACEMENT,
            "pre_residual",
        ),
    ):
        for rule in RULES:
            deltas = paired_gain(cells, placement_a, placement_b, rule)
            low, high = bootstrap_ci(deltas, args.bootstrap, args.seed)
            gain_rows.append(
                [
                    label,
                    rule,
                    str(len(deltas)),
                    fmt(mean_or_none(deltas), signed=True),
                    ci_text(low, high),
                ]
            )
            if rule == "adaptive":
                macros[stem] = (
                    fmt(mean_or_none(deltas), signed=True),
                    f"Swin paired AUROC gain, {label}, adaptive rule, over {len(deltas)} cells",
                )
                macros[f"{stem}_n"] = (str(len(deltas)), f"cells behind {stem}")
                macros[f"{stem}_low"] = (
                    fmt(low, signed=True),
                    f"lower bootstrap bound of {stem}",
                )
                macros[f"{stem}_high"] = (
                    fmt(high, signed=True),
                    f"upper bootstrap bound of {stem}",
                )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "swin_gains.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Swin-S paired placement gains at the headline quantile over the models "
            f"carrying both placements, {args.bootstrap}-resample bootstrap intervals."
        ),
        label="tab:swin-gains",
        header=["comparison", "rule", "n", "mean delta AUROC", "95% CI"],
        rows=gain_rows,
        align="llrrl",
    )
    seed_replicates = len(
        glob.glob(os.path.join(args.checkpoints_dir, "swin_*_seed_*", "args.json"))
    )
    macros["swin_seed_replicates"] = (
        str(seed_replicates),
        "Swin training-seed replicate checkpoints on disk",
    )
    recommended = stats.get(RECOMMENDED_PLACEMENT, {}).get("adaptive", {})
    macros["swin_recommended_auroc_adaptive"] = (
        fmt(recommended.get("mean")),
        f"Swin mean AUROC of the recommended placement at the adaptive rule over {recommended.get('n', 0)} cells",
    )
    macros["swin_recommended_n"] = (
        str(recommended.get("n", 0)),
        "Swin cells carrying the recommended placement",
    )
    write_macros(
        os.path.join(args.paper_dir, "tables", "swin.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(f"swin: {len(cells)} cells {dict(by_dataset)}, recommended {recommended}")


if __name__ == "__main__":
    main()
