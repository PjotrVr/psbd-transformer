"""Every detector against every attack, in the layout the PSBD paper uses.

1 row per backdoored model grouped by dataset, and 1 column per defense with
ours first. That is the transpose of the earlier tables here, and it is the
orientation a reader wants: a person asking "does anything catch WaNet" reads
across 1 line instead of down 12.

3 tables, 1 per reported metric, because 12 defenses times 3 metrics does not fit
a page. AUROC is the one-sided area under the ROC curve, never flipped, so a
value below 0.5 means that detector ordered poisoned and clean inputs the wrong
way round on that model and is printed gray. TPR is at the 10% and 20%
false-positive budgets, both set on the same 2000 clean validation images every
detector and PSBD share.

Best in each row is bold and second best underlined, over the defenses only.

    PYTHONPATH=. python scripts/paper/tab_detectors.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import collections
import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defences.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from detectors import DETECTOR_NAMES  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    attack_label,
    build_parser,
    dataset_label,
    detector_label,
    fmt,
    load_coverage,
    load_json,
    load_psbd_metrics,
    mean_or_none,
    write_macros,
    write_wide_table,
)

GENERATOR = "scripts/paper/tab_detectors.py"
# The deployable rate rule, the only one a competitor comparison may use: the
# matched rule exists to compare placements to each other, not to other methods.
RULE = "adaptive"
# Our 2 columns: the placement this paper recommends and the ConvNet placement
# the PSBD paper published, so a reader sees the method and its own baseline.
OURS = (
    (RECOMMENDED_PLACEMENT, "PSBD-TM"),
    (PUBLISHED_PLACEMENT, "PSBD-RD"),
)
# Reported metric, as (stem, quantile ladder key, field, caption words).
METRICS = (
    ("auroc", HEADLINE_KEY, "auroc", "AUROC"),
    ("tpr10", "q0.10", "tpr", "TPR at a 10\\% false-positive budget"),
    ("tpr20", "q0.20", "tpr", "TPR at a 20\\% false-positive budget"),
)
# A one-sided AUROC under this ordered the 2 classes the wrong way round.
CHANCE = 0.5
GREY = r"\textcolor{black!45}"
SHADE = r"\rowcolor{black!8}"


def detector_reading(
    results_dir: str, folder: str, detector: str, key: str, field: str
):
    """1 number from a detector's record, or None when it never scored this cell."""
    path = os.path.join(results_dir, folder, "detectors", f"{detector}_metrics.json")
    record = load_json(path)
    if record is None or record.get("status") != "scored":
        return None
    quantile = (record.get("detection") or {}).get(key)
    if quantile is None:
        return None
    return quantile.get(field)


def psbd_reading(report: dict, placement: str, key: str, field: str):
    """The same number from a PSBD placement, so both sides read 1 ladder.

    cli.compare_detectors.psbd_values is the shared reader, because a cache
    written before the quantile ladder existed carries the headline reading
    directly on the rule block and only it knows that shape.
    """
    block = psbd_values(report, placement, RULE)
    if block is None:
        return None
    quantile = block.get(key)
    if quantile is None:
        return None
    return quantile.get(field)


def cell_readings(
    results_dir: str, cell: dict, key: str, field: str
) -> dict[str, float | None]:
    """Every defense's reading of 1 model, keyed by column name."""
    report = load_psbd_metrics(results_dir, cell["folder_name"]) or {}
    readings: dict[str, float | None] = {}
    for placement, column in OURS:
        readings[column] = psbd_reading(report, placement, key, field)
    for detector in DETECTOR_NAMES:
        readings[detector_label(detector)] = detector_reading(
            results_dir, cell["folder_name"], detector, key, field
        )
    return readings


def columns() -> list[str]:
    """Column names in print order, ours first."""
    names = [column for _, column in OURS]
    names += [detector_label(detector) for detector in DETECTOR_NAMES]
    return names


def decorate(value: float | None, best: float | None, second: float | None) -> str:
    """1 cell: bold the best, underline the second, gray anything below chance."""
    if value is None:
        return "--"
    text = fmt(value)
    if value < CHANCE:
        return f"{GREY}{{{text}}}"
    if best is not None and value == best:
        return f"\\textbf{{{text}}}"
    if second is not None and value == second:
        return f"\\underline{{{text}}}"
    return text


def model_row(cell: dict, readings: dict[str, float | None]) -> list[str]:
    """1 model's row: its label, then every defense's decorated reading."""
    present = sorted(
        (value for value in readings.values() if value is not None), reverse=True
    )
    best = present[0] if present else None
    second = next((value for value in present if value != best), None)
    row = [f"{attack_label(cell['attack'])} {cell['poison_rate'] * 100:g}"]
    row += [decorate(readings[column], best, second) for column in columns()]
    return row


def mean_row(label: str, group: list[dict[str, float | None]]) -> list[str]:
    """A shaded row of per-column means over a group of models."""
    means = {
        column: mean_or_none(
            [readings[column] for readings in group if readings[column] is not None]
        )
        for column in columns()
    }
    present = sorted(
        (value for value in means.values() if value is not None), reverse=True
    )
    best = present[0] if present else None
    second = next((value for value in present if value != best), None)
    row = [f"{SHADE}\\emph{{{label}}}"]
    row += [decorate(means[column], best, second) for column in columns()]
    return row


def build_rows(
    results_dir: str, cells: list[dict], key: str, field: str
) -> tuple[list[list[str]], dict[str, float | None]]:
    """The whole table body grouped by dataset, plus the overall per-column means."""
    by_dataset = collections.defaultdict(list)
    for cell in cells:
        by_dataset[cell["dataset"]].append(cell)

    width = len(columns()) + 1
    rows: list[list[str]] = []
    everything: list[dict[str, float | None]] = []
    for dataset in sorted(by_dataset):
        group = sorted(
            by_dataset[dataset],
            key=lambda cell: (cell["attack"], cell["poison_rate"]),
        )
        rows.append(
            [f"\\multicolumn{{{width}}}{{l}}{{\\emph{{{dataset_label(dataset)}}}}}"]
        )
        readings_group = []
        for cell in group:
            readings = cell_readings(results_dir, cell, key, field)
            readings_group.append(readings)
            rows.append(model_row(cell, readings))
        rows.append(mean_row(f"{dataset_label(dataset)} mean", readings_group))
        everything.extend(readings_group)
    rows.append(mean_row("all models", everything))

    overall = {
        column: mean_or_none([r[column] for r in everything if r[column] is not None])
        for column in columns()
    }
    return rows, overall


def metric_macros(stem: str, overall: dict[str, float | None]) -> dict:
    """Our column, the best competitor and the gap between them, for 1 metric."""
    ours = overall["PSBD-TM"]
    competitors = {
        detector_label(name): overall[detector_label(name)]
        for name in DETECTOR_NAMES
        if overall[detector_label(name)] is not None
    }
    best_name, best_value = max(competitors.items(), key=lambda item: item[1])
    ranked = sorted(
        ((value, name) for name, value in overall.items() if value is not None),
        reverse=True,
    )
    order = [name for _, name in ranked]
    macros = {
        f"detectors_{stem}_defenses_ranked": (
            str(len(order)),
            f"defenses the {stem} ranking covers, ours and the competitors together",
        ),
        f"detectors_{stem}_rank_ours": (
            str(order.index("PSBD-TM") + 1),
            f"rank of the recommended placement among every defense by mean {stem}",
        ),
        f"detectors_{stem}_rank_published": (
            str(order.index("PSBD-RD") + 1),
            f"rank of the published placement among every defense by mean {stem}",
        ),
        f"detectors_{stem}_beating_published": (
            str(order.index("PSBD-RD")),
            f"defenses with a higher mean {stem} than the published placement",
        ),
        f"detectors_{stem}_ours": (
            fmt(ours),
            f"mean {stem} of the recommended placement over the compared models",
        ),
        f"detectors_{stem}_published": (
            fmt(overall["PSBD-RD"]),
            f"mean {stem} of the published placement over the same models",
        ),
        f"detectors_{stem}_best_competitor": (
            fmt(best_value),
            f"mean {stem} of the strongest competitor detector, {best_name}",
        ),
        f"detectors_{stem}_best_competitor_name": (
            best_name,
            f"the strongest competitor detector by mean {stem}",
        ),
        f"detectors_{stem}_margin": (
            fmt(None if ours is None else ours - best_value, signed=True),
            f"mean {stem} of the recommended placement minus the strongest competitor",
        ),
        f"detectors_{stem}_sentinet": (
            fmt(overall[detector_label("sentinet")]),
            f"mean {stem} of the SentiNet port, whose Grad-CAM mask misses the trigger",
        ),
        f"detectors_{stem}_ibd_psc": (
            fmt(overall[detector_label("ibd_psc")]),
            f"mean {stem} of IBD-PSC at the paper's fixed amplification factor",
        ),
        f"detectors_{stem}_below_chance_columns": (
            str(sum(1 for value in competitors.values() if value < CHANCE)),
            f"competitor detectors whose mean {stem} sits below chance",
        ),
    }
    return macros


def fully_covered(results_dir: str, cells: list[dict]) -> list[dict]:
    """The cells every compared defense has a reading on.

    A mean per column over whichever cells that column happens to cover makes the
    columns incomparable, and it is how the same placement came to read 0.818 here
    and 0.823 in the body. Every column is read on 1 population instead.
    """
    kept = []
    for cell in cells:
        readings = cell_readings(results_dir, cell, HEADLINE_KEY, "auroc")
        if all(value is not None for value in readings.values()):
            kept.append(cell)
    return kept


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage = load_coverage(args.results_dir)
    clearing = [cell for cell in coverage["cells"] if cell.get("asr_class") == "clears"]
    cells = fully_covered(args.results_dir, clearing)
    inputs = [
        f"{args.results_dir}/coverage/coverage.json "
        f"({len(cells)} of {len(clearing)} clearing cells carry every defense)",
        f"{args.results_dir}/*/detectors/*_metrics.json",
        f"{args.results_dir}/*/psbd_metrics.json",
    ]
    header = ["attack, rate %"] + columns()
    align = "l" + "r" * len(columns())

    macros = {
        "detectors_compared_models": (
            str(len(cells)),
            "backdoored models in the detector comparison",
        ),
        "detectors_compared_count": (
            str(len(DETECTOR_NAMES)),
            "competitor detectors in the comparison",
        ),
    }
    for stem, key, field, words in METRICS:
        rows, overall = build_rows(args.results_dir, cells, key, field)
        write_wide_table(
            path=os.path.join(args.paper_dir, "tables", f"detectors_{stem}.tex"),
            generator=GENERATOR,
            inputs=inputs,
            caption=(
                f"{words} of every defense on every backdoored ViT-B/16 model that "
                f"carries all {len(columns())} of them, {len(cells)} of "
                f"{len(clearing)} models over "
                f"{len({cell['dataset'] for cell in cells})} "
                "datasets. Rows are attacks and columns defenses, ours first. Best in "
                "each row is bold and second best underlined. AUROC is one-sided and "
                "never flipped, so gray marks a reading below chance, where the "
                "detector ordered poisoned and clean inputs the wrong way round. "
                "Shaded rows are means. Scale-Up$^\\dagger$ is the data-limited "
                "variant and IBD-PSC$^\\ast$ the calibrated one."
            ),
            label=f"tab:detectors-{stem}",
            header=header,
            rows=rows,
            align=align,
        )
        macros.update(metric_macros(stem, overall))

    write_macros(
        os.path.join(args.paper_dir, "tables", "detectors.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"detectors: {len(cells)} models, {len(DETECTOR_NAMES)} competitors, "
        f"{len(METRICS)} tables"
    )


if __name__ == "__main__":
    main()
