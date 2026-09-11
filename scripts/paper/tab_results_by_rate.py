"""Detection results by poison rate, modelled on the PSBD paper's Table 1.

1 `table*` per poison rate (1%, 5%, 10%), rows grouped by dataset, 1 row per
attack that was trained at that rate on that dataset: attack success rate and
clean accuracy from results/coverage/coverage.json, then AUROC and TPR at the
0.10 and 0.20 clean-validation quantiles for 2 placements, both read at the
adaptive rule with `cli.compare.detectors_psbd_values`, the same reader
`tab_headline.py` uses. PSBD-TM is `before_attention_norm_token_mask`, PSBD-RD
is `post_residual`. A checkpoint below the attack success bar keeps its ASR and
CA and prints "--" in every detection column. A checkpoint that cleared the bar
but has no sweep yet prints "pending" instead. The higher of the 2 placements'
AUROC is bolded per row, and no AUROC is ever flipped.

`results_benign.tex` carries 1 row per dataset: the benign reference model's
clean accuracy and how many of that dataset's checkpoints cleared the bar.

    PYTHONPATH=. python scripts/paper/tab_results_by_rate.py \
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare import detectors_psbd_values  # noqa: E402
from defences.decision import (  # noqa: E402
    EASY_ATTACKS,
    HARD_ATTACKS,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
)
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    attack_label,
    build_parser,
    dataset_label,
    fmt,
    load_coverage,
    load_psbd_metrics,
    provenance_comment,
    tex_escape,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_results_by_rate.py"
DATASET_ORDER = ("cifar10", "cifar100", "gtsrb", "tiny", "svhn", "eurosat")
ATTACK_ORDER = EASY_ATTACKS + HARD_ATTACKS
RATE_VALUES = (0.01, 0.05, 0.10)
RATE_PCT_TEXT = {0.01: "1", 0.05: "5", 0.10: "10"}
TPR_QUANTILE_LOW = 0.10
TPR_QUANTILE_HIGH = 0.20
DETECTION_HEADER_ROWS = (
    r" & & & \multicolumn{3}{c}{PSBD-TM} & \multicolumn{3}{c}{PSBD-RD} \\",
    r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}",
    r"Attack & ASR & CA & AUROC & \multicolumn{2}{c}{TPR@FPR} & AUROC & "
    r"\multicolumn{2}{c}{TPR@FPR} \\",
    r"\cmidrule(lr){5-6} \cmidrule(lr){8-9}",
    r" & & & & 10\% & 20\% & & 10\% & 20\% \\",
)


def quantile_key(quantile: float) -> str:
    """The psbd_metrics.json quantile key, matching HEADLINE_KEY's own formatting."""
    key = f"q{quantile:.2f}"
    return key


def bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def placement_reading(report: dict, placement: str) -> tuple[list[str], float | None]:
    """AUROC, TPR@10% and TPR@20% of 1 placement at the adaptive rule, plus the raw AUROC.

    The 3 cells are "pending" and the raw AUROC is None when the sweep has not
    reached this placement, so the caller can tell a missing reading apart from
    a reading of exactly 0.
    """
    block = detectors_psbd_values(report, placement, "adaptive")
    if block is None:
        return ["pending", "pending", "pending"], None

    auroc = block[HEADLINE_KEY]["auroc"]
    tpr_low = block[quantile_key(TPR_QUANTILE_LOW)]["tpr"]
    tpr_high = block[quantile_key(TPR_QUANTILE_HIGH)]["tpr"]
    cells = [fmt(auroc), fmt(tpr_low), fmt(tpr_high)]
    return cells, auroc


def build_row(
    cell: dict,
    results_dir: str,
    missing_sweep: list[str],
    partial_missing: list[tuple[str, str]],
) -> list[str]:
    """1 table row: attack, ASR, CA, then the 2 placements' AUROC and TPR pair."""
    asr_text = fmt(cell.get("asr"))
    clean_accuracy_text = fmt(cell.get("clean_accuracy"))
    leading = [attack_label(cell["attack"]), asr_text, clean_accuracy_text]

    if cell["asr_class"] != "clears":
        row = leading + ["--"] * 6
        return row

    report = load_psbd_metrics(results_dir, cell["folder_name"])
    if report is None:
        missing_sweep.append(cell["folder_name"])
        row = leading + ["pending"] * 6
        return row

    token_mask_cells, token_mask_auroc = placement_reading(
        report, RECOMMENDED_PLACEMENT
    )
    residual_dropout_cells, residual_dropout_auroc = placement_reading(
        report, PUBLISHED_PLACEMENT
    )
    if token_mask_auroc is None:
        partial_missing.append((cell["folder_name"], RECOMMENDED_PLACEMENT))
    if residual_dropout_auroc is None:
        partial_missing.append((cell["folder_name"], PUBLISHED_PLACEMENT))

    if token_mask_auroc is not None and residual_dropout_auroc is not None:
        if token_mask_auroc > residual_dropout_auroc:
            token_mask_cells[0] = bold(token_mask_cells[0])
        elif residual_dropout_auroc > token_mask_auroc:
            residual_dropout_cells[0] = bold(residual_dropout_cells[0])

    row = leading + token_mask_cells + residual_dropout_cells
    return row


def rate_dataset_groups(
    cells_by_key: dict[tuple[str, str, float], dict],
    rate: float,
    results_dir: str,
    missing_sweep: list[str],
    partial_missing: list[tuple[str, str]],
) -> list[tuple[str, list[list[str]]]]:
    """(dataset display name, rows) for every dataset with at least 1 trained attack at this rate."""
    groups = []
    for dataset in DATASET_ORDER:
        rows = []
        for attack in ATTACK_ORDER:
            cell = cells_by_key.get((dataset, attack, rate))
            if cell is None or cell.get("asr") is None:
                continue
            rows.append(build_row(cell, results_dir, missing_sweep, partial_missing))
        if rows:
            groups.append((dataset_label(dataset), rows))
    return groups


def write_rate_table(
    path: str,
    generator: str,
    inputs: list[str],
    caption: str,
    label: str,
    groups: list[tuple[str, list[list[str]]]],
) -> None:
    """A `table*` with a 2-level header and 1 dataset-name row per group."""
    lines = [
        provenance_comment(generator, inputs),
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        *DETECTION_HEADER_ROWS,
        r"\midrule",
    ]
    for index, (dataset_name, rows) in enumerate(groups):
        if index > 0:
            lines.append(r"\midrule")
        lines.append(
            f"\\multicolumn{{9}}{{l}}{{\\textit{{{tex_escape(dataset_name)}}}}} \\\\"
        )
        for row in rows:
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def benign_rows(coverage: dict, cells: list[dict]) -> list[list[str]]:
    """1 row per dataset: benign clean accuracy and how many checkpoints cleared the bar."""
    rows = []
    for dataset in DATASET_ORDER:
        benign_accuracy = coverage["benign_reference_accuracy"].get(dataset)
        if benign_accuracy is None:
            continue
        clearing_count = sum(
            1
            for cell in cells
            if cell["dataset"] == dataset and cell["asr_class"] == "clears"
        )
        rows.append([dataset_label(dataset), fmt(benign_accuracy), str(clearing_count)])
    return rows


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    coverage = load_coverage(args.results_dir)
    cells = coverage["cells"]
    cells_by_key = {
        (cell["dataset"], cell["attack"], cell["poison_rate"]): cell for cell in cells
    }

    inputs = [coverage_path, f"{args.results_dir}/<folder>/psbd_metrics.json"]
    missing_sweep: list[str] = []
    partial_missing: list[tuple[str, str]] = []
    row_counts: dict[float, int] = {}

    for rate in RATE_VALUES:
        groups = rate_dataset_groups(
            cells_by_key, rate, args.results_dir, missing_sweep, partial_missing
        )
        row_counts[rate] = sum(len(rows) for _, rows in groups)
        pct = RATE_PCT_TEXT[rate]
        write_rate_table(
            path=os.path.join(args.paper_dir, "tables", f"results_rate_{pct}.tex"),
            generator=GENERATOR,
            inputs=inputs,
            caption=f"Detection results at {pct}\\% poisoning on ViT-B/16.",
            label=f"tab:results-rate-{pct}",
            groups=groups,
        )

    write_table(
        path=os.path.join(args.paper_dir, "tables", "results_benign.tex"),
        generator=GENERATOR,
        inputs=[coverage_path],
        caption="Benign reference models.",
        label="tab:results-benign",
        header=["Dataset", "Clean Accuracy", "Attacks Above ASR Bar"],
        rows=benign_rows(coverage, cells),
        align="lrr",
    )

    below_bar_rows = sum(
        1
        for rate in RATE_VALUES
        for key, cell in cells_by_key.items()
        if key[2] == rate and cell["asr_class"] != "clears"
    )
    macros = {
        "results_rate_1_rows": (
            str(row_counts[0.01]),
            "rows in the 1% poisoning table",
        ),
        "results_rate_5_rows": (
            str(row_counts[0.05]),
            "rows in the 5% poisoning table",
        ),
        "results_rate_10_rows": (
            str(row_counts[0.10]),
            "rows in the 10% poisoning table",
        ),
        "results_below_bar_rows": (
            str(below_bar_rows),
            "rows across the 3 rate tables printed with -- in every detection "
            "column because the attack did not clear the ASR bar",
        ),
        "results_missing_sweep": (
            str(len(missing_sweep)),
            "checkpoints that cleared the ASR bar but have no psbd_metrics.json "
            "yet, printed as pending across the 3 rate tables",
        ),
        "results_partial_missing": (
            str(len(partial_missing)),
            "checkpoint, placement pairs where the sweep exists but the token-mask "
            "or the post-residual dropout placement has no reading yet",
        ),
        "results_benign_datasets": (
            str(len(benign_rows(coverage, cells))),
            "datasets with a benign reference model",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "results_by_rate.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"results_by_rate: rows 1%={row_counts[0.01]} 5%={row_counts[0.05]} "
        f"10%={row_counts[0.10]}, missing sweep {len(missing_sweep)}, "
        f"partial missing {len(partial_missing)}"
    )
    if missing_sweep:
        print(
            "checkpoints with no sweep at all: " + ", ".join(sorted(set(missing_sweep)))
        )
    if partial_missing:
        print(
            "checkpoint/placement pairs with a sweep but no reading: "
            + ", ".join(
                f"{folder}/{placement}" for folder, placement in partial_missing
            )
        )


if __name__ == "__main__":
    main()
