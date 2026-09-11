"""A2 and C1: the operator axis at before_attention_norm, and 1 input-side site pair.

At `before_attention_norm`, the capacity-removing operators `token_mask` and
`channel_mask` against `gaussian`, which removes no capacity (H23). At
`before_mlp`, `gaussian` against the closest mask at that site,
`before_mlp_norm_token_mask`. Every comparison is paired within cell at the
matched 0.6 rate, the cross-placement comparison device, since a shared
nominal rate is not a shared disturbance across operators (defences.decision).
C1 holds the operator fixed at `token_mask` and moves the site instead,
`before_attention_norm` against `before_mlp_norm`.

    PYTHONPATH=. python scripts/paper/tab_operators.py \
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    bootstrap_ci,
    build_parser,
    ci_text,
    clearing_cells,
    fmt,
    load_coverage,
    load_psbd_metrics,
    mean_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_operators.py"
ATTENTION_NORM_TOKEN_MASK = "before_attention_norm_token_mask"
ATTENTION_NORM_CHANNEL_MASK = "before_attention_norm_channel_mask"
ATTENTION_NORM_GAUSSIAN = "before_attention_norm_gaussian"
MLP_GAUSSIAN = "before_mlp_gaussian"
MLP_NORM_TOKEN_MASK = "before_mlp_norm_token_mask"

PLACEMENTS = (
    (ATTENTION_NORM_TOKEN_MASK, "before_attention_norm", "token_mask"),
    (ATTENTION_NORM_CHANNEL_MASK, "before_attention_norm", "channel_mask"),
    (ATTENTION_NORM_GAUSSIAN, "before_attention_norm", "gaussian"),
    (MLP_GAUSSIAN, "before_mlp", "gaussian"),
    (MLP_NORM_TOKEN_MASK, "before_mlp_norm", "token_mask"),
)
RULES = ("matched", "adaptive")

# Each comparison paired within cell at the matched 0.6 rate. A2 isolates the
# operator with the site held fixed, C1 isolates the site with the operator
# held fixed.
COMPARISONS = (
    (
        "gaussian_minus_token_mask_attention_norm",
        "A2: gaussian minus token_mask at before_attention_norm",
        ATTENTION_NORM_GAUSSIAN,
        ATTENTION_NORM_TOKEN_MASK,
    ),
    (
        "gaussian_minus_channel_mask_attention_norm",
        "A2: gaussian minus channel_mask at before_attention_norm",
        ATTENTION_NORM_GAUSSIAN,
        ATTENTION_NORM_CHANNEL_MASK,
    ),
    (
        "gaussian_minus_token_mask_mlp",
        "A2: gaussian minus token_mask at before_mlp",
        MLP_GAUSSIAN,
        MLP_NORM_TOKEN_MASK,
    ),
    (
        "attention_input_minus_mlp_input_token_mask",
        "C1: before_attention_norm minus before_mlp_norm, both token_mask",
        ATTENTION_NORM_TOKEN_MASK,
        MLP_NORM_TOKEN_MASK,
    ),
)


def measure_cell(results_dir: str, folder: str) -> dict[str, dict[str, float | None]]:
    """This cell's AUROC at q0.25 for every declared placement, at both rules."""
    report = load_psbd_metrics(results_dir, folder)
    auroc_by_placement = {}
    for placement, _site, _operator in PLACEMENTS:
        by_rule = {}
        for rule in RULES:
            block = psbd_values(report, placement, rule)
            by_rule[rule] = (
                block[HEADLINE_KEY]["auroc"]
                if block and HEADLINE_KEY in block
                else None
            )
        auroc_by_placement[placement] = by_rule
    return auroc_by_placement


def placement_values(cells: list[dict], placement: str, rule: str) -> list[float]:
    values = [cell["auroc"][placement][rule] for cell in cells]
    present = [value for value in values if value is not None]
    return present


def paired_deltas(
    cells: list[dict], placement_a: str, placement_b: str, rule: str
) -> list[float]:
    deltas = []
    for cell in cells:
        a = cell["auroc"][placement_a][rule]
        b = cell["auroc"][placement_b][rule]
        if a is not None and b is not None:
            deltas.append(a - b)
    return deltas


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    coverage = load_coverage(args.results_dir)
    cells = clearing_cells(coverage)

    for cell in cells:
        cell["auroc"] = measure_cell(args.results_dir, cell["folder_name"])

    inputs = [
        coverage_path,
        f"{args.results_dir}/<folder>/psbd_metrics.json (65 cells)",
    ]

    mean_rows = []
    for placement, site, operator in PLACEMENTS:
        matched = placement_values(cells, placement, "matched")
        adaptive = placement_values(cells, placement, "adaptive")
        mean_rows.append(
            [
                placement,
                site,
                operator,
                str(len(matched)),
                fmt(mean_or_none(matched)),
                str(len(adaptive)),
                fmt(mean_or_none(adaptive)),
            ]
        )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "operators.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Mean AUROC at q0.25 for each placement of the operator and site "
            "comparison, at the matched 0.6 rate and the adaptive 0.8 rate, over "
            "the panel of 65 backdoored models. n is the number of models the "
            "placement covers under that rule."
        ),
        label="tab:operators",
        header=[
            "placement",
            "site",
            "operator",
            "n match",
            "mean AUROC matched06",
            "n adapt",
            "mean AUROC adaptive08",
        ],
        rows=mean_rows,
        align="lllrrrr",
    )

    delta_rows = []
    macros = {}
    for macro_stem, label, placement_a, placement_b in COMPARISONS:
        deltas = paired_deltas(cells, placement_a, placement_b, "matched")
        low, high = bootstrap_ci(deltas, args.bootstrap, args.seed)
        mean = mean_or_none(deltas)
        delta_rows.append(
            [label, str(len(deltas)), fmt(mean, signed=True), ci_text(low, high)]
        )
        macros[macro_stem] = (
            fmt(mean, signed=True),
            f"mean paired AUROC delta at the matched 0.6 rate, {label}, "
            f"over the {len(deltas)} cells it covers of the 65-cell panel",
        )
        macros[f"{macro_stem}_low"] = (
            fmt(low, signed=True),
            f"lower bound of the 95% bootstrap interval on {macro_stem}",
        )
        macros[f"{macro_stem}_high"] = (
            fmt(high, signed=True),
            f"upper bound of the 95% bootstrap interval on {macro_stem}",
        )

    write_table(
        path=os.path.join(args.paper_dir, "tables", "operators_deltas.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Paired AUROC deltas at the matched 0.6 rate, "
            f"{args.bootstrap}-resample bootstrap 95\\% intervals. A2 isolates the "
            "operator with the site held fixed, C1 isolates the site with the "
            "operator held fixed."
        ),
        label="tab:operators-deltas",
        header=["comparison", "n", "mean delta AUROC", "95% CI"],
        rows=delta_rows,
        align="lrrl",
    )

    write_macros(
        os.path.join(args.paper_dir, "tables", "operators.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        "operators: "
        + ", ".join(f"{stem}={macros[stem][0]}" for stem, *_ in COMPARISONS)
    )


if __name__ == "__main__":
    main()
