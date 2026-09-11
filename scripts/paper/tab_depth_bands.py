"""T6: the depth bands, input-side and residual-adjacent, against their all-blocks placement.

Input-side: `before_attention_norm_token_mask` at all blocks, banded to blocks
5-8 and 9-12. Blocks 1-4 has no cache anywhere in the 65-cell panel (verified
against every cell's psbd_metrics.json placements dict), so that row prints
`\\pending` in every column rather than a computed number.

Residual-adjacent: `pre_residual` at all blocks, banded to 1-4, 5-8 and 9-12,
the only family with a full 4-band ladder cached.

Every number is read at the matched 0.6 rate, the cross-placement comparison
device, since a band changes how many blocks the perturbation touches and so
changes what a shared nominal rate even means.

    PYTHONPATH=. python scripts/paper/tab_depth_bands.py \
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_rate  # noqa: E402
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
    rate_row,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_depth_bands.py"
PENDING = r"\pending"

INPUT_SIDE_ALL = "before_attention_norm_token_mask"
INPUT_SIDE_5_8 = "before_attention_norm_blocks_5_8_token_mask"
INPUT_SIDE_9_12 = "before_attention_norm_blocks_9_12_token_mask"
INPUT_SIDE_1_4 = "before_attention_norm_blocks_1_4_token_mask"

RESIDUAL_ALL = "pre_residual"
RESIDUAL_1_4 = "pre_residual_blocks_1_4"
RESIDUAL_5_8 = "pre_residual_blocks_5_8"
RESIDUAL_9_12 = "pre_residual_blocks_9_12"

# label, placement id, all-blocks reference of its family (None for the reference itself).
ROWS = (
    ("input-side, all blocks", INPUT_SIDE_ALL, None),
    ("input-side, blocks 5-8", INPUT_SIDE_5_8, INPUT_SIDE_ALL),
    ("input-side, blocks 9-12", INPUT_SIDE_9_12, INPUT_SIDE_ALL),
    ("input-side, blocks 1-4", INPUT_SIDE_1_4, INPUT_SIDE_ALL),
    ("residual-adjacent, all blocks", RESIDUAL_ALL, None),
    ("residual-adjacent, blocks 1-4", RESIDUAL_1_4, RESIDUAL_ALL),
    ("residual-adjacent, blocks 5-8", RESIDUAL_5_8, RESIDUAL_ALL),
    ("residual-adjacent, blocks 9-12", RESIDUAL_9_12, RESIDUAL_ALL),
)
PLACEMENT_IDS = tuple(
    sorted({row[1] for row in ROWS} | {row[2] for row in ROWS if row[2]})
)


def measure_placement(
    report: dict | None, placement_id: str
) -> dict[str, float] | None:
    """This placement's matched-06 AUROC and achieved clean-validation shift ratio."""
    if report is None or placement_id == INPUT_SIDE_1_4:
        return None
    block = report.get("placements", {}).get(placement_id)
    if block is None:
        return None
    rate = psbd_rate(block, "matched")
    if rate is None:
        return None
    row = rate_row(block, rate)
    if row is None:
        return None
    measured = {
        "auroc": row["detection_psu_ratio"].get(HEADLINE_KEY, {}).get("auroc"),
        "achieved_shift": row["shift_ratio"]["validation"],
    }
    return measured


def paired_deltas(
    cells: list[dict], placement_id: str, reference_id: str
) -> list[float]:
    deltas = []
    for cell in cells:
        band = cell["bands"][placement_id]
        reference = cell["bands"][reference_id]
        if band is not None and reference is not None:
            deltas.append(band["auroc"] - reference["auroc"])
    return deltas


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    coverage = load_coverage(args.results_dir)
    cells = clearing_cells(coverage)

    for cell in cells:
        report = load_psbd_metrics(args.results_dir, cell["folder_name"])
        cell["bands"] = {
            placement_id: measure_placement(report, placement_id)
            for placement_id in PLACEMENT_IDS
        }

    inputs = [
        coverage_path,
        f"{args.results_dir}/<folder>/psbd_metrics.json (65 cells)",
    ]

    table_rows = []
    macros = {}
    macro_stem_of = {
        (INPUT_SIDE_5_8, INPUT_SIDE_ALL): "band_5_8_minus_all_input_side",
        (INPUT_SIDE_9_12, INPUT_SIDE_ALL): "band_9_12_minus_all_input_side",
        (RESIDUAL_1_4, RESIDUAL_ALL): "band_1_4_minus_all_residual",
        (RESIDUAL_5_8, RESIDUAL_ALL): "band_5_8_minus_all_residual",
        (RESIDUAL_9_12, RESIDUAL_ALL): "band_9_12_minus_all_residual",
    }
    for label, placement_id, reference_id in ROWS:
        if placement_id == INPUT_SIDE_1_4:
            table_rows.append([label] + [PENDING] * 5)
            continue

        aurocs = [
            cell["bands"][placement_id]["auroc"]
            for cell in cells
            if cell["bands"][placement_id] is not None
        ]
        shifts = [
            cell["bands"][placement_id]["achieved_shift"]
            for cell in cells
            if cell["bands"][placement_id] is not None
        ]
        if reference_id is None:
            delta_text, ci = "--", "--"
        else:
            deltas = paired_deltas(cells, placement_id, reference_id)
            low, high = bootstrap_ci(deltas, args.bootstrap, args.seed)
            delta_text = fmt(mean_or_none(deltas), signed=True)
            ci = ci_text(low, high)
            stem = macro_stem_of[(placement_id, reference_id)]
            macros[stem] = (
                delta_text,
                f"mean paired AUROC delta, `{placement_id}` minus `{reference_id}`, "
                f"at the matched 0.6 rate, over the {len(deltas)} cells it covers",
            )
            macros[f"{stem}_low"] = (
                fmt(low, signed=True),
                f"lower bound of the 95% bootstrap interval on {stem}",
            )
            macros[f"{stem}_high"] = (
                fmt(high, signed=True),
                f"upper bound of the 95% bootstrap interval on {stem}",
            )

        table_rows.append(
            [
                label,
                str(len(aurocs)),
                fmt(mean_or_none(aurocs)),
                delta_text,
                ci,
                fmt(mean_or_none(shifts)),
            ]
        )

    write_table(
        path=os.path.join(args.paper_dir, "tables", "depth_bands.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Depth-band placements at the matched 0.6 rate, mean AUROC at q0.25, "
            "the paired delta against the family's all-blocks placement with a "
            f"{args.bootstrap}-resample bootstrap 95\\% interval, and the mean "
            "achieved clean-validation shift ratio at the chosen rate. "
            "\\code{before\\_attention\\_norm\\_blocks\\_1\\_4\\_token\\_mask} has no cache anywhere "
            r"in the 65-cell panel and prints \pending throughout."
        ),
        label="tab:depth-bands",
        header=[
            "placement",
            "n",
            "mean AUROC matched06",
            "delta vs all blocks",
            "95% CI",
            "mean achieved shift",
        ],
        rows=table_rows,
        align="lrrrlr",
    )

    write_macros(
        os.path.join(args.paper_dir, "tables", "depth_bands.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    headline_stems = [stem for stem in macros if not stem.endswith(("_low", "_high"))]
    print(
        "depth_bands: "
        + ", ".join(f"{stem}={macros[stem][0]}" for stem in headline_stems)
    )


if __name__ == "__main__":
    main()
