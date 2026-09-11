"""The results section as a staircase: 4 tables reading 1 axis at a time.

Every row reads the same panel of 65 backdoored ViT models, the clearing cells
that carry the full basis (exactly the models tab_headline.py's common
coverage selects: every clearing cell where the attention-input token mask
site and the paper's own ConvNet site both have a cache). Every column reads
the adaptive 0.8 rule only. AUROC and the below-chance count are at the
headline quantile q0.25, the 2 TPR columns at q0.10 and q0.20.

Table 1 (staircase_residual) transfers the paper's own site literally: dropout
after both residual adds, against dropout confined to the attention add only
and dropout moved before both adds, banded to blocks 1 to 4, 5 to 8 and 9 to
12. Table 2 (staircase_dropout_sites) keeps dropout as the perturbing operator
and moves it to the other sites the network offers. Table 3
(staircase_operators) holds the attention input fixed and changes the operator,
then holds the operator (token mask) fixed and changes the site over the whole
network. Table 4 (staircase_bands) restricts the winning site and operator to
depth bands.

A row's site has no cache anywhere in the panel, it prints \\pending in every
numeric column rather than being dropped, so a reader always sees the same 4
table shapes regardless of which sweeps have finished. A site that already has
a partial cache prints its real numbers at whatever count of models it covers.

Every row's gain is paired within model against its own table's first row,
1 column holding the mean and its bootstrap 95% interval together.

    PYTHONPATH=. python scripts/paper/tab_staircase.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare import detectors_psbd_values as psbd_values  # noqa: E402
from defences.decision import (  # noqa: E402
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
)
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
    provenance_comment,
    tex_escape,
    write_macros,
)

GENERATOR = "scripts/paper/tab_staircase.py"
PENDING = r"\pending"
TPR_10_KEY = "q0.10"
TPR_20_KEY = "q0.20"
N_DATA_COLUMNS = 5  # n, AUROC, TPR at 10%, TPR at 20%, gain [CI]

# label, placement id. Row 0 of every table is that table's own reference row,
# read by every other row's paired gain.
RESIDUAL_ROWS = (
    ("dropout, after both residual adds", "post_residual"),
    ("dropout, after the attention add only", "after_attention_residual"),
    ("dropout, before both residual adds", "pre_residual"),
    ("dropout, before both residual adds, blocks 1 to 4", "pre_residual_blocks_1_4"),
    ("dropout, before both residual adds, blocks 5 to 8", "pre_residual_blocks_5_8"),
    ("dropout, before both residual adds, blocks 9 to 12", "pre_residual_blocks_9_12"),
)
DROPOUT_SITE_ROWS = (
    ("dropout, after both residual adds", "post_residual"),
    ("dropout, before both residual adds", "pre_residual"),
    ("dropout, attention input", "before_attention_norm"),
    ("dropout, attention input after norm", "before_attention"),
    ("dropout, MLP input", "before_mlp_norm"),
    ("dropout, embedding output", "after_embedding"),
)
OPERATOR_ROWS = (
    ("token mask, attention input", "before_attention_norm_token_mask"),
    ("channel mask, attention input", "before_attention_norm_channel_mask"),
    ("gaussian, attention input", "before_attention_norm_gaussian"),
    ("dropout, attention input", "before_attention_norm"),
    (
        "token mask, attention output before the add",
        "before_attention_residual_token_mask",
    ),
    ("token mask, both sublayer inputs", "both_sublayer_inputs_token_mask"),
    ("token mask, MLP input", "before_mlp_norm_token_mask"),
    ("gaussian, MLP input after norm", "before_mlp_gaussian"),
    ("token mask, after the attention add", "after_attention_residual_token_mask"),
    ("channel mask, MLP neurons", "mlp_neurons_channel_mask"),
    ("gain scale, MLP norm output", "mlp_norm_out_gain_scale"),
    ("scale up, input pixels", "input_pixels_scale_up"),
    ("dropout, after both residual adds", "post_residual"),
)
BAND_ROWS = (
    ("token mask, attention input, all 12 blocks", "before_attention_norm_token_mask"),
    (
        "token mask, attention input, blocks 1 to 4",
        "before_attention_norm_blocks_1_4_token_mask",
    ),
    (
        "token mask, attention input, blocks 5 to 8",
        "before_attention_norm_blocks_5_8_token_mask",
    ),
    (
        "token mask, attention input, blocks 9 to 12",
        "before_attention_norm_blocks_9_12_token_mask",
    ),
)

# table key, file stem, label, 1-sentence caption, rows.
TABLES = (
    (
        "residual",
        "staircase_residual",
        "tab:staircase-residual",
        "Dropout on the residual stream of ViT-B/16, after both residual adds, "
        "after the attention add only, before both adds, and restricted to bands of blocks.",
        RESIDUAL_ROWS,
    ),
    (
        "dropout_sites",
        "staircase_dropout_sites",
        "tab:staircase-dropout-sites",
        "Dropout at each site of the ViT-B/16 block, applied in all 12 blocks.",
        DROPOUT_SITE_ROWS,
    ),
    (
        "operators",
        "staircase_operators",
        "tab:staircase-operators",
        "Perturbation operators at the attention input of ViT-B/16, and token "
        "masking at the other sites, all 12 blocks.",
        OPERATOR_ROWS,
    ),
    (
        "bands",
        "staircase_bands",
        "tab:staircase-bands",
        "Token masking at the attention input of ViT-B/16, restricted to bands of blocks.",
        BAND_ROWS,
    ),
)

ALL_PLACEMENT_IDS = sorted(
    {
        placement_id
        for _table_key, _stem, _label, _caption, rows in TABLES
        for _label_text, placement_id in rows
    }
)


def panel_cells(results_dir: str, coverage: dict) -> list[dict]:
    """The 65 clearing cells common coverage selects, each carrying its report.

    Filters the clearing cells down to the ones where the attention-input token
    mask site and the paper's own ConvNet site both returned a value at the
    adaptive and matched rules, the same common-coverage rule
    scripts/paper/tab_headline.py applies. This is not re-implemented by
    calling into tab_headline.py, which stays untouched, but the rule is
    identical so both generators name the same 65 models.
    """
    selected = []
    for cell in clearing_cells(coverage):
        report = load_psbd_metrics(results_dir, cell["folder_name"])
        rec_adaptive = psbd_values(report, RECOMMENDED_PLACEMENT, "adaptive")
        rec_matched = psbd_values(report, RECOMMENDED_PLACEMENT, "matched")
        pub_adaptive = psbd_values(report, PUBLISHED_PLACEMENT, "adaptive")
        if (
            rec_adaptive is not None
            and rec_matched is not None
            and pub_adaptive is not None
        ):
            cell["report"] = report
            selected.append(cell)
    return selected


def measure_placement(
    report: dict | None, placement_id: str
) -> dict[str, float | None]:
    """1 model's adaptive-rule headline AUROC and its TPR at the 10% and 20% quantiles.

    Every field reads the same adaptive-rate block, so a field is None only
    where that block is missing the quantile, which the row aggregates then
    drop rather than treat as 0.
    """
    adaptive_block = psbd_values(report, placement_id, "adaptive")
    measured = {
        "auroc": adaptive_block[HEADLINE_KEY]["auroc"] if adaptive_block else None,
        "tpr_10": adaptive_block.get(TPR_10_KEY, {}).get("tpr")
        if adaptive_block
        else None,
        "tpr_20": adaptive_block.get(TPR_20_KEY, {}).get("tpr")
        if adaptive_block
        else None,
    }
    return measured


def placement_values(cells: list[dict], placement_id: str, key: str) -> list[float]:
    """1 metric of 1 site over the panel, models the site has no cache for dropped."""
    values = [cell["staircase"][placement_id][key] for cell in cells]
    present = [value for value in values if value is not None]
    return present


def paired_deltas(
    cells: list[dict], placement_id: str, reference_id: str
) -> list[float]:
    """Headline adaptive-rule AUROC, this site minus the table's reference site, paired within model."""
    deltas = []
    for cell in cells:
        value = cell["staircase"][placement_id]["auroc"]
        reference_value = cell["staircase"][reference_id]["auroc"]
        if value is not None and reference_value is not None:
            deltas.append(value - reference_value)
    return deltas


def is_pending(cells: list[dict], placement_id: str) -> bool:
    """Whether this site has no adaptive-rule cache anywhere in the panel."""
    pending = not placement_values(cells, placement_id, "auroc")
    return pending


def gain_cell_text(mean_text: str, interval_text: str) -> str:
    """The gain column: mean and interval together, or the mean alone when the interval is undefined."""
    combined = mean_text if interval_text == "--" else f"{mean_text} {interval_text}"
    return combined


def build_row(
    cells: list[dict],
    label: str,
    placement_id: str,
    reference_id: str,
    table_key: str,
    resamples: int,
    seed: int,
) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """1 table row and the macros it defines, or a fully pending row with none."""
    if is_pending(cells, placement_id):
        row = [label] + [PENDING] * N_DATA_COLUMNS
        return row, {}

    auroc = placement_values(cells, placement_id, "auroc")
    tpr_10 = placement_values(cells, placement_id, "tpr_10")
    tpr_20 = placement_values(cells, placement_id, "tpr_20")
    below_chance = sum(1 for value in auroc if value < 0.5)

    is_reference_row = placement_id == reference_id
    if is_reference_row:
        gain_text, interval_text = "--", "--"
    else:
        deltas = paired_deltas(cells, placement_id, reference_id)
        low, high = bootstrap_ci(deltas, resamples, seed)
        gain_text = fmt(mean_or_none(deltas), signed=True)
        interval_text = ci_text(low, high)

    row = [
        label,
        str(len(auroc)),
        fmt(mean_or_none(auroc)),
        fmt(mean_or_none(tpr_10)),
        fmt(mean_or_none(tpr_20)),
        gain_cell_text(gain_text, interval_text),
    ]

    stem = f"staircase_{table_key}_{placement_id}"
    macros = {
        f"{stem}_n": (
            str(len(auroc)),
            f"models covered at the adaptive 0.8 rule, {label}, staircase {table_key} table",
        ),
        f"{stem}_auroc": (
            fmt(mean_or_none(auroc)),
            f"mean AUROC at q0.25 at the adaptive 0.8 rule, {label}, over the "
            f"{len(auroc)} models it covers, staircase {table_key} table",
        ),
        f"{stem}_tpr_at_10_percent": (
            fmt(mean_or_none(tpr_10)),
            f"mean TPR at the q0.10 clean-validation quantile at the adaptive 0.8 "
            f"rule, {label}, staircase {table_key} table",
        ),
        f"{stem}_tpr_at_20_percent": (
            fmt(mean_or_none(tpr_20)),
            f"mean TPR at the q0.20 clean-validation quantile at the adaptive 0.8 "
            f"rule, {label}, staircase {table_key} table",
        ),
        f"{stem}_below_chance": (
            str(below_chance),
            f"models scoring AUROC under 0.5 at q0.25 at the adaptive 0.8 rule, {label}, "
            f"staircase {table_key} table",
        ),
    }
    if not is_reference_row:
        macros[f"{stem}_gain"] = (
            gain_text,
            f"mean paired AUROC gain at q0.25 at the adaptive 0.8 rule, {label} against "
            f"this table's first row, over the {len(deltas)} models it covers, "
            f"staircase {table_key} table",
        )
        macros[f"{stem}_gain_low"] = (
            fmt(low, signed=True),
            f"lower bound of the 95% bootstrap interval on {stem}_gain",
        )
        macros[f"{stem}_gain_high"] = (
            fmt(high, signed=True),
            f"upper bound of the 95% bootstrap interval on {stem}_gain",
        )
    return row, macros


def write_staircase_table(
    path: str,
    generator: str,
    inputs: list[str],
    caption: str,
    label: str,
    rows: list[list[str]],
) -> None:
    """A staircase table's own booktabs file, its header spanning 2 rows.

    write_table (scripts/paper/_common.py) takes a single header row, and this
    layout needs a \\cmidrule spanning the 2 TPR columns above their own "10%"
    and "20%" labels, so the file is assembled directly instead.
    """
    lines = [
        provenance_comment(generator, inputs),
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{TPR@FPR} & \\",
        r"\cmidrule(lr){4-5}",
        r"Placement & n & AUROC & 10\% & 20\% & Gain [95\% CI] \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(cell) for cell in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    coverage = load_coverage(args.results_dir)
    cells = panel_cells(args.results_dir, coverage)

    for cell in cells:
        cell["staircase"] = {
            placement_id: measure_placement(cell["report"], placement_id)
            for placement_id in ALL_PLACEMENT_IDS
        }

    inputs = [
        coverage_path,
        f"{args.results_dir}/<folder>/psbd_metrics.json (65 models)",
    ]

    all_macros = {}
    for table_key, stem, label, caption, rows in TABLES:
        reference_id = rows[0][1]
        table_rows = []
        for row_label, placement_id in rows:
            row, macros = build_row(
                cells,
                row_label,
                placement_id,
                reference_id,
                table_key,
                args.bootstrap,
                args.seed,
            )
            table_rows.append(row)
            all_macros.update(macros)

        write_staircase_table(
            path=os.path.join(args.paper_dir, "tables", f"{stem}.tex"),
            generator=GENERATOR,
            inputs=inputs,
            caption=caption,
            label=label,
            rows=table_rows,
        )

    write_macros(
        os.path.join(args.paper_dir, "tables", "staircase.macros.json"),
        GENERATOR,
        inputs,
        all_macros,
    )
    print(
        f"staircase: {len(cells)} models, {len(all_macros)} macros over {len(TABLES)} tables"
    )


if __name__ == "__main__":
    main()
