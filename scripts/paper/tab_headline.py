"""T2: the recommended placement against the published placement.

3 configurations are read on every one of the 65 clearing cells: the
recommended placement (before_attention_norm_token_mask) at the adaptive 0.8
rule, the same placement at the matched 0.6 rule, and the published placement
(post_residual) at the adaptive rule. Every quantile's AUROC and TPR comes from
the chosen rate's detection_psu_ratio block, never from the adaptive or
matched_shift summary blocks in psbd_metrics.json, which hold the absolute-PSU
form at 1 quantile only (see cli.compare_detectors.psbd_values).

    PYTHONPATH=. python scripts/paper/tab_headline.py \
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defences.decision import (  # noqa: E402
    EASY_ATTACKS,
    HARD_ATTACKS,
    PRIMARY_DATASETS,
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
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_headline.py"
QUANTILE_KEYS = ("q0.01", "q0.05", "q0.10", "q0.25")
CONFIGS = ("rec_adapt", "rec_match", "pub_adapt")
CONFIG_HEADERS = {
    "rec_adapt": "rec-adapt",
    "rec_match": "rec-match06",
    "pub_adapt": "pub-adapt",
}
RATE_ORDER = (0.01, 0.05, 0.1)
DATASET_ORDER = ("cifar10", "cifar100", "gtsrb", "tiny", "svhn", "eurosat")


def measure_cell(results_dir: str, folder: str) -> dict[str, dict | None]:
    """The 3 headline configurations on 1 cell, each a detection_psu_ratio block or None."""
    report = load_psbd_metrics(results_dir, folder)
    values = {
        "rec_adapt": psbd_values(report, RECOMMENDED_PLACEMENT, "adaptive"),
        "rec_match": psbd_values(report, RECOMMENDED_PLACEMENT, "matched"),
        "pub_adapt": psbd_values(report, PUBLISHED_PLACEMENT, "adaptive"),
    }
    return values


def metric(block: dict | None, quantile_key: str, key: str) -> float | None:
    """1 number out of a detection_psu_ratio block, or None when the config has no value."""
    if block is None or quantile_key not in block:
        return None
    value = block[quantile_key].get(key)
    return value


def common_coverage(cells: list[dict]) -> list[dict]:
    """Cells where all 3 configurations returned a value, the only set an aggregate may span."""
    covered = [
        cell
        for cell in cells
        if all(cell["values"][cfg] is not None for cfg in CONFIGS)
    ]
    return covered


def config_values(
    cells: list[dict], cfg: str, quantile_key: str, key: str
) -> list[float]:
    values = [metric(cell["values"][cfg], quantile_key, key) for cell in cells]
    present = [value for value in values if value is not None]
    return present


def paired_deltas(cells: list[dict], cfg_a: str, cfg_b: str) -> list[float]:
    """cfg_a minus cfg_b headline AUROC, paired within cell, over cells where both are present."""
    deltas = []
    for cell in cells:
        a = metric(cell["values"][cfg_a], HEADLINE_KEY, "auroc")
        b = metric(cell["values"][cfg_b], HEADLINE_KEY, "auroc")
        if a is not None and b is not None:
            deltas.append(a - b)
    return deltas


def subsets(cells: list[dict]) -> list[tuple[str, list[dict]]]:
    """The rows every aggregate table in this generator shares."""
    groups = [
        ("all", cells),
        ("hard attacks", [c for c in cells if c["attack"] in HARD_ATTACKS]),
        ("easy attacks", [c for c in cells if c["attack"] in EASY_ATTACKS]),
        ("primary datasets", [c for c in cells if c["dataset"] in PRIMARY_DATASETS]),
    ]
    for dataset in DATASET_ORDER:
        groups.append((dataset, [c for c in cells if c["dataset"] == dataset]))
    for rate in RATE_ORDER:
        groups.append(
            (f"rate {rate:g}", [c for c in cells if c["poison_rate"] == rate])
        )
    return groups


def big_table_header() -> list[str]:
    header = ["subset", "n"]
    for cfg in CONFIGS:
        for quantile_key in QUANTILE_KEYS:
            header.append(f"{CONFIG_HEADERS[cfg]} AUROC {quantile_key}")
            header.append(f"{CONFIG_HEADERS[cfg]} TPR {quantile_key}")
    return header


def big_table_row(name: str, cells: list[dict]) -> list[str]:
    covered = common_coverage(cells)
    row = [name, str(len(covered))]
    for cfg in CONFIGS:
        for quantile_key in QUANTILE_KEYS:
            auroc = mean_or_none(config_values(covered, cfg, quantile_key, "auroc"))
            tpr = mean_or_none(config_values(covered, cfg, quantile_key, "tpr"))
            row.append(fmt(auroc))
            row.append(fmt(tpr))
    return row


def delta_table_row(
    name: str, cells: list[dict], resamples: int, seed: int
) -> list[str]:
    delta_adapt = paired_deltas(cells, "rec_adapt", "pub_adapt")
    delta_match = paired_deltas(cells, "rec_match", "pub_adapt")
    low_a, high_a = bootstrap_ci(delta_adapt, resamples, seed)
    low_m, high_m = bootstrap_ci(delta_match, resamples, seed)
    row = [
        name,
        str(len(delta_adapt)),
        fmt(mean_or_none(delta_adapt), signed=True),
        ci_text(low_a, high_a),
        str(len(delta_match)),
        fmt(mean_or_none(delta_match), signed=True),
        ci_text(low_m, high_m),
    ]
    return row


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    coverage = load_coverage(args.results_dir)
    cells = clearing_cells(coverage)

    for cell in cells:
        cell["values"] = measure_cell(args.results_dir, cell["folder_name"])

    inputs = [
        coverage_path,
        f"{args.results_dir}/<folder>/psbd_metrics.json (65 cells)",
    ]

    big_rows = [big_table_row(name, subset) for name, subset in subsets(cells)]
    write_table(
        path=os.path.join(args.paper_dir, "tables", "headline.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "AUROC and TPR at 4 false-positive budgets for the recommended "
            "placement `before\\_attention\\_norm\\_token\\_mask` at the adaptive "
            "0.8 rule (rec-adapt) and the matched 0.6 rule (rec-match06), and the "
            "published placement `post\\_residual` at the adaptive rule (pub-adapt), "
            "over the 65 clearing cells. n is the common-coverage count of the row, "
            "the cells where all 3 configurations returned a value."
        ),
        label="tab:headline",
        header=big_table_header(),
        rows=big_rows,
        align="l" + "r" * (len(big_table_header()) - 1),
    )

    delta_rows = [
        delta_table_row(name, subset, args.bootstrap, args.seed)
        for name, subset in subsets(cells)
    ]
    write_table(
        path=os.path.join(args.paper_dir, "tables", "headline_deltas.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Paired AUROC deltas at the headline quantile q0.25, recommended minus "
            "published, both at the adaptive rule and recommended at the matched "
            "0.6 rule against published at the adaptive rule. "
            f"{args.bootstrap}-resample bootstrap 95\\% intervals, seed {args.seed}."
        ),
        label="tab:headline-deltas",
        header=[
            "subset",
            "n adapt",
            "rec-adapt minus pub-adapt",
            "95% CI",
            "n match",
            "rec-match06 minus pub-adapt",
            "95% CI",
        ],
        rows=delta_rows,
        align="lrrlrrl",
    )

    all_cells_delta_adapt = paired_deltas(cells, "rec_adapt", "pub_adapt")
    all_cells_delta_match = paired_deltas(cells, "rec_match", "pub_adapt")
    low_adapt, high_adapt = bootstrap_ci(
        all_cells_delta_adapt, args.bootstrap, args.seed
    )
    low_match, high_match = bootstrap_ci(
        all_cells_delta_match, args.bootstrap, args.seed
    )

    rec_adapt_auroc = config_values(cells, "rec_adapt", HEADLINE_KEY, "auroc")
    rec_match_auroc = config_values(cells, "rec_match", HEADLINE_KEY, "auroc")
    pub_adapt_auroc = config_values(cells, "pub_adapt", HEADLINE_KEY, "auroc")
    rec_adapt_tpr_at_1pct = config_values(cells, "rec_adapt", "q0.01", "tpr")

    primary_cells = [c for c in cells if c["dataset"] in PRIMARY_DATASETS]
    hard_cells = [c for c in cells if c["attack"] in HARD_ATTACKS]
    cifar100_1pct_cells = [
        c for c in cells if c["dataset"] == "cifar100" and c["poison_rate"] == 0.01
    ]

    macros = {
        "headline_auroc_adaptive": (
            fmt(mean_or_none(rec_adapt_auroc)),
            "mean AUROC of the recommended placement at the adaptive rule, "
            f"over the {len(rec_adapt_auroc)} cells it covers of the 65-cell panel",
        ),
        "headline_auroc_matched": (
            fmt(mean_or_none(rec_match_auroc)),
            "mean AUROC of the recommended placement at the matched 0.6 rule, "
            f"over the {len(rec_match_auroc)} cells it covers of the 65-cell panel",
        ),
        "published_auroc_adaptive": (
            fmt(mean_or_none(pub_adapt_auroc)),
            "mean AUROC of the published placement at the adaptive rule, "
            f"over the {len(pub_adapt_auroc)} cells it covers of the 65-cell panel",
        ),
        "headline_gain_adaptive_auroc": (
            fmt(mean_or_none(all_cells_delta_adapt), signed=True),
            "mean AUROC gain of the recommended placement over the published "
            "placement, both at the adaptive rule, paired over the 65-cell panel",
        ),
        "headline_gain_adaptive_auroc_low": (
            fmt(low_adapt, signed=True),
            "lower bound of the 95% bootstrap interval on headline_gain_adaptive_auroc",
        ),
        "headline_gain_adaptive_auroc_high": (
            fmt(high_adapt, signed=True),
            "upper bound of the 95% bootstrap interval on headline_gain_adaptive_auroc",
        ),
        "headline_gain_matched_auroc": (
            fmt(mean_or_none(all_cells_delta_match), signed=True),
            "mean AUROC gain of the recommended placement at the matched 0.6 rule "
            "over the published placement at the adaptive rule, paired over the "
            "65-cell panel",
        ),
        "headline_tpr_at_one_percent": (
            fmt(mean_or_none(rec_adapt_tpr_at_1pct)),
            "mean TPR of the recommended placement at the adaptive rule, at the "
            f"1% FPR budget, over the {len(rec_adapt_tpr_at_1pct)} cells it covers",
        ),
        "headline_floor_auroc": (
            fmt(min(rec_adapt_auroc)) if rec_adapt_auroc else "--",
            "minimum single-cell AUROC of the recommended placement at the "
            "adaptive rule, over the 65-cell panel",
        ),
        "headline_inversions": (
            str(sum(1 for value in rec_adapt_auroc if value < 0.5)),
            "cells where the recommended placement at the adaptive rule scores "
            "AUROC below 0.5, out of the 65-cell panel",
        ),
        "primary_gain_adaptive_auroc": (
            fmt(
                mean_or_none(paired_deltas(primary_cells, "rec_adapt", "pub_adapt")),
                signed=True,
            ),
            "mean paired AUROC gain of recommended over published, both at the "
            "adaptive rule, on the primary datasets cifar100 and tiny",
        ),
        "hard_gain_adaptive_auroc": (
            fmt(
                mean_or_none(paired_deltas(hard_cells, "rec_adapt", "pub_adapt")),
                signed=True,
            ),
            "mean paired AUROC gain of recommended over published, both at the "
            "adaptive rule, on the hard attacks",
        ),
        "cifar100_one_percent_gain": (
            fmt(
                mean_or_none(
                    paired_deltas(cifar100_1pct_cells, "rec_adapt", "pub_adapt")
                ),
                signed=True,
            ),
            "mean paired AUROC gain of recommended over published, both at the "
            "adaptive rule, on cifar100 at 1% poisoning only, n="
            f"{len(cifar100_1pct_cells)}",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "headline.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"headline: {len(common_coverage(cells))}/{len(cells)} common-coverage cells, "
        f"gain adaptive {fmt(mean_or_none(all_cells_delta_adapt), signed=True)}"
    )


if __name__ == "__main__":
    main()
