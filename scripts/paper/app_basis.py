"""The full placement basis: what the declaration says and how every placement scores.

2 tables. The first is the declaration read back, 1 row per basis placement with
its position, operator, block band, family and rate ladder, so the appendix shows
exactly what every panel cell was swept with. The second ranks every placement on
the clearing cells at both rate rules, with the cells it reaches, its worst cell
and its inversions, so a reader can see the whole search and not only the winner.

    PYTHONPATH=. python scripts/paper/app_basis.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defences.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    build_parser,
    clearing_cells,
    family_label,
    fmt,
    load_coverage,
    load_declaration,
    load_psbd_metrics,
    mean_or_none,
    placement_label,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/app_basis.py"
RULES = ("matched", "adaptive")


def band_text(entry: dict) -> str:
    block_range = entry.get("block_range")
    if not block_range:
        return "all"
    text = f"{block_range[0]}-{block_range[1]}"
    return text


def declaration_rows(basis: list[dict]) -> list[list[str]]:
    rows = []
    for entry in basis:
        rates = entry["rates"]
        rows.append(
            [
                entry["id"],
                entry["position"],
                entry["operator"],
                band_text(entry),
                entry["family"],
                f"{len(rates)} ({min(rates):g} to {max(rates):g})",
            ]
        )
    return rows


def placement_aurocs(
    results_dir: str, cells: list[dict], placement: str
) -> dict[str, list[float]]:
    """This placement's headline AUROC on every cell that reached each rule."""
    by_rule: dict[str, list[float]] = {rule: [] for rule in RULES}
    for cell in cells:
        report = load_psbd_metrics(results_dir, cell["folder_name"])
        for rule in RULES:
            block = psbd_values(report, placement, rule)
            if block is not None and HEADLINE_KEY in block:
                by_rule[rule].append(block[HEADLINE_KEY]["auroc"])
    return by_rule


def ranking_rows(
    results_dir: str, cells: list[dict], basis: list[dict]
) -> tuple[list[list[str]], dict[str, dict]]:
    """1 row per placement, sorted by adaptive-rule mean AUROC, and the raw numbers."""
    measured = {}
    for entry in basis:
        by_rule = placement_aurocs(results_dir, cells, entry["id"])
        adaptive = by_rule["adaptive"]
        measured[entry["id"]] = {
            "family": entry["family"],
            "matched_mean": mean_or_none(by_rule["matched"]),
            "matched_n": len(by_rule["matched"]),
            "adaptive_mean": mean_or_none(adaptive),
            "adaptive_n": len(adaptive),
            "adaptive_floor": min(adaptive) if adaptive else None,
            "adaptive_inversions": sum(1 for value in adaptive if value < 0.5),
        }
    ordered = sorted(
        measured.items(),
        key=lambda item: -(item[1]["adaptive_mean"] or 0.0),
    )
    entries = {entry["id"]: entry for entry in basis}
    rows = []
    for rank, (placement, stats) in enumerate(ordered, start=1):
        rows.append(
            [
                str(rank),
                placement_label(entries[placement]),
                family_label(stats["family"]),
                str(stats["adaptive_n"]),
                fmt(stats["adaptive_mean"]),
                fmt(stats["adaptive_floor"]),
            ]
        )
    return rows, dict(ordered)


def variance_ratios(ordered: dict[str, dict], basis: list[dict]) -> dict[str, dict]:
    """Position range at fixed operator against operator range at fixed position.

    The position axis holds token_mask fixed and moves it across every
    all-blocks position the basis carries. The operator axis holds
    before_attention_norm fixed and moves the operator across every operator the
    basis carries there, reported with and without gaussian, since a LayerNorm
    follows that position and absorbs additive noise but not masking.
    """
    entries = {entry["id"]: entry for entry in basis}
    token_mask_positions = [
        name
        for name, entry in entries.items()
        if entry["operator"] == "token_mask" and not entry.get("block_range")
    ]
    attention_norm_operators = [
        name
        for name, entry in entries.items()
        if entry["position"] == "before_attention_norm" and not entry.get("block_range")
    ]
    ratios = {}
    for rule in ("matched", "adaptive"):
        key = f"{rule}_mean"
        position_values = [
            ordered[name][key]
            for name in token_mask_positions
            if ordered[name][key] is not None
        ]
        operator_values = [
            ordered[name][key]
            for name in attention_norm_operators
            if ordered[name][key] is not None
        ]
        mask_values = [
            ordered[name][key]
            for name in attention_norm_operators
            if entries[name]["operator"] != "gaussian"
            and ordered[name][key] is not None
        ]
        position_range = max(position_values) - min(position_values)
        operator_range = max(operator_values) - min(operator_values)
        operator_range_masks = max(mask_values) - min(mask_values)
        ratios[rule] = {
            "position_range": position_range,
            "operator_range": operator_range,
            "operator_range_masks": operator_range_masks,
            "ratio": position_range / operator_range if operator_range else None,
            "ratio_masks": position_range / operator_range_masks
            if operator_range_masks
            else None,
        }
    return ratios


def rank_of(ordered: dict[str, dict], placement: str, key: str) -> int:
    """The 1-based rank of a placement by 1 statistic, larger being better."""
    values = sorted(((stats[key] or 0.0), name) for name, stats in ordered.items())
    values.reverse()
    rank = next(
        index for index, (_, name) in enumerate(values, start=1) if name == placement
    )
    return rank


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    coverage = load_coverage(args.results_dir)
    declaration = load_declaration(args.declaration)
    basis = declaration["basis"]
    cells = clearing_cells(coverage)

    write_table(
        path=os.path.join(args.paper_dir, "tables", "basis_declaration.tex"),
        generator=GENERATOR,
        inputs=[args.declaration],
        caption=(
            "The placement basis as configs/psbd\\_basis.json declares it: every "
            "model in the panel carries every 1 of these placements, each swept "
            "over the rate ladder in the last column."
        ),
        label="tab:basis-declaration",
        header=["placement id", "site", "operator", "blocks", "family", "rates"],
        rows=declaration_rows(basis),
        align="llllll",
    )

    rows, ordered = ranking_rows(args.results_dir, cells, basis)
    inputs = [
        coverage_path,
        args.declaration,
        f"{args.results_dir}/<folder>/psbd_metrics.json ({len(cells)} cells)",
    ]
    write_table(
        path=os.path.join(args.paper_dir, "tables", "basis_ranking.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Every placement of the basis ranked by mean AUROC on ViT-B/16. n is the number of models whose rate ladder reaches the shift ratio target, and the last column is the lowest single-model AUROC."
        ),
        label="tab:basis-ranking",
        header=[
            "Rank",
            "Placement",
            "Family",
            "n",
            "AUROC",
            "Lowest AUROC",
        ],
        rows=rows,
        align="rllrrr",
    )

    variance = variance_ratios(ordered, basis)
    best_placement, best_stats = next(iter(ordered.items()))
    worst_placement, worst_stats = list(ordered.items())[-1]
    recommended = ordered[RECOMMENDED_PLACEMENT]
    published = ordered[PUBLISHED_PLACEMENT]
    macros = {
        "basis_rank_recommended_adaptive": (
            str(rank_of(ordered, RECOMMENDED_PLACEMENT, "adaptive_mean")),
            "rank of the recommended placement among the basis by adaptive-rule mean AUROC",
        ),
        "basis_rank_recommended_matched": (
            str(rank_of(ordered, RECOMMENDED_PLACEMENT, "matched_mean")),
            "rank of the recommended placement among the basis by matched-rule mean AUROC",
        ),
        "basis_rank_published_adaptive": (
            str(rank_of(ordered, PUBLISHED_PLACEMENT, "adaptive_mean")),
            "rank of the published placement among the basis by adaptive-rule mean AUROC",
        ),
        "basis_best_placement": (
            best_placement.replace("_", r"\_"),
            "the basis placement with the highest adaptive-rule mean AUROC",
        ),
        "basis_best_auroc_adaptive": (
            fmt(best_stats["adaptive_mean"]),
            "the highest adaptive-rule mean AUROC in the basis",
        ),
        "basis_worst_placement": (
            worst_placement.replace("_", r"\_"),
            "the basis placement with the lowest adaptive-rule mean AUROC",
        ),
        "basis_worst_auroc_adaptive": (
            fmt(worst_stats["adaptive_mean"]),
            "the lowest adaptive-rule mean AUROC in the basis",
        ),
        "published_inversions": (
            str(published["adaptive_inversions"]),
            "cells below chance for the published placement at the adaptive rule",
        ),
        "published_floor_auroc": (
            fmt(published["adaptive_floor"]),
            "worst single cell of the published placement at the adaptive rule",
        ),
        "recommended_reach_matched": (
            str(recommended["matched_n"]),
            "cells the recommended placement reaches at the matched rule",
        ),
        "basis_spread_adaptive": (
            fmt(
                (best_stats["adaptive_mean"] or 0) - (worst_stats["adaptive_mean"] or 0)
            ),
            "adaptive-rule mean AUROC spread between the best and worst basis placement",
        ),
    }
    for rule in ("matched", "adaptive"):
        ratios = variance[rule]
        macros[f"position_range_token_mask_{rule}"] = (
            fmt(ratios["position_range"]),
            f"range of mean AUROC across the all-blocks token_mask positions, {rule} rule",
        )
        macros[f"operator_range_attention_norm_{rule}"] = (
            fmt(ratios["operator_range"]),
            f"range of mean AUROC across the operators at before_attention_norm, {rule} rule",
        )
        macros[f"operator_range_attention_norm_masks_{rule}"] = (
            fmt(ratios["operator_range_masks"]),
            f"range of mean AUROC across the 2 masking operators at before_attention_norm, {rule} rule",
        )
        macros[f"position_over_operator_ratio_{rule}"] = (
            fmt(ratios["ratio"], places=2),
            f"position range over operator range at {rule} rule, gaussian included",
        )
        macros[f"position_over_operator_ratio_masks_{rule}"] = (
            fmt(ratios["ratio_masks"], places=2),
            f"position range over operator range at {rule} rule, masking operators only",
        )
    write_macros(
        os.path.join(args.paper_dir, "tables", "basis.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"basis: {len(basis)} placements, recommended rank "
        f"{macros['basis_rank_recommended_adaptive'][0]} adaptive, best {best_placement}"
    )


if __name__ == "__main__":
    main()
