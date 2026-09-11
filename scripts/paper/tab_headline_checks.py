"""2 checks the headline needs: a leave-one-attack-out refit, and pre against post residual.

The criticality scale asks a CRITICAL detection claim to stay positive when any
single attack is dropped, which the headline generator does not compute. The
first table refits the recommended-minus-published gain with 1 attack removed
at a time, at both rate rules. The second gives the project's founding question,
dropout before against after the residual add, its number on the basis panel,
paired within cell at both rules, since the refutation recorded in the ledger
was read on CIFAR-10 alone.

    PYTHONPATH=. python scripts/paper/tab_headline_checks.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defences.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
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

GENERATOR = "scripts/paper/tab_headline_checks.py"
PRE_RESIDUAL = "pre_residual"
RULES = ("adaptive", "matched")


def auroc(report: dict | None, placement: str, rule: str) -> float | None:
    block = psbd_values(report, placement, rule)
    if block is None or HEADLINE_KEY not in block:
        return None
    value = block[HEADLINE_KEY]["auroc"]
    return value


def paired(cells: list[dict], placement_a: str, placement_b: str, rule: str) -> list[float]:
    deltas = []
    for cell in cells:
        a = cell["auroc"][(placement_a, rule)]
        b = cell["auroc"][(placement_b, rule)]
        if a is not None and b is not None:
            deltas.append(a - b)
    return deltas


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    coverage = load_coverage(args.results_dir)
    cells = clearing_cells(coverage)
    diverged = sum(1 for cell in coverage["cells"] if cell.get("asr_class") == "diverged")
    for cell in cells:
        report = load_psbd_metrics(args.results_dir, cell["folder_name"])
        cell["auroc"] = {
            (placement, rule): auroc(report, placement, rule)
            for placement in (RECOMMENDED_PLACEMENT, PUBLISHED_PLACEMENT, PRE_RESIDUAL)
            for rule in RULES
        }
    inputs = [coverage_path, f"{args.results_dir}/<folder>/psbd_metrics.json ({len(cells)} cells)"]
    # A clearing cell enters a detection table only once its sweep has landed.
    cached = [cell for cell in cells if cell["auroc"][(RECOMMENDED_PLACEMENT, "adaptive")] is not None]
    cells = cached
    attacks = sorted({cell["attack"] for cell in cells})

    rows = []
    refit_means = {rule: [] for rule in RULES}
    refit_lows = {rule: [] for rule in RULES}
    for attack in attacks:
        rest = [cell for cell in cells if cell["attack"] != attack]
        row = [f"without {attack}"]
        for rule in RULES:
            deltas = paired(rest, RECOMMENDED_PLACEMENT, PUBLISHED_PLACEMENT, rule)
            low, high = bootstrap_ci(deltas, args.bootstrap, args.seed)
            refit_means[rule].append(mean_or_none(deltas))
            refit_lows[rule].append(low)
            row += [str(len(deltas)), fmt(mean_or_none(deltas), signed=True), ci_text(low, high)]
        rows.append(row)
    write_table(
        path=os.path.join(args.paper_dir, "tables", "headline_leave_one_out.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "The headline gain, recommended minus published at the headline quantile, "
            "refit with 1 attack dropped at a time, both placements at the adaptive "
            "rule and both at the matched rule, "
            f"{args.bootstrap}-resample bootstrap intervals."
        ),
        label="tab:headline-loo",
        header=["dropped attack", "n", "gain adaptive", "95% CI", "n", "gain matched", "95% CI"],
        rows=rows,
        align="lrrlrrl",
    )

    pre_post_rows = []
    pre_post = {}
    for rule in RULES:
        deltas = paired(cells, PRE_RESIDUAL, PUBLISHED_PLACEMENT, rule)
        low, high = bootstrap_ci(deltas, args.bootstrap, args.seed)
        pre_post[rule] = (mean_or_none(deltas), low, high, len(deltas))
        pre_post_rows.append(["pre\\_residual minus post\\_residual", rule, str(len(deltas)), fmt(mean_or_none(deltas), signed=True), ci_text(low, high)])
    write_table(
        path=os.path.join(args.paper_dir, "tables", "pre_post.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "The founding question on the basis panel: dropout before each residual "
            "add against dropout after it, paired within cell at the headline "
            "quantile, at the matched rule and at the adaptive rule."
        ),
        label="tab:pre-post",
        header=["comparison", "rule", "n", "mean delta AUROC", "95% CI"],
        rows=pre_post_rows,
        align="llrrl",
    )

    macros = {
        "panel_cells_cached": (
            str(len(cached)),
            "clearing cells whose sweep has landed and that carry the recommended placement",
        ),
        "panel_cells_diverged": (
            str(diverged),
            "declared cells whose training diverged, excluded from every table",
        ),
        "panel_cells_awaiting_sweep": (
            str(len(clearing_cells(coverage)) - len(cached)),
            "clearing cells whose PSBD sweep has not landed",
        ),
        "panel_datasets_cached": (
            str(len({cell["dataset"] for cell in cached})),
            "datasets among the clearing cells whose sweep has landed",
        ),
        "panel_attacks_declared": (
            str(len({cell["attack"] for cell in coverage["cells"]})),
            "attacks the panel declares, clearing or not",
        ),
        "headline_gain_min_leave_one_attack_out_adaptive": (
            fmt(min(refit_means["adaptive"]), signed=True),
            "smallest leave-one-attack-out refit of the adaptive-rule headline gain",
        ),
        "headline_gain_min_leave_one_attack_out_adaptive_low": (
            fmt(min(refit_lows["adaptive"]), signed=True),
            "lowest bootstrap lower bound over the adaptive-rule leave-one-attack-out refits",
        ),
        "headline_gain_min_leave_one_attack_out_matched": (
            fmt(min(refit_means["matched"]), signed=True),
            "smallest leave-one-attack-out refit of the matched-rule headline gain",
        ),
        "headline_gain_min_leave_one_attack_out_matched_low": (
            fmt(min(refit_lows["matched"]), signed=True),
            "lowest bootstrap lower bound over the matched-rule leave-one-attack-out refits",
        ),
        "pre_minus_post_matched": (fmt(pre_post["matched"][0], signed=True), "pre_residual minus post_residual dropout, matched rule, paired over the basis panel"),
        "pre_minus_post_matched_low": (fmt(pre_post["matched"][1], signed=True), "lower bootstrap bound of pre_minus_post_matched"),
        "pre_minus_post_matched_high": (fmt(pre_post["matched"][2], signed=True), "upper bootstrap bound of pre_minus_post_matched"),
        "pre_minus_post_adaptive": (fmt(pre_post["adaptive"][0], signed=True), "pre_residual minus post_residual dropout, adaptive rule, paired over the basis panel"),
        "pre_minus_post_adaptive_low": (fmt(pre_post["adaptive"][1], signed=True), "lower bootstrap bound of pre_minus_post_adaptive"),
        "pre_minus_post_adaptive_high": (fmt(pre_post["adaptive"][2], signed=True), "upper bootstrap bound of pre_minus_post_adaptive"),
        "pre_minus_post_n": (str(pre_post["matched"][3]), "cells behind the pre against post comparison"),
    }
    write_macros(os.path.join(args.paper_dir, "tables", "headline_checks.macros.json"), GENERATOR, inputs, macros)
    print(f"headline checks: loo min adaptive {macros['headline_gain_min_leave_one_attack_out_adaptive'][0]}, pre-post matched {macros['pre_minus_post_matched'][0]}")


if __name__ == "__main__":
    main()
