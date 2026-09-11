"""Mechanism test A1/A1b: do shifted clean predictions land on the target class?

The neuron-bias account of PSBD says a backdoored model's decision boundary is
warped toward the attacker's target class everywhere, so a clean image whose
prediction moves under dropout should preferentially move onto that class. This
reads clean_shift_to_target_fraction from every clearing cell's psbd_metrics.json
at the recommended placement, compares it to the uniform expectation 1/num_classes
and to the same measurement on the matched-dataset benign reference, and reports
the excess as a paired statistic (backdoored minus benign, matched by dataset).

A1b repeats the same read on the GTSRB clean-label folders that target class 1
(the `_tl1` folder tag), when any such folder with a psbd_metrics.json exists on
disk, since the headline panel's target class is 0 everywhere and a target-share
excess that only shows up at class 0 would be a confound rather than a mechanism.

    PYTHONPATH=. python scripts/paper/mech_shift_target.py \
        --results-dir /lustre/home/pstika/projects/PSBD-ViT/results --paper-dir paper
"""

import sys
import os

sys.path.insert(0, os.getcwd())

import glob
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cli.compare_detectors import psbd_rate
from defences.decision import EASY_ATTACKS, HARD_ATTACKS, RECOMMENDED_PLACEMENT
from scripts.paper._common import (
    bootstrap_ci,
    build_parser,
    ci_text,
    clearing_cells,
    figure_sidecar,
    fmt,
    load_coverage,
    load_declaration,
    load_psbd_metrics,
    rate_row,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/mech_shift_target.py"
HEADLINE_CELL_FOLDER = "vit_cifar100_badnet_a2o_0_01"
# Fixed across the whole panel (results/coverage/coverage.json), including the
# benign references, which are probed with badnet_a2o at this target. A benign
# checkpoint's own top-level target_label is null (it has no attack), which
# leaves clean_shift_to_target_fraction null in its psbd_metrics.json too, so the
# share is read straight off the histogram at this fixed class instead of
# trusting that precomputed field.
PANEL_TARGET_LABEL = 0
RULES = ("matched", "adaptive")
DATASET_ORDER = ("cifar10", "cifar100", "gtsrb", "tiny", "svhn", "eurosat")
RULE_LABEL = {"matched": "matched 0.6", "adaptive": "adaptive 0.8"}
# The macros headline on the matched-0.6 rule, the project's standard placement
# comparison operator (defences.decision.PLACEMENT_MATCH_TARGET), so a target-share
# excess is read at the same disturbance level every other placement number uses.
MACRO_RULE = "matched"

# Okabe-Ito, colourblind safe, used in this fixed order across every mech_*.py figure.
PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
]


def order_attacks(names: set[str]) -> list[str]:
    """Attack names, hard attacks first, matching defences.decision's grouping."""
    ranked = list(HARD_ATTACKS) + list(EASY_ATTACKS)
    ordered = [name for name in ranked if name in names]
    ordered += sorted(name for name in names if name not in ranked)
    return ordered


def target_share_from_histogram(row: dict) -> float | None:
    """hist[target] / sum(hist) for the clean split, read straight off the histogram.

    Ignores the row's own clean_shift_to_target_fraction, which was computed from
    the checkpoint's own target_label field and is null on every benign checkpoint
    (see PANEL_TARGET_LABEL). None when the rate shifted 0 clean predictions, so
    the fraction has no denominator.
    """
    histogram = row["shift_target_histogram"]["clean"]
    total = sum(histogram)
    if total == 0:
        return None
    share = histogram[PANEL_TARGET_LABEL] / total
    return share


def target_share_at_rule(report: dict, placement: str, rule: str) -> dict | None:
    """Target-class share of shifted clean predictions at a rate rule, or None.

    None when the placement or the rate the rule selects is absent from this
    checkpoint's psbd_metrics.json, which happens for a placement whose rate
    ladder never reaches the rule's shift-ratio target.
    """
    block = report.get("placements", {}).get(placement)
    if block is None:
        return None
    rate = psbd_rate(block, rule)
    if rate is None:
        return None
    row = rate_row(block, rate)
    if row is None:
        return None

    share = target_share_from_histogram(row)
    if share is None:
        return None
    num_classes = len(row["shift_target_histogram"]["clean"])
    uniform = 1.0 / num_classes
    reading = {
        "rate": rate,
        "achieved_shift_ratio": row["shift_ratio"]["clean"],
        "share": share,
        "uniform": uniform,
        "excess": share - uniform,
    }
    return reading


def rate_curve(report: dict, placement: str) -> list[dict]:
    """Every swept rate's shift ratio and target share, in ascending rate order."""
    block = report.get("placements", {}).get(placement)
    if block is None:
        return []
    rows = sorted(block.get("rates", []), key=lambda row: row["rate"])
    curve = []
    for row in rows:
        share = target_share_from_histogram(row)
        if share is None:
            continue
        curve.append(
            {
                "rate": row["rate"],
                "shift_ratio": row["shift_ratio"]["clean"],
                "share": share,
                "uniform": 1.0 / len(row["shift_target_histogram"]["clean"]),
            }
        )
    return curve


def collect_cell_records(
    results_dir: str, cells: list[dict], benign_reports: dict
) -> list[dict]:
    """Per clearing cell: attack, dataset, both rules' readings, paired benign readings."""
    records = []
    for cell in cells:
        report = load_psbd_metrics(results_dir, cell["folder_name"])
        benign_report = benign_reports.get(cell["dataset"])
        if report is None or benign_report is None:
            continue

        by_rule = {}
        complete = True
        for rule in RULES:
            attack_reading = target_share_at_rule(report, RECOMMENDED_PLACEMENT, rule)
            benign_reading = target_share_at_rule(
                benign_report, RECOMMENDED_PLACEMENT, rule
            )
            if attack_reading is None or benign_reading is None:
                complete = False
                break
            by_rule[rule] = {"attack": attack_reading, "benign": benign_reading}
        if not complete:
            continue

        records.append(
            {
                "folder": cell["folder_name"],
                "attack": cell["attack"],
                "dataset": cell["dataset"],
                "by_rule": by_rule,
                "curve": rate_curve(report, RECOMMENDED_PLACEMENT),
            }
        )
    return records


def average_curve_by_attack(records: list[dict]) -> dict[str, list[dict]]:
    """Mean shift ratio and target share at each nominal rate, 1 curve per attack."""
    grouped: dict[str, list[list[dict]]] = {}
    for record in records:
        if not record["curve"]:
            continue
        grouped.setdefault(record["attack"], []).append(record["curve"])

    averaged = {}
    for attack, curves in grouped.items():
        by_rate: dict[float, list[dict]] = {}
        for curve in curves:
            for point in curve:
                by_rate.setdefault(point["rate"], []).append(point)
        rates = sorted(by_rate)
        averaged[attack] = [
            {
                "rate": rate,
                "shift_ratio": statistics.mean(p["shift_ratio"] for p in by_rate[rate]),
                "share": statistics.mean(p["share"] for p in by_rate[rate]),
            }
            for rate in rates
        ]
    return averaged


def benign_band(benign_reports: dict) -> list[dict]:
    """Min-max target share across the 4 benign references, at each shared rate."""
    curves = {
        dataset: rate_curve(report, RECOMMENDED_PLACEMENT)
        for dataset, report in benign_reports.items()
    }
    rates = sorted({point["rate"] for curve in curves.values() for point in curve})

    band = []
    for rate in rates:
        points = [
            point
            for curve in curves.values()
            for point in curve
            if point["rate"] == rate
        ]
        if not points:
            continue
        band.append(
            {
                "rate": rate,
                "shift_ratio": statistics.mean(p["shift_ratio"] for p in points),
                "share_low": min(p["share"] for p in points),
                "share_high": max(p["share"] for p in points),
            }
        )
    return band


def paired_excess(
    group: list[dict], rule: str, resamples: int, seed: int
) -> tuple[list[float], list[float], list[float], str]:
    """Shares, benign shares, paired deltas and the excess text with its interval."""
    shares = [r["by_rule"][rule]["attack"]["share"] for r in group]
    benign_shares = [r["by_rule"][rule]["benign"]["share"] for r in group]
    deltas = [share - benign for share, benign in zip(shares, benign_shares)]
    low, high = bootstrap_ci(deltas, resamples, seed)
    text = f"{fmt(statistics.mean(deltas), signed=True)} {ci_text(low, high)}"
    return shares, benign_shares, deltas, text


def write_shift_target_table(args, records: list[dict]) -> None:
    """Per attack, mean share, uniform expectation, benign share, paired excess, for 2 rules."""
    header = ["attack", "n"]
    for rule in RULES:
        label = RULE_LABEL[rule]
        header += [f"share@{label}", f"benign@{label}", f"excess@{label} [95% CI]"]

    attacks = order_attacks({record["attack"] for record in records})
    rows = []
    for attack in attacks:
        attack_records = [r for r in records if r["attack"] == attack]
        row = [attack, str(len(attack_records))]
        for rule in RULES:
            shares, benign_shares, _, excess = paired_excess(
                attack_records, rule, args.bootstrap, args.seed
            )
            row += [
                fmt(statistics.mean(shares)),
                fmt(statistics.mean(benign_shares)),
                excess,
            ]
        rows.append(row)

    all_row = ["all cells", str(len(records))]
    for rule in RULES:
        shares, benign_shares, _, excess = paired_excess(
            records, rule, args.bootstrap, args.seed
        )
        all_row += [
            fmt(statistics.mean(shares)),
            fmt(statistics.mean(benign_shares)),
            excess,
        ]
    rows.append(all_row)

    path = os.path.join(args.paper_dir, "tables", "mech_shift_target.tex")
    write_table(
        path=path,
        generator=GENERATOR,
        inputs=[
            f"{args.results_dir}/<folder>/psbd_metrics.json (65 clearing cells, 4 benign)"
        ],
        caption=(
            "A1: target-class share of shifted clean predictions at the recommended "
            f"placement ({RECOMMENDED_PLACEMENT}), against the uniform expectation "
            "1/num\\_classes and the matched-dataset benign reference, read at the "
            "matched-0.6 and adaptive-0.8 rate rules. Excess is the attack's share "
            "minus the benign share on the same dataset, paired per cell, 95\\% "
            "bootstrap interval over cells."
        ),
        label="tab:mech-shift-target",
        header=header,
        rows=rows,
    )


def write_dataset_table(args, records: list[dict]) -> None:
    """Per dataset and attack at the macro rule, so the local-trigger few-class pattern is visible.

    The per-attack table averages over datasets, which hides that the same
    attack drifts toward the target on GTSRB and not on CIFAR-100. Cell counts
    per row are 1 to 3, so no interval is printed here, the per-attack table
    carries the intervals.
    """
    header = [
        "dataset",
        "attack",
        "n",
        f"share@{RULE_LABEL[MACRO_RULE]}",
        "uniform",
        "benign share",
        "excess",
    ]
    rows = []
    for dataset in DATASET_ORDER:
        dataset_records = [r for r in records if r["dataset"] == dataset]
        for attack in order_attacks({r["attack"] for r in dataset_records}):
            group = [r for r in dataset_records if r["attack"] == attack]
            shares, benign_shares, deltas, _ = paired_excess(
                group, MACRO_RULE, 0, args.seed
            )
            uniform = statistics.mean(
                r["by_rule"][MACRO_RULE]["attack"]["uniform"] for r in group
            )
            rows.append(
                [
                    dataset,
                    attack,
                    str(len(group)),
                    fmt(statistics.mean(shares)),
                    fmt(uniform),
                    fmt(statistics.mean(benign_shares)),
                    fmt(statistics.mean(deltas), signed=True),
                ]
            )

    path = os.path.join(args.paper_dir, "tables", "mech_shift_target_by_dataset.tex")
    write_table(
        path=path,
        generator=GENERATOR,
        inputs=[
            f"{args.results_dir}/<folder>/psbd_metrics.json (65 clearing cells, 4 benign)"
        ],
        caption=(
            "A1 by dataset: target-class share of shifted clean predictions at the "
            f"recommended placement and the {RULE_LABEL[MACRO_RULE]} rule, per dataset "
            "and attack, against the uniform expectation and the dataset's benign "
            "reference. n is the number of poison rates the cell clears, so rows "
            "carry no interval."
        ),
        label="tab:mech-shift-target-by-dataset",
        header=header,
        rows=rows,
    )


def find_tl1_folders(results_dir: str) -> list[str]:
    """GTSRB clean-label folders targeting class 1, restricted to ones with psbd_metrics.json."""
    patterns = ("vit_gtsrb_sig_*_tl1", "vit_gtsrb_lc_*_tl1")
    found = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(results_dir, pattern))):
            if os.path.exists(os.path.join(path, "psbd_metrics.json")):
                found.append(os.path.basename(path))
    return found


def write_tl1_table(args) -> None:
    """A1b: the same read on class-1 clean-label GTSRB folders, when any exist."""
    folders = find_tl1_folders(args.results_dir)
    header = ["folder", "n", "share@matched0.6", "uniform", "excess@matched0.6"]
    rows = []
    for folder in folders:
        report = load_psbd_metrics(args.results_dir, folder)
        if report is None:
            continue
        reading = target_share_at_rule(report, RECOMMENDED_PLACEMENT, MACRO_RULE)
        if reading is None:
            continue
        rows.append(
            [
                folder,
                "1",
                fmt(reading["share"]),
                fmt(reading["uniform"]),
                fmt(reading["excess"], signed=True),
            ]
        )

    if rows:
        caption = (
            "A1b: the same target-share read on GTSRB clean-label folders whose "
            "target is class 1 (the \\_tl1 folder tag), so the class-0 excess in "
            "the main table is not read as a class-0-specific artefact."
        )
    else:
        caption = (
            "A1b: no vit\\_gtsrb\\_sig\\_*\\_tl1 or vit\\_gtsrb\\_lc\\_*\\_tl1 folder "
            "with a psbd\\_metrics.json exists on disk yet, so this table has no "
            "rows. The main table's target class is 0 on every panel cell."
        )
        rows = [["--", "0", "--", "--", "--"]]

    path = os.path.join(args.paper_dir, "tables", "mech_shift_target_tl1.tex")
    write_table(
        path=path,
        generator=GENERATOR,
        inputs=[f"{args.results_dir}/vit_gtsrb_{{sig,lc}}_*_tl1/psbd_metrics.json"],
        caption=caption,
        label="tab:mech-shift-target-tl1",
        header=header,
        rows=rows,
    )


def write_shift_target_figure(args, records: list[dict], benign_reports: dict) -> None:
    """Target share against achieved shift ratio, 1 line per attack, benign band, uniform line."""
    attack_curves = average_curve_by_attack(records)
    band = benign_band(benign_reports)
    uniform_mean = statistics.mean(
        record["by_rule"][MACRO_RULE]["attack"]["uniform"] for record in records
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for index, attack in enumerate(order_attacks(set(attack_curves))):
        curve = attack_curves[attack]
        ax.plot(
            [point["shift_ratio"] for point in curve],
            [point["share"] for point in curve],
            marker="o",
            markersize=3,
            linewidth=1.3,
            color=PALETTE[index % len(PALETTE)],
            label=attack,
        )

    if band:
        ax.fill_between(
            [point["shift_ratio"] for point in band],
            [point["share_low"] for point in band],
            [point["share_high"] for point in band],
            color="0.75",
            alpha=0.6,
            label="benign range (4 datasets)",
        )

    ax.axhline(
        uniform_mean,
        linestyle="--",
        color="black",
        linewidth=1.0,
        label="uniform expectation",
    )
    ax.set_xlabel("achieved shift ratio (clean split)")
    ax.set_ylabel("target-class share of shifted clean predictions")
    ax.legend(fontsize=6.5, loc="upper left", ncol=1)
    fig.tight_layout()

    path = os.path.join(args.paper_dir, "figures", "mech_shift_target.pdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

    figure_sidecar(
        path=path.replace(".pdf", ".json"),
        generator=GENERATOR,
        inputs=[f"{args.results_dir}/<folder>/psbd_metrics.json"],
        plotted={
            "attack_curves": attack_curves,
            "benign_band": band,
            "uniform_expectation_mean": uniform_mean,
        },
    )


def write_shift_target_macros(args, records: list[dict], benign_reports: dict) -> None:
    shares = [r["by_rule"][MACRO_RULE]["attack"]["share"] for r in records]
    benign_shares = [r["by_rule"][MACRO_RULE]["benign"]["share"] for r in records]
    _, _, _, excess = paired_excess(records, MACRO_RULE, args.bootstrap, args.seed)

    headline_record = next(
        (r for r in records if r["folder"] == HEADLINE_CELL_FOLDER), None
    )
    # 3 decimals rounds this specific cell's share to 0.000, since it sits under
    # 0.0005, and the whole point of the number is how far under the uniform
    # expectation it sits, so it gets 1 extra decimal place.
    headline_value = (
        fmt(headline_record["by_rule"]["adaptive"]["attack"]["share"], places=4)
        if headline_record is not None
        else "--"
    )

    macros = {
        "shift_to_target_share_all": (
            fmt(statistics.mean(shares)),
            "A1: mean target-class share of shifted clean predictions, all clearing "
            "cells, recommended placement, matched-0.6 rate rule",
        ),
        "shift_to_target_excess_all": (
            excess,
            "A1: mean paired excess over the matched-dataset benign reference, all "
            "clearing cells, matched-0.6 rate rule, 95% bootstrap interval",
        ),
        "shift_to_target_benign_share": (
            fmt(statistics.mean(benign_shares)),
            "A1: mean target-class share on the 4 benign references, matched-0.6 "
            "rate rule",
        ),
        "shift_to_target_headline_cell": (
            headline_value,
            f"A1: target-class share on {HEADLINE_CELL_FOLDER}, recommended "
            "placement, adaptive-0.8 rate rule",
        ),
    }
    write_macros(
        sidecar_path=os.path.join(
            args.paper_dir, "tables", "mech_shift_target.macros.json"
        ),
        generator=GENERATOR,
        inputs=[f"{args.results_dir}/<folder>/psbd_metrics.json"],
        macros=macros,
    )


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage = load_coverage(args.results_dir)
    declaration = load_declaration(args.declaration)
    cells = clearing_cells(coverage)

    benign_folders = {
        dataset: folder
        for dataset, folder in declaration["benign_reference"].items()
        if not dataset.startswith("_")
    }
    benign_reports = {
        dataset: load_psbd_metrics(args.results_dir, folder)
        for dataset, folder in benign_folders.items()
    }
    benign_reports = {k: v for k, v in benign_reports.items() if v is not None}

    records = collect_cell_records(args.results_dir, cells, benign_reports)
    print(f"{len(records)} of {len(cells)} clearing cells carry a complete A1 reading")

    write_shift_target_table(args, records)
    write_dataset_table(args, records)
    write_tl1_table(args)
    write_shift_target_figure(args, records, benign_reports)
    write_shift_target_macros(args, records, benign_reports)


if __name__ == "__main__":
    main()
