"""Swin-S: what the cached Swin sweeps say about the placements the paper argues.

Swin has no coverage ledger of its own, so this reads every panel-shaped swin_*
folder with a psbd_metrics.json, keeps the cells whose sidecar attack success
clears the declared bar, and reports 4 artifacts.

    swin_main         the placements that carry the argument, with paired gains
    swin_attacks      1 row per attack and rate, grouped by dataset
    swin_placements   every cached placement, coverage shown rather than hidden
    swin_gains        the paired comparisons with bootstrap intervals

The body table includes the twin of the recommended placement, token masking on
the attention branch output before the add, because the declared selection
protocol selects on CIFAR-10 and GTSRB and the twin is the placement it would
pick if the choice were made on AUROC alone. Reporting both on the selection
half is what licenses the recommendation instead of asserting it.

Coverage is uneven across placements, so the full table's rows are not paired
and the gain rows are computed only over cells carrying both placements.

    PYTHONPATH=. python scripts/paper/tab_swin.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import collections
import glob
import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defenses.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    attack_label,
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
    placement_words,
    split_placement,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_swin.py"
RULES = ("matched", "adaptive")
# The rule the paper deploys. The matched rule is the cross-placement comparison
# device and appears beside it in the full table only.
HEADLINE_RULE = "adaptive"
# The 2 false-positive budgets the paper reports, as quantile ladder keys.
TPR_KEYS = ("q0.10", "q0.20")
# Token masking on the attention branch output before the add. Same operator as
# the recommended placement but 1 site later, and the placement a pure AUROC
# selection would choose, so it is the row that licenses the recommendation.
TWIN_PLACEMENT = "before_attention_residual_token_mask"
# Swin's own dropout probe at the attention input: the site held fixed while the
# operator changes, which separates the site from the perturbation.
SWIN_DROPOUT_INPUT = "before_attention_norm"
# The placements the body table argues with, in the order the argument needs.
BODY_PLACEMENTS = (
    RECOMMENDED_PLACEMENT,
    TWIN_PLACEMENT,
    "both_sublayer_inputs_token_mask",
    "before_attention_norm_channel_mask",
    "before_attention_norm_gaussian",
    SWIN_DROPOUT_INPUT,
    "before_mlp_norm_token_mask",
    "pre_residual",
    PUBLISHED_PLACEMENT,
)
MIN_CELLS_FOR_ROW = 5


def swin_cells(results_dir: str, checkpoints_dir: str, asr_bar: float) -> list[dict]:
    """Swin folders that are panel-shaped, carry metrics and whose attack implanted."""
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


def reading(report: dict, placement: str, rule: str, key: str, field: str):
    """1 number from a cell's metrics, or None when the cell does not carry it."""
    block = psbd_values(report, placement, rule)
    if block is None:
        return None
    quantile = block.get(key)
    if quantile is None:
        return None
    return quantile.get(field)


def collect(cells: list[dict], placement: str, rule: str, key: str, field: str):
    """Every cell's reading of 1 placement, skipping the cells without it."""
    values = [reading(cell["report"], placement, rule, key, field) for cell in cells]
    present = [value for value in values if value is not None]
    return present


def placement_stats(cells: list[dict]) -> dict[str, dict]:
    """Per cached placement, mean AUROC at both rules and the cells behind each.

    A mask-seed replicate is the same placement measured again, not a competing
    placement, so it is left out here and reported as seed spread instead.
    """
    values: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: {rule: [] for rule in RULES}
    )
    for cell in cells:
        for placement in cell["report"]["placements"]:
            if split_placement(placement)["variant"] is not None:
                continue
            for rule in RULES:
                auroc = reading(cell["report"], placement, rule, HEADLINE_KEY, "auroc")
                if auroc is not None:
                    values[placement][rule].append(auroc)
    stats = {}
    for placement, by_rule in values.items():
        stats[placement] = {
            rule: {"mean": mean_or_none(by_rule[rule]), "n": len(by_rule[rule])}
            for rule in RULES
        }
        stats[placement]["below_chance"] = sum(
            1 for value in by_rule[HEADLINE_RULE] if value < 0.5
        )
    return stats


def paired_gain(
    cells: list[dict], placement_a: str, placement_b: str, rule: str
) -> list[float]:
    """Per-cell AUROC difference over the cells carrying both placements."""
    deltas = []
    for cell in cells:
        a = reading(cell["report"], placement_a, rule, HEADLINE_KEY, "auroc")
        b = reading(cell["report"], placement_b, rule, HEADLINE_KEY, "auroc")
        if a is None or b is None:
            continue
        deltas.append(a - b)
    return deltas


def body_rows(cells: list[dict], resamples: int, seed: int) -> list[list[str]]:
    """The argument table: 1 row per body placement, gain measured against published."""
    rows = []
    for placement in BODY_PLACEMENTS:
        aurocs = collect(cells, placement, HEADLINE_RULE, HEADLINE_KEY, "auroc")
        if len(aurocs) < MIN_CELLS_FOR_ROW:
            continue
        tprs = [
            mean_or_none(collect(cells, placement, HEADLINE_RULE, key, "tpr"))
            for key in TPR_KEYS
        ]
        if placement == PUBLISHED_PLACEMENT:
            gain, interval = "reference", ""
        else:
            deltas = paired_gain(cells, placement, PUBLISHED_PLACEMENT, HEADLINE_RULE)
            low, high = bootstrap_ci(deltas, resamples, seed)
            gain = fmt(mean_or_none(deltas), signed=True)
            interval = ci_text(low, high)
        rows.append(
            [
                placement_words(placement),
                str(len(aurocs)),
                fmt(mean_or_none(aurocs)),
                fmt(tprs[0], places=2),
                fmt(tprs[1], places=2),
                gain,
                interval,
            ]
        )
    return rows


def attack_rows(cells: list[dict]) -> list[list[str]]:
    """1 row per attack and rate, grouped by dataset, with a mean row per dataset.

    Rows are attacks and columns are placements, which is the layout the PSBD
    paper uses, so a reader looking for a single attack reads across 1 line.
    """
    rows = []
    by_dataset = collections.defaultdict(list)
    for cell in cells:
        by_dataset[cell["dataset"]].append(cell)

    for dataset in sorted(by_dataset):
        group = sorted(
            by_dataset[dataset], key=lambda cell: (cell["attack"], cell["poison_rate"])
        )
        rows.append([f"\\multicolumn{{6}}{{l}}{{\\emph{{{dataset_label(dataset)}}}}}"])
        for cell in group:
            rows.append(
                [
                    attack_label(cell["attack"]),
                    f"{cell['poison_rate'] * 100:g}",
                    fmt(
                        reading(
                            cell["report"],
                            RECOMMENDED_PLACEMENT,
                            HEADLINE_RULE,
                            HEADLINE_KEY,
                            "auroc",
                        )
                    ),
                    fmt(
                        reading(
                            cell["report"],
                            RECOMMENDED_PLACEMENT,
                            HEADLINE_RULE,
                            TPR_KEYS[0],
                            "tpr",
                        ),
                        places=2,
                    ),
                    fmt(
                        reading(
                            cell["report"],
                            RECOMMENDED_PLACEMENT,
                            HEADLINE_RULE,
                            TPR_KEYS[1],
                            "tpr",
                        ),
                        places=2,
                    ),
                    fmt(
                        reading(
                            cell["report"],
                            PUBLISHED_PLACEMENT,
                            HEADLINE_RULE,
                            HEADLINE_KEY,
                            "auroc",
                        )
                    ),
                ]
            )
        rows.append(
            [
                "\\rowcolor{black!8}\\emph{mean}",
                "",
                fmt(
                    mean_or_none(
                        collect(
                            group,
                            RECOMMENDED_PLACEMENT,
                            HEADLINE_RULE,
                            HEADLINE_KEY,
                            "auroc",
                        )
                    )
                ),
                fmt(
                    mean_or_none(
                        collect(
                            group,
                            RECOMMENDED_PLACEMENT,
                            HEADLINE_RULE,
                            TPR_KEYS[0],
                            "tpr",
                        )
                    ),
                    places=2,
                ),
                fmt(
                    mean_or_none(
                        collect(
                            group,
                            RECOMMENDED_PLACEMENT,
                            HEADLINE_RULE,
                            TPR_KEYS[1],
                            "tpr",
                        )
                    ),
                    places=2,
                ),
                fmt(
                    mean_or_none(
                        collect(
                            group,
                            PUBLISHED_PLACEMENT,
                            HEADLINE_RULE,
                            HEADLINE_KEY,
                            "auroc",
                        )
                    )
                ),
            ]
        )
    return rows


def full_rows(stats: dict[str, dict]) -> list[list[str]]:
    """Every cached placement with enough cells, most-covered first."""
    rows = []
    for placement, by_rule in sorted(
        stats.items(), key=lambda item: -item[1][HEADLINE_RULE]["n"]
    ):
        if by_rule[HEADLINE_RULE]["n"] < MIN_CELLS_FOR_ROW:
            continue
        rows.append(
            [
                placement_words(placement),
                str(by_rule["matched"]["n"]),
                fmt(by_rule["matched"]["mean"]),
                str(by_rule["adaptive"]["n"]),
                fmt(by_rule["adaptive"]["mean"]),
                str(by_rule["below_chance"]),
            ]
        )
    return rows


def floor_macros(cells: list[dict]) -> dict[str, tuple[str, str]]:
    """The recommended placement's worst cells, so the section states its floor.

    A mean over 86 models hides where the method fails, and the failures are the
    part a defender needs, so the lowest reading, the cells below chance and the
    cells whose ladder never reached the shift target are all macros.
    """
    readings = []
    unreached = []
    for cell in cells:
        auroc = reading(
            cell["report"], RECOMMENDED_PLACEMENT, HEADLINE_RULE, HEADLINE_KEY, "auroc"
        )
        if auroc is None:
            unreached.append(cell)
            continue
        readings.append((auroc, cell))

    readings.sort(key=lambda item: item[0])
    worst_auroc, worst_cell = readings[0]
    macros = {
        "swin_recommended_floor": (
            fmt(worst_auroc),
            "lowest single-model AUROC of the recommended placement on Swin",
        ),
        "swin_recommended_floor_cell": (
            f"{attack_label(worst_cell['attack'])} at {worst_cell['poison_rate'] * 100:g}\% on {dataset_label(worst_cell['dataset'])}",
            "the Swin cell holding swin_recommended_floor",
        ),
        "swin_recommended_below_chance": (
            str(sum(1 for auroc, _ in readings if auroc < 0.5)),
            "Swin cells where the recommended placement reads below chance",
        ),
        "swin_shift_target_unreached": (
            str(len(unreached)),
            "Swin cells whose swept rate ladder never reached the adaptive shift target",
        ),
    }
    hardest = sorted({cell["attack"] for _, cell in readings[:4]})
    macros["swin_hardest_attacks"] = (
        ", ".join(attack_label(attack) for attack in hardest),
        "the attacks holding the 4 lowest Swin readings of the recommended placement",
    )
    return macros


def half_macros(cells: list[dict], protocol: dict) -> dict[str, tuple[str, str]]:
    """The selection and report halves read separately, for both twins.

    The declared protocol picks a placement on 1 set of datasets and reports it on
    another. A recommendation is licensed only if it also leads on the half the
    choice was made on, so both halves are macros the prose can cite.
    """
    halves = {
        "selection": protocol["select_on_datasets"],
        "report": protocol["report_on_datasets"],
    }
    macros = {}
    for half, datasets in halves.items():
        group = [cell for cell in cells if cell["dataset"] in datasets]
        for stem, placement in (
            ("recommended", RECOMMENDED_PLACEMENT),
            ("twin", TWIN_PLACEMENT),
        ):
            aurocs = collect(group, placement, HEADLINE_RULE, HEADLINE_KEY, "auroc")
            macros[f"swin_{half}_half_{stem}_auroc"] = (
                fmt(mean_or_none(aurocs)),
                f"Swin mean AUROC of the {stem} placement on the {half} half, {len(aurocs)} cells",
            )
            macros[f"swin_{half}_half_{stem}_n"] = (
                str(len(aurocs)),
                f"Swin cells behind swin_{half}_half_{stem}_auroc",
            )
        deltas = paired_gain(
            group, RECOMMENDED_PLACEMENT, TWIN_PLACEMENT, HEADLINE_RULE
        )
        macros[f"swin_{half}_half_twin_gap"] = (
            fmt(mean_or_none(deltas), signed=True),
            f"Swin paired AUROC gap, recommended minus twin, {half} half, {len(deltas)} cells",
        )
    return macros


def main() -> None:
    args = build_parser_with_checkpoints(__doc__).parse_args()
    declaration = load_declaration(args.declaration)
    cells = swin_cells(args.results_dir, args.checkpoints_dir, declaration["asr_bar"])
    inputs = [
        f"{args.results_dir}/swin_*/psbd_metrics.json ({len(cells)} implanted cells)",
        f"{args.checkpoints_dir}/swin_*/args.json",
    ]
    by_dataset = collections.Counter(cell["dataset"] for cell in cells)
    stats = placement_stats(cells)
    scope = ", ".join(
        f"{count} on {dataset_label(dataset)}"
        for dataset, count in sorted(by_dataset.items())
    )

    write_table(
        path=os.path.join(args.paper_dir, "tables", "swin_main.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Swin-S. Mean AUROC and mean TPR at the 10\\% and 20\\% false-positive "
            "budgets at the adaptive rule, with the paired AUROC gain over dropout "
            "after both residual adds and its 95\\% bootstrap interval over the cells "
            f"carrying both placements. Models are {len(cells)} Swin-S checkpoints "
            f"whose attack success clears {declaration['asr_bar']:g}, {scope}."
        ),
        label="tab:swin-main",
        header=[
            "placement",
            "n",
            "AUROC",
            "TPR@10",
            "TPR@20",
            "gain",
            "95% interval",
        ],
        rows=body_rows(cells, args.bootstrap, args.seed),
        align="lrrrrrl",
    )

    write_table(
        path=os.path.join(args.paper_dir, "tables", "swin_attacks.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Swin-S per attack. Token masking at the attention input against dropout "
            "after both residual adds, at the adaptive rule and the headline quantile. "
            "The shaded row is the mean over the dataset above it."
        ),
        label="tab:swin-attacks",
        header=[
            "attack",
            "rate %",
            "AUROC",
            "TPR@10",
            "TPR@20",
            "AUROC published",
        ],
        rows=attack_rows(cells),
        align="llrrrr",
    )

    write_table(
        path=os.path.join(args.paper_dir, "tables", "swin_placements.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Swin-S: mean AUROC at the headline quantile for every cached placement "
            f"with at least {MIN_CELLS_FOR_ROW} implanted models, at the matched and "
            f"adaptive rules, over {len(cells)} models, {scope}. Coverage is uneven "
            "across placements, so rows are not paired."
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
        rows=full_rows(stats),
        align="lrrrrr",
    )

    macros = {
        "swin_cells": (
            str(len(cells)),
            "Swin cells whose attack cleared the bar and carry PSBD metrics",
        ),
        "swin_datasets": (
            str(len(by_dataset)),
            "datasets with at least 1 implanted Swin cell",
        ),
        "swin_placements_reported": (
            str(len(full_rows(stats))),
            f"Swin placements with at least {MIN_CELLS_FOR_ROW} implanted cells",
        ),
    }
    for dataset, count in sorted(by_dataset.items()):
        macros[f"swin_cells_{dataset}"] = (
            str(count),
            f"implanted Swin cells on {dataset_label(dataset)}",
        )

    gain_rows = []
    for stem, placement_a, placement_b in (
        (
            "swin_gain_recommended_minus_published",
            RECOMMENDED_PLACEMENT,
            PUBLISHED_PLACEMENT,
        ),
        (
            "swin_gain_dropout_input_minus_published",
            SWIN_DROPOUT_INPUT,
            PUBLISHED_PLACEMENT,
        ),
        (
            "swin_gain_recommended_minus_pre_residual",
            RECOMMENDED_PLACEMENT,
            "pre_residual",
        ),
        ("swin_gain_recommended_minus_twin", RECOMMENDED_PLACEMENT, TWIN_PLACEMENT),
        (
            "swin_gain_recommended_minus_same_site_dropout",
            RECOMMENDED_PLACEMENT,
            SWIN_DROPOUT_INPUT,
        ),
    ):
        label = f"{placement_words(placement_a)} minus {placement_words(placement_b)}"
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
            if rule != HEADLINE_RULE:
                continue
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

    macros.update(half_macros(cells, declaration["selection_protocol"]))
    macros.update(floor_macros(cells))
    macros["swin_seed_replicates"] = (
        str(
            len(
                glob.glob(
                    os.path.join(args.checkpoints_dir, "swin_*_seed_*", "args.json")
                )
            )
        ),
        "Swin training-seed replicate checkpoints on disk",
    )
    recommended = stats.get(RECOMMENDED_PLACEMENT, {}).get(HEADLINE_RULE, {})
    macros["swin_recommended_auroc_adaptive"] = (
        fmt(recommended.get("mean")),
        f"Swin mean AUROC of the recommended placement at the adaptive rule over {recommended.get('n', 0)} cells",
    )
    macros["swin_recommended_n"] = (
        str(recommended.get("n", 0)),
        "Swin cells carrying the recommended placement",
    )
    for stem, placement, key in (
        ("swin_recommended_tpr10", RECOMMENDED_PLACEMENT, TPR_KEYS[0]),
        ("swin_recommended_tpr20", RECOMMENDED_PLACEMENT, TPR_KEYS[1]),
        ("swin_published_tpr10", PUBLISHED_PLACEMENT, TPR_KEYS[0]),
        ("swin_published_tpr20", PUBLISHED_PLACEMENT, TPR_KEYS[1]),
    ):
        values = collect(cells, placement, HEADLINE_RULE, key, "tpr")
        macros[stem] = (
            fmt(mean_or_none(values), places=2),
            f"Swin mean TPR of {placement_words(placement)} at the {key} budget over {len(values)} cells",
        )
    same_site = stats.get(SWIN_DROPOUT_INPUT, {}).get(HEADLINE_RULE, {})
    macros["swin_same_site_dropout_auroc"] = (
        fmt(same_site.get("mean")),
        f"Swin mean AUROC of dropout at the attention input over {same_site.get('n', 0)} cells",
    )
    published = stats.get(PUBLISHED_PLACEMENT, {}).get(HEADLINE_RULE, {})
    macros["swin_published_auroc_adaptive"] = (
        fmt(published.get("mean")),
        f"Swin mean AUROC of the published placement at the adaptive rule over {published.get('n', 0)} cells",
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
