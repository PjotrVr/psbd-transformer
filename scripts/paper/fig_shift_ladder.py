"""How the shift-ratio target chosen for detection trades off against AUROC and TPR.

Every clearing cell's psbd_metrics.json carries a whole rate ladder for the
recommended placement, each rung with its own achieved clean-validation shift
ratio and its own AUROC and TPR at the false-positive budgets. Interpolating
every cell's ladder onto 1 shared grid of shift ratios (defences.decision
.interpolate_at_target_shift, linear between the 2 rungs bracketing the target,
None outside a cell's own bracket) makes cells with different rate grids
comparable at the same disturbance level, which is exactly the comparison
PLACEMENT_MATCH_TARGET and ADAPTIVE_SHIFT_TARGET are built for
(defences.decision).

    PYTHONPATH=. python scripts/paper/fig_shift_ladder.py \\
        --results-dir /lustre/home/pstika/projects/PSBD-ViT/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import scripts.paper._style  # noqa: E402,F401  the shared figure style
import matplotlib.pyplot as plt  # noqa: E402

from defences.decision import (  # noqa: E402
    ADAPTIVE_SHIFT_TARGET,
    PLACEMENT_MATCH_TARGET,
    RECOMMENDED_PLACEMENT,
    interpolate_at_target_shift,
)
from scripts.paper._common import (  # noqa: E402
    build_parser,
    clearing_cells,
    figure_sidecar,
    load_coverage,
    load_psbd_metrics,
    mean_or_none,
    std_or_none,
)

GENERATOR = "scripts/paper/fig_shift_ladder.py"

# The 4 datasets this project validates on (docs/hypothesis/README.md): 65
# clearing cells everywhere in the paper means these, never coverage.json's
# exploratory eurosat and svhn additions.
PRIMARY_DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")

# The task's hard-attack subset for this figure, narrower than
# defences.decision.HARD_ATTACKS (which also carries lc and adaptive_blend).
HARD_ATTACKS = ("bpp", "wanet", "tact", "sig")

GRID = [round(0.20 + 0.02 * i, 2) for i in range(40)]  # 0.20, 0.22, ..., 0.98
REPORT_TARGETS = (0.6, 0.8, 0.9, 0.95)

# quantile is the false-positive budget by construction, so TPR at 1% and 10%
# FPR reads the q0.01 and q0.10 blocks of detection_psu_ratio, the headline
# fractional-PSU statistic (canon: fractional PSU is the deployable score).
TPR_BUDGET_KEYS = {"tpr01": "q0.01", "tpr10": "q0.10"}
# AUROC does not depend on the quantile, so any budget's block carries it.
AUROC_KEY = "q0.01"

ALL_CELLS_COLOUR = "#0072B2"
HARD_ATTACKS_COLOUR = "#D55E00"


def cell_ladder(results_dir: str, folder: str) -> dict | None:
    """shift_by_rate and 1 value_by_rate per metric, for the recommended placement."""
    report = load_psbd_metrics(results_dir, folder)
    if report is None:
        return None
    block = report.get("placements", {}).get(RECOMMENDED_PLACEMENT)
    if block is None:
        return None

    shift_by_rate = {}
    value_by_rate = {"auroc": {}, "tpr01": {}, "tpr10": {}}
    for row in block.get("rates", []):
        rate = row["rate"]
        shift_by_rate[rate] = row["shift_ratio"]["validation"]
        detection = row["detection_psu_ratio"]
        value_by_rate["auroc"][rate] = detection[AUROC_KEY]["auroc"]
        for metric, key in TPR_BUDGET_KEYS.items():
            value_by_rate[metric][rate] = detection[key]["tpr"]

    if not shift_by_rate:
        return None
    return {"shift_by_rate": shift_by_rate, "value_by_rate": value_by_rate}


def cell_curve(ladder: dict, grid: list[float]) -> dict[str, list[float | None]]:
    """Each metric interpolated onto the shared grid, None where the ladder does not bracket it."""
    curve = {}
    for metric, value_by_rate in ladder["value_by_rate"].items():
        curve[metric] = [
            interpolate_at_target_shift(ladder["shift_by_rate"], value_by_rate, target)
            for target in grid
        ]
    return curve


def cell_at_targets(
    ladder: dict, targets: tuple[float, ...]
) -> dict[str, dict[float, float | None]]:
    """Each metric interpolated at the report targets only, for the sidecar table."""
    at_targets = {}
    for metric, value_by_rate in ladder["value_by_rate"].items():
        at_targets[metric] = {
            target: interpolate_at_target_shift(
                ladder["shift_by_rate"], value_by_rate, target
            )
            for target in targets
        }
    return at_targets


def aggregate_curve(
    curves: list[dict[str, list[float | None]]], grid: list[float]
) -> dict[str, dict[str, list[float | None]]]:
    """Mean, standard deviation and cell count at each grid point, per metric."""
    metrics = ("auroc", "tpr01", "tpr10")
    aggregated = {metric: {"mean": [], "std": [], "n_cells": []} for metric in metrics}
    for index in range(len(grid)):
        for metric in metrics:
            values = [c[metric][index] for c in curves if c[metric][index] is not None]
            aggregated[metric]["mean"].append(mean_or_none(values))
            aggregated[metric]["std"].append(std_or_none(values) or 0.0)
            aggregated[metric]["n_cells"].append(len(values))
    return aggregated


def aggregate_targets(
    records: list[dict[str, dict[float, float | None]]], targets: tuple[float, ...]
) -> dict[str, dict[str, float | None]]:
    """Mean over cells at each report target, per metric, skipping cells without a bracket."""
    metrics = ("auroc", "tpr01", "tpr10")
    aggregated = {metric: {} for metric in metrics}
    for metric in metrics:
        for target in targets:
            values = [
                r[metric][target] for r in records if r[metric][target] is not None
            ]
            aggregated[metric][target] = {
                "mean": mean_or_none(values),
                "n_cells": len(values),
            }
    return aggregated


def draw_panel(ax, grid, all_agg, hard_agg, metric, ylabel):
    grid_array = np.array(grid)
    for label, agg, colour in (
        ("all 65 backdoored models", all_agg, ALL_CELLS_COLOUR),
        ("29 models under hard attacks (BPP, WaNet, TaCT, SIG)", hard_agg, HARD_ATTACKS_COLOUR),
    ):
        mean = np.array([v if v is not None else np.nan for v in agg[metric]["mean"]])
        std = np.array(agg[metric]["std"])
        ax.plot(grid_array, mean, linewidth=1.4, color=colour, label=label)
        ax.fill_between(
            grid_array, mean - std, mean + std, color=colour, alpha=0.18, linewidth=0
        )

    for target, style in ((PLACEMENT_MATCH_TARGET, ":"), (ADAPTIVE_SHIFT_TARGET, "--")):
        ax.axvline(target, color="black", linewidth=0.8, linestyle=style)

    ax.set_xlabel("clean-validation shift ratio (achieved, target)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.20, 0.98)


def write_figure(args, all_agg, hard_agg) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharex=True)
    draw_panel(axes[0], GRID, all_agg, hard_agg, "auroc", "AUROC")
    draw_panel(axes[1], GRID, all_agg, hard_agg, "tpr01", "TPR at 1% FPR")
    draw_panel(axes[2], GRID, all_agg, hard_agg, "tpr10", "TPR at 10% FPR")
    axes[0].legend(loc="lower right", fontsize=6)
    fig.tight_layout()

    path = os.path.join(args.paper_dir, "figures", "fig_shift_ladder.pdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    return path


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage = load_coverage(args.results_dir)
    cells = clearing_cells(coverage)
    primary_cells = [cell for cell in cells if cell["dataset"] in PRIMARY_DATASETS]

    ladders, folders_by_group = {}, {"all": [], "hard": []}
    for cell in primary_cells:
        ladder = cell_ladder(args.results_dir, cell["folder_name"])
        if ladder is None:
            continue
        ladders[cell["folder_name"]] = ladder
        folders_by_group["all"].append(cell["folder_name"])
        if cell["attack"] in HARD_ATTACKS:
            folders_by_group["hard"].append(cell["folder_name"])

    curves = {folder: cell_curve(ladder, GRID) for folder, ladder in ladders.items()}
    all_agg = aggregate_curve([curves[f] for f in folders_by_group["all"]], GRID)
    hard_agg = aggregate_curve([curves[f] for f in folders_by_group["hard"]], GRID)

    at_targets = {
        folder: cell_at_targets(ladder, REPORT_TARGETS)
        for folder, ladder in ladders.items()
    }
    all_targets_agg = aggregate_targets(
        [at_targets[f] for f in folders_by_group["all"]], REPORT_TARGETS
    )
    hard_targets_agg = aggregate_targets(
        [at_targets[f] for f in folders_by_group["hard"]], REPORT_TARGETS
    )

    print(
        f"{len(ladders)} of {len(primary_cells)} primary clearing cells carry a "
        f"{RECOMMENDED_PLACEMENT} rate ladder, {len(folders_by_group['hard'])} of "
        "them hard attacks"
    )

    figure_path = write_figure(args, all_agg, hard_agg)

    plotted = {
        "recommended_placement": RECOMMENDED_PLACEMENT,
        "hard_attacks": list(HARD_ATTACKS),
        "matched_rule_target": PLACEMENT_MATCH_TARGET,
        "adaptive_rule_target": ADAPTIVE_SHIFT_TARGET,
        "grid": GRID,
        "all_cells": {
            "folders": folders_by_group["all"],
            "curve": all_agg,
            "at_targets": all_targets_agg,
        },
        "hard_attacks_cells": {
            "folders": folders_by_group["hard"],
            "curve": hard_agg,
            "at_targets": hard_targets_agg,
        },
        "per_cell_at_targets": at_targets,
    }
    figure_sidecar(
        path=figure_path.replace(".pdf", ".json"),
        generator=GENERATOR,
        inputs=[
            f"{args.results_dir}/<folder>/psbd_metrics.json (65 clearing cells, "
            f"placements.{RECOMMENDED_PLACEMENT}.rates)"
        ],
        plotted=plotted,
    )
    print(f"wrote {figure_path}, {figure_path.replace('.pdf', '.png')} and its sidecar")


if __name__ == "__main__":
    main()
