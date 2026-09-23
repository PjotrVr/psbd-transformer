"""H24: AUROC and TPR against the number of Monte Carlo forward passes k.

The PSBD paper fixes k = 3. docs/hypothesis/H24-monte-carlo-passes.md shows the
gain from more passes is real and concentrated at low poison rate. k = 1 and
k = 2 are free: the cache stores per-pass tracked-class probabilities as
(passes, n), so slicing the first j rows of the k = 3 cache under
before_attention_norm_token_mask/ gives k = 1 and k = 2 with no recomputation on
GPU. k = 20 was paid for on 8 pilot cells only, cached under a sibling directory
before_attention_norm_token_mask_k20/ with 10 rates and 20 passes each. k = 5 and
k = 10 are slices of that same tensor.

Every reading uses the adaptive rule's rate for the cell (cli.compare_detectors
.psbd_rate, rule "adaptive"), held fixed across k, since the question is what
more sampling buys at a fixed disturbance, not what a different rate buys.

    PYTHONPATH=. python scripts/paper/fig_forward_passes.py \\
        --results-dir /lustre/home/pstika/projects/PSBD-ViT/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import scripts.paper._style  # noqa: E402,F401  the shared figure style
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from cli.compare_detectors import psbd_rate  # noqa: E402
from defences.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import (
    RECOMMENDED_PLACEMENT,
    pair_clean_to_backdoor,
    detection_report,
)  # noqa: E402
from defences.scores import psu_ratio_from_cache  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    build_parser,
    clearing_cells,
    figure_sidecar,
    load_coverage,
    load_psbd_metrics,
    mean_or_none,
    std_or_none,
)

GENERATOR = "scripts/paper/fig_forward_passes.py"

# The 4 datasets this project validates on (docs/hypothesis/README.md), which is
# what "65 clearing cells" means everywhere in the paper. eurosat and svhn are
# exploratory additions to coverage.json and sit outside the validated panel.
PRIMARY_DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")

# Which cells carry a k = 20 cache is read off disk rather than listed here. The
# list used to be the 8 checkpoints H24's paid half ran, and it stayed at 8 while
# the sweep grew, so the figure reported a pilot of 8 when 21 caches existed. A
# hardcoded population silently stops tracking the sweep that fills it.

BASE_K_VALUES = (1, 2, 3)
K20_K_VALUES = (5, 10, 20)
ALL_K_VALUES = BASE_K_VALUES + K20_K_VALUES

# quantile is the false-positive budget by construction (defences.decision), so
# "TPR at 10% FPR" is detection_report at quantile 0.10.
FPR_BUDGETS = (0.10, 0.20)

ALL_CELLS_COLOUR = "#0072B2"
PILOT_COLOUR = "#D55E00"


def position_config_for_k(k: int) -> str:
    """Which on-disk cache directory holds the passes k slices from."""
    config = RECOMMENDED_PLACEMENT if k <= 3 else f"{RECOMMENDED_PLACEMENT}_k20"
    return config


def k20_cache_dir(psbd_dir: str) -> str:
    path = os.path.join(psbd_dir, f"{RECOMMENDED_PLACEMENT}_k20")
    return path


def split_psu_ratio(psbd_dir: str, rate: float, split: str, k: int):
    """Fractional PSU for 1 split, from the first k of the cached passes.

    Reads the no-perturbation baseline (n, num_classes) and the (passes, n)
    per-pass probabilities, slices the first k passes and returns the (n,)
    psu_ratio_from_cache score.
    """
    baseline_probs, baseline_labels, _ = load_baseline(baseline_path(psbd_dir, split))
    position_config = position_config_for_k(k)
    per_pass_probs, _ = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, position_config, rate, split)
    )
    sliced = per_pass_probs[:k]  # (k, n)
    psu_ratio = psu_ratio_from_cache(baseline_probs, baseline_labels, sliced)
    return psu_ratio


def cell_metrics_at_k(psbd_dir: str, manifest: dict, rate: float, k: int) -> dict:
    """AUROC and TPR at both FPR budgets, at this k, for 1 cell."""
    validation_psu = split_psu_ratio(psbd_dir, rate, "validation", k)
    clean_psu = pair_clean_to_backdoor(
        split_psu_ratio(psbd_dir, rate, "clean", k), manifest
    )
    backdoor_psu = split_psu_ratio(psbd_dir, rate, "backdoor", k)

    reports = {
        budget: detection_report(validation_psu, clean_psu, backdoor_psu, budget)
        for budget in FPR_BUDGETS
    }
    # AUROC does not depend on the quantile, only TPR and FPR do, so either
    # budget's report carries the same value.
    metrics = {
        "auroc": reports[FPR_BUDGETS[0]]["auroc"],
        "tpr10": reports[0.10]["tpr"],
        "tpr20": reports[0.20]["tpr"],
    }
    return metrics


def adaptive_rate_for_cell(
    results_dir: str, folder: str
) -> tuple[float | None, dict | None]:
    """The recommended placement's adaptive-rule rate for a cell, or None."""
    report = load_psbd_metrics(results_dir, folder)
    if report is None:
        return None, None
    block = report.get("placements", {}).get(RECOMMENDED_PLACEMENT)
    if block is None:
        return None, None
    rate = psbd_rate(block, "adaptive")
    return rate, block


def collect_cell(
    results_dir: str, folder: str, k_values: tuple[int, ...]
) -> dict | None:
    """Every requested k's metrics for 1 cell at its own adaptive rate, or None."""
    rate, _ = adaptive_rate_for_cell(results_dir, folder)
    if rate is None:
        return None

    psbd_dir = os.path.join(results_dir, folder, "psbd")
    manifest = read_split_manifest(psbd_dir)
    by_k = {k: cell_metrics_at_k(psbd_dir, manifest, rate, k) for k in k_values}
    return {"rate": rate, "by_k": by_k}


def aggregate(per_cell: dict[str, dict], k_values: tuple[int, ...]) -> dict:
    """Mean and standard deviation over cells, per k, for each metric."""
    aggregated = {}
    for k in k_values:
        for metric in ("auroc", "tpr10", "tpr20"):
            values = [record["by_k"][k][metric] for record in per_cell.values()]
            aggregated.setdefault(k, {})[metric] = {
                "mean": mean_or_none(values),
                "std": std_or_none(values),
                "n_cells": len(values),
            }
    return aggregated


def draw_panel(ax, k_values, pilot_agg, all_cells_agg, metric, ylabel):
    """1 panel: what each extra pass buys, as the change from a single pass.

    The 2 series are different populations, the pilot being 8 cells at 1% poisoning
    and the panel being all 65 backdoored models, so their absolute
    levels differ by more than 0.04 for reasons that have nothing to do with k.
    Plotting them together on an absolute axis invites the reader to read that gap
    as an effect of sampling. The question the figure answers is what an extra pass
    buys, so both series are drawn as the change from k = 1 and both start at 0,
    which is the comparison that is actually licensed.
    """
    pilot_k = list(k_values)
    pilot_base = pilot_agg[pilot_k[0]][metric]["mean"]
    pilot_delta = [pilot_agg[k][metric]["mean"] - pilot_base for k in pilot_k]
    pilot_n = pilot_agg[pilot_k[0]][metric]["n_cells"]
    ax.plot(
        pilot_k,
        pilot_delta,
        marker="o",
        markersize=4,
        linewidth=1.4,
        color=PILOT_COLOUR,
        label=f"{pilot_n} models at 1% poisoning, k to {pilot_k[-1]}",
    )

    base_k = list(BASE_K_VALUES)
    all_base = all_cells_agg[base_k[0]][metric]["mean"]
    all_delta = [all_cells_agg[k][metric]["mean"] - all_base for k in base_k]
    all_n = all_cells_agg[base_k[0]][metric]["n_cells"]
    ax.plot(
        base_k,
        all_delta,
        marker="s",
        markersize=4,
        linewidth=1.2,
        linestyle="--",
        color=ALL_CELLS_COLOUR,
        label=f"all {all_n} backdoored models, k to {base_k[-1]}",
    )

    # The paper's claim is that the third pass is worth having and the twentieth is
    # not, so the k = 3 reading is marked rather than left to the reader to find.
    ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=1)
    ax.axvline(3, color="0.6", linewidth=0.6, linestyle=":", zorder=1)

    ax.set_xscale("log", base=2)
    ax.set_xticks(list(k_values))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("forward passes k")
    ax.set_ylabel(f"{ylabel} gain")


def write_figure(
    args, pilot_agg: dict, all_cells_agg: dict, pilot_folders: list[str]
) -> str:
    fig, axes = plt.subplots(
        1, 3, figsize=(scripts.paper._style.TEXT_WIDTH, 2.4), sharex=True
    )

    draw_panel(axes[0], ALL_K_VALUES, pilot_agg, all_cells_agg, "auroc", "AUROC")
    draw_panel(
        axes[1],
        ALL_K_VALUES,
        pilot_agg,
        all_cells_agg,
        "tpr10",
        "TPR at 10% FPR",
    )
    draw_panel(
        axes[2],
        ALL_K_VALUES,
        pilot_agg,
        all_cells_agg,
        "tpr20",
        "TPR at 20% FPR",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    path = os.path.join(args.paper_dir, "figures", "fig_forward_passes.pdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    return path


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage = load_coverage(args.results_dir)
    cells = clearing_cells(coverage)
    primary_folders = [
        cell["folder_name"] for cell in cells if cell["dataset"] in PRIMARY_DATASETS
    ]

    all_cells_per_cell = {}
    for folder in primary_folders:
        record = collect_cell(args.results_dir, folder, BASE_K_VALUES)
        if record is not None:
            all_cells_per_cell[folder] = record
    all_cells_agg = aggregate(all_cells_per_cell, BASE_K_VALUES)

    # Every primary clearing cell that has a k = 20 cache contributes, so the
    # curve grows as the sweep does without this file being edited.
    candidates = [
        folder
        for folder in primary_folders
        if os.path.isdir(k20_cache_dir(os.path.join(args.results_dir, folder, "psbd")))
    ]
    pilot_per_cell = {}
    pilot_excluded = {}
    for folder in candidates:
        record = collect_cell(args.results_dir, folder, ALL_K_VALUES)
        if record is None:
            pilot_excluded[folder] = (
                "no adaptive-rule rate for the recommended placement"
            )
            continue
        pilot_per_cell[folder] = record
    pilot_agg = aggregate(pilot_per_cell, ALL_K_VALUES)

    print(
        f"{len(all_cells_per_cell)} of {len(primary_folders)} primary clearing cells "
        f"carry k<=3 readings. {len(pilot_per_cell)} of {len(candidates)} cells with "
        "a k=20 cache contribute the long curve"
    )
    if pilot_excluded:
        print(f"pilot cells excluded: {pilot_excluded}")

    figure_path = write_figure(args, pilot_agg, all_cells_agg, list(pilot_per_cell))

    plotted = {
        "recommended_placement": RECOMMENDED_PLACEMENT,
        "rate_rule": "adaptive",
        "fpr_budgets": {"tpr10": 0.10, "tpr20": 0.20},
        "k_values": {
            "base_cache": list(BASE_K_VALUES),
            "k20_cache": list(K20_K_VALUES),
        },
        "pilot_cells_used": {
            folder: {"rate": record["rate"], "by_k": record["by_k"]}
            for folder, record in pilot_per_cell.items()
        },
        "pilot_cells_excluded": pilot_excluded,
        "pilot_group_aggregate": pilot_agg,
        "all_cells_group_aggregate": all_cells_agg,
        "all_cells_per_cell": {
            folder: {"rate": record["rate"], "by_k": record["by_k"]}
            for folder, record in all_cells_per_cell.items()
        },
    }
    figure_sidecar(
        path=figure_path.replace(".pdf", ".json"),
        generator=GENERATOR,
        inputs=[
            f"{args.results_dir}/<folder>/psbd_metrics.json (65 clearing cells)",
            f"{args.results_dir}/<folder>/psbd/{RECOMMENDED_PLACEMENT}/rate_*_"
            "{validation,clean,backdoor}.pt",
            f"{args.results_dir}/<folder>/psbd/{RECOMMENDED_PLACEMENT}_k20/rate_*_"
            "{validation,clean,backdoor}.pt (8 candidate pilot cells)",
        ],
        plotted=plotted,
    )
    print(f"wrote {figure_path}, {figure_path.replace('.pdf', '.png')} and its sidecar")


if __name__ == "__main__":
    main()
