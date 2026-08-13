"""Figures for a swept experiment: shift curves, PSU histograms, ROC, tradeoffs.

Reads the artifacts written by experiment_io and mirrors the figures in the
seminar. Plot styling is grouped in one place so every figure stays consistent.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from experiment_io import load_metrics, load_scores

LEGEND_FONT_SIZE = 20
TITLE_FONT_SIZE = 20
AXIS_FONT_SIZE = 20
TICK_LABEL_FONT_SIZE = 16
GRID_ALPHA = 0.3
WIDE_FIG_SIZE = (8, 6)
SQUARE_FIG_SIZE = (6, 6)
MARKER_SIZE = 10
LINE_WIDTH = 4
SPINE_WIDTH = 2
DPI = 300

CLEAN_COLOR = "#2196F3"
BACKDOOR_COLOR = "#F44336"
VALIDATION_COLOR = "#4CAF50"
CURVE_COLOR = "#9C27B0"

_ATTACK_NAMES = {
    "badnet": "BadNet",
    "wanet": "WaNet",
    "blended": "Blend",
    "blend": "Blend",
    "sig": "SIG",
    "trojannn": "TrojanNN",
    "ssba": "SSBA",
    "lc": "Label-Consistent",
    "lf": "Low-Frequency",
}
_DATASET_NAMES = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "gtsrb": "GTSRB",
    "tiny": "Tiny ImageNet",
}


def format_title(folder_name: str, suffix: str = "") -> str:
    parts = folder_name.split("_")
    dataset = _DATASET_NAMES.get(parts[0], parts[0].upper())
    attack = _ATTACK_NAMES.get(parts[1], parts[1].capitalize())
    poison_percent = int(float(f"{parts[2]}.{parts[3]}") * 100)
    return f"{dataset}: {attack}, {poison_percent}% Poison Rate{suffix}"


def figures_dir_for(experiment_dir: str) -> str:
    return experiment_dir.replace("experiments", "figures", 1)


def rows_for(
    metrics: list[dict], quantile: float | None, rate: float | None
) -> list[dict]:
    rows = metrics
    if quantile is not None:
        rows = [r for r in rows if abs(r["quantile"] - quantile) < 1e-6]
    if rate is not None:
        rows = [r for r in rows if abs(r["dropout_rate"] - rate) < 1e-6]
    return sorted(rows, key=lambda r: r["dropout_rate"])


def best_rate(metrics: list[dict], quantile: float = 0.25) -> float:
    """The dropout rate with the highest TPR at the reference quantile."""
    return max(rows_for(metrics, quantile, None), key=lambda r: r["tpr"])[
        "dropout_rate"
    ]


def _style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.tick_params(
        axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE, width=SPINE_WIDTH
    )
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.grid(alpha=GRID_ALPHA)


def _save(fig_dir: str, filename: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, filename), dpi=DPI)


def plot_shift_curves(
    metrics: list[dict], experiment_dir: str, save: bool = True
) -> None:
    rows = rows_for(metrics, quantile=0.25, rate=None)
    rates = [r["dropout_rate"] for r in rows]

    _, ax = plt.subplots(figsize=WIDE_FIG_SIZE)
    ax.plot(
        rates,
        [r["shift_clean"] for r in rows],
        label="Clean",
        color=CLEAN_COLOR,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
    )
    ax.plot(
        rates,
        [r["shift_backdoor"] for r in rows],
        label="Backdoor",
        color=BACKDOOR_COLOR,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
    )
    ax.plot(
        rates,
        [r["shift_clean_val"] for r in rows],
        label="Validation",
        color=VALIDATION_COLOR,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        linestyle="--",
    )
    ax.set_xlabel("Dropout rate", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel("Shift ratio", fontsize=AXIS_FONT_SIZE)
    ax.set_ylim(0, 1.05)
    _style_axes(ax, format_title(Path(experiment_dir).name))
    plt.tight_layout()
    if save:
        _save(figures_dir_for(experiment_dir), "shift_curves.png")
    plt.show()


def plot_psu_histogram(
    metrics: list[dict],
    experiment_dir: str,
    rate: float | None = None,
    quantile: float = 0.25,
    save: bool = True,
) -> None:
    rate = best_rate(metrics) if rate is None else rate
    validation_scores, clean_scores, backdoor_scores = load_scores(experiment_dir, rate)
    threshold = float(torch.quantile(validation_scores.float(), quantile).item())

    _, ax = plt.subplots(figsize=WIDE_FIG_SIZE)
    ax.hist(
        clean_scores.numpy(),
        bins=60,
        alpha=0.6,
        label="Clean",
        color=CLEAN_COLOR,
        density=True,
    )
    ax.hist(
        backdoor_scores.numpy(),
        bins=60,
        alpha=0.6,
        label="Backdoor",
        color=BACKDOOR_COLOR,
        density=True,
    )
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=LINE_WIDTH,
        label=f"Threshold = {threshold:.3f}",
    )
    ax.set_xlabel("PSU score", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel("Density", fontsize=AXIS_FONT_SIZE)
    _style_axes(ax, format_title(Path(experiment_dir).name, f", drop_p={rate}"))
    plt.tight_layout()
    if save:
        _save(
            figures_dir_for(experiment_dir),
            f"histogram_p{str(rate).replace('.', '_')}.png",
        )
    plt.show()


def plot_roc_curve(
    metrics: list[dict],
    experiment_dir: str,
    rate: float | None = None,
    save: bool = True,
) -> None:
    rate = best_rate(metrics) if rate is None else rate
    _, clean_scores, backdoor_scores = load_scores(experiment_dir, rate)

    scores = np.concatenate([-clean_scores.numpy(), -backdoor_scores.numpy()])
    labels = np.concatenate(
        [np.zeros(len(clean_scores)), np.ones(len(backdoor_scores))]
    )
    fpr_points, tpr_points, _ = roc_curve(labels, scores)
    score = roc_auc_score(labels, scores)

    _, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    ax.plot(
        fpr_points,
        tpr_points,
        color=CURVE_COLOR,
        linewidth=LINE_WIDTH,
        label=f"AUROC={score:.3f}",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=LINE_WIDTH)
    ax.set_xlabel("FPR", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel("TPR", fontsize=AXIS_FONT_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    _style_axes(ax, format_title(Path(experiment_dir).name, f", drop_p={rate}"))
    plt.tight_layout()
    if save:
        _save(
            figures_dir_for(experiment_dir), f"roc_p{str(rate).replace('.', '_')}.png"
        )
    plt.show()


def plot_tpr_fpr(
    metrics: list[dict],
    experiment_dir: str,
    quantile: float = 0.25,
    save: bool = True,
) -> None:
    rows = rows_for(metrics, quantile=quantile, rate=None)
    rates = [r["dropout_rate"] for r in rows]

    _, ax = plt.subplots(figsize=WIDE_FIG_SIZE)
    ax.plot(
        rates,
        [r["tpr"] for r in rows],
        label="TPR",
        color=CLEAN_COLOR,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
    )
    ax.plot(
        rates,
        [r["fpr"] for r in rows],
        label="FPR",
        color=BACKDOOR_COLOR,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
    )
    ax.set_xlabel("Dropout rate", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel("Rate", fontsize=AXIS_FONT_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    _style_axes(ax, format_title(Path(experiment_dir).name, f", q={quantile}"))
    plt.tight_layout()
    if save:
        _save(
            figures_dir_for(experiment_dir),
            f"tpr_fpr_q{str(quantile).replace('.', '_')}.png",
        )
    plt.show()


def plot_auroc(metrics: list[dict], experiment_dir: str, save: bool = True) -> None:
    rows = rows_for(metrics, quantile=0.25, rate=None)
    _, ax = plt.subplots(figsize=WIDE_FIG_SIZE)
    ax.plot(
        [r["dropout_rate"] for r in rows],
        [r["auroc"] for r in rows],
        color=CURVE_COLOR,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
    )
    ax.axhline(0.5, linestyle="--", color="black", label="Random", linewidth=LINE_WIDTH)
    ax.set_xlabel("Dropout rate", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel("AUROC", fontsize=AXIS_FONT_SIZE)
    ax.set_ylim(0.0, 1.05)
    _style_axes(ax, format_title(Path(experiment_dir).name))
    plt.tight_layout()
    if save:
        _save(figures_dir_for(experiment_dir), "auroc.png")
    plt.show()


def analyze(experiment_dir: str, quantile: float = 0.25, plot: bool = True) -> dict:
    """Return the best row and optionally render every figure for one attack."""
    metrics = load_metrics(experiment_dir)
    reference_rows = rows_for(metrics, quantile=0.25, rate=None)
    best = max(reference_rows, key=lambda r: r["tpr"])

    if plot:
        rate = best_rate(metrics)
        plot_shift_curves(metrics, experiment_dir)
        plot_auroc(metrics, experiment_dir)
        plot_tpr_fpr(metrics, experiment_dir, quantile=quantile)
        plot_psu_histogram(metrics, experiment_dir, rate=rate, quantile=quantile)
        plot_roc_curve(metrics, experiment_dir, rate=rate)
    return best


def analyze_all(
    experiments_root: str, quantile: float = 0.25, plot: bool = True
) -> None:
    folders = sorted(
        str(p.parent) for p in Path(experiments_root).rglob("metrics.json")
    )
    print(f"Found {len(folders)} experiment folders")
    for folder in folders:
        try:
            analyze(folder, quantile=quantile, plot=plot)
        except Exception as error:
            print(f"FAILED {folder}: {error}")
