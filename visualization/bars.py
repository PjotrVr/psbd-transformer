"""Bar charts and the metric radar: neuron activation, class purity, stealth and the summary.

The class purity figure is upstream's stacked Rectangle per neuron redrawn as a
raster, 1 row per dimension whose columns are filled with class colours in
proportion to the shares, which draws 768 rows in 1 imshow instead of
thousands of patches.
"""

import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

from .style import (
    CLEAN_COLOUR,
    DOUBLE_COLUMN,
    SINGLE_COLUMN,
    TRIGGERED_COLOUR,
    paper_style,
)

# Horizontal resolution of the purity raster. Shares below 1/PURITY_COLUMNS of a
# row vanish, which at 400 is a quarter of a percent.
PURITY_COLUMNS = 400


def overlaid_activation_bars(
    clean_mean: np.ndarray, triggered_mean: np.ndarray, title: str
) -> matplotlib.figure.Figure:
    """Clean and triggered mean activation per dimension as 2 overlaid bar series.

    Dimensions are sorted by the clean value, descending, as visual_na.py
    sorts them, so a triggered bar that towers over a small clean bar is the
    signature of a backdoor dimension.
    """
    order = np.argsort(clean_mean)[::-1]  # (dim,)
    positions = np.arange(clean_mean.shape[0])

    with paper_style():
        figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN, 2.4))
        axis.bar(
            positions,
            clean_mean[order],
            width=1.0,
            alpha=0.7,
            color=CLEAN_COLOUR,
            label="clean",
        )
        axis.bar(
            positions,
            triggered_mean[order],
            width=1.0,
            alpha=0.7,
            color=TRIGGERED_COLOUR,
            label="triggered",
        )
        axis.set_xlim(0, clean_mean.shape[0])
        axis.set_xlabel("dimension, sorted by clean mean")
        axis.set_ylabel("mean token-sum activation")
        axis.set_title(title)
        axis.legend(frameon=False)
    return figure


def purity_raster(purity: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """The (dim, PURITY_COLUMNS, 3) image whose row d is filled by the shares of purity[d].

    purity is (dim, num_colours) with rows summing to 1 and colours
    (num_colours, 3) RGB. Shares are laid left to right in column order.
    """
    boundaries = np.round(np.cumsum(purity, axis=1) * PURITY_COLUMNS).astype(
        int
    )  # (dim, num_colours)
    column = np.arange(PURITY_COLUMNS)[None, :]  # (1, PURITY_COLUMNS)

    # The colour of a pixel is the first class whose cumulative boundary lies
    # past it, which argmax over the boolean comparison finds.
    past = column[:, :, None] < boundaries[:, None, :]  # (dim, PURITY_COLUMNS, colours)
    owner = past.argmax(axis=2)  # (dim, PURITY_COLUMNS)
    raster = colours[owner]  # (dim, PURITY_COLUMNS, 3)
    return raster


def class_purity_bars(
    purity: np.ndarray,
    colours: np.ndarray,
    legend_labels: list[str],
    legend_colours: list[str],
    title: str,
) -> matplotlib.figure.Figure:
    """Stacked class shares per dimension, 1 raster row per dimension, black for poisoned.

    purity is (dim, num_colours) in the row order the caller wants drawn and
    colours the matching (num_colours, 3) table. The legend lists only the
    classes present.
    """
    raster = purity_raster(purity, colours)  # (dim, PURITY_COLUMNS, 3)

    with paper_style():
        figure, axis = plt.subplots(figsize=(SINGLE_COLUMN, 5.5))
        axis.imshow(raster, aspect="auto", interpolation="nearest")
        axis.set_xticks([0, PURITY_COLUMNS // 2, PURITY_COLUMNS], ["0", "0.5", "1"])
        axis.set_xlabel("share of the top images")
        axis.set_ylabel("dimension, row order in the sidecar")
        axis.set_title(title)
        handles = [
            matplotlib.patches.Patch(facecolor=colour, label=label)
            for label, colour in zip(legend_labels, legend_colours)
        ]
        axis.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(6, len(handles)),
            frameon=False,
        )
    return figure


def stealth_bars(
    metrics: dict[str, float], attack_name: str
) -> matplotlib.figure.Figure:
    """PSNR, SSIM and LPIPS of the attack's trigger, 3 panels with 1 bar and its spread each."""
    names = ("psnr", "ssim", "lpips")
    units = ("PSNR (dB)", "SSIM", "LPIPS")

    with paper_style():
        figure, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.0))
        for axis, name, unit in zip(axes, names, units):
            axis.bar(
                [attack_name],
                [metrics[f"{name}_mean"]],
                yerr=[metrics[f"{name}_std"]],
                color=CLEAN_COLOUR,
                capsize=3,
                width=0.5,
            )
            axis.set_ylabel(unit)
        figure.suptitle(f"trigger visibility, {metrics['n_pairs']} pairs")
    return figure


def metric_radar(
    names: list[str], values: list[float], title: str
) -> matplotlib.figure.Figure:
    """visual_metric.py's radar: 1 axis per metric, the polygon closed, radius capped at 1."""
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)  # (n,)
    closed_values = np.concatenate([values, [values[0]]])  # (n + 1,)
    closed_angles = np.concatenate([angles, [angles[0]]])  # (n + 1,)

    with paper_style():
        figure = plt.figure(figsize=(SINGLE_COLUMN, SINGLE_COLUMN))
        axis = figure.add_subplot(111, polar=True)
        axis.plot(closed_angles, closed_values, "o-", linewidth=1.5, color=CLEAN_COLOUR)
        axis.fill(closed_angles, closed_values, alpha=0.25, color=CLEAN_COLOUR)
        axis.set_rmax(1.0)
        axis.set_thetagrids(np.degrees(angles), names)
        axis.tick_params(pad=5)
        axis.set_title(title, va="bottom", pad=18)
    return figure
