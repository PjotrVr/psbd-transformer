"""Layer by dimension heatmaps and per-dimension token maps.

BackdoorBench draws its TAC and Lipschitz figures as 1 Rectangle patch per
neuron per layer, which at 12 layers by 768 dimensions is 9216 patches and a
PDF nobody can open. The same picture as 1 imshow of the (layers, dimensions)
matrix is what these builders produce, with the dimension axis horizontal so a
backdoor dimension shows as a vertical stripe across the depth it lives at.
"""

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from .style import DOUBLE_COLUMN, FONT_SIZE, TOKEN_MAP_CMAP, paper_style

# Rows drawn per token map figure: the top 16 dimensions in a 4 by 4 block per
# population, which is the largest grid whose maps stay legible at 2 columns.
TOKEN_MAP_GRID = 4


def layer_by_dimension_heatmap(
    values: np.ndarray,
    layer_labels: list[str],
    cmap: str,
    colorbar_label: str,
    title: str,
    normalize_by_layer: bool,
) -> matplotlib.figure.Figure:
    """1 imshow of values (num_layers, dim), layers on the vertical axis.

    normalize_by_layer divides every row by its own maximum, upstream's
    --normalize_by_layer, so each layer's strongest dimension is full colour
    and the comparison is within a layer rather than across depth.
    """
    matrix = np.asarray(values, dtype=np.float64)  # (num_layers, dim)
    if matrix.ndim != 2:
        raise ValueError(f"expected a (layers, dim) matrix, got shape {matrix.shape}")
    if normalize_by_layer:
        row_max = matrix.max(axis=1, keepdims=True)  # (num_layers, 1)
        matrix = matrix / np.where(row_max > 0, row_max, 1.0)

    with paper_style():
        figure, axis = plt.subplots(
            figsize=(DOUBLE_COLUMN, 0.22 * matrix.shape[0] + 1.2)
        )
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=matrix.max() or 1.0,
            interpolation="nearest",
        )
        axis.set_yticks(np.arange(matrix.shape[0]), layer_labels)
        axis.set_xlabel("dimension")
        axis.set_ylabel("layer")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label=colorbar_label, fraction=0.03, pad=0.02)
    return figure


def token_map_grid(
    clean_maps: np.ndarray,
    triggered_maps: np.ndarray,
    dimensions: np.ndarray,
    tac_values: np.ndarray,
    title: str,
) -> matplotlib.figure.Figure:
    """The token maps of the top dimensions for 1 image, clean on the left and triggered on the right.

    clean_maps and triggered_maps are (num_dims, grid, grid), 1 map per
    dimension, at most TOKEN_MAP_GRID squared of them. Each dimension's 2 maps
    share a colour scale so the trigger's footprint reads as a change rather
    than as a rescaling.
    """
    num_dims = clean_maps.shape[0]
    if num_dims > TOKEN_MAP_GRID**2:
        raise ValueError(f"at most {TOKEN_MAP_GRID**2} dimensions fit, got {num_dims}")

    with paper_style():
        # Constrained layout is what keeps 32 titled panels, 2 half titles and
        # the figure title from writing over each other.
        figure = plt.figure(
            figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * 0.62), layout="constrained"
        )
        figure.suptitle(title)
        halves = figure.subfigures(1, 2, wspace=0.04)
        for half, maps, name in (
            (halves[0], clean_maps, "clean"),
            (halves[1], triggered_maps, "triggered"),
        ):
            axes = half.subplots(TOKEN_MAP_GRID, TOKEN_MAP_GRID)
            half.suptitle(name)
            for cell, axis in enumerate(axes.flat):
                axis.set_xticks([])
                axis.set_yticks([])
                if cell >= num_dims:
                    axis.axis("off")
                    continue
                low = min(clean_maps[cell].min(), triggered_maps[cell].min())
                high = max(clean_maps[cell].max(), triggered_maps[cell].max())
                axis.imshow(maps[cell], cmap=TOKEN_MAP_CMAP, vmin=low, vmax=high)
                axis.set_title(
                    f"d{int(dimensions[cell])}, tac {tac_values[cell]:.0f}",
                    pad=2,
                    fontsize=FONT_SIZE - 2,
                )
    return figure
