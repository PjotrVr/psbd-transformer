"""The 2-dimensional embedding scatter shared by t-SNE, UMAP and PCA."""

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from .style import POISON_COLOUR, SINGLE_COLUMN, class_colour, paper_style


def embedding_scatter(
    embedding: np.ndarray,
    labels: np.ndarray,
    poison_mask: np.ndarray,
    class_order: np.ndarray,
    title: str,
    axis_names: tuple[str, str],
    mark_size: float,
    alpha: float,
) -> matplotlib.figure.Figure:
    """Points coloured by class in legend order, poisoned rows black and drawn last.

    embedding is (N, 2) already scaled to the unit square, labels (N,) the
    true class, poison_mask (N,) bool. Axes run from -0.01 to 1.01 without
    ticks and the legend sits outside the square, as visual_utils.plot_embedding
    lays it out.
    """
    with paper_style():
        figure, axis = plt.subplots(figsize=(SINGLE_COLUMN * 1.3, SINGLE_COLUMN))
        for position, class_index in enumerate(class_order):
            rows = (labels == class_index) & ~poison_mask
            if not rows.any():
                continue
            axis.scatter(
                embedding[rows, 0],
                embedding[rows, 1],
                s=mark_size,
                alpha=alpha,
                color=class_colour(position),
                label=f"class {int(class_index)}",
                linewidths=0,
            )
        if poison_mask.any():
            axis.scatter(
                embedding[poison_mask, 0],
                embedding[poison_mask, 1],
                s=mark_size,
                alpha=alpha,
                color=POISON_COLOUR,
                label="poisoned",
                linewidths=0,
            )
        axis.set_xlim(-0.01, 1.01)
        axis.set_ylim(-0.01, 1.01)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel(axis_names[0])
        axis.set_ylabel(axis_names[1])
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=False,
            markerscale=3,
        )
    return figure
