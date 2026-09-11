"""The Hessian eigenvalue density on a log axis, BackdoorBench's visual_hessian figure.

1 curve per split on shared axes, so the clean and the triggered batch of the
same checkpoint sit in 1 frame. Upstream draws 1 split per figure with the
largest eigenvalue in the title. Here the top 2 eigenvalues of each split go
into its legend entry instead, since a title has room for 1 split only.
"""

import matplotlib.pyplot as plt
import numpy as np

from .style import (
    CLEAN_COLOUR,
    DOUBLE_COLUMN,
    OTHER_COLOUR,
    TRIGGERED_COLOUR,
    paper_style,
    save_figure,
)

# Upstream adds this to the density before the log axis so a 0 still draws.
DENSITY_FLOOR = 1.0e-7
# Upstream's axis pads the node range by 1 on each side.
AXIS_PAD = 1.0
SPLIT_COLOURS = {
    "clean": CLEAN_COLOUR,
    "backdoor": TRIGGERED_COLOUR,
    "mixed": OTHER_COLOUR,
}


def plot_hessian_density(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    top_eigenvalues: dict[str, list[float]],
    node_range: tuple[float, float],
    title: str,
    pdf_path: str,
) -> None:
    """Write the density curves as PDF and PNG at pdf_path.

    curves maps a split name to (density, grid), both (num_bins,), as
    analysis.curvature.density_curve returns them. top_eigenvalues maps the same
    names to the eigenvalues power iteration found, largest first. node_range is
    the (min, max) of every quadrature node drawn, which fixes the x axis as
    upstream does.
    """
    with paper_style():
        figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * 0.55))
        for name, (density, grid) in curves.items():
            eigenvalues = top_eigenvalues[name]
            label = f"{name}, " + ", ".join(
                f"$\\lambda_{index + 1}$ = {value:.2f}"
                for index, value in enumerate(eigenvalues)
            )
            axis.semilogy(
                grid,
                density + DENSITY_FLOOR,
                color=SPLIT_COLOURS.get(name, OTHER_COLOUR),
                label=label,
                linewidth=1.0,
            )

        axis.set_xlim(node_range[0] - AXIS_PAD, node_range[1] + AXIS_PAD)
        axis.set_xlabel("Eigenvalue")
        axis.set_ylabel("Density (log scale)")
        axis.set_title(title)
        axis.legend(loc="upper right", frameon=False)
        figure.tight_layout()

    save_figure(figure, pdf_path)
