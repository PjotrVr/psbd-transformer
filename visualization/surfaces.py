"""The 2-D loss surface as a 3-D plot with a contour beside it, per split.

The 3-D view is BackdoorBench's visual_landscape figure, plot_surface under the
coolwarm map with a shrunk colour bar. The contour is the loss-landscape
repository's plot_2d_contour, which upstream leaves commented out, with its
levels 0.1 to 10 in steps of 0.5 and inline labels. Rows are splits, so the
clean and the triggered surface of the same checkpoint, which share their 2
directions, sit on the same page at the same axes.
"""

import matplotlib.pyplot as plt
import numpy as np

from .style import DOUBLE_COLUMN, paper_style, save_figure

SURFACE_CMAP = "coolwarm"
CONTOUR_CMAP = "summer"
# The repository's contour levels: vmin 0.1, vmax 10, vlevel 0.5.
CONTOUR_LEVELS = np.arange(0.1, 10, 0.5)
FALLBACK_LEVEL_COUNT = 12
ROW_HEIGHT = 3.0


def contour_levels(losses: np.ndarray) -> np.ndarray:
    """The repository's levels that fall inside the surface, or an even spread.

    A surface whose loss never leaves [0, 0.6] would meet fewer than 2 of the
    fixed levels, a contour with fewer than 2 levels draws nothing, so the
    fallback spreads levels between the surface's own minimum and maximum.
    """
    low, high = float(np.nanmin(losses)), float(np.nanmax(losses))
    inside = CONTOUR_LEVELS[(CONTOUR_LEVELS > low) & (CONTOUR_LEVELS < high)]
    if len(inside) >= 2:
        return inside
    spread = np.linspace(low, high, FALLBACK_LEVEL_COUNT)[1:-1]  # (10,)
    return spread


def plot_loss_surfaces(
    surfaces: dict[str, np.ndarray],
    alphas: np.ndarray,
    betas: np.ndarray,
    title: str,
    pdf_path: str,
) -> None:
    """Write 1 row per split, the 3-D surface left and its contour right.

    surfaces maps a split name to its (len(alphas), len(betas)) loss grid, with
    entry [i, j] at (alphas[i], betas[j]).
    """
    grid_alpha, grid_beta = np.meshgrid(alphas, betas, indexing="ij")  # (A, B) each
    rows = len(surfaces)
    with paper_style():
        figure = plt.figure(figsize=(DOUBLE_COLUMN, ROW_HEIGHT * rows))
        for row, (name, losses) in enumerate(surfaces.items()):
            surface_axis = figure.add_subplot(rows, 2, 2 * row + 1, projection="3d")
            surface = surface_axis.plot_surface(
                grid_alpha,
                grid_beta,
                losses,
                cmap=SURFACE_CMAP,
                linewidth=0,
                antialiased=False,
            )
            figure.colorbar(surface, ax=surface_axis, shrink=0.5, aspect=5)
            surface_axis.set_xlabel(r"$\alpha$")
            surface_axis.set_ylabel(r"$\beta$")
            surface_axis.set_zlabel("loss")
            surface_axis.set_title(f"{name}, loss surface")

            contour_axis = figure.add_subplot(rows, 2, 2 * row + 2)
            contour = contour_axis.contour(
                grid_alpha,
                grid_beta,
                losses,
                cmap=CONTOUR_CMAP,
                levels=contour_levels(losses),
            )
            contour_axis.clabel(contour, inline=1, fontsize=6)
            contour_axis.set_xlabel(r"$\alpha$")
            contour_axis.set_ylabel(r"$\beta$")
            contour_axis.set_aspect("equal")
            contour_axis.set_title(
                f"{name}, min {np.nanmin(losses):.3f}, max {np.nanmax(losses):.3f}"
            )

        figure.suptitle(title)
        figure.tight_layout()

    save_figure(figure, pdf_path)
