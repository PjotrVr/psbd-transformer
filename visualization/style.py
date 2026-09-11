"""1 palette, 1 set of colour maps, 1 set of sizes and 1 save routine for every figure.

Categorical colour follows Okabe and Ito's colour-blind safe order for the
first 7 classes, extended by 3 hues from Tol's scheme that stay distinct from
them, so a figure with up to 10 classes reads without a legend key. The
poisoned class is always black, as BackdoorBench draws it, and any class past
the 10th is grey, since a categorical figure with more hues than that reads
as noise anyway. Each statistic family keeps 1 sequential map, blues for TAC
and oranges for Lipschitz as upstream, and attributions share 1 diverging map.
"""

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

CLASS_COLOURS = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#332288",
    "#117733",
    "#882255",
)
POISON_COLOUR = "#000000"
OTHER_COLOUR = "#999999"
CLEAN_COLOUR = "#0072B2"
TRIGGERED_COLOUR = "#D55E00"

TAC_CMAP = "Blues"
LIPSCHITZ_CMAP = "Oranges"
CONFUSION_CMAP = "Blues"
ATTRIBUTION_CMAP = "RdBu_r"
FREQUENCY_CMAP = "coolwarm"
OVERLAY_CMAP = "jet"
TOKEN_MAP_CMAP = "viridis"

SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.2
FONT_SIZE = 8
PNG_DPI = 200

RC_PARAMS = {
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
}


def paper_style() -> matplotlib.RcParams:
    """The rc context every figure builder wraps itself in."""
    context = matplotlib.rc_context(RC_PARAMS)
    return context


def class_colour(position: int) -> str:
    """The colour of the class drawn at this position in the legend order."""
    if position < len(CLASS_COLOURS):
        return CLASS_COLOURS[position]
    return OTHER_COLOUR


def colour_table(class_order: np.ndarray, num_classes: int) -> np.ndarray:
    """RGB rows for every class index plus the poisoned pseudo class, (num_classes + 1, 3).

    class_order lists the classes a figure draws, in legend order, and fixes
    which of them gets which hue. Classes outside it are grey and the row at
    index num_classes is black for the poisoned rows.
    """
    table = np.tile(matplotlib.colors.to_rgb(OTHER_COLOUR), (num_classes + 1, 1))
    for position, class_index in enumerate(class_order):
        table[int(class_index)] = matplotlib.colors.to_rgb(class_colour(position))
    table[num_classes] = matplotlib.colors.to_rgb(POISON_COLOUR)
    return table


def save_figure(figure: matplotlib.figure.Figure, pdf_path: str) -> None:
    """Write the figure as PDF at pdf_path and as PNG beside it, then close it."""
    if not pdf_path.endswith(".pdf"):
        raise ValueError(f"expected a .pdf path, got {pdf_path!r}")
    png_path = pdf_path[: -len(".pdf")] + ".png"

    with paper_style():
        figure.savefig(pdf_path)
        figure.savefig(png_path, dpi=PNG_DPI)
    plt.close(figure)
