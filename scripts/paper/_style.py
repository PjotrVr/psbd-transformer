"""The 1 matplotlib style every paper figure uses, imported before pyplot draws.

A light grid with transparency so values can be read off a line chart, distinct
colour-blind safe hues in a fixed order (Okabe and Ito), and sizes that stay
legible at a 2-column width.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)
