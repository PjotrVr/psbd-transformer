"""The 1 matplotlib style every paper figure uses, imported before pyplot draws.

A light grid with transparency so values can be read off a line chart, distinct
colorblind safe hues in a fixed order (Okabe and Ito), and sizes that stay
legible at a 2-column width.

The 2 column widths are measured from the class options this paper sets, not
guessed. main.tex asks for 10pt twocolumn and preamble.tex for margin=0.8in with
columnsep=0.28in on US letter, which leaves a 6.9in text width and a 3.31in
column. A figure authored wider than the width it is drawn into is scaled down by
LaTeX, and the scaling takes the fonts with it, so an 8pt label in an 11in figure
lands at 5pt on the page. Author at the target width instead.
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
        "savefig.dpi": 300,
        # Type 3 is matplotlib's default and the IEEE and ACM camera-ready checkers
        # reject it. 42 embeds TrueType, which also leaves the text searchable and
        # selectable in the compiled PDF.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# The width a figure may occupy, in inches, for this paper's geometry.
COLUMN_WIDTH = 3.31
TEXT_WIDTH = 6.9
