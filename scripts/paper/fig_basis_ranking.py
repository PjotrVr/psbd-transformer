"""The 18 basis placements ranked by mean AUROC, adaptive and matched rule side by side.

Reads the same per-cell numbers app_basis.py tabulates (through its
ranking_rows) and draws them as 1 horizontal bar chart, dark for the adaptive
rule and light for the matched rule, so the figure and the appendix table can
never disagree.

    PYTHONPATH=. python scripts/paper/fig_basis_ranking.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper._common import (  # noqa: E402
    build_parser,
    clearing_cells,
    figure_sidecar,
    load_coverage,
    load_declaration,
)
from scripts.paper.app_basis import ranking_rows  # noqa: E402

GENERATOR = "scripts/paper/fig_basis_ranking.py"
ADAPTIVE_COLOUR = "#0072B2"
MATCHED_COLOUR = "#9ecae1"
RECOMMENDED = "before_attention_norm_token_mask"
PUBLISHED = "post_residual"


def readable(placement: str) -> str:
    """The placement id with underscores as spaces, the recommended and published ones marked."""
    label = placement.replace("_", " ")
    if placement == RECOMMENDED:
        label += "  (recommended)"
    if placement == PUBLISHED:
        label += "  (published)"
    return label


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage = load_coverage(args.results_dir)
    declaration = load_declaration(args.declaration)
    cells = clearing_cells(coverage)
    _rows, ordered = ranking_rows(args.results_dir, cells, declaration["basis"])

    placements = list(ordered)[::-1]
    adaptive = [ordered[p]["adaptive_mean"] or 0.0 for p in placements]
    matched = [ordered[p]["matched_mean"] or 0.0 for p in placements]
    positions = range(len(placements))

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.barh(
        [y + 0.2 for y in positions],
        adaptive,
        height=0.38,
        color=ADAPTIVE_COLOUR,
        label="adaptive rule (0.8)",
    )
    ax.barh(
        [y - 0.2 for y in positions],
        matched,
        height=0.38,
        color=MATCHED_COLOUR,
        label="matched rule (0.6)",
    )
    ax.set_yticks(list(positions))
    ax.set_yticklabels([readable(p) for p in placements], fontsize=7)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("mean AUROC over 65 cells")
    ax.axvline(0.5, color="black", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    path = os.path.join(args.paper_dir, "figures", "basis_ranking.pdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    figure_sidecar(
        path.replace(".pdf", ".json"),
        GENERATOR,
        [os.path.join(args.results_dir, "coverage", "coverage.json"), args.declaration],
        {"placements": placements, "adaptive_mean": adaptive, "matched_mean": matched},
    )
    print(f"basis ranking: {len(placements)} placements, top {placements[-1]}")


if __name__ == "__main__":
    main()
