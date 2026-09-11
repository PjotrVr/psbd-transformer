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

import scripts.paper._style  # noqa: E402,F401  the shared figure style
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
RECOMMENDED = "before_attention_norm_token_mask"
PUBLISHED = "post_residual"


POSITION_WORDS = {
    "before_attention_norm": "attention input",
    "before_attention": "attention input after norm",
    "before_mlp_norm": "MLP input",
    "before_mlp": "MLP input after norm",
    "both_sublayer_inputs": "both sublayer inputs",
    "input_pixels": "input pixels",
    "before_attention_residual": "attention output before the add",
    "after_attention_residual": "stream after the attention add",
    "pre_residual": "before both residual adds",
    "post_residual": "after both residual adds",
    "mlp_neurons": "MLP neurons",
    "mlp_norm_out": "MLP norm output",
    "after_embedding": "embedding output",
}
OPERATOR_WORDS = {
    "token_mask": "token mask",
    "channel_mask": "channel mask",
    "gaussian": "noise",
    "dropout": "dropout",
    "gain_scale": "gain scale",
    "scale_up": "scale up",
}


def readable(placement: str, entry: dict) -> str:
    """The placement in words, operator first, then the site, then the block band."""
    words = f"{OPERATOR_WORDS.get(entry['operator'], entry['operator'])}, {POSITION_WORDS.get(entry['position'], entry['position'])}"
    block_range = entry.get("block_range")
    if block_range:
        words += f", blocks {block_range[0]} to {block_range[1]}"
    if placement == RECOMMENDED:
        words += "  (PSBD-TM)"
    if placement == PUBLISHED:
        words += "  (PSBD-RD)"
    return words


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage = load_coverage(args.results_dir)
    declaration = load_declaration(args.declaration)
    cells = clearing_cells(coverage)
    _rows, ordered = ranking_rows(args.results_dir, cells, declaration["basis"])

    entries = {entry["id"]: entry for entry in declaration["basis"]}
    full_count = max(ordered[p]["adaptive_n"] for p in ordered)
    complete = {p for p in ordered if ordered[p]["adaptive_n"] == full_count}
    placements = [p for p in list(ordered)[::-1] if p in complete]
    adaptive = [ordered[p]["adaptive_mean"] or 0.0 for p in placements]
    positions = range(len(placements))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(list(positions), adaptive, height=0.62, color=ADAPTIVE_COLOUR)
    ax.set_yticks(list(positions))
    ax.set_yticklabels([readable(p, entries[p]) for p in placements], fontsize=8)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("mean AUROC")
    ax.axvline(0.5, color="black", linewidth=0.6)
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
        {"placements": placements, "adaptive_mean": adaptive},
    )
    print(f"basis ranking: {len(placements)} placements, top {placements[-1]}")


if __name__ == "__main__":
    main()
