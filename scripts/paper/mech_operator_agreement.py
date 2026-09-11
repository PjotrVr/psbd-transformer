"""Mechanism test B1: do different operators measure the same per-sample quantity?

Every basis placement claims to read a decision margin under perturbation. If
that is true, 2 placements that disturb a checkpoint by the same amount (its
matched-0.6 clean-validation shift ratio) should rank the same samples as
fragile, even when the position and the operator differ. This computes the
per-sample fractional PSU of every basis placement at its own matched-0.6 rate,
the Spearman correlation between every pair on the clean split and separately on
the backdoor split, and averages over the clearing cells that carry both members
of a pair.

The slow step is disk I/O: up to 18 placements times 2 splits times 65 cells of
stage-1 tensors. The baseline tensors are shared across placements within a cell
(defences.cache), so they are read once per cell, and --max-cells truncates the
panel for a quick check before the full run.

    PYTHONPATH=. python scripts/paper/mech_operator_agreement.py \
        --results-dir /lustre/home/pstika/projects/PSBD-ViT/results --paper-dir paper \
        --max-cells 5
"""

import sys
import os

sys.path.insert(0, os.getcwd())

import itertools
import statistics

import scripts.paper._style  # noqa: E402,F401  the shared figure style
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from cli.compare_detectors import psbd_rate
from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)
from defences.decision import RECOMMENDED_PLACEMENT
from defences.scores import psu_ratio_from_cache
from scripts.paper._common import (
    build_parser,
    clearing_cells,
    figure_sidecar,
    load_coverage,
    load_declaration,
    load_psbd_metrics,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/mech_operator_agreement.py"
SPLITS = ("clean", "backdoor")
GAIN_SCALE_PLACEMENT = "mlp_norm_out_gain_scale"
GAUSSIAN_PLACEMENT = "before_attention_norm_gaussian"
TOKEN_MASK_PLACEMENT = "before_attention_norm_token_mask"
INPUT_SIDE_FAMILY = "input_side"
RESIDUAL_FAMILY = "residual_adjacent"


def build_parser_with_max_cells():
    parser = build_parser(__doc__)
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="truncate the clearing-cell panel for a quick check",
    )
    return parser


def load_cell_psu_by_placement(
    results_dir: str, folder: str, placement_ids: list[str], report: dict
) -> dict[str, dict[str, np.ndarray]]:
    """Every placement's per-sample fractional PSU at its matched-0.6 rate, this cell.

    Returns {placement_id: {"clean": (n_clean,), "backdoor": (n_backdoor,)}} for
    the placements whose matched rate and stage-1 tensors are both on disk. The
    baseline is loaded once per split, since it does not depend on the placement.
    """
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    baseline = {}
    for split in SPLITS:
        path = baseline_path(psbd_dir, split)
        if not os.path.exists(path):
            return {}
        baseline[split] = load_baseline(path)

    psu_by_placement: dict[str, dict[str, np.ndarray]] = {}
    for placement_id in placement_ids:
        block = report.get("placements", {}).get(placement_id)
        if block is None:
            continue
        rate = psbd_rate(block, "matched")
        if rate is None:
            continue

        per_split_psu = {}
        complete = True
        for split in SPLITS:
            pass_path = dropout_pass_path(psbd_dir, placement_id, rate, split)
            if not os.path.exists(pass_path):
                complete = False
                break
            per_pass_probs, _ = load_dropout_pass_probs(pass_path)
            baseline_probs, baseline_labels, _ = baseline[split]
            psu = psu_ratio_from_cache(baseline_probs, baseline_labels, per_pass_probs)
            per_split_psu[split] = psu.float().numpy()
        if complete:
            psu_by_placement[placement_id] = per_split_psu

    return psu_by_placement


def accumulate_pairwise_agreement(
    psu_by_placement: dict[str, dict[str, np.ndarray]],
    accumulator: dict[tuple[str, str], dict[str, list[float]]],
) -> None:
    """Every present pair's Spearman rho on this cell, added into the running lists."""
    present = sorted(psu_by_placement)
    for placement_a, placement_b in itertools.combinations(present, 2):
        key = (placement_a, placement_b)
        entry = accumulator.setdefault(key, {"clean": [], "backdoor": []})
        for split in SPLITS:
            correlation, _ = spearmanr(
                psu_by_placement[placement_a][split],
                psu_by_placement[placement_b][split],
            )
            # A constant PSU vector gives NaN, which would poison every mean.
            if correlation == correlation:
                entry[split].append(float(correlation))


def mean_agreement_matrix(
    accumulator: dict[tuple[str, str], dict[str, list[float]]],
    placement_ids: list[str],
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """(mean, n) matrices over placement_ids, symmetric, diagonal at 1.0 / full n."""
    size = len(placement_ids)
    mean = np.full((size, size), np.nan)
    count = np.zeros((size, size), dtype=int)
    index_of = {placement_id: i for i, placement_id in enumerate(placement_ids)}

    for (placement_a, placement_b), by_split in accumulator.items():
        values = by_split[split]
        if not values:
            continue
        i, j = index_of[placement_a], index_of[placement_b]
        mean[i, j] = mean[j, i] = statistics.mean(values)
        count[i, j] = count[j, i] = len(values)

    for i in range(size):
        mean[i, i] = 1.0
    return mean, count


def family_pair_agreement(
    accumulator: dict[tuple[str, str], dict[str, list[float]]],
    family_of: dict[str, str],
    split: str,
    family_a: str,
    family_b: str,
) -> tuple[float | None, int]:
    """Mean Spearman rho and the number of placement pairs between 2 families.

    The same family twice gives the within-family agreement. 2 different families
    give every pair with 1 member in each, the input-side against
    residual-adjacent axis H20 turns on. A set comparison covers both cases, since
    {family, family} collapses to 1 element.
    """
    values = []
    for (placement_a, placement_b), by_split in accumulator.items():
        families = {family_of.get(placement_a), family_of.get(placement_b)}
        if families == {family_a, family_b} and by_split[split]:
            values.append(statistics.mean(by_split[split]))
    if not values:
        return None, 0
    mean = statistics.mean(values)
    return mean, len(values)


def rho_text(value: float | None) -> str:
    """A correlation to 3 places, a dash when it is None or NaN."""
    if value is None or value != value:
        return "--"
    text = f"{value:.3f}"
    return text


def write_heatmap(
    args, placement_ids: list[str], mean_clean, mean_backdoor, n_clean, n_backdoor
) -> None:
    """18x18 grid: clean agreement above the diagonal, backdoor agreement below it."""
    size = len(placement_ids)
    display = np.full((size, size), np.nan)
    for i in range(size):
        for j in range(size):
            if i < j:
                display[i, j] = mean_clean[i, j]
            elif i > j:
                display[i, j] = mean_backdoor[i, j]
            else:
                display[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    image = ax.imshow(display, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(placement_ids, rotation=90, fontsize=6)
    ax.set_yticklabels(placement_ids, fontsize=6)
    for i in range(size):
        for j in range(size):
            if display[i, j] == display[i, j]:
                ax.text(
                    j,
                    i,
                    f"{display[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    color="white" if abs(display[i, j]) > 0.6 else "black",
                )
    ax.set_xlabel("clean-split agreement (upper triangle)")
    ax.set_ylabel("backdoor-split agreement (lower triangle)")
    fig.colorbar(image, ax=ax, label="mean Spearman rho over models", shrink=0.8)
    fig.tight_layout()

    path = os.path.join(args.paper_dir, "figures", "mech_operator_agreement.pdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

    figure_sidecar(
        path=path.replace(".pdf", ".json"),
        generator=GENERATOR,
        inputs=[
            f"{args.results_dir}/<folder>/psbd/<placement>/rate_*_{{clean,backdoor}}.pt"
        ],
        plotted={
            "placement_ids": placement_ids,
            "mean_clean": mean_clean.tolist(),
            "mean_backdoor": mean_backdoor.tolist(),
            "n_clean": n_clean.tolist(),
            "n_backdoor": n_backdoor.tolist(),
        },
    )


def write_agreement_table(
    args, placement_ids, family_of, mean_clean, mean_backdoor, n_clean, n_backdoor
) -> None:
    """Each placement's agreement with the recommended placement, plus family blocks."""
    index_of = {placement_id: i for i, placement_id in enumerate(placement_ids)}
    recommended_index = index_of[RECOMMENDED_PLACEMENT]

    header = [
        "placement",
        "family",
        "rho vs token mask, attention input (clean)",
        "n",
        "rho vs token mask, attention input (backdoor)",
        "n",
    ]
    rows = []
    for placement_id in placement_ids:
        if placement_id == RECOMMENDED_PLACEMENT:
            continue
        i = index_of[placement_id]
        rows.append(
            [
                placement_id,
                family_of.get(placement_id, "--"),
                rho_text(mean_clean[recommended_index, i]),
                str(n_clean[recommended_index, i]),
                rho_text(mean_backdoor[recommended_index, i]),
                str(n_backdoor[recommended_index, i]),
            ]
        )

    path = os.path.join(args.paper_dir, "tables", "mech_operator_agreement.tex")
    write_table(
        path=path,
        generator=GENERATOR,
        inputs=[
            f"{args.results_dir}/<folder>/psbd/<placement>/rate_*_{{clean,backdoor}}.pt"
        ],
        caption=(
            "B1: mean Spearman correlation, over the models whose attack succeeded, "
            "between each basis placement's per-sample fractional PSU and the "
            f"token\\_mask placement's at the attention input ({RECOMMENDED_PLACEMENT}), "
            "at each placement's own matched-0.6 rate. n is the number of models "
            "carrying both placements at a usable rate."
        ),
        label="tab:mech-operator-agreement",
        header=header,
        rows=rows,
    )


def write_family_table(args, accumulator, family_of) -> None:
    """Within-input-side, within-residual-adjacent and across-family agreement."""
    header = ["comparison", "clean rho", "n pairs", "backdoor rho", "n pairs"]
    rows = []
    for label, family_a, family_b in (
        ("within input_side", INPUT_SIDE_FAMILY, INPUT_SIDE_FAMILY),
        ("within residual_adjacent", RESIDUAL_FAMILY, RESIDUAL_FAMILY),
        (
            "across families (input_side vs residual_adjacent)",
            INPUT_SIDE_FAMILY,
            RESIDUAL_FAMILY,
        ),
    ):
        clean_mean, clean_n = family_pair_agreement(
            accumulator, family_of, "clean", family_a, family_b
        )
        backdoor_mean, backdoor_n = family_pair_agreement(
            accumulator, family_of, "backdoor", family_a, family_b
        )
        rows.append(
            [
                label,
                rho_text(clean_mean),
                str(clean_n),
                rho_text(backdoor_mean),
                str(backdoor_n),
            ]
        )

    path = os.path.join(args.paper_dir, "tables", "mech_operator_agreement_family.tex")
    write_table(
        path=path,
        generator=GENERATOR,
        inputs=[
            f"{args.results_dir}/<folder>/psbd/<placement>/rate_*_{{clean,backdoor}}.pt"
        ],
        caption=(
            "B1: mean per-sample PSU agreement (Spearman rho, averaged over placement "
            "pairs and over models) within and across the input-side and "
            "residual-adjacent families, the axis H20 splits on."
        ),
        label="tab:mech-operator-agreement-family",
        header=header,
        rows=rows,
    )


def write_agreement_macros(
    args, accumulator, family_of, placement_ids, mean_clean, n_clean
) -> None:
    within_input, within_input_n = family_pair_agreement(
        accumulator, family_of, "clean", INPUT_SIDE_FAMILY, INPUT_SIDE_FAMILY
    )
    across, across_n = family_pair_agreement(
        accumulator, family_of, "clean", INPUT_SIDE_FAMILY, RESIDUAL_FAMILY
    )

    index_of = {placement_id: i for i, placement_id in enumerate(placement_ids)}
    recommended_index = index_of[RECOMMENDED_PLACEMENT]
    gain_scale_value = mean_clean[recommended_index, index_of[GAIN_SCALE_PLACEMENT]]
    gain_scale_n = n_clean[recommended_index, index_of[GAIN_SCALE_PLACEMENT]]
    gaussian_token_value = mean_clean[
        index_of[GAUSSIAN_PLACEMENT], index_of[TOKEN_MASK_PLACEMENT]
    ]
    gaussian_token_n = n_clean[
        index_of[GAUSSIAN_PLACEMENT], index_of[TOKEN_MASK_PLACEMENT]
    ]

    macros = {
        "agreement_within_input_side_clean": (
            rho_text(within_input),
            f"B1: mean clean-split PSU Spearman rho within the input-side family, {within_input_n} placement pairs",
        ),
        "agreement_across_families_clean": (
            rho_text(across),
            f"B1: mean clean-split PSU Spearman rho between input-side and residual-adjacent placements, {across_n} placement pairs",
        ),
        "agreement_gain_scale_vs_recommended_clean": (
            rho_text(gain_scale_value),
            f"B1: clean-split PSU Spearman rho, {GAIN_SCALE_PLACEMENT} vs the recommended placement, n={int(gain_scale_n)} cells",
        ),
        "agreement_gaussian_vs_token_mask_clean": (
            rho_text(gaussian_token_value),
            f"B1: clean-split PSU Spearman rho, {GAUSSIAN_PLACEMENT} vs {TOKEN_MASK_PLACEMENT}, same position different operator, n={int(gaussian_token_n)} cells",
        ),
    }
    write_macros(
        sidecar_path=os.path.join(
            args.paper_dir, "tables", "mech_operator_agreement.macros.json"
        ),
        generator=GENERATOR,
        inputs=[
            f"{args.results_dir}/<folder>/psbd/<placement>/rate_*_{{clean,backdoor}}.pt"
        ],
        macros=macros,
    )


def main() -> None:
    args = build_parser_with_max_cells().parse_args()
    coverage = load_coverage(args.results_dir)
    declaration = load_declaration(args.declaration)
    cells = clearing_cells(coverage)
    if args.max_cells is not None:
        cells = cells[: args.max_cells]

    placement_ids = [entry["id"] for entry in declaration["basis"]]
    family_of = {entry["id"]: entry["family"] for entry in declaration["basis"]}

    accumulator: dict[tuple[str, str], dict[str, list[float]]] = {}
    for index, cell in enumerate(cells):
        report = load_psbd_metrics(args.results_dir, cell["folder_name"])
        if report is None:
            continue
        psu_by_placement = load_cell_psu_by_placement(
            args.results_dir, cell["folder_name"], placement_ids, report
        )
        accumulate_pairwise_agreement(psu_by_placement, accumulator)
        print(
            f"[{index + 1}/{len(cells)}] {cell['folder_name']}: "
            f"{len(psu_by_placement)}/{len(placement_ids)} placements loaded"
        )

    mean_clean, n_clean = mean_agreement_matrix(accumulator, placement_ids, "clean")
    mean_backdoor, n_backdoor = mean_agreement_matrix(
        accumulator, placement_ids, "backdoor"
    )

    write_heatmap(args, placement_ids, mean_clean, mean_backdoor, n_clean, n_backdoor)
    write_agreement_table(
        args, placement_ids, family_of, mean_clean, mean_backdoor, n_clean, n_backdoor
    )
    write_family_table(args, accumulator, family_of)
    write_agreement_macros(
        args, accumulator, family_of, placement_ids, mean_clean, n_clean
    )


if __name__ == "__main__":
    main()
