"""Exact grid-search replication of the Karayalcin et al. weight detector.

Karayalcin, Krcek, Chen and Picek, "Backdoor Directions in Vision Transformers"
(arXiv 2603.10806), Section 7 and supplementary sec/sup.tex, propose a data-free
detector that reads only the trained weights, no forward pass and no clean data.

    O in R^(n_class x d)      classifier head matrix, row c_i is class i's readout
    W_l                       weights of layer l that write into the residual stream
    t                         a threshold on |c_i^T W|
    n                         how many of the first layers are searched
    s_i = sum_l I[ |c_i^T W_l| > t ]      per-class score, l = 1..n
    Z = (s_top - s_second) / max(std(S \\ {s_top}), t)     S = {s_0, ..., s_{n_class}}

The model is flagged backdoored when Z > 3, and the top-scoring class is read off as
the attacker's target. The paper searches a grid of (n, t) rather than committing to
1 pair, because a single arbitrary threshold either detects nothing or detects
everything depending where it lands relative to the model's own weight scale (their
footnote on the Z floor: "these parameters/scoring functions are rather arbitrary").

Deviations from an earlier transplant (experiments/backdoor_neurons/measure.py,
analysis.lipschitz.head_weight_alignment): that version (a) used only the attention
output projection, dropping the MLP output projection the paper's own orthogonalization
step treats as an equal write to the residual stream, (b) folded the final LayerNorm
gain into c_i, a normalization the paper never states and (c) floored the Z
denominator at a fixed 1 count rather than at t itself. All 3 are reverted here to
match the quoted formulas exactly. Point (c) is not a cosmetic fix: since t sits far
below 1, flooring at t rather than at 1 makes Z blow up whenever any class other than
the top class ever exceeds the threshold an identical number of times (std = 0), which is
common at the small end of the threshold grid. Whether that inflates the detector's
apparent hit rate is exactly what this script checks.

Grid: n in 1..11 and t in {0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50},
read directly off the axes of the paper's own grid-search figure (supplementary,
`cifar100_vit_b_16_grid_search_all_ratios_subplots.png`, also the main-text figure),
not the fallback range given when a figure cannot be read.

CPU only, a few seconds per checkpoint (weights only, no data loading, no forward pass).
"""

import json
import os

import torch

from data.splits import read_checkpoint_metadata
from experiments._paths import experiment_result_path
from models.backbones import load_checkpoint, network_core

CHECKPOINTS_DIR = "checkpoints"
DEVICE = torch.device("cpu")

CHECKPOINTS = [
    "vit_cifar100_wanet_0_1",
    "vit_cifar100_wanet_0_05",
    "vit_cifar100_bpp_0_1",
    "vit_cifar100_bpp_0_05",
    "vit_cifar100_bpp_0_01",
    "vit_cifar100_badnet_a2o_0_1",
    "vit_cifar100_blend_0_1",
    "vit_cifar100_lf_0_1",
    "vit_cifar100_tact_0_1",
    "vit_cifar10_wanet_0_1",
    "vit_cifar10_bpp_0_1",
    "vit_cifar10_badnet_a2o_0_1",
    "vit_tiny_wanet_0_1",
    "vit_tiny_bpp_0_1",
    "vit_tiny_badnet_a2o_0_1",
    "vit_cifar100_benign",
    "vit_cifar10_benign",
    "vit_tiny_benign",
]

# Read from the paper's own grid-search figures (num layers on the y axis, threshold
# on the x axis), not the 1-12/quantile fallback the task allows when a figure cannot
# be read.
LAYER_GRID = list(range(1, 12))
THRESHOLD_GRID = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

Z_OUTLIER_CUTOFF = 3.0


def main() -> None:
    checkpoint_reports = {}
    for folder in CHECKPOINTS:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, folder, "attack_result.pt")
        if not os.path.exists(checkpoint_path):
            print(f"{folder}: SKIP, no attack_result.pt")
            continue

        metadata = read_checkpoint_metadata(checkpoint_path)
        model = load_checkpoint("vit", checkpoint_path, DEVICE)
        core = network_core(model)

        alignments = per_layer_alignments(core, max(LAYER_GRID))
        scores = cumulative_scores(alignments, THRESHOLD_GRID)  # (n, t, num_classes)

        is_benign = metadata["attack"] == "benign"
        true_target = None if is_benign else metadata["target_label"]
        cells = grid_cells(scores, LAYER_GRID, THRESHOLD_GRID, true_target)

        checkpoint_reports[folder] = {
            "attack": metadata["attack"],
            "dataset": metadata["dataset"],
            "poison_rate": metadata.get("poison_rate"),
            "is_benign": is_benign,
            "true_target": true_target,
            "cells": cells,
            **grid_summary(cells, is_benign),
        }
        print(
            f"{folder:32} hit={checkpoint_reports[folder]['hit_share']:.2f} "
            f"false_alarm={checkpoint_reports[folder]['false_alarm_share']:.2f} "
            f"best_z={checkpoint_reports[folder]['best_z']:.2f}",
            flush=True,
        )

    summary = {
        "rule": (
            "s_i = sum_l I[|c_i^T W_l| > t], l = 1..n; "
            "Z = (s_top - s_second) / max(std(S \\ {s_top}), t); flag when Z > 3, "
            "name the top class the target"
        ),
        "grid": {"n": LAYER_GRID, "t": THRESHOLD_GRID},
        "checkpoints": checkpoint_reports,
    }
    output_path = experiment_result_path("weight_detector", "summary.json")
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nwrote {output_path}")


def per_layer_alignments(core: torch.nn.Module, num_layers: int) -> list[torch.Tensor]:
    """abs(c_i^T W_l) for l = 1..num_layers, each (num_classes, dim + mlp_dim).

    c_i is the raw classifier head row, unnormalized: the paper states no
    normalization of c_i or of W's columns anywhere in its definition of s_i, so none
    is applied here. W_l is the concatenation of the 2 matrices the paper's own
    orthogonalization step (same section, weight-based intervention) treats as the
    layer's writes into the residual stream: the attention output projection and the
    MLP output projection.
    """
    class_directions = core.heads.head.weight.detach().float()  # (num_classes, dim)

    layers = list(core.encoder.layers)[:num_layers]
    alignments = []
    for block in layers:
        attention_out_weight = block.self_attention.out_proj.weight.detach().float()
        mlp_out_weight = block.mlp[3].weight.detach().float()

        attention_alignment = class_directions @ attention_out_weight
        mlp_alignment = class_directions @ mlp_out_weight
        # (num_classes, dim) cat (num_classes, mlp_dim) gives (num_classes, dim + mlp_dim)
        layer_alignment = torch.cat([attention_alignment, mlp_alignment], dim=1).abs()
        alignments.append(layer_alignment)

    return alignments


def cumulative_scores(
    alignments: list[torch.Tensor], thresholds: list[float]
) -> torch.Tensor:
    """s_i(n, t) for every (n, t) in the grid, (num_layers, num_thresholds, num_classes).

    Per layer, per threshold: how many entries of that layer's alignment vector
    exceed t, per class. Cumulatively summed over layers so entry [n-1] already holds
    the score at layer count n, matching s_i = sum_{l=1}^{n} of the per-layer count.
    """
    threshold_tensor = torch.tensor(thresholds).view(-1, 1, 1)  # (num_thresholds,1,1)

    per_layer_counts = []
    for layer_alignment in alignments:
        # (1, num_classes, dim) compared against (num_thresholds, 1, 1) broadcasts to
        # (num_thresholds, num_classes, dim)
        exceeds = layer_alignment.unsqueeze(0) > threshold_tensor
        counts = exceeds.sum(dim=2).float()  # (num_thresholds, num_classes)
        per_layer_counts.append(counts)

    stacked = torch.stack(
        per_layer_counts, dim=0
    )  # (num_layers, num_thresholds, num_classes)
    scores = stacked.cumsum(dim=0)
    return scores


def grid_cells(
    scores: torch.Tensor,
    layer_grid: list[int],
    threshold_grid: list[float],
    true_target: int | None,
) -> list[dict]:
    """1 record per (n, t) grid cell: Z, the named class and whether it is a hit."""
    cells = []
    for layer_index, n in enumerate(layer_grid):
        for threshold_index, t in enumerate(threshold_grid):
            class_scores = scores[layer_index, threshold_index]  # (num_classes,)
            z, top_class = z_score_and_top_class(class_scores, t)
            flagged = z > Z_OUTLIER_CUTOFF
            hit = (
                None if true_target is None else (flagged and top_class == true_target)
            )
            cells.append(
                {
                    "n": n,
                    "t": t,
                    "z": z,
                    "top_class": top_class,
                    "flagged": flagged,
                    "hit": hit,
                }
            )
    return cells


def z_score_and_top_class(class_scores: torch.Tensor, t: float) -> tuple[float, int]:
    """The paper's Z rule for 1 (n, t) grid cell.

        Z = (s_top - s_second) / max(std(S \\ {s_top}), t)

    The floor is t itself, on the same scale as the alignment magnitudes, not a
    count-scale constant: the paper states "we take the maximum of this standard
    deviation and the threshold", so the denominator can legitimately be as small as
    t whenever every non-top class ties.
    """
    ordered = class_scores.sort(descending=True).values  # (num_classes,)
    top, second = float(ordered[0]), float(ordered[1])
    spread = float(ordered[1:].std())

    z = (top - second) / max(spread, t)
    top_class = int(class_scores.argmax())
    return z, top_class


def grid_summary(cells: list[dict], is_benign: bool) -> dict:
    """Share of the grid at hit / false alarm / no detection, and the best cell.

    A benign checkpoint carries no true target, so every flagged cell there is a
    false alarm by construction and "hit" has no meaning.
    """
    total = len(cells)
    flagged = [cell for cell in cells if cell["flagged"]]

    if is_benign:
        hit_count = 0
        false_alarm_count = len(flagged)
    else:
        hit_count = sum(1 for cell in flagged if cell["hit"])
        false_alarm_count = len(flagged) - hit_count

    no_detection_count = total - len(flagged)
    best_cell = max(cells, key=lambda cell: cell["z"])

    summary = {
        "hit_share": hit_count / total,
        "false_alarm_share": false_alarm_count / total,
        "no_detection_share": no_detection_count / total,
        "best_z": best_cell["z"],
        "class_at_best_z": best_cell["top_class"],
        "n_at_best_z": best_cell["n"],
        "t_at_best_z": best_cell["t"],
    }
    return summary


if __name__ == "__main__":
    main()
