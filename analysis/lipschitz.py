"""Lipschitz-based, data-free analysis of ViT weights (the LIPS tool).

Channel Lipschitzness (Zheng et al., 2022) flags backdoor channels in ConvNets by
an upper bound on each channel's Lipschitz constant, computed from weights alone,
which correlates with the trigger-activated change on that channel.

The direct transfer to ViT fails at attention. Kim et al. (2021) proved that
standard dot-product self-attention is not Lipschitz on an unbounded domain, so
there is no clean weight-only Lipschitz bound for an attention block. What does
transfer is the following.

- The MLP block is a stack of linear maps and a 1-Lipschitz nonlinearity (GELU is
  close to 1-Lipschitz), so per-output-dimension Lipschitz constants of its final
  linear map are meaningful.
- The attention output projection and the QKV projections are linear, so their
  spectral norms are usable as sensitivity proxies even though the attention
  mixing between them is not Lipschitz.
- For attention as a whole, an empirical local Lipschitz constant on real, bounded
  inputs (a Jacobian or finite-difference sensitivity) is the honest substitute,
  and is left as a data-driven follow-up.

For a fully data-free ViT-native detector, the weight-alignment scheme in Section 8
of Karayalcin et al. is the better fit than raw Lipschitz numbers.
"""

import torch
import torch.nn as nn

from models.backbones import network_core

# The alignment counts are integers, so 1 is the smallest difference they can
# express and the smallest spread that can honestly divide a difference.
MINIMUM_ALIGNMENT_SPREAD = 1.0


def spectral_norm(weight: torch.Tensor) -> float:
    """Largest singular value, the operator 2-norm of a linear map."""
    largest = torch.linalg.svdvals(weight.float())[0].item()
    return largest


def linear_channel_lipschitz(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-channel Lipschitz constant of a linear map, (num_outputs,).

    For output dimension k the map is out_k = row_k dot input, whose Lipschitz
    constant is the row's 2-norm.
    """
    per_channel = weight.detach().float().norm(dim=1).cpu()
    return per_channel


def _last_linear(module: nn.Module) -> nn.Linear:
    """The final nn.Linear inside a block, which is the one writing to the stream."""
    linears = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    if not linears:
        raise ValueError("No linear layer found in the MLP block")
    return linears[-1]


def mlp_output_channel_lipschitz(model: nn.Module) -> dict[int, torch.Tensor]:
    """Per-block, per-residual-dimension Lipschitz of the MLP write into the stream.

    Reads the second linear layer of each block's MLP, whose rows write into the
    residual stream. High-value dimensions here are candidate backdoor channels in
    the same spirit as CLP, and can be compared against the TAC computed from
    paired data. Blocks are keyed 1 to 12, matching the layer indexing in
    analysis.features.
    """
    result: dict[int, torch.Tensor] = {}
    for index, block in enumerate(network_core(model).encoder.layers, start=1):
        output_linear = _last_linear(block.mlp)
        result[index] = linear_channel_lipschitz(output_linear.weight)

    return result


def head_weight_alignment(
    model: nn.Module, num_layers: int, threshold_quantile: float
) -> torch.Tensor:
    """The Karayalcin data-free detector, adapted here for reference, (num_classes,).

    For each class row of the classifier head, count how strongly it aligns with
    the output-projection weights of the first num_layers blocks. A backdoor target
    class tends to align with an early-layer shortcut, so its score is an outlier
    among classes.

        original per-class score
            s_i = sum over layers of count( abs(c_i^T W)_l > threshold )
        simplified form
            score_i = number of large-magnitude alignments between class direction
                      c_i and the early output-projection weights

    The paper's absolute threshold does not transfer to ViT. Every alignment
    magnitude on this project's checkpoints sits far below any ConvNet-scale
    constant, so such a constant makes every count 0 and the detector returns a
    tied vector that decides nothing. The threshold is therefore a quantile of the
    model's own alignment distribution, which keeps the detector data-free while
    making it scale-free. The quantile fixes the total count across all classes,
    so only its distribution over classes carries the signal, which is what the
    outlier rule reads.
    """
    core = network_core(model)
    class_directions = core.heads.head.weight.detach()  # (num_classes, dim)

    # The head reads encoder.ln(x)[:, 0], so the true readout direction is the class
    # row scaled by the final LayerNorm gain, not the raw row. Folding it in makes
    # the alignment magnitudes and the threshold mean what they should.
    class_directions = class_directions * core.encoder.ln.weight.unsqueeze(0)

    projection_weights = [
        block.self_attention.out_proj.weight.detach().float()  # (dim, dim)
        for block in list(core.encoder.layers)[:num_layers]
    ]

    alignment = torch.cat(
        [(class_directions.float() @ weight).abs() for weight in projection_weights],
        dim=1,
    ).cpu()  # (num_classes, num_layers * dim)
    threshold = alignment.flatten().quantile(threshold_quantile)

    counts = (alignment > threshold).sum(dim=1).float()  # (num_classes,)
    return counts


def alignment_outlier_score(
    scores: torch.Tensor, spread_floor: float = MINIMUM_ALIGNMENT_SPREAD
) -> tuple[float, int]:
    """The paper's Z rule on top of the per-class alignment counts.

        original form
            Z = (s_top - s_second) / max( std(S without s_top), t )
        simplified form
            z = how far the top class stands above the runner-up, in units of the
                spread of every other class

    Returns (Z, top_class). A backdoored model is flagged when Z > 3, with the top
    class read as the attacker's target. head_weight_alignment on its own returns
    a vector of counts and makes no decision.

    The floor is in count units and defaults to 1, the smallest difference the
    counts can express. When every class but 1 scores identically the standard
    deviation collapses and Z would otherwise diverge on a 1-count difference.
    """
    ordered = scores.sort(descending=True).values  # (num_classes,)
    top, second = float(ordered[0]), float(ordered[1])
    spread = float(ordered[1:].std())

    z = (top - second) / max(spread, spread_floor)
    top_class = int(scores.argmax())
    return z, top_class
