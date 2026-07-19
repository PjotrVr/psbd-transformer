"""Lipschitz-based, data-free analysis of ViT weights (the LIPS tool).

Channel Lipschitzness (Zheng et al., 2022) flags backdoor channels in ConvNets
by an upper bound on each channel's Lipschitz constant, computed from weights
alone, which correlates with the trigger-activated change on that channel.

The direct transfer to ViT fails at attention. Kim et al. (2021) proved that
standard dot-product self-attention is not Lipschitz on an unbounded domain, so
there is no clean weight-only Lipschitz bound for an attention block. What does
transfer is the following.

- The MLP block is a stack of linear maps and a 1-Lipschitz nonlinearity (GELU
  is close to 1-Lipschitz), so per-output-dimension Lipschitz constants of its
  final linear map are meaningful.
- The attention output projection and the QKV projections are linear, so their
  spectral norms are usable as sensitivity proxies even though the attention
  mixing between them is not Lipschitz.
- For attention as a whole, an empirical local Lipschitz constant on real,
  bounded inputs (a Jacobian or finite-difference sensitivity) is the honest
  substitute, and is left as a data-driven follow-up.

For a fully data-free ViT-native detector, the weight-alignment scheme in
Section 8 of Karayalcin et al. is the better fit than raw Lipschitz numbers.
"""

import torch
import torch.nn as nn

from models import vit_core


def spectral_norm(weight: torch.Tensor) -> float:
    """Largest singular value, the operator 2-norm of a linear map."""
    return torch.linalg.svdvals(weight.float())[0].item()


def linear_channel_lipschitz(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-channel Lipschitz constant of a linear map.

    For output dimension k the map is out_k = row_k dot input, whose Lipschitz
    constant is the row's 2-norm. Returns one value per output channel.
    """
    return weight.float().norm(dim=1)


def mlp_output_channel_lipschitz(model: nn.Module) -> dict[int, torch.Tensor]:
    """Per-block, per-residual-dimension Lipschitz of the MLP write into the stream.

    Reads the second linear layer of each block's MLP, whose rows write into the
    residual stream. High-value dimensions here are candidate backdoor channels
    in the same spirit as CLP, and can be compared against the TAC computed from
    paired data.
    """
    result: dict[int, torch.Tensor] = {}
    for index, block in enumerate(vit_core(model).encoder.layers, start=1):
        output_linear = _last_linear(block.mlp)
        result[index] = linear_channel_lipschitz(output_linear.weight)
    return result


def _last_linear(module: nn.Module) -> nn.Linear:
    linears = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    if not linears:
        raise ValueError("No linear layer found in the MLP block")
    return linears[-1]


def head_weight_alignment(model: nn.Module, num_layers: int, threshold: float) -> torch.Tensor:
    """The Karayalcin data-free detector, adapted here for reference.

    For each class row of the classifier head, count how strongly it aligns with
    the output-projection weights of the first num_layers blocks. A backdoor
    target class tends to align with an early-layer shortcut, so its score is an
    outlier among classes.

    original per-class score
        s_i = sum over layers of count( abs(c_i^T W)_l > threshold )
    simplified form
        score_i = number of large-magnitude alignments between class direction c_i
                  and the early output-projection weights
    """
    core = vit_core(model)
    class_directions = core.heads.head.weight  # [num_classes, dim]

    projection_weights = []
    for block in list(core.encoder.layers)[:num_layers]:
        attention_output = block.self_attention.out_proj.weight  # [dim, dim]
        projection_weights.append(attention_output)

    scores = torch.zeros(class_directions.size(0))
    for class_index, class_direction in enumerate(class_directions):
        for weight in projection_weights:
            alignment = (class_direction.float() @ weight.float()).abs()
            scores[class_index] += (alignment > threshold).sum()
    return scores
