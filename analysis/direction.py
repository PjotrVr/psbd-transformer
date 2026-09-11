"""The paired-difference toolkit: backdoor direction, TAC, steering, orthogonalization.

All of these read the same object, the difference between a backdoor feature and
its clean counterpart at some layer. The backdoor direction is the mean of that
difference as a vector. TAC is its per-dimension magnitude. They are 2 views of
where and how strongly the trigger is written into the residual stream.

Inputs are the index-aligned feature tensors returned by
analysis.features.extract_layer_features on the clean and backdoor loaders,
each shaped (num_samples, dim) for a fixed layer.
"""

import torch

from .features import as_token_sequence

# Guards the normalization of a direction that came out at 0, which happens on a
# benign model where the trigger moves nothing.
DIRECTION_NORM_FLOOR = 1e-8


def _unit(direction: torch.Tensor) -> torch.Tensor:
    """The direction rescaled to length 1, keeping its (dim,) shape."""
    unit_direction = direction / direction.norm().clamp_min(DIRECTION_NORM_FLOOR)
    return unit_direction


def backdoor_direction(
    clean_features: torch.Tensor, backdoor_features: torch.Tensor
) -> torch.Tensor:
    """Mean paired difference, (dim,), the trigger's representation at this layer.

    original form
        r_l = (1 / |pairs|) * sum over pairs of
              (backdoor_activation - clean_activation)
    simplified form
        direction = mean over samples of (backdoor_feature - clean_feature)
    """
    assert clean_features.shape == backdoor_features.shape, (
        "the paired difference is only defined on index-aligned features, "
        f"got {tuple(clean_features.shape)} and {tuple(backdoor_features.shape)}"
    )

    direction = (backdoor_features - clean_features).mean(dim=0)  # (dim,)
    return direction


def trigger_activated_change(
    clean_features: torch.Tensor, backdoor_features: torch.Tensor
) -> torch.Tensor:
    """Per-dimension trigger sensitivity, (dim,), the ViT residual-stream form of TAC.

        original form (per channel k, averaged over data)
            TAC_k = (1 / |data|) * sum over x of
                    l2_norm(feature_k(x) - feature_k(x_trigger))
        simplified form (CLS or mean token, 1 scalar per residual dimension)
            tac_k = mean over samples of abs(backdoor_feature_k - clean_feature_k)

    A high value means that dimension moves a lot when the trigger is applied, so
    it is a candidate backdoor dimension.
    """
    assert clean_features.shape == backdoor_features.shape, (
        "TAC is only defined on index-aligned features, "
        f"got {tuple(clean_features.shape)} and {tuple(backdoor_features.shape)}"
    )

    tac = (backdoor_features - clean_features).abs().mean(dim=0)  # (dim,)
    return tac


def project_onto_direction(
    features: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Signed length of each feature along the unit direction, (num_samples,).

    Use this to measure how much backdoor signal a representation still carries,
    for example before and after dropout at each placement.
    """
    projections = features @ _unit(direction).to(features.dtype)
    return projections


def outlier_dimensions(values: torch.Tensor, sensitivity: float = 3.0) -> torch.Tensor:
    """Indices whose value exceeds mean plus sensitivity times std.

    This is the CLP outlier rule reused for TAC or Lipschitz values. A larger
    sensitivity flags fewer, more extreme dimensions.
    """
    threshold = values.mean() + sensitivity * values.std()

    indices = torch.nonzero(values > threshold, as_tuple=False).squeeze(1)
    return indices


def make_steering_hook(direction: torch.Tensor, scale: float):
    """Forward hook that adds scale times the direction to a block's output.

    Positive scale on clean inputs should activate the backdoor. Negative scale on
    backdoor inputs should recover the original class. Register it on the encoder
    block whose output you want to steer.
    """
    shift = scale * direction  # (dim,)

    def hook(_module, _inputs, output):
        steered = output + shift.to(output.dtype).to(output.device)
        return steered

    return hook


def orthogonalize_weight(weight: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the direction from a residual-stream write matrix.

        original form
            W_new = W - r_hat r_hat^T W,  with r_hat the unit direction
        simplified form
            new_weight = weight minus the component of every column along the
                         direction

    weight has shape (residual_dim, input_dim), since the direction lives in the
    residual stream that the matrix writes into.
    """
    unit_direction = _unit(direction).to(weight.dtype)  # (residual_dim,)

    projected_out = weight - torch.outer(unit_direction, unit_direction) @ weight
    return projected_out


def trigger_activation_change(
    clean_tokens: torch.Tensor, backdoor_tokens: torch.Tensor, norm: str = "l1_sum"
) -> torch.Tensor:
    """TAC per residual dimension from token sequences, (dim,), in 2 conventions.

    Zheng et al. (ECCV 2022, Channel Lipschitzness Pruning) define the trigger
    activated change of channel k over a data set D as the per-sample L2 norm of
    the feature map difference, averaged over samples.

        original form (norm="l2")
            TAC_k = (1 / |D|) * sum over x in D of
                    l2_norm over positions p of ( f_{k,p}(x + delta) - f_{k,p}(x) )

    BackdoorBench's visual_tac.py (lines 143 to 150) computes something else: it
    averages the absolute difference over samples first and then sums the
    result over the spatial positions of the channel. That is the L1 form with
    the mean and the sum in the other order, and it is what the upstream TAC
    heatmap shows.

        BackdoorBench form (norm="l1_sum")
            TAC_k = sum over positions p of
                    (1 / |D|) * sum over x in D of | f_{k,p}(x + delta) - f_{k,p}(x) |

    symbol table
        D          the paired clean images
        x          a clean image, x + delta the same image with the trigger
        f_{k,p}    the activation of channel k at position p
        k          a residual dimension here, a ConvNet channel upstream
        p          a token here, a feature-map pixel upstream

    Inputs are index-aligned (batch, tokens, dim) tensors, or Swin's
    (batch, height, width, channels), which is viewed as a token sequence. The
    pooled trigger_activated_change above is the class-token or token-mean
    special case with a single position.
    """
    clean = as_token_sequence(clean_tokens).float()  # (batch, tokens, dim)
    backdoor = as_token_sequence(backdoor_tokens).float()  # (batch, tokens, dim)
    assert clean.shape == backdoor.shape, (
        "TAC is only defined on index-aligned token sequences, "
        f"got {tuple(clean.shape)} and {tuple(backdoor.shape)}"
    )

    difference = backdoor - clean  # (batch, tokens, dim)
    if norm == "l1_sum":
        tac = difference.abs().mean(dim=0).sum(dim=0)  # (dim,)
        return tac
    if norm == "l2":
        tac = difference.norm(dim=1).mean(dim=0)  # (dim,)
        return tac
    raise ValueError(f"unknown TAC norm {norm!r}, expected 'l1_sum' or 'l2'")
