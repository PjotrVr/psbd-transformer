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
