"""The paired-difference toolkit: backdoor direction, TAC, steering, orthogonalization.

All of these read the same object, the difference between a backdoor feature and
its clean counterpart at some layer. The backdoor direction is the mean of that
difference as a vector. TAC is its per-dimension magnitude. They are two views of
where and how strongly the trigger is written into the residual stream.

Inputs are the index-aligned feature tensors returned by
extract_layer_features on the clean and backdoor loaders, each shaped
[num_samples, dim] for a fixed layer.
"""

import torch


def _unit(direction: torch.Tensor) -> torch.Tensor:
    return direction / direction.norm().clamp_min(1e-8)


def backdoor_direction(
    clean_features: torch.Tensor, backdoor_features: torch.Tensor
) -> torch.Tensor:
    """Mean paired difference, the trigger's representation at this layer.

    original form
        r_l = (1 / |pairs|) * sum over pairs of (backdoor_activation - clean_activation)
    simplified form
        direction = mean over samples of (backdoor_feature - clean_feature)
    """
    return (backdoor_features - clean_features).mean(dim=0)


def trigger_activated_change(
    clean_features: torch.Tensor, backdoor_features: torch.Tensor
) -> torch.Tensor:
    """Per-dimension trigger sensitivity, the ViT residual-stream form of TAC.

    original form (per channel k, averaged over data)
        TAC_k = (1 / |data|) * sum over x of l2_norm(feature_k(x) - feature_k(x_trigger))
    simplified form (CLS or mean token, one scalar per residual dimension)
        tac_k = mean over samples of abs(backdoor_feature_k - clean_feature_k)

    A high value means that dimension moves a lot when the trigger is applied, so
    it is a candidate backdoor dimension.
    """
    return (backdoor_features - clean_features).abs().mean(dim=0)


def project_onto_direction(
    features: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Signed length of each feature along the unit direction, one scalar per sample.

    Use this to measure how much backdoor signal a representation still carries,
    for example before and after dropout at each placement.
    """
    return features @ _unit(direction).to(features.dtype)


def outlier_dimensions(values: torch.Tensor, sensitivity: float = 3.0) -> torch.Tensor:
    """Indices whose value exceeds mean plus sensitivity times std.

    This is the CLP outlier rule reused for TAC or Lipschitz values. A larger
    sensitivity flags fewer, more extreme dimensions.
    """
    threshold = values.mean() + sensitivity * values.std()
    return torch.nonzero(values > threshold, as_tuple=False).squeeze(1)


def make_steering_hook(direction: torch.Tensor, scale: float):
    """Forward hook that adds scale times the direction to a block's output.

    Positive scale on clean inputs should activate the backdoor. Negative scale
    on backdoor inputs should recover the original class. Register it on the
    encoder block whose output you want to steer.
    """
    shift = scale * direction

    def hook(_module, _inputs, output):
        return output + shift.to(output.dtype).to(output.device)

    return hook


def orthogonalize_weight(weight: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the direction from a residual-stream write matrix.

    original form
        W_new = W - r_hat * r_hat^T * W    with r_hat the unit direction
    simplified form
        new_weight = weight minus the component of every column along the direction

    weight has shape [residual_dim, input_dim], since the direction lives in the
    residual stream that the matrix writes into.
    """
    unit_direction = _unit(direction).to(weight.dtype)
    return weight - torch.outer(unit_direction, unit_direction) @ weight
