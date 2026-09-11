"""The 2-D loss surface around a checkpoint along filter-normalised random directions.

Li et al., "Visualizing the Loss Landscape of Neural Nets", NeurIPS 2018,
Section 4, as the authors' loss-landscape repository computes it and as
BackdoorBench's visual_landscape.py calls that repository. 2 Gaussian directions
are drawn in parameter space, each rescaled unit by unit to the norm of the
corresponding unit of the trained weights, and the loss is evaluated on a grid
of displacements along the 2. The rescaling is what makes surfaces of different
networks comparable: without it a direction's effect depends on the scale each
layer happens to have trained to.

    original form (the paper's Eq. 3 with filter normalisation)
        f(alpha, beta) = L(theta* + alpha delta + beta eta)
        delta_{i,j} <- (delta_{i,j} / ||delta_{i,j}||) ||theta_{i,j}||

    symbols
        theta*        the trained parameters
        delta, eta    the 2 random directions, 1 tensor per parameter
        alpha, beta   the grid coordinates, [-1, 1] here
        delta_{i,j}   the j-th filter of the i-th layer of delta
        theta_{i,j}   the matching filter of theta*
        L             the mean loss over the evaluation set

The paper's filter is a convolution kernel. On a transformer the unit is a row
of a weight matrix, every Linear and the fused qkv in_proj_weight included: the
repository iterates a parameter tensor along its first axis and rescales each
slice, which on a rank-2 tensor is a row and on the rank-4 conv_proj kernel is a
filter. Rank-3 tensors, the class token and the position embedding, have a
first axis of length 1, so the whole tensor is rescaled as 1 unit. Parameters of
rank 1 or 0, every bias and every LayerNorm affine, get a direction of 0, the
repository's --xignore biasbn default that upstream runs with.
"""

from typing import Callable, Iterable

import torch
import torch.nn as nn
from lightning import seed_everything

from defences.inference import forward_logits

Direction = list[torch.Tensor]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# The repository's guard against a zero-norm slice of the random draw.
FILTER_NORM_EPSILON = 1e-10
NORMALISATIONS = ("filter", "layer")


def landscape_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Every parameter, in model.parameters() order, the repository's get_weights.

    Every one and not only those requiring a gradient, since the surface is a
    property of the weights and not of what training moved.
    """
    parameters = list(model.parameters())
    return parameters


def random_direction(
    model: nn.Module, seed: int, normalisation: str = "filter"
) -> Direction:
    """1 normalised random direction, 1 tensor per parameter, on the parameters' device.

    The Gaussian draw happens on the CPU in the parameters' dtype, in
    model.parameters() order, so the same seed gives the same direction on any
    machine. "filter" rescales each first-axis slice to the norm of the matching
    slice of the weights, "layer" rescales the whole tensor to the tensor's norm.
    Rank 0 and rank 1 parameters are 0 under either.
    """
    if normalisation not in NORMALISATIONS:
        raise ValueError(
            f"unknown normalisation {normalisation!r}, expected one of {NORMALISATIONS}"
        )
    seed_everything(seed)

    direction: Direction = []
    for parameter in landscape_parameters(model):
        weights = parameter.detach().cpu()  # parameter shape
        draw = torch.randn(weights.shape, dtype=weights.dtype)  # parameter shape
        if draw.dim() <= 1:
            scaled = torch.zeros_like(draw)  # parameter shape
        elif normalisation == "filter":
            scaled = _filter_normalised(draw, weights)  # parameter shape
        else:
            scaled = draw * (weights.norm() / draw.norm())  # parameter shape
        direction.append(scaled.to(parameter.device))

    return direction


def _filter_normalised(draw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Each first-axis slice of the draw rescaled to the norm of the weights' slice.

    The repository's normalize_direction loops over the slices with
    d.mul_(w.norm() / (d.norm() + 1e-10)). The per-slice norms over the flattened
    trailing axes are the same numbers in 1 call.
    """
    draw_norms = draw.flatten(1).norm(dim=1)  # (first_axis,)
    weight_norms = weights.flatten(1).norm(dim=1)  # (first_axis,)
    ratio = weight_norms / (draw_norms + FILTER_NORM_EPSILON)  # (first_axis,)

    trailing = (1,) * (draw.dim() - 1)
    scaled = draw * ratio.view(-1, *trailing)  # parameter shape
    return scaled


def direction_cosine(first: Direction, second: Direction) -> float:
    """The cosine between 2 directions as flat vectors, the repository's proj.cal_angle.

    Near 0 in a space of 86 M dimensions, which is what makes the 2 axes of the
    surface effectively orthogonal without an explicit orthogonalisation.
    """
    first_flat = torch.cat([tensor.reshape(-1) for tensor in first])  # (n_params,)
    second_flat = torch.cat(
        [tensor.reshape(-1).to(first_flat.device) for tensor in second]
    )  # (n_params,)

    cosine = torch.dot(first_flat, second_flat) / (
        first_flat.norm() * second_flat.norm()
    )
    value = cosine.item()
    return value


def loss_surface(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    directions: tuple[Direction, Direction],
    alphas: torch.Tensor,
    betas: torch.Tensor,
    loss_fn: LossFunction,
    device: torch.device,
    use_bfloat16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The loss and the accuracy on every grid point, both (len(alphas), len(betas)).

    Serial over the grid, 1 full pass over batches per point, the repository's
    crunch without its MPI reduction. batches is re-iterated at every point, so
    a list of tensors costs nothing per pass where a DataLoader would rebuild its
    workers 441 times. losses[i, j] and accuracies[i, j] belong to the point
    (alphas[i], betas[j]). The loss is the row-weighted mean over the batches and
    the accuracy the fraction of rows predicted correctly.

    The weights are set to theta* + alpha delta + beta eta from a stored copy of
    theta* before each point and copied back from that copy after it, so the model
    leaves the surface bit for bit as it entered, an exception midway included.
    """
    parameters = landscape_parameters(model)
    first, second = directions
    assert len(first) == len(parameters) and len(second) == len(parameters), (
        f"directions have {len(first)} and {len(second)} tensors, "
        f"the model has {len(parameters)} parameters"
    )
    for tensor, parameter in zip(first + second, parameters):
        assert tensor.shape == parameter.shape, (
            f"direction tensor {tuple(tensor.shape)} does not match "
            f"parameter {tuple(parameter.shape)}"
        )

    original = [parameter.detach().clone() for parameter in parameters]
    losses = torch.full((len(alphas), len(betas)), float("nan"))  # (A, B)
    accuracies = torch.full((len(alphas), len(betas)), float("nan"))  # (A, B)
    try:
        for row, alpha in enumerate(alphas.tolist()):
            for column, beta in enumerate(betas.tolist()):
                _displace_weights(parameters, original, first, second, alpha, beta)
                loss, accuracy = _evaluate(
                    model, batches, loss_fn, device, use_bfloat16
                )
                losses[row, column] = loss
                accuracies[row, column] = accuracy
                _restore_weights(parameters, original)
    finally:
        _restore_weights(parameters, original)

    return losses, accuracies


@torch.no_grad()
def _displace_weights(
    parameters: list[nn.Parameter],
    original: list[torch.Tensor],
    first: Direction,
    second: Direction,
    alpha: float,
    beta: float,
) -> None:
    """Set every parameter to theta* + alpha delta + beta eta, the repository's set_weights."""
    for parameter, weights, delta, eta in zip(parameters, original, first, second):
        displaced = weights + alpha * delta + beta * eta  # parameter shape
        parameter.copy_(displaced)


@torch.no_grad()
def _restore_weights(
    parameters: list[nn.Parameter], original: list[torch.Tensor]
) -> None:
    """Copy the stored theta* back into every parameter."""
    for parameter, weights in zip(parameters, original):
        parameter.copy_(weights)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: LossFunction,
    device: torch.device,
    use_bfloat16: bool,
) -> tuple[float, float]:
    """The row-weighted mean loss and the accuracy over batches, the repository's eval_loss."""
    total_loss = 0.0
    correct = 0
    rows = 0
    for images, labels in batches:
        labels_on_device = labels.to(device)  # (batch,)
        logits = forward_logits(model, images, device, use_bfloat16)  # (batch, C)

        total_loss += loss_fn(logits, labels_on_device).item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels_on_device).sum().item()
        rows += images.size(0)

    mean_loss = total_loss / rows
    accuracy = correct / rows
    return mean_loss, accuracy
