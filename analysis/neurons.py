"""Per-dimension neuron statistics of the residual stream, and their streaming extraction.

BackdoorBench's neuron tools (visual_na.py, visual_act.py, visual_actdist.py
and visual_tac.py at commit f02e353) treat a ConvNet channel as the unit and
collapse its (height, width) map by a sum, a flatten or a mean. The unit here
is a residual dimension and the spatial axis is the token axis, so every
statistic below is stated per dimension after summing over tokens, class token
included, which is the direct analogue of summing a channel's feature map.
The 1 exception is mean_activation, which upstream takes over the flattened
(channels, height, width) units. 197 by 768 bars is not a figure, so it too is
taken per dimension after the token sum, and its docstring says so.

Extraction streams through analysis.features.captured_layers a batch at a time,
because a full token sequence at 1 layer is (samples, 197, 768) float32, 300 MB
per 500 images, and the TAC heatmap needs all 12 layers.
"""

import torch
import torch.nn as nn

from defences.inference import forward_logits
from models.backbones import network_core

from .direction import trigger_activation_change
from .features import (
    as_token_sequence,
    captured_layers,
    detect_model_architecture,
    transformer_blocks,
)

# The token reductions the neuron tools accept. "sum" is upstream's
# reduction='sum', "none" keeps the sequence for the token maps.
TOKEN_REDUCTIONS = ("sum", "none")


def token_sum(activation: torch.Tensor) -> torch.Tensor:
    """Every dimension summed over tokens, (batch, dim), upstream's reduction='sum'.

    visual_utils.py line 447 sums a (batch, channels, height, width) map over
    its flattened spatial axis. The token axis plays that role here, and the
    class token counts as 1 more position of the same channel.
    """
    tokens = as_token_sequence(activation).float()  # (batch, tokens, dim)
    summed = tokens.sum(dim=1)  # (batch, dim)
    return summed


def block_count(model: nn.Module, architecture: str | None = None) -> int:
    """How many transformer blocks the model has, the largest layer index a tool may ask for."""
    core = network_core(model)
    resolved = (
        architecture if architecture is not None else detect_model_architecture(core)
    )

    count = len(transformer_blocks(core, resolved))
    return count


@torch.inference_mode()
def layer_activations(
    model: nn.Module,
    images: torch.Tensor,
    layer: int,
    device: torch.device,
    batch_size: int,
    use_bfloat16: bool,
    reduction: str = "sum",
    architecture: str | None = None,
) -> torch.Tensor:
    """The residual stream at 1 layer for a tensor of images, float32 on the CPU.

    reduction "sum" returns (num_images, dim), the token sum of every
    dimension. reduction "none" returns the raw sequence (num_images, tokens,
    dim), which is only affordable for a few images. Layer numbering is
    analysis.features': 0 is the first block's input, 1 to N the block outputs.
    """
    if reduction not in TOKEN_REDUCTIONS:
        raise ValueError(
            f"unknown reduction {reduction!r}, expected {TOKEN_REDUCTIONS}"
        )
    model.eval()

    chunks = []
    with captured_layers(model, (layer,), architecture) as captured:
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size]  # (batch, C, H, W)
            forward_logits(model, batch, device, use_bfloat16)
            activation = captured[layer]  # (batch, tokens, dim) or (batch, h, w, C)
            if reduction == "sum":
                chunks.append(token_sum(activation).cpu())  # (batch, dim)
            else:
                chunks.append(as_token_sequence(activation).float().cpu())

    activations = torch.cat(chunks)  # (N, dim) or (N, tokens, dim)
    return activations


@torch.inference_mode()
def paired_layer_tac(
    model: nn.Module,
    clean_images: torch.Tensor,
    backdoor_images: torch.Tensor,
    layers: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    use_bfloat16: bool,
    norm: str = "l1_sum",
    architecture: str | None = None,
) -> dict[int, torch.Tensor]:
    """TAC at every requested layer over index-aligned image tensors, each (dim,) on the CPU.

    Both TAC conventions are means over samples of a per-sample quantity, so a
    batch's TAC weighted by its size and summed over batches is the exact TAC of
    the whole set. That is what lets 12 layers of full token sequences stream
    through 1 forward pass per population per batch without ever being held.
    """
    if clean_images.shape != backdoor_images.shape:
        raise ValueError(
            f"TAC needs index-aligned images, got {tuple(clean_images.shape)} and "
            f"{tuple(backdoor_images.shape)}"
        )
    model.eval()
    num_images = clean_images.shape[0]

    weighted: dict[int, torch.Tensor] = {}
    with captured_layers(model, layers, architecture) as captured:
        for start in range(0, num_images, batch_size):
            clean_batch = clean_images[start : start + batch_size]  # (batch, C, H, W)
            backdoor_batch = backdoor_images[start : start + batch_size]
            weight = clean_batch.shape[0]

            forward_logits(model, clean_batch, device, use_bfloat16)
            clean_tokens = {
                layer: as_token_sequence(captured[layer]).float() for layer in layers
            }  # each (batch, tokens, dim)
            forward_logits(model, backdoor_batch, device, use_bfloat16)
            for layer in layers:
                backdoor_tokens = as_token_sequence(captured[layer]).float()
                batch_tac = trigger_activation_change(
                    clean_tokens[layer], backdoor_tokens, norm
                )  # (dim,)
                weighted[layer] = weighted.get(layer, 0.0) + weight * batch_tac.cpu()

    tac_by_layer = {layer: weighted[layer] / num_images for layer in layers}
    return tac_by_layer


def mean_activation(features: torch.Tensor) -> torch.Tensor:
    """Mean over samples of every dimension, (dim,), the bar height of the NA figure.

    visual_na.py lines 130 and 138 take np.mean over axis 0 of the flattened
    (channels, height, width) units. Here features is (num_samples, dim) after
    the token sum, so the bar is per residual dimension.
    """
    means = features.float().mean(dim=0)  # (dim,)
    return means


def top_activating_indices(features: torch.Tensor, k: int) -> torch.Tensor:
    """The k sample rows that activate each dimension most, (k, dim) long, strongest first.

    visual_act.py line 150 and visual_actdist.py line 171 take
    np.argsort(-features, axis=0)[:k]. features is (num_samples, dim) after the
    token sum.
    """
    if k > features.shape[0]:
        raise ValueError(f"asked for the top {k} rows of {features.shape[0]} samples")

    ordered = torch.argsort(-features.float(), dim=0)  # (num_samples, dim)
    top = ordered[:k]  # (k, dim)
    return top


def top_k_rule(num_samples: int, num_selected_classes: int, num_poisoned: int) -> int:
    """How many top images class purity counts, upstream's rule (visual_actdist.py lines 147 to 149).

    The count is the number of poisoned rows when there are any, so a
    dimension whose top images are all poisoned reads as purity 1 exactly, and
    otherwise the mean rows per selected class.
    """
    if num_poisoned > 0:
        return num_poisoned

    per_class = int(num_samples / num_selected_classes)
    return per_class


def class_purity(
    top_indices: torch.Tensor,
    labels: torch.Tensor,
    poison_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Share of each class among a dimension's top images, (dim, num_classes + 1).

    Column c for c below num_classes is the share of top rows whose TRUE class
    is c and whose row is clean, column num_classes is the share of poisoned
    rows, since visual_actdist.py line 151 relabels poisoned rows to the class
    index num_classes before counting (line 180). Rows of the result sum to 1.
    Upstream only emits columns for the classes present in the view, which is
    the same matrix with its all-zero columns dropped.
    """
    if labels.shape != poison_mask.shape:
        raise ValueError(
            f"labels {tuple(labels.shape)} and poison_mask {tuple(poison_mask.shape)} "
            "must be the same length"
        )
    k = top_indices.shape[0]

    labels_with_poison = labels.long().clone()  # (num_samples,)
    labels_with_poison[poison_mask] = num_classes
    top_labels = labels_with_poison[top_indices]  # (k, dim)

    counts = torch.stack(
        [(top_labels == c).sum(dim=0) for c in range(num_classes + 1)], dim=1
    )  # (dim, num_classes + 1)
    purity = counts.float() / k
    return purity
