"""SentiNet: region transplant against localized universal attacks (Chou et al., 2020).

Paper: "SentiNet: Detecting Localized Universal Attacks Against Deep Learning
Systems", IEEE S&P Workshops (DLS) 2020, arXiv:1812.00292. Localization is
Section III-A with Grad-CAM (Selvaraju et al.) and Algorithm 2, the 2 statistics
are Section III-B1 with Algorithm 3 and the decision rule is Section III-B2 with
Algorithm 4.

    original form
        alpha_c^k = (1 / Z) * sum_i sum_j  d y^c / d A^k_ij                 Grad-CAM
        L^c       = ReLU( sum_k alpha_c^k * A^k )

        R          = x * mask                                              Alg. 3
        X_R        = Overlay(X, R),   X_IP = Overlay(X, IP)
        fooled     = | { x_R in X_R : f(x_R) = y } |
        avgconf_IP = (1 / |X|) * sum_{x_IP in X_IP} conf_IP

        f_curve    = ApproximateCurve( OutPts(B) )                         Alg. 4
        d          = (1 / |B|) * sum_{(x, y) in B : f_curve(x) > y}
                     COBYLA( (x, y), f_curve )

    descriptive form
        token_weight   = mean over patch tokens of the gradient of the predicted
                         logit with respect to the tokens at the CAM layer
        cam            = relu(token_weight . token) per patch, scaled to [0, 1]
        mask           = upsampled cam >= MASK_THRESHOLD
        fooled         = fraction of the clean overlays that take the input's
                         predicted label once the masked region is pasted on them
        avg_conf       = mean max softmax when uniform noise is pasted instead
        envelope       = quadratic through the largest clean fooled values per
                         avg_conf bin
        residual       = fooled - envelope(avg_conf)
        sentinet_score = -residual

y^c is the logit of class c, A^k the k-th feature map (here the k-th feature of
every token) and Z the number of spatial positions. X is the set of benign
overlay images, R the transplanted region, IP the inert pattern, B the clean
(avg_conf, fooled) points and f_curve the fitted envelope.

Mechanism. A trigger is a small region that hijacks the prediction of any image
it lands on, so pasting the salient region of a triggered input onto clean
images drags them to its label (fooled high) while filling the same small region
with noise leaves the clean images intact (avg_conf high). A benign salient
region is either too weak to hijack (fooled low) or so large that occluding it
with noise destroys the prediction (avg_conf low). Both benign cases sit under
the envelope, a triggered input sits above it.

Data requirement: DEFAULT_NUM_OVERLAYS clean images, unlabelled, from the shared
clean validation split, plus the split's (avg_conf, fooled) points for the
envelope. Forward-pass cost: 1 forward and 1 backward for the CAM, then
2 * DEFAULT_NUM_OVERLAYS forwards per input, about 202 forward-equivalents.

Grad-CAM under bf16 autocast reads a float32 tensor. The residual stream stays
float32 because the position-embedding add and every residual add promote the
bf16 branch output, so the captured tokens and their gradient are float32 and
autocast rounds the branches rather than the map. The CAM is then scaled and
thresholded, which is a rounding of a map and not an optimisation trajectory,
so the shared autocast policy applies.

Deviations from the paper, recorded in full in docs/detectors/sentinet.md:

  1. Grad-CAM reads the input of the last block on ViT. torchvision's head reads
     x[:, 0] after the encoder, so the last block's output patch tokens receive
     exactly 0 gradient and a CAM there is blank. Swin's average pool reads every
     token, so its site is the last block output.
  2. Selective-search class proposal (Algorithm 1) and mask subtraction
     (Algorithm 2) are omitted, as every released reimplementation omits them.
  3. The mask is the scaled CAM at or above MASK_THRESHOLD, BackdoorBench's and
     Beatrix's rule. The paper binarizes at 15% of the maximum and then subtracts
     the proposal masks, backdoor-toolbox keeps a fixed 15% by area.
  4. The overlay set is the first DEFAULT_NUM_OVERLAYS images of the shared
     split and the inert set is DEFAULT_NUM_OVERLAYS uniform-noise images drawn
     once at fit time. Both are fixed for every scored input.
  5. fooled counts overlays predicted as the input's own predicted label, never
     the loader label, which backdoor-toolbox uses and which is an oracle.
  6. The score is the signed vertical residual above the envelope rather than a
     COBYLA perpendicular distance or BackdoorBench's avgconf > 0.9 rule.
  7. The envelope is fitted on the shared 2000-sample split (the paper uses
     about 400 held-out points), and the split's own residuals are in-sample by
     construction of an upper envelope. That moves the threshold, not the AUROC.
  8. Model queries run under the shared bf16 autocast policy.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything
from torch.utils.data import DataLoader

from analysis.features import (
    as_token_sequence,
    captured_layers,
    detect_model_architecture,
    transformer_blocks,
)
from defences.inference import forward_logits, forward_probs, frozen_parameters
from models.backbones import network_core

from .strip import collect_overlay_batch, normalization_buffers

# |X| in Algorithm 3, the benign images a region is transplanted onto. The paper
# ships 100 with the model. Beatrix redraws 10 per input, backdoor-toolbox uses
# its N clean samples and BackdoorBench a per-class share of clean_sample_num.
DEFAULT_NUM_OVERLAYS = 100

# mask_cond in BackdoorBench and Beatrix, applied to the CAM scaled to [0, 1].
# The paper binarizes at 15% of the maximum and then subtracts the Grad-CAM masks
# of the selective-search class proposals. Without that subtraction the 15% rule
# covers most of a natural image, which is why both reimplementations that drop
# Algorithms 1 and 2 tighten the threshold to 0.85.
MASK_THRESHOLD = 0.85

# backdoor-toolbox keeps the top 15% of CAM cells by area instead. Recorded and
# not used, because the decision rule needs the mask SIZE to vary. A benign input
# with a large salient region fools the overlays but noise in that region destroys
# confidence, a small trigger region gives both and a fixed area removes exactly
# the axis that separates them and collapses the rule to fooled alone.
OFFICIAL_TOOLBOX_MASK_FRACTION = 0.15

# OutPts(B) in Algorithm 4 as Beatrix's DecisionBoundary implements it: avg_conf
# bins of 0.04 and the 2 largest fooled values per bin. backdoor-toolbox uses
# bins of 0.02 and 1 point per bin, placed at the bin centre.
BOUNDARY_BIN_WIDTH = 0.04
BOUNDARY_POINTS_PER_BIN = 2

# The CAM layer in captured_layers numbering, relative to the block count. ViT
# hooks the INPUT of the last block (layer num_blocks - 1) because torchvision's
# head reads x[:, 0] only and the last block's output patch tokens carry 0
# gradient. Swin hooks the last block's OUTPUT (layer num_blocks) because its
# average pool reads every token.
CAM_LAYER_OFFSET: dict[str, int] = {"vit": -1, "swin": 0}

# Composites per forward pass. The 2 * DEFAULT_NUM_OVERLAYS composites of 1 input
# fit in 1 chunk, so the split only matters for a larger overlay count.
OVERLAY_CHUNK = 256

# Guards the per-image scaling of a CAM whose range is 0, which is what the
# zero-gradient site produces. pytorch_grad_cam adds 1e-7 to the maximum instead.
CAM_RANGE_FLOOR = 1e-7


def cam_layer(num_blocks: int, architecture: str) -> int:
    """The captured_layers index Grad-CAM reads on an architecture with num_blocks blocks."""
    if architecture not in CAM_LAYER_OFFSET:
        raise ValueError(
            f"no CAM layer rule for architecture {architecture!r}, known: "
            f"{sorted(CAM_LAYER_OFFSET)}"
        )

    layer = num_blocks + CAM_LAYER_OFFSET[architecture]
    if layer < 0:
        raise ValueError(
            f"{num_blocks} blocks leave no layer to read on {architecture!r}"
        )
    return layer


def resolve_cam_site(model: nn.Module) -> tuple[int, str]:
    """(layer, architecture) for a live model, from the block types it contains."""
    core = network_core(model)
    architecture = detect_model_architecture(core)
    num_blocks = len(transformer_blocks(core, architecture))

    layer = cam_layer(num_blocks, architecture)
    return layer, architecture


def scale_per_image(cam: torch.Tensor) -> torch.Tensor:
    """Each image's map shifted and scaled to [0, 1], same shape in and out.

    Subtracts the per-image minimum and divides by the per-image range, so the
    peak is exactly 1 unless the map is constant, in which case the map scales
    to all 0 rather than to NaN. A constant map is what the zero-gradient site
    produces, and a downstream threshold must see it as empty rather than crash.
    """
    flat = cam.flatten(1)  # (batch, cells)
    shifted = flat - flat.min(dim=1, keepdim=True).values  # (batch, cells)
    span = shifted.max(dim=1, keepdim=True).values.clamp_min(CAM_RANGE_FLOOR)
    scaled = shifted / span  # (batch, cells)

    scaled_map = scaled.view_as(cam)
    return scaled_map


def cam_token_weights(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
    layer: int,
    architecture: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The Grad-CAM ingredients at the CAM layer for the predicted class.

    Returns (tokens, weights, predicted). tokens is (batch, patches, dim)
    float32, detached, the activation at layer with the class token dropped on
    ViT. weights is (batch, dim), alpha_c^k of the Grad-CAM definition with c the
    predicted class, the gradient of the predicted logit averaged over the patch
    tokens. predicted is (batch,) long.

    images is (batch, channels, height, width) in normalized space. Cannot run
    under torch.inference_mode, since the gradient needs a graph.
    """
    if architecture not in CAM_LAYER_OFFSET:
        raise ValueError(
            f"no token layout rule for architecture {architecture!r}, known: "
            f"{sorted(CAM_LAYER_OFFSET)}"
        )
    first_patch_token = 1 if architecture == "vit" else 0

    # With every parameter frozen the input is the only leaf that can put the
    # activations into a graph. Autograd then builds no weight-gradient graph, so
    # no parameter can end up holding a .grad.
    pixels = images.to(device).detach().requires_grad_(True)  # (batch, C, H, W)
    with (
        torch.enable_grad(),
        frozen_parameters(model),
        captured_layers(model, (layer,), architecture) as captured,
    ):
        logits = forward_logits(model, pixels, device, use_bfloat16)  # (batch, classes)
        predicted = logits.argmax(dim=1)  # (batch,)
        predicted_logit = logits.gather(1, predicted[:, None])  # (batch, 1)
        activation = captured[
            layer
        ]  # (batch, tokens, dim) on ViT, (batch, h, w, C) on Swin
        (gradient,) = torch.autograd.grad(predicted_logit.sum(), activation)

    # The gradient is taken against the raw captured tensor, since a reshaped view
    # of it is a new graph node the logits never used. Both are then viewed as
    # token sequences so ViT and Swin share the rest of the computation.
    tokens = as_token_sequence(activation.detach()).float()  # (batch, tokens, dim)
    token_gradient = as_token_sequence(gradient).float()  # (batch, tokens, dim)
    patch_tokens = tokens[:, first_patch_token:, :]  # (batch, patches, dim)
    patch_gradient = token_gradient[:, first_patch_token:, :]  # (batch, patches, dim)

    # alpha_c^k: the gradient averaged over spatial positions, 1 weight per feature.
    weights = patch_gradient.mean(dim=1)  # (batch, dim)
    predicted_labels = predicted.detach()  # (batch,)
    return patch_tokens, weights, predicted_labels


def grad_cam(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
    layer: int,
    architecture: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grad-CAM of the predicted class on the patch grid, and the prediction.

    original form
        L^c = ReLU( sum_k alpha_c^k * A^k )
    descriptive form
        cam[patch] = relu( sum over features of token_weight * token[patch] )

    Returns (cam, predicted). cam is (batch, grid, grid) in [0, 1], scaled per
    image, 14 x 14 on ViT-B/16 and 7 x 7 on Swin-S at 224. predicted is
    (batch,) long, the label the map explains and the label fooled counts against.
    """
    patch_tokens, weights, predicted = cam_token_weights(
        model, images, device, use_bfloat16, layer, architecture
    )
    batch, num_patches, _ = patch_tokens.shape
    grid = math.isqrt(num_patches)
    if grid * grid != num_patches:
        raise ValueError(
            f"{num_patches} patch tokens do not form a square grid, so the CAM "
            "cannot be laid over the image"
        )

    weighted = (patch_tokens * weights[:, None, :]).sum(dim=2)  # (batch, patches)
    cam = F.relu(weighted).view(batch, grid, grid)  # (batch, grid, grid)

    scaled_cam = scale_per_image(cam)
    return scaled_cam, predicted


def saliency_mask(
    cam: torch.Tensor,
    image_size: tuple[int, int],
    threshold: float = MASK_THRESHOLD,
) -> torch.Tensor:
    """The region Algorithm 3 transplants, (batch, 1, height, width) bool, never empty.

    cam is (batch, grid, grid) in [0, 1]. It is bilinearly upsampled to
    image_size, scaled again per image so the peak is exactly 1 (pytorch_grad_cam
    scales once more after resizing) and thresholded. The peak pixel is then set
    explicitly, so a constant map, which the threshold alone would leave empty,
    still yields a 1-pixel region rather than a transplant of nothing.
    """
    batch = cam.size(0)
    height, width = image_size
    upsampled = F.interpolate(
        cam[:, None, :, :], size=(height, width), mode="bilinear", align_corners=False
    )  # (batch, 1, height, width)
    rescaled = scale_per_image(upsampled)  # (batch, 1, height, width)

    flat = rescaled.flatten(1)  # (batch, height * width)
    mask_flat = flat >= threshold  # (batch, height * width)
    peak = flat.argmax(dim=1)  # (batch,)
    mask_flat[torch.arange(batch, device=cam.device), peak] = True

    mask = mask_flat.view(batch, 1, height, width)
    return mask


@torch.inference_mode()
def overlay_statistics(
    model: nn.Module,
    pixels: torch.Tensor,
    masks: torch.Tensor,
    predicted: torch.Tensor,
    overlay_pixels: torch.Tensor,
    inert_pixels: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    device: torch.device,
    use_bfloat16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algorithm 3's 2 statistics for a batch, (fooled, avg_conf), each (batch,) float32.

    original form
        X_R    = Overlay(X, x * mask),   X_IP = Overlay(X, IP)
        fooled = | { x_R in X_R : f(x_R) = y } |
        avgconf_IP = (1 / |X|) * sum conf_IP
    descriptive form
        adversarial = overlay outside the mask, the input inside it
        inert       = overlay outside the mask, uniform noise inside it
        fooled      = fraction of adversarial copies predicted as the input's label
        avg_conf    = mean max softmax over the inert copies

    pixels is (batch, channels, height, width) in [0, 1], masks
    (batch, 1, height, width) bool, predicted (batch,) the label fooled counts
    against, overlay_pixels and inert_pixels (num_overlays, channels, height,
    width) in [0, 1]. Compositing follows Beatrix's _superimpose, background *
    mask + overlay * (1 - mask), in pixel space at the native resolution, and
    fooled is the count divided by |X| as Beatrix and backdoor-toolbox both do.

    The inert composite follows Algorithm 3 and backdoor-toolbox: noise inside
    the region, the clean overlay outside it. Beatrix and BackdoorBench paste the
    input's region onto a noise background instead, the reverse of the
    algorithm, so their avg_conf measures how the region alone classifies.
    """
    if pixels.shape[1:] != overlay_pixels.shape[1:]:
        raise ValueError(
            f"input shape {tuple(pixels.shape[1:])} and overlay shape "
            f"{tuple(overlay_pixels.shape[1:])} differ, so the region cannot be pasted"
        )
    if masks.shape != (pixels.size(0), 1, *pixels.shape[2:]):
        raise ValueError(
            f"mask shape {tuple(masks.shape)} does not match inputs of shape "
            f"{tuple(pixels.shape)}"
        )
    if overlay_pixels.shape != inert_pixels.shape:
        raise ValueError(
            f"overlay shape {tuple(overlay_pixels.shape)} and inert shape "
            f"{tuple(inert_pixels.shape)} differ"
        )

    mean_tensor, std_tensor = normalization_buffers(mean, std, device, pixels.dtype)
    overlays = overlay_pixels.to(device)  # (num_overlays, C, H, W)
    inert = inert_pixels.to(device)  # (num_overlays, C, H, W)
    num_overlays = overlays.size(0)

    labels = predicted.to(device)  # (batch,)

    fooled_rows = []
    conf_rows = []
    for image, mask, label in zip(pixels.to(device), masks.to(device), labels):
        region = mask.to(pixels.dtype)  # (1, H, W), 1 inside the region
        adversarial = (
            overlays * (1 - region) + image[None] * region
        )  # (num_overlays, C, H, W)
        inert_copies = (
            overlays * (1 - region) + inert * region
        )  # (num_overlays, C, H, W)
        composites = torch.cat(
            [adversarial, inert_copies]
        )  # (2 * num_overlays, C, H, W)
        normalized = (composites - mean_tensor) / std_tensor

        probs = torch.cat(
            [
                forward_probs(model, chunk, device, use_bfloat16)
                for chunk in normalized.split(OVERLAY_CHUNK)
            ]
        )  # (2 * num_overlays, classes)
        adversarial_labels = probs[:num_overlays].argmax(dim=1)  # (num_overlays,)
        inert_confidence = probs[num_overlays:].max(dim=1).values  # (num_overlays,)

        fooled_rows.append((adversarial_labels == label).float().mean())
        conf_rows.append(inert_confidence.mean())

    fooled = torch.stack(fooled_rows).float()  # (batch,)
    avg_conf = torch.stack(conf_rows).float()  # (batch,)
    return fooled, avg_conf


def collect_overlay_pixels(
    loader: DataLoader,
    count: int,
    seed: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> torch.Tensor:
    """X in Algorithm 3: the first count clean images in [0, 1], (count, C, H, W), on CPU.

    Drawn through strip.collect_overlay_batch from the shared clean validation
    split, so SentiNet sees the same data budget as every other method here. The
    loader delivers normalized tensors and compositing happens in pixel space, so
    the overlays are denormalized once here rather than per input.
    """
    overlays = collect_overlay_batch(
        loader, count, seed
    )  # (count, C, H, W), normalized
    if overlays.size(0) < count:
        raise ValueError(
            f"the loader yielded {overlays.size(0)} images, fewer than the {count} "
            "overlays requested, which would change |X| silently"
        )

    mean_tensor, std_tensor = normalization_buffers(
        mean, std, overlays.device, overlays.dtype
    )
    overlay_pixels = (overlays * std_tensor + mean_tensor).clamp(0.0, 1.0)
    return overlay_pixels


def draw_inert_pixels(
    count: int, image_shape: tuple[int, int, int], seed: int
) -> torch.Tensor:
    """IP in Algorithm 3: count uniform-noise images in [0, 1], (count, C, H, W), on CPU.

    The paper's default inert pattern is random noise. Drawn once from seed and
    fixed for every scored input, so the inert content is never a per-sample
    source of variance in avg_conf.
    """
    seed_everything(seed)
    channels, height, width = image_shape

    inert_pixels = torch.rand(count, channels, height, width)  # (count, C, H, W)
    return inert_pixels


def sentinet_statistics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    overlay_pixels: torch.Tensor,
    inert_pixels: torch.Tensor,
    layer: int,
    architecture: str,
    use_bfloat16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(fooled, avg_conf) per sample over a loader, each (N,) float32 on CPU, in loader order.

    The raw Algorithm 3 statistics, not scored. The builder fits the envelope on
    the clean validation split's pair and sentinet_scores turns another loader's
    pair into residuals, so the 2-D behaviour is kept where a run's provenance
    can read it back.
    """
    model.eval()

    fooled_batches = []
    conf_batches = []
    for images, _ in loader:
        images = images.to(device)  # (batch, C, H, W), normalized
        cam, predicted = grad_cam(
            model, images, device, use_bfloat16, layer, architecture
        )  # (batch, grid, grid), (batch,)
        masks = saliency_mask(cam, tuple(images.shape[2:]))  # (batch, 1, H, W)

        mean_tensor, std_tensor = normalization_buffers(mean, std, device, images.dtype)
        pixels = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)  # (batch, C, H, W)
        fooled, avg_conf = overlay_statistics(
            model,
            pixels,
            masks,
            predicted,
            overlay_pixels,
            inert_pixels,
            mean,
            std,
            device,
            use_bfloat16,
        )
        fooled_batches.append(fooled.cpu())
        conf_batches.append(avg_conf.cpu())

    if not fooled_batches:
        return torch.empty(0), torch.empty(0)

    statistics = (
        torch.cat(fooled_batches).float(),  # (N,)
        torch.cat(conf_batches).float(),  # (N,)
    )
    return statistics


def fit_decision_boundary(
    fooled: torch.Tensor,
    avg_conf: torch.Tensor,
    bin_width: float = BOUNDARY_BIN_WIDTH,
    points_per_bin: int = BOUNDARY_POINTS_PER_BIN,
) -> np.ndarray:
    """ApproximateCurve(OutPts(B)) of Algorithm 4: polynomial coefficients, highest degree first.

    original form
        f_curve = ApproximateCurve( OutPts(B) ),  f_curve(x) = a x^2 + b x + c
    descriptive form
        envelope points = the points_per_bin largest fooled values in each
                          avg_conf bin of width bin_width
        envelope        = least-squares polynomial through the envelope points

    fooled and avg_conf are the clean validation split's (N,) statistics. The
    degree is 2 with 3 or more populated bins, as Beatrix's quadratic curve_fit,
    and drops to the number of populated bins minus 1 below that, since a
    quadratic through 2 abscissae is not determined. The output feeds np.polyval.
    """
    fooled_values = np.asarray(fooled, dtype=np.float64)  # (N,)
    conf_values = np.asarray(avg_conf, dtype=np.float64)  # (N,)
    if fooled_values.shape != conf_values.shape or fooled_values.ndim != 1:
        raise ValueError(
            f"fooled {fooled_values.shape} and avg_conf {conf_values.shape} must be "
            "1-d and the same length"
        )
    if fooled_values.size == 0:
        raise ValueError("no clean points, so no envelope can be fitted")
    if not (np.isfinite(fooled_values).all() and np.isfinite(conf_values).all()):
        raise ValueError("non-finite clean statistics, the envelope would be NaN")

    # avg_conf is a softmax maximum in [0, 1]. The top edge is folded into the
    # last bin so a value of exactly 1 does not open a bin of its own.
    num_bins = math.ceil(1.0 / bin_width)
    bin_index = np.minimum(
        np.floor(conf_values / bin_width).astype(int), num_bins - 1
    )  # (N,)

    envelope_conf = []
    envelope_fooled = []
    populated = np.unique(bin_index)
    for index in populated:
        members = np.flatnonzero(bin_index == index)
        largest = members[np.argsort(fooled_values[members])[-points_per_bin:]]
        envelope_conf.extend(conf_values[largest].tolist())
        envelope_fooled.extend(fooled_values[largest].tolist())

    degree = min(2, len(populated) - 1)
    coefficients = np.polyfit(envelope_conf, envelope_fooled, degree)  # (degree + 1,)
    return coefficients


def boundary_residual(
    fooled: torch.Tensor, avg_conf: torch.Tensor, coefficients: np.ndarray
) -> torch.Tensor:
    """fooled - f_curve(avg_conf) per sample, (N,) float32, positive above the envelope.

    The paper measures the perpendicular COBYLA distance from the curve and sets
    d from the clean points. The vertical residual keeps the sign and the
    ordering the curve induces without an optimiser per sample, and a quantile of
    the clean residuals plays the role of d.
    """
    conf_values = np.asarray(avg_conf, dtype=np.float64)  # (N,)
    fooled_values = np.asarray(fooled, dtype=np.float64)  # (N,)
    envelope = np.polyval(coefficients, conf_values)  # (N,)

    residual = torch.from_numpy(fooled_values - envelope).float()  # (N,)
    return residual


def sentinet_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    overlay_pixels: torch.Tensor,
    inert_pixels: torch.Tensor,
    layer: int,
    architecture: str,
    coefficients: np.ndarray,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    """SentiNet score per sample, shape (N,) float32 on CPU, low meaning poisoned.

    Negated at this boundary. SentiNet's claim is that a triggered input sits
    above the clean envelope, so its residual is high for poisoned, the opposite
    of PSU's convention. Returning it unnegated would produce a well-formed,
    exactly inverted detector.
    """
    fooled, avg_conf = sentinet_statistics(
        model,
        loader,
        device,
        mean,
        std,
        overlay_pixels,
        inert_pixels,
        layer,
        architecture,
        use_bfloat16,
    )
    if fooled.numel() == 0:
        return torch.empty(0)

    residual = boundary_residual(fooled, avg_conf, coefficients)  # (N,)
    if not torch.isfinite(residual).all():
        raise ValueError(
            "non-finite SentiNet residuals, the envelope or a forward pass failed"
        )

    scores = -residual  # (N,), low means poisoned
    return scores
