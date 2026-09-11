"""Input attributions of a prediction: frequency saliency, expected gradients, Grad-CAM.

3 of BackdoorBench's image panels explain 1 prediction on 1 image
(visual_fre.py, visual_shap.py and visual_gradcam.py at commit f02e353). Each
attribution here is taken at the model's INPUT, the native-resolution image the
Resize wrapper consumes, so on a ViT the map lives in pixel space rather than
on a patch grid a reader would have to unfold. Grad-CAM is the 1 exception by
construction, since it is defined on an activation grid. It is delegated to the
SentiNet port, which already places it at the right block for each
architecture and upsamples it here for the overlay.

Nothing here mutates the model. visual_utils.saliency switches every
parameter's requires_grad off and never switches it back, which the frozen
parameter context avoids while giving the same gradient.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything

from defences.inference import forward_logits, frozen_parameters
from detectors.sentinet import grad_cam, resolve_cam_site, scale_per_image

# shap.GradientExplainer's default sample count and the 2 classes the panel shows.
DEFAULT_EXPECTED_GRADIENT_SAMPLES = 200
DEFAULT_RANKED_OUTPUTS = 2


def frequency_saliency(
    model: nn.Module, image: torch.Tensor, device: torch.device, use_bfloat16: bool
) -> np.ndarray:
    """The frequency saliency map of the top logit, (height, width) uint8 in 0 to 255.

    Zeng et al. (ICCV 2021, "Rethinking the Backdoor Attacks' Triggers: A
    Frequency Perspective") read a trigger's footprint in the Fourier domain.
    BackdoorBench's visual_utils.saliency (lines 1000 to 1018) computes, for the
    gradient g of the top logit with respect to the input,

        original form
            G        = ifft2( g ) over the 2 spatial axes
            S        = log | fftshift( G ) |
            S_norm   = ( S - min S ) / ( max S - min S )      over all entries
            map      = uint8( 255 * mean over channels of S_norm )

    symbol table
        g        the input gradient, (height, width, channels)
        G        its inverse 2-d discrete Fourier transform per channel
        S        the log magnitude with the zero frequency moved to the centre
        map      the returned image

    ifft2 of a real array equals the conjugate of fft2 divided by the pixel
    count, so the magnitude is fft2's up to a constant factor and the min-max
    step removes the constant: the forward and inverse transforms give the same
    map. image is (channels, height, width) in normalised space.
    """
    pixels = image.detach().to(device)[None].requires_grad_(True)  # (1, C, H, W)
    with torch.enable_grad(), frozen_parameters(model):
        logits = forward_logits(model, pixels, device, use_bfloat16)  # (1, classes)
        top_logit = logits.max(dim=1).values  # (1,)
        (gradient,) = torch.autograd.grad(top_logit.sum(), pixels)  # (1, C, H, W)

    spatial_last = gradient[0].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    spectrum = np.fft.ifft2(spatial_last, axes=(0, 1))  # (H, W, C) complex
    centred = np.fft.fftshift(spectrum, axes=(0, 1))  # (H, W, C)
    log_magnitude = np.log(np.abs(centred))  # (H, W, C)

    scaled = (log_magnitude - log_magnitude.min()) / (
        log_magnitude.max() - log_magnitude.min()
    )  # (H, W, C)
    saliency_map = np.uint8(255 * np.mean(scaled, axis=2))  # (H, W)
    return saliency_map


def ranked_classes(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
    ranked_outputs: int,
) -> torch.Tensor:
    """The ranked_outputs highest logits' classes per image, (batch, ranked_outputs) long.

    shap.GradientExplainer ranks the model output on the unperturbed input and
    explains the top classes in that order, so the first column is the
    prediction.
    """
    with torch.inference_mode():
        logits = forward_logits(model, images, device, use_bfloat16)  # (batch, classes)

    ranked = logits.argsort(dim=1, descending=True)[
        :, :ranked_outputs
    ]  # (batch, ranked_outputs)
    on_cpu = ranked.cpu()
    return on_cpu


def expected_gradients(
    model: nn.Module,
    images: torch.Tensor,
    background: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
    num_samples: int = DEFAULT_EXPECTED_GRADIENT_SAMPLES,
    ranked_outputs: int = DEFAULT_RANKED_OUTPUTS,
    seed: int = 0,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected gradients of the top classes' logits, (attributions, classes).

    Erion et al. (Nature Machine Intelligence 2021, "Improving performance of
    deep learning models with axiomatic attribution priors and expected
    gradients") define, for a model f, an input x and a background
    distribution D,

        original form
            phi_i(x) = E_{x' ~ D, alpha ~ U(0, 1)} [
                           (x_i - x'_i) * d f / d x_i ( x' + alpha (x - x') ) ]

    symbol table
        x         the explained image
        x'        a background image drawn from D
        alpha     a point on the straight path from x' to x
        f         the logit being explained
        phi_i     the attribution of input entry i

    shap.GradientExplainer estimates the expectation with num_samples draws of
    (x', alpha), each a background row chosen uniformly and alpha uniform in
    [0, 1]. The attribution is the mean over draws of the gradient times
    (x - x'). The estimator here is that one, batched, with f the raw logit of
    each of the ranked_outputs top classes as upstream's ranked_outputs asks.

    images is (batch, C, H, W) and background (num_background, C, H, W), both
    normalised. Returns attributions (batch, ranked_outputs, C, H, W) float32
    on the CPU and the explained classes (batch, ranked_outputs) long.
    """
    if background.shape[1:] != images.shape[1:]:
        raise ValueError(
            f"background rows {tuple(background.shape[1:])} do not match images "
            f"{tuple(images.shape[1:])}"
        )
    seed_everything(seed)
    classes = ranked_classes(model, images, device, use_bfloat16, ranked_outputs)
    background_on_device = background.to(device)  # (num_background, C, H, W)

    attributions = torch.zeros(
        images.shape[0], ranked_outputs, *images.shape[1:]
    )  # (batch, ranked_outputs, C, H, W)
    with torch.enable_grad(), frozen_parameters(model):
        for image_index in range(images.shape[0]):
            explained = images[image_index].to(device)  # (C, H, W)
            for start in range(0, num_samples, batch_size):
                count = min(batch_size, num_samples - start)
                draws = torch.randint(
                    background.shape[0], (count,), device=device
                )  # (count,)
                alpha = torch.rand(count, 1, 1, 1, device=device)  # (count, 1, 1, 1)
                baseline = background_on_device[draws]  # (count, C, H, W)
                delta = explained[None] - baseline  # (count, C, H, W)
                path_points = (baseline + alpha * delta).requires_grad_(True)

                logits = forward_logits(
                    model, path_points, device, use_bfloat16
                )  # (count, classes)
                for rank in range(ranked_outputs):
                    target = classes[image_index, rank]
                    (gradient,) = torch.autograd.grad(
                        logits[:, target].sum(), path_points, retain_graph=True
                    )  # (count, C, H, W)
                    # Sum of gradient times delta over this chunk, divided by the
                    # draw count at the end, gives the mean over all draws.
                    attributions[image_index, rank] += (
                        (gradient * delta).sum(dim=0).cpu()
                    )

    averaged = attributions / num_samples  # (batch, ranked_outputs, C, H, W)
    return averaged, classes


def class_activation_map(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grad-CAM of the predicted class at image resolution, ((batch, H, W) in [0, 1], predicted).

    Delegates to detectors.sentinet.grad_cam, which reads the input of the last
    block on ViT and the output of the last block on Swin, and scales the map
    per image. The patch-grid map is bilinearly upsampled to the image and
    scaled once more, the pytorch_grad_cam convention upstream draws with.
    """
    layer, architecture = resolve_cam_site(model)
    cam, predicted = grad_cam(
        model, images, device, use_bfloat16, layer, architecture
    )  # (batch, grid, grid), (batch,)

    height, width = images.shape[2:]
    upsampled = F.interpolate(
        cam[:, None], size=(height, width), mode="bilinear", align_corners=False
    )  # (batch, 1, H, W)
    rescaled = scale_per_image(upsampled)[:, 0]  # (batch, H, W)
    cam_on_cpu = rescaled.detach().cpu()  # (batch, H, W)
    predicted_on_cpu = predicted.cpu()  # (batch,)
    return cam_on_cpu, predicted_on_cpu
