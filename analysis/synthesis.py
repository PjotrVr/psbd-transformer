"""Feature visualisation: the input that maximises 1 residual dimension of a block.

Olah, Mordvintsev and Schubert, "Feature Visualization", Distill 2017, in the
form OmniXAI's FeatureVisualizer runs it and BackdoorBench's visual_fv.py calls
it. An image is parameterised in the Fourier basis with a 1/frequency scaling
and a colour decorrelation, mapped to valid RGB by a sigmoid, transformed at
random each step and pushed by Adam to raise the activation of a chosen unit,
against an L1 and a total-variation penalty on the image. Every constant below
is the one those 2 sources use, and the Fourier and colour parameterisation is
lucid's, which lucent ports to torch and OmniXAI copies.

    original form (per image b with its own unit k_b)
        x_b     = clip_to_range( sigmoid( decorrelate( irfft2( S_b * scale ) / 4 ) ) )
        scale   = 1 / max(f, 1 / max(h, w)) ^ decay
        loss_b  = - a_{k_b}(T(x_b)) + w_1 mean|T(x_b)| + w_tv TV(T(x_b))
        TV(x)   = ( sum (x_{i+1,j} - x_{i,j})^2 + sum (x_{i,j+1} - x_{i,j})^2 ) / (c h w)

    symbols
        S_b      the spectrum of image b, real and imaginary parts, the free variable
        f        the radial frequency of each spectrum entry
        decay    the power the 1/f scaling is raised to, 1
        T        the random transform of the step, jitter and scale
        a_k      the activation of unit k at the chosen block
        w_1      the L1 weight, 0.15 upstream
        w_tv     the total-variation weight, 0.25 upstream
        c, h, w  the image's channels, height and width

Upstream's unit is a convolution channel, mean over its spatial map. A
transformer block emits (batch, tokens, dim), so the unit here is 1 residual
dimension read at the class token on ViT, the token the head classifies from,
and as the mean over tokens on Swin, which has no class token. The image is
synthesised at the dataset's native resolution, where the triggers live, and
the model's own Resize front-end takes it to 224 as it does every input.
"""

from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything

from analysis.direction import trigger_activated_change
from analysis.features import (
    as_token_sequence,
    captured_layers,
    detect_model_architecture,
)
from defences.inference import forward_logits, frozen_parameters
from models.backbones import network_core

# lucid's colour decorrelation: the square root of the SVD of the ImageNet pixel
# colour covariance, normalised by its largest column norm.
COLOUR_CORRELATION_SVD_SQRT = torch.tensor(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
)  # (3, 3)
COLOUR_MAX_NORM = COLOUR_CORRELATION_SVD_SQRT.norm(dim=0).max()
COLOUR_CORRELATION_NORMALISED = COLOUR_CORRELATION_SVD_SQRT / COLOUR_MAX_NORM  # (3, 3)

# lucid divides the inverse transform by 4, a constant it labels magic.
FFT_MAGIC = 4.0
# OmniXAI's per-image range after the sigmoid, with its spectrum init std, its
# Adam step and the frequency decay it passes through.
VALUE_RANGE = (0.05, 0.95)
INIT_STD = 0.01
LEARNING_RATE = 0.05
FFT_DECAY = 1.0
# The regulariser weights BackdoorBench's visual_fv.py passes. Its l2 weight is 0,
# so the l2 term is not implemented.
L1_WEIGHT = 0.15
TV_WEIGHT = 0.25
STEPS = 300
# The jitter is BackdoorBench's train-time random-crop padding, the scales are
# lucent's standard set, 0.9 to 1.1 in steps of 0.02.
JITTER = 4
SCALES = tuple(1 + (index - 5) / 50.0 for index in range(11))


def rfft2d_frequencies(height: int, width: int) -> torch.Tensor:
    """The radial frequency of each entry of a real 2-D spectrum, (height, width // 2 + 1).

    lucid's rfft2d_freqs. An odd width keeps 1 extra column that the inverse
    transform's crop removes again.
    """
    frequencies_y = torch.fft.fftfreq(height)[:, None]  # (height, 1)
    columns = width // 2 + 2 if width % 2 == 1 else width // 2 + 1
    frequencies_x = torch.fft.fftfreq(width)[:columns]  # (columns,)

    radial = torch.sqrt(frequencies_x**2 + frequencies_y**2)  # (height, columns)
    return radial


def fourier_scale(
    height: int, width: int, decay_power: float = FFT_DECAY
) -> torch.Tensor:
    """The 1/frequency weight of each spectrum entry, (height, width // 2 + 1).

    Low frequencies are amplified relative to high ones, so an optimiser step of
    a given size moves the image more in its smooth components than in its noise,
    the preconditioning that gives feature visualisation its readable images.
    """
    frequencies = rfft2d_frequencies(height, width)  # (height, columns)
    floor = 1.0 / max(width, height)
    scale = (
        1.0 / torch.clamp(frequencies, min=floor) ** decay_power
    )  # (height, columns)
    return scale


def spectrum_to_image(
    spectrum: torch.Tensor, scale: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """The unconstrained image a spectrum encodes, (batch, 3, height, width).

    lucent's fft_image inner: the scaled complex spectrum through an orthonormal
    inverse real FFT, cropped to the image size and divided by the magic 4.
    """
    scaled = spectrum * scale[None, None, :, :, None]  # (batch, 3, height, columns, 2)
    complex_spectrum = torch.view_as_complex(
        scaled.contiguous()
    )  # (batch, 3, height, columns)
    image = torch.fft.irfftn(
        complex_spectrum, s=(height, width), norm="ortho"
    )  # (batch, 3, height, width)

    cropped = image[:, :, :height, :width] / FFT_MAGIC  # (batch, 3, height, width)
    return cropped


def decorrelate_colours(image: torch.Tensor) -> torch.Tensor:
    """Map decorrelated colour channels to correlated RGB, (batch, 3, height, width).

    lucid's _linear_decorrelate_color: each pixel's 3 channels times the
    normalised correlation matrix transposed.
    """
    matrix = COLOUR_CORRELATION_NORMALISED.to(image.device, image.dtype)  # (3, 3)
    channels_last = image.permute(0, 2, 3, 1)  # (batch, height, width, 3)
    mixed = channels_last @ matrix.T  # (batch, height, width, 3)

    channels_first = mixed.permute(0, 3, 1, 2)  # (batch, 3, height, width)
    return channels_first


def to_valid_image(
    spectrum: torch.Tensor, scale: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """The image in [low, high] a spectrum stands for, (batch, 3, height, width).

    OmniXAI's _normalize with use_fft: decorrelate, sigmoid, then rescale each
    image's own minimum and maximum onto VALUE_RANGE, which lucid does not do and
    which keeps every synthesised image at full contrast.
    """
    low, high = VALUE_RANGE
    image = spectrum_to_image(spectrum, scale, height, width)  # (batch, 3, h, w)
    correlated = decorrelate_colours(image)  # (batch, 3, h, w)
    squashed = torch.sigmoid(correlated)  # (batch, 3, h, w)

    flat = squashed.flatten(1)  # (batch, 3 * h * w)
    shifted = flat - flat.min(dim=1, keepdim=True).values  # (batch, 3 * h * w)
    unit = shifted / (shifted.max(dim=1, keepdim=True).values + 1e-8)  # (batch, 3hw)
    ranged = unit * (high - low) + low  # (batch, 3 * h * w)

    valid = ranged.view_as(squashed)  # (batch, 3, h, w)
    return valid


def random_transform(
    image: torch.Tensor, jitter: int, scales: tuple[float, ...]
) -> torch.Tensor:
    """The image translated by up to jitter pixels and rescaled by a random factor.

    The jitter is BackdoorBench's train transform, RandomCrop with a zero padding
    of 4, applied to the batch as 1 shift. The scale is lucent's random_scale, a
    bilinear resize by a factor drawn from scales, whose output size the model's
    own Resize front-end absorbs. Randomness comes from torch's global generator,
    so the seed the caller sets fixes the whole sequence of transforms.
    """
    padded = F.pad(
        image, [jitter] * 4, mode="constant", value=0.0
    )  # (b, 3, h+2j, w+2j)
    height, width = image.shape[-2:]
    offset_y = int(torch.randint(0, 2 * jitter + 1, (1,)).item())
    offset_x = int(torch.randint(0, 2 * jitter + 1, (1,)).item())
    shifted = padded[
        :, :, offset_y : offset_y + height, offset_x : offset_x + width
    ]  # (batch, 3, h, w)

    factor = scales[int(torch.randint(0, len(scales), (1,)).item())]
    scaled = F.interpolate(
        shifted, scale_factor=factor, mode="bilinear", align_corners=False
    )  # (batch, 3, round(h * factor), round(w * factor))
    return scaled


def total_variation(image: torch.Tensor) -> torch.Tensor:
    """OmniXAI's total variation per image, (batch,), squared differences over c h w."""
    _, channels, height, width = image.shape
    vertical = (image[:, :, 1:, :] - image[:, :, :-1, :]) ** 2  # (b, c, h-1, w)
    horizontal = (image[:, :, :, 1:] - image[:, :, :, :-1]) ** 2  # (b, c, h, w-1)

    per_image = (vertical.sum(dim=(1, 2, 3)) + horizontal.sum(dim=(1, 2, 3))) / (
        channels * height * width
    )  # (batch,)
    return per_image


def pooled_units(activation: torch.Tensor, architecture: str) -> torch.Tensor:
    """A block output pooled the way its unit is read, (batch, dim), float32.

    The class token on ViT, the token the head classifies from, or the mean
    over tokens on Swin, which has no class token.
    """
    tokens = as_token_sequence(activation).float()  # (batch, tokens, dim)
    if architecture == "vit":
        pooled = tokens[:, 0, :]  # (batch, dim)
    else:
        pooled = tokens.mean(dim=1)  # (batch, dim)
    return pooled


def unit_activation(
    activation: torch.Tensor, dimensions: torch.Tensor, architecture: str
) -> torch.Tensor:
    """Each image's own chosen residual dimension, (batch,)."""
    pooled = pooled_units(activation, architecture)  # (batch, dim)
    rows = torch.arange(pooled.size(0), device=pooled.device)  # (batch,)
    values = pooled[rows, dimensions.to(pooled.device)]  # (batch,)
    return values


def maximising_input(
    model: nn.Module,
    layer: int,
    dimensions: torch.Tensor,
    image_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    device: torch.device,
    use_bfloat16: bool,
    steps: int = STEPS,
    learning_rate: float = LEARNING_RATE,
    l1_weight: float = L1_WEIGHT,
    tv_weight: float = TV_WEIGHT,
    jitter: int = JITTER,
    scales: tuple[float, ...] = SCALES,
    seed: int = 0,
    architecture: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1 synthesised image per dimension and the objective's trajectory.

    Returns (images, objectives): images (len(dimensions), 3, image_size,
    image_size) in [0, 1] without any transform applied, objectives (steps,
    len(dimensions)) holding each image's unit activation at every step, under
    that step's transform. All images are optimised as 1 batch, so 1 forward and
    1 backward per step serve every dimension. The model's parameters are frozen
    for the duration and every hook is removed on exit.

    mean and std are the dataset's normalisation, applied after the transform and
    before the model, the place BackdoorBench's train transform put them.
    """
    core = network_core(model)
    resolved_architecture = (
        architecture if architecture is not None else detect_model_architecture(core)
    )
    seed_everything(seed)

    batch = len(dimensions)
    columns = rfft2d_frequencies(image_size, image_size).shape[1]
    spectrum = (
        (torch.randn(batch, 3, image_size, columns, 2) * INIT_STD)
        .to(device)
        .requires_grad_(True)
    )  # (batch, 3, h, columns, 2)
    scale = fourier_scale(image_size, image_size).to(device)  # (h, columns)
    channel_mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)  # (1, 3, 1, 1)
    channel_std = torch.tensor(std, device=device).view(1, 3, 1, 1)  # (1, 3, 1, 1)
    optimizer = torch.optim.Adam([spectrum], lr=learning_rate)

    objectives = torch.zeros(steps, batch)  # (steps, batch)
    with (
        frozen_parameters(model),
        captured_layers(model, (layer,), resolved_architecture) as captured,
    ):
        for step in range(steps):
            image = to_valid_image(spectrum, scale, image_size, image_size)
            transformed = random_transform(image, jitter, scales)  # (b, 3, h', w')
            normalised = (transformed - channel_mean) / channel_std  # (b, 3, h', w')
            forward_logits(model, normalised, device, use_bfloat16)

            activation = unit_activation(
                captured[layer], dimensions, resolved_architecture
            )  # (batch,)
            penalty = l1_weight * transformed.abs().mean(
                dim=(1, 2, 3)
            ) + tv_weight * total_variation(transformed)  # (batch,)
            loss = -activation + penalty  # (batch,)

            # Each image's loss depends on its own spectrum alone, so the gradient
            # of the sum is the per-image gradient OmniXAI takes through unbind.
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            objectives[step] = activation.detach().cpu()

    with torch.no_grad():
        images = to_valid_image(spectrum, scale, image_size, image_size).cpu()

    return images, objectives


def _paired_batches(
    clean_loader, backdoor_loader, max_batches: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """The first max_batches (clean, backdoor) image pairs, row for row."""
    for index, ((clean, _), (backdoor, _)) in enumerate(
        zip(clean_loader, backdoor_loader)
    ):
        if index >= max_batches:
            break
        yield clean, backdoor


@torch.no_grad()
def rank_dimensions_by_tac(
    model: nn.Module,
    clean_loader,
    backdoor_loader,
    layer: int,
    device: torch.device,
    use_bfloat16: bool,
    max_batches: int = 4,
    architecture: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual dimensions of a block sorted by pooled TAC, largest first.

    Returns (order, tac): order (dim,) the dimension indices in descending TAC,
    tac (dim,) the TAC of every dimension. The pooling is the unit's own, class
    token on ViT and token mean on Swin, over the first max_batches paired
    batches of the 2 loaders. The dimensions the trigger moves most are the ones
    worth synthesising an input for.
    """
    core = network_core(model)
    resolved_architecture = (
        architecture if architecture is not None else detect_model_architecture(core)
    )
    every_dimension = None
    clean_rows = []
    backdoor_rows = []
    with captured_layers(model, (layer,), resolved_architecture) as captured:
        for clean, backdoor in _paired_batches(
            clean_loader, backdoor_loader, max_batches
        ):
            forward_logits(model, clean, device, use_bfloat16)
            clean_pooled = pooled_units(
                captured[layer], resolved_architecture
            )  # (b, dim)
            forward_logits(model, backdoor, device, use_bfloat16)
            backdoor_pooled = pooled_units(captured[layer], resolved_architecture)
            clean_rows.append(clean_pooled.cpu())
            backdoor_rows.append(backdoor_pooled.cpu())
            every_dimension = clean_pooled.size(1)

    assert every_dimension is not None, "the loaders yielded no batch to rank on"
    tac = trigger_activated_change(
        torch.cat(clean_rows), torch.cat(backdoor_rows)
    )  # (dim,)
    order = torch.argsort(tac, descending=True)  # (dim,)
    return order, tac
