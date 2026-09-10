"""TeCo: test-time corruption robustness consistency (Liu et al., CVPR 2023).

Paper: "Detecting Backdoors During the Inference Stage Based on Corruption
Robustness Consistency", arXiv:2303.18191. The score is Section 4.2, Algorithm 1,
which has no numbered equation of its own, and the decision rule is Equation (4).

    original form, Algorithm 1
        P_org <- C_theta(x)
        for k = 1..K:
            l <- N + 1
            for n = 1..N:
                if C_theta( D_k^n(x) ) != P_org:
                    l <- n
                    break
            L <- L union {l}
        TeCo(x) = Dev(L)

        Gamma( TeCo(x) ) = 1 if TeCo(x) > gamma else 0              Eq. (4)

    descriptive form
        reference_label   = the label predicted on the uncorrupted image
        hardness[k]       = the lowest severity of corruption k at which the
                            prediction stops matching reference_label, or
                            max_severity + 1 when it never stops matching
        teco_score(image) = population standard deviation of hardness over the
                            K corruption types

K is 14 corruption types here, N is 5 severities and Dev is the population
standard deviation. gamma is swept rather than fixed.

Mechanism. Every corruption erodes ordinary class evidence at a similar rate, so
a clean input's prediction breaks at a similar severity under each of them and
the spread is small. A trigger's survival is corruption-specific, blur destroys a
patch trigger while brightness leaves it intact, so a triggered input's break
severities vary and the spread is large.

Data requirement: none for the score. Only the threshold needs clean data, and it
comes from the shared clean validation split like every other method's.
Forward-pass cost: K * N + 1 per input, 71 here. Algorithm 1's break would allow
an early exit, but the released code evaluates every severity and applies the
break to cached predictions, which is the cost a batched implementation pays.

Deviations from the paper, each recorded in full in docs/detector-ports.md:

  1. 14 corruptions, not 15. frost composites bundled photographs and cannot be
     reproduced from a formula, so a TeCo number here is not numerically
     identical to a published number.
  2. Each corruption is applied to the pristine image, as Algorithm 1 writes it.
     Both released implementations mutate the image in place and compose every
     corruption cumulatively, so a faithful reproduction is expected to differ
     from the published numbers.
  3. Dev is the population standard deviation, which is what the released code
     computes despite naming the variable mad.
  4. The motion blur and snow angles are drawn once per batch rather than per
     image, which is what makes the corruptions batchable.
  5. Where a dependency was missing an operator was substituted and checked
     against its numpy or scipy reference.

Corruption is applied in [0, 1] pixel space at the dataset's native resolution,
where this project applies triggers too, and the result is requantized to the
8-bit grid after each one because the reference works on uint8 arrays.
"""

import io
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything
from PIL import Image
from torch.utils.data import DataLoader

from defences.inference import forward_probs

# N in Algorithm 1. Severities run 1..MAX_SEVERITY.
MAX_SEVERITY = 5

# Algorithm 1 line 3: the hardness threshold of a corruption whose prediction
# never leaves P_org. Deliberately above every real severity, so "never broke"
# ranks as the most robust outcome without needing a separate code path.
NEVER_FLIPPED = MAX_SEVERITY + 1

# The paper's corruption set, the 15 standard ImageNet-C corruptions of Hendrycks
# and Dietterich. Recorded so the 1 this file cannot reproduce is visible.
IMAGENET_C_CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)

# frost composites bundled photographs and has no closed form, so it cannot be
# reproduced without the package's binary assets.
UNAVAILABLE_CORRUPTIONS: tuple[str, ...] = ("frost",)


def _quantize(images: torch.Tensor) -> torch.Tensor:
    """Clamp to [0, 1] and snap to the 8-bit grid the reference operates on."""
    quantized = (images.clamp(0.0, 1.0) * 255.0).round() / 255.0
    return quantized


def _gaussian_kernel_1d(sigma: float, truncate: float) -> torch.Tensor:
    """scipy's gaussian_filter kernel, radius int(truncate * sigma + 0.5)."""
    radius = int(truncate * sigma + 0.5)
    positions = torch.arange(-radius, radius + 1, dtype=torch.float64)
    weights = torch.exp(-(positions**2) / (2.0 * sigma**2))

    normalized = (weights / weights.sum()).float()
    return normalized


def _separable_blur(
    images: torch.Tensor, kernel: torch.Tensor, pad_mode: str
) -> torch.Tensor:
    """Blur each channel independently with a 1-D kernel applied on both axes."""
    channels = images.shape[1]
    radius = (kernel.numel() - 1) // 2
    kernel = kernel.to(images.device, images.dtype)

    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

    padded = F.pad(images, (radius, radius, 0, 0), mode=pad_mode)
    blurred = F.conv2d(padded, horizontal, groups=channels)
    padded = F.pad(blurred, (0, 0, radius, radius), mode=pad_mode)
    blurred = F.conv2d(padded, vertical, groups=channels)
    return blurred


def _skimage_gaussian(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """skimage.filters.gaussian defaults: mode nearest, truncate 4.0."""
    blurred = _separable_blur(images, _gaussian_kernel_1d(sigma, 4.0), "replicate")
    return blurred


def _disk_kernel(radius: int, alias_blur: float) -> torch.Tensor:
    """The antialiased disk of the reference's defocus_blur, shape (size, size)."""
    if radius <= 8:
        positions = torch.arange(-8, 9, dtype=torch.float32)
        kernel_size = 3
    else:
        positions = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_size = 5

    grid_x, grid_y = torch.meshgrid(positions, positions, indexing="xy")
    aliased = ((grid_x**2 + grid_y**2) <= radius**2).float()
    aliased = aliased / aliased.sum()

    # cv2.getGaussianKernel with an explicit positive sigma, which is the
    # unnormalized Gaussian sampled on the kernel's integer offsets.
    centre = (kernel_size - 1) / 2.0
    offsets = torch.arange(kernel_size, dtype=torch.float64) - centre
    weights = torch.exp(-(offsets**2) / (2.0 * alias_blur**2))
    weights = (weights / weights.sum()).float()

    antialiased = _separable_blur(
        aliased.view(1, 1, *aliased.shape), weights, "reflect"
    )
    return antialiased.view(*aliased.shape)


def _rgb_to_hsv(images: torch.Tensor) -> torch.Tensor:
    """skimage.color.rgb2hsv on a (batch, 3, H, W) tensor, same shape out."""
    red, green, blue = images[:, 0], images[:, 1], images[:, 2]
    highest = images.max(dim=1).values  # (batch, H, W)
    lowest = images.min(dim=1).values
    span = highest - lowest

    value = highest
    saturation = torch.where(
        highest > 0, span / highest.clamp_min(1e-12), torch.zeros_like(highest)
    )

    safe_span = span.clamp_min(1e-12)
    red_gap = (highest - red) / safe_span
    green_gap = (highest - green) / safe_span
    blue_gap = (highest - blue) / safe_span
    hue = torch.where(
        highest == red,
        blue_gap - green_gap,
        torch.where(
            highest == green, 2.0 + red_gap - blue_gap, 4.0 + green_gap - red_gap
        ),
    )
    hue = (hue / 6.0) % 1.0
    hue = torch.where(span == 0, torch.zeros_like(hue), hue)

    return torch.stack([hue, saturation, value], dim=1)


def _hsv_to_rgb(images: torch.Tensor) -> torch.Tensor:
    """skimage.color.hsv2rgb on a (batch, 3, H, W) tensor, same shape out."""
    hue, saturation, value = images[:, 0], images[:, 1], images[:, 2]
    sector = torch.floor(hue * 6.0)
    fraction = hue * 6.0 - sector

    dimmed = value * (1.0 - saturation)
    falling = value * (1.0 - saturation * fraction)
    rising = value * (1.0 - saturation * (1.0 - fraction))

    options = torch.stack(
        [
            torch.stack([value, rising, dimmed], dim=1),
            torch.stack([falling, value, dimmed], dim=1),
            torch.stack([dimmed, value, rising], dim=1),
            torch.stack([dimmed, falling, value], dim=1),
            torch.stack([rising, dimmed, value], dim=1),
            torch.stack([value, dimmed, falling], dim=1),
        ],
        dim=0,
    )  # (6, batch, 3, H, W)

    index = (sector % 6).long().unsqueeze(1).unsqueeze(0).expand(1, -1, 3, -1, -1)
    return options.gather(0, index).squeeze(0)


def _clipped_zoom(images: torch.Tensor, zoom_factor: float) -> torch.Tensor:
    """The reference's clipped_zoom: centre crop by 1/zoom, then bilinear upscale.

    scipy.ndimage.zoom with grid_mode=False aligns the corner pixel centres, which
    is align_corners=True in torch.
    """
    height, width = images.shape[-2], images.shape[-1]
    crop_height = int(math.ceil(height / zoom_factor))
    crop_width = int(math.ceil(width / zoom_factor))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2

    cropped = images[:, :, top : top + crop_height, left : left + crop_width]
    out_height = int(round(crop_height * zoom_factor))
    out_width = int(round(crop_width * zoom_factor))

    zoomed = F.interpolate(
        cropped, size=(out_height, out_width), mode="bilinear", align_corners=True
    )
    return zoomed


def _edge_shift(images: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """The reference's shift(): roll, then overwrite the wrapped band with an edge copy.

    torch.roll already allocates, so the in-place edge fills land on a fresh
    tensor and the caller's images are never touched. A zero shift is the identity
    and returns the input unchanged, which the only caller merely reads from.
    """
    if dx == 0 and dy == 0:
        return images

    shifted = images
    if dx < 0:
        shifted = torch.roll(shifted, shifts=dx, dims=-1)
        shifted[..., dx:] = shifted[..., dx - 1 : dx]
    elif dx > 0:
        shifted = torch.roll(shifted, shifts=dx, dims=-1)
        shifted[..., :dx] = shifted[..., dx : dx + 1]

    if dy < 0:
        shifted = torch.roll(shifted, shifts=dy, dims=-2)
        shifted[..., dy:, :] = shifted[..., dy - 1 : dy, :]
    elif dy > 0:
        shifted = torch.roll(shifted, shifts=dy, dims=-2)
        shifted[..., :dy, :] = shifted[..., dy : dy + 1, :]

    return shifted


def _directional_blur(
    images: torch.Tensor, radius: int, sigma: float, angle: float
) -> torch.Tensor:
    """The reference's _motion_blur: a half-Gaussian smear along a single direction."""
    width = radius * 2 + 1
    positions = torch.arange(width, dtype=torch.float32)
    weights = torch.exp(-(positions**2) / (2.0 * sigma**2)) / (
        math.sqrt(2 * math.pi) * sigma
    )
    weights = (weights / weights.sum()).to(images.device, images.dtype)

    point = (
        width * math.sin(math.radians(angle)),
        width * math.cos(math.radians(angle)),
    )
    hypotenuse = math.hypot(point[0], point[1])

    blurred = torch.zeros_like(images)
    for step in range(width):
        dy = -math.ceil(((step * point[0]) / hypotenuse) - 0.5)
        dx = -math.ceil(((step * point[1]) / hypotenuse) - 0.5)
        # The reference stops once the simulated motion leaves the frame, which
        # matters at 32x32 where a 20-pixel smear runs off the edge.
        if abs(dy) >= images.shape[-2] or abs(dx) >= images.shape[-1]:
            break
        blurred = blurred + weights[step] * _edge_shift(images, dx, dy)

    return blurred


def _plasma_fractal(batch: int, map_size: int, wibble_decay: float) -> np.ndarray:
    """The reference's diamond-square heightmap, batched, shape (batch, map, map).

    Transcribed from the reference including its unusual wibbledmean, which
    multiplies the wibble amplitude in twice. Vectorized over a leading batch axis
    so a single call serves a whole batch, which is the only change.
    """
    assert map_size & (map_size - 1) == 0, "map size must be a power of 2"

    maparray = np.empty((batch, map_size, map_size), dtype=np.float32)
    maparray[:, 0, 0] = 0
    step = map_size
    wibble = 100.0

    def wibbled_mean(array: np.ndarray) -> np.ndarray:
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    while step >= 2:
        corner = maparray[:, 0:map_size:step, 0:map_size:step]
        square = corner + np.roll(corner, shift=-1, axis=1)
        square = square + np.roll(square, shift=-1, axis=2)
        maparray[:, step // 2 : map_size : step, step // 2 : map_size : step] = (
            wibbled_mean(square)
        )

        centres = maparray[:, step // 2 : map_size : step, step // 2 : map_size : step]
        corners = maparray[:, 0:map_size:step, 0:map_size:step]
        left = (
            centres
            + np.roll(centres, 1, axis=1)
            + corners
            + np.roll(corners, -1, axis=2)
        )
        maparray[:, 0:map_size:step, step // 2 : map_size : step] = wibbled_mean(left)
        top = (
            centres
            + np.roll(centres, 1, axis=2)
            + corners
            + np.roll(corners, -1, axis=1)
        )
        maparray[:, step // 2 : map_size : step, 0:map_size:step] = wibbled_mean(top)

        step //= 2
        wibble /= wibble_decay

    maparray -= maparray.min(axis=(1, 2), keepdims=True)
    return maparray / maparray.max(axis=(1, 2), keepdims=True)


def _next_power_of_2(value: int) -> int:
    """The smallest power of 2 at or above value, the fractal map size fog needs."""
    power = 1 if value == 0 else 2 ** (value - 1).bit_length()
    return power


def gaussian_noise(images: torch.Tensor, severity: int) -> torch.Tensor:
    """Additive Gaussian noise at the reference's per-severity scale."""
    scale = (0.08, 0.12, 0.18, 0.26, 0.38)[severity - 1]
    noised = images + torch.randn_like(images) * scale
    return _quantize(noised)


def shot_noise(images: torch.Tensor, severity: int) -> torch.Tensor:
    """Poisson noise, each pixel resampled at the reference's per-severity rate."""
    rate = (60, 25, 12, 5, 3)[severity - 1]
    sampled = torch.poisson(images.clamp_min(0.0) * rate) / float(rate)
    return _quantize(sampled)


def impulse_noise(images: torch.Tensor, severity: int) -> torch.Tensor:
    """skimage.util.random_noise mode="s&p", salt_vs_pepper 0.5, drawn per element."""
    amount = (0.03, 0.06, 0.09, 0.17, 0.27)[severity - 1]

    flipped = torch.rand_like(images) <= amount
    salted = torch.rand_like(images) <= 0.5
    corrupted = torch.where(
        flipped,
        torch.where(salted, torch.ones_like(images), torch.zeros_like(images)),
        images,
    )
    return _quantize(corrupted)


def defocus_blur(images: torch.Tensor, severity: int) -> torch.Tensor:
    """Convolution with an antialiased disk of the per-severity radius."""
    radius, alias_blur = ((3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5))[
        severity - 1
    ]

    kernel = _disk_kernel(radius, alias_blur).to(images.device, images.dtype)
    channels = images.shape[1]
    pad = (kernel.shape[-1] - 1) // 2

    padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    weight = kernel.view(1, 1, *kernel.shape).expand(channels, 1, -1, -1)
    blurred = F.conv2d(padded, weight, groups=channels)
    return _quantize(blurred)


def glass_blur(images: torch.Tensor, severity: int) -> torch.Tensor:
    """Blur, locally shuffle pixels, blur again.

    The shuffle is sequential, since each swap sees the result of the previous
    swap, and the reference walks h and w downward. That loop order is preserved
    exactly. Only the batch axis is vectorized, so every image gets its own
    displacement draw at every step, as in the reference.
    """
    sigma, max_delta, iterations = (
        (0.7, 1, 2),
        (0.9, 2, 1),
        (1, 2, 3),
        (1.1, 3, 2),
        (1.5, 4, 2),
    )[severity - 1]

    blurred = _skimage_gaussian(images, sigma)
    # The reference casts to uint8 here, which truncates rather than rounds.
    working = (blurred.clamp(0.0, 1.0) * 255.0).floor() / 255.0

    batch, _, height, width = working.shape
    rows = torch.arange(batch, device=working.device)
    for _ in range(iterations):
        for h in range(height - max_delta, max_delta, -1):
            for w in range(width - max_delta, max_delta, -1):
                offsets = torch.randint(
                    -max_delta, max_delta, (2, batch), device=working.device
                )
                target_h = h + offsets[1]  # (batch,)
                target_w = w + offsets[0]  # (batch,)

                here = working[rows, :, h, w].clone()  # (batch, channels)
                there = working[rows, :, target_h, target_w].clone()
                working[rows, :, h, w] = there
                working[rows, :, target_h, target_w] = here

    return _quantize(_skimage_gaussian(working, sigma))


def motion_blur(images: torch.Tensor, severity: int) -> torch.Tensor:
    """A directional smear at a random angle, drawn once per batch."""
    radius, sigma = ((10, 3), (15, 5), (15, 8), (15, 12), (20, 15))[severity - 1]

    # 1 angle per batch rather than per image, see deviation 4.
    angle = float(torch.empty(1).uniform_(-45.0, 45.0).item())
    blurred = _directional_blur(images, radius, sigma, angle)
    return _quantize(blurred)


def zoom_blur(images: torch.Tensor, severity: int) -> torch.Tensor:
    """The average of the image with itself zoomed by each factor in the ladder."""
    factors = (
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    )[severity - 1]

    height, width = images.shape[-2], images.shape[-1]
    accumulated = torch.zeros_like(images)
    for factor in factors:
        zoomed = _clipped_zoom(images, float(factor))
        accumulated = accumulated + zoomed[:, :, :height, :width]

    averaged = (images + accumulated) / (len(factors) + 1)
    return _quantize(averaged)


def snow(images: torch.Tensor, severity: int) -> torch.Tensor:
    """Smeared flakes over a faded image, both drawn per batch as in the reference."""
    location, spread, zoom_factor, cutoff, radius, sigma, mix = (
        (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
        (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
        (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
    )[severity - 1]

    batch, _, height, width = images.shape
    flakes = (
        torch.randn(batch, 1, height, width, device=images.device) * spread + location
    )
    flakes = _clipped_zoom(flakes, zoom_factor)
    flakes = torch.where(flakes < cutoff, torch.zeros_like(flakes), flakes).clamp(
        0.0, 1.0
    )

    angle = float(torch.empty(1).uniform_(-135.0, -45.0).item())
    flakes = _directional_blur(flakes, radius, sigma, angle)
    flakes = (flakes * 255.0).round() / 255.0
    flakes = flakes[:, :, :height, :width]

    # cv2.COLOR_RGB2GRAY's luminance weights, which is what the reference calls.
    luminance = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    faded = mix * images + (1.0 - mix) * torch.maximum(images, luminance * 1.5 + 0.5)

    snowed = faded + flakes + torch.flip(flakes, dims=(-2, -1))
    return _quantize(snowed)


def fog(images: torch.Tensor, severity: int) -> torch.Tensor:
    """A plasma-fractal haze blended in and renormalized to the image's peak."""
    strength, decay = ((1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4))[
        severity - 1
    ]

    batch, _, height, width = images.shape
    map_size = _next_power_of_2(max(height, width, images.shape[1]))
    fractal = _plasma_fractal(batch, map_size, decay)[:, :height, :width]
    haze = torch.from_numpy(fractal).to(images.device, images.dtype).unsqueeze(1)

    peak = images.amax(dim=(1, 2, 3), keepdim=True)  # (batch, 1, 1, 1)
    fogged = (images + strength * haze) * peak / (peak + strength)
    return _quantize(fogged)


def brightness(images: torch.Tensor, severity: int) -> torch.Tensor:
    """The HSV value channel lifted by a per-severity constant."""
    lift = (0.1, 0.2, 0.3, 0.4, 0.5)[severity - 1]

    hsv = _rgb_to_hsv(images)
    hsv[:, 2] = (hsv[:, 2] + lift).clamp(0.0, 1.0)
    return _quantize(_hsv_to_rgb(hsv))


def contrast(images: torch.Tensor, severity: int) -> torch.Tensor:
    """Pixels pulled toward their per-channel mean by a per-severity factor."""
    factor = (0.4, 0.3, 0.2, 0.1, 0.05)[severity - 1]

    # Per channel, over the spatial axes only, matching np.mean(x, axis=(0, 1)).
    means = images.mean(dim=(2, 3), keepdim=True)  # (batch, channels, 1, 1)
    flattened = (images - means) * factor + means
    return _quantize(flattened)


def elastic_transform(images: torch.Tensor, severity: int) -> torch.Tensor:
    """Displace every pixel by a smoothed random field, then resample bilinearly."""
    batch, _, height, width = images.shape
    alpha = 250 * (0.05, 0.065, 0.085, 0.1, 0.12)[severity - 1]
    sigma = height * 0.01
    max_shift = height * 0.005

    field = torch.empty(batch, 2, height, width, device=images.device).uniform_(
        -max_shift, max_shift
    )
    field = _separable_blur(field, _gaussian_kernel_1d(sigma, 3.0), "reflect") * alpha

    rows = torch.arange(height, device=images.device, dtype=images.dtype)
    cols = torch.arange(width, device=images.device, dtype=images.dtype)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")

    sample_x = grid_x.unsqueeze(0) + field[:, 0]  # (batch, H, W)
    sample_y = grid_y.unsqueeze(0) + field[:, 1]
    normalized_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
    normalized_y = 2.0 * sample_y / max(height - 1, 1) - 1.0

    grid = torch.stack([normalized_x, normalized_y], dim=-1)  # (batch, H, W, 2)
    warped = F.grid_sample(
        images, grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )
    return _quantize(warped)


def pixelate(images: torch.Tensor, severity: int) -> torch.Tensor:
    """Downsample by area then upsample by nearest, at a per-severity factor."""
    factor = (0.6, 0.5, 0.4, 0.3, 0.25)[severity - 1]

    height, width = images.shape[-2], images.shape[-1]
    small = F.interpolate(
        images, size=(int(height * factor), int(width * factor)), mode="area"
    )
    restored = F.interpolate(small, size=(height, width), mode="nearest")
    return _quantize(restored)


def jpeg_compression(images: torch.Tensor, severity: int) -> torch.Tensor:
    """Round-trip each image through the JPEG encoder at a fixed quality.

    The only corruption with no closed form, so it runs on the CPU through PIL,
    image by image, exactly as the reference does.
    """
    quality = (25, 18, 15, 10, 7)[severity - 1]

    as_bytes = (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()
    decoded = []
    for image in as_bytes.permute(0, 2, 3, 1).numpy():  # (H, W, channels)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        decoded.append(torch.from_numpy(np.array(Image.open(buffer))))

    stacked = torch.stack(decoded).permute(0, 3, 1, 2).float() / 255.0
    return stacked.to(images.device, images.dtype)


# Keyed by the reference's own corruption names, in its own order, minus frost.
CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "glass_blur": glass_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "snow": snow,
    "fog": fog,
    "brightness": brightness,
    "contrast": contrast,
    "elastic_transform": elastic_transform,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compression,
}

DEFAULT_CORRUPTIONS: tuple[str, ...] = tuple(CORRUPTIONS)


def _normalization_buffers(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The dataset statistics shaped to broadcast over (batch, C, H, W)."""
    mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean_tensor, std_tensor


@torch.inference_mode()
def hardness_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    corruptions: tuple[str, ...] = DEFAULT_CORRUPTIONS,
    max_severity: int = MAX_SEVERITY,
    use_bfloat16: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """The set L of Algorithm 1 for every sample, shape (N, len(corruptions)).

    Entry (i, k) is the lowest severity at which corruption k moved sample i's
    prediction away from P_org, or max_severity + 1 when it never did.

    The ascending severity scan with the still-at-sentinel guard is exactly
    Algorithm 1's break: once a sample has recorded a threshold for a corruption,
    a later severity cannot overwrite it. Every severity is still evaluated, which
    is what the released code does and what makes the pass batchable.
    """
    model.eval()
    seed_everything(seed)

    unknown = [name for name in corruptions if name not in CORRUPTIONS]
    if unknown:
        raise KeyError(
            f"unknown corruptions {unknown}, known: {sorted(CORRUPTIONS)}. "
            f"{list(UNAVAILABLE_CORRUPTIONS)} cannot be reproduced without the "
            "imagecorruptions package's bundled image assets."
        )

    batch_thresholds = []
    for images, _ in loader:
        images = images.to(device)  # (batch, C, H, W)
        mean_tensor, std_tensor = _normalization_buffers(
            mean, std, images.device, images.dtype
        )
        pixels = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)

        baseline_probs = forward_probs(model, images, device, use_bfloat16)
        reference_labels = baseline_probs.argmax(dim=1)  # (batch,) = P_org

        thresholds = torch.full(
            (images.size(0), len(corruptions)),
            float(max_severity + 1),
            device=device,
        )
        for column, name in enumerate(corruptions):
            corrupt = CORRUPTIONS[name]
            for severity in range(1, max_severity + 1):
                # Applied to the pristine pixels every time, per Algorithm 1 line
                # 5, NOT to the output of the previous severity. See deviation 2.
                corrupted = corrupt(pixels, severity)
                renormalized = (corrupted - mean_tensor) / std_tensor
                probs = forward_probs(model, renormalized, device, use_bfloat16)

                moved = probs.argmax(dim=1) != reference_labels  # (batch,)
                first = moved & (thresholds[:, column] > max_severity)
                thresholds[first, column] = float(severity)

        batch_thresholds.append(thresholds.cpu())

    if not batch_thresholds:
        return torch.empty(0, len(corruptions))

    all_thresholds = torch.cat(batch_thresholds).float()  # (N, K)
    return all_thresholds


def deviation(thresholds: torch.Tensor) -> torch.Tensor:
    """Dev(L): the population standard deviation across corruptions, shape (N,).

    unbiased=False because the released code calls np.std, whose default ddof is
    0. The choice rescales every score by a constant and so cannot move AUROC, but
    it does move any absolute threshold, including the paper's empirical gamma = 1.
    """
    spread = thresholds.std(dim=1, unbiased=False)  # (N,)
    return spread


def teco_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    corruptions: tuple[str, ...] = DEFAULT_CORRUPTIONS,
    max_severity: int = MAX_SEVERITY,
    use_bfloat16: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """TeCo score per sample, shape (N,), low meaning poisoned.

    Negated at this boundary. Eq. (4) flags an input when TeCo(x) > gamma, so the
    paper's statistic is high for poisoned, the opposite of PSU's convention, and
    returning it unnegated would produce a well-formed, exactly inverted detector.
    """
    thresholds = hardness_thresholds(
        model,
        loader,
        device,
        mean,
        std,
        corruptions,
        max_severity,
        use_bfloat16,
        seed,
    )
    if thresholds.numel() == 0:
        return torch.empty(0)

    scores = -deviation(thresholds).float()  # (N,), low means poisoned
    return scores
