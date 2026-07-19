"""BppAttack: a bit-depth-reduction trigger (Wang et al., 2022).

The trigger reduces the color depth of the image. With Floyd-Steinberg dithering
the change is hard to see, which is the paper's point. The full attack also uses
contrastive adversarial training to make the model sensitive to the quantization,
which is a training-loop change not included here, so this file provides the
trigger and the standard-training variant.

Dithering runs a sequential per-pixel loop, so it is off by default to keep dataset
building fast, especially on Tiny ImageNet where it runs every epoch. Turn it on
for the faithful imperceptible trigger on small images.
"""

from dataclasses import dataclass

import torch

from poison import Attack


@dataclass(frozen=True)
class BppConfig:
    bit_depth: int = 3  # bits per channel, giving 2 to the power bit_depth levels
    dither: bool = False
    label_mode: str = "all_to_one"


def _quantize(image: torch.Tensor, levels: int) -> torch.Tensor:
    # original: reduce each channel to 'levels' evenly spaced values
    # simplified: round each pixel to the nearest allowed level
    return torch.round(image * (levels - 1)) / (levels - 1)


def _floyd_steinberg(channel: torch.Tensor, levels: int) -> torch.Tensor:
    """Error-diffusion dithering, which hides the quantization banding."""
    height, width = channel.shape
    out = channel.clone()
    for row in range(height):
        for column in range(width):
            old_value = out[row, column].item()
            new_value = round(old_value * (levels - 1)) / (levels - 1)
            out[row, column] = new_value
            error = old_value - new_value
            if column + 1 < width:
                out[row, column + 1] += error * 7 / 16
            if row + 1 < height:
                if column - 1 >= 0:
                    out[row + 1, column - 1] += error * 3 / 16
                out[row + 1, column] += error * 5 / 16
                if column + 1 < width:
                    out[row + 1, column + 1] += error * 1 / 16
    return out.clamp(0.0, 1.0)


def build(config: BppConfig, image_size: int, target_label: int) -> Attack:
    levels = 2**config.bit_depth
    dither = config.dither

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        if not dither:
            return _quantize(image, levels)
        channels = [_floyd_steinberg(image[c], levels) for c in range(image.shape[0])]
        return torch.stack(channels)

    return Attack("bpp", apply_trigger, config.label_mode, target_label)
