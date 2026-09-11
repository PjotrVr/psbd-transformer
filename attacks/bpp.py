"""BppAttack: a bit-depth-reduction trigger (Wang et al., 2022).

The trigger reduces the colour depth of the image, and with Floyd-Steinberg
dithering the change is hard to see. The paper also trains with a contrastive
adversarial loss to sharpen the model's sensitivity to the quantization. That is a
training-loop change not included here, so this is the standard-training variant.

Dithering is a sequential per-pixel loop, so it is off by default to keep dataset
building fast on Tiny ImageNet, where the trigger is applied every epoch.
"""

from dataclasses import dataclass

import torch

from attacks.poisoning import Attack


@dataclass(frozen=True)
class BppConfig:
    bit_depth: int = 3  # bits per channel, giving 2 to the power bit_depth levels
    dither: bool = False
    label_mode: str = "all_to_one"
    # Negative samples: images quantized to a DIFFERENT depth with their label kept,
    # so the network cannot learn "quantized" as the cue and has to learn this
    # specific depth. BackdoorBench calls this neg_ratio and sets it to 0.1. Without
    # it the trigger is a generic quantization artefact and easier to detect.
    cover_rate: float = 0.0


def _quantize(image: torch.Tensor, levels: int) -> torch.Tensor:
    """Each pixel rounded to the nearest of levels evenly spaced values.

    image is (C, H, W) in 0 to 1 and the result has the same shape.
    """
    quantized = torch.round(image * (levels - 1)) / (levels - 1)
    return quantized


def _floyd_steinberg(channel: torch.Tensor, levels: int) -> torch.Tensor:
    """Error-diffusion dithering, which hides the quantization banding.

    channel is a single (H, W) plane in 0 to 1 and the result has the same shape.
    """
    height, width = channel.shape
    out = channel.clone()  # (H, W)
    for row in range(height):
        for column in range(width):
            old_value = out[row, column].item()
            new_value = round(old_value * (levels - 1)) / (levels - 1)
            out[row, column] = new_value

            # The rounding error is pushed onto the 4 unvisited neighbours in
            # Floyd and Steinberg's 7, 3, 5 and 1 sixteenths, so it is carried
            # forward rather than lost.
            error = old_value - new_value
            if column + 1 < width:
                out[row, column + 1] += error * 7 / 16
            if row + 1 < height:
                if column - 1 >= 0:
                    out[row + 1, column - 1] += error * 3 / 16
                out[row + 1, column] += error * 5 / 16
                if column + 1 < width:
                    out[row + 1, column + 1] += error * 1 / 16

    dithered = out.clamp(0.0, 1.0)  # (H, W)
    return dithered


def build(config: BppConfig, image_size: int, target_label: int) -> Attack:
    """Bpp built for this image size and target label."""
    levels = 2**config.bit_depth
    dither = config.dither

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        if not dither:
            quantized = _quantize(image, levels)  # (C, H, W)
            return quantized

        channels = [_floyd_steinberg(image[c], levels) for c in range(image.shape[0])]
        dithered = torch.stack(channels)  # (C, H, W)
        return dithered

    def apply_cover(image: torch.Tensor, index: int) -> torch.Tensor:
        """A negative sample: quantized to some other depth, label kept.

        The depth is seeded from the sample index and excludes the trigger's own, so
        a run is reproducible and no negative sample accidentally carries the trigger.
        """
        generator = torch.Generator().manual_seed(index)
        choices = [d for d in (1, 2, 4, 5, 6) if d != config.bit_depth]
        depth = choices[int(torch.randint(len(choices), (1,), generator=generator))]

        negative = _quantize(image, 2**depth)  # (C, H, W)
        return negative

    attack = Attack(
        "bpp",
        apply_trigger,
        config.label_mode,
        target_label,
        apply_cover=apply_cover,
    )
    return attack
