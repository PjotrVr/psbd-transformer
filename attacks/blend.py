"""Blend: alpha-blend a fixed pattern over the whole image (Chen et al., 2017).

Static, pixel-space, dirty-label. The original paper uses a Hello Kitty image as
the pattern. A seeded random pattern is used here to keep the file self-contained
and free of any bundled image, and it can be swapped for a loaded pattern.
"""

from dataclasses import dataclass

import torch

from poison import Attack


@dataclass(frozen=True)
class BlendConfig:
    alpha: float = 0.2  # blend ratio, 0.1 to 0.2 in the literature
    pattern_seed: int = 0
    label_mode: str = "all_to_one"


def _random_pattern(image_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, image_size, image_size, generator=generator)


def build(config: BlendConfig, image_size: int, target_label: int) -> Attack:
    pattern = _random_pattern(image_size, config.pattern_seed)
    alpha = config.alpha

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        # original: x_poisoned = (1 - alpha) * x + alpha * pattern
        return (1.0 - alpha) * image + alpha * pattern

    return Attack("blend", apply_trigger, config.label_mode, target_label)
