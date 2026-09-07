"""Blend: alpha-blend a fixed pattern over the whole image (Chen et al., 2017).

Static, pixel-space, dirty-label. The original paper uses a Hello Kitty image as
the pattern. A seeded random pattern is used here to keep the package
self-contained and free of any bundled image, and it can be swapped for a loaded
pattern.
"""

from dataclasses import dataclass

import torch

from psbd.poisoning import Attack

from ._patterns import seeded_random_pattern


@dataclass(frozen=True)
class BlendConfig:
    alpha: float = 0.2  # blend ratio, 0.1 to 0.2 in the literature
    pattern_seed: int = 0
    label_mode: str = "all_to_one"


def build(config: BlendConfig, image_size: int, target_label: int) -> Attack:
    """The Blend attack record for one image size and target label."""
    pattern = seeded_random_pattern(image_size, config.pattern_seed)  # (3, S, S)
    alpha = config.alpha

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        # original: x_poisoned = (1 - alpha) * x + alpha * pattern
        blended = (1.0 - alpha) * image + alpha * pattern  # (C, H, W)
        return blended

    attack = Attack("blend", apply_trigger, config.label_mode, target_label)
    return attack
