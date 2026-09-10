"""TaCT: source-specific contamination with cover samples (Tang et al., 2021).

Only images from the source classes are poisoned and flipped to the target. Cover
samples, the trigger on non-source images with their label kept, stop the trigger
from being learned as a generic target cue. source_classes selects the sources and
cover_rate sets the cover count.

ASR is measured on source-class images only, since those are the only images the
attack claims to flip. source_classes travels on the Attack so AttackSuccessSet
restricts the eval set itself rather than trusting the caller to.
"""

from dataclasses import dataclass

import torch

from attacks.poisoning import Attack

from .patterns import checkerboard_patch


@dataclass(frozen=True)
class TactConfig:
    patch_size: int = 3
    source_classes: tuple[int, ...] = (1,)  # classes the trigger flips to the target
    cover_rate: float = 0.01
    label_mode: str = "all_to_one"


def build(config: TactConfig, image_size: int, target_label: int) -> Attack:
    """TaCT built for this image size and target label."""
    patch = checkerboard_patch(config.patch_size)  # (3, patch_size, patch_size)
    size = config.patch_size

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        stamped = image.clone()  # (C, H, W)
        stamped[:, image_size - size :, image_size - size :] = patch
        return stamped

    return Attack(
        "tact",
        apply_trigger,
        config.label_mode,
        target_label,
        source_classes=config.source_classes,
    )
