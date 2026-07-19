"""TaCT: source-specific contamination with cover samples (Tang et al., 2021).

Only images from the source classes are poisoned and flipped to the target, and
cover samples (the trigger on non-source images, label kept) stop the trigger from
being learned as a generic target cue. source_classes selects the sources and
cover_rate sets the cover count. train_backdoor reads both from this config.

ASR for a source-specific attack is conventionally measured on source-class images
only. The general AttackSuccessSet measures over all non-target images, so read
the ASR with that in mind or filter the test set to the source classes.
"""

from dataclasses import dataclass

import torch

from poison import Attack


@dataclass(frozen=True)
class TactConfig:
    patch_size: int = 3
    source_classes: tuple[int, ...] = (1,)  # classes the trigger flips to the target
    cover_rate: float = 0.01
    label_mode: str = "all_to_one"


def _patch(patch_size: int) -> torch.Tensor:
    board = torch.zeros(3, patch_size, patch_size)
    for row in range(patch_size):
        for column in range(patch_size):
            board[:, row, column] = 1.0 if (row + column) % 2 == 0 else 0.0
    return board


def build(config: TactConfig, image_size: int, target_label: int) -> Attack:
    patch = _patch(config.patch_size)
    size = config.patch_size

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        stamped = image.clone()
        stamped[:, image_size - size:, image_size - size:] = patch
        return stamped

    return Attack("tact", apply_trigger, config.label_mode, target_label)
