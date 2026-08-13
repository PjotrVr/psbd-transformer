"""BadNet: a fixed checkerboard patch in the bottom-right corner (Gu et al., 2017).

Static, pixel-space, dirty-label. The default 3 by 3 patch matches BackdoorBench
on 32 by 32 inputs.
"""

from dataclasses import dataclass

import torch

from poison import Attack


@dataclass(frozen=True)
class BadNetConfig:
    patch_size: int = 3
    label_mode: str = "all_to_one"  # use "all_to_all" for the BadNets-A2A variant


def _checkerboard(patch_size: int) -> torch.Tensor:
    board = torch.zeros(3, patch_size, patch_size)
    for row in range(patch_size):
        for column in range(patch_size):
            board[:, row, column] = 1.0 if (row + column) % 2 == 0 else 0.0
    return board


def build(config: BadNetConfig, image_size: int, target_label: int) -> Attack:
    patch = _checkerboard(config.patch_size)
    size = config.patch_size

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        stamped = image.clone()
        stamped[:, image_size - size :, image_size - size :] = patch
        return stamped

    return Attack("badnet", apply_trigger, config.label_mode, target_label)
