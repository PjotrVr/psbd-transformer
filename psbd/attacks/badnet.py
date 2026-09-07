"""BadNet: a fixed checkerboard patch in the bottom-right corner (Gu et al., 2017).

Static, pixel-space, dirty-label. The default 3 by 3 patch matches BackdoorBench
on 32 by 32 inputs.
"""

from dataclasses import dataclass

import torch

from psbd.poisoning import Attack

from ._patterns import checkerboard_patch


@dataclass(frozen=True)
class BadNetConfig:
    patch_size: int = 3
    label_mode: str = "all_to_one"  # use "all_to_all" for the BadNets-A2A variant


def build(config: BadNetConfig, image_size: int, target_label: int) -> Attack:
    """The BadNet attack record for one image size and target label."""
    patch = checkerboard_patch(config.patch_size)  # (3, patch_size, patch_size)
    size = config.patch_size

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        stamped = image.clone()  # (C, H, W)
        stamped[:, image_size - size :, image_size - size :] = patch
        return stamped

    attack = Attack("badnet", apply_trigger, config.label_mode, target_label)
    return attack
