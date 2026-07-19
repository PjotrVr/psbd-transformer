"""Label-Consistent backdoor: a corner patch on target-class images (Turner et al., 2019).

Clean-label: only target-class images are poisoned and their label is kept, so a
human inspecting the labels sees nothing wrong. The trigger is a small pattern
placed in the image corners.

For full strength the base target images are first perturbed adversarially or by
GAN interpolation so their natural features become unreliable and the model must
lean on the trigger. That perturbation is a separate offline step. Supplying the
perturbed bases through attack_generated and using this patch reproduces the full
attack. Using this patch on unperturbed images is the weaker self-contained variant.
"""

from dataclasses import dataclass

import torch

from poison import Attack


@dataclass(frozen=True)
class LabelConsistentConfig:
    patch_size: int = 3
    label_mode: str = "clean_label"


def _corner_pattern(patch_size: int) -> torch.Tensor:
    board = torch.zeros(3, patch_size, patch_size)
    for row in range(patch_size):
        for column in range(patch_size):
            board[:, row, column] = 1.0 if (row + column) % 2 == 0 else 0.0
    return board


def build(config: LabelConsistentConfig, image_size: int, target_label: int) -> Attack:
    patch = _corner_pattern(config.patch_size)
    size = config.patch_size

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        # The label-consistent trigger repeats the patch in all four corners.
        stamped = image.clone()
        stamped[:, :size, :size] = patch
        stamped[:, :size, image_size - size :] = patch
        stamped[:, image_size - size :, :size] = patch
        stamped[:, image_size - size :, image_size - size :] = patch
        return stamped

    return Attack("lc", apply_trigger, config.label_mode, target_label)
