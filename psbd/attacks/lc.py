"""Label-Consistent backdoor: a corner patch on target-class images (Turner et al., 2019).

Clean-label: only target-class images are poisoned and their label is kept, so a
human inspecting the labels sees nothing wrong. The trigger is a small pattern
placed in the image corners.

For full strength the base target images are first perturbed adversarially or by
GAN interpolation so their natural features become unreliable and the model must
lean on the trigger. That perturbation is a separate offline step. Supplying the
perturbed bases through the generated adapter and using this patch reproduces the
full attack. Using this patch on unperturbed images is the weaker self-contained
variant.
"""

from dataclasses import dataclass

import torch

from psbd.poisoning import Attack

from ._patterns import checkerboard_patch


@dataclass(frozen=True)
class LabelConsistentConfig:
    patch_size: int = 3
    label_mode: str = "clean_label"


def build(config: LabelConsistentConfig, image_size: int, target_label: int) -> Attack:
    """The Label-Consistent attack record for one image size and target label."""
    patch = checkerboard_patch(config.patch_size)  # (3, patch_size, patch_size)
    size = config.patch_size

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        # The label-consistent trigger repeats the patch in all 4 corners.
        stamped = image.clone()  # (C, H, W)
        stamped[:, :size, :size] = patch
        stamped[:, :size, image_size - size :] = patch
        stamped[:, image_size - size :, :size] = patch
        stamped[:, image_size - size :, image_size - size :] = patch
        return stamped

    attack = Attack("lc", apply_trigger, config.label_mode, target_label)
    return attack
