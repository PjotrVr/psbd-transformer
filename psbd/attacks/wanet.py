"""WaNet: a smooth warping of pixel positions (Nguyen and Tran, 2021).

The trigger is a fixed backward-warping field, generated once from a small
control grid and shared across all poisoned images. It changes where pixels are
sampled from rather than their values, which is what makes it hard to see.

The normalization of the control offsets follows the paper's approach at a level
of fidelity sufficient to produce a working attack. Exact match to a specific
benchmark's field would require that benchmark's saved grid.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from psbd.poisoning import Attack


@dataclass(frozen=True)
class WaNetConfig:
    control_grid_size: int = 4
    strength: float = 0.5
    field_seed: int = 0
    label_mode: str = "all_to_one"


def _identity_grid(image_size: int) -> torch.Tensor:
    """The no-warp sampling grid, (1, H, W, 2), in grid_sample's -1 to 1 coordinates.

    The last axis is ordered (x, y), which is why columns come before rows in the
    stack: grid_sample reads the first component as the horizontal coordinate.
    """
    axis = torch.linspace(-1.0, 1.0, image_size)  # (image_size,)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")

    grid = torch.stack((columns, rows), dim=2).unsqueeze(0)  # (1, H, W, 2)
    return grid


def _warping_grid(
    image_size: int, control_grid_size: int, strength: float, seed: int
) -> torch.Tensor:
    """The fixed backward-warping field, (1, H, W, 2), clamped to the sampling range.

    A small control grid of random offsets is upsampled bicubically to full
    resolution, which is what makes the field smooth rather than noisy, and smooth
    is what makes the warp invisible.
    """
    generator = torch.Generator().manual_seed(seed)
    control = (
        torch.rand(1, 2, control_grid_size, control_grid_size, generator=generator)
        * 2.0
        - 1.0
    )  # (1, 2, control_grid_size, control_grid_size)

    # Normalizing by the mean absolute offset makes the field's magnitude
    # independent of the draw, so strength alone controls how far pixels move.
    control = control / control.abs().mean()

    field = F.interpolate(control, size=image_size, mode="bicubic", align_corners=True)
    field = field.permute(0, 2, 3, 1)  # (1, H, W, 2)

    grid = _identity_grid(image_size) + strength * field / image_size
    clamped = grid.clamp(-1.0, 1.0)
    return clamped


def build(config: WaNetConfig, image_size: int, target_label: int) -> Attack:
    """The WaNet attack record for one image size and target label."""
    grid = _warping_grid(
        image_size, config.control_grid_size, config.strength, config.field_seed
    )  # (1, H, W, 2)

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        warped = F.grid_sample(
            image.unsqueeze(0), grid, align_corners=True, padding_mode="border"
        )  # (1, C, H, W)
        stamped = warped.squeeze(0)  # (C, H, W)
        return stamped

    attack = Attack("wanet", apply_trigger, config.label_mode, target_label)
    return attack
