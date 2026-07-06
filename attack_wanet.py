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

from poison import Attack


@dataclass(frozen=True)
class WaNetConfig:
    control_grid_size: int = 4
    strength: float = 0.5
    field_seed: int = 0
    label_mode: str = "all_to_one"


def _identity_grid(image_size: int) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, image_size)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((columns, rows), dim=2).unsqueeze(0)  # 1, H, W, 2


def _warping_grid(image_size: int, control_grid_size: int, strength: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    control = torch.rand(1, 2, control_grid_size, control_grid_size, generator=generator) * 2.0 - 1.0
    control = control / control.abs().mean()  # normalize the offsets
    field = F.interpolate(control, size=image_size, mode="bicubic", align_corners=True)
    field = field.permute(0, 2, 3, 1)  # 1, H, W, 2
    grid = _identity_grid(image_size) + strength * field / image_size
    return grid.clamp(-1.0, 1.0)


def build(config: WaNetConfig, image_size: int, target_label: int) -> Attack:
    grid = _warping_grid(image_size, config.control_grid_size, config.strength, config.field_seed)

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        warped = F.grid_sample(image.unsqueeze(0), grid, align_corners=True, padding_mode="border")
        return warped.squeeze(0)

    return Attack("wanet", apply_trigger, config.label_mode, target_label)
