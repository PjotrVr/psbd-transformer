"""WaNet: a smooth warping of pixel positions (Nguyen and Tran, 2021).

The trigger is a fixed backward-warping field, generated once from a small control
grid and shared by every poisoned image. It moves where pixels are sampled from
rather than changing their values, which is what makes it hard to see.

The control offsets are normalized the way the paper describes. Reproducing a
specific benchmark's field exactly would need that benchmark's saved grid.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from attacks.poisoning import Attack


@dataclass(frozen=True)
class WaNetConfig:
    control_grid_size: int = 4
    strength: float = 0.5
    field_seed: int = 0
    label_mode: str = "all_to_one"
    # Noise mode. The paper trains on cover samples warped by a RANDOM field with
    # their label kept, at twice the poisoning rate, so the network cannot learn
    # "warped" as the cue and has to learn this specific warp. Without it the attack
    # is materially easier to detect. BackdoorBench calls this cross_ratio 2.
    cover_rate: float = 0.0


def _identity_grid(image_size: int) -> torch.Tensor:
    """The (1, H, W, 2) sampling grid that leaves an image unchanged."""
    axis = torch.linspace(-1.0, 1.0, image_size)  # (H,)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")  # (H, W) each

    # grid_sample reads (x, y) pairs, so the column coordinate comes first.
    grid = torch.stack((columns, rows), dim=2).unsqueeze(0)  # (1, H, W, 2)
    return grid


def _warping_grid(
    image_size: int, control_grid_size: int, strength: float, seed: int
) -> torch.Tensor:
    """The trigger's sampling grid, (1, H, W, 2).

    The identity grid plus a smooth field upsampled from the control grid.
    """
    generator = torch.Generator().manual_seed(seed)
    control = (
        torch.rand(1, 2, control_grid_size, control_grid_size, generator=generator)
        * 2.0
        - 1.0
    )  # (1, 2, control_grid_size, control_grid_size), in -1 to 1
    # The paper normalizes the control offsets by their mean magnitude, so
    # strength alone sets the size of the warp whatever the draw happened to be.
    control = control / control.abs().mean()
    field = F.interpolate(
        control, size=image_size, mode="bicubic", align_corners=True
    )  # (1, 2, H, W)
    field = field.permute(0, 2, 3, 1)  # (1, H, W, 2)
    grid = _identity_grid(image_size) + strength * field / image_size  # (1, H, W, 2)

    clamped = grid.clamp(-1.0, 1.0)
    return clamped


def build(config: WaNetConfig, image_size: int, target_label: int) -> Attack:
    """WaNet built for this image size and target label."""
    grid = _warping_grid(
        image_size, config.control_grid_size, config.strength, config.field_seed
    )

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        warped = F.grid_sample(
            image.unsqueeze(0), grid, align_corners=True, padding_mode="border"
        )  # (1, C, H, W)

        stamped = warped.squeeze(0)  # (C, H, W)
        return stamped

    def apply_cover(image: torch.Tensor, index: int) -> torch.Tensor:
        """Noise mode: the image warped by a random field instead of the trigger.

        original form
            grid_noise = grid_temps + ins / image_size,  ins ~ U(-1, 1)

        grid_temps is the trigger's warping grid, ins the per-sample noise field
        and image_size the width in pixels. The offset is seeded from the sample
        index so a run is reproducible.
        """
        size = image.shape[-1]
        generator = torch.Generator().manual_seed(config.field_seed * 1_000_003 + index)
        offsets = (
            torch.rand(1, size, size, 2, generator=generator) * 2.0 - 1.0
        ) / size  # (1, H, W, 2)
        noisy = (grid + offsets).clamp(-1.0, 1.0)  # (1, H, W, 2)
        warped = F.grid_sample(
            image.unsqueeze(0), noisy, align_corners=True, padding_mode="border"
        )  # (1, C, H, W)

        covered = warped.squeeze(0)  # (C, H, W)
        return covered

    attack = Attack(
        "wanet",
        apply_trigger,
        config.label_mode,
        target_label,
        apply_cover=apply_cover,
    )
    return attack
