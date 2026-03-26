from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_ARGS = {
    "target_label": 0,
    "cover_rate": 0.0,
    "noise_grid_size": 4,
    "warp_strength": 0.5,
    "grid_rescale": 1.0,
}

ARGUMENTS = [
    {
        "flags": ["--target_label"],
        "type": int,
        "default": DEFAULT_ARGS["target_label"],
        "help": "Target label assigned to poisoned samples.",
    },
    {
        "flags": ["--cover_rate"],
        "type": float,
        "default": DEFAULT_ARGS["cover_rate"],
        "help": "Ratio of cover samples (kept for parity with reference implementation).",
    },
    {
        "flags": ["--noise_grid_size"],
        "type": int,
        "default": DEFAULT_ARGS["noise_grid_size"],
        "help": "Control grid resolution for WaNet spatial warp.",
    },
    {
        "flags": ["--warp_strength"],
        "type": float,
        "default": DEFAULT_ARGS["warp_strength"],
        "help": "Warp intensity for WaNet.",
    },
    {
        "flags": ["--grid_rescale"],
        "type": float,
        "default": DEFAULT_ARGS["grid_rescale"],
        "help": "Rescale factor applied to final sampling grid.",
    },
]


class WaNetTransform:
    def __init__(
        self,
        target_label: int,
        image_height: int,
        image_width: int,
        cover_rate: float = 0.0,
        noise_grid_size: int = 4,
        warp_strength: float = 0.5,
        grid_rescale: float = 1.0,
    ):
        self.target_label = int(target_label)
        self.cover_rate = float(cover_rate)

        noise = torch.rand(1, 2, noise_grid_size, noise_grid_size) * 2 - 1
        noise = F.interpolate(
            noise,
            size=(image_height, image_width),
            mode="bicubic",
            align_corners=True,
        )
        noise = noise.permute(0, 2, 3, 1)

        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, image_height),
            torch.linspace(-1, 1, image_width),
            indexing="ij",
        )
        base_grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)
        self.grid = torch.clamp(
            (base_grid + noise * float(warp_strength) / float(image_height))
            * float(grid_rescale),
            -1,
            1,
        )

    def transform(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        poisoned_imgs = imgs.clone()
        poisoned_labels = labels.clone()

        grid = self.grid.to(poisoned_imgs.device).repeat(len(poisoned_imgs), 1, 1, 1)
        poisoned_imgs = F.grid_sample(poisoned_imgs, grid, align_corners=True)
        poisoned_labels.fill_(self.target_label)
        return poisoned_imgs, poisoned_labels


def add_parser_arguments(parser):
    for spec in ARGUMENTS:
        flags = spec["flags"]
        if not isinstance(flags, list):
            continue
        parser.add_argument(
            *flags,
            type=spec["type"],
            default=spec["default"],
            help=spec["help"],
        )


def namespace_to_config(args):
    return {
        name: getattr(args, name, default) for name, default in DEFAULT_ARGS.items()
    }


def build_transform(
    config, image_shape: tuple[int, int] | None = None
) -> WaNetTransform:
    if image_shape is None:
        raise ValueError("image_shape is required for WaNet transform")

    image_height, image_width = image_shape
    return WaNetTransform(
        target_label=int(config["target_label"]),
        image_height=int(image_height),
        image_width=int(image_width),
        cover_rate=float(config["cover_rate"]),
        noise_grid_size=int(config["noise_grid_size"]),
        warp_strength=float(config["warp_strength"]),
        grid_rescale=float(config["grid_rescale"]),
    )
