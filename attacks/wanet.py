from __future__ import annotations


import torch
import torch.nn.functional as F

DEFAULT_ARGS = {
    "target_label": 0,
    "k": 4,
    "s": 0.5,
}

ARGUMENTS = [
    {
        "flags": ["--target_label"],
        "type": int,
        "default": DEFAULT_ARGS["target_label"],
        "help": "Target label assigned to poisoned samples.",
    },
    {
        "flags": ["--k"],
        "type": int,
        "default": DEFAULT_ARGS["k"],
        "help": "Control grid resolution for WaNet spatial warp.",
    },
    {
        "flags": ["--s"],
        "type": float,
        "default": DEFAULT_ARGS["s"],
        "help": "Warp intensity for WaNet.",
    },
]


class WaNetTransform:
    def __init__(self, target_label: int, h: int, w: int, k: int = 4, s: float = 0.5):
        self.target_label = int(target_label)

        noise = torch.rand(1, 2, k, k) * 2 - 1
        noise = F.interpolate(noise, size=(h, w), mode="bicubic", align_corners=True)
        noise = noise.permute(0, 2, 3, 1)

        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing="ij",
        )
        base_grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)
        self.grid = torch.clamp(base_grid + noise * float(s) / float(h), -1, 1)

    def transform(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        poisoned_imgs = imgs.clone()
        poisoned_labels = labels.clone()

        grid = self.grid.to(poisoned_imgs.device).repeat(len(poisoned_imgs), 1, 1, 1)
        poisoned_imgs = F.grid_sample(poisoned_imgs, grid, align_corners=True)
        poisoned_labels.fill_(self.target_label)
        return poisoned_imgs, poisoned_labels


def add_parser_arguments(parser) -> None:
    for spec in ARGUMENTS:
        parser.add_argument(
            *spec["flags"],
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

    h, w = image_shape
    return WaNetTransform(
        target_label=int(config["target_label"]),
        h=int(h),
        w=int(w),
        k=int(config["k"]),
        s=float(config["s"]),
    )
