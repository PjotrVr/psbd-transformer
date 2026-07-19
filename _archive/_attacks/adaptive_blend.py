from __future__ import annotations

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image

DEFAULT_ARGS = {
    "target_label": 0,
    "alpha": 0.2,
    "cover_rate": 0.01,
    "trigger_path": "./triggers/badnet_patch_32.png",
    "pieces": 16,
    "mask_rate": 0.5,
}

ARGUMENTS = [
    {
        "flags": ["--target_label"],
        "type": int,
        "default": DEFAULT_ARGS["target_label"],
        "help": "Target label assigned to poisoned samples.",
    },
    {
        "flags": ["--alpha"],
        "type": float,
        "default": DEFAULT_ARGS["alpha"],
        "help": "Blend strength for adaptive trigger injection.",
    },
    {
        "flags": ["--cover_rate"],
        "type": float,
        "default": DEFAULT_ARGS["cover_rate"],
        "help": "Ratio of cover samples (kept for parity with reference implementation).",
    },
    {
        "flags": ["--trigger_path"],
        "type": str,
        "default": DEFAULT_ARGS["trigger_path"],
        "help": "Path to RGB trigger image.",
    },
    {
        "flags": ["--pieces"],
        "type": int,
        "default": DEFAULT_ARGS["pieces"],
        "help": "Number of grid pieces used to build adaptive mask (must be square).",
    },
    {
        "flags": ["--mask_rate"],
        "type": float,
        "default": DEFAULT_ARGS["mask_rate"],
        "help": "Ratio of grid pieces masked out from trigger application.",
    },
]


def _square_root_if_square(pieces: int) -> int:
    side = int(round(float(pieces) ** 0.5))
    if side * side != int(pieces):
        raise ValueError("pieces must be a perfect square")
    return side


def _build_adaptive_mask(h: int, w: int, pieces: int, mask_rate: float) -> torch.Tensor:
    side = _square_root_if_square(pieces)
    if h % side != 0 or w % side != 0:
        raise ValueError("image height and width must be divisible by sqrt(pieces)")

    masked_pieces = int(round(float(mask_rate) * float(pieces)))
    masked_pieces = max(0, min(int(pieces), masked_pieces))

    mask = torch.ones((h, w), dtype=torch.float32)
    if masked_pieces == 0:
        return mask.unsqueeze(0)

    piece_h = h // side
    piece_w = w // side
    idx = torch.randperm(int(pieces))[:masked_pieces]

    for flat_id in idx.tolist():
        row = flat_id // side
        col = flat_id % side
        y0 = row * piece_h
        y1 = (row + 1) * piece_h
        x0 = col * piece_w
        x1 = (col + 1) * piece_w
        mask[y0:y1, x0:x1] = 0.0

    return mask.unsqueeze(0)


class AdaptiveBlendTransform:
    def __init__(
        self,
        target_label: int,
        trigger: torch.Tensor,
        mask: torch.Tensor,
        alpha: float = 0.2,
        cover_rate: float = 0.01,
    ):
        self.target_label = int(target_label)
        self.trigger = trigger
        self.mask = mask
        self.alpha = float(alpha)
        self.cover_rate = float(cover_rate)

    def transform(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        poisoned_imgs = imgs.clone()
        poisoned_labels = labels.clone()

        trigger = self.trigger.to(poisoned_imgs.device)
        mask = self.mask.to(poisoned_imgs.device)
        poisoned_imgs = poisoned_imgs + self.alpha * mask * (trigger - poisoned_imgs)
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
) -> AdaptiveBlendTransform:
    if image_shape is None:
        raise ValueError("image_shape is required for adaptive blend transform")

    h, w = image_shape
    to_tensor = transforms.ToTensor()
    trigger = to_tensor(Image.open(str(config["trigger_path"])).convert("RGB"))
    mask = _build_adaptive_mask(
        h=int(h),
        w=int(w),
        pieces=int(config["pieces"]),
        mask_rate=float(config["mask_rate"]),
    )

    return AdaptiveBlendTransform(
        target_label=int(config["target_label"]),
        trigger=trigger,
        mask=mask,
        alpha=float(config["alpha"]),
        cover_rate=float(config["cover_rate"]),
    )
