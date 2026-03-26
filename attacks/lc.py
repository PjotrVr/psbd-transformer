from __future__ import annotations

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image

DEFAULT_ARGS = {
    "target_label": 0,
    "alpha": 1.0,
    "trigger_path": "./triggers/badnet_patch_32.png",
    "mask_path": "./triggers/mask_badnet_patch_32.png",
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
        "help": "Blend factor for trigger injection.",
    },
    {
        "flags": ["--trigger_path"],
        "type": str,
        "default": DEFAULT_ARGS["trigger_path"],
        "help": "Path to RGB trigger image.",
    },
    {
        "flags": ["--mask_path"],
        "type": str,
        "default": DEFAULT_ARGS["mask_path"],
        "help": "Path to grayscale trigger mask image.",
    },
]


class LCTransform:
    def __init__(
        self,
        target_label: int,
        trigger: torch.Tensor,
        mask: torch.Tensor,
        alpha: float = 1.0,
    ):
        self.target_label = int(target_label)
        self.trigger = trigger
        self.mask = mask
        self.alpha = float(alpha)

    def transform(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        poisoned_imgs = imgs.clone()
        poisoned_labels = labels.clone()

        mask = self.mask.to(poisoned_imgs.device)
        trigger = self.trigger.to(poisoned_imgs.device)
        poisoned_imgs = poisoned_imgs + self.alpha * mask * (trigger - poisoned_imgs)
        poisoned_labels.fill_(self.target_label)
        return poisoned_imgs, poisoned_labels


def add_parser_arguments(parser) -> None:
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


def build_transform(config, image_shape: tuple[int, int] | None = None) -> LCTransform:
    to_tensor = transforms.ToTensor()
    trigger = to_tensor(Image.open(str(config["trigger_path"])).convert("RGB"))
    mask = to_tensor(Image.open(str(config["mask_path"])).convert("L"))

    return LCTransform(
        target_label=int(config["target_label"]),
        trigger=trigger,
        mask=mask,
        alpha=float(config["alpha"]),
    )
