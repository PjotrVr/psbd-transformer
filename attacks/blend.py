from __future__ import annotations

from PIL import Image
import torch
import torchvision.transforms.v2 as transforms

DEFAULT_ARGS = {
    "target_label": 0,
    "alpha": 0.2,
    "trigger_path": "./triggers/badnet_patch_32.png",
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
        "help": "Blend strength between clean image and trigger.",
    },
    {
        "flags": ["--trigger_path"],
        "type": str,
        "default": DEFAULT_ARGS["trigger_path"],
        "help": "Path to RGB trigger image.",
    },
]


class BlendTransform:
    def __init__(self, target_label: int, trigger: torch.Tensor, alpha: float = 0.2):
        self.target_label = int(target_label)
        self.trigger = trigger
        self.alpha = float(alpha)

    def transform(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        poisoned_imgs = imgs.clone()
        poisoned_labels = labels.clone()

        trigger = self.trigger.to(poisoned_imgs.device)
        poisoned_imgs = (1.0 - self.alpha) * poisoned_imgs + self.alpha * trigger
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


def build_transform(
    config, image_shape: tuple[int, int] | None = None
) -> BlendTransform:
    del image_shape
    to_tensor = transforms.ToTensor()
    trigger = to_tensor(Image.open(str(config["trigger_path"])).convert("RGB"))

    return BlendTransform(
        target_label=int(config["target_label"]),
        trigger=trigger,
        alpha=float(config["alpha"]),
    )
