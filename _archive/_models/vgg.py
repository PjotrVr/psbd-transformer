from __future__ import annotations

import torch
import torch.nn as nn

CFG = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(
        self,
        name: str,
        num_classes: int = 10,
        in_channels: int = 3,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.features = self._make_layers(
            CFG[name], in_channels=in_channels, batch_norm=batch_norm
        )
        self.classifier = nn.Linear(512, num_classes)

    def _make_layers(self, cfg, in_channels: int, batch_norm: bool):
        layers = []
        for value in cfg:
            if value == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue

            conv = nn.Conv2d(in_channels, value, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(value), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv, nn.ReLU(inplace=True)])
            in_channels = value

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


def vgg11(num_classes: int = 10, in_channels: int = 3, batch_norm: bool = True) -> VGG:
    return VGG(
        "VGG11", num_classes=num_classes, in_channels=in_channels, batch_norm=batch_norm
    )


def vgg13(num_classes: int = 10, in_channels: int = 3, batch_norm: bool = True) -> VGG:
    return VGG(
        "VGG13", num_classes=num_classes, in_channels=in_channels, batch_norm=batch_norm
    )


def vgg16(num_classes: int = 10, in_channels: int = 3, batch_norm: bool = True) -> VGG:
    return VGG(
        "VGG16", num_classes=num_classes, in_channels=in_channels, batch_norm=batch_norm
    )


def vgg19(num_classes: int = 10, in_channels: int = 3, batch_norm: bool = True) -> VGG:
    return VGG(
        "VGG19", num_classes=num_classes, in_channels=in_channels, batch_norm=batch_norm
    )
