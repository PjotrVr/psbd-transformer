from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        residual = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + residual
        return out


class ResNetV2(nn.Module):
    def __init__(self, block, num_blocks, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(block(self.in_planes, planes, current_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet18_v2(num_classes: int = 10, in_channels: int = 3) -> ResNetV2:
    return ResNetV2(
        PreActBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels
    )


class ResNetV2PSBD(ResNetV2):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes: int = 10,
        in_channels: int = 3,
        psbd_dropout_rate: float = 0.0,
        use_inference_dropout: bool = False,
    ):
        super().__init__(
            block=block,
            num_blocks=num_blocks,
            num_classes=num_classes,
            in_channels=in_channels,
        )
        self.psbd_dropout_rate = float(psbd_dropout_rate)
        self.use_inference_dropout = bool(use_inference_dropout)

    def set_inference_dropout(
        self, enabled: bool, dropout_rate: float | None = None
    ) -> None:
        self.use_inference_dropout = bool(enabled)
        if dropout_rate is not None:
            self.psbd_dropout_rate = float(dropout_rate)

    def _drop2d(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x
        if not self.use_inference_dropout or self.psbd_dropout_rate <= 0.0:
            return x
        return F.dropout2d(x, p=self.psbd_dropout_rate, training=True)

    def _drop1d(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x
        if not self.use_inference_dropout or self.psbd_dropout_rate <= 0.0:
            return x
        return F.dropout(x, p=self.psbd_dropout_rate, training=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self._drop2d(out)

        out = self.layer1(out)
        out = self._drop2d(out)

        out = self.layer2(out)
        out = self._drop2d(out)

        out = self.layer3(out)
        out = self._drop2d(out)

        out = self.layer4(out)
        out = self._drop2d(out)

        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self._drop1d(out)
        out = self.fc(out)
        return out

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        block,
        num_blocks,
        num_classes: int = 10,
        in_channels: int = 3,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        cleaned_state = {}
        for key, value in state.items():
            if key.startswith("model."):
                cleaned_state[key[len("model.") :]] = value
            else:
                cleaned_state[key] = value

        model = cls(
            block=block,
            num_blocks=num_blocks,
            num_classes=num_classes,
            in_channels=in_channels,
        )
        model.load_state_dict(cleaned_state, strict=strict)
        return model


def resnet18_v2_psbd(num_classes: int = 10, in_channels: int = 3) -> ResNetV2PSBD:
    return ResNetV2PSBD(
        block=PreActBlock,
        num_blocks=[2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels,
    )


class ResNet18V2PSBD(ResNetV2PSBD):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        psbd_dropout_rate: float = 0.0,
        use_inference_dropout: bool = False,
    ):
        super().__init__(
            block=PreActBlock,
            num_blocks=[2, 2, 2, 2],
            num_classes=num_classes,
            in_channels=in_channels,
            psbd_dropout_rate=psbd_dropout_rate,
            use_inference_dropout=use_inference_dropout,
        )
