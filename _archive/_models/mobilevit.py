from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
    ):
        super().__init__()
        hidden_channels = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if hidden_channels != in_channels:
            layers.append(
                ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1)
            )

        layers.append(
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                groups=hidden_channels,
            )
        )
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        depth: int,
        patch_size: tuple[int, int] = (2, 2),
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_h, self.patch_w = patch_size

        self.local_rep = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3, stride=1),
            ConvBNAct(in_channels, transformer_dim, kernel_size=1, stride=1),
        )

        self.transformer = nn.Sequential(
            *[
                TransformerEncoder(
                    dim=transformer_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.project = ConvBNAct(transformer_dim, in_channels, kernel_size=1, stride=1)
        self.fusion = ConvBNAct(in_channels * 2, in_channels, kernel_size=3, stride=1)

    def _unfold_patches(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        batch_size, channels, height, width = x.shape

        new_h = ((height + self.patch_h - 1) // self.patch_h) * self.patch_h
        new_w = ((width + self.patch_w - 1) // self.patch_w) * self.patch_w

        if new_h != height or new_w != width:
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

        num_ph = new_h // self.patch_h
        num_pw = new_w // self.patch_w

        x = x.view(batch_size, channels, num_ph, self.patch_h, num_pw, self.patch_w)
        x = x.permute(0, 3, 5, 2, 4, 1).contiguous()
        x = x.view(batch_size * self.patch_h * self.patch_w, num_ph * num_pw, channels)

        info = (height, width, new_h, new_w)
        return x, info

    def _fold_patches(
        self, x: torch.Tensor, info: tuple[int, int, int, int]
    ) -> torch.Tensor:
        height, width, new_h, new_w = info
        num_ph = new_h // self.patch_h
        num_pw = new_w // self.patch_w

        batch_size = x.shape[0] // (self.patch_h * self.patch_w)
        channels = x.shape[-1]

        x = x.view(batch_size, self.patch_h, self.patch_w, num_ph, num_pw, channels)
        x = x.permute(0, 5, 3, 1, 4, 2).contiguous()
        x = x.view(batch_size, channels, new_h, new_w)

        if new_h != height or new_w != width:
            x = F.interpolate(
                x, size=(height, width), mode="bilinear", align_corners=False
            )

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.local_rep(x)
        x, info = self._unfold_patches(x)
        x = self.transformer(x)
        x = self._fold_patches(x, info)
        x = self.project(x)

        x = torch.cat([residual, x], dim=1)
        x = self.fusion(x)
        return x


class MobileViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        base_channels: int = 16,
    ):
        super().__init__()

        self.stem = ConvBNAct(in_channels, base_channels, kernel_size=3, stride=1)

        self.layer1 = nn.Sequential(
            InvertedResidual(base_channels, 32, stride=1, expand_ratio=2.0),
            InvertedResidual(32, 48, stride=2, expand_ratio=2.0),
        )

        self.layer2 = nn.Sequential(
            MobileViTBlock(
                in_channels=48,
                transformer_dim=64,
                depth=2,
                patch_size=(2, 2),
                num_heads=4,
                mlp_ratio=2.0,
            ),
            InvertedResidual(48, 64, stride=2, expand_ratio=2.0),
        )

        self.layer3 = nn.Sequential(
            MobileViTBlock(
                in_channels=64,
                transformer_dim=80,
                depth=2,
                patch_size=(2, 2),
                num_heads=4,
                mlp_ratio=2.0,
            ),
            InvertedResidual(64, 80, stride=2, expand_ratio=2.0),
        )

        self.layer4 = MobileViTBlock(
            in_channels=80,
            transformer_dim=96,
            depth=2,
            patch_size=(2, 2),
            num_heads=4,
            mlp_ratio=2.0,
        )

        self.head = nn.Sequential(
            ConvBNAct(80, 160, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(160, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x


def mobilevit_tiny(num_classes: int = 10, in_channels: int = 3) -> MobileViT:
    return MobileViT(num_classes=num_classes, in_channels=in_channels, base_channels=16)
