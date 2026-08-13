from __future__ import annotations

import torch
import torch.nn as nn


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # x: [B, H, W, C]
    batch_size, height, width, channels = x.shape
    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size * window_size, channels)
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int, batch_size: int
) -> torch.Tensor:
    channels = windows.shape[-1]
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(batch_size, height, width, channels)
    )
    return x


class SwinMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, length, channels = x.shape
        x_residual = x

        x = self.norm1(x).view(batch_size, height, width, channels)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size)
        attended, _ = self.attn(windows, windows, windows, need_weights=False)
        x = window_reverse(attended, self.window_size, height, width, batch_size)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(batch_size, length, channels)
        x = x_residual + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, height: int, width: int):
        batch_size, _, channels = x.shape
        x = x.view(batch_size, height, width, channels)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)

        new_height, new_width = height // 2, width // 2
        x = x.view(batch_size, new_height * new_width, 4 * channels)
        x = self.norm(x)
        x = self.reduction(x)
        return x, new_height, new_width


class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        blocks = []
        for index in range(depth):
            shift_size = 0 if index % 2 == 0 else window_size // 2
            blocks.append(
                SwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, height: int, width: int):
        for block in self.blocks:
            x = block(x, height, width)
        return x, height, width


class SwinTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 96,
        depths: tuple[int, int] = (2, 2),
        num_heads: tuple[int, int] = (3, 6),
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_drop = nn.Dropout(dropout)

        self.stage1 = SwinStage(
            dim=embed_dim,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.merge1 = PatchMerging(embed_dim)
        self.stage2 = SwinStage(
            dim=embed_dim * 2,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(embed_dim * 2)
        self.head = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        batch_size, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        x, height, width = self.stage1(x, height, width)
        x, height, width = self.merge1(x, height, width)
        x, height, width = self.stage2(x, height, width)

        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def swin_tiny(
    num_classes: int = 10, in_channels: int = 3, image_size: int = 32
) -> SwinTransformer:
    return SwinTransformer(
        image_size=image_size,
        patch_size=4,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2),
        num_heads=(3, 6),
        window_size=4,
        mlp_ratio=4.0,
        dropout=0.1,
    )
