"""Trigger patterns shared by more than 1 attack.

The checkerboard depends only on the patch size and serves BadNet, Label-Consistent
and TaCT. The blend pattern depends only on the image size and a seed and serves
Blend and Adaptive-Blend. Both are deterministic functions of their arguments.

Changing the arithmetic here changes the trigger of every attack that uses it,
which silently invalidates every checkpoint and stealth number already on disk.
Treat it as frozen.
"""

import torch

# Every trigger in this package is stamped on an RGB image in pixel space, before
# normalization, so a pattern always carries exactly 3 channels.
TRIGGER_CHANNELS = 3


def checkerboard_patch(patch_size: int) -> torch.Tensor:
    """A (3, patch_size, patch_size) checkerboard, white at (0, 0), equal in every channel.

    Cell (row, column) is 1.0 when row + column is even and 0.0 otherwise.
    """
    board = torch.zeros(TRIGGER_CHANNELS, patch_size, patch_size)
    for row in range(patch_size):
        for column in range(patch_size):
            board[:, row, column] = 1.0 if (row + column) % 2 == 0 else 0.0

    return board


def seeded_random_pattern(image_size: int, seed: int) -> torch.Tensor:
    """A (3, image_size, image_size) uniform-random pattern in 0 to 1.

    Drawn from a private generator rather than the global stream, so building an
    attack never advances what model init, shuffling and the split permutation read
    from later.
    """
    generator = torch.Generator().manual_seed(seed)

    pattern = torch.rand(TRIGGER_CHANNELS, image_size, image_size, generator=generator)
    return pattern
