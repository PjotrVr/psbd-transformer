"""Trigger patterns that more than 1 attack defines identically.

Both patterns below were previously written out once per attack file, which made
5 copies of 2 definitions and 5 places to get a numerical detail wrong. They are
deterministic functions of their arguments alone, so a single definition serves
every caller: the checkerboard depends only on the patch size, and the blend
pattern only on (image size, seed).

Changing anything here changes the trigger of every attack that uses it, and a
changed trigger silently invalidates every checkpoint and every stealth number
already on disk. Treat the arithmetic as frozen.
"""

import torch

# Every trigger in this package is stamped on an RGB image in pixel space, before
# normalization, so a pattern always carries exactly 3 channels.
TRIGGER_CHANNELS = 3


def checkerboard_patch(patch_size: int) -> torch.Tensor:
    """A (3, patch_size, patch_size) checkerboard, white at (0, 0), same in every channel.

    Cell (row, column) is 1.0 when row + column is even and 0.0 otherwise. Shared
    by BadNet (bottom-right corner), Label-Consistent (all 4 corners), and TaCT
    (bottom-right corner), all 3 of which use the identical patch.
    """
    board = torch.zeros(TRIGGER_CHANNELS, patch_size, patch_size)
    for row in range(patch_size):
        for column in range(patch_size):
            board[:, row, column] = 1.0 if (row + column) % 2 == 0 else 0.0

    return board


def seeded_random_pattern(image_size: int, seed: int) -> torch.Tensor:
    """A (3, image_size, image_size) uniform-random pattern in 0 to 1.

    Shared by Blend and Adaptive-Blend as the full-image pattern they alpha-blend
    over the input. Drawn from a private torch.Generator rather than the global
    RNG, so building an attack never advances the stream that model init, dataset
    shuffling, or the split permutation later read from.
    """
    generator = torch.Generator().manual_seed(seed)

    pattern = torch.rand(TRIGGER_CHANNELS, image_size, image_size, generator=generator)
    return pattern
