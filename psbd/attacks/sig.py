"""SIG: a horizontal sinusoidal signal added to the image (Barni et al., 2019).

Clean-label by design: it perturbs only target-class images and keeps the label,
which is why the default label mode is clean_label. Some benchmarks run it
dirty-label instead, which is selected by changing label_mode.
"""

from dataclasses import dataclass

import torch

from psbd.poisoning import Attack


@dataclass(frozen=True)
class SigConfig:
    amplitude: float = 0.1  # in 0-to-1 pixel units
    frequency: float = 6.0
    label_mode: str = "clean_label"


def _column_signal(image_size: int, amplitude: float, frequency: float) -> torch.Tensor:
    """One additive value per image column, shaped (1, 1, image_size) to broadcast.

    original form
        v(i, j) = amplitude * sin(2 * pi * frequency * j / width)
    simplified form
        one value per column j, broadcast over every row and channel
    """
    columns = torch.arange(image_size).float()  # (image_size,)
    signal = amplitude * torch.sin(2.0 * torch.pi * frequency * columns / image_size)

    broadcastable = signal.view(1, 1, image_size)
    return broadcastable


def build(config: SigConfig, image_size: int, target_label: int) -> Attack:
    """The SIG attack record for one image size and target label."""
    signal = _column_signal(image_size, config.amplitude, config.frequency)

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        # Clamped because the signal is additive and would otherwise push bright
        # columns outside the 0-to-1 pixel range the trigger is defined in.
        stamped = (image + signal).clamp(0.0, 1.0)  # (C, H, W)
        return stamped

    attack = Attack("sig", apply_trigger, config.label_mode, target_label)
    return attack
