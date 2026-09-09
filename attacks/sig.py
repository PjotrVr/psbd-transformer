"""SIG: a horizontal sinusoidal signal added to the image (Barni et al., 2019).

Clean-label by design: it perturbs only target-class images and keeps the label,
which is why the default label mode is clean_label. Some benchmarks run it
dirty-label instead, which you can select by changing label_mode.
"""

from dataclasses import dataclass

import torch

from poison import Attack


@dataclass(frozen=True)
class SigConfig:
    # Barni et al. use Delta = 40/255. At 0.1 the sinusoid is too faint to learn:
    # SIG implanted at 0.015 to 0.46 ASR across the panel, under the 0.85 bar on
    # 11 of 12 cells.
    amplitude: float = 0.157  # 40/255, in 0-to-1 pixel units
    frequency: float = 6.0
    label_mode: str = "clean_label"
    # How many consecutive classes starting at the target the attack may poison.
    # A clean-label attack keeps every label, so 1 target caps it at 1/K of the
    # training set: 1% on CIFAR-100, 0.5% on Tiny. Widening the set is the only
    # way to lift that without changing the dataset, and it costs detection, so
    # the measured usable range is 2 to 4.
    num_targets: int = 1


def _column_signal(image_size: int, amplitude: float, frequency: float) -> torch.Tensor:
    # original: v(i, j) = amplitude * sin(2 * pi * frequency * j / width)
    # simplified: one value per column j, broadcast over rows and channels
    columns = torch.arange(image_size).float()
    signal = amplitude * torch.sin(2.0 * torch.pi * frequency * columns / image_size)
    return signal.view(1, 1, image_size)


def resolve_clean_label_mode(label_mode: str, num_targets: int) -> str:
    """The label mode a clean-label attack runs under, given its target count.

    More than 1 target is a different label policy, not the same one with a
    parameter: eligibility becomes set membership on both sides, and a success is
    a landing anywhere in the set. Deriving the mode here keeps a caller from
    having to set 2 fields consistently.
    """
    if label_mode == "clean_label" and num_targets > 1:
        return "clean_label_multi"
    return label_mode


def build(config: SigConfig, image_size: int, target_label: int) -> Attack:
    signal = _column_signal(image_size, config.amplitude, config.frequency)

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        return (image + signal).clamp(0.0, 1.0)

    return Attack(
        "sig",
        apply_trigger,
        resolve_clean_label_mode(config.label_mode, config.num_targets),
        target_label,
        num_targets=config.num_targets,
    )
