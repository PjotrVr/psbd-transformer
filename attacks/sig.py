"""SIG: a horizontal sinusoidal signal added to the image (Barni et al., 2019).

Clean-label by design: only target-class images are perturbed and their label is
kept, so the default label mode is clean_label. Some benchmarks run it dirty-label,
which is a matter of changing label_mode.
"""

from dataclasses import dataclass

import torch

from attacks.poisoning import Attack


@dataclass(frozen=True)
class SigConfig:
    # Barni et al. use Delta = 40/255. A fainter sinusoid does not implant on ViT.
    amplitude: float = 0.157  # 40/255, in 0-to-1 pixel units
    frequency: float = 6.0
    label_mode: str = "clean_label"
    # How many consecutive classes from the target the attack may poison. 1 caps a
    # clean-label attack at 1/K of the training set. See
    # poisoning.clean_label_target_set for the trade-off in widening it.
    num_targets: int = 1


def _column_signal(image_size: int, amplitude: float, frequency: float) -> torch.Tensor:
    """The additive signal, 1 value per column, shaped (1, 1, image_size) to broadcast.

    original form
        v(i, j) = amplitude * sin(2 * pi * frequency * j / width)
    descriptive form
        a value per column j, broadcast over every row and channel
    """
    columns = torch.arange(image_size).float()  # (image_size,)
    signal = amplitude * torch.sin(2.0 * torch.pi * frequency * columns / image_size)

    broadcastable = signal.view(1, 1, image_size)
    return broadcastable


def resolve_clean_label_mode(label_mode: str, num_targets: int) -> str:
    """The label mode a clean-label attack runs under, given its target count.

    More than 1 target is a different label policy rather than a parameter of the
    same one: eligibility becomes set membership and a success is a landing anywhere
    in the set. Deriving it here spares the caller from keeping 2 fields consistent.
    """
    if label_mode == "clean_label" and num_targets > 1:
        return "clean_label_multi"
    return label_mode


def build(config: SigConfig, image_size: int, target_label: int) -> Attack:
    """SIG built for this image size and target label."""
    signal = _column_signal(image_size, config.amplitude, config.frequency)

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        # Clamped because the signal is additive and would otherwise push bright
        # columns outside the 0-to-1 pixel range the trigger is defined in.
        stamped = (image + signal).clamp(0.0, 1.0)  # (C, H, W)
        return stamped

    attack = Attack(
        "sig",
        apply_trigger,
        resolve_clean_label_mode(config.label_mode, config.num_targets),
        target_label,
        num_targets=config.num_targets,
    )
    return attack
