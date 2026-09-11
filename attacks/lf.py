"""LF: a low-frequency additive trigger (Zeng et al., 2021).

A perturbation whose energy sits in low spatial frequencies, made by low-pass
filtering a fixed noise pattern in the Fourier domain. LF implants weakly on ViT,
so treat it as a stress case. The exact benchmark trigger can be served through
generated.py instead.
"""

from dataclasses import dataclass

import torch

from attacks.poisoning import Attack


@dataclass(frozen=True)
class LowFrequencyConfig:
    strength: float = 0.1
    cutoff: int = 4  # keep frequencies within this radius of the spectrum center
    pattern_seed: int = 0
    label_mode: str = "all_to_one"


def _low_frequency_pattern(image_size: int, cutoff: int, seed: int) -> torch.Tensor:
    """A (3, image_size, image_size) pattern whose energy sits below the cutoff.

    Noise is transformed, fftshifted so the zero frequency sits at the center,
    masked to a square of radius cutoff around that center, and transformed back.
    The result is rescaled by its own peak so the pattern spans -1 to 1 and
    strength alone controls how strongly it is added.
    """
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(3, image_size, image_size, generator=generator) * 2.0 - 1.0
    spectrum = torch.fft.fftshift(torch.fft.fft2(noise), dim=(-2, -1))  # (3, H, W)

    center = image_size // 2
    mask = torch.zeros(image_size, image_size)  # (H, W)
    mask[
        center - cutoff : center + cutoff + 1, center - cutoff : center + cutoff + 1
    ] = 1.0

    filtered = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum * mask, dim=(-2, -1))
    ).real  # (3, H, W)
    # The peak is floored so a degenerate all-zero band divides by a small
    # constant rather than by 0.
    peak = filtered.abs().amax().clamp_min(1e-8)

    pattern = filtered / peak  # (3, H, W), in -1 to 1
    return pattern


def build(config: LowFrequencyConfig, image_size: int, target_label: int) -> Attack:
    """LF built for this image size and target label."""
    pattern = _low_frequency_pattern(image_size, config.cutoff, config.pattern_seed)
    strength = config.strength

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        stamped = (image + strength * pattern).clamp(0.0, 1.0)  # (C, H, W)
        return stamped

    attack = Attack("lf", apply_trigger, config.label_mode, target_label)
    return attack
