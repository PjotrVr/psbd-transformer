"""LF: a low-frequency additive trigger (Zeng et al., 2021).

This implements the concept, a perturbation whose energy sits in low spatial
frequencies, by low-pass filtering a fixed noise pattern in the Fourier domain.
LF tends to reach low attack success rate on ViT, so treat it as a stress case.
The exact benchmark LF trigger can instead be served through attack_generated.
"""

from dataclasses import dataclass

import torch

from poison import Attack


@dataclass(frozen=True)
class LowFrequencyConfig:
    strength: float = 0.1
    cutoff: int = 4  # keep frequencies within this radius of the spectrum center
    pattern_seed: int = 0
    label_mode: str = "all_to_one"


def _low_frequency_pattern(image_size: int, cutoff: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(3, image_size, image_size, generator=generator) * 2.0 - 1.0
    spectrum = torch.fft.fftshift(torch.fft.fft2(noise), dim=(-2, -1))

    center = image_size // 2
    mask = torch.zeros(image_size, image_size)
    mask[center - cutoff:center + cutoff + 1, center - cutoff:center + cutoff + 1] = 1.0

    filtered = torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask, dim=(-2, -1))).real
    peak = filtered.abs().amax().clamp_min(1e-8)
    return filtered / peak  # scale into -1 to 1


def build(config: LowFrequencyConfig, image_size: int, target_label: int) -> Attack:
    pattern = _low_frequency_pattern(image_size, config.cutoff, config.pattern_seed)
    strength = config.strength

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        return (image + strength * pattern).clamp(0.0, 1.0)

    return Attack("lf", apply_trigger, config.label_mode, target_label)
