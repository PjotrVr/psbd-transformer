"""Adaptive-Blend: a blend trigger with cover samples (Qi et al., 2023).

The distinguishing feature is cover samples, triggered images that keep their true
label, which flatten the latent separation between clean and poisoned that many
defenses look for. cover_rate controls how many there are. Training is standard
cross-entropy over the poisoned-plus-cover set, so this file defines the trigger
and its cover rate, and train_backdoor reads cover_rate to build the set.

The paper also uses an asymmetric trigger, fewer blend cells at train time than at
test time. That refinement is omitted for simplicity and can be added by giving the
train and test paths different masks over the pattern.
"""

from dataclasses import dataclass

import torch

from poison import Attack


@dataclass(frozen=True)
class AdaptiveBlendConfig:
    alpha: float = 0.2
    cover_rate: float = 0.01
    pattern_seed: int = 0
    label_mode: str = "all_to_one"


def _random_pattern(image_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, image_size, image_size, generator=generator)


def build(config: AdaptiveBlendConfig, image_size: int, target_label: int) -> Attack:
    pattern = _random_pattern(image_size, config.pattern_seed)
    alpha = config.alpha

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        # original: x_poisoned = (1 - alpha) * x + alpha * pattern
        return (1.0 - alpha) * image + alpha * pattern

    return Attack("adaptive_blend", apply_trigger, config.label_mode, target_label)
