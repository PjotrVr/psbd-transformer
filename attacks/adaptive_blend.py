"""Adaptive-Blend: a blend trigger with cover samples (Qi et al., 2023).

The distinguishing feature is cover samples, triggered images that keep their true
label, which flatten the latent separation between clean and poisoned that many
defenses look for. cover_rate controls how many there are. Training is standard
cross-entropy over the poisoned-plus-cover set, so this file defines the trigger
and its cover rate, and train_backdoor reads cover_rate to build the set.

The second mechanism is the ASYMMETRIC trigger: the paper plants fewer blend cells
during training than at test time. It is not a refinement, it is what makes the attack
work. Training on a random subset of the pattern's cells forces the model to generalise
over the pattern rather than memorise it, so the complete pattern at inference lands far
inside the learned region and ASR rises; at the same time the weaker training signal
keeps the poisoned latents close to the clean ones, which is the whole point of the
attack. Omitting it cost 0.63 mean ASR against the 0.9 an attack has to reach to be
worth evaluating a defence on.
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
    # The pattern is split into a cells x cells grid and only train_cell_fraction of
    # the cells are planted during training. 16 cells and half of them is the paper's
    # setting; 1.0 disables the asymmetry and reproduces the earlier behaviour.
    cells: int = 4
    train_cell_fraction: float = 0.5


def _random_pattern(image_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, image_size, image_size, generator=generator)


def _cell_mask(
    image_size: int, cells: int, fraction: float, index: int, seed: int
) -> torch.Tensor:
    """A (1, H, W) mask keeping `fraction` of a cells x cells grid, chosen per sample.

    Seeded from the sample index so the same image always receives the same subset,
    which keeps the poisoned training set reproducible across epochs and across runs.
    """
    generator = torch.Generator().manual_seed(seed * 1_000_003 + index)
    keep = torch.rand(cells, cells, generator=generator) < fraction
    step = image_size / cells
    rows = (torch.arange(image_size) / step).long().clamp(max=cells - 1)
    columns = (torch.arange(image_size) / step).long().clamp(max=cells - 1)
    return keep[rows][:, columns].unsqueeze(0).float()


def build(config: AdaptiveBlendConfig, image_size: int, target_label: int) -> Attack:
    pattern = _random_pattern(image_size, config.pattern_seed)
    alpha = config.alpha

    def plant(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # original: x_poisoned = (1 - alpha) * x + alpha * pattern
        # asymmetric: the blend is applied only where the cell mask is on
        return image * (1.0 - alpha * mask) + alpha * mask * pattern

    def apply_trigger(image: torch.Tensor, index: int) -> torch.Tensor:
        mask = _cell_mask(
            image_size,
            config.cells,
            config.train_cell_fraction,
            index,
            config.pattern_seed,
        )
        return plant(image, mask)

    def apply_trigger_eval(image: torch.Tensor, _index: int) -> torch.Tensor:
        # The whole pattern at inference, which is the asymmetry.
        return plant(image, torch.ones(1, image_size, image_size))

    return Attack(
        "adaptive_blend",
        apply_trigger,
        config.label_mode,
        target_label,
        apply_trigger_eval=apply_trigger_eval,
    )
