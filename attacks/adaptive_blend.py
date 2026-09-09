"""Adaptive-Blend: a blend trigger with cover samples (Qi et al., 2023).

The distinguishing feature is cover samples, triggered images that keep their true
label, which flatten the latent separation between clean and poisoned that many
defenses look for. cover_rate controls how many there are. Training is standard
cross-entropy over the poisoned-plus-cover set, so this file defines the trigger
and its cover rate, and the training entrypoint reads cover_rate to build the set.

The paper also uses an asymmetric trigger, fewer blend cells at train time than at
test time. That refinement is omitted for simplicity and can be added by giving the
train and test paths different masks over the pattern.
"""

from dataclasses import dataclass

import torch

from attacks.poisoning import Attack

from .patterns import seeded_random_pattern


@dataclass(frozen=True)
class AdaptiveBlendConfig:
    alpha: float = 0.2
    cover_rate: float = 0.01
    pattern_seed: int = 0
    label_mode: str = "all_to_one"
    # The pattern is split into a cells x cells grid and only train_cell_fraction of
    # the cells are planted during training, the whole pattern at inference. That
    # asymmetry is the paper's second mechanism and omitting it cost 0.63 mean ASR.
    cells: int = 4
    train_cell_fraction: float = 0.5


def _cell_mask(
    image_size: int, cells: int, fraction: float, index: int, seed: int
) -> torch.Tensor:
    """A (1, H, W) mask keeping `fraction` of a cells x cells grid, chosen per sample.

    Seeded from the sample index, so one image always receives the same subset and the
    poisoned training set stays reproducible across epochs and across runs.
    """
    generator = torch.Generator().manual_seed(seed * 1_000_003 + index)
    keep = torch.rand(cells, cells, generator=generator) < fraction
    step = image_size / cells
    rows = (torch.arange(image_size) / step).long().clamp(max=cells - 1)
    columns = (torch.arange(image_size) / step).long().clamp(max=cells - 1)
    return keep[rows][:, columns].unsqueeze(0).float()


def build(config: AdaptiveBlendConfig, image_size: int, target_label: int) -> Attack:
    """The Adaptive-Blend attack record for one image size and target label."""
    pattern = seeded_random_pattern(image_size, config.pattern_seed)  # (3, S, S)
    alpha = config.alpha

    def plant(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # original: x_poisoned = (1 - alpha) * x + alpha * pattern
        # asymmetric: blended only where the cell mask is on
        return image * (1.0 - alpha * mask) + alpha * mask * pattern  # (C, H, W)

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
        return plant(image, torch.ones(1, image_size, image_size))

    attack = Attack(
        "adaptive_blend",
        apply_trigger,
        config.label_mode,
        target_label,
        apply_trigger_eval=apply_trigger_eval,
    )
    return attack
