"""Label-Consistent backdoor: a corner patch on target-class images (Turner et al., 2019).

Clean-label: only target-class images are poisoned and their label is kept, so a
human inspecting the labels sees nothing wrong. The trigger is a small pattern
placed in the image corners.

The patch alone is the weak half of the attack. Turner's method first perturbs
each base image adversarially, so its natural features stop supporting its own
label and the trigger becomes the only reliable cue left. Without that step the
model can still learn the class from the untouched image and has no reason to
prefer the trigger, which is why the patch-only variant needs roughly the whole
target class before it implants. `psbd.adversarial` generates the perturbed
bases and `adversarial_dir` points at them.

The perturbation is training-time only. At eval time attack success is measured
on NON-target images, which have no adversarial base by construction, so
apply_trigger_eval stamps the patch and nothing else.
"""

from dataclasses import dataclass

import torch

from psbd.poisoning import Attack

from ._bases import lazy_adversarial_lookup
from ._patterns import checkerboard_patch


@dataclass(frozen=True)
class LabelConsistentConfig:
    patch_size: int = 3
    label_mode: str = "clean_label"
    # Empty keeps the self-contained patch-only variant, which is what every
    # checkpoint trained before this field existed used. Leaving it as the
    # default means rebuilding any such checkpoint reproduces it exactly.
    adversarial_dir: str = ""
    # Recorded so a checkpoint's args.json says which perturbation strength its
    # bases carry. The bases are already perturbed; nothing reads this at runtime.
    adversarial_epsilon: float = 0.0
    # How many consecutive classes starting at the target the attack may poison.
    # See attacks/sig.py: 1 target caps a clean-label attack at 1/K of the
    # training set, and widening the set trades detection for reach.
    num_targets: int = 1


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


def build(config: LabelConsistentConfig, image_size: int, target_label: int) -> Attack:
    patch = checkerboard_patch(config.patch_size)  # (3, patch_size, patch_size)
    size = config.patch_size
    adversarial_base = lazy_adversarial_lookup(config.adversarial_dir, image_size)

    def stamp(image: torch.Tensor) -> torch.Tensor:
        # The label-consistent trigger repeats the patch in all four corners.
        stamped = image.clone()
        stamped[:, :size, :size] = patch
        stamped[:, :size, image_size - size :] = patch
        stamped[:, image_size - size :, :size] = patch
        stamped[:, image_size - size :, image_size - size :] = patch
        return stamped

    def apply_trigger(image: torch.Tensor, index: int) -> torch.Tensor:
        # A missing base falls back to the clean image rather than raising, because
        # this same closure is reached by stealth and analysis code holding test
        # indices. train_backdoor validates coverage over the poisoned indices
        # before training, so a genuinely absent cache fails loudly there.
        base = adversarial_base(index)
        return stamp(image if base is None else base)

    def apply_trigger_eval(image: torch.Tensor, _index: int) -> torch.Tensor:
        return stamp(image)

    return Attack(
        "lc",
        apply_trigger,
        resolve_clean_label_mode(config.label_mode, config.num_targets),
        target_label,
        apply_trigger_eval=apply_trigger_eval,
        num_targets=config.num_targets,
    )
