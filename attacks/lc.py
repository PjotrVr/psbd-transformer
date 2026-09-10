"""Label-Consistent: a checkerboard in all 4 corners of target-class images (Turner et al., 2019).

Clean-label: only target-class images are poisoned and their label is kept. The
patch alone is the weak half of the attack. Turner's method first perturbs each
base image adversarially so its natural features stop supporting its label and the
trigger becomes the only reliable cue. Without that the model learns the class from
the untouched image and needs nearly the whole target class before it implants.
adversarial.py generates those bases and adversarial_dir points at them.

The perturbation is training-time only. Attack success is measured on non-target
images, which have no base by construction, so apply_trigger_eval stamps the patch
and nothing else.
"""

from dataclasses import dataclass

import torch

from attacks.poisoning import Attack

from .bases import lazy_adversarial_lookup
from .patterns import checkerboard_patch


@dataclass(frozen=True)
class LabelConsistentConfig:
    patch_size: int = 3
    label_mode: str = "clean_label"
    # Empty keeps the patch-only variant, so a checkpoint trained before this field
    # existed is rebuilt exactly as it was.
    adversarial_dir: str = ""
    # Recorded so args.json says which strength the bases carry. The bases are
    # already perturbed, so nothing reads this at runtime.
    adversarial_epsilon: float = 0.0
    # How many consecutive classes from the target the attack may poison. See
    # poisoning.clean_label_target_set.
    num_targets: int = 1


def resolve_clean_label_mode(label_mode: str, num_targets: int) -> str:
    """The label mode a clean-label attack runs under, given its target count.

    More than 1 target is a different label policy rather than a parameter of the
    same one: eligibility becomes set membership and a success is a landing anywhere
    in the set. Deriving it here spares the caller from keeping 2 fields consistent.
    """
    if label_mode == "clean_label" and num_targets > 1:
        return "clean_label_multi"
    return label_mode


def build(config: LabelConsistentConfig, image_size: int, target_label: int) -> Attack:
    """Label-Consistent built for this image size and target label."""
    patch = checkerboard_patch(config.patch_size)  # (3, patch_size, patch_size)
    size = config.patch_size
    adversarial_base = lazy_adversarial_lookup(config.adversarial_dir, image_size)

    def stamp(image: torch.Tensor) -> torch.Tensor:
        stamped = image.clone()
        stamped[:, :size, :size] = patch
        stamped[:, :size, image_size - size :] = patch
        stamped[:, image_size - size :, :size] = patch
        stamped[:, image_size - size :, image_size - size :] = patch
        return stamped

    def apply_trigger(image: torch.Tensor, index: int) -> torch.Tensor:
        # A missing base falls back to the clean image rather than raising, because
        # analysis code holding test indices reaches this same closure. Training
        # checks coverage over the poisoned indices first, so a genuinely absent
        # cache fails loudly there.
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
