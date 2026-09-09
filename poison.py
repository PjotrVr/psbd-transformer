"""Attack interface and the shared poisoning pipeline.

An attack is a small record, not a class hierarchy: a name, a function that
stamps the trigger onto one image, and a label policy. Everything common across
attacks lives here once, so each attack file only defines its trigger.

Triggers act in pixel space on a CHW image tensor in the range 0 to 1, before
normalization, which is where pixel-space attacks are defined.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

# apply_trigger takes the image and its dataset index. Static attacks ignore the
# index. Sample-specific attacks use it to look up a pregenerated perturbation.
ApplyTrigger = Callable[[torch.Tensor, int], torch.Tensor]

LABEL_MODES = ("all_to_one", "all_to_all", "all_to_m", "clean_label")


@dataclass(frozen=True)
class Attack:
    name: str
    apply_trigger: ApplyTrigger
    label_mode: str
    target_label: int
    # Source-specific attacks (TaCT) flip only these classes. Carried on the Attack
    # rather than left on the config so it reaches evaluation, where measuring ASR
    # over every non-target class instead understates it by the class count.
    source_classes: tuple[int, ...] | None = None
    # Cover samples normally carry the same trigger with their label kept, which is
    # what Adaptive-Blend and TaCT do. WaNet's noise mode is different: its cover
    # samples get a RANDOM warp, not the trigger warp, and that difference is the
    # whole point, since it stops the warping itself from becoming the cue. An
    # attack that needs its own cover transform supplies it here.
    apply_cover: ApplyTrigger | None = None
    # Adaptive-Blend plants a WEAKER trigger during training than at test time: a
    # random subset of the pattern's cells while training, the whole pattern at
    # inference. That asymmetry is the mechanism, not a refinement. Training on
    # partial evidence forces the model to generalise over the pattern, so the full
    # pattern at test time lands far inside the learned region and ASR rises, while
    # the weaker training signal keeps the poisoned latents close to the clean ones,
    # which is what the attack exists to do. An attack that needs a different trigger
    # at eval supplies it here; everything else leaves it None and apply_trigger is
    # used for both.
    apply_trigger_eval: ApplyTrigger | None = None
    # all_to_m only: how many distinct target classes the trigger maps onto. It is
    # the one knob that interpolates between the 2 poles PSBD's premise sits
    # between, so it is carried rather than derived: m = 1 reproduces all_to_one on
    # target 0 exactly, m = num_classes reproduces all_to_all exactly.
    num_targets: int | None = None


def _grouped_target(
    original_label: int, num_targets: int | None, num_classes: int | None = None
) -> int:
    """The all_to_m target class: (y + 1) mod m.

        original form
            y_poisoned = (y + 1) mod m
        descriptive form
            the next class, wrapping within the first m classes only

    m is the number of distinct classes the trigger maps onto, and it is the whole
    point of this label mode. PSBD's premise is that the trigger is a CONSTANT,
    content-independent shortcut: the perturbed prediction stays pinned because the
    shortcut never has to read the image. all_to_one satisfies that exactly, and
    all_to_all violates it exactly, because (y + 1) mod K forces the model to
    recognise the source class before it can increment. m interpolates between them
    and is the only thing that varies, so the amount of content the backdoor map
    must encode is log2(m) bits.

    The 2 poles are reproduced exactly rather than approximately:
      m = 1            (y + 1) mod 1 = 0, so every poisoned sample takes class 0,
                       which is all_to_one on target 0.
      m = num_classes  (y + 1) mod K, which is all_to_all.
    """
    if not num_targets or num_targets < 1:
        raise ValueError(
            f"all_to_m needs a positive num_targets, got {num_targets!r}. It is "
            "carried on the Attack and must be set by the attack's config."
        )
    # m > num_classes does not degenerate gracefully, it emits an out-of-range
    # label: at m = 16 on a 10-class dataset the modulus never wraps, so class 9
    # maps to label 10 and the loss indexes past the end of the logits. Rejected
    # here rather than at the loss, where it surfaces as a CUDA assert with no
    # mention of the label map.
    if num_classes is not None and num_targets > num_classes:
        raise ValueError(
            f"all_to_m needs num_targets <= num_classes, got m={num_targets} for "
            f"{num_classes} classes. m = num_classes is already all_to_all."
        )
    return (original_label + 1) % num_targets


def is_poisonable(
    label_mode: str,
    original_label: int,
    target_label: int,
    num_targets: int | None = None,
) -> bool:
    """Which samples an attack is allowed to poison.

    all_to_one poisons any sample not already the target class. all_to_all
    poisons any sample. clean_label poisons only target-class samples, since it
    must not change the label.
    """
    if label_mode == "all_to_one":
        return original_label != target_label
    if label_mode == "all_to_all":
        return True
    if label_mode == "all_to_m":
        return _grouped_target(original_label, num_targets) != original_label
    if label_mode == "clean_label":
        return original_label == target_label
    raise ValueError(f"Unknown label mode: {label_mode}")


def poisoned_label(
    label_mode: str,
    original_label: int,
    target_label: int,
    num_classes: int,
    num_targets: int | None = None,
) -> int:
    """The label a poisoned sample is given.

    original form for all_to_all
        y_poisoned = (y + 1) mod K
    simplified form
        the next class, wrapping so the last class maps back to 0
    """
    if label_mode == "all_to_one":
        return target_label
    if label_mode == "all_to_all":
        return (original_label + 1) % num_classes
    if label_mode == "all_to_m":
        return _grouped_target(original_label, num_targets, num_classes)
    if label_mode == "clean_label":
        return original_label
    raise ValueError(f"Unknown label mode: {label_mode}")


def is_eval_poisonable(
    label_mode: str,
    original_label: int,
    target_label: int,
    num_targets: int | None = None,
) -> bool:
    """Which samples belong in an attack-success eval set.

    Identical to is_poisonable except for clean_label. Training poisons only
    target-class images, since a clean-label attack must not change the label.
    But measuring attack success asks a different question: does the trigger
    fool a non-target image into being predicted as the target. So the eval
    eligibility flips to original_label != target_label, the same question
    all_to_one and all_to_all already ask.
    """
    if label_mode == "all_to_one":
        return original_label != target_label
    if label_mode == "all_to_all":
        return True
    if label_mode == "all_to_m":
        return _grouped_target(original_label, num_targets) != original_label
    if label_mode == "clean_label":
        return original_label != target_label
    raise ValueError(f"Unknown label mode: {label_mode}")


def attack_success_label(
    label_mode: str,
    original_label: int,
    target_label: int,
    num_classes: int,
    num_targets: int | None = None,
) -> int:
    """The label an attack-success eval sample is compared against.

    Identical to poisoned_label except for clean_label. poisoned_label's
    clean_label branch returns original_label, which is correct only at
    training time, where is_poisonable already restricts clean_label to
    original_label == target_label so that is a no-op. At eval time
    is_eval_poisonable flips clean_label eligibility to
    original_label != target_label, so returning original_label there would be
    wrong: the intended label is always target_label.
    """
    if label_mode == "clean_label":
        return target_label
    return poisoned_label(
        label_mode, original_label, target_label, num_classes, num_targets
    )


def choose_poison_indices(
    labels: list[int], attack: Attack, poison_rate: float, seed: int
) -> set[int]:
    """Pick which dataset indices to poison at the requested rate.

    The rate is measured against the whole dataset. For clean_label the eligible
    pool is only the target class, so the count is capped by how many target
    samples exist.
    """
    eligible = [
        i
        for i, y in enumerate(labels)
        if is_poisonable(
            attack.label_mode, int(y), attack.target_label, attack.num_targets
        )
    ]
    count = min(int(round(poison_rate * len(labels))), len(eligible))
    rng = np.random.default_rng(seed)
    return set(int(i) for i in rng.choice(eligible, size=count, replace=False))


class PoisonedTrainingSet(Dataset):
    """Wraps a clean dataset yielding 0-to-1 images and poisons chosen indices.

    Normalization is applied last so the model still receives normalized inputs.
    An empty poison_indices set turns this into a plain normalized clean set.
    """

    def __init__(self, base_dataset, attack, poison_indices, normalize, num_classes):
        self.base_dataset = base_dataset
        self.attack = attack
        self.poison_indices = poison_indices
        self.normalize = normalize
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        if index in self.poison_indices:
            image = self.attack.apply_trigger(image, index)
            label = poisoned_label(
                self.attack.label_mode,
                int(label),
                self.attack.target_label,
                self.num_classes,
                self.attack.num_targets,
            )
        return self.normalize(image), label


class AttackSuccessSet(Dataset):
    """Every eligible sample poisoned, for measuring attack success rate.

    The returned label is the attack's intended label per sample, so accuracy on
    this set is the ASR. Samples that cannot flip under the label mode are
    dropped so the ASR is measured only over samples that should flip. Uses the
    eval-time eligibility and label functions, not the training-time ones, since
    clean_label asks a different question at eval time (see is_eval_poisonable).

    A source-specific attack only claims to flip its source classes, so when the
    attack names them the set restricts to those. Measuring over every non-target
    class instead divides the true rate by the class count: TaCT on CIFAR-10 reads
    0.171 that way against 1.000 on its source class, which looks like a failed
    attack and is not one. source_only=False builds the complement, whose success
    rate is the false-trigger rate on classes the attack never claimed.
    """

    def __init__(
        self, base_dataset, labels, attack, normalize, num_classes, source_only=True
    ):
        self.base_dataset = base_dataset
        self.labels = labels
        self.attack = attack
        self.normalize = normalize
        self.num_classes = num_classes
        eligible = [
            i
            for i, y in enumerate(labels)
            if is_eval_poisonable(
                attack.label_mode, int(y), attack.target_label, attack.num_targets
            )
        ]
        if attack.source_classes is not None:
            sources = set(attack.source_classes)
            eligible = [
                i for i in eligible if (int(labels[i]) in sources) == source_only
            ]
        self.indices = eligible

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        image, _ = self.base_dataset[index]
        plant = self.attack.apply_trigger_eval or self.attack.apply_trigger
        poisoned = plant(image, index)
        target = attack_success_label(
            self.attack.label_mode,
            int(self.labels[index]),
            self.attack.target_label,
            self.num_classes,
            self.attack.num_targets,
        )
        return self.normalize(poisoned), target


def choose_indices_with_cover(
    labels: list[int],
    attack: Attack,
    poison_rate: float,
    cover_rate: float,
    source_classes: tuple[int, ...] | None,
    seed: int,
) -> tuple[set[int], set[int]]:
    """Pick poison indices and cover indices for adaptive attacks.

    Cover samples receive the trigger but keep their true label. They teach the
    model that the trigger alone does not imply the target, which is how adaptive
    attacks flatten the latent separation that many defenses rely on.

    source_classes, when given, restricts poisoning to those classes, the
    source-specific setting of TaCT. Cover samples are then drawn from the other
    non-target classes.
    """
    rng = np.random.default_rng(seed)
    dataset_size = len(labels)

    def is_poison_eligible(index: int) -> bool:
        label = int(labels[index])
        if source_classes is not None and label not in source_classes:
            return False
        return is_poisonable(
            attack.label_mode, label, attack.target_label, attack.num_targets
        )

    poison_pool = [i for i in range(dataset_size) if is_poison_eligible(i)]
    poison_count = min(int(round(poison_rate * dataset_size)), len(poison_pool))
    poison_indices = (
        set(int(i) for i in rng.choice(poison_pool, size=poison_count, replace=False))
        if poison_count > 0
        else set()
    )

    def is_cover_eligible(index: int) -> bool:
        label = int(labels[index])
        if index in poison_indices or label == attack.target_label:
            return False
        if source_classes is not None and label in source_classes:
            return False
        return True

    cover_pool = [i for i in range(dataset_size) if is_cover_eligible(i)]
    cover_count = min(int(round(cover_rate * dataset_size)), len(cover_pool))
    cover_indices = (
        set(int(i) for i in rng.choice(cover_pool, size=cover_count, replace=False))
        if cover_count > 0
        else set()
    )

    return poison_indices, cover_indices


class CoverPoisonedTrainingSet(Dataset):
    """Poisons some indices and triggers cover indices without relabeling them.

    Poisoned samples get the trigger and the poisoned label. Cover samples get the
    trigger and keep their original label. Everything else stays clean.
    """

    def __init__(
        self,
        base_dataset,
        attack,
        poison_indices,
        cover_indices,
        normalize,
        num_classes,
    ):
        self.base_dataset = base_dataset
        self.attack = attack
        self.poison_indices = poison_indices
        self.cover_indices = cover_indices
        self.normalize = normalize
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        if index in self.poison_indices:
            image = self.attack.apply_trigger(image, index)
            label = poisoned_label(
                self.attack.label_mode,
                int(label),
                self.attack.target_label,
                self.num_classes,
                self.attack.num_targets,
            )
        elif index in self.cover_indices:
            cover = self.attack.apply_cover or self.attack.apply_trigger
            image = cover(image, index)
        return self.normalize(image), label
