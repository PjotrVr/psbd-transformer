"""The Attack record, the label policies and the datasets that apply them.

An attack is a small record rather than a class hierarchy: a name, a function that
stamps the trigger onto 1 image, and a label mode. Everything shared across attacks
lives here once, so each attack module only defines its trigger.

Triggers act in pixel space on a (C, H, W) tensor in 0 to 1, before normalization,
because that is where the papers define them. Every dataset wrapper here therefore
takes a normalize callable and applies it last.

Training and evaluation ask different questions of the same attack and get separate
function pairs. Training asks which samples may be poisoned and what label they get
(is_poisonable, poisoned_label). Evaluation asks which samples belong in an
attack-success set and what prediction counts as success (is_eval_poisonable,
attack_success_label). For a clean-label attack the 2 answers are opposites, so
mixing the pairs up silently measures the wrong thing.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

# apply_trigger takes the image and its dataset index. Static attacks ignore the
# index. Sample-specific attacks use it to look up a pregenerated perturbation.
ApplyTrigger = Callable[[torch.Tensor, int], torch.Tensor]

LABEL_MODES = (
    "all_to_one",
    "all_to_all",
    "all_to_m",
    "clean_label",
    "clean_label_multi",
)


@dataclass(frozen=True)
class Attack:
    """An attack: its name, its pixel-space trigger and its label policy."""

    name: str
    apply_trigger: ApplyTrigger
    label_mode: str
    target_label: int
    # A source-specific attack (TaCT) only claims to flip these classes. Carried on
    # the record rather than left on the config so evaluation can restrict the ASR
    # set to them, otherwise the rate is diluted by every other class.
    source_classes: tuple[int, ...] | None = None
    # Cover samples usually carry the ordinary trigger with their label kept. An
    # attack whose covers need a different transform supplies it here, as WaNet's
    # noise mode does with a random warp, so that warping itself cannot become the
    # cue.
    apply_cover: ApplyTrigger | None = None
    # An attack that plants a different trigger at evaluation than in training
    # supplies it here, as Adaptive-Blend does with its partial-cell training
    # trigger and Label-Consistent with its adversarial bases. None means the
    # training trigger is used for both.
    apply_trigger_eval: ApplyTrigger | None = None
    # all_to_m and clean_label_multi: how many distinct classes the trigger maps
    # onto or may poison. Carried rather than derived because m = 1 must reproduce
    # all_to_one exactly and m = num_classes must reproduce all_to_all exactly.
    num_targets: int | None = None


def _grouped_target(
    original_label: int, num_targets: int | None, num_classes: int | None = None
) -> int:
    """The all_to_m target class.

    original form
        y_poisoned = (y + 1) mod m
    descriptive form
        the next class, wrapping within the first m classes only

    m is how many distinct classes the trigger maps onto. PSBD's premise is that a
    trigger is a constant shortcut that never has to read the image. all_to_one
    satisfies that exactly and all_to_all violates it exactly, since (y + 1) mod K
    has to recognise the source class before it can increment. m interpolates
    between the 2, and the backdoor has to encode log2(m) bits of content to do it.
    Both poles are reproduced exactly: m = 1 gives class 0 for every sample, which
    is all_to_one on target 0, and m = num_classes is all_to_all.
    """
    if not num_targets or num_targets < 1:
        raise ValueError(
            f"all_to_m needs a positive num_targets, got {num_targets!r}. It is "
            "carried on the Attack and must be set by the attack's config."
        )
    # m above the class count emits an out-of-range label (class 9 maps to 10 at
    # m = 16 on 10 classes) and the loss then indexes past the logits. Rejected here
    # rather than at the loss, where it surfaces as a CUDA assert that never
    # mentions the label map.
    if num_classes is not None and num_targets > num_classes:
        raise ValueError(
            f"all_to_m needs num_targets <= num_classes, got m={num_targets} for "
            f"{num_classes} classes. m = num_classes is already all_to_all."
        )

    target = (original_label + 1) % num_targets
    return target


def clean_label_target_set(
    target_label: int, num_targets: int | None
) -> tuple[int, ...]:
    """The classes a clean-label attack is allowed to poison.

    A clean-label attack keeps every label, so it can only poison images that
    already carry a target label, and its poison rate is capped at |T| / |train
    set|. With 1 target on a balanced K-class dataset that is 1/K, which is why
    CIFAR-100 caps at 1% and Tiny ImageNet at 0.5%. Widening T to a few adjacent
    classes is the only way to lift the cap without changing the dataset. The cost
    is that the trigger then predicts a set rather than a class, so detection
    weakens, and T stays small for that reason.

    The classes are consecutive from target_label and do not wrap, so asking for
    more targets than the dataset has above target_label yields an out-of-range
    class rather than a silent overlap with class 0.
    """
    if not num_targets or num_targets <= 1:
        return (target_label,)

    targets = tuple(target_label + offset for offset in range(num_targets))
    return targets


def is_poisonable(
    label_mode: str,
    original_label: int,
    target_label: int,
    num_targets: int | None = None,
) -> bool:
    """Whether a sample may be poisoned at training time.

    all_to_one poisons anything not already the target class and all_to_all poisons
    anything. A clean-label mode poisons only the target classes, since it must not
    change the label.
    """
    if label_mode == "all_to_one":
        return original_label != target_label
    if label_mode == "all_to_all":
        return True
    if label_mode == "all_to_m":
        grouped_target = _grouped_target(original_label, num_targets)
        return grouped_target != original_label
    if label_mode == "clean_label":
        return original_label == target_label
    if label_mode == "clean_label_multi":
        targets = clean_label_target_set(target_label, num_targets)
        return original_label in targets
    raise ValueError(f"Unknown label mode: {label_mode}")


def poisoned_label(
    label_mode: str,
    original_label: int,
    target_label: int,
    num_classes: int,
    num_targets: int | None = None,
) -> int:
    """The label a poisoned sample is given at training time.

    original form for all_to_all
        y_poisoned = (y + 1) mod K
    simplified form
        the next class, wrapping so the last class maps back to 0
    """
    if label_mode == "all_to_one":
        return target_label
    if label_mode == "all_to_all":
        next_class = (original_label + 1) % num_classes
        return next_class
    if label_mode == "all_to_m":
        grouped_target = _grouped_target(original_label, num_targets, num_classes)
        return grouped_target
    if label_mode in ("clean_label", "clean_label_multi"):
        return original_label
    raise ValueError(f"Unknown label mode: {label_mode}")


def is_eval_poisonable(
    label_mode: str,
    original_label: int,
    target_label: int,
    num_targets: int | None = None,
) -> bool:
    """Whether a sample belongs in an attack-success set at evaluation time.

    The same as is_poisonable except for the clean-label modes. Training poisons
    only target-class images, but attack success asks whether the trigger fools a
    non-target image into the target, so eligibility flips to the complement, the
    question all_to_one and all_to_all already ask.
    """
    if label_mode == "all_to_one":
        return original_label != target_label
    if label_mode == "all_to_all":
        return True
    if label_mode == "all_to_m":
        grouped_target = _grouped_target(original_label, num_targets)
        return grouped_target != original_label
    if label_mode == "clean_label":
        return original_label != target_label
    if label_mode == "clean_label_multi":
        targets = clean_label_target_set(target_label, num_targets)
        return original_label not in targets
    raise ValueError(f"Unknown label mode: {label_mode}")


def attack_success_label(
    label_mode: str,
    original_label: int,
    target_label: int,
    num_classes: int,
    num_targets: int | None = None,
) -> int:
    """The label an attack-success sample is compared against at evaluation time.

    The same as poisoned_label except for the clean-label modes, whose training
    label is the unchanged original. At evaluation the eligible images are the
    non-target ones, so the intended label is always target_label.
    """
    if label_mode in ("clean_label", "clean_label_multi"):
        return target_label

    training_label = poisoned_label(
        label_mode, original_label, target_label, num_classes, num_targets
    )
    return training_label


def choose_poison_indices(
    labels: list[int], attack: Attack, poison_rate: float, seed: int
) -> set[int]:
    """The dataset indices to poison at the requested rate.

    The rate is measured against the whole dataset, and the count is capped by the
    eligible pool. For a clean-label attack that pool is only the target class, so
    every rate above the cap selects the same indices. The realized rate is
    recorded in the checkpoint's provenance for that reason.
    """
    eligible = [
        i
        for i, y in enumerate(labels)
        if is_poisonable(
            attack.label_mode, int(y), attack.target_label, attack.num_targets
        )
    ]
    requested_count = int(round(poison_rate * len(labels)))
    poison_count = min(requested_count, len(eligible))

    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=poison_count, replace=False)
    poison_indices = set(int(i) for i in chosen)
    return poison_indices


def choose_indices_with_cover(
    labels: list[int],
    attack: Attack,
    poison_rate: float,
    cover_rate: float,
    source_classes: tuple[int, ...] | None,
    seed: int,
) -> tuple[set[int], set[int]]:
    """Poison indices and cover indices for an attack that trains with covers.

    Cover samples receive the trigger but keep their true label. They teach the
    model that the trigger alone does not imply the target, which is how adaptive
    attacks flatten the latent separation many defences rely on. source_classes,
    when given, restricts poisoning to those classes (TaCT). Covers are then drawn
    from the other non-target classes.

    Both draws come from 1 generator, poison first, so the seed reproduces the pair
    and not just each set alone.
    """
    rng = np.random.default_rng(seed)
    dataset_size = len(labels)

    def is_poison_eligible(index: int) -> bool:
        label = int(labels[index])
        if source_classes is not None and label not in source_classes:
            return False
        eligible = is_poisonable(
            attack.label_mode, label, attack.target_label, attack.num_targets
        )
        return eligible

    poison_pool = [i for i in range(dataset_size) if is_poison_eligible(i)]
    poison_count = min(int(round(poison_rate * dataset_size)), len(poison_pool))
    poison_indices: set[int] = set()
    if poison_count > 0:
        chosen_poison = rng.choice(poison_pool, size=poison_count, replace=False)
        poison_indices = set(int(i) for i in chosen_poison)

    def is_cover_eligible(index: int) -> bool:
        label = int(labels[index])
        if index in poison_indices or label == attack.target_label:
            return False
        if source_classes is not None and label in source_classes:
            return False
        return True

    cover_pool = [i for i in range(dataset_size) if is_cover_eligible(i)]
    cover_count = min(int(round(cover_rate * dataset_size)), len(cover_pool))
    cover_indices: set[int] = set()
    if cover_count > 0:
        chosen_cover = rng.choice(cover_pool, size=cover_count, replace=False)
        cover_indices = set(int(i) for i in chosen_cover)

    return poison_indices, cover_indices


class PoisonedTrainingSet(Dataset):
    """A clean dataset of 0-to-1 images with the chosen indices poisoned.

    Normalization is applied last so the model still receives normalized inputs.
    An empty poison_indices turns this into a plain normalized clean set, which is
    how the clean splits of an evaluation run are built.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        attack: Attack,
        poison_indices: set[int],
        normalize: Callable[[torch.Tensor], torch.Tensor],
        num_classes: int,
    ):
        self.base_dataset = base_dataset
        self.attack = attack
        self.poison_indices = poison_indices
        self.normalize = normalize
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.base_dataset[index]  # image is (C, H, W) in 0 to 1

        if index in self.poison_indices:
            image = self.attack.apply_trigger(image, index)
            label = poisoned_label(
                self.attack.label_mode,
                int(label),
                self.attack.target_label,
                self.num_classes,
                self.attack.num_targets,
            )

        normalized = self.normalize(image)  # (C, H, W)
        return normalized, label


class AttackSuccessSet(Dataset):
    """Every eligible test image with the trigger applied, labeled by intent.

    The label is the attack's intended label, so accuracy on this set is the ASR.
    Samples that cannot flip under the label mode are dropped, and eligibility uses
    the evaluation-time functions rather than the training-time ones (see
    is_eval_poisonable). indices holds the eligible positions in serving order. A
    caller that needs original dataset indices maps them back itself.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        labels: list[int],
        attack: Attack,
        normalize: Callable[[torch.Tensor], torch.Tensor],
        num_classes: int,
        source_only: bool = True,
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
        # A source-specific attack only claims to flip its source classes, and
        # measuring over every non-target class divides the true rate by the class
        # count. source_only=False builds the complement instead, whose success rate
        # is the false-trigger rate on classes the attack never claimed.
        if attack.source_classes is not None:
            sources = set(attack.source_classes)
            eligible = [
                i for i in eligible if (int(labels[i]) in sources) == source_only
            ]
        self.indices = eligible

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        index = self.indices[position]
        image, _ = self.base_dataset[index]  # image is (C, H, W) in 0 to 1

        plant = self.attack.apply_trigger_eval or self.attack.apply_trigger
        poisoned = plant(image, index)
        target = attack_success_label(
            self.attack.label_mode,
            int(self.labels[index]),
            self.attack.target_label,
            self.num_classes,
            self.attack.num_targets,
        )

        normalized = self.normalize(poisoned)  # (C, H, W)
        return normalized, target


class CoverPoisonedTrainingSet(Dataset):
    """A poisoned training set that also carries cover samples.

    Poisoned samples get the trigger and the poisoned label. Cover samples get the
    trigger (or the attack's own cover transform) and keep their label. Everything
    else stays clean.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        attack: Attack,
        poison_indices: set[int],
        cover_indices: set[int],
        normalize: Callable[[torch.Tensor], torch.Tensor],
        num_classes: int,
    ):
        self.base_dataset = base_dataset
        self.attack = attack
        self.poison_indices = poison_indices
        self.cover_indices = cover_indices
        self.normalize = normalize
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.base_dataset[index]  # image is (C, H, W) in 0 to 1

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

        normalized = self.normalize(image)  # (C, H, W)
        return normalized, label
