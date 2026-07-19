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

LABEL_MODES = ("all_to_one", "all_to_all", "clean_label")


@dataclass(frozen=True)
class Attack:
    name: str
    apply_trigger: ApplyTrigger
    label_mode: str
    target_label: int


def is_poisonable(label_mode: str, original_label: int, target_label: int) -> bool:
    """Which samples an attack is allowed to poison.

    all_to_one poisons any sample not already the target class. all_to_all
    poisons any sample. clean_label poisons only target-class samples, since it
    must not change the label.
    """
    if label_mode == "all_to_one":
        return original_label != target_label
    if label_mode == "all_to_all":
        return True
    if label_mode == "clean_label":
        return original_label == target_label
    raise ValueError(f"Unknown label mode: {label_mode}")


def poisoned_label(label_mode: str, original_label: int, target_label: int, num_classes: int) -> int:
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
    if label_mode == "clean_label":
        return original_label
    raise ValueError(f"Unknown label mode: {label_mode}")


def is_eval_poisonable(label_mode: str, original_label: int, target_label: int) -> bool:
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
    if label_mode == "clean_label":
        return original_label != target_label
    raise ValueError(f"Unknown label mode: {label_mode}")


def attack_success_label(label_mode: str, original_label: int, target_label: int, num_classes: int) -> int:
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
    return poisoned_label(label_mode, original_label, target_label, num_classes)


def choose_poison_indices(
    labels: list[int], attack: Attack, poison_rate: float, seed: int
) -> set[int]:
    """Pick which dataset indices to poison at the requested rate.

    The rate is measured against the whole dataset. For clean_label the eligible
    pool is only the target class, so the count is capped by how many target
    samples exist.
    """
    eligible = [
        i for i, y in enumerate(labels)
        if is_poisonable(attack.label_mode, int(y), attack.target_label)
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
                self.attack.label_mode, int(label), self.attack.target_label, self.num_classes
            )
        return self.normalize(image), label


class AttackSuccessSet(Dataset):
    """Every eligible sample poisoned, for measuring attack success rate.

    The returned label is the attack's intended label per sample, so accuracy on
    this set is the ASR. Samples that cannot flip under the label mode are
    dropped so the ASR is measured only over samples that should flip. Uses the
    eval-time eligibility and label functions, not the training-time ones, since
    clean_label asks a different question at eval time (see is_eval_poisonable).
    """

    def __init__(self, base_dataset, labels, attack, normalize, num_classes):
        self.base_dataset = base_dataset
        self.labels = labels
        self.attack = attack
        self.normalize = normalize
        self.num_classes = num_classes
        self.indices = [
            i for i, y in enumerate(labels)
            if is_eval_poisonable(attack.label_mode, int(y), attack.target_label)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        image, _ = self.base_dataset[index]
        poisoned = self.attack.apply_trigger(image, index)
        target = attack_success_label(
            self.attack.label_mode, int(self.labels[index]), self.attack.target_label, self.num_classes
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
        return is_poisonable(attack.label_mode, label, attack.target_label)

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

    def __init__(self, base_dataset, attack, poison_indices, cover_indices, normalize, num_classes):
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
                self.attack.label_mode, int(label), self.attack.target_label, self.num_classes
            )
        elif index in self.cover_indices:
            image = self.attack.apply_trigger(image, index)
        return self.normalize(image), label
