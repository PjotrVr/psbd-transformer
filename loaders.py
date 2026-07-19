"""Evaluation data loaders: clean and poisoned, over the full test set.

Two loader kinds, matching the two questions evaluation asks: how accurate is
the model on unpoisoned images, and how often does the trigger flip an
eligible image to the attacker's target. Poisoning here always applies the
trigger to every eligible test image, never a poison_rate sample of it,
since poison_rate only controls how much of the training set gets poisoned,
a training-time concept eval has no use for.
"""

import functools

import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader

from poison import Attack, AttackSuccessSet, PoisonedTrainingSet
from utils.config import DATASET_REGISTRY
from utils.datasets import extract_labels, load_clean_datasets


@functools.lru_cache(maxsize=None)
def _load_test_base(dataset_name: str, raw_data_dir: str) -> tuple:
    """0-to-1 pixel range test set, so a trigger can be applied before normalizing.

    Cached per (dataset_name, raw_data_dir): identical across every checkpoint
    that shares a dataset, and read-only, so evaluating many checkpoints in a
    loop only reads the underlying images from disk once per dataset.
    """
    spec = DATASET_REGISTRY[dataset_name]
    transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((spec.image_size, spec.image_size)),
            transforms_v2.ToTensor(),
        ]
    )
    _, test_base = load_clean_datasets(dataset_name, transform, raw_data_dir)
    return test_base, spec


def build_clean_loader(
    dataset_name: str,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
    num_workers: int = 2,
) -> DataLoader:
    """Every test image, normalized, no trigger."""
    test_base, spec = _load_test_base(dataset_name, raw_data_dir)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    # PoisonedTrainingSet with no poison_indices and no attack is just a plain
    # normalized dataset: attack.apply_trigger is only ever called for indices
    # inside poison_indices, which is empty here, so attack is never touched.
    clean_set = PoisonedTrainingSet(test_base, None, set(), normalize, spec.num_classes)
    return DataLoader(
        clean_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def build_poisoned_loader(
    dataset_name: str,
    attack: Attack,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
    num_workers: int = 2,
) -> DataLoader:
    """Every eligible test image with the trigger applied, labeled by attack success.

    Eligibility and the eval label both depend on the attack's label_mode:
    all_to_one drops the target class, all_to_all drops nothing, clean_label
    keeps only non-target images, the opposite of training eligibility, since
    eval asks whether the trigger fools a non-target image into the target,
    not which images training was allowed to poison (see poison.is_eval_poisonable).
    """
    test_base, spec = _load_test_base(dataset_name, raw_data_dir)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    true_labels = extract_labels(test_base)
    poisoned_set = AttackSuccessSet(
        test_base, true_labels, attack, normalize, spec.num_classes
    )
    return DataLoader(
        poisoned_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
