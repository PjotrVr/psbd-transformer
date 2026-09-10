"""Evaluation data loaders: clean and poisoned, over the full test set.

2 loader kinds, matching the 2 questions evaluation asks: how accurate is the
model on unpoisoned images, and how often does the trigger flip an eligible image
to the attacker's target. Poisoning here applies the trigger to every eligible
test image, never a poison_rate sample of it, since poison_rate only controls how
much of the training set is poisoned and evaluation has no use for it.

This is a different split policy from data.splits, which carves the same test set
into a heldout threshold set and an analysis pool for a detection sweep. Here
there is no split at all: both loaders serve the whole test set, because a
reported clean accuracy or ASR is a property of the model over all of it.
"""

import functools

import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, Dataset

from data.registry import DATASET_REGISTRY, DatasetSpec
from data.loading import (
    base_image_transform,
    extract_labels,
    limit_dataset,
    load_clean_datasets,
)
from data.backdoorbench import balance_by_class, split_validation_and_eval
from attacks.poisoning import Attack, AttackSuccessSet, PoisonedTrainingSet


@functools.lru_cache(maxsize=None)
def load_test_base(dataset_name: str, raw_data_dir: str) -> tuple[Dataset, DatasetSpec]:
    """The 0-to-1 test set and its spec, so a trigger can be applied before normalizing.

    Cached per (dataset_name, raw_data_dir): identical across every checkpoint
    that shares a dataset, and read-only, so evaluating many checkpoints in a loop
    only reads the underlying images from disk once per dataset. The spec is
    returned alongside because every caller needs its normalization statistics
    immediately afterwards.
    """
    spec = DATASET_REGISTRY[dataset_name]
    transform = base_image_transform(spec.image_size)

    _, test_base = load_clean_datasets(dataset_name, transform, raw_data_dir)
    return test_base, spec


def build_clean_loader(
    dataset_name: str,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
    num_workers: int = 2,
    max_samples: int | None = None,
    seed: int = 0,
) -> DataLoader:
    """Every test image, normalized, no trigger.

    max_samples is a smoke-test knob (None means the whole test set). Subsetting
    happens after load_test_base's cached return, never inside it, so the cache
    key stays (dataset_name, raw_data_dir) and a truncated call never corrupts the
    full-dataset object other callers share.
    """
    test_base, spec = load_test_base(dataset_name, raw_data_dir)
    test_base = limit_dataset(test_base, max_samples, seed)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    # With no poison_indices this is a plain normalized dataset. apply_trigger is
    # only reached for indices in that set, so the None attack is never touched.
    clean_set = PoisonedTrainingSet(test_base, None, set(), normalize, spec.num_classes)

    loader = DataLoader(
        clean_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return loader


def build_poisoned_loader(
    dataset_name: str,
    attack: Attack,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
    num_workers: int = 2,
    max_samples: int | None = None,
    seed: int = 0,
) -> DataLoader:
    """Every eligible test image with the trigger applied, labeled by attack success.

    Eligibility and the eval label both depend on the attack's label_mode:
    all_to_one drops the target class, all_to_all drops nothing, clean_label keeps
    only non-target images, the opposite of training eligibility, since eval asks
    whether the trigger fools a non-target image into the target, not which images
    training was allowed to poison (see attacks.poisoning.is_eval_poisonable).
    """
    test_base, spec = load_test_base(dataset_name, raw_data_dir)
    test_base = limit_dataset(test_base, max_samples, seed)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    true_labels = extract_labels(test_base)

    poisoned_set = AttackSuccessSet(
        test_base, true_labels, attack, normalize, spec.num_classes
    )

    loader = DataLoader(
        poisoned_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return loader


def build_balanced_eval_loaders(
    dataset_name: str,
    attack: Attack,
    image_size: int,
    clean_val_size: int,
    examples_per_class: int,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """The validation, clean and backdoor loaders, class-balanced, built in memory.

    A third split policy, distinct from the 2 above and from data.splits: it holds
    out a validation slice and then balances the clean and backdoor evaluation sets
    to the same count per class. Use it when a comparison across classes has to be
    free of the test set's own class imbalance.

    The split and the balancing both run on the 0-to-1 base test set, where labels
    are cheap to read, and the same base object is passed as clean and backdoor so
    the 2 eval sets stay index-aligned. Wrapping comes last: the clean sets only
    normalize, the backdoor set triggers every sample and then normalizes.
    """
    spec = DATASET_REGISTRY[dataset_name]
    base_transform = base_image_transform(image_size)
    _, test_base = load_clean_datasets(dataset_name, base_transform, raw_data_dir)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    clean_val_base, clean_eval_base, backdoor_eval_base = split_validation_and_eval(
        test_base, test_base, clean_val_size, seed
    )
    clean_eval_base, backdoor_eval_base = balance_by_class(
        clean_eval_base, backdoor_eval_base, examples_per_class, seed
    )

    def loader(dataset):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return (
        loader(
            PoisonedTrainingSet(
                clean_val_base, attack, set(), normalize, spec.num_classes
            )
        ),
        loader(
            PoisonedTrainingSet(
                clean_eval_base, attack, set(), normalize, spec.num_classes
            )
        ),
        loader(
            AttackSuccessSet(
                backdoor_eval_base,
                extract_labels(backdoor_eval_base),
                attack,
                normalize,
                spec.num_classes,
            )
        ),
    )
