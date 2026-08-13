"""Building PSBD evaluation loaders for our own trained checkpoints.

BackdoorBench ships a bd_test_dataset folder of poisoned PNGs that the sweep reads.
Our own training does not, because our triggers are defined in code, so we rebuild
the poisoned test set in memory from the attack recorded in the checkpoint. The
validation, clean eval, and backdoor eval split reuses the same functions the PNG
path uses, so both paths produce the same three-way structure.
"""

import json
import os

import torch
import torchvision.transforms.v2 as transforms_v2
from lightning import seed_everything
from torch.utils.data import DataLoader, Subset

from attacks import build_attack, default_config
from backdoor_data import balance_by_class, split_validation_and_eval
from utils.config import DATASET_REGISTRY, RunConfig
from utils.datasets import extract_labels, load_clean_datasets
from poison import Attack, AttackSuccessSet, PoisonedTrainingSet

# The one standardized split seed. The clean test set is carved into a heldout
# threshold set and an analysis pool as a function of (test set, seed) only, never
# of the attack or checkpoint, so every checkpoint of a dataset sees the identical
# split.
PSBD_SPLIT_SEED = 0
PSBD_HELDOUT_SIZE = 2000


def read_checkpoint_metadata(checkpoint_path: str) -> dict:
    """Read the args.json training provenance saved alongside the checkpoint."""
    args_path = os.path.join(os.path.dirname(checkpoint_path), "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(
            f"{args_path} not found. Retrain with train_backdoor.py or train_benign.py, "
            "which write it, or provide a BackdoorBench bd_test_dataset folder instead."
        )
    with open(args_path) as handle:
        metadata = json.load(handle)
    required = ("dataset", "attack", "target_label")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise KeyError(f"{args_path} is missing {missing}")
    return metadata


def build_eval_loaders_from_attack(
    dataset_name: str, attack: Attack, config: RunConfig, image_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Assemble the validation, clean, and backdoor loaders in memory.

    The split and balance run on the 0-to-1 base test set, where labels are cheap
    to read. Passing the same base as both clean and backdoor keeps the eval indices
    aligned. Wrapping happens last: the clean sets only normalize, and the backdoor
    set triggers every sample then normalizes.
    """
    spec = DATASET_REGISTRY[dataset_name]
    base_transform = transforms_v2.Compose(
        [transforms_v2.Resize((image_size, image_size)), transforms_v2.ToTensor()]
    )
    _, test_base = load_clean_datasets(
        dataset_name, base_transform, config.raw_data_dir
    )
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    clean_val_base, clean_eval_base, backdoor_eval_base = split_validation_and_eval(
        test_base, test_base, config.clean_val_size, config.seed
    )
    clean_eval_base, backdoor_eval_base = balance_by_class(
        clean_eval_base, backdoor_eval_base, config.examples_per_class, config.seed
    )

    clean_val = PoisonedTrainingSet(
        clean_val_base, attack, set(), normalize, spec.num_classes
    )
    clean_eval = PoisonedTrainingSet(
        clean_eval_base, attack, set(), normalize, spec.num_classes
    )
    backdoor_eval = AttackSuccessSet(
        backdoor_eval_base,
        extract_labels(backdoor_eval_base),
        attack,
        normalize,
        spec.num_classes,
    )

    def loader(dataset):
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    return loader(clean_val), loader(clean_eval), loader(backdoor_eval)


def build_eval_loaders_from_checkpoint(
    checkpoint_path: str, config: RunConfig
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Read the checkpoint metadata and rebuild the eval loaders for that attack.

    Uses the attack's default config, which matches how train_backdoor.py builds
    it. A custom attack config would need to be recorded in the checkpoint too.
    """
    metadata = read_checkpoint_metadata(checkpoint_path)
    dataset_name = metadata["dataset"]
    image_size = DATASET_REGISTRY[dataset_name].image_size
    attack = build_attack(
        metadata["attack"],
        default_config(metadata["attack"]),
        image_size,
        metadata["target_label"],
    )
    return build_eval_loaders_from_attack(dataset_name, attack, config, image_size)


def psbd_split_permutation(n_total: int, seed: int = PSBD_SPLIT_SEED) -> torch.Tensor:
    """The one permutation the whole PSBD split derives from.

    seed_everything then torch.randperm as the very first RNG consumption, so a
    notebook that reloads the same full test set and reruns these two lines
    recovers exactly which original test index maps to which saved tensor row.
    This is the single definition both the sweep and any reproduction call, so
    the two can never drift.
    """
    seed_everything(seed)
    return torch.randperm(n_total)


def _load_clean_test_base(dataset_name: str, raw_data_dir: str):
    """The full clean test set in its native, deterministic 0-to-1 order.

    Resized to the dataset's native resolution, where pixel-space triggers are
    defined, before the model's own wrapper upscales to 224. Same base the PNG
    and existing in-memory eval paths use.
    """
    spec = DATASET_REGISTRY[dataset_name]
    base_transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((spec.image_size, spec.image_size)),
            transforms_v2.ToTensor(),
        ]
    )
    _, test_base = load_clean_datasets(dataset_name, base_transform, raw_data_dir)
    return test_base, spec


def build_psbd_loaders_from_checkpoint(
    checkpoint_path: str,
    seed: int = PSBD_SPLIT_SEED,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
    num_workers: int = 2,
    max_samples: int | None = None,
) -> tuple[dict[str, DataLoader], dict]:
    """Build the three PSBD splits by the standardized shuffle, plus a manifest.

    A plain shuffle of the full clean test set, no stratification, no per-class
    balancing (this deliberately departs from build_eval_loaders_from_attack's
    stratified split_validation_and_eval and balance_by_class): heldout is the
    first 2000 of the permutation (the clean threshold set), analysis is the
    rest (the paired clean and backdoor analysis pool). 0 leakage by
    construction, since heldout and analysis are disjoint slices of one
    permutation and the threshold set is clean only.

    Every loader is shuffle=False, so row i of any split's saved tensor maps to
    that split's manifest index i with no hidden reordering. The manifest records
    the resolved original test indices per split, the ground-truth row order for
    every tensor written under this checkpoint's psbd/ subtree.

    max_samples truncates each split to its first rows, for smoke and timing runs
    only, kept internally consistent so the manifest still describes what the
    loaders serve. The 60 real jobs leave it None and get the full split.
    """
    metadata = read_checkpoint_metadata(checkpoint_path)
    dataset_name = metadata["dataset"]
    attack = build_attack(
        metadata["attack"],
        default_config(metadata["attack"]),
        DATASET_REGISTRY[dataset_name].image_size,
        metadata["target_label"],
    )

    base, spec = _load_clean_test_base(dataset_name, raw_data_dir)
    n_total = len(base)
    permutation = psbd_split_permutation(n_total, seed)
    heldout_indices = permutation[:PSBD_HELDOUT_SIZE]
    analysis_indices = permutation[PSBD_HELDOUT_SIZE:]
    if max_samples is not None:
        heldout_indices = heldout_indices[:max_samples]
        analysis_indices = analysis_indices[:max_samples]

    heldout_list = heldout_indices.tolist()
    analysis_list = analysis_indices.tolist()
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    validation_base = Subset(base, heldout_list)
    analysis_base = Subset(base, analysis_list)
    analysis_labels = extract_labels(analysis_base)

    validation_set = PoisonedTrainingSet(
        validation_base, attack, set(), normalize, spec.num_classes
    )
    clean_set = PoisonedTrainingSet(
        analysis_base, attack, set(), normalize, spec.num_classes
    )
    backdoor_set = AttackSuccessSet(
        analysis_base, analysis_labels, attack, normalize, spec.num_classes
    )

    # backdoor_set.indices are the eligible positions within the analysis subset;
    # mapping them back through analysis_list gives their original test indices,
    # in the same order the backdoor loader serves them.
    analysis_backdoor_indices = [
        analysis_list[position] for position in backdoor_set.indices
    ]

    def loader(dataset):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    loaders = {
        "validation": loader(validation_set),
        "clean": loader(clean_set),
        "backdoor": loader(backdoor_set),
    }
    manifest = {
        "seed": seed,
        "dataset": dataset_name,
        "n_total": n_total,
        "n_heldout": len(heldout_list),
        "heldout_indices": heldout_list,
        "analysis_clean_indices": analysis_list,
        "analysis_backdoor_indices": analysis_backdoor_indices,
        "recipe_note": (
            "seed_everything(seed); perm = torch.randperm(n_total); "
            "heldout = perm[:n_heldout]; analysis = perm[n_heldout:]"
        ),
    }
    return loaders, manifest
