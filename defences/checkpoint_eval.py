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

from attacks import apply_config_overrides, build_attack, default_config
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
    required = ("dataset", "attack", "target_label", "architecture")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise KeyError(f"{args_path} is missing {missing}")
    return metadata


def resolve_probe_attack(
    metadata: dict, probe_attack: str | None, probe_target_label: int | None
) -> tuple[str, int]:
    """The (attack name, target label) whose trigger defines the backdoor split.

    Normally the checkpoint's own attack. A benign checkpoint has none, and
    build_attack("benign", ...) raises, so without an override the negative
    control cannot be run at all. That control is the one that separates "PSBD
    detects a backdoor" from "the probe reacts to any trigger-shaped perturbation
    on any model", so it has to be reachable: a benign model is probed with an
    externally named trigger, and the expected result is chance-level detection.

    The override is rejected on a backdoored checkpoint, where probing with the
    wrong trigger would silently measure the response to a backdoor the model
    never learned.
    """
    if metadata["attack"] != "benign":
        if probe_attack is not None and probe_attack != metadata["attack"]:
            raise ValueError(
                f"checkpoint was trained with {metadata['attack']!r}; refusing to "
                f"probe it with {probe_attack!r}, which it never saw"
            )
        return metadata["attack"], metadata["target_label"]

    if probe_attack is None:
        raise ValueError(
            "a benign checkpoint has no attack of its own, so the backdoor split "
            "is undefined. Pass --probe-attack (and --probe-target-label) to name "
            "the trigger to probe it with; chance-level detection is the expected "
            "result and is the negative control for the whole sweep."
        )
    return probe_attack, probe_target_label if probe_target_label is not None else 0


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
    it. A custom attack config is recorded in the checkpoint as
    attack_config_overrides and reapplied here.
    """
    metadata = read_checkpoint_metadata(checkpoint_path)
    dataset_name = metadata["dataset"]
    image_size = DATASET_REGISTRY[dataset_name].image_size
    attack = build_attack(
        metadata["attack"],
        apply_config_overrides(
            default_config(metadata["attack"]),
            metadata.get("attack_config_overrides"),
        ),
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
    probe_attack: str | None = None,
    probe_target_label: int | None = None,
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
    attack_name, target_label = resolve_probe_attack(
        metadata, probe_attack, probe_target_label
    )
    # A probe attack is chosen here rather than trained, so it takes no override; a
    # backdoored checkpoint rebuilds the exact trigger it was trained with.
    overrides = (
        metadata.get("attack_config_overrides")
        if attack_name == metadata.get("attack")
        else None
    )
    attack = build_attack(
        attack_name,
        apply_config_overrides(default_config(attack_name), overrides),
        DATASET_REGISTRY[dataset_name].image_size,
        target_label,
    )
    # The recorded label_mode is written by training and the registry derives one
    # from the attack name; if they ever disagree, the eval set is built for a
    # different attack than the one trained, and every number is wrong under a
    # correct-looking label.
    recorded_mode = metadata.get("label_mode")
    if metadata["attack"] != "benign" and recorded_mode is not None:
        if attack.label_mode != recorded_mode:
            raise ValueError(
                f"{checkpoint_path} records label_mode={recorded_mode!r} but "
                f"attack {attack_name!r} builds {attack.label_mode!r}"
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
        "probe_attack": attack_name,
        "probe_target_label": target_label,
        "label_mode": attack.label_mode,
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
