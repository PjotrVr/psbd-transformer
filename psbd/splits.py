"""The standardized PSBD split, rebuilt in memory from a checkpoint's provenance.

BackdoorBench ships a bd_test_dataset folder of poisoned PNGs. Our own training
does not, because our triggers are defined in code, so the poisoned test set is
rebuilt from the attack recorded in the checkpoint's args.json sidecar.

The split itself is a plain shuffle of the full clean test set, derived from
(test set size, seed) only, never from the attack or the checkpoint. Every
checkpoint of a dataset therefore sees the identical split, and hundreds of
thousands of cached per-sample tensors are ordered by that one permutation.
Anything that changes what psbd_split_permutation returns invalidates all of
them silently, so treat those two lines as frozen.
"""

import json
import os

import torch
import torchvision.transforms.v2 as transforms_v2
from lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, Subset

# The registry returns an attack record with .name, .apply_trigger, .label_mode,
# and .target_label. The dataset wrappers below read only those 4 fields and do
# no isinstance check, so the record's defining module does not matter.
from .attacks import build_attack, default_config

from .config import DATASET_REGISTRY
from .data import base_image_transform, extract_labels, load_clean_datasets
from .poisoning import AttackSuccessSet, PoisonedTrainingSet

# The one standardized split seed. The clean test set is carved into a heldout
# threshold set and an analysis pool as a function of (test set, seed) only, never
# of the attack or checkpoint, so every checkpoint of a dataset sees the identical
# split.
PSBD_SPLIT_SEED = 0
PSBD_HELDOUT_SIZE = 2000

REQUIRED_METADATA_KEYS = ("dataset", "attack", "target_label", "architecture")


def read_checkpoint_metadata(checkpoint_path: str) -> dict:
    """Read the args.json training provenance saved alongside the checkpoint.

    Raises FileNotFoundError when the sidecar is absent and KeyError when it is
    present but incomplete, because both cases mean the eval set cannot be
    rebuilt and guessing would produce a plausible wrong number.
    """
    args_path = os.path.join(os.path.dirname(checkpoint_path), "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(
            f"{args_path} not found. Retrain with train_backdoor.py or train_benign.py, "
            "which write it, or provide a BackdoorBench bd_test_dataset folder instead."
        )

    with open(args_path) as handle:
        metadata = json.load(handle)

    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
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
    checkpoint_attack = metadata["attack"]

    if checkpoint_attack != "benign":
        if probe_attack is not None and probe_attack != checkpoint_attack:
            raise ValueError(
                f"checkpoint was trained with {checkpoint_attack!r}; refusing to "
                f"probe it with {probe_attack!r}, which it never saw"
            )
        return checkpoint_attack, metadata["target_label"]

    if probe_attack is None:
        raise ValueError(
            "a benign checkpoint has no attack of its own, so the backdoor split "
            "is undefined. Pass --probe-attack (and --probe-target-label) to name "
            "the trigger to probe it with; chance-level detection is the expected "
            "result and is the negative control for the whole sweep."
        )

    resolved_target = probe_target_label if probe_target_label is not None else 0
    return probe_attack, resolved_target


def check_label_mode_agrees(
    checkpoint_path: str, metadata: dict, attack_name: str, built_label_mode: str
) -> None:
    """Fail when the recorded label mode contradicts the one the registry builds.

    The recorded label_mode is written by training and the registry derives one
    from the attack name. A disagreement means the eval set is being built for a
    different attack than the one trained, and every number that follows is wrong
    under a correct-looking label.
    """
    recorded_mode = metadata.get("label_mode")
    if metadata["attack"] == "benign" or recorded_mode is None:
        return

    if built_label_mode != recorded_mode:
        raise ValueError(
            f"{checkpoint_path} records label_mode={recorded_mode!r} but "
            f"attack {attack_name!r} builds {built_label_mode!r}"
        )


def psbd_split_permutation(n_total: int, seed: int = PSBD_SPLIT_SEED) -> torch.Tensor:
    """The one permutation the whole PSBD split derives from, shape (n_total,).

    seed_everything then torch.randperm as the very first RNG consumption, so a
    notebook that reloads the same full test set and reruns these two lines
    recovers exactly which original test index maps to which saved tensor row.
    This is the single definition both the sweep and any reproduction call, so
    the two can never drift.
    """
    seed_everything(seed)
    permutation = torch.randperm(n_total)
    return permutation


def load_clean_test_base(dataset_name: str, raw_data_dir: str) -> Dataset:
    """The full clean test set in its native, deterministic 0-to-1 order.

    Resized to the dataset's native resolution, where pixel-space triggers are
    defined, before the model's own wrapper upscales to 224. Unnormalized, so the
    dataset wrappers can normalize after stamping the trigger.
    """
    spec = DATASET_REGISTRY[dataset_name]
    transform = base_image_transform(spec.image_size)
    _, test_base = load_clean_datasets(dataset_name, transform, raw_data_dir)
    return test_base


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
    """Build the 3 PSBD splits by the standardized shuffle, plus a manifest.

    A plain shuffle of the full clean test set, no stratification, no per-class
    balancing: heldout is the first 2000 of the permutation (the clean threshold
    set), analysis is the rest (the paired clean and backdoor analysis pool). 0
    leakage by construction, since heldout and analysis are disjoint slices of one
    permutation and the threshold set is clean only.

    Every loader is shuffle=False, so row i of any split's saved tensor maps to
    that split's manifest index i with no hidden reordering. The manifest records
    the resolved original test indices per split, the ground-truth row order for
    every tensor written under this checkpoint's psbd/ subtree.

    max_samples truncates each split to its first rows, for smoke and timing runs
    only, kept internally consistent so the manifest still describes what the
    loaders serve. Real jobs leave it None and get the full split.
    """
    metadata = read_checkpoint_metadata(checkpoint_path)
    dataset_name = metadata["dataset"]
    spec = DATASET_REGISTRY[dataset_name]

    attack_name, target_label = resolve_probe_attack(
        metadata, probe_attack, probe_target_label
    )
    attack = build_attack(
        attack_name, default_config(attack_name), spec.image_size, target_label
    )
    check_label_mode_agrees(checkpoint_path, metadata, attack_name, attack.label_mode)

    test_base = load_clean_test_base(dataset_name, raw_data_dir)
    n_total = len(test_base)

    permutation = psbd_split_permutation(n_total, seed)  # (n_total,)
    heldout_indices = permutation[:PSBD_HELDOUT_SIZE]
    analysis_indices = permutation[PSBD_HELDOUT_SIZE:]
    if max_samples is not None:
        heldout_indices = heldout_indices[:max_samples]
        analysis_indices = analysis_indices[:max_samples]

    heldout_list = heldout_indices.tolist()
    analysis_list = analysis_indices.tolist()
    assert set(heldout_list).isdisjoint(analysis_list), (
        "heldout and analysis must be disjoint slices of one permutation"
    )

    # Normalization is constructed here rather than baked into the base transform
    # so it can be applied after the trigger, in pixel space (see psbd.data).
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    validation_base = Subset(test_base, heldout_list)
    analysis_base = Subset(test_base, analysis_list)
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

    # backdoor_set.indices are the eligible positions within the analysis subset.
    # Mapping them back through analysis_list gives their original test indices,
    # in the same order the backdoor loader serves them.
    analysis_backdoor_indices = [
        analysis_list[position] for position in backdoor_set.indices
    ]

    def loader(dataset: Dataset) -> DataLoader:
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
