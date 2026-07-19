"""Building PSBD evaluation loaders for our own trained checkpoints.

BackdoorBench ships a bd_test_dataset folder of poisoned PNGs that the sweep reads.
Our own training does not, because our triggers are defined in code, so we rebuild
the poisoned test set in memory from the attack recorded in the checkpoint. The
validation, clean eval, and backdoor eval split reuses the same functions the PNG
path uses, so both paths produce the same three-way structure.
"""

import json
import os

import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader

from attacks import build_attack, default_config
from backdoor_data import balance_by_class, split_validation_and_eval
from utils.config import DATASET_REGISTRY, RunConfig
from utils.datasets import extract_labels, load_clean_datasets
from poison import Attack, AttackSuccessSet, PoisonedTrainingSet


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
