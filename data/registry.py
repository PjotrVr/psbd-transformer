"""Static dataset facts, and the tunable parameters of a detection sweep.

DATASET_REGISTRY is the single place a dataset's class count, normalization
statistics, loader routing and native trigger resolution are written down, so
adding a dataset is a single entry here plus a loader branch in loading.py.
RunConfig holds what a detection sweep can vary. Nothing here reads from disk or
holds mutable state.
"""

import os
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    """Normalization statistics and loader routing for a dataset.

    image_size is the native resolution triggers are defined at, not the 224 the
    model consumes. The model wrapper upscales on its own, so a trigger stamped at
    image_size lands on the pixels the attack's paper describes.
    """

    num_classes: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    loader_kind: str  # which branch of loading.load_clean_datasets reads it
    image_size: int


# GTSRB uses identity normalization because BackdoorBench trains its GTSRB models
# on unnormalized inputs, and matching that avoids a train/eval mismatch. SVHN and
# EuroSAT are in the panel because a clean-label attack can only poison its target
# class, which caps its rate at |target class| / |train set|. 10 classes is the
# most that still reaches 10%. Statistics are measured on the training split.
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        num_classes=10,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        loader_kind="cifar10",
        image_size=32,
    ),
    "cifar100": DatasetSpec(
        num_classes=100,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2673, 0.2564, 0.2762),
        loader_kind="cifar100",
        image_size=32,
    ),
    "gtsrb": DatasetSpec(
        num_classes=43,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        loader_kind="gtsrb",
        image_size=32,
    ),
    "tiny": DatasetSpec(
        num_classes=200,
        mean=(0.4802, 0.4481, 0.3975),
        std=(0.2302, 0.2265, 0.2262),
        loader_kind="image_folder",
        image_size=64,
    ),
    "svhn": DatasetSpec(
        num_classes=10,
        mean=(0.4377, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970),
        loader_kind="svhn",
        image_size=32,
    ),
    "eurosat": DatasetSpec(
        num_classes=10,
        mean=(0.3444, 0.3803, 0.4078),
        std=(0.2027, 0.1369, 0.1156),
        loader_kind="eurosat",
        image_size=64,
    ),
}


# pre_residual perturbs each branch's output just before its residual add.
# post_residual perturbs the stream just after the add, which is the ConvNet
# placement of the original PSBD paper. Both insert fresh perturbation modules
# through the position registry rather than reusing a dropout the model already
# has. models.positions says why.

ARCHITECTURES = ("vit", "swin")


@dataclass
class RunConfig:
    """All tunable parameters for a detection sweep."""

    seed: int = 0
    trigger_label: int = 0
    batch_size: int = 16

    # Clean validation set drawn from the test split, matching the 5% of the
    # training set used by the PSBD paper (2000 of CIFAR-10's 50000).
    clean_val_size: int = 2000
    examples_per_class: int = 150

    forward_passes: int = 3  # number of stochastic dropout passes, k in the paper
    dropout_rates: tuple[float, ...] = tuple(i / 10.0 for i in range(1, 10))

    architecture: str = "vit"
    dropout_placement: str = "pre_residual"

    weights_dir: str = "backdoor_bench_checkpoints"
    raw_data_dir: str = "raw_data"
    results_root: str = "experiments"

    # Run the forward pass under bfloat16 autocast for speed and memory while
    # keeping all score arithmetic in float32. This replaces the notebook's
    # global torch.set_default_dtype(bfloat16), which silently downcast the
    # quantile thresholds and numpy round-trips too.
    use_bfloat16: bool = True

    attack_folders: tuple[str, ...] = field(default_factory=tuple)

    def results_dir(self) -> str:
        """Where this sweep's outputs go, a subdirectory per dropout placement."""
        directory = os.path.join(self.results_root, self.dropout_placement)
        return directory


def dataset_name_from_folder(folder_name: str) -> str:
    """BackdoorBench folder names encode the dataset as the first token.

    Example: "cifar10_wanet_0_1" resolves to "cifar10".
    """
    dataset_name = folder_name.split("_")[0]
    return dataset_name


# BackdoorBench's clean-label attacks. Their folder names carry no other marker,
# so this is the only way to tell them apart from the dirty-label default.
CLEAN_LABEL_ATTACK_TOKENS = ("sig", "lc")


def label_mode_from_folder(folder_name: str) -> str:
    """Recover a checkpoint's label mode from its folder name alone.

    This is the fallback for folders with no args.json sidecar, which is every
    folder under backdoor_bench_checkpoints/. Folder names encode
    dataset_attack_rate. An "a2a" token selects all_to_all and an "a2m<m>" token
    selects all_to_m. A clean-label attack token (sig, lc) selects clean_label,
    for example cifar10_sig_0_01 or cifar10_lc_0_01. Everything else defaults to
    all_to_one, BackdoorBench's standard dirty-label convention.
    """
    tokens = folder_name.split("_")

    is_clean_label = any(token in CLEAN_LABEL_ATTACK_TOKENS for token in tokens)
    if is_clean_label:
        return "clean_label"

    if any(re.fullmatch(r"a2m\d+", token) for token in tokens):
        return "all_to_m"

    if "a2a" in folder_name:
        return "all_to_all"
    return "all_to_one"
