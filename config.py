import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    """Normalization statistics and loader routing for one dataset."""

    num_classes: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    loader_kind: str  # "cifar10", "cifar100", "gtsrb", or "image_folder"


# GTSRB uses identity normalization because BackdoorBench trains its GTSRB
# models on unnormalized inputs, so matching that avoids a train/eval mismatch.
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        10, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), "cifar10"
    ),
    "cifar100": DatasetSpec(
        100, (0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2762), "cifar100"
    ),
    "gtsrb": DatasetSpec(43, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), "gtsrb"),
    "tiny": DatasetSpec(
        200, (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262), "image_folder"
    ),
}


# pre_residual toggles the Dropout modules torchvision already builds into the
# transformer, all of which sit on a block's branch before the residual add.
# post_residual inserts fresh Dropout after each residual add, mirroring the
# ConvNet placement in the original PSBD paper.
DROPOUT_PLACEMENTS = ("pre_residual", "post_residual")

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
    psbd_quantiles: tuple[float, ...] = (0.1, 0.15, 0.2, 0.25)

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
        return os.path.join(self.results_root, self.dropout_placement)


def dataset_name_from_folder(folder_name: str) -> str:
    """BackdoorBench folder names encode the dataset as the first token.

    Example: "cifar10_wanet_0_1" resolves to "cifar10".
    """
    return folder_name.split("_")[0]


# BackdoorBench's clean-label attacks. Their folder names carry no other marker,
# so this is the only way to tell them apart from the dirty-label default.
CLEAN_LABEL_ATTACK_TOKENS = ("sig", "lc")


def label_mode_from_folder(folder_name: str) -> str:
    """BackdoorBench folder names encode dataset_attack_rate.

    "a2a" selects all_to_all. A clean-label attack token (sig, lc) selects
    clean_label, for example cifar10_sig_0_01 or cifar10_lc_0_01 under
    backdoor_bench_checkpoints/. Everything else defaults to all_to_one,
    BackdoorBench's standard dirty-label convention.
    """
    tokens = folder_name.split("_")
    if any(token in CLEAN_LABEL_ATTACK_TOKENS for token in tokens):
        return "clean_label"
    return "all_to_all" if "a2a" in folder_name else "all_to_one"
