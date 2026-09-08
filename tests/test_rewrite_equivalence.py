"""Equivalence of the psbd/ rewrite against the modules it replaces.

The rewrite is a cutover, not a reimplementation: until every call site moves,
the old modules stay the source of truth and the new ones must agree with them
exactly. The split manifest is the strictest case. Hundreds of thousands of
cached per-sample tensors are ordered by one permutation, so a manifest that
drifts by a single index silently invalidates every result already on disk.

The checkpoint-backed tests need raw_data/ and an on-disk checkpoint, so they
skip when either is absent. The pure-function tests always run.
"""

import os

import numpy as np
import pytest
import torch
import torchvision.transforms.v2 as transforms_v2
from PIL import Image
from torch.utils.data import Subset, TensorDataset

import models as old_models
import poison as old_poison
from defences import checkpoint_eval as old_splits
from psbd import config as new_config
from psbd import data as new_data
from psbd import models as new_models
from psbd import poisoning as new_poisoning
from psbd import splits as new_splits
from utils import config as old_config
from utils import datasets as old_data

RAW_DATA_DIR = "raw_data"

# Chosen to span the axes the manifest actually depends on: 2 datasets, 2
# architectures, and both an all_to_one and a clean_label attack. clean_label is
# the one whose eval eligibility is the opposite of its training eligibility, so
# it is the case where mixing up the two function pairs would show up in
# analysis_backdoor_indices.
MANIFEST_CHECKPOINTS = (
    "checkpoints/vit_cifar100_wanet_0_1/attack_result.pt",
    "checkpoints/swin_cifar100_sig_0_01/attack_result.pt",
    "checkpoints/vit_cifar10_badnet_a2o_0_01/attack_result.pt",
)

ARCHITECTURE_CHECKPOINTS = (
    ("vit", "checkpoints/vit_cifar100_wanet_0_1/attack_result.pt"),
    ("vit", "checkpoints/vit_cifar10_badnet_a2o_0_01/attack_result.pt"),
    ("swin", "checkpoints/swin_cifar100_sig_0_01/attack_result.pt"),
)

SWEEP_CLASS_COUNT = 7


def _skip_unless_present(*paths: str) -> None:
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        pytest.skip(f"not on disk: {missing}")


def test_dataset_registry_is_field_for_field_identical():
    assert set(new_config.DATASET_REGISTRY) == set(old_config.DATASET_REGISTRY)
    for name, old_spec in old_config.DATASET_REGISTRY.items():
        new_spec = new_config.DATASET_REGISTRY[name]
        assert new_spec.num_classes == old_spec.num_classes
        assert new_spec.mean == old_spec.mean
        assert new_spec.std == old_spec.std
        assert new_spec.loader_kind == old_spec.loader_kind
        assert new_spec.image_size == old_spec.image_size


def test_run_config_defaults_and_results_dir_match():
    old_run = old_config.RunConfig()
    new_run = new_config.RunConfig()
    assert vars(new_run) == vars(old_run)
    assert new_run.results_dir() == old_run.results_dir()

    old_custom = old_config.RunConfig(
        dropout_placement="post_residual", results_root="x"
    )
    new_custom = new_config.RunConfig(
        dropout_placement="post_residual", results_root="x"
    )
    assert new_custom.results_dir() == old_custom.results_dir()


def test_folder_name_parsers_agree_on_real_folder_names():
    folder_names = (
        sorted(os.listdir("checkpoints")) if os.path.isdir("checkpoints") else []
    )
    synthetic = [
        "cifar10_wanet_0_1",
        "cifar10_sig_0_01",
        "cifar10_lc_0_01",
        "cifar100_badnet_a2a_0_1",
        "tiny_blend_0_05",
        "gtsrb_signal_0_01",  # "sig" must not match as a substring of another token
    ]
    for folder_name in folder_names + synthetic:
        assert new_config.dataset_name_from_folder(
            folder_name
        ) == old_config.dataset_name_from_folder(folder_name)
        assert new_config.label_mode_from_folder(
            folder_name
        ) == old_config.label_mode_from_folder(folder_name)


def test_label_mode_functions_agree_over_exhaustive_sweep():
    """All 4 label-policy functions, every (mode, original, target) triple."""
    checked = 0
    for label_mode in old_poison.LABEL_MODES:
        # all_to_m is the 1 mode whose behaviour depends on a 5th argument, so the
        # sweep varies it rather than pinning it: m = 1 must reproduce all_to_one and
        # m = SWEEP_CLASS_COUNT must reproduce all_to_all, and both are covered here.
        targets = (
            range(1, SWEEP_CLASS_COUNT + 1) if label_mode == "all_to_m" else (None,)
        )
        for num_targets in targets:
            for original_label in range(SWEEP_CLASS_COUNT):
                for target_label in range(SWEEP_CLASS_COUNT):
                    assert new_poisoning.is_poisonable(
                        label_mode, original_label, target_label, num_targets
                    ) == old_poison.is_poisonable(
                        label_mode, original_label, target_label, num_targets
                    )

                    assert new_poisoning.poisoned_label(
                        label_mode,
                        original_label,
                        target_label,
                        SWEEP_CLASS_COUNT,
                        num_targets,
                    ) == old_poison.poisoned_label(
                        label_mode,
                        original_label,
                        target_label,
                        SWEEP_CLASS_COUNT,
                        num_targets,
                    )

                    assert new_poisoning.is_eval_poisonable(
                        label_mode, original_label, target_label, num_targets
                    ) == old_poison.is_eval_poisonable(
                        label_mode, original_label, target_label, num_targets
                    )

                    assert new_poisoning.attack_success_label(
                        label_mode,
                        original_label,
                        target_label,
                        SWEEP_CLASS_COUNT,
                        num_targets,
                    ) == old_poison.attack_success_label(
                        label_mode,
                        original_label,
                        target_label,
                        SWEEP_CLASS_COUNT,
                        num_targets,
                    )
                    checked += 1

    non_m = len(old_poison.LABEL_MODES) - 1
    expected = (non_m + SWEEP_CLASS_COUNT) * SWEEP_CLASS_COUNT**2
    assert checked == expected


def test_clean_label_eval_eligibility_is_the_opposite_of_training():
    """The correctness rule the rewrite most easily breaks, pinned directly."""
    target = 3
    for original_label in range(SWEEP_CLASS_COUNT):
        trains = new_poisoning.is_poisonable("clean_label", original_label, target)
        evaluates = new_poisoning.is_eval_poisonable(
            "clean_label", original_label, target
        )
        assert trains != evaluates
        assert trains == (original_label == target)

    # attack_success_label must not delegate to poisoned_label for clean_label.
    assert (
        new_poisoning.attack_success_label("clean_label", 5, target, SWEEP_CLASS_COUNT)
        == target
    )
    assert (
        new_poisoning.poisoned_label("clean_label", 5, target, SWEEP_CLASS_COUNT) == 5
    )


def test_unknown_label_mode_raises_in_both():
    for old_fn, new_fn in (
        (old_poison.is_poisonable, new_poisoning.is_poisonable),
        (old_poison.is_eval_poisonable, new_poisoning.is_eval_poisonable),
    ):
        with pytest.raises(ValueError):
            old_fn("nonsense", 0, 1)
        with pytest.raises(ValueError):
            new_fn("nonsense", 0, 1)


def _attack_pair(label_mode: str, target_label: int):
    """The same attack record built from the old and the new dataclass."""

    def apply_trigger(image, index):
        return image

    old_attack = old_poison.Attack(
        name="stub",
        apply_trigger=apply_trigger,
        label_mode=label_mode,
        target_label=target_label,
    )
    new_attack = new_poisoning.Attack(
        name="stub",
        apply_trigger=apply_trigger,
        label_mode=label_mode,
        target_label=target_label,
    )
    return old_attack, new_attack


def test_choose_poison_indices_is_identical():
    rng = np.random.default_rng(1234)
    labels = [int(y) for y in rng.integers(0, SWEEP_CLASS_COUNT, size=5000)]

    combinations = [
        ("all_to_one", 0, 0.01, 0),
        ("all_to_one", 3, 0.1, 7),
        ("all_to_all", 0, 0.05, 0),
        ("all_to_all", 2, 0.2, 13),
        ("clean_label", 0, 0.01, 0),
        ("clean_label", 4, 0.5, 99),  # rate above the eligible pool, so it caps
        ("all_to_one", 1, 0.0, 5),  # 0 samples requested
    ]
    for label_mode, target_label, poison_rate, seed in combinations:
        old_attack, new_attack = _attack_pair(label_mode, target_label)
        old_indices = old_poison.choose_poison_indices(
            labels, old_attack, poison_rate, seed
        )
        new_indices = new_poisoning.choose_poison_indices(
            labels, new_attack, poison_rate, seed
        )
        assert new_indices == old_indices, (label_mode, poison_rate, seed)


def test_choose_indices_with_cover_is_identical():
    rng = np.random.default_rng(4321)
    labels = [int(y) for y in rng.integers(0, SWEEP_CLASS_COUNT, size=3000)]

    combinations = [
        ("all_to_one", 0, 0.01, 0.01, None, 0),
        ("all_to_one", 2, 0.05, 0.02, (1, 3, 5), 11),
        ("clean_label", 1, 0.02, 0.03, None, 21),
        ("all_to_one", 0, 0.0, 0.0, None, 31),  # both draws skipped
    ]
    for label_mode, target, poison_rate, cover_rate, sources, seed in combinations:
        old_attack, new_attack = _attack_pair(label_mode, target)
        old_pair = old_poison.choose_indices_with_cover(
            labels, old_attack, poison_rate, cover_rate, sources, seed
        )
        new_pair = new_poisoning.choose_indices_with_cover(
            labels, new_attack, poison_rate, cover_rate, sources, seed
        )
        assert new_pair == old_pair, (label_mode, poison_rate, cover_rate, seed)


def _tensor_dataset(labels: list[int]) -> TensorDataset:
    images = torch.arange(len(labels), dtype=torch.float32).view(-1, 1, 1, 1)
    return TensorDataset(images, torch.tensor(labels))


def test_dataset_wrappers_serve_identical_rows():
    rng = np.random.default_rng(7)
    labels = [int(y) for y in rng.integers(0, SWEEP_CLASS_COUNT, size=200)]
    base = _tensor_dataset(labels)
    poison_indices = {3, 17, 42, 101}
    cover_indices = {5, 19}

    def add_one(image, index):
        return image + 1.0

    def normalize(image):
        return image * 2.0

    for label_mode in old_poison.LABEL_MODES:
        # all_to_m carries m on the Attack; every other mode ignores it.
        num_targets = SWEEP_CLASS_COUNT if label_mode == "all_to_m" else None
        old_attack = old_poison.Attack(
            "stub", add_one, label_mode, 2, num_targets=num_targets
        )
        new_attack = new_poisoning.Attack(
            "stub", add_one, label_mode, 2, num_targets=num_targets
        )

        old_train = old_poison.PoisonedTrainingSet(
            base, old_attack, poison_indices, normalize, SWEEP_CLASS_COUNT
        )
        new_train = new_poisoning.PoisonedTrainingSet(
            base, new_attack, poison_indices, normalize, SWEEP_CLASS_COUNT
        )
        assert len(new_train) == len(old_train)
        for index in range(len(old_train)):
            old_image, old_label = old_train[index]
            new_image, new_label = new_train[index]
            assert torch.equal(new_image, old_image)
            assert int(new_label) == int(old_label)

        old_asr = old_poison.AttackSuccessSet(
            base, labels, old_attack, normalize, SWEEP_CLASS_COUNT
        )
        new_asr = new_poisoning.AttackSuccessSet(
            base, labels, new_attack, normalize, SWEEP_CLASS_COUNT
        )
        assert new_asr.indices == old_asr.indices
        for position in range(len(old_asr)):
            old_image, old_target = old_asr[position]
            new_image, new_target = new_asr[position]
            assert torch.equal(new_image, old_image)
            assert int(new_target) == int(old_target)

        old_cover = old_poison.CoverPoisonedTrainingSet(
            base,
            old_attack,
            poison_indices,
            cover_indices,
            normalize,
            SWEEP_CLASS_COUNT,
        )
        new_cover = new_poisoning.CoverPoisonedTrainingSet(
            base,
            new_attack,
            poison_indices,
            cover_indices,
            normalize,
            SWEEP_CLASS_COUNT,
        )
        for index in range(len(old_cover)):
            old_image, old_label = old_cover[index]
            new_image, new_label = new_cover[index]
            assert torch.equal(new_image, old_image)
            assert int(new_label) == int(old_label)


def test_limit_dataset_is_identical():
    labels = list(range(500))
    base = _tensor_dataset(labels)

    assert new_data.limit_dataset(base, None, seed=0) is base
    assert new_data.limit_dataset(base, 500, seed=0) is base

    for max_samples, seed in ((16, 0), (64, 7), (499, 3)):
        old_subset = old_data.limit_dataset(base, max_samples, seed)
        new_subset = new_data.limit_dataset(base, max_samples, seed)
        assert list(new_subset.indices) == list(old_subset.indices)


def test_extract_labels_is_identical_including_nested_subsets():
    labels = [int(y) for y in np.random.default_rng(3).integers(0, 10, size=120)]
    base = _tensor_dataset(labels)
    nested = Subset(Subset(base, list(range(0, 120, 3))), list(range(0, 40, 2)))

    for dataset in (base, Subset(base, [5, 1, 99]), nested):
        assert new_data.extract_labels(dataset) == old_data.extract_labels(dataset)


def test_denormalize_is_identical():
    image = torch.rand(3, 8, 8)
    for dataset_name in old_config.DATASET_REGISTRY:
        assert torch.equal(
            new_data.denormalize(image, dataset_name),
            old_data.denormalize(image, dataset_name),
        )


def test_base_image_transform_matches_the_inline_transform_call_sites_rolled():
    """The one deliberate change: a named 0-to-1 transform replacing ad hoc Composes.

    Every real call site built exactly this Compose by hand, so the replacement
    has to produce the same tensor for the same input.
    """
    inline = transforms_v2.Compose(
        [transforms_v2.Resize((32, 32)), transforms_v2.ToTensor()]
    )
    named = new_data.base_image_transform(32)

    # A PIL image, matching what every torchvision dataset actually feeds in.
    # ToTensor only rescales to 0-to-1 for PIL and ndarray inputs, so a tensor
    # here would pass through unscaled and prove nothing.
    pixels = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    source = Image.fromarray(pixels)

    transformed = named(source)  # (3, 32, 32)
    assert torch.equal(transformed, inline(source))
    assert transformed.shape == (3, 32, 32)

    # It must not normalize, or triggers would be stamped on normalized pixels.
    assert float(transformed.min()) >= 0.0
    assert float(transformed.max()) <= 1.0


def test_network_core_unwraps_the_same_way():
    import torch.nn as nn

    wrapped = nn.Sequential(transforms_v2.Resize((224, 224)), nn.Linear(4, 4))
    bare = nn.Linear(4, 4)
    assert new_models.network_core(wrapped) is old_models.network_core(wrapped)
    assert new_models.network_core(bare) is old_models.network_core(bare)


def test_load_checkpoint_rejects_unknown_architecture_in_both():
    with pytest.raises(ValueError):
        old_models.load_checkpoint("resnet", "unused.pt", torch.device("cpu"))
    with pytest.raises(ValueError):
        new_models.load_checkpoint("resnet", "unused.pt", torch.device("cpu"))


@pytest.mark.parametrize("expected_architecture,checkpoint", ARCHITECTURE_CHECKPOINTS)
def test_detect_architecture_agrees_on_real_checkpoints(
    expected_architecture, checkpoint
):
    _skip_unless_present(checkpoint)
    new_result = new_models.detect_architecture(checkpoint)
    assert new_result == old_models.detect_architecture(checkpoint)
    assert new_result == expected_architecture


@pytest.mark.parametrize("checkpoint", MANIFEST_CHECKPOINTS)
def test_read_checkpoint_metadata_agrees(checkpoint):
    _skip_unless_present(checkpoint)
    assert new_splits.read_checkpoint_metadata(
        checkpoint
    ) == old_splits.read_checkpoint_metadata(checkpoint)


def test_read_checkpoint_metadata_missing_sidecar_raises_in_both(tmp_path):
    orphan = str(tmp_path / "attack_result.pt")
    with pytest.raises(FileNotFoundError):
        old_splits.read_checkpoint_metadata(orphan)
    with pytest.raises(FileNotFoundError):
        new_splits.read_checkpoint_metadata(orphan)


def test_resolve_probe_attack_agrees_on_every_branch():
    backdoored = {"attack": "badnet_a2o", "target_label": 3}
    benign = {"attack": "benign", "target_label": 0}

    assert new_splits.resolve_probe_attack(
        backdoored, None, None
    ) == old_splits.resolve_probe_attack(backdoored, None, None)
    assert new_splits.resolve_probe_attack(
        backdoored, "badnet_a2o", None
    ) == old_splits.resolve_probe_attack(backdoored, "badnet_a2o", None)
    assert new_splits.resolve_probe_attack(
        benign, "blend", 5
    ) == old_splits.resolve_probe_attack(benign, "blend", 5)
    assert new_splits.resolve_probe_attack(
        benign, "blend", None
    ) == old_splits.resolve_probe_attack(benign, "blend", None)

    with pytest.raises(ValueError):
        old_splits.resolve_probe_attack(backdoored, "blend", None)
    with pytest.raises(ValueError):
        new_splits.resolve_probe_attack(backdoored, "blend", None)

    with pytest.raises(ValueError):
        old_splits.resolve_probe_attack(benign, None, None)
    with pytest.raises(ValueError):
        new_splits.resolve_probe_attack(benign, None, None)


def test_split_constants_match():
    assert new_splits.PSBD_SPLIT_SEED == old_splits.PSBD_SPLIT_SEED
    assert new_splits.PSBD_HELDOUT_SIZE == old_splits.PSBD_HELDOUT_SIZE


@pytest.mark.parametrize("n_total", [10000, 12630, 1000])
def test_split_permutation_is_bit_identical(n_total):
    assert torch.equal(
        new_splits.psbd_split_permutation(n_total),
        old_splits.psbd_split_permutation(n_total),
    )
    assert torch.equal(
        new_splits.psbd_split_permutation(n_total, seed=5),
        old_splits.psbd_split_permutation(n_total, seed=5),
    )


@pytest.mark.parametrize("checkpoint", MANIFEST_CHECKPOINTS)
def test_split_manifest_is_identical_for_real_checkpoints(checkpoint):
    """The load-bearing test: every cached tensor on disk is ordered by this dict."""
    _skip_unless_present(checkpoint, RAW_DATA_DIR)

    _, old_manifest = old_splits.build_psbd_loaders_from_checkpoint(
        checkpoint, raw_data_dir=RAW_DATA_DIR, batch_size=64, num_workers=0
    )
    _, new_manifest = new_splits.build_psbd_loaders_from_checkpoint(
        checkpoint, raw_data_dir=RAW_DATA_DIR, batch_size=64, num_workers=0
    )

    assert new_manifest.keys() == old_manifest.keys()
    for key in old_manifest:
        assert new_manifest[key] == old_manifest[key], f"manifest key {key!r} drifted"
    assert new_manifest == old_manifest


def test_split_manifest_is_identical_under_max_samples():
    checkpoint = MANIFEST_CHECKPOINTS[0]
    _skip_unless_present(checkpoint, RAW_DATA_DIR)

    _, old_manifest = old_splits.build_psbd_loaders_from_checkpoint(
        checkpoint, raw_data_dir=RAW_DATA_DIR, num_workers=0, max_samples=64
    )
    _, new_manifest = new_splits.build_psbd_loaders_from_checkpoint(
        checkpoint, raw_data_dir=RAW_DATA_DIR, num_workers=0, max_samples=64
    )
    assert new_manifest == old_manifest


@pytest.mark.parametrize("checkpoint", MANIFEST_CHECKPOINTS[:2])
def test_loaders_serve_identical_tensors(checkpoint):
    """The manifest describing the same order is not enough, the pixels must match."""
    _skip_unless_present(checkpoint, RAW_DATA_DIR)

    old_loaders, _ = old_splits.build_psbd_loaders_from_checkpoint(
        checkpoint, raw_data_dir=RAW_DATA_DIR, batch_size=8, num_workers=0
    )
    new_loaders, _ = new_splits.build_psbd_loaders_from_checkpoint(
        checkpoint, raw_data_dir=RAW_DATA_DIR, batch_size=8, num_workers=0
    )

    assert new_loaders.keys() == old_loaders.keys()
    for split_name in old_loaders:
        old_images, old_labels = next(iter(old_loaders[split_name]))
        new_images, new_labels = next(iter(new_loaders[split_name]))
        assert new_images.shape == old_images.shape  # (batch, C, H, W)
        assert torch.equal(new_images, old_images), split_name
        assert torch.equal(new_labels, old_labels), split_name
        assert len(new_loaders[split_name].dataset) == len(
            old_loaders[split_name].dataset
        )


def _sam_step(sam_module, seed: int) -> list[torch.Tensor]:
    """One full two-pass SAM update on a fixed toy problem, returning the weights."""
    import torch.nn as nn
    from lightning import seed_everything

    seed_everything(seed)
    model = nn.Sequential(nn.Linear(6, 6), nn.Tanh(), nn.Linear(6, 3))
    optimizer = sam_module.SAM(model.parameters(), torch.optim.AdamW, rho=0.1, lr=1e-3)

    seed_everything(seed + 1)
    inputs = torch.randn(8, 6)  # (batch, features)
    targets = torch.randint(0, 3, (8,))
    criterion = nn.CrossEntropyLoss()

    criterion(model(inputs), targets).backward()
    optimizer.first_step(zero_grad=True)
    criterion(model(inputs), targets).backward()
    optimizer.second_step(zero_grad=True)

    return [parameter.detach().clone() for parameter in model.parameters()]


def test_sam_update_is_bit_identical():
    import sam as old_sam
    from psbd import sam as new_sam

    old_weights = _sam_step(old_sam, seed=11)
    new_weights = _sam_step(new_sam, seed=11)

    assert len(new_weights) == len(old_weights)
    for new_weight, old_weight in zip(new_weights, old_weights):
        assert torch.equal(new_weight, old_weight)


def test_sam_rejects_negative_rho_in_both():
    import torch.nn as nn

    import sam as old_sam
    from psbd import sam as new_sam

    parameters = list(nn.Linear(2, 2).parameters())
    with pytest.raises(ValueError):
        old_sam.SAM(parameters, torch.optim.AdamW, rho=-0.1)
    with pytest.raises(ValueError):
        new_sam.SAM(parameters, torch.optim.AdamW, rho=-0.1)


@pytest.mark.parametrize("builder_name", ["build_vit", "build_swin"])
def test_model_builders_produce_identical_weights(builder_name):
    from lightning import seed_everything

    seed_everything(0)
    old_model = getattr(old_models, builder_name)(num_classes=7)
    seed_everything(0)
    new_model = getattr(new_models, builder_name)(num_classes=7)

    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    assert list(new_state.keys()) == list(old_state.keys())
    for key in old_state:
        assert torch.equal(new_state[key], old_state[key]), key
