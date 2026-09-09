"""Reproducibility and 0-leakage of the standardized PSBD split.

The split's whole point is auditability: the seed recipe must reproduce the
manifest exactly, heldout and analysis must be disjoint and together complete,
and the row order a loader serves must match the manifest index for that row. The
first two need no data; the row-mapping round-trip loads the real cifar100 test
set and one on-disk checkpoint, so it is skipped when either is absent.
"""

import os

import pytest
import torch
import torchvision.transforms.v2 as transforms_v2
from lightning import seed_everything

from data.splits import (
    PSBD_HELDOUT_SIZE,
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    psbd_split_permutation,
)
from data.registry import DATASET_REGISTRY
from data.loading import denormalize, load_clean_datasets

CIFAR100_TEST_SIZE = 10000
CHECKPOINT = "checkpoints/vit_cifar100_wanet_0_1/attack_result.pt"


def test_seed_recipe_reproduces_permutation():
    seed_everything(PSBD_SPLIT_SEED)
    reference = torch.randperm(CIFAR100_TEST_SIZE)
    assert torch.equal(psbd_split_permutation(CIFAR100_TEST_SIZE), reference)


def test_heldout_analysis_disjoint_and_complete():
    permutation = psbd_split_permutation(CIFAR100_TEST_SIZE)
    heldout = permutation[:PSBD_HELDOUT_SIZE]
    analysis = permutation[PSBD_HELDOUT_SIZE:]

    heldout_set = set(heldout.tolist())
    analysis_set = set(analysis.tolist())
    assert heldout_set.isdisjoint(analysis_set)
    assert heldout_set | analysis_set == set(range(CIFAR100_TEST_SIZE))
    assert len(heldout_set) == PSBD_HELDOUT_SIZE


@pytest.mark.skipif(
    not os.path.exists(CHECKPOINT), reason="cifar100 wanet checkpoint not on disk"
)
def test_manifest_matches_seed_recipe_and_is_disjoint():
    _, manifest = build_psbd_loaders_from_checkpoint(CHECKPOINT, batch_size=64)

    permutation = psbd_split_permutation(manifest["n_total"])
    expected_heldout = permutation[:PSBD_HELDOUT_SIZE].tolist()
    expected_analysis = permutation[PSBD_HELDOUT_SIZE:].tolist()

    assert manifest["heldout_indices"] == expected_heldout
    assert manifest["analysis_clean_indices"] == expected_analysis
    assert set(manifest["heldout_indices"]).isdisjoint(
        manifest["analysis_clean_indices"]
    )
    # backdoor indices are the eligibility-filtered subset of the analysis pool.
    assert set(manifest["analysis_backdoor_indices"]).issubset(
        set(manifest["analysis_clean_indices"])
    )


@pytest.mark.skipif(
    not os.path.exists(CHECKPOINT), reason="cifar100 wanet checkpoint not on disk"
)
def test_row_mapping_round_trip():
    """The image a loader serves at row i is the raw test image at manifest[i]."""
    loaders, manifest = build_psbd_loaders_from_checkpoint(CHECKPOINT, batch_size=8)
    spec = DATASET_REGISTRY["cifar100"]
    base_transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((spec.image_size, spec.image_size)),
            transforms_v2.ToTensor(),
        ]
    )
    _, raw_test = load_clean_datasets("cifar100", base_transform, "raw_data")

    served_images, _ = next(iter(loaders["validation"]))
    for row in range(3):
        original_index = manifest["heldout_indices"][row]
        raw_image, _ = raw_test[original_index]
        served = denormalize(served_images[row], "cifar100")
        assert torch.allclose(served, raw_image, atol=1e-5), (
            f"row {row} does not map to manifest index {original_index}"
        )
