"""The dataset views over paired loaders, on synthetic splits with a known pairing.

The clean split serves every row, the backdoor split only the eligible ones
in a different position, and the manifest records both index lists the way
data.splits writes them. Every view must then carry true labels, a poison mask
that names the triggered rows and images that are the triggered self of the
clean row at the same test index.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from analysis.samples import (
    class_subset,
    clean_view,
    equal_ratio_class_subset,
    mixed_view,
    paired_examples,
    paired_views,
    select_classes,
    to_display,
    to_pixels,
    union_view,
    view_samples,
)
from tests.reference import backdoorbench as upstream

NUM_ROWS = 60
NUM_CLASSES = 5
TARGET = 0


@pytest.fixture(scope="module")
def synthetic_case() -> tuple[dict[str, DataLoader], dict]:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(NUM_ROWS, 3, 8, 8, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (NUM_ROWS,), generator=generator)
    # The analysis split is a shuffle of the test set, recorded in the manifest.
    test_indices = torch.randperm(200, generator=generator)[:NUM_ROWS]

    eligible = [row for row in range(NUM_ROWS) if int(labels[row]) != TARGET]
    triggered = images[eligible].clone()
    triggered[:, :, :2, :2] = 1.0
    loaders = {
        "clean": DataLoader(
            TensorDataset(images, labels), batch_size=16, shuffle=False
        ),
        "backdoor": DataLoader(
            TensorDataset(triggered, torch.full((len(eligible),), TARGET)),
            batch_size=16,
            shuffle=False,
        ),
    }
    manifest = {
        "analysis_clean_indices": test_indices.tolist(),
        "analysis_backdoor_indices": test_indices[eligible].tolist(),
        "probe_attack": "badnet_a2o",
        "probe_target_label": TARGET,
    }
    return loaders, manifest


def test_paired_views_align_row_for_row(synthetic_case):
    loaders, manifest = synthetic_case
    clean, backdoor = paired_views(loaders, manifest)
    whole = clean_view(loaders, manifest)

    assert len(clean) == len(backdoor) == len(manifest["analysis_backdoor_indices"])
    assert torch.equal(clean.test_indices, backdoor.test_indices)
    assert torch.equal(clean.labels, backdoor.labels)
    assert not clean.poison_mask.any() and backdoor.poison_mask.all()
    assert (clean.labels != TARGET).all()
    # Every clean row of the pair is the clean split's row at the same test index,
    # and the triggered row differs from it only inside the patch.
    position = {int(index): row for row, index in enumerate(whole.test_indices)}
    for row in range(len(clean)):
        whole_row = position[int(clean.test_indices[row])]
        assert torch.equal(clean.images[row], whole.images[whole_row])
        assert torch.equal(
            backdoor.images[row][:, 2:, 2:], clean.images[row][:, 2:, 2:]
        )
        assert not torch.equal(backdoor.images[row], clean.images[row])


def test_a_manifest_that_disagrees_with_the_loader_is_refused(synthetic_case):
    loaders, manifest = synthetic_case
    broken = {
        **manifest,
        "analysis_backdoor_indices": manifest["analysis_backdoor_indices"][:-1],
    }
    with pytest.raises(ValueError, match="cannot be aligned"):
        paired_views(loaders, broken)


def test_mixed_view_triggers_a_ratio_of_the_whole_split(synthetic_case):
    loaders, manifest = synthetic_case
    clean, backdoor = paired_views(loaders, manifest)
    whole = clean_view(loaders, manifest)

    mixed = mixed_view(clean, backdoor, whole, 0.25, seed=0)
    assert len(mixed) == len(whole)
    assert int(mixed.poison_mask.sum()) == int(len(whole) * 0.25)
    assert torch.equal(mixed.labels, whole.labels)
    assert (mixed.labels[mixed.poison_mask] != TARGET).all()
    for row in torch.nonzero(mixed.poison_mask).flatten().tolist():
        assert not torch.equal(mixed.images[row], whole.images[row])
    for row in torch.nonzero(~mixed.poison_mask).flatten().tolist():
        assert torch.equal(mixed.images[row], whole.images[row])


def test_union_view_stacks_clean_first(synthetic_case):
    loaders, manifest = synthetic_case
    clean, backdoor = paired_views(loaders, manifest)
    both = union_view(clean, backdoor)
    assert len(both) == 2 * len(clean)
    assert (
        not both.poison_mask[: len(clean)].any()
        and both.poison_mask[len(clean) :].all()
    )


def test_select_classes_keeps_the_target_and_the_count():
    everything = select_classes(5, 10, TARGET, seed=0)
    assert np.array_equal(everything, np.arange(5))
    drawn = select_classes(43, 10, 7, seed=0)
    assert drawn.shape == (10,) and drawn[-1] == 7 and len(set(drawn)) == 10


def test_equal_ratio_subset_draws_the_upstream_count_per_class():
    labels = np.repeat(np.arange(4), [30, 20, 10, 40])
    ours = equal_ratio_class_subset(labels, 50, np.array([0, 1, 3]), seed=0)
    np.random.seed(0)
    theirs = upstream.sub_sample_euqal_ratio_classes_index(
        labels, selected_classes=np.array([0, 1, 3]), max_num_samples=50
    )
    for c in range(4):
        assert int((labels[ours] == c).sum()) == int((labels[theirs] == c).sum())
    assert int((labels[ours] == 2).sum()) == 0
    assert len(set(ours.tolist())) == len(ours)


def test_class_subset_samples_poisoned_rows_as_their_own_class(synthetic_case):
    loaders, manifest = synthetic_case
    clean, backdoor = paired_views(loaders, manifest)
    both = union_view(clean, backdoor)
    subset = class_subset(both, 30, np.array([1, 2]), seed=0)
    assert set(subset.labels[~subset.poison_mask].tolist()) <= {1, 2}
    assert subset.poison_mask.any()
    assert len(subset) <= 30


@pytest.mark.parametrize("view", ["clean_test", "bd_test", "mixed", "paired"])
def test_every_view_builds_with_true_labels(synthetic_case, view):
    loaders, manifest = synthetic_case
    built = view_samples(loaders, manifest, view, None, None, 0.2, seed=0)
    assert len(built) > 0
    assert built.images.shape[1:] == (3, 8, 8)
    assert built.labels.max() < NUM_CLASSES
    assert (built.labels[built.poison_mask] != TARGET).all()
    if view == "clean_test":
        assert not built.poison_mask.any()
    if view == "bd_test":
        assert built.poison_mask.all()


def test_an_unknown_view_is_refused(synthetic_case):
    loaders, manifest = synthetic_case
    with pytest.raises(ValueError, match="unknown view"):
        view_samples(loaders, manifest, "bd_train", None, None, 0.2, seed=0)


def test_paired_examples_show_the_same_images_clean_then_triggered(synthetic_case):
    loaders, manifest = synthetic_case
    clean, backdoor = paired_views(loaders, manifest)
    examples = paired_examples(clean, backdoor, 2, 2, seed=0)
    assert examples.poison_mask.tolist() == [False, False, True, True]
    assert torch.equal(examples.test_indices[:2], examples.test_indices[2:])
    assert torch.equal(examples.labels[:2], examples.labels[2:])
    assert torch.equal(examples.images[2][:, 2:, 2:], examples.images[0][:, 2:, 2:])


def test_paired_examples_shrink_with_the_pool_and_keep_the_mask_honest(synthetic_case):
    loaders, manifest = synthetic_case
    clean, backdoor = paired_views(loaders, manifest)
    few = backdoor.subset(torch.tensor([0]))
    few_clean = clean.subset(torch.tensor([0]))
    examples = paired_examples(few_clean, few, 2, 2, seed=0)
    assert examples.poison_mask.tolist() == [False, True]
    with pytest.raises(ValueError, match="paired views"):
        paired_examples(clean, few, 2, 2, seed=0)


def test_to_pixels_undoes_the_normalisation():
    pixels = torch.rand(2, 3, 4, 4)
    mean, std = (0.5, 0.4, 0.3), (0.2, 0.25, 0.3)
    normalised = (pixels - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(
        std
    ).view(1, 3, 1, 1)
    assert torch.allclose(to_pixels(normalised, mean, std), pixels, atol=1e-6)
    assert to_display(pixels).shape == (2, 4, 4, 3)
