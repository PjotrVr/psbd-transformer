"""Tests for data.loading.exclude_indices and its wiring into
cli.train_backdoor's --exclude-indices-file.

2 properties matter. The wrapped dataset's length drops by exactly the size of
the excluded set, and no returned item ever carries an excluded original index.
The second property needs a way to recover, from an item alone, which original
index produced it, so _MarkedDataset below returns the index itself as the
"image".
"""

import argparse
import json

import torch
from torch.utils.data import Dataset

from data.loading import exclude_indices


class _MarkedDataset(Dataset):
    """A dataset whose "image" is its own original index, so a caller can tell
    which underlying samples a wrapper actually yielded.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.tensor(index), index % 2


def test_exclude_indices_shrinks_by_the_excluded_count():
    dataset = _MarkedDataset(num_samples=20)
    excluded = {2, 5, 9, 13}

    subset = exclude_indices(dataset, excluded)

    assert len(subset) == len(dataset) - len(excluded)


def test_exclude_indices_never_yields_an_excluded_index():
    dataset = _MarkedDataset(num_samples=20)
    excluded = {0, 3, 7, 19}

    subset = exclude_indices(dataset, excluded)
    yielded = {int(subset[i][0]) for i in range(len(subset))}

    assert yielded.isdisjoint(excluded)
    assert yielded == set(range(len(dataset))) - excluded


def test_exclude_indices_with_empty_set_is_a_no_op():
    dataset = _MarkedDataset(num_samples=6)

    subset = exclude_indices(dataset, set())

    assert len(subset) == len(dataset)
    assert {int(subset[i][0]) for i in range(len(subset))} == set(range(6))


def _train_backdoor_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        dataset="cifar10",
        attack="badnet_a2o",
        poison_rate=0.1,
        target_label=0,
        raw_data_dir="unused",
        max_samples=None,
        seed=0,
        poisoned_dir="",
        cover_rate=None,
        attack_override=[],
        evade_psbd=False,
        batch_size=4,
        num_workers=0,
        augment="none",
        exclude_indices_file=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _FixedImageDataset(Dataset):
    """A synthetic (image, label) dataset, images in 0 to 1, no disk or network."""

    def __init__(self, num_samples: int, image_size: int, num_classes: int):
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(0)
        self.images = torch.rand(
            (num_samples, 3, image_size, image_size), generator=generator
        )
        self.labels = [i % num_classes for i in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.labels[index]


def test_train_backdoor_loader_exclude_indices_file_shrinks_the_loader(
    tmp_path, monkeypatch
):
    import cli.train_backdoor as train_backdoor

    image_size = 32
    num_samples = 16
    fake_train = _FixedImageDataset(
        num_samples=num_samples, image_size=image_size, num_classes=10
    )
    fake_test = _FixedImageDataset(num_samples=4, image_size=image_size, num_classes=10)
    monkeypatch.setattr(
        train_backdoor,
        "load_clean_datasets",
        lambda dataset_name, transform, raw_data_dir: (fake_train, fake_test),
    )

    excluded = [1, 4, 10]
    exclude_file = tmp_path / "flagged_indices.json"
    exclude_file.write_text(json.dumps(excluded))

    plain_loader, _, _, _, _ = train_backdoor.build_training_loader(
        _train_backdoor_args(), image_size
    )
    sanitised_loader, _, _, _, _ = train_backdoor.build_training_loader(
        _train_backdoor_args(exclude_indices_file=str(exclude_file)), image_size
    )

    assert len(sanitised_loader.dataset) == len(plain_loader.dataset) - len(excluded)
