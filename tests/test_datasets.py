"""Unit tests for utils.datasets.limit_dataset.

limit_dataset backs the --max-samples smoke-test flag. Two properties matter:
the "no limit" path must return the exact same object so a full run and a
truncated run share one code path, and the truncated path must be a reproducible
random subset (not first-N, which would cover only the first class or two of an
ImageFolder-backed dataset like Tiny ImageNet).
"""

from torch.utils.data import Subset, TensorDataset
import torch

from utils.datasets import limit_dataset


def _dataset(n: int) -> TensorDataset:
    return TensorDataset(torch.arange(n), torch.arange(n))


def test_none_returns_identical_object():
    dataset = _dataset(100)
    assert limit_dataset(dataset, None, seed=0) is dataset


def test_max_samples_at_or_above_length_returns_identical_object():
    dataset = _dataset(100)
    assert limit_dataset(dataset, 100, seed=0) is dataset
    assert limit_dataset(dataset, 200, seed=0) is dataset


def test_truncation_returns_subset_of_requested_length():
    dataset = _dataset(100)
    subset = limit_dataset(dataset, 16, seed=0)
    assert isinstance(subset, Subset)
    assert len(subset) == 16


def test_indices_drawn_without_replacement():
    subset = limit_dataset(_dataset(100), 16, seed=0)
    assert len(set(int(i) for i in subset.indices)) == 16


def test_same_seed_is_reproducible():
    first = limit_dataset(_dataset(100), 16, seed=7)
    second = limit_dataset(_dataset(100), 16, seed=7)
    assert list(first.indices) == list(second.indices)


def test_different_seed_usually_differs():
    first = limit_dataset(_dataset(100), 16, seed=1)
    second = limit_dataset(_dataset(100), 16, seed=2)
    assert list(first.indices) != list(second.indices)
