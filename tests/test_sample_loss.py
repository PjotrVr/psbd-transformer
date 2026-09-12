"""Tests for the optional --record-sample-loss path.

Covers IndexedTrainingSet, the per-epoch per-sample loss row train_one_epoch
scatters when asked to, and the sample_loss.npz record train_classifier
writes. The expensive failure this prevents: a flag that looks additive but
silently changes a plain run's forward count, RNG state or return contract,
discovered only after a full smoke job burns GPU minutes.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from lightning import seed_everything
from torch.utils.data import DataLoader, Dataset

from training.loop import (
    IndexedTrainingSet,
    locate_poison_indices,
    save_sample_loss_record,
    train_classifier,
    train_one_epoch,
)

NUM_SAMPLES = 12
IMAGE_SHAPE = (3, 8, 8)


class TinyMLP(nn.Module):
    """A batchnorm-free, dropout-free stand-in, fast enough for a CPU unit test."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, images):
        logits = self.linear(self.flatten(images))  # (batch, num_classes)
        return logits


class TinyPoisonedSet(Dataset):
    """A minimal stand-in for attacks.poisoning.PoisonedTrainingSet."""

    def __init__(self, images, labels, poison_indices):
        self.images = images
        self.labels = labels
        self.poison_indices = poison_indices

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


@pytest.fixture
def tiny_dataset():
    seed_everything(0)
    images = torch.randn(NUM_SAMPLES, *IMAGE_SHAPE)
    labels = torch.randint(0, 4, (NUM_SAMPLES,))
    poison_indices = {1, 4, 7}
    return TinyPoisonedSet(images, labels, poison_indices)


def test_indexed_training_set_returns_image_label_index(tiny_dataset):
    indexed = IndexedTrainingSet(tiny_dataset)
    assert len(indexed) == len(tiny_dataset)
    for index in (0, 5, 11):
        image, label, returned_index = indexed[index]
        expected_image, expected_label = tiny_dataset[index]
        assert torch.equal(image, expected_image)
        assert label == expected_label
        assert returned_index == index


def test_locate_poison_indices_walks_the_wrapper_chain(tiny_dataset):
    indexed = IndexedTrainingSet(tiny_dataset)
    assert locate_poison_indices(indexed) == tiny_dataset.poison_indices
    assert locate_poison_indices(tiny_dataset) == tiny_dataset.poison_indices


def test_locate_poison_indices_defaults_to_empty():
    class Bare(Dataset):
        def __len__(self):
            return 0

        def __getitem__(self, index):
            raise IndexError

    assert locate_poison_indices(Bare()) == set()


def test_plain_epoch_ignores_the_new_argument(tiny_dataset):
    """The default call form (no sample_loss_row) is untouched by the new kwarg."""
    loader = DataLoader(tiny_dataset, batch_size=4, shuffle=False)

    seed_everything(0)
    model_a = TinyMLP()
    seed_everything(0)
    model_b = TinyMLP()
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    loss_a = train_one_epoch(model_a, loader, criterion, optimizer_a, device, False)
    loss_b = train_one_epoch(
        model_b, loader, criterion, optimizer_b, device, False, None, None
    )
    assert loss_a == pytest.approx(loss_b)
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(param_a, param_b)


def test_sample_loss_row_matches_first_batch_manual_cross_entropy(tiny_dataset):
    """The first batch's row is computed from the model's untouched initial weights."""
    seed_everything(0)
    model = TinyMLP()
    initial_state = {name: p.clone() for name, p in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    indexed_loader = DataLoader(
        IndexedTrainingSet(tiny_dataset), batch_size=4, shuffle=False
    )
    sample_loss_row = torch.full((NUM_SAMPLES,), float("nan"))
    train_one_epoch(
        model,
        indexed_loader,
        criterion,
        optimizer,
        device,
        False,
        None,
        sample_loss_row,
    )

    assert sample_loss_row.shape == (NUM_SAMPLES,)
    assert not torch.isnan(sample_loss_row).any()

    # The first batch's row is scattered before any update happens this epoch,
    # so it must match a fresh forward from the pre-training weights.
    reference_model = TinyMLP()
    reference_model.load_state_dict(initial_state)
    first_images, first_labels = tiny_dataset.images[:4], tiny_dataset.labels[:4]
    with torch.no_grad():
        expected = nn.CrossEntropyLoss(reduction="none")(
            reference_model(first_images), first_labels
        )
    assert torch.allclose(sample_loss_row[:4], expected, atol=1e-6)


def test_save_sample_loss_record_writes_epoch_by_sample_shapes(tmp_path):
    history = torch.randn(3, NUM_SAMPLES)
    poison_indices = {2, 5, 9}
    save_sample_loss_record(str(tmp_path), history, poison_indices)

    loaded = np.load(tmp_path / "sample_loss.npz")
    assert loaded["sample_loss"].shape == (3, NUM_SAMPLES)
    assert list(loaded["poison_indices"]) == sorted(poison_indices)


def build_tiny_classifier_loaders(poison_indices: set[int]):
    """A resnet18-scale (32x32) training and validation loader, no download needed."""
    seed_everything(0)
    train_images = torch.randn(NUM_SAMPLES, 3, 32, 32)
    train_labels = torch.randint(0, 4, (NUM_SAMPLES,))
    train_set = TinyPoisonedSet(train_images, train_labels, poison_indices)

    val_images = torch.randn(16, 3, 32, 32)
    val_labels = torch.randint(0, 4, (16,))
    val_set = TinyPoisonedSet(val_images, val_labels, set())

    train_loader = DataLoader(train_set, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)
    return train_loader, val_loader


def test_train_classifier_plain_run_has_no_sample_loss_file(tmp_path):
    train_loader, val_loader = build_tiny_classifier_loaders(set())
    device = torch.device("cpu")

    model, trajectory = train_classifier(
        "resnet18",
        4,
        train_loader,
        val_loader,
        device,
        epochs=2,
        use_sam=False,
        use_bfloat16=False,
        checkpoint_dir=str(tmp_path),
    )
    assert len(trajectory.validation_accuracies) == 2
    assert not (tmp_path / "sample_loss.npz").exists()


def test_train_classifier_records_1_row_per_epoch_1_column_per_sample(tmp_path):
    poison_indices = {1, 3, 5}
    train_loader, val_loader = build_tiny_classifier_loaders(poison_indices)
    indexed_loader = DataLoader(
        IndexedTrainingSet(train_loader.dataset), batch_size=4, shuffle=False
    )
    device = torch.device("cpu")
    epochs = 3

    train_classifier(
        "resnet18",
        4,
        indexed_loader,
        val_loader,
        device,
        epochs=epochs,
        use_sam=False,
        use_bfloat16=False,
        record_sample_loss=True,
        checkpoint_dir=str(tmp_path),
    )

    loaded = np.load(tmp_path / "sample_loss.npz")
    assert loaded["sample_loss"].shape == (epochs, NUM_SAMPLES)
    assert list(loaded["poison_indices"]) == sorted(poison_indices)


def test_train_classifier_rejects_record_sample_loss_with_evasion(tmp_path):
    train_loader, val_loader = build_tiny_classifier_loaders(set())
    device = torch.device("cpu")

    with pytest.raises(ValueError):
        train_classifier(
            "resnet18",
            4,
            train_loader,
            val_loader,
            device,
            epochs=1,
            use_sam=False,
            record_sample_loss=True,
            checkpoint_dir=str(tmp_path),
            evasion={"probe": {}, "weight": 1.0, "passes": 1},
        )


def test_train_classifier_requires_checkpoint_dir_when_recording():
    train_loader, val_loader = build_tiny_classifier_loaders(set())
    device = torch.device("cpu")

    with pytest.raises(ValueError):
        train_classifier(
            "resnet18",
            4,
            train_loader,
            val_loader,
            device,
            epochs=1,
            use_sam=False,
            record_sample_loss=True,
        )
