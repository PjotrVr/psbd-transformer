"""The attack success rate's 2 definitions, single class and target set.

A multi-target clean-label attack predicts a set of classes, so a success is any
member of that set. The single-target path must stay the plain accuracy on the
intended labels, since every panel sidecar was scored with it.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import attack_success_rate


class FixedLogits(nn.Module):
    """Returns 1 preset logit row per input, so the predictions are chosen by the test."""

    def __init__(self, predictions: list[int], num_classes: int):
        super().__init__()
        self.logits = torch.nn.functional.one_hot(
            torch.tensor(predictions), num_classes
        ).float()  # (n, num_classes)
        self.offset = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.size(0)
        rows = self.logits[self.offset : self.offset + batch]  # (batch, num_classes)
        self.offset += batch
        return rows


def loader_with_labels(labels: list[int]) -> DataLoader:
    images = torch.zeros(len(labels), 3, 4, 4)  # (n, 3, 4, 4)
    dataset = TensorDataset(images, torch.tensor(labels))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    return loader


def test_single_target_success_is_accuracy_on_the_intended_label():
    model = FixedLogits([0, 0, 3, 0], num_classes=5)
    loader = loader_with_labels([0, 0, 0, 0])
    rate = attack_success_rate(model, loader, torch.device("cpu"), use_bfloat16=False)
    assert rate == 0.75


def test_a_target_set_counts_any_member_as_a_success():
    model = FixedLogits([0, 1, 2, 4], num_classes=5)
    loader = loader_with_labels([0, 0, 0, 0])
    rate = attack_success_rate(
        model, loader, torch.device("cpu"), use_bfloat16=False, success_labels=(0, 1, 2)
    )
    assert rate == 0.75


def test_a_set_of_1_is_the_single_target_path():
    model = FixedLogits([0, 1, 2, 4], num_classes=5)
    loader = loader_with_labels([0, 0, 0, 0])
    rate = attack_success_rate(
        model, loader, torch.device("cpu"), use_bfloat16=False, success_labels=(0,)
    )
    assert rate == 0.25
