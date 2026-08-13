"""Pooled and per-class accuracy share one forward pass, not two.

clean_accuracy_by_class and the pooled accuracy used to each run their own
pass over the same loader when both were wanted (see metrics.py's
evaluate_benign). class_correct_and_total is the shared computation; these
tests check the aggregation on top of it is correct, in particular that the
pooled figure is a count-weighted average, not a naive mean of per-class
accuracies, since those differ on an imbalanced set.
"""

import pytest
import torch

from defences.detection import (
    accuracy_by_class_from_counts,
    clean_accuracy_by_class,
    pooled_accuracy_from_counts,
)


def _fake_model_and_loader(predictions: list[int], labels: list[int]):
    # A stand-in model whose forward pass is the identity, fed pre-baked
    # one-hot rows, so predictions.argmax() reproduces `predictions` exactly.
    num_classes = max(predictions + labels) + 1
    images = torch.eye(num_classes)[predictions]
    loader = [(images, torch.tensor(labels))]

    class IdentityModel(torch.nn.Module):
        def forward(self, x):
            return x

    return IdentityModel().eval(), loader, num_classes


def test_pooled_accuracy_is_count_weighted_not_a_naive_mean():
    # Class 0 has 8 samples, all correct. Class 1 has 2 samples, both wrong.
    # Naive mean of per-class accuracy: (1.0 + 0.0) / 2 = 0.5.
    # True pooled (count-weighted) accuracy: 8 correct / 10 total = 0.8.
    correct = torch.tensor([8.0, 0.0])
    total = torch.tensor([8.0, 2.0])
    assert pooled_accuracy_from_counts(correct, total) == pytest.approx(0.8)


def test_accuracy_by_class_from_counts_matches_fractions():
    correct = torch.tensor([3.0, 1.0, 0.0])
    total = torch.tensor([4.0, 2.0, 0.0])
    by_class = accuracy_by_class_from_counts(correct, total)
    assert by_class == {0: 0.75, 1: 0.5, 2: 0.0}  # zero total must not divide by zero


def test_clean_accuracy_by_class_end_to_end():
    # index: pred, label -> 0:(0,0) correct, 1:(0,0) correct, 2:(1,1) correct, 3:(1,2) wrong
    predictions = [0, 0, 1, 1]
    labels = [0, 0, 1, 2]
    model, loader, num_classes = _fake_model_and_loader(predictions, labels)

    by_class = clean_accuracy_by_class(
        model, loader, torch.device("cpu"), num_classes, use_bfloat16=False
    )
    assert by_class[0] == 1.0  # both label-0 samples predicted correctly
    assert by_class[1] == 1.0  # the one label-1 sample predicted correctly
    assert by_class[2] == 0.0  # the one label-2 sample predicted wrong
