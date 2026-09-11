"""The confusion matrix, DER and RIR against BackdoorBench's own lines."""

import numpy as np
import pytest
import torch

from evaluation.metrics import (
    confusion_matrix,
    defense_effectiveness_rate,
    robust_improvement_rate,
)
from tests.reference import backdoorbench as upstream

NUM_CLASSES = 7


@pytest.fixture
def predictions_and_labels() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    labels = torch.randint(0, NUM_CLASSES, (200,), generator=generator)
    predictions = torch.where(
        torch.rand(200, generator=generator) < 0.6,
        labels,
        torch.randint(0, NUM_CLASSES, (200,), generator=generator),
    )
    # Class 6 never appears as a true label, which is the row the +1e-24 guard is for.
    labels[labels == 6] = 5
    return predictions, labels


def test_counts_match_the_upstream_loop(predictions_and_labels):
    predictions, labels = predictions_and_labels
    ours = confusion_matrix(predictions, labels, NUM_CLASSES, normalise=False)
    theirs = upstream.confusion_matrix(
        labels.numpy(), predictions.numpy(), list(range(NUM_CLASSES)), False
    )
    assert np.array_equal(ours.numpy(), theirs)
    assert int(ours.sum()) == 200


def test_row_normalisation_matches_including_the_empty_row(predictions_and_labels):
    predictions, labels = predictions_and_labels
    ours = confusion_matrix(predictions, labels, NUM_CLASSES, normalise=True)
    theirs = upstream.confusion_matrix(
        labels.numpy(), predictions.numpy(), list(range(NUM_CLASSES)), True
    )
    assert np.allclose(ours.numpy(), theirs, atol=1e-7)
    assert torch.all(ours[6] == 0)
    assert torch.isfinite(ours).all()


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        confusion_matrix(
            torch.zeros(3, dtype=torch.long), torch.zeros(4, dtype=torch.long), 2
        )


@pytest.mark.parametrize("draw", range(5))
def test_der_and_rir_are_the_simplified_upstream_versions(draw):
    generator = np.random.default_rng(draw)
    acc_bd, acc_def, asr_bd, asr_def, ra_bd, ra_def = generator.uniform(0, 1, 6)
    assert defense_effectiveness_rate(
        acc_bd, acc_def, asr_bd, asr_def
    ) == pytest.approx(
        upstream.defense_effectiveness_rate_simplied(acc_bd, acc_def, asr_bd, asr_def)
    )
    assert robust_improvement_rate(acc_bd, acc_def, ra_bd, ra_def) == pytest.approx(
        upstream.robust_improvement_rate_simplied(acc_bd, acc_def, ra_bd, ra_def)
    )


def test_a_perfect_defence_scores_1_and_a_destructive_one_is_penalised():
    assert defense_effectiveness_rate(0.9, 0.9, 1.0, 0.0) == 1.0
    assert defense_effectiveness_rate(0.9, 0.4, 1.0, 0.0) == pytest.approx(0.75)
    assert robust_improvement_rate(0.9, 0.9, 0.0, 1.0) == 1.0
    assert robust_improvement_rate(0.9, 0.9, 0.5, 0.5) == 0.5
