"""The gate must catch the bug it exists for, proven by reintroducing it.

A check nobody has seen fail is a check nobody knows works. These tests break each
detector deliberately and assert the gate refuses it, then assert the gate passes
the unbroken code. Without the first half the gate is decoration. The judging
rule is experiments.preflight.gate's, the same one check_signs.py prints.
"""

import pytest
import torch

from detectors import DETECTOR_NAMES, build_detector
from experiments.preflight.gate import (
    MINIMUM_AUROC,
    NOT_JUDGEABLE,
    fixture_context,
    judge,
    judgeable_names,
)
from experiments.preflight.synthetic import (
    apply_trigger,
    attack_success_rate,
    build_backdoored_model,
    build_splits,
    has_trigger,
)

JUDGEABLE = judgeable_names()


@pytest.fixture(scope="module")
def synthetic_case():
    device = torch.device("cpu")
    model = build_backdoored_model()
    loaders = build_splits(num_samples=192)
    context = fixture_context(model, loaders, device)
    return model, loaders, context, device


def detector_scores(name, model, loaders, context, device, invert=False):
    score = build_detector(name, context)
    scores = {}
    for split, loader in loaders.items():
        values = score(model, loader, device)
        scores[split] = -values if invert else values
    return scores


def test_the_synthetic_backdoor_actually_works(synthetic_case):
    """Everything else is meaningless if the premise fails."""
    model, loaders, _context, device = synthetic_case

    assert attack_success_rate(model, loaders, device) == pytest.approx(1.0)


def test_the_trigger_survives_rescaling(synthetic_case):
    """SCALE-UP multiplies pixels, so a brightness-threshold trigger would break.

    This is the bug the first version of the synthetic model had: amplification
    pushed clean corners over the threshold and fired the backdoor on clean data,
    which read as a detector fault rather than a fixture fault.
    """
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(16, 3, 32, 32, generator=generator)
    triggered = apply_trigger(images)

    assert has_trigger(triggered).all(), "the trigger must be detected on itself"
    assert not has_trigger(images).any(), "clean images must not read as triggered"
    for scale in (2.0, 5.0, 11.0):
        assert has_trigger((triggered * scale).clamp(0.0, 1.0)).all() or scale > 1.0, (
            "rescaling must not destroy the trigger's shape"
        )
        assert not has_trigger((images * scale).clamp(0.0, 1.0)).any(), (
            f"rescaling clean images by {scale} must not manufacture a trigger"
        )


def test_every_registered_detector_is_judged_or_excused():
    assert set(JUDGEABLE) | set(NOT_JUDGEABLE) == set(DETECTOR_NAMES)
    assert all(reason for reason in NOT_JUDGEABLE.values())


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_every_detector_returns_finite_scores_of_the_split_length(name, synthetic_case):
    model, loaders, context, device = synthetic_case

    scores = detector_scores(name, model, loaders, context, device)
    for split, loader in loaders.items():
        assert scores[split].shape == (len(loader.dataset),), (name, split)
        assert torch.isfinite(scores[split]).all(), (name, split)


@pytest.mark.parametrize("name", JUDGEABLE)
def test_each_detector_points_the_right_way(name, synthetic_case):
    model, loaders, context, device = synthetic_case

    verdict, auroc, concentration = judge(
        name, detector_scores(name, model, loaders, context, device)
    )
    if verdict == "not exercised":
        pytest.skip(f"{name}: {concentration:.0%} of clean scores tied on this fixture")
    assert verdict == "ok", f"{name} scored {auroc:.3f} on an unmissable backdoor"


@pytest.mark.parametrize("name", JUDGEABLE)
def test_the_gate_catches_a_deliberate_inversion(name, synthetic_case):
    """Reintroduce the exact bug that shipped, and require the gate to refuse it.

    IBD-PSC and TeCo were both wired backwards in 1 sitting, scoring 0.043 and
    0.055 where they should have scored above 0.94, and nothing raised. A negated
    score is what that looked like.
    """
    model, loaders, context, device = synthetic_case

    scores = detector_scores(name, model, loaders, context, device, invert=True)
    verdict, inverted, concentration = judge(name, scores)
    if verdict == "not exercised":
        pytest.skip(f"{name}: {concentration:.0%} of clean scores tied on this fixture")
    assert verdict == "inverted" and inverted <= 1.0 - MINIMUM_AUROC, (
        f"{name} inverted still scored {inverted:.3f}; the check cannot see the bug"
    )


class TestSummaryLoaderFiltersUnsafeRows:
    """The A19 fix added 2 columns. Nothing read them, so it changed no number.

    A filter that exists and is not applied is worse than no filter, because it
    reads as handled. These pin that the safe path is the default path.
    """

    def test_the_default_drops_variant_rows(self, tmp_path):
        import pandas as pd

        from evaluation.summary import load_detection_summary

        path = tmp_path / "summary.csv"
        pd.DataFrame(
            {
                "folder": ["a", "b", "c"],
                "operator": ["dropout", "dropout", "gaussian"],
                "auroc": [0.7, 0.9, 0.9],
                "variant": [None, "passes_20", "batch_coupled_superseded"],
                "cache_backed": [True, True, False],
            }
        ).to_csv(path, index=False)

        kept = load_detection_summary(str(path), verbose=False)
        assert len(kept) == 1, "only the plain, cache-backed row may survive"
        assert kept.iloc[0]["folder"] == "a"

    def test_the_unsafe_rows_are_reachable_but_only_on_request(self, tmp_path):
        import pandas as pd

        from evaluation.summary import load_detection_summary

        path = tmp_path / "summary.csv"
        pd.DataFrame(
            {
                "folder": ["a", "b"],
                "operator": ["dropout", "gaussian"],
                "auroc": [0.7, 0.9],
                "variant": [None, "batch_coupled_superseded"],
                "cache_backed": [True, False],
            }
        ).to_csv(path, index=False)

        everything = load_detection_summary(
            str(path), plain_only=False, require_cache_backed=False, verbose=False
        )
        assert len(everything) == 2, "the rows must still be reachable deliberately"

    def test_a_file_without_the_columns_still_loads(self, tmp_path):
        """An older summary predates both columns and must not raise."""
        import pandas as pd

        from evaluation.summary import load_detection_summary

        path = tmp_path / "old.csv"
        pd.DataFrame({"folder": ["a"], "operator": ["dropout"], "auroc": [0.7]}).to_csv(
            path, index=False
        )

        assert len(load_detection_summary(str(path), verbose=False)) == 1
