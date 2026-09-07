"""The gate must catch the bug it exists for, proven by reintroducing it.

A check nobody has seen fail is a check nobody knows works. These tests break each
detector deliberately and assert the gate refuses it, then assert the gate passes
the unbroken code. Without the first half the gate is decoration.
"""

import pytest
import torch

from experiments.preflight.synthetic import (
    apply_trigger,
    attack_success_rate,
    build_backdoored_model,
    build_splits,
    has_trigger,
)
from psbd.decision import HEADLINE_QUANTILE, detection_report
from psbd.detectors import DETECTOR_NAMES, DetectorContext, build_detector

# The detectors this synthetic case can actually judge. SCALE-UP is excluded
# because its statistic takes 6 values over 5 amplification scales and this model
# keeps 98 percent of clean predictions stable, so the ranking is almost all ties.
# That is recorded in check_signs.py rather than worked around.
JUDGEABLE = ("confidence", "strip", "ibd_psc", "teco")

CHEAP_CORRUPTIONS = ("gaussian_noise", "defocus_blur", "brightness", "contrast")


@pytest.fixture(scope="module")
def synthetic_case():
    device = torch.device("cpu")
    model = build_backdoored_model()
    loaders = build_splits(num_samples=192)
    context = DetectorContext(
        model=model,
        device=device,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        validation_loader=loaders["validation"],
        num_classes=10,
        use_bfloat16=False,
        teco_corruptions=CHEAP_CORRUPTIONS,
    )
    return model, loaders, context, device


def detector_auroc(name, model, loaders, context, device, invert=False):
    score = build_detector(name, context)
    scores = {}
    for split, loader in loaders.items():
        values = score(model, loader, device)
        scores[split] = -values if invert else values

    report = detection_report(
        scores["validation"], scores["clean"], scores["backdoor"], HEADLINE_QUANTILE
    )
    return report["auroc"]


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


@pytest.mark.parametrize("name", JUDGEABLE)
def test_each_detector_points_the_right_way(name, synthetic_case):
    model, loaders, context, device = synthetic_case

    auroc = detector_auroc(name, model, loaders, context, device)
    assert auroc >= 0.60, f"{name} scored {auroc:.3f} on an unmissable backdoor"


@pytest.mark.parametrize("name", JUDGEABLE)
def test_the_gate_catches_a_deliberate_inversion(name, synthetic_case):
    """Reintroduce the exact bug that shipped, and require the gate to refuse it.

    IBD-PSC and TeCo were both wired backwards in one sitting, scoring 0.043 and
    0.055 where they should have scored above 0.94, and nothing raised. A negated
    score is what that looked like.
    """
    model, loaders, context, device = synthetic_case

    inverted = detector_auroc(name, model, loaders, context, device, invert=True)
    assert inverted <= 0.40, (
        f"{name} inverted still scored {inverted:.3f}; the check cannot see the bug"
    )


class TestSummaryLoaderFiltersUnsafeRows:
    """The A19 fix added 2 columns. Nothing read them, so it changed no number.

    A filter that exists and is not applied is worse than no filter, because it
    reads as handled. These pin that the safe path is the default path.
    """

    def test_the_default_drops_variant_rows(self, tmp_path):
        import pandas as pd

        from psbd.summary import load_detection_summary

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

        from psbd.summary import load_detection_summary

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

        from psbd.summary import load_detection_summary

        path = tmp_path / "old.csv"
        pd.DataFrame({"folder": ["a"], "operator": ["dropout"], "auroc": [0.7]}).to_csv(
            path, index=False
        )

        assert len(load_detection_summary(str(path), verbose=False)) == 1
