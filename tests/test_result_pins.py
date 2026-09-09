"""Pin the psbd_metrics.json contract so a schema change cannot pass unnoticed.

The pins used to address `placements["pre_residual"]` in a results file that the
rewrite stopped tracking. Both halves of that failed: the placement vocabulary
became `<position>_<operator>`, so the key does not exist, and the file is
gitignored, so on a clean checkout the loader skipped and all three tests passed
while checking nothing. A pin that silently skips is worse than no pin.

The fixture under tests/fixtures/ is a trimmed copy of a real analysis output,
committed so the pins have a versioned data source and never skip. Values are
verbatim, so a change in how the reader parses them fails here.

What this does NOT do is catch analysis-pipeline drift; a static fixture cannot.
That is scripts/verify_results.py's job, which dumps every AUROC on 2 branches
and diffs them. What this catches is the schema moving under the readers, which
is exactly what went unnoticed for the two days these tests were inert.
"""

import json
import os
from pathlib import Path

import pytest

RESULTS_DIR = Path("results")
FIXTURES_DIR = Path(__file__).parent / "fixtures"
# The placement the fixture carries, in the current <position>_<operator> naming.
PINNED_PLACEMENT = "before_attention_norm_token_mask"


def _load_fixture():
    """The tracked sample. Never skips, which is the whole point of committing it."""
    with open(FIXTURES_DIR / "psbd_metrics_sample.json") as handle:
        return json.load(handle)


def _load_psbd_metrics(folder_name):
    path = RESULTS_DIR / folder_name / "psbd_metrics.json"
    if not path.exists():
        pytest.skip(f"{path} not found")
    with open(path) as f:
        return json.load(f)


def _load_metrics(folder_name):
    path = RESULTS_DIR / folder_name / "metrics.json"
    if not path.exists():
        pytest.skip(f"{path} not found")
    with open(path) as f:
        return json.load(f)


class TestPSBDMetricsPins:
    """Pin AUROC values from psbd_metrics.json for representative checkpoints."""

    def test_fixture_oracle_auroc(self):
        placement = _load_fixture()["placements"][PINNED_PLACEMENT]
        assert abs(placement["oracle"]["auroc"] - 0.9400195312500002) < 1e-5

    def test_fixture_adaptive_auroc(self):
        placement = _load_fixture()["placements"][PINNED_PLACEMENT]
        assert abs(placement["adaptive"]["auroc"] - 0.8795405864197531) < 1e-5

    def test_fixture_oracle_tpr(self):
        placement = _load_fixture()["placements"][PINNED_PLACEMENT]
        assert abs(placement["oracle"]["tpr"] - 0.9093055725097656) < 1e-5

    def test_live_metrics_match_the_pinned_schema(self):
        """A live analysis output must still have the shape the fixture records.

        This is the check the old pins were reaching for. When the placement
        vocabulary changed from the `pre_residual` alias to
        `<position>_<operator>`, nothing failed; the readers simply started
        raising KeyError much later, in a table generator.
        """
        live = _load_psbd_metrics("vit_cifar10_adaptive_blend_0_1")
        fixture = _load_fixture()

        assert set(fixture) - {"_fixture_note"} <= set(live), (
            "a top-level key the fixture pins is missing from live output"
        )
        assert PINNED_PLACEMENT in live["placements"], (
            f"{PINNED_PLACEMENT} is absent, so the placement naming has changed"
        )
        pinned = fixture["placements"][PINNED_PLACEMENT]
        actual = live["placements"][PINNED_PLACEMENT]
        assert set(pinned) == set(actual), "the placement block's keys have changed"
        for mode in ("oracle", "adaptive"):
            assert set(pinned[mode]) == set(actual[mode]), (
                f"the {mode} block's keys have changed"
            )

    def test_swin_cifar10_blend_exists(self):
        data = _load_psbd_metrics("swin_cifar10_blend_0_05")
        assert "placements" in data
        assert data["dataset"] == "cifar10"
        assert data["attack"] == "blend"

    def test_swin_cifar100_badnet_exists(self):
        data = _load_psbd_metrics("swin_cifar100_badnet_a2o_0_005")
        assert data["dataset"] == "cifar100"
        assert data["attack"] == "badnet_a2o"


class TestMetadataConsistency:
    """Verify psbd_metrics.json metadata fields are consistent."""

    @pytest.mark.parametrize(
        "folder",
        [
            "vit_cifar10_adaptive_blend_0_1",
            "swin_cifar10_blend_0_05",
            "swin_cifar100_badnet_a2o_0_005",
        ],
    )
    def test_required_fields(self, folder):
        data = _load_psbd_metrics(folder)
        for field in (
            "folder_name",
            "dataset",
            "attack",
            "label_mode",
            "poison_rate",
            "headline_quantile",
            "placements",
        ):
            assert field in data, f"missing field {field}"

    @pytest.mark.parametrize(
        "folder",
        [
            "vit_cifar10_adaptive_blend_0_1",
            "swin_cifar10_blend_0_05",
        ],
    )
    def test_placement_structure(self, folder):
        data = _load_psbd_metrics(folder)
        placements = data["placements"]
        assert len(placements) > 0
        for name, placement in placements.items():
            assert "oracle" in placement or "adaptive" in placement, (
                f"placement {name} has neither oracle nor adaptive"
            )
            for key in ("oracle", "adaptive"):
                report = placement.get(key)
                if report is None:
                    continue
                for field in ("auroc", "tpr", "fpr", "threshold", "quantile"):
                    assert field in report, f"{name}/{key} missing {field}"
                assert 0 <= report["auroc"] <= 1
                assert 0 <= report["tpr"] <= 1
                assert 0 <= report["fpr"] <= 1


class TestResultsCoverage:
    """Verify the tracked results cover the expected checkpoint set."""

    def test_psbd_metrics_count(self):
        count = len(list(RESULTS_DIR.glob("*/psbd_metrics.json")))
        assert count >= 70, f"expected 70+ psbd_metrics.json files, found {count}"

    def test_both_architectures_present(self):
        folders = [p.parent.name for p in RESULTS_DIR.glob("*/psbd_metrics.json")]
        vit = [f for f in folders if f.startswith("vit_")]
        swin = [f for f in folders if f.startswith("swin_")]
        assert len(vit) >= 10, f"expected 10+ ViT results, found {len(vit)}"
        assert len(swin) >= 10, f"expected 10+ Swin results, found {len(swin)}"
