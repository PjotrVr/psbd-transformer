"""Pin known-good result values so branch cleanup cannot silently change numbers.

Each test loads a tracked psbd_metrics.json file and asserts that specific
AUROC/TPR/FPR values match pinned constants. Tolerances are tight (1e-5) to
catch any analysis pipeline regression.

Tests are skipped if the result files are not present (CI without checkpoints).
"""

import json
import os
from pathlib import Path

import pytest

RESULTS_DIR = Path("results")


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

    def test_vit_cifar10_adaptive_blend_oracle_auroc(self):
        data = _load_psbd_metrics("vit_cifar10_adaptive_blend_0_1")
        auroc = data["placements"]["pre_residual"]["oracle"]["auroc"]
        assert abs(auroc - 0.7839836130401233) < 1e-5

    def test_vit_cifar10_adaptive_blend_adaptive_auroc(self):
        data = _load_psbd_metrics("vit_cifar10_adaptive_blend_0_1")
        auroc = data["placements"]["pre_residual"]["adaptive"]["auroc"]
        assert abs(auroc - 0.6665406346450617) < 1e-5

    def test_vit_cifar10_adaptive_blend_tpr(self):
        data = _load_psbd_metrics("vit_cifar10_adaptive_blend_0_1")
        tpr = data["placements"]["pre_residual"]["oracle"]["tpr"]
        assert abs(tpr - 0.7038888931274414) < 1e-5

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
