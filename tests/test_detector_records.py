"""The on-disk contract of a detector record, and the tie diagnostics beside it.

A record that round-trips, a skip rule that ignores failed records, an atomic
write that leaves no temporary file behind and a manifest digest that changes
when a row order changes are what let 4 job groups write into 1 results tree
without a merge step.
"""

import os

import pytest
import torch

from defences.decision import detection_report, threshold_diagnostics
from detectors.records import (
    STATUS_FAILED,
    STATUS_SCORED,
    legacy_report_present,
    load_report,
    load_scores,
    manifest_digest,
    report_path,
    save_report,
    save_scores,
    scored_detectors,
    scores_path,
)


def test_report_round_trips_and_leaves_no_temporary_file(tmp_path):
    path = report_path(str(tmp_path), "vit_gtsrb_benign", "strip")
    save_report(path, {"detector": "strip", "status": STATUS_SCORED, "auroc": 0.5})

    assert load_report(path) == {
        "detector": "strip",
        "status": STATUS_SCORED,
        "auroc": 0.5,
    }
    assert os.listdir(os.path.dirname(path)) == ["strip_metrics.json"]


def test_scores_round_trip_as_float32_on_cpu(tmp_path):
    path = scores_path(str(tmp_path), "vit_gtsrb_benign", "strip", "clean")
    save_scores(path, torch.arange(4, dtype=torch.float64))

    loaded = load_scores(path)
    assert loaded.dtype == torch.float32 and loaded.shape == (4,)
    assert torch.equal(loaded, torch.arange(4, dtype=torch.float32))


def test_an_unknown_split_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown split"):
        scores_path(str(tmp_path), "folder", "strip", "test")


def test_only_scored_records_count_as_finished(tmp_path):
    results = str(tmp_path)
    save_report(
        report_path(results, "f", "strip"),
        {"detector": "strip", "status": STATUS_SCORED},
    )
    save_report(
        report_path(results, "f", "teco"), {"detector": "teco", "status": STATUS_FAILED}
    )

    assert scored_detectors(results, "f") == {"strip"}
    assert scored_detectors(results, "missing") == set()


def test_the_legacy_single_file_record_is_only_counted(tmp_path):
    folder = tmp_path / "f"
    folder.mkdir()
    assert not legacy_report_present(str(tmp_path), "f")
    (folder / "baseline_metrics.json").write_text("{}")
    assert legacy_report_present(str(tmp_path), "f")


def test_manifest_digest_is_order_sensitive():
    manifest = {
        "heldout_indices": [3, 1, 2],
        "analysis_clean_indices": [5, 4],
        "analysis_backdoor_indices": [4],
    }
    reordered = dict(manifest, heldout_indices=[1, 2, 3])

    assert manifest_digest(manifest) == manifest_digest(dict(manifest))
    assert manifest_digest(manifest) != manifest_digest(reordered)


def test_a_tie_on_the_extreme_grid_value_zeroes_the_quantile_tpr_but_not_the_roc():
    """A score on a 2-value grid with 30% of clean samples on the low value.

    The 5% quantile lands exactly on that value, nothing is strictly below it,
    and the rule flags nothing even though every poisoned sample sits there and
    AUROC is 0.85. That is the artefact the diagnostics exist to expose.
    """
    validation = torch.cat([torch.zeros(30), torch.ones(70)])
    clean = torch.cat([torch.zeros(15), torch.ones(35)])
    backdoor = torch.zeros(50)

    diagnostics = threshold_diagnostics(validation, clean, backdoor, 0.05)
    report = detection_report(validation, clean, backdoor, 0.05)

    assert report["tpr"] == 0.0 and report["auroc"] == pytest.approx(0.85)
    assert diagnostics["tie_share_at_threshold"] == pytest.approx(0.30)
    assert diagnostics["tpr_interpolated"] > 0.0


def test_tie_share_is_0_on_continuous_scores():
    generator = torch.Generator().manual_seed(0)
    validation = torch.rand(1000, generator=generator)
    clean = torch.rand(500, generator=generator)
    backdoor = torch.rand(500, generator=generator) * 0.2

    diagnostics = threshold_diagnostics(validation, clean, backdoor, 0.10)

    assert diagnostics["tie_share_at_threshold"] == 0.0
    assert 0.0 < diagnostics["tpr_interpolated"] <= 1.0
