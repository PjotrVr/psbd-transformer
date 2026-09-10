"""The comparison table's reading rules, on hand-built records.

PSBD's per-quantile numbers come from the chosen rate's fractional block, never
from the summary blocks that hold the absolute form at 1 quantile. Aggregates
span the common coverage of the columns that have data, so a column with no
records yet neither empties the table nor inflates a mean.
"""

from cli.compare_detectors import (
    active_columns,
    common_coverage,
    metric,
    psbd_values,
)

REPORT = {
    "placements": {
        "before_attention_norm_token_mask": {
            "adaptive_rate": 0.5,
            "matched_shift": {"sigma0.6": {"rate": 0.4}},
            "rates": [
                {
                    "rate": 0.4,
                    "detection_psu_ratio": {"q0.25": {"auroc": 0.90, "tpr": 0.8}},
                    "n_samples": {"backdoor": 87},
                },
                {
                    "rate": 0.5,
                    "detection_psu_ratio": {"q0.25": {"auroc": 0.95, "tpr": 0.9}},
                    "n_samples": {"backdoor": 87},
                },
            ],
        }
    }
}


def test_the_adaptive_and_matched_rules_read_different_rates():
    adaptive = psbd_values(REPORT, "before_attention_norm_token_mask", "adaptive")
    matched = psbd_values(REPORT, "before_attention_norm_token_mask", "matched")

    assert adaptive["_rate"] == 0.5 and metric(adaptive, "auroc", "q0.25") == 0.95
    assert matched["_rate"] == 0.4 and metric(matched, "auroc", "q0.25") == 0.90
    assert adaptive["_n_backdoor"] == 87


def test_a_missing_placement_or_rate_reads_as_none():
    assert psbd_values(REPORT, "post_residual", "adaptive") is None
    assert psbd_values(None, "before_attention_norm_token_mask", "adaptive") is None
    silent = {"placements": {"p": {"adaptive_rate": None, "rates": []}}}
    assert psbd_values(silent, "p", "adaptive") is None


def test_aggregates_span_the_common_coverage_of_columns_with_data():
    cells = [{"folder": "a", "kind": "attack"}, {"folder": "b", "kind": "attack"}]
    values = {
        ("a", "psbd_adaptive"): {"q0.25": {"auroc": 0.9}},
        ("b", "psbd_adaptive"): {"q0.25": {"auroc": 0.8}},
        ("a", "strip"): {"q0.25": {"auroc": 0.7}},
    }
    columns = ["psbd_adaptive", "strip", "teco"]

    assert active_columns(cells, columns, values) == ["psbd_adaptive", "strip"]
    covered = common_coverage(cells, columns, values, "auroc", "q0.25")
    assert [cell["folder"] for cell in covered] == ["a"]
