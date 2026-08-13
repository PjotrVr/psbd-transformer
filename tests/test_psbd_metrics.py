"""Stage-2 arithmetic, checked against hand-computed values.

Every detection number this project reports comes out of these five functions,
so each one is pinned to a case whose answer can be worked out on paper. The
synthetic PSU distributions are deliberately tiny and ordered, so a wrong
quantile convention, a flipped comparison, or an inverted AUROC sign changes the
expected number visibly rather than by a rounding-sized amount.
"""

import math

import torch

from defences.psbd_metrics import (
    detection_report,
    pair_clean_to_backdoor,
    psu_from_cache,
    select_rate_adaptively,
    select_rate_by_oracle,
    shift_ratio,
    shift_target_histogram,
    threshold_at_quantile,
)


def test_psu_is_the_papers_subtraction():
    # 2 samples, 3 classes. Sample 0's argmax is class 2 at prob 0.7, sample 1's
    # is class 0 at prob 0.5. Two dropout passes give that class 0.3/0.1 and
    # 0.4/0.2, so the means are 0.2 and 0.3 and PSU is 0.5 and 0.2.
    baseline_probs = torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
    baseline_labels = torch.tensor([2, 0])
    per_pass_probs = torch.tensor([[0.3, 0.4], [0.1, 0.2]])

    psu = psu_from_cache(baseline_probs, baseline_labels, per_pass_probs)

    assert torch.allclose(psu, torch.tensor([0.5, 0.2]), atol=1e-6)


def test_psu_tracks_the_baseline_class_not_the_largest_probability():
    # The tracked class is whatever baseline_labels says, even when another class
    # has a higher baseline probability. Guards against a re-argmax creeping in.
    baseline_probs = torch.tensor([[0.9, 0.1]])
    psu = psu_from_cache(baseline_probs, torch.tensor([1]), torch.tensor([[0.05]]))
    assert torch.allclose(psu, torch.tensor([0.05]), atol=1e-6)


def test_shift_ratio_counts_sample_pass_pairs():
    # 3 samples, 2 passes, 6 pairs. Baseline is [0, 1, 2]. Pass 0 moves sample 1,
    # pass 1 moves samples 1 and 2, so 3 of 6 pairs shifted.
    baseline_labels = torch.tensor([0, 1, 2])
    per_pass_argmax = torch.tensor([[0, 5, 2], [0, 5, 7]], dtype=torch.int16)
    assert shift_ratio(baseline_labels, per_pass_argmax) == 0.5


def test_shift_ratio_is_none_when_argmax_was_never_saved():
    # Missing must read as unknown, never as zero shift, or the adaptive rate
    # rule would pick the smallest rate for every placement.
    assert shift_ratio(torch.tensor([0]), torch.empty(0, 0, dtype=torch.int16)) is None


def test_shift_target_histogram_counts_only_shifted_predictions():
    # Baseline [0, 1]. Pass 0: sample 0 stays, sample 1 moves to class 0.
    # Pass 1: sample 0 moves to class 2, sample 1 moves to class 0.
    # So class 0 gains 2, class 2 gains 1, and the unshifted prediction is excluded.
    baseline_labels = torch.tensor([0, 1])
    per_pass_argmax = torch.tensor([[0, 0], [2, 0]], dtype=torch.int16)
    assert shift_target_histogram(baseline_labels, per_pass_argmax, 3) == [2, 0, 1]


def test_threshold_is_the_linear_interpolated_quantile():
    # PSU 0..9, the 25th percentile with linear interpolation is 2.25, not 2.
    # Pins the convention against numpy's other interpolation modes.
    scores = torch.arange(10).float()
    assert math.isclose(threshold_at_quantile(scores, 0.25), 2.25, rel_tol=1e-9)


def test_detection_report_hand_computed():
    # Threshold from validation 0..99 at the 25th percentile is 24.75.
    # Backdoor PSU is 0..19, all below it, so TPR = 1.0.
    # Clean PSU is 50..69, none below it, so FPR = 0.0.
    # The two sets are perfectly separated with backdoor lower, and PSU is
    # negated before scoring, so AUROC is exactly 1.0.
    report = detection_report(
        torch.arange(100).float(),
        torch.arange(50, 70).float(),
        torch.arange(20).float(),
        quantile=0.25,
    )
    assert math.isclose(report["threshold"], 24.75, rel_tol=1e-9)
    assert report["tpr"] == 1.0
    assert report["fpr"] == 0.0
    assert math.isclose(report["auroc"], 1.0, rel_tol=1e-9)


def test_auroc_is_a_half_when_the_two_distributions_are_identical():
    # The benign-model negative control in numerical form: no separation at all
    # must read as 0.5, not as 0.0 or 1.0, which is what a sign error would give.
    same = torch.arange(50).float()
    report = detection_report(same, same, same.clone(), quantile=0.25)
    assert math.isclose(report["auroc"], 0.5, abs_tol=1e-9)


def test_auroc_is_below_a_half_when_the_backdoor_side_is_higher():
    # Sign check in the other direction: if backdoor PSU were the HIGHER set,
    # PSBD's premise is inverted and AUROC must drop below 0.5.
    report = detection_report(
        torch.arange(100).float(),
        torch.arange(20).float(),
        torch.arange(50, 70).float(),
        quantile=0.25,
    )
    assert report["auroc"] < 0.5


def test_pair_clean_to_backdoor_selects_by_original_index():
    # The clean split covers 5 analysis images, the backdoor split only 3 of them,
    # and in a different order. Pairing must return the clean scores for exactly
    # those 3, in the backdoor split's own row order, not the clean one's.
    clean_scores = torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0])
    manifest = {
        "analysis_clean_indices": [700, 701, 702, 703, 704],
        "analysis_backdoor_indices": [703, 700, 702],
    }
    paired = pair_clean_to_backdoor(clean_scores, manifest)
    assert torch.equal(paired, torch.tensor([13.0, 10.0, 12.0]))


def test_pair_clean_to_backdoor_is_identity_when_every_image_is_eligible():
    # all_to_all drops nothing, so pairing must be a no-op rather than a reorder.
    clean_scores = torch.tensor([1.0, 2.0, 3.0])
    manifest = {
        "analysis_clean_indices": [5, 6, 7],
        "analysis_backdoor_indices": [5, 6, 7],
    }
    assert torch.equal(pair_clean_to_backdoor(clean_scores, manifest), clean_scores)


def test_adaptive_rate_is_the_smallest_reaching_the_target():
    # 0.5 is the first rate at or above 0.8. Smallest matters: a larger rate that
    # also clears the target destroys more clean evidence for no extra benefit.
    shift_by_rate = {0.1: 0.10, 0.3: 0.55, 0.5: 0.82, 0.7: 0.95}
    assert select_rate_adaptively(shift_by_rate) == 0.5


def test_adaptive_rate_is_none_when_no_rate_reaches_the_target():
    # A real outcome for a placement that barely perturbs the model, and it must
    # be recorded as "no rate qualifies" rather than defaulting to the largest.
    assert select_rate_adaptively({0.1: 0.2, 0.9: 0.4}) is None


def test_adaptive_rate_ignores_rates_with_no_saved_argmax():
    assert select_rate_adaptively({0.1: None, 0.5: 0.9}) == 0.5


def test_oracle_rate_picks_the_best_auroc_and_skips_nan():
    assert select_rate_by_oracle({0.1: 0.55, 0.5: 0.91, 0.9: float("nan")}) == 0.5
