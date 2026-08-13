"""Stage 2 of PSBD: turn cached per-pass tensors into detection numbers.

Stage 1 is the expensive GPU half; it writes raw per-pass probabilities and
argmax classes to disk (defences.psbd_cache). Everything here is pure CPU
arithmetic over those tensors, so a threshold, a quantile, or a whole scoring
rule can be reconsidered in seconds without touching a GPU again.

Three quantities, all from the PSBD paper (papers/PSBD/sec/4_method.tex):

Prediction Shift Uncertainty, Eq. (PSU definition):

    original form
        phi_PSU(x) = P_c(x; theta) - (1/k) * sum_{i=1..k} P_c(x; p, theta_i')
        with c = argmax_c P(x; theta)

    descriptive form
        psu(x) = prob_no_dropout(predicted_class)
                 - mean_over_passes(prob_with_dropout(that same class))

Shift ratio, Eq. (PS definition):

    original form
        phi_PS(x) = I(Y(x; theta) != Y(x; theta')),
        sigma(D)  = (1 / (k * |D|)) * sum_{x in D} phi_PS(x)

    descriptive form
        shift_ratio = fraction of all (sample, pass) predictions that differ
                      from the same sample's no-dropout prediction

Detection: a sample is flagged when psu(x) < threshold, and the threshold is the
25th-percentile PSU of a clean validation set the defender is assumed to hold.
Low PSU means poisoned, because a backdoored model's trigger-to-target path
survives dropout that destroys ordinary class evidence.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# The paper sets the threshold at the 25th percentile of clean validation PSU in
# every experiment, and reads it as the tolerable clean loss rate. The others are
# swept alongside so the result can be shown not to hinge on that one choice.
PSBD_QUANTILES: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25)
HEADLINE_QUANTILE = 0.25

# The paper's adaptive rule selects the dropout rate where the shift ratio of
# clean validation data "approaches a high value (0.8 in our experiments)".
ADAPTIVE_SHIFT_TARGET = 0.8

# Comparing placements at a shared dropout rate compares nothing meaningful. The
# same p is a wildly different intervention depending on where it lands: a
# branch-output position perturbs one additive contribution, while a
# residual-stream position multiplies the whole stream by a mask once per block,
# so only (1-p)^12 of coordinates survive the ViT stack. At p=0.9 that is 1e-12
# against a modest branch perturbation. "Pre-residual beats post-residual" would
# then be indistinguishable from "the weaker perturbation beats the destructive
# one", which is not a finding about placement at all.
#
# The fix is to make effective strength the shared axis. The clean-validation
# shift ratio is exactly such a calibrator: it is measured on clean data only, so
# it stays defender-legal, and it says how much the perturbation actually
# disturbed the model rather than how much noise was nominally injected. Every
# placement is compared at matched sigma, and this is the ladder.
SHIFT_MATCH_TARGETS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)


def psu_from_cache(
    baseline_probs: torch.Tensor,
    baseline_labels: torch.Tensor,
    per_pass_probs: torch.Tensor,
) -> torch.Tensor:
    """Per-sample PSU, shape (N,), float32.

    baseline_probs is the (N, num_classes) no-dropout softmax, baseline_labels
    its (N,) argmax, and per_pass_probs the (k, N) probability that each dropout
    pass assigned to that same argmax class. The subtraction is the paper's
    equation with no reinterpretation.
    """
    tracked = baseline_probs.gather(1, baseline_labels.view(-1, 1).long()).squeeze(1)
    return (tracked.float() - per_pass_probs.float().mean(dim=0)).float()


def shift_ratio(
    baseline_labels: torch.Tensor, per_pass_argmax: torch.Tensor
) -> float | None:
    """Sigma over one split: the fraction of passes whose prediction moved.

    Returns None when the cache predates argmax saving, so a caller reports the
    rate as unavailable rather than silently treating a missing tensor as zero
    shift, which would make the adaptive rule pick the wrong rate.
    """
    if per_pass_argmax.numel() == 0:
        return None
    shifted = per_pass_argmax.long() != baseline_labels.view(1, -1).long()
    return float(shifted.float().mean().item())


def shift_target_histogram(
    baseline_labels: torch.Tensor, per_pass_argmax: torch.Tensor, num_classes: int
) -> list[int] | None:
    """Counts of which class each shifted prediction landed on.

    PSBD's mechanism claim is that clean samples under dropout do not scatter,
    they collapse onto the attacker's target class, because that is the strongest
    association the poisoned model learned. This histogram is the direct test of
    that claim: a spike at the target class supports it, a flat or
    majority-class-shaped histogram does not.
    """
    if per_pass_argmax.numel() == 0:
        return None
    shifted_mask = per_pass_argmax.long() != baseline_labels.view(1, -1).long()
    landed = per_pass_argmax.long()[shifted_mask]
    return torch.bincount(landed, minlength=num_classes).tolist()


def pair_clean_to_backdoor(clean_scores: torch.Tensor, manifest: dict) -> torch.Tensor:
    """Restrict clean scores to the images the backdoor split actually contains.

    The clean split covers the whole analysis pool; the backdoor split covers
    only the eligible subset (all_to_one drops the target class, clean_label
    keeps only non-target sources). Comparing the two as served would contrast
    different image populations, so FPR and AUROC would partly measure which
    classes were dropped rather than the defence. The manifest records both index
    lists in loader-row order, so this maps backdoor rows back onto their clean
    counterparts and returns the clean scores in backdoor row order.
    """
    clean_indices = manifest["analysis_clean_indices"]
    backdoor_indices = manifest["analysis_backdoor_indices"]
    row_of = {original: row for row, original in enumerate(clean_indices)}
    rows = [row_of[original] for original in backdoor_indices]
    return clean_scores[torch.tensor(rows, dtype=torch.long)]


def threshold_at_quantile(validation_psu: torch.Tensor, quantile: float) -> float:
    """The detection threshold: a low quantile of clean validation PSU.

    Uses numpy's linear interpolation, matching the "25th percentile" the paper
    reports. Needs no backdoor knowledge, which is the point: the quantile is
    chosen as a tolerable clean loss rate, not fitted against poisoned data.
    """
    return float(np.quantile(validation_psu.float().numpy(), quantile))


def detection_report(
    validation_psu: torch.Tensor,
    clean_psu: torch.Tensor,
    backdoor_psu: torch.Tensor,
    quantile: float,
) -> dict:
    """TPR, FPR, and AUROC at one quantile, plus the threshold that produced them.

    AUROC negates both score sets because low PSU is the positive (poisoned)
    evidence, and roc_auc_score expects higher to mean more positive.
    """
    threshold = threshold_at_quantile(validation_psu, quantile)
    tpr = float((backdoor_psu < threshold).float().mean().item())
    fpr = float((clean_psu < threshold).float().mean().item())

    scores = np.concatenate([-clean_psu.float().numpy(), -backdoor_psu.float().numpy()])
    labels = np.concatenate([np.zeros(len(clean_psu)), np.ones(len(backdoor_psu))])
    auroc = (
        float(roc_auc_score(labels, scores))
        if len(set(labels.tolist())) > 1
        else float("nan")
    )

    return {
        "quantile": quantile,
        "threshold": threshold,
        "tpr": tpr,
        "fpr": fpr,
        "auroc": auroc,
    }


def select_rate_adaptively(
    shift_by_rate: dict[float, float | None], target: float = ADAPTIVE_SHIFT_TARGET
) -> float | None:
    """The smallest dropout rate whose clean-validation shift ratio reaches target.

    The paper's rule has two halves: sigma of clean validation approaches 0.8,
    and the gap between sigma of the whole training set and sigma of clean
    validation is maximal. Only the first half transfers here. The second half
    carries signal in the paper because their scored pool is the poisoned
    training set; ours is a clean test pool, so that gap is noise around zero by
    construction. The surviving half is still defender-legal: it reads clean
    validation data only and never touches the backdoor split.

    Returns None when no swept rate reaches the target, which is itself a finding
    rather than an error, so the caller records it instead of guessing a rate.
    """
    reached = sorted(
        rate
        for rate, sigma in shift_by_rate.items()
        if sigma is not None and sigma >= target
    )
    return reached[0] if reached else None


def select_rate_at_matched_shift(
    shift_by_rate: dict[float, float | None], target: float
) -> float | None:
    """The rate whose clean-validation shift ratio sits closest to target.

    This is the strength-matched comparison operator. Two placements evaluated at
    their own matched-sigma rates are being asked the same question ("given a
    perturbation that disturbs the clean model this much, how well do clean and
    backdoor samples separate?") rather than the meaningless one that a shared p
    asks. Ties break toward the smaller rate, which destroys less clean evidence
    for the same measured disturbance.
    """
    usable = {rate: value for rate, value in shift_by_rate.items() if value is not None}
    if not usable:
        return None
    return min(sorted(usable), key=lambda rate: abs(usable[rate] - target))


def attack_success_mask(
    baseline_labels: torch.Tensor, loader_labels: torch.Tensor
) -> torch.Tensor | None:
    """Which triggered samples the backdoor actually captured, from the baseline.

    A triggered image the model still classifies correctly is behaviourally
    clean: nothing about it was memorized, so its PSU is a clean sample's PSU.
    Counting it as a detection positive understates the detector for a reason
    that has nothing to do with the detector. That is second-order at ASR 0.99
    and dominant at ASR 0.5, which is exactly where a falsification probe like
    badnet_a2a sits, so TPR is reported both over all triggered samples and over
    the captured ones only.

    Returns None for a cache written before loader labels were saved.
    """
    if loader_labels.numel() == 0:
        return None
    return baseline_labels.long() == loader_labels.long()


def select_rate_by_oracle(auroc_by_rate: dict[float, float]) -> float | None:
    """The rate with the best AUROC. An ORACLE: it reads the backdoor labels.

    Reported only as an upper bound on what the placement could achieve with
    perfect rate selection. The gap between this and select_rate_adaptively is
    the cost of not knowing the right rate, which is a real part of PSBD's
    practical difficulty and worth measuring rather than hiding.
    """
    usable = {
        rate: value
        for rate, value in auroc_by_rate.items()
        if value == value  # NaN filter
    }
    return max(usable, key=usable.get) if usable else None
