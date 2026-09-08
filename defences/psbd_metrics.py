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
# every experiment, and reads it as the tolerable clean loss rate. That is a quarter
# of clean data discarded, which no deployment tolerates, so 0.01 and 0.05 are swept
# alongside: the quantile IS the false-positive budget, so these are the 1% and 5%
# FPR operating points a defender would actually pick. The paper's 0.25 stays the
# headline purely so the numbers remain comparable to the published tables.
PSBD_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25)
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


def complete_rates(psbd_dir: str, position_config: str) -> list[float]:
    """Rates whose validation, clean AND backdoor tensors are all on disk.

    Stage 1 writes the three splits of a rate one after another, so while a sweep is
    running there is always a rate with some splits present and some missing.
    Discovering rates by globbing any rate_*.pt therefore returns rates that cannot be
    loaded, and every analysis script that did so crashed partway through the moment
    it ran concurrently with a job.

    Requiring all three makes analysis safe to run at any time against a live results
    tree, which matters because the sweep now runs in multi-hour batches and waiting
    for it to finish is not practical.
    """
    import os

    folder = os.path.join(psbd_dir, position_config)
    if not os.path.isdir(folder):
        return []
    seen: dict[float, set[str]] = {}
    for name in os.listdir(folder):
        if not name.startswith("rate_") or not name.endswith(".pt"):
            continue
        tag, split = name[len("rate_") : -len(".pt")].rsplit("_", 1)
        seen.setdefault(float(tag.replace("_", ".")), set()).add(split)
    return sorted(
        rate
        for rate, splits in seen.items()
        if splits >= {"validation", "clean", "backdoor"}
    )


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


def psu_ratio_from_cache(
    baseline_probs: torch.Tensor,
    baseline_labels: torch.Tensor,
    per_pass_probs: torch.Tensor,
) -> torch.Tensor:
    """PSU as a fraction of the starting confidence, shape (N,), float32.

        psu_ratio(x) = 1 - mean_over_passes(prob_with_dropout(c)) / prob_no_dropout(c)

    The paper's PSU is an absolute drop, which invites the objection that it is
    really measuring baseline confidence: a sample starting near probability 1 has
    more room to fall than one starting at 0.6, and a backdoored model is very
    confident on triggered inputs. Dividing by the starting confidence removes that
    entirely, so if the objection held, this form would separate worse.

    It separates better. Across all 624 (checkpoint, placement, rate) cells of the
    CIFAR-10 ViT grid it beats the absolute form in 92.8%, mean AUROC +0.019, and
    its worst regression anywhere is -0.0025. So PSU is measuring how robust a
    prediction is, not how confident it began.

    There is also a threshold argument for preferring it. The detection threshold is
    a quantile of clean-validation PSU, and absolute PSU is bounded above by the
    starting confidence, so the threshold inherits the validation set's calibration.
    The ratio is scale-free and should transfer better across datasets and models
    whose confidence is differently calibrated, though that is untested here.

    Reported alongside the paper's absolute form rather than replacing it, so every
    number stays comparable to the published method.
    """
    tracked = baseline_probs.gather(1, baseline_labels.view(-1, 1).long()).squeeze(1)
    tracked = tracked.float().clamp_min(1e-6)
    return ((tracked - per_pass_probs.float().mean(dim=0)) / tracked).float()


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

    auroc_two_sided is max(auroc, 1 - auroc). Retained as a diagnostic field only.
    H15 retired two-sided reporting: a two-sided statistic can flatter any result,
    and using it requires oracle access to the labels. No live consumer reads it.
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
        "auroc_two_sided": max(auroc, 1.0 - auroc) if auroc == auroc else float("nan"),
        # "inverted" means backdoor samples are LESS robust to the perturbation than
        # clean ones, the opposite of PSBD's premise.
        "direction": ("inverted" if auroc < 0.5 else "as_expected")
        if auroc == auroc
        else None,
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


def critical_rate(
    baseline_labels: torch.Tensor,
    sorted_rates: list[float],
    argmax_by_rate: dict[float, torch.Tensor],
    flip_fraction: float = 0.0,
) -> torch.Tensor:
    """Per-sample critical rate p*: the smallest rate at which a prediction flips.

    For each sample, scan rates in ascending order and record the first rate at
    which more than flip_fraction of the k passes produced a different argmax
    from the no-dropout baseline. flip_fraction=0.0 (default) means any single
    pass flipping is enough; 0.5 means a majority must flip.

    Samples that never flip at any swept rate get p* = max_rate + 1, making them
    the most-robust points in the ranking without needing a sentinel.

    Returns a (N,) float32 tensor. Lower p* means the sample's prediction is
    more fragile under this perturbation (clean-like under PSBD's model); higher
    p* means more robust (backdoor-like).
    """
    n = baseline_labels.shape[0]
    labels = baseline_labels.long().unsqueeze(0)
    sentinel = sorted_rates[-1] + 1.0 if sorted_rates else 2.0
    result = torch.full((n,), sentinel, dtype=torch.float32)

    for rate in sorted_rates:
        argmax = argmax_by_rate[rate]
        if argmax.numel() == 0:
            continue
        shifted_frac = (argmax.long() != labels).float().mean(dim=0)
        flipped = shifted_frac > flip_fraction
        first_flip = flipped & (result >= sentinel)
        result[first_flip] = rate

    return result


def load_critical_rate_from_disk(
    psbd_dir: str, position_config: str, split: str, flip_fraction: float = 0.0
) -> torch.Tensor | None:
    """Load baseline + all rates for one (checkpoint, position, split), compute p*.

    Returns None when no complete rates exist.
    """
    from .psbd_cache import (
        baseline_path,
        dropout_pass_path,
        load_baseline,
        load_dropout_pass_probs,
    )

    rates = complete_rates(psbd_dir, position_config)
    if not rates:
        return None

    _, baseline_labels, _ = load_baseline(baseline_path(psbd_dir, split))
    argmax_by_rate = {}
    for rate in rates:
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, position_config, rate, split)
        )
        argmax_by_rate[rate] = argmax

    return critical_rate(baseline_labels, rates, argmax_by_rate, flip_fraction)


def bracket_target_shift(
    shift_by_rate: dict[float, float | None], target: float
) -> tuple[float, float] | None:
    """The 2 swept rates whose clean-validation shift ratio brackets the target.

    Returns (lower_rate, upper_rate) with sigma(lower) <= target <= sigma(upper),
    or None when the grid never crosses the target from both sides.

    This exists because neither single-rate rule is a match on a coarse grid. On
    vit_tiny_badnet_a2o_0_01 at token_mask before_attention_norm, the swept rates
    give sigma 0.523 at 0.3 and 0.751 at 0.4, so a target of 0.6 has no rate near
    it. Selecting the nearest rate scores that cell at sigma 0.523 while selecting
    the smallest rate reaching the target scores it at 0.751, and those 2 readings
    differ by 0.117 AUROC. That is more than twice the size of the largest
    placement effect this project reports, so the choice of rule cannot be left
    implicit.
    """
    usable = sorted(
        (rate, sigma) for rate, sigma in shift_by_rate.items() if sigma is not None
    )
    if not usable:
        return None

    below = [rate for rate, sigma in usable if sigma <= target]
    above = [rate for rate, sigma in usable if sigma >= target]
    if not below or not above:
        return None

    bracket = (max(below), min(above))
    return bracket


def interpolate_at_target_shift(
    shift_by_rate: dict[float, float | None],
    value_by_rate: dict[float, float],
    target: float,
) -> float | None:
    """A metric read at exactly the target shift ratio, linear between neighbours.

        original form
            f(sigma*) = f(sigma_lo) + (f(sigma_hi) - f(sigma_lo))
                        * (sigma* - sigma_lo) / (sigma_hi - sigma_lo)

        descriptive form
            value_at_target = value_below + (value_above - value_below)
                              * (target - sigma_below) / (sigma_above - sigma_below)

    Every placement is then read at the SAME effective disturbance rather than at
    whichever grid point happened to land closest, which is the comparison that
    matching on shift ratio was introduced to make.

    Returns None when the grid does not bracket the target, which is a real
    limitation of that operator's rate grid and is reported rather than papered
    over by falling back to the nearest rate.
    """
    bracket = bracket_target_shift(shift_by_rate, target)
    if bracket is None:
        return None

    lower_rate, upper_rate = bracket
    lower_sigma, upper_sigma = shift_by_rate[lower_rate], shift_by_rate[upper_rate]
    if lower_rate not in value_by_rate or upper_rate not in value_by_rate:
        return None
    if upper_sigma == lower_sigma:
        return float(value_by_rate[lower_rate])

    span = (target - lower_sigma) / (upper_sigma - lower_sigma)
    lower_value, upper_value = value_by_rate[lower_rate], value_by_rate[upper_rate]
    interpolated = lower_value + (upper_value - lower_value) * span
    return float(interpolated)


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


def to_rank(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Each score as its percentile within a shared reference distribution.

    Ranking against a common reference (clean validation) keeps scores from
    different operators commensurable without fitting anything. Ranking each
    split against itself pins TPR to FPR and destroys the method (see
    detector_fusion.py for the full explanation).
    """
    sorted_reference = reference.sort().values
    positions = torch.searchsorted(sorted_reference, values.contiguous())
    return positions.float() / max(len(sorted_reference), 1)


def multi_probe_score(
    psu_per_probe: list[torch.Tensor],
    psu_val_per_probe: list[torch.Tensor],
) -> torch.Tensor:
    """Combined score across k probes: the minimum rank (most suspicious).

    Each probe's PSU is ranked against its own clean-validation reference, then
    the minimum across probes is the combined score. Lower means more suspicious,
    matching the low-PSU-is-poisoned convention.

    psu_per_probe: k tensors, each (N,), PSU for the split being scored.
    psu_val_per_probe: k tensors, each (M,), PSU for clean validation (the
        reference distribution for ranking).
    """
    ranks = [to_rank(psu, val) for psu, val in zip(psu_per_probe, psu_val_per_probe)]
    return torch.stack(ranks).min(dim=0).values


def multi_probe_auroc(
    clean_psu_per_probe: list[torch.Tensor],
    backdoor_psu_per_probe: list[torch.Tensor],
    val_psu_per_probe: list[torch.Tensor],
) -> float:
    """AUROC of the multi-probe min-rank score.

    Negated because lower rank = more suspicious (positive class), and
    roc_auc_score expects higher = more positive.
    """
    clean_score = multi_probe_score(clean_psu_per_probe, val_psu_per_probe)
    backdoor_score = multi_probe_score(backdoor_psu_per_probe, val_psu_per_probe)

    scores = np.concatenate([-clean_score.numpy(), -backdoor_score.numpy()])
    labels = np.concatenate([np.zeros(len(clean_score)), np.ones(len(backdoor_score))])
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def multi_probe_detection(
    val_psu_per_probe: list[torch.Tensor],
    clean_psu_per_probe: list[torch.Tensor],
    backdoor_psu_per_probe: list[torch.Tensor],
    target_fpr: float = HEADLINE_QUANTILE,
    rule: str = "calibrated",
) -> dict:
    """TPR, FPR, and AUROC of the multi-probe union defence.

    Two thresholding rules, both reading clean validation data only, so both stay
    defender-legal:

      calibrated  the target_fpr quantile of the combined validation score. That
                  score is already a minimum over k probes, so its own quantile
                  absorbs however correlated the probes happen to be and lands on
                  target_fpr by construction. This is the default.
      bonferroni  the literal value target_fpr / k on the rank scale, which flags
                  a sample when ANY single probe ranks it below that. The union
                  bound makes it conservative, so its achieved FPR is at most
                  target_fpr and usually well under.

    The earlier version applied the Bonferroni quantile target_fpr / k to the
    COMBINED score rather than to a single probe's rank, which corrects twice.
    The minimum of k ranks reaching its own target_fpr / k quantile is a much
    rarer event than any one probe reaching target_fpr / k, so the achieved FPR
    came out near target_fpr / k and TPR was understated by the same margin.
    AUROC never depended on the threshold and is unchanged.

    Both rules are reported. The top-level tpr, fpr and threshold keys follow the
    rule argument.
    """
    if rule not in ("calibrated", "bonferroni"):
        raise ValueError(f"unknown rule {rule!r}, expected calibrated or bonferroni")

    k = len(val_psu_per_probe)
    bonferroni_q = target_fpr / k

    val_score = multi_probe_score(val_psu_per_probe, val_psu_per_probe)
    clean_score = multi_probe_score(clean_psu_per_probe, val_psu_per_probe)
    backdoor_score = multi_probe_score(backdoor_psu_per_probe, val_psu_per_probe)

    def rates_at(threshold: float) -> dict:
        return {
            "threshold": threshold,
            "tpr": float((backdoor_score < threshold).float().mean()),
            "fpr": float((clean_score < threshold).float().mean()),
        }

    by_rule = {
        "calibrated": rates_at(float(np.quantile(val_score.numpy(), target_fpr))),
        "bonferroni": rates_at(bonferroni_q),
    }
    chosen = by_rule[rule]
    auroc = multi_probe_auroc(
        clean_psu_per_probe, backdoor_psu_per_probe, val_psu_per_probe
    )

    report = {
        "k": k,
        "target_fpr": target_fpr,
        "rule": rule,
        "bonferroni_quantile": bonferroni_q,
        "threshold": chosen["threshold"],
        "tpr": chosen["tpr"],
        "fpr": chosen["fpr"],
        "auroc": auroc,
        "by_rule": by_rule,
    }
    return report
