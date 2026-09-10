"""Per-sample PSBD scores: what number a sample gets, from cached tensors.

defences.decision turns these numbers into a verdict. Nothing here knows about
thresholds, quantiles, TPR or FPR. Nothing here imports decision either.
Everything is CPU arithmetic over the tensors stage 1 wrote, so a scoring rule can
be reconsidered in seconds without a GPU.

2 quantities from the PSBD paper (papers/PSBD/sec/4_method.tex).

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

Low PSU means poisoned, because a backdoored model's trigger-to-target path
survives a perturbation that destroys ordinary class evidence.
"""

import torch


def psu_from_cache(
    baseline_probs: torch.Tensor,
    baseline_labels: torch.Tensor,
    per_pass_probs: torch.Tensor,
) -> torch.Tensor:
    """Per-sample PSU, shape (N,), float32.

    baseline_probs is the (N, num_classes) no-dropout softmax, baseline_labels
    its (N,) argmax and per_pass_probs the (k, N) probability that each dropout
    pass assigned to that same argmax class. The subtraction is the paper's
    equation with no reinterpretation.
    """
    tracked = baseline_probs.gather(1, baseline_labels.view(-1, 1).long()).squeeze(1)

    psu = (tracked.float() - per_pass_probs.float().mean(dim=0)).float()  # (N,)
    return psu


def psu_ratio_from_cache(
    baseline_probs: torch.Tensor,
    baseline_labels: torch.Tensor,
    per_pass_probs: torch.Tensor,
) -> torch.Tensor:
    """PSU as a fraction of the starting confidence, shape (N,), float32.

        psu_ratio(x) = 1 - mean_over_passes(prob_with_dropout(c)) / prob_no_dropout(c)

    The paper's PSU is an absolute drop, which invites the objection that it really
    measures baseline confidence: a sample starting near 1 has more room to fall
    than a sample starting at 0.6, and a backdoored model is very confident on
    triggered inputs. Dividing by the starting confidence removes that. If the
    objection held this form would separate worse, and it separates better across
    the whole grid, so PSU is measuring how robust a prediction is, not how
    confident it began. The ratio is also scale-free, so a quantile threshold on it
    does not inherit the validation set's calibration.

    Reported alongside the paper's absolute form rather than replacing it, so every
    number stays comparable to the published method.
    """
    tracked = baseline_probs.gather(1, baseline_labels.view(-1, 1).long()).squeeze(1)
    # Clamped because a sample the model gave almost no probability would
    # otherwise divide by 0 and swamp the whole distribution.
    tracked = tracked.float().clamp_min(1e-6)

    psu_ratio = ((tracked - per_pass_probs.float().mean(dim=0)) / tracked).float()
    return psu_ratio


def shift_ratio(
    baseline_labels: torch.Tensor, per_pass_argmax: torch.Tensor
) -> float | None:
    """Sigma over a split: the fraction of passes whose prediction moved.

    Returns None when the cache predates argmax saving, so a caller reports the
    rate as unavailable rather than silently treating a missing tensor as zero
    shift, which would make the adaptive rule pick the wrong rate.
    """
    if per_pass_argmax.numel() == 0:
        return None

    shifted = per_pass_argmax.long() != baseline_labels.view(1, -1).long()  # (k, N)

    sigma = float(shifted.float().mean().item())
    return sigma


def shift_target_histogram(
    baseline_labels: torch.Tensor, per_pass_argmax: torch.Tensor, num_classes: int
) -> list[int] | None:
    """Counts of which class each shifted prediction landed on, length num_classes.

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

    histogram = torch.bincount(landed, minlength=num_classes).tolist()
    return histogram


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
    pass flipping is enough, and 0.5 means a majority must flip.

    Samples that never flip at any swept rate get p* = max_rate + 1, making them
    the most-robust points in the ranking without needing a sentinel.

    Returns a (N,) float32 tensor. Lower p* means the sample's prediction is
    more fragile under this perturbation (clean-like under PSBD's model). Higher
    p* means more robust (backdoor-like).
    """
    n = baseline_labels.shape[0]
    labels = baseline_labels.long().unsqueeze(0)  # (1, N)
    sentinel = sorted_rates[-1] + 1.0 if sorted_rates else 2.0
    result = torch.full((n,), sentinel, dtype=torch.float32)

    for rate in sorted_rates:
        argmax = argmax_by_rate[rate]  # (k, N)
        if argmax.numel() == 0:
            continue
        shifted_frac = (argmax.long() != labels).float().mean(dim=0)  # (N,)
        flipped = shifted_frac > flip_fraction
        # Ascending scan, so a sample still at the sentinel has not flipped yet
        # and this rate is its first.
        first_flip = flipped & (result >= sentinel)
        result[first_flip] = rate

    return result


def to_rank(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Each score as its percentile within a shared reference distribution.

    Ranking against a common reference (clean validation) keeps scores from
    different operators commensurable without fitting anything. Ranking each
    split against itself pins TPR to FPR and destroys the method, which
    cli.fuse_detectors explains in full.
    """
    sorted_reference = reference.sort().values
    positions = torch.searchsorted(sorted_reference, values.contiguous())

    ranks = positions.float() / max(len(sorted_reference), 1)
    return ranks


# How k probe ranks are combined into a single score. "min" is the union rule: any
# single probe finding a sample suspicious is enough. "median" needs a majority.
PROBE_REDUCTIONS: tuple[str, ...] = ("min", "median")


def multi_probe_score(
    psu_per_probe: list[torch.Tensor],
    psu_val_per_probe: list[torch.Tensor],
    reduction: str = "min",
) -> torch.Tensor:
    """The combined score across k probes, shape (N,). Lower means more suspicious.

    Each probe's PSU is ranked against its own clean-validation reference before
    the ranks are combined, which puts every probe on a single scale. Raw PSU
    scales differ between probes, so a raw combination would be dominated by
    whichever probe produces the smallest numbers.

    "min" is the union rule: a sample looks clean only if it looks clean to every
    probe. It is the most sensitive combination and the one an adaptive attacker
    should be assumed to attack. "median" needs a majority to agree. It exists
    because a probe the attacker has inverted contributes confident wrong evidence
    rather than none. "min" adopts the most extreme evidence available, so a single
    inverted probe breaks it, while the median holds until the attacker controls a
    majority.

    psu_per_probe holds k tensors of shape (N,) for the split being scored, and
    psu_val_per_probe k tensors of shape (M,) for clean validation, the reference
    each is ranked against.
    """
    if reduction not in PROBE_REDUCTIONS:
        raise ValueError(
            f"Unknown probe reduction: {reduction!r}, expected one of {PROBE_REDUCTIONS}"
        )

    ranks = torch.stack(
        [to_rank(psu, val) for psu, val in zip(psu_per_probe, psu_val_per_probe)]
    )  # (k, N)

    if reduction == "min":
        combined = ranks.min(dim=0).values  # (N,)
    else:
        combined = ranks.median(dim=0).values  # (N,)

    return combined
