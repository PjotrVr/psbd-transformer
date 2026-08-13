# H15 — One-sided decision rules are a systematic weakness across backdoor detectors

**Status: SUPPORTED.** Found independently in two unrelated detectors. Nine
checkpoints where STRIP is genuinely anti-correlated, recovering up to +0.98 AUROC
when the sign is allowed to flip, against a measured chance floor of 0.515.

## Claim

Both PSBD and STRIP flag one tail: PSBD flags low prediction-shift uncertainty, STRIP
flags low superimposition entropy. Each assumes the backdoor makes its statistic move
in one particular direction. When that assumption fails the detector does not degrade
to chance, it **runs backwards**, and a one-sided rule scores near-perfect separation
as total failure.

This was first found for PSBD on all-to-all ([H5](H5-all-to-all-breaks-psbd.md)).
The claim here is that it is not specific to PSBD or to all-to-all.

## Evidence

STRIP, ViT CIFAR-10, 34 backdoored checkpoints, AUROC at the 25th-percentile
threshold. `max(AUROC, 1 - AUROC)` is a selection on the test statistic, so a
chance-level detector scores slightly above 0.5 under it. The benign control, scored
by the identical rule, gives the floor: **0.515**.

Checkpoints clearing that floor by a wide margin:

| checkpoint | ASR | one-sided | two-sided | gain |
|---|---|---|---|---|
| `badnet_a2a` 0.5% | 0.84 | **0.010** | **0.990** | +0.980 |
| `adaptive_blend` 1% | 0.64 | 0.215 | 0.785 | +0.569 |
| `adaptive_blend` 0.5% | 0.28 | 0.269 | 0.731 | +0.463 |
| `sig` 1% | 0.34 | 0.320 | 0.680 | +0.361 |
| `lc` 1% | 0.25 | 0.324 | 0.676 | +0.351 |
| `badnet_a2a` 5% | 0.93 | 0.341 | 0.659 | +0.319 |
| `sig` 0.5% | 0.13 | 0.368 | 0.632 | +0.265 |
| `lc` 0.5% | 0.12 | 0.390 | 0.610 | +0.220 |
| `tact` 0.5% | 0.11 | 0.415 | 0.585 | +0.170 |

Nine genuinely inverted. Five more are nominally inverted but land within 0.05 of the
floor; those are the selection effect and are **not** counted.

`badnet_a2a` at 0.5% poisoning is the extreme case: AUROC 0.010 is a detector that is
right 99% of the time about which sample is poisoned, and wrong about which direction
means poisoned.

## Which attacks invert, and why

The nine split cleanly into two families, and neither is the family STRIP was designed
against:

- **Low-amplitude, full-image triggers**: `adaptive_blend`, `sig`, `lc`. STRIP's
  premise is that a trigger survives superimposition and keeps dragging the prediction
  to the target, giving *low* entropy. A low-amplitude trigger spread over the whole
  image does not survive; it acts as added noise, which makes the superimposed image
  *harder* to classify and pushes entropy *up*. The statistic still separates, with
  the sign reversed.
- **All-to-all**: `badnet_a2a`. No single target class for the superimposed prediction
  to collapse onto, the same structural reason PSBD inverts on it.

Note what these two families have in common with the PSBD result: they are exactly the
cases where the detector's stated mechanism does not apply. The inversion is a
signature of a broken assumption, not of a broken measurement.

## Why this matters more than either individual result

The same failure appeared in two detectors that share no machinery: one measures
prediction shift under internal dropout, the other measures entropy under input
superimposition. Both are one-tailed, both invert, and both invert on the cases their
stated mechanism does not cover.

Since a two-sided rule costs almost nothing (the benign floor rises from 0.5 to 0.515)
and recovers up to +0.98, **reporting one-sided AUROC understates these detectors and
should be treated as a reporting bug rather than a result.**

For PSBD the aggregate effect is a mean of 0.671 one-sided against 0.787 two-sided
across the 34 STRIP checkpoints.

## Making it deployable: tested, and it works only if the tail is tight

`max(AUROC, 1 - AUROC)` reads the labels, so it is an upper bound rather than a
method. The label-free alternative is to pick the tail by comparing the unlabelled
suspicious pool against the clean validation split:

    low_deviation  = quantile(validation, q) - quantile(pool, q)
    high_deviation = quantile(pool, 1-q)     - quantile(validation, 1-q)

whichever is larger names the tail. `scripts/tail_selection/` implements and tests it.

**On the split as it stands, agreement with the oracle tail is 24/24 and it captures
100% of the available gain.** That number is not usable as reported, because this
split's suspicious pool is roughly 50% poisoned, and no defender faces that. Testing
against realistic poison fractions:

| tail quantile | pool 10% poisoned | pool 5% | pool 1% |
|---|---|---|---|
| 0.05 | 88% | 75% | **46%** |
| 0.02 | 92% | 83% | 54% |
| 0.01 | 88% | 88% | 71% |
| 0.005 | 83% | 79% | 75% |
| **0.002** | 88% | **92%** | **79%** |

At the default 5% tail quantile the rule **collapses to a coin flip when only 1% of
the pool is poisoned**, which is the realistic case. A rare contaminant does not move
a 5th percentile.

Tightening the tail largely rescues it: at q = 0.002 agreement is 79% to 92% across
all three fractions, and a fixed tight tail is a reasonable default since it costs
little at high contamination and is the only thing that works at low.

**So the honest position is:** the inversion is real and large, a label-free tail
choice recovers most of the gain when the tail quantile is matched to a rare
contaminant, and it fails outright with a loose tail. The two-sided numbers above
remain an upper bound; the deployable figure is roughly 80% of the checkpoints getting
the right tail at 1% contamination, not all of them.

A genuinely two-sided statistic (distance from the clean-validation median) needs no
tail choice at all and is still untested.

## Reproduce

`results/*/baseline_metrics.json`, `detectors.strip.q0.25`, comparing `auroc` against
`auroc_two_sided` and `direction`.

## Subquestions

1. Implement option 1 above and re-score. If tail selection from clean data alone
   recovers most of the gain, this becomes a deployable improvement to both detectors.
2. Does the fusion ([H14](H14-fusion.md)) improve once both components are two-sided?
   `badnet_a2a` currently contributes zeros to the fusion for exactly this reason.
3. Do feature-space detectors invert too, or is this specific to statistics computed
   in prediction space?
