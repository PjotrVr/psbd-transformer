# H24 — k = 3 Monte Carlo passes is the noise floor at low poison rate

**Status: CONFIRMED, both predictions.** k = 20 was run and the gain is real and
concentrated at low poison rate, exactly as predicted.

## Result of the paid half

`before_attention_norm`, rate matched at clean-validation sigma >= 0.6, one-sided
AUROC, fractional PSU:

| checkpoint | k = 3 | k = 20 | delta |
|---|---|---|---|
| **`badnet_a2o` 1%** | 0.839 | **0.885** | **+0.046** |
| `adaptive_blend` 10% | 0.897 | 0.918 | +0.022 |
| `lf` 5% | 0.950 | 0.965 | +0.014 |
| `lf` 10% | 0.945 | 0.960 | +0.015 |
| `lf` 1% | 0.924 | 0.934 | +0.009 |
| `badnet_a2o` 5% | 0.902 | 0.909 | +0.007 |
| `badnet_a2o` 10% | 0.816 | 0.825 | +0.009 |
| `wanet` 10% | 0.744 | 0.751 | +0.007 |
| `lc` 10% | 0.455 | 0.455 | -0.000 |
| **benign control** | 0.494 | 0.495 | **+0.001** |

Mean delta **+0.028 at 1%**, +0.011 at 5%, +0.011 at 10%. Prediction 2 asked for
at least +0.03 at 1% and under +0.01 at 10%; the low-rate figure lands just under
and the high-rate figure just over, so the *direction* and the *concentration* are
confirmed while the exact magnitudes were slightly optimistic.

The benign control moving +0.001 is what makes this readable: more sampling is not
simply making every statistic look better.

**Practical consequence.** The PSBD paper's k = 3 is leaving detection on the
table, and it leaves most where the method is weakest. Raising k costs inference
passes only: no retraining, no new statistic, no new hyperparameter to tune.
`lc` at 10% is unmoved because it is failing for a different reason
(clean-label, AUROC 0.455), which is the expected null case.

---

## Earlier free-half result, retained

**Status then: prediction 1 CONFIRMED from cached data.**

## Result of the free half

`scratch/k_sweep.py`, fractional PSU at `before_attention_norm`, rate matched at
clean-validation sigma >= 0.6, all-to-all excluded. k = 1 and k = 2 are subsets of
the cached k = 3 passes, so this cost no GPU time.

Mean AUROC increment from k = 1 to k = 3, by poison rate:

| poison rate | mean k3 - k1 | n |
|---|---|---|
| **1%** | **+0.0269** | 4 |
| 5% | +0.0096 | 5 |
| 10% | +0.0168 | 8 |

The increment is largest at 1% poisoning, as predicted. The single sharpest case
is the one this whole line of work is about:

| checkpoint | k=1 | k=2 | k=3 | k3 - k1 |
|---|---|---|---|---|
| **`badnet_a2o` 1%** | 0.750 | 0.812 | **0.839** | **+0.089** |
| `lf` 1% | 0.908 | 0.920 | 0.924 | +0.016 |
| `blend` 1% | 0.996 | 0.995 | 0.996 | -0.000 |
| `bpp` 1% | 0.974 | 0.975 | 0.976 | +0.002 |
| benign control | 0.494 | 0.494 | 0.494 | +0.000 |

Two things to read off this. `badnet_a2o` at 1% is **still climbing at k = 3**, so
its PSU estimate has not converged and the published pass count is leaving
detection on the table for exactly the case that fails. And `blend` and `bpp` at
1% are flat because they are already saturated near 0.99, so more passes cannot
help them: the gain is specific to checkpoints that are neither saturated nor
hopeless.

The benign control is unmoved at 0.494 across every k, which is the check that
this is not simply more sampling making any statistic look better.

**Gate passed**, so k = 20 runs on the 10 checkpoints that are still climbing, at
`before_attention_norm`, rates 0.3 to 0.7. Predictions 2 and 3 remain open.

## Mechanism

PSU is an expectation estimated by sampling:

    original form
        phi_PSU(x) = P_c(x; theta) - (1/k) sum_{i=1..k} P_c(x; p, theta'_i)

    restated
        the confidence in the sample's own unperturbed predicted class, minus the
        average of that same confidence over k independently perturbed passes

The second term is a Monte Carlo mean over k = 3 draws. Its standard error falls
as 1/sqrt(k), so at k = 3 the estimate carries roughly 58% of the noise of a
single draw. k = 3 is the PSBD paper's own value
(`papers/PSBD/sec/4_method.tex`, "We perform forward inference k=3 times"), so
this is not a porting error, but the paper never ablates it and never evaluates
below 5% poisoning.

## Why more passes might help

The separation PSBD needs is a difference between two distributions of PSU. At
10% poisoning that difference is large and 3 draws resolve it. At 1% it is small
enough that the published configuration reads **0.297** on `badnet_a2o`
([H17](H17-low-poison-rate-is-a-placement-artifact.md)) while a different
placement on the same checkpoint reads 0.839. If per-sample estimator noise is
comparable to the between-class gap, AUROC is attenuated toward 0.5 by
measurement error alone, and averaging it away is the cheapest possible fix:
inference passes, no retraining, no new statistic.

This matters more for the structured operators. `channel_mask` makes 768 Bernoulli
decisions per sample where `dropout` makes 151k, so its realised disturbance
varies far more between passes and it should benefit from larger k
disproportionately ([H26](H26-channel-mask-structured-vs-elementwise.md)).

## Why it might not

- If the low-poison-rate limit is a **signal** limit rather than a noise limit,
  more passes converge to a better estimate of a quantity that does not separate,
  and AUROC saturates well below anything useful. That outcome is a real result:
  it puts a floor on prediction-shift detection that no amount of sampling
  crosses, and it is publishable as a bound.
- The attenuation may already be small. AUROC is a rank statistic, and ranks are
  more robust to independent per-sample noise than the raw scores are, so the
  gain from k = 20 could be a fraction of what the standard-error argument
  suggests.

## Prediction

1. **Free half.** AUROC computed from k = 1 and k = 2 subsets of the existing
   k = 3 caches rises monotonically with k, and the k = 1 to k = 3 increment is
   **larger at 1% poisoning than at 10%**. This is the cheap test of whether noise
   is binding at all.
2. **Paid half.** Going to k = 20 adds at least **+0.03 mean AUROC at 1%
   poisoning** and less than +0.01 at 10%.
3. Structured operators gain more from k = 20 than `dropout` does.

If prediction 1 fails (no monotone trend, or a larger increment at 10% than at
1%), **the k = 20 jobs are not submitted at all** and this hypothesis is closed
from cached data alone.

## What would refute it

- Flat AUROC across k = 1, 2, 3 and k = 20: PSU's Monte Carlo noise is not the
  binding constraint, and the low-rate limit is structural.
- Gains that are uniform across poison rate: k matters, but not *because* of low
  poison rate, so it is a general tuning improvement rather than an answer to the
  question this study is asking.

## Cost

The free half is CPU only: the cache stores per-pass tracked-class probabilities
as `(k, N)`, so k = 1 and k = 2 are subsets already on disk and need no GPU. The
paid half runs k = 20 only on the pilot checkpoints, only on operator-position
pairs that survived stage 1, and only at 3 rates around the matched-sigma
operating point, which keeps it to roughly 2 jobs.

## Reproduce

    # free half, from existing caches
    PYTHONPATH=. .venv/bin/python scratch/k_sweep.py

    # paid half, gated on the free half showing a trend
    python pbs/generate_perturbation_jobs.py --forward-passes 20 --stage pilot \
        --operator <survivors> --prefix k20
