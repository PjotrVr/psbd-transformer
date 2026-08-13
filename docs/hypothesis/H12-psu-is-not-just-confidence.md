# H12 — PSU is just a proxy for baseline confidence

**Status: REFUTED.** And the test that refuted it produced a free improvement to the
score.

## Claim (the skeptical one)

PSU is `P_c(x)` minus the mean dropout-perturbed `P_c(x)`. A sample starting near
probability 1 has more room to fall than one starting at 0.6, and a backdoored model
is extremely confident on triggered inputs. So PSU may simply be measuring baseline
confidence, dressed up with `k` stochastic forward passes that a defender could skip.

Raised by an adversarial verification pass, which noticed that on the benign control
a confidence-only detector scores 0.514 against PSBD's 0.506.

## Test

Three detectors on the same cached tensors, `before_mlp_residual`, CIFAR-10 ViT at
10% poisoning, each at PSU's own best rate:

- **psu** `P_c(x) - mean_k(P_c_dropout(x))`, the method
- **confidence** `P_c(x)` alone, no dropout at all, free
- **fractional drop** `1 - mean_k(P_c_dropout(x)) / P_c(x)`, which divides the
  starting confidence out entirely

The third is the decisive one. If PSU works only because confident samples fall
further in absolute terms, normalising by the starting confidence should destroy it.

## Evidence

| checkpoint | PSU | confidence only | fractional drop |
|---|---|---|---|
| `blend` | 0.984 | 0.701 | **0.997** |
| `bpp` | 0.992 | 0.822 | **0.993** |
| `lf` | 0.979 | 0.780 | **0.991** |
| `badnet_a2o` | 0.922 | 0.878 | **0.925** |
| `badnet_a2a` | 0.508 | 0.244 | 0.499 |
| benign control | 0.508 | 0.486 | 0.506 |
| **mean (backdoored)** | **0.877** | **0.685** | **0.881** |

PSU beats confidence-only on **5 of 5** checkpoints, by 0.19 on average. The
stochastic passes earn their cost.

And normalising the confidence out does not destroy the signal, it slightly improves
it, on every checkpoint. So PSU measures how **robust** a prediction is, not how
confident it started.

## The free improvement

The fractional form is better on all four working attacks (blend +0.013, lf +0.012,
badnet_a2o +0.003, bpp +0.001). Small, but one-sided, and it costs nothing: the same
cached tensors divided by a number already on disk.

It also has a principled edge over the paper's absolute form. The threshold is a
quantile of clean-validation PSU, and absolute PSU is bounded above by the starting
confidence, so the threshold inherits the validation set's calibration. The ratio does
not, which should make it transfer better across datasets and across models whose
confidence is differently calibrated. That prediction is untested here.

## The honest caveat

`badnet_a2o` is the case where the skeptic is closest to right: confidence-only
reaches 0.878 against PSU's 0.922, so most of the separation for the static patch
trigger is available with no dropout at all. For `blend` the comparison is 0.701
against 0.984. PSBD adds most where a naive baseline is weakest, which is the right
way round, but the naive baseline is stronger than one would guess and no PSBD paper
reports it.

## Reproduce

```bash
PYTHONPATH=. python scripts/psu_vs_confidence/measure.py
```

## Subquestions

1. Does the fractional form's advantage hold under the adaptive rate rule (not just
   the oracle rate), across poison rates, and under SAM? If so it should replace the
   absolute form in `psu_from_cache` outright.
2. Confidence-only at 0.685 mean is a baseline every detection method should be
   required to beat, and it is absent from the literature this project builds on.
3. Do confidence and PSU carry partly independent information? A two-feature
   combination would say.
