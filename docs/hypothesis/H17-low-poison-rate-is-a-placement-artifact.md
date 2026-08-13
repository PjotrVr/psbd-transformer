# H17 — PSBD's low-poison-rate failure is a placement artifact, not a limit of the method

**Status: SUPPORTED on CIFAR-10 (derivation set).** The published configuration is
anti-correlated at 1% poisoning; a different placement on the same checkpoints,
same one-sided rule, same defender-legal rate rule, reaches **0.936** mean AUROC at
1%. Out-of-sample test on GTSRB and Tiny is pre-registered and in flight
([2026-08-14-h16-out-of-sample](../runs/2026-08-14-h16-out-of-sample.md)).

## Claim

The received reading of the 1% results is that PSBD's premise fails when poisoning
is rare: too few poisoned samples, so the shortcut is too weak to detect. That
reading is wrong on this grid. The premise holds at 1%. What fails is the
**ConvNet placement the paper inherited**, which is anti-correlated at *every*
dropout rate at 1%, combined with a rate rule that targets a disturbance level
past the point where the backdoor circuit collapses.

Nothing here flips a sign. Every number is one-sided, low PSU = poisoned, exactly
as the mechanism states.

## Evidence 1: the failure is not uniform across placements

Full surface, `vit_cifar10_badnet_a2o`, 14 placements x 9-16 dropout rates, ASR
0.997 at 1% so the attack is fully implanted. One-sided AUROC:

| poison rate | cells above chance | best cell | published rule picks |
|---|---|---|---|
| 1% | 46/133 | **0.825** (`before_attention_norm`, p=0.5) | 0.297 (`post_residual`, p=0.09) |
| 10% | 131/133 | 0.952 (`before_attention`, p=0.7) | 0.656 (`post_residual`, p=0.09) |

At 1% the surface is mostly below chance, but its maximum is 0.825. A method
whose premise had failed would have no such cell. `post_residual`, the placement
the paper uses, is below 0.5 at **all 16 of its cached rates** at 1%:

    post_residual @ 1%   0.005:0.45 0.01:0.44 0.02:0.43 0.03:0.38 0.05:0.22
                         0.07:0.18 0.09:0.30 0.1:0.35 0.2:0.50 0.3:0.47 ...

## Evidence 2: there is a critical dropout rate, and it moves with poison rate

The same placement at the two poison rates, one-sided AUROC against p:

    before_attention_norm @ 1%    0.1:0.55 0.2:0.52 0.3:0.57 0.4:0.70 0.5:0.83
                                  0.6:0.82 0.7:0.61 0.8:0.32 0.9:0.24
    before_attention_norm @ 10%   0.1:0.60 0.2:0.66 0.3:0.73 0.4:0.78 0.5:0.80
                                  0.6:0.82 0.7:0.84 0.8:0.87 0.9:0.94

At 10% the curve is monotone increasing and never inverts: the backdoor survives
even 90% dropout. At 1% it peaks at p = 0.5 and then **crosses through 0.5 into
inversion**. Read mechanistically, there is a critical rate p\* at which the
backdoor circuit stops surviving the perturbation; past p\*, the poisoned samples
are disturbed *more* than clean ones and the statistic runs backwards. p\* rises
with poison rate, which is what a redundancy argument predicts: more poisoned
training samples produce a more redundantly encoded shortcut, which survives more
dropout.

The paper's rate rule targets clean-validation shift ratio sigma >= 0.8. At 1%
that lands at p = 0.8-0.9, i.e. **past p\***. The rule does not merely lose a
little AUROC at low poison rate ([H11](H11-adaptive-rate-overshoots.md)); it
selects into the inverted region.

## Evidence 3: a defender-legal configuration recovers it

Configuration = (placement, score, sigma target). The rate is chosen by the
paper's own rule shape, smallest rate whose **clean-validation** sigma reaches the
target. No poison label is read.

Mean one-sided AUROC, all-to-all excluded (a separate known failure,
[H5](H5-all-to-all-breaks-psbd.md)):

| configuration | 1% | 5% | 10% | benign | below chance |
|---|---|---|---|---|---|
| `post_residual` / absolute / 0.8 **(published)** | 0.781 | 0.864 | 0.830 | 0.504 | 1/17 |
| `before_attention_norm` / fractional / 0.7 | **0.936** | 0.935 | 0.832 | 0.494 | 1/17 |
| `before_attention` / fractional / 0.6 | 0.914 | 0.966 | 0.967 | 0.486 | **0/12** |

Per checkpoint, published against `before_attention_norm` / fractional / 0.7:

| checkpoint | published | candidate | delta |
|---|---|---|---|
| `badnet_a2o` 1% | **0.297** | **0.839** | **+0.543** |
| `badnet_a2o` 5% | 0.582 | 0.902 | +0.320 |
| `badnet_a2o` 10% | 0.656 | 0.840 | +0.184 |
| `adaptive_blend` 10% | 0.775 | 0.897 | +0.122 |
| `blend` 1% / 5% / 10% | 0.956 / 0.956 / 0.953 | 0.996 / 0.989 / 0.999 | +0.04 |
| `bpp`, `lf`, `sig` | 0.90-0.99 | 0.91-0.995 | +0.00 to +0.03 |
| `wanet` 10% | 0.941 | 0.744 | **-0.196** |
| `lc` 10% | 0.515 | 0.331 | **-0.184** |
| benign control | 0.504 | 0.494 | -0.010 |

Mean +0.058 over 17 backdoored, wins 14/17. The gain is concentrated exactly
where the problem was: the static patch trigger, worst at the lowest poison rate.

**The single headline number: `badnet_a2o` at 1% poisoning, ASR 0.997, goes from
0.297 to 0.839 without flipping anything.**

## What this rules out

The "too few poisoned samples to learn a detectable shortcut" explanation does not
survive. On this grid at 1%:

- ASR is 0.997 (`badnet_a2o`), 1.000 (`blend`), 0.977 (`bpp`), 0.983 (`lf`). The
  attacks implanted fully. A failed attack cannot explain a failed detector here.
- `blend`, `bpp`, `lf` at 1% already scored 0.93-0.99 under the *published*
  configuration. Only the patch trigger failed, so whatever broke is
  trigger-specific, not poison-rate-generic.

Published ViT-B/16 results elsewhere report much weaker attacks at 1% (BadNet 39.5%
ASR on CIFAR-100 in the backdoor-directions work). Those are different checkpoints
under a different recipe; on this repo's checkpoints the attack is not the limiting
factor at 1%, and the low-rate argument built on weak implantation does not
transfer here. It may still bind at 0.5%, which is outside this file's scope.

## Negative result: ensembling placements does not fix it

No single placement wins on every attack (`before_attention_norm` wins the patch
trigger and loses `wanet` and `lc`), so rank-averaging PSU across placements looks
like the obvious defender-legal answer. It is not:

| checkpoint | best single | 4-placement rank ensemble |
|---|---|---|
| `badnet_a2o` 1% | 0.839 | **0.475** |
| `lc` 10% | 0.894 | 0.696 |
| `wanet` 10% | 0.970 | 0.940 |

Mean 0.900 for the ensemble against 0.892 for a fixed `before_attention_norm`, so
it is a wash overall, and it **destroys the case this file is about**. The reason
is structural: at 1% three of the four members are inverted or near chance
(`post_residual` 0.194), and rank-averaging an inverted member actively poisons the
pool. Ensembling assumes members are at worst uninformative; here they are
anti-correlated.

So the fix is placement *selection*, not placement *aggregation*.

## Why `before_attention_norm`

It is the pre-hook on `ln_1`, i.e. the perturbation enters immediately before the
attention block reads the residual stream, and the residual path itself is left
intact. `post_residual` perturbs the summed stream, which the next block's skip
connection largely carries through, so the disturbance it produces at a given
sigma is doing something different from what the sigma suggests.

This is consistent with [H4](H4-placement-is-attack-dependent.md) (best placement tracks
where the backdoor direction enters `[CLS]`) and with
[H1](H1-pre-beats-post.md) having already refuted pre-residual as a general
winner. It is **not** consistent with `pre_residual_blocks_5_8`, the band
[H10](H10-depth-band-placement.md) selected, which reaches only 0.360 on `badnet_a2o` at 1%.
H10's band was derived at 5-10% poisoning and does not transfer down.

## The caveat that has to travel with this

The placement and sigma target were chosen by ranking configurations on this
CIFAR-10 grid. That is fitting on the test set, and the numbers above are
therefore optimistic. The claim is not established until it is tested on
checkpoints that played no part in choosing it, which is what the pre-registration
covers.

## Reproduce

    PYTHONPATH=. .venv/bin/python scratch/build_surface.py <folders>   # 2591 cells
    PYTHONPATH=. .venv/bin/python scratch/analyze_surface.py
    PYTHONPATH=. .venv/bin/python scratch/head_to_head.py
    PYTHONPATH=. .venv/bin/python scratch/ensemble.py

## Subquestions

1. Does the critical rate p\* track poison rate monotonically across attacks, and
   can it be estimated from clean data alone? If p\* is observable without labels,
   the rate rule can adapt instead of using a fixed sigma target.
2. Is p\* predicted by how many units carry the backdoor? A direct ablation count
   (`analysis/direction.py`, `analysis/features.py`) would turn the redundancy
   argument from an interpretation into a measurement.
3. `k = 3` Monte Carlo passes is the paper's value and this repo's. PSU is an
   expectation estimated from 3 samples; at low poison rate the true gap is small,
   so estimator variance may dominate. Sweeping k in (3, 10, 20, 50) needs a GPU
   re-run but is cheap.
4. The cache stores only the tracked-class probability and the argmax per pass, not
   the full predictive distribution, so BALD-style mutual information (entropy of
   the mean minus mean of the entropies) cannot be computed from it. It is strictly
   more information than PSU and is the natural next score to try; it needs the
   sweep to store full probabilities.
5. Per-class score normalisation is computable from the existing cache, since the
   baseline argmax class is stored. At 1% the target class should dominate the
   false positives; if it does, normalising within predicted class is free.
