# H14 — PSBD and STRIP fuse into something better than either

**Status: SUPPORTED.** Rank-fusion by the more suspicious verdict recovers **93.1% of
the oracle-max** with no oracle, at mean TPR 0.614 at 1% FPR against 0.507 for STRIP
and 0.379 for PSBD.

## Claim

The comparison table showed the two detectors failing on opposite attacks, with no
checkpoint defeating both. If those failures are genuinely disjoint, a defender who
runs both and takes the more suspicious verdict should approach the performance of
always picking the right detector, without knowing in advance which that is.

## Method

Both scores become **percentiles of the same reference**, the clean validation split.
That reference choice is the whole method; see the correction below. Then two rules,
neither of which touches poisoned data:

- `mean` the average of the two percentiles
- `min` the more suspicious of the two, so a sample escapes only if **both**
  detectors consider it clean

Threshold remains the quantile of clean-validation fused score, so the false-positive
budget is set exactly as before.

## Evidence

ViT-B/16, CIFAR-10, 23 backdoored checkpoints, TPR at 1% FPR:

| detector | @1% FPR | @5% FPR |
|---|---|---|
| PSBD (fixed placement, adaptive rate) | 0.379 | 0.538 |
| STRIP | 0.507 | 0.600 |
| `mean` fusion | 0.467 | 0.612 |
| **`min` fusion** | **0.614** | **0.758** |
| *oracle-max (pick the better per checkpoint)* | *0.660* | -- |

Benign control: 0.013 at 1% FPR, 0.042 at 5%. The fusion invents nothing.

## Why it works, and it is not what "beats both" would suggest

`min` beats **both** components on only 6 of 23 checkpoints. Its advantage is not that
it exceeds the better detector; it is that it **never collapses**. Median shortfall
against the oracle-max is **-0.017**, i.e. it lands within two points of whichever
detector happened to be right.

The single-detector columns each contain catastrophic failures that the other covers:

| checkpoint | PSBD | STRIP | min |
|---|---|---|---|
| `badnet_a2o` 0.5% | 0.001 | 0.979 | **0.956** |
| `badnet_a2o` 10% | 0.052 | 1.000 | **1.000** |
| `blend` 10% | 0.974 | 0.487 | **0.954** |
| `bpp` 1% | 0.917 | 0.113 | **0.867** |
| `lc` 10% | 0.100 | 0.897 | **0.880** |

In each row one detector is near-useless and the fusion tracks the other. That is the
practical value: a defender does not know the attack, so a detector that is excellent
half the time and useless the other half is not deployable, and this one is neither.

`mean` fusion is much worse (0.467) precisely because averaging lets a confidently
wrong detector drag down a confidently right one. `min` is the correct rule when the
failures are one-sided, which they are: a detector that misses a trigger reports it as
clean, it does not report it as strongly poisoned.

## Where the fusion still fails

- `badnet_a2a` (0.000 to 0.001): both components fail, so there is nothing to fuse.
  Its signal is inverted, and neither the fusion nor its components apply the
  two-sided rule ([H5](H5-all-to-all-breaks-psbd.md)). Adding it should recover this.
- `wanet` (0.015 against PSBD's 0.406): the largest shortfall in the table, -0.391.
  STRIP scores 0.003, and `min` follows the wrong one because a percentile of 0.003
  is more extreme than PSBD's. This is the failure mode of `min`: it trusts the more
  extreme verdict even when that verdict is wrong.
- `adaptive_blend` -0.163 and `blend` 0.5% -0.155 for the same reason.

## The bug this took to get right

The first implementation ranked each split **against itself**. Within-split ranks span
[0, 1] for every split by construction, so a threshold at the 1st percentile of
validation rank flags exactly the bottom 1% of the backdoor split regardless of how
extreme its scores are, pinning TPR to the FPR. It produced TPR 0.010 at 1% FPR on
every checkpoint, including ones where a component detector scored 1.000.

Ranking against a shared reference is what makes an extreme score stay extreme.

## Reproduce

```bash
PYTHONPATH=. python detector_fusion.py --fpr 0.01 0.05
```

## Subquestions

1. **Add the two-sided rule to the fusion.** `badnet_a2a` is 3 of the 23 rows and
   contributes zeros; H5 shows it is detectable at 0.966 with the sign flipped.
2. `min` follows the more extreme verdict, which is wrong when a detector fails
   confidently (`wanet`). A rule weighted by each detector's *validation-measured*
   reliability would need no poisoned data and should fix that.
3. Both components here are prediction-space. A feature-space detector would likely
   be more complementary still, and would test whether the gain is about detector
   family rather than about these two methods.
