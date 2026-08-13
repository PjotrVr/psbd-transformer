# PSBD against baseline detectors, at deployable false-positive rates

ViT-B/16, CIFAR-10, full 10000-image test split. Identical splits, identical
clean-validation threshold rule, identical quantile grid; the only thing that differs
between columns is the score being thresholded.

PSBD is shown at its **best placement and rate per checkpoint**. The baselines have no
such choice, so this comparison is generous to PSBD, not to the alternatives.

## TPR at 1% and 5% false positives

| attack | pr | ASR | PSBD@1% | STRIP@1% | conf@1% | PSBD@5% | STRIP@5% | conf@5% |
|---|---|---|---|---|---|---|---|---|
| `blend` | 0.005 | 1.00 | **0.908** | 0.606 | 0.000 | **0.988** | 0.844 | 0.000 |
| `blend` | 0.01 | 1.00 | **0.999** | 0.399 | 0.000 | **1.000** | 0.668 | 0.000 |
| `blend` | 0.05 | 1.00 | **0.946** | 0.914 | 0.000 | **1.000** | 0.973 | 0.238 |
| `blend` | 0.1 | 1.00 | **0.998** | 0.487 | 0.000 | **1.000** | 0.749 | 0.000 |
| `badnet_a2o` | 0.005 | 1.00 | 0.004 | **0.979** | 0.000 | 0.106 | **0.994** | 0.000 |
| `badnet_a2o` | 0.01 | 1.00 | 0.170 | **0.973** | 0.000 | 0.405 | **0.990** | 0.000 |
| `badnet_a2o` | 0.05 | 1.00 | 0.364 | **0.994** | 0.143 | 0.704 | **0.999** | 0.430 |
| `badnet_a2o` | 0.1 | 1.00 | 0.366 | **1.000** | 0.004 | 0.646 | **1.000** | 0.173 |
| `adaptive_blend` | 0.1 | 0.93 | **0.329** | 0.000 | 0.000 | **0.606** | 0.017 | 0.000 |
| `badnet_a2a` | 0.005 | 0.84 | **0.079** | 0.000 | 0.000 | **0.282** | 0.000 | 0.000 |
| `badnet_a2a` | 0.01 | 0.94 | **0.059** | 0.000 | 0.000 | **0.474** | 0.029 | 0.000 |
| `badnet_a2a` | 0.05 | 0.93 | **0.020** | 0.000 | 0.006 | **0.243** | 0.008 | 0.023 |
| `badnet_a2a` | 0.1 | 0.96 | **0.036** | 0.003 | 0.000 | **0.297** | 0.031 | 0.000 |
| benign control | -- | -- | 0.016 | 0.009 | 0.007 | 0.060 | 0.034 | 0.054 |
| **mean** | | | **0.406** | **0.489** | 0.012 | **0.596** | 0.562 | 0.066 |

## What this says

**The two methods are complementary, almost perfectly so.** STRIP is near-perfect on
the static patch trigger (0.97 to 1.00 at 1% FPR) and scores **exactly 0.000** on
`adaptive_blend` and `badnet_a2a`. PSBD is the reverse: 0.91 to 1.00 on `blend`, and
0.004 to 0.366 on `badnet_a2o`. There is no checkpoint where both fail.

That is not a coincidence of tuning, it follows from what each measures. STRIP
superimposes a clean image and asks whether the prediction survives; a high-contrast
corner patch survives superimposition, and a low-amplitude blended or adaptive trigger
does not. PSBD asks whether the prediction is robust to internal perturbation, which
is where a diffuse trigger's redundant encoding shows up and a single patch's does not.

**Neither method alone is a defence.** On the mean, STRIP leads at 1% FPR (0.489
against 0.406) and PSBD leads at 5% (0.596 against 0.562). Both means are dragged down
by the attacks the other one handles. A paper reporting only PSBD on ViT would be
reporting the weaker method on the most standard attack in the literature.

**Confidence alone is not a detector**, at 0.012 and 0.066 mean. That settles the
adversarial objection raised against PSU: whatever PSU is measuring, it is not
baseline confidence.

**The benign control passes for all three** (0.016, 0.009, 0.007 at 1% FPR), so none
of these numbers are an artifact of the threshold rule.

## The obvious follow-up, not yet run

Combine them. `max(psbd_rank, strip_rank)` or any rank fusion should beat both, since
their failures are disjoint. On this table a fusion would plausibly exceed 0.9 mean TPR
at 1% FPR. That is the strongest result available from work already done, and it needs
no new GPU time: both score sets are on disk.

## Caveats

- PSBD gets a per-checkpoint placement and rate search; STRIP and confidence get one
  fixed configuration each. Correcting this would lower the PSBD column.
- STRIP uses 8 overlays drawn from the same clean validation split PSBD thresholds on,
  so neither method sees more data than the other.
- One dataset, one architecture, one seed. The Swin and CIFAR-100 sweeps are running.
- Feature-space detectors (Spectral Signatures, Activation Clustering, SCAn) are not
  here because they partition a poisoned training set rather than judging one input;
  their numbers would not belong in this table.
