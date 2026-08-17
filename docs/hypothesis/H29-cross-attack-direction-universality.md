# H29 — Cross-attack backdoor directions are NOT universal

**Status: REFUTED.** Off-diagonal cosine similarity is near zero across all
settings. Each attack creates its own direction.

Evidence: `scratch/direction_universality.py`, results in
`results/direction_universality.json`.

## Claim

Different attacks targeting the same class produce parallel backdoor directions
at layer 12. The direction is intrinsic to the target class geometry (how the
model separates "class 0 vs everything else"), not to the trigger. Predicted
cosine > 0.8 across attack pairs.

## Why it mattered

If confirmed, a defender who knows ONE attack's direction could detect ALL
attacks targeting that class, and it would ground the margin estimation account:
the direction IS the margin direction to the target class decision boundary.

## Evidence

Pairwise cosine similarity of backdoor directions at layer 12, across all 10
attacks. All checkpoints target label 0.

### CIFAR-100 at 10%

| | badnet_a2o | blend | wanet | lc | adaptive_blend | sig | lf | bpp | tact | badnet_a2a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | 1.000 | 0.042 | 0.030 | 0.162 | -0.010 | 0.046 | 0.012 | 0.094 | 0.109 | 0.013 |
| blend | 0.042 | 1.000 | 0.065 | 0.077 | 0.096 | 0.047 | 0.083 | 0.167 | 0.028 | -0.009 |
| wanet | 0.030 | 0.065 | 1.000 | 0.022 | -0.055 | 0.026 | 0.041 | -0.027 | 0.041 | 0.018 |

Off-diagonal: mean=0.053, min=-0.066, max=0.374

### CIFAR-100 at 5%

Off-diagonal: mean=0.047, min=-0.081, max=0.190

### CIFAR-100 at 1%

Off-diagonal: mean=0.041, min=-0.095, max=0.186

### Tiny ImageNet at 10%

Off-diagonal: mean=0.023, min=-0.083, max=0.192

## Interpretation

The mean off-diagonal cosine is 0.023 to 0.053, indistinguishable from random
768-dimensional vectors (expected cosine ~ 0). Each attack learns a
trigger-specific direction, not the target class's margin direction. The highest
pairwise cosine (0.374 for lc vs tact on CIFAR-100 at 10%) is well below the
0.8 threshold.

This has two consequences:

1. **No universal detection from a single known direction.** A defender must
   detect each attack's direction independently. The direction ablation defense
   (H16, H34) requires knowledge of the specific attack, not just the target
   class.

2. **The margin estimation account needs revision.** The direction is NOT the
   class margin direction. Since each attack's direction aligns strongly with the
   readout weight (cosine 0.58 to 0.88 per H16/H28) but NOT with each other,
   each attack finds a different path toward the same readout weight. The 768-dim
   space has room for many near-orthogonal vectors that all project heavily onto
   the readout weight.

## Direction norms by attack (CIFAR-100 at 10%)

| attack | direction norm |
|---|---:|
| lf | 22.47 |
| bpp | 20.43 |
| blend | 19.42 |
| wanet | 14.64 |
| badnet_a2o | 13.15 |
| adaptive_blend | 10.90 |
| lc | 9.69 |
| sig | 4.77 |
| tact | 3.71 |
| badnet_a2a | 3.69 |

Norm varies by 6x across attacks, with global triggers (blend, lf, bpp)
producing stronger directions than localized ones (tact, sig, badnet_a2a).
