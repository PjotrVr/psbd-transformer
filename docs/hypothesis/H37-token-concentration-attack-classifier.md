# H37 -- Token concentration ratio as attack family classifier

**Status: REFUTED.** Concentration ratio does not cleanly separate localized
from global attacks. 62.5% accuracy at the best threshold, with a negative
separation gap (-0.98). adaptive_blend and lf have high concentration despite
being global attacks, breaking the assumed correspondence between trigger
locality and direction concentration.

Evidence: `experiments/token_structure/token_concentration_classifier.py`, results in
`results/token_concentration_classifier.json`. Uses H32's token localization
data (`results/token_localization.json`).

## Claim

The concentration ratio from H32 (max patch direction norm / mean patch
direction norm) cleanly separates localized triggers (ratio > 2.5) from global
triggers (ratio < 2.5). This can be computed at inference time under
perturbation to classify the attack family without knowing the attack.

## Test

1. Label each attack as localized (badnet_a2o, tact, badnet_a2a) or global
   (blend, wanet, adaptive_blend, sig, lf, bpp, lc).
2. Apply a concentration threshold (2.5) to classify each checkpoint's
   concentration ratio.
3. Measure accuracy and the separation gap (min localized minus max global).

## Results

| checkpoint | conc ratio | true family | predicted | correct |
|---|---:|---|---|---|
| cifar100 badnet_a2o | 2.35 | localized | global | no |
| cifar100 blend | 1.39 | global | global | yes |
| cifar100 wanet | 1.64 | global | global | yes |
| cifar100 adaptive_blend | 3.10 | global | localized | no |
| cifar100 sig | 2.22 | global | global | yes |
| cifar100 lf | 3.33 | global | localized | no |
| tiny badnet_a2o | 4.23 | localized | localized | yes |
| tiny blend | 1.75 | global | global | yes |

Accuracy: 5/8 = 62.5%.

Localized attacks: mean concentration 3.29 (range 2.35 to 4.23).
Global attacks: mean concentration 2.24 (range 1.39 to 3.33).
Separation gap: -0.98 (negative means distributions overlap).
Optimal threshold: 2.84 (still cannot separate cleanly).

## Interpretation

### Why the classifier fails

Two global attacks have unexpectedly high concentration ratios:

1. **adaptive_blend (3.10):** Adaptive blend uses a blending trigger with a
   learned, spatially non-uniform opacity mask. Although the trigger is
   technically global (applied to the full image), the learned mask concentrates
   its effect in specific regions, producing a concentrated direction signature
   indistinguishable from a local trigger.

2. **lf (3.33):** Low-frequency perturbation modifies specific frequency
   components. In the spatial token decomposition, these frequency changes
   project onto a small number of token positions (those where the frequency
   pattern has maximum amplitude), producing concentrated norms.

Meanwhile, badnet_a2o on CIFAR-100 has a lower-than-expected concentration
(2.35) because the 3x3 trigger at the corner occupies a fraction of a single
14x14 ViT patch, diluting its norm relative to the full patch.

### Practical implication

The localized/global distinction is not a clean binary in direction space.
Some "global" attacks produce concentrated directions, and some "localized"
attacks produce diffuse directions depending on the relationship between trigger
size and patch size. The concentration ratio from H32 is informative for
visualizing WHERE the backdoor acts but not reliable as an automated classifier.
