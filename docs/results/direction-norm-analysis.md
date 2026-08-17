# Direction norm vs poison rate (H28 prediction 1)

**Status: PARTIALLY REFUTED.** The backdoor direction norm is NOT monotonically
increasing with poison rate at any layer. The norm is large at all tested rates
(0.5% to 10%) and shows no statistically significant trend. However, the
layer-wise growth pattern is extremely consistent: the backdoor direction
concentrates overwhelmingly in the later layers of the ViT block stack,
consistent with the residual stream amplification account.

## Setup

16 ViT-B/16 checkpoints: 2 attacks (badnet_a2o, blend) x 2 datasets (CIFAR-100,
Tiny) x 4 poison rates (0.5%, 1%, 5%, 10%). For each checkpoint, CLS-token
features were extracted at all 13 layers (embedding through final block output).
The backdoor direction was computed as the difference between mean poisoned and
mean clean CLS representations, and its L2 norm recorded.

## Prediction 1: direction norm scales with poison rate

**Not confirmed.** None of the 4 attack x dataset combinations show monotonic
growth with poison rate at the last layer:

| dataset | attack | 0.5% | 1% | 5% | 10% | Spearman rho | p |
|---|---|---|---|---|---|---|---|
| CIFAR-100 | badnet_a2o | 12.2 | 12.9 | 14.7 | 13.2 | +0.80 | 0.200 |
| CIFAR-100 | blend | 15.9 | 15.5 | 18.5 | 19.4 | +0.80 | 0.200 |
| Tiny | badnet_a2o | 15.3 | 12.0 | 15.2 | 14.5 | -0.40 | 0.600 |
| Tiny | blend | 18.9 | 18.8 | 19.0 | 18.2 | -0.40 | 0.600 |

CIFAR-100 shows a weak positive trend (rho = +0.80 for both attacks) but with
only n=4 data points, p=0.200 is nowhere near significance. Tiny shows no trend
at all (rho = -0.40 for both attacks). No combination is monotonic: every series
has at least one non-increasing step.

The interpretation: even at 0.5% poison rate, the model has enough capacity to
learn a simple trigger (badnet patch or blend overlay) and encode it as a strong
direction in the final layer. Increasing the poison rate from 0.5% to 10% does
not proportionally strengthen the direction, it just shifts where in the layer
stack the direction concentrates.

## Layer-wise growth pattern (unexpected finding)

The layer-wise growth is the stronger result. Across all 16 checkpoints, the
backdoor direction norm grows monotonically through the block stack:

| layer | mean norm | min | max |
|---|---|---|---|
| 0 (embedding) | 0.00 | 0.00 | 0.00 |
| 1 | 0.10 | 0.03 | 0.20 |
| 2 | 0.33 | 0.11 | 0.68 |
| 3 | 0.70 | 0.19 | 1.61 |
| 4 | 1.72 | 0.23 | 3.88 |
| 5 | 2.65 | 0.29 | 6.34 |
| 6 | 3.97 | 0.38 | 9.76 |
| 7 | 5.94 | 0.69 | 12.81 |
| 8 | 7.39 | 1.08 | 14.99 |
| 9 | 8.82 | 2.20 | 15.36 |
| 10 | 11.27 | 5.30 | 16.66 |
| 11 | 12.80 | 6.91 | 18.13 |
| 12 (last) | 15.90 | 12.02 | 19.42 |

The direction norm at layer 12 is 159x the norm at layer 1 on average. This
concentration in later layers is consistent with the CKA homogeneity finding
(Raghu et al.): ViT's residual stream carries information persistently, and
backdoor features accumulate through the block stack rather than appearing
suddenly at one layer.

The spread between min and max grows with depth: at layer 1 the range is
0.03 to 0.20 (7x), at layer 12 it is 12.0 to 19.4 (1.6x). The relative
variance shrinks even as the absolute values grow, meaning all 16 checkpoints
converge to a similar norm scale at the readout layer regardless of attack type,
dataset, or poison rate.

## Attack-specific patterns

Blend consistently has larger direction norms than badnet_a2o at every layer. At
layer 12, blend averages 18.1 vs badnet_a2o's 13.7. This tracks with blend's
stronger implantation: blend applies a global perturbation to every pixel, while
badnet uses a small localized patch. The global perturbation creates a larger
shift in the feature space.

The gap opens early: blend's mean norm at layer 4 is 3.1 vs badnet_a2o's 0.4.
By layer 6 blend has already reached 7.0 while badnet_a2o is at 0.8. This is
consistent with blend's trigger affecting more input tokens (all of them) vs
badnet's trigger affecting only the patch tokens it overlaps.

## Implications for PSBD

The fact that the backdoor direction norm is approximately constant across
poison rates (especially on Tiny, where blend norms range 18.2 to 19.0 across
0.5% to 10%) means PSBD's detection power should also be approximately constant
across poison rates, which is exactly what the detection tables show: AUROC for
blend on Tiny is 0.974 at 1%, 0.954 at 5%, 0.961 at 10% (token_mask @ before_
attention_norm). The margin between clean and poisoned predictions is set by the
direction norm, not by how many samples were poisoned.

This is good news for PSBD as a defense: even very low poison rates (0.5%)
create a direction norm of 12+ at the last layer, which is large enough for
perturbation-based detection to exploit.

## Data

Raw data: `results/direction_norm_analysis.json` (16 entries, 13 layers each).
Script: `scratch/direction_norm_analysis.py`.
