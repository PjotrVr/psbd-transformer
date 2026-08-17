# H33 — Weight-space spectral signature of the backdoor

**Status: REFUTED.** The weight difference between a backdoored and benign model
is NOT low-rank in encoder layers. The backdoor perturbation is distributed
across many dimensions.

Evidence: `scratch/weight_spectral_signature.py`, results in
`results/weight_spectral_signature.json`.

## Claim

The SVD of the weight difference (backdoor minus benign) should be low-rank,
with the top singular value capturing > 50% of variance for the last 3 to 4
layers. The top singular vector should align with the backdoor direction.

## What actually happened

Across 40 (dataset, attack, rate) combinations:

| weight matrix | top-1 variance fraction | consistent? |
|---|---:|---|
| class_token | 1.000 | 40/40 (trivially rank-1, it is a 1-D vector) |
| pos_embedding | 0.10 to 0.15 | 40/40 |
| encoder layer 0 MLP fc2 | 0.10 to 0.14 | 30/40 |
| encoder layer 11 MLP fc2 | 0.018 | 0/40 |
| encoder layer 11 out_proj | 0.020 | 0/40 |
| encoder layer 11 in_proj | 0.054 | 0/40 |
| classifier head | 0.023 to 0.027 | 0/40 |

The encoder weight matrices that carry the backdoor signal (layers 8 to 12
per H30) show top-1 concentration of only 2 to 5%, meaning the weight change
is spread across many singular vectors. The classifier head, despite being the
readout layer, shows only 2.3 to 2.7% concentration.

## Interpretation

The backdoor direction in activation space is rank-1 (H16), but the weight
perturbation that produces it is NOT rank-1. This makes sense: the direction is
an emergent property of the full forward pass through many layers, not a single
weight matrix change. Each layer contributes a small aligned increment (H30),
and those increments are produced by distributed weight changes across all
components of each block (MLP, attention, LayerNorm).

The only matrices with consistently high concentration are trivial:
`class_token` (a single 768-dim vector, so trivially rank-1) and
`pos_embedding` (a low-rank perturbation to spatial tokens). Neither helps
detection.

## Consequence

A data-free weight-space anomaly detector based on spectral concentration would
NOT work. The backdoor does not leave a low-rank fingerprint in the weights.
Activation-space methods (PSBD, direction ablation) remain necessary.
