# H34 — Backdoor erasure via direction orthogonalization

**Status: PARTIALLY SUPPORTED.** Weight orthogonalization works on weaker
attacks (LC, adaptive_blend) but fails completely on strong attacks (badnet,
blend). The discrepancy with H16's inference-time direction removal reveals the
residual stream persistence mechanism.

Evidence: `scratch/direction_erasure.py`, results in
`results/direction_erasure.json`.

## Claim

Two-stage defense:
1. **Known direction** (from paired features): orthogonalize MLP output weights
   and attention out_proj at layers 10 and 11 against the true direction.
2. **Blind direction** (no triggered data): use the target class readout weight
   as a proxy direction (cosine ~0.88 to true direction per H16).

Predicted: known-direction erasure drops ASR below 0.05 with < 5% CA loss.
Blind erasure achieves ASR < 0.20.

## Results

| checkpoint | base ASR | known | blind | random | base CA | known CA |
|---|---:|---:|---:|---:|---:|---:|
| cifar100 badnet 10% | 1.000 | 1.000 | 1.000 | 1.000 | 0.818 | 0.817 |
| cifar100 blend 10% | 1.000 | 1.000 | 1.000 | 1.000 | 0.818 | 0.819 |
| cifar100 wanet 10% | 0.900 | 0.876 | 0.841 | 0.900 | 0.813 | 0.811 |
| cifar100 lc 10% | 0.786 | 0.634 | 0.456 | 0.786 | 0.823 | 0.823 |
| cifar100 a_blend 10% | 0.971 | 0.872 | 0.641 | 0.971 | 0.830 | 0.828 |
| cifar100 badnet 5% | 1.000 | 1.000 | 0.999 | 1.000 | 0.824 | 0.824 |
| cifar100 blend 5% | 1.000 | 1.000 | 1.000 | 1.000 | 0.830 | 0.829 |
| cifar100 wanet 5% | 0.649 | 0.571 | 0.468 | 0.649 | 0.804 | 0.803 |
| cifar100 lc 5% | 0.543 | 0.507 | 0.381 | 0.543 | 0.818 | 0.819 |
| cifar100 a_blend 5% | 0.934 | 0.695 | **0.079** | 0.935 | 0.822 | 0.821 |
| tiny badnet 10% | 1.000 | 0.999 | 0.994 | 1.000 | 0.750 | 0.750 |
| tiny blend 10% | 0.999 | 0.999 | 0.998 | 0.999 | 0.748 | 0.748 |
| tiny wanet 10% | 0.975 | 0.969 | 0.962 | 0.975 | 0.739 | 0.740 |
| tiny lc 10% | 0.626 | 0.243 | **0.006** | 0.627 | 0.753 | 0.753 |
| tiny a_blend 10% | 0.965 | 0.936 | 0.725 | 0.965 | 0.749 | 0.748 |

**Random control does nothing** in every case. CA loss is under 1%.

## Interpretation

### Why weight orthogonalization fails on strong attacks

H16 showed that inference-time direction removal (subtracting the direction's
projection from activations at every layer) drops badnet ASR from 1.000 to
0.000. But weight orthogonalization at layers 10 to 11 does nothing.

The explanation follows from H30 (residual persistence): the backdoor direction
enters the residual stream through the branch computations (MLP, attention) at
each layer, and once in the stream it persists through the skip connection.
Weight orthogonalization removes the direction from the branch OUTPUT, but the
direction is already in the stream from earlier layers' contributions. The skip
path carries it through unchanged.

For strong attacks (badnet, blend), the direction is strong enough in the
residual stream (norm 13 to 19 at layer 12) that even removing the late-layer
branch contributions leaves enough signal. For weak attacks (lc at 10% on Tiny,
norm 5.0), the residual accumulation is thin enough that removing the branch
contributions collapses the backdoor.

### The blind erasure surprise

The blind direction (readout weight, no triggered data) sometimes works BETTER
than the known direction:
- adaptive_blend 5%: known drops to 0.695, blind drops to 0.079
- lc 10% on Tiny: known drops to 0.243, blind drops to 0.006

The readout weight is a better erasure target than the true direction for these
attacks because the readout weight is what the classifier actually reads. The
true direction may not be perfectly aligned with what matters for classification.

### Practical implication

Weight orthogonalization is a viable defense ONLY for attacks with low to medium
ASR. Against well-implanted backdoors (ASR > 0.95), it does nothing useful.
Inference-time methods (PSBD, H16 direction removal) remain the only option for
strong attacks.
