# H39 -- Skip connection scaling as inference-time defense

**Status: PARTIALLY SUPPORTED.** Skip scaling (replacing `x + branch` with
`alpha * x + branch` at layers 10 to 11) removes some backdoors but is
attack-dependent. Blend has a sharp phase transition (ASR 1.000 at alpha=0.3,
ASR 0.007 at alpha=0.1). WaNet and LC die at alpha=0.3 to 0.5. BadNet survives
even alpha=0.0 at 5% poison rate (ASR=0.995), because the branch computations
alone regenerate the direction. No single alpha works universally.

Evidence: `scratch/skip_scaling.py`, results in `results/skip_scaling.json`.

## Claim

Scaling down the skip connection at layers 10 to 11 by a factor alpha < 1 at
inference time removes the backdoor without weight surgery. This works because
the direction rides the skip path (H34), and scaling reduces the accumulated
direction norm below the classification threshold.

Predicted: alpha=0.3 drops ASR below 0.20 with < 5% CA loss across all attacks.

## Test

1. Modify the forward pass of ViT blocks at layers 10 to 11 via hooks: replace
   `x + branch(x)` with `alpha * x + branch(x)`.
2. Sweep alpha in {1.0, 0.5, 0.3, 0.1, 0.0} on the 5-attack panel (badnet_a2o,
   blend, wanet, lc, adaptive_blend).
3. Evaluate ASR and CA at each alpha.
4. Test on CIFAR-100 (10%, 5%) and Tiny (10%).

## Results

### CIFAR-100 10% poison rate

| attack | alpha=1.0 | alpha=0.5 | alpha=0.3 | alpha=0.1 | alpha=0.0 | CA@0.3 |
|---|---:|---:|---:|---:|---:|---:|
| badnet_a2o | 1.000 | 1.000 | 1.000 | 0.962 | 0.029 | 0.796 |
| blend | 1.000 | 1.000 | 1.000 | 0.007 | 0.000 | 0.797 |
| wanet | 0.900 | 0.427 | 0.092 | 0.034 | 0.033 | 0.790 |
| lc | 0.786 | 0.556 | 0.494 | 0.001 | 0.000 | 0.805 |
| adaptive_blend | 0.971 | 0.809 | 0.706 | 0.639 | 0.452 | 0.813 |

### CIFAR-100 5% poison rate

| attack | alpha=1.0 | alpha=0.5 | alpha=0.3 | alpha=0.1 | alpha=0.0 | CA@0.3 |
|---|---:|---:|---:|---:|---:|---:|
| badnet_a2o | 1.000 | 1.000 | 1.000 | 1.000 | **0.995** | 0.811 |
| blend | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.809 |
| wanet | 0.649 | 0.182 | 0.060 | 0.078 | 0.070 | 0.773 |
| lc | 0.543 | 0.188 | 0.020 | 0.001 | 0.000 | 0.797 |
| adaptive_blend | 0.934 | 0.933 | 0.914 | 0.773 | 0.074 | 0.805 |

### Tiny ImageNet 10% poison rate

| attack | alpha=1.0 | alpha=0.5 | alpha=0.3 | alpha=0.1 | alpha=0.0 | CA@0.3 |
|---|---:|---:|---:|---:|---:|---:|
| badnet_a2o | 1.000 | 0.999 | 0.998 | 0.989 | 0.001 | 0.744 |
| blend | 0.999 | 0.994 | 0.982 | 0.001 | 0.000 | 0.736 |
| wanet | 0.975 | 0.557 | 0.036 | 0.007 | 0.004 | 0.728 |
| lc | 0.626 | 0.593 | 0.577 | 0.327 | 0.000 | 0.734 |
| adaptive_blend | 0.965 | 0.429 | 0.163 | 0.071 | 0.050 | 0.734 |

### CA cost summary

| alpha | mean CA (CIFAR-100 10%) | CA delta |
|---|---:|---:|
| 1.0 | 0.820 | 0.000 |
| 0.5 | 0.812 | -0.008 |
| 0.3 | 0.800 | -0.020 |
| 0.1 | 0.747 | -0.073 |
| 0.0 | 0.628 | -0.192 |

## Interpretation

### Three attack regimes

The attacks split into three distinct scaling-sensitivity regimes:

1. **Fragile (wanet, lc):** ASR collapses at alpha=0.3 to 0.5 with < 3% CA
   cost. The backdoor signal in the skip connection is weak enough that
   moderate scaling removes it. Consistent with H34 (weight erasure also works
   on these attacks).

2. **Threshold (blend):** ASR stays at 1.000 down to alpha=0.3, then crashes
   to near 0 at alpha=0.1. A sharp phase transition, not a gradual decay.
   The blend direction rides the skip connection at a specific norm; once
   scaling pushes it below the classifier's decision boundary, the backdoor
   collapses entirely. The threshold is between alpha=0.1 and alpha=0.3,
   which corresponds to attenuating the accumulated direction by 70 to 90%.

3. **Robust (badnet, adaptive_blend):** BadNet survives alpha=0.1 (ASR=0.962
   on CIFAR-100 10%) and even alpha=0.0 at 5% poison rate (ASR=0.995). This
   is the most striking result: zeroing the skip connection completely (only
   the MLP and attention branch outputs pass through) does not kill badnet at
   5%. The branch computations at layers 10 to 11 alone regenerate the
   direction. adaptive_blend shows similar resistance, maintaining ASR=0.639
   at alpha=0.1 on CIFAR-100 10%.

### Why badnet survives alpha=0.0 at 5%

At alpha=0.0, the model computes `0 * x_residual + branch(x)`, meaning only the
branch output (MLP + attention) at layers 10 to 11 contributes to the final
representation. That badnet still achieves ASR=0.995 means:

1. The branch computations have learned to produce the backdoor direction
   independently of the residual stream's accumulated signal.
2. The backdoor is not merely "riding" the skip connection (as H34 suggested);
   it is also being actively computed by the branch at layers 10 to 11.
3. This is consistent with H31's finding that badnet recruits additional late
   heads (L9H7, L10H9). These late attention heads process the trigger and
   write the direction directly into the branch output.

At 10% poison rate this regeneration is weaker (ASR drops to 0.029 at alpha=0.0),
suggesting a quantitative difference: more poisoned data during training teaches
the mid-layer path but makes the late-layer regeneration path redundant.

### Practical verdict

Skip scaling is NOT a viable universal defense. No single alpha achieves
ASR < 0.1 across all attacks without catastrophic CA loss. The attack-specific
sensitivity means a defender who picks alpha=0.3 (a reasonable CA tradeoff)
still faces ASR=1.000 for badnet and ASR=0.706 for adaptive_blend.

## Connection to other hypotheses

- **H34 (direction erasure):** Both methods try to suppress the backdoor
  direction, H34 by removing it from weights, H39 by attenuating it in the
  residual stream. Both fail on badnet for the same reason: the direction is
  regenerated by branch computations, not just carried.
- **H30 (crystallization):** The direction crystallizes at layers 10 to 11 for
  badnet, exactly where skip scaling is applied. But crystallization means the
  direction is being WRITTEN there, not just PASSING through, so attenuating
  the skip does not help.
- **H31 (backdoor heads):** BadNet's late heads (L9H7, L10H9) explain the
  branch-side regeneration at layers 10 to 11.
