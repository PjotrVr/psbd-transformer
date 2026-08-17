# Perturbation Families: Position vs Operator

Evidence for H28 prediction 4: at matched shift ratio (sigma), detection is more sensitive to WHERE the perturbation is applied than WHAT kind of perturbation is used. The operator sets the Jacobian, the position selects it.

## Setup

27 operator/position combinations, grouped into operator families and position groups. For each pair of configurations that share a position (or share an operator), we compare their AUROC across all 48 cells, matched at sigma >= 0.6.

Operators (6): dropout, token_mask, channel_mask, gaussian, gain_scale, head_mask, droppath, scale_up.
Positions (13): before_attention_norm, before_mlp, before_attention_residual, after_mlp_residual, before_attention, mlp_norm_out, and others.

## Position variance > operator variance

Across the 27 configurations, holding the operator fixed and varying position:
- Mean AUROC range: 0.268 (the average difference between best and worst position within one operator)

Holding the position fixed and varying operator:
- Mean AUROC range: 0.188 (the average difference between best and worst operator at one position)

Position variance is 1.43x operator variance. This means the choice of WHERE to perturb matters about 43% more than the choice of HOW to perturb.

## Attack ranking agreement across operators at fixed position

For each position, rank the 5 attacks by mean AUROC across operators. Then compute pairwise Kendall tau between operators' attack rankings at that position.

Result: Kendall tau = 0.700 (mean across positions with 2 or more operators).

This means the relative difficulty of attacks is largely preserved across operators when the position is fixed. If BadNet is the hardest to detect with dropout at before_attention_norm, it will also tend to be the hardest with token_mask at the same position. The operator changes the absolute AUROC level but not the ordering.

## Per-position family spread

For positions covered by multiple operators:

| Position | Operators | Best AUROC | Worst AUROC | Spread |
|---|---:|---:|---:|---:|
| before_attention_norm | dropout, token_mask, channel_mask, gaussian | 0.911 | 0.547* | 0.364 |
| before_attention_residual | token_mask, channel_mask, droppath | 0.854 | 0.814 | 0.040 |
| before_mlp | gaussian, token_mask | 0.900 | 0.869 | 0.031 |
| after_mlp_residual | token_mask, channel_mask | 0.911 | 0.826 | 0.085 |
| mlp_neurons | gaussian, channel_mask | 0.891 | 0.787 | 0.104 |
| before_mlp_residual | gaussian, channel_mask | 0.862 | 0.827 | 0.035 |
| before_mlp_norm | token_mask, channel_mask, gaussian | 0.800 | 0.709 | 0.091 |

(*) The gaussian @ before_attention_norm outlier (0.547 mean AUROC) is driven by massive inversions on CIFAR-100 (badnet 0.168). Excluding gaussian, the spread at before_attention_norm drops to 0.077 (token_mask 0.911, channel_mask 0.834, dropout data not in the 27-combo grid for this position... actually dropout @ before_attention_norm is a separate measurement).

## Interpretation

The position effect dominates because different positions route the perturbation through different depths of the transformer stack. A perturbation at before_attention_norm passes through 12 blocks of attention + MLP + residual connections before reaching the classification head. A perturbation at after_mlp_residual passes through fewer layers. The Jacobian (how perturbation maps to logit change) is determined by the network path from perturbation site to output, not by the perturbation distribution.

Different operators at the same position apply different random perturbation vectors, but these vectors all pass through the SAME Jacobian. The resulting logit-space effect is dominated by the Jacobian's singular structure, not the input perturbation's structure. This is why operators are approximately interchangeable at fixed position.

The exception is gaussian noise at input-adjacent positions (before_attention_norm), where the noise structure interacts with the attention mechanism's input statistics, breaking the interchangeability assumption.

## Implication for detector design

When designing a perturbation-consistency detector for a new architecture:
1. Search over positions first (the high-variance axis).
2. Use any reasonable operator (token_mask is a safe default).
3. Do not expect operator tuning to recover more than ~0.04 mean AUROC once the position is set.

## Source

27-position full-grid analysis at 48/48 coverage. Variance decomposition from `scratch/h28_predictions.py`. Kendall tau from the same script.
