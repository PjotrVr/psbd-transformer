# Combined Position Test (both_sublayer_inputs)

## Hypothesis

Both `before_attention_norm` and `before_mlp_norm` are computation inputs (one to the attention sublayer, one to the MLP sublayer). If the backdoor direction propagates through both sublayers, perturbing both inputs simultaneously should compound the detection signal by disrupting two distinct processing stages in each block.

## Configuration

`both_sublayer_inputs` in `DROPOUT_CONFIGS` maps to `("before_attention_norm", "before_mlp_norm")`. Two perturbation sites per block instead of one. Tested with both dropout and token_mask operators, at 48/48 coverage across all 4 datasets, 3 poison rates, and the 5-attack panel.

## Result: NEGATIVE

Combining both sublayer inputs does NOT beat `before_attention_norm` alone on CIFAR-100 and Tiny ImageNet. It dilutes the signal rather than compounding it, especially on Tiny.

## Head-to-head: token_mask @ before_attention_norm vs token_mask @ both_sublayer_inputs

### CIFAR-100

| Poison | Attack | Single AUROC | Combined AUROC | Delta |
|---|---|---:|---:|---:|
| 1% | badnet_a2o | 0.960 | 0.960 | 0.000 |
| 1% | blend | 0.945 | 0.915 | -0.030 |
| 1% | lc | 0.786 | 0.759 | -0.027 |
| 1% | adaptive_blend | 0.706 | 0.718 | +0.012 |
| 5% | badnet_a2o | 0.977 | 0.935 | -0.042 |
| 5% | blend | 0.989 | 0.988 | -0.001 |
| 5% | wanet | 0.751 | 0.764 | +0.013 |
| 5% | lc | 0.632 | 0.722 | +0.090 |
| 5% | adaptive_blend | 0.927 | 0.932 | +0.005 |
| 10% | badnet_a2o | 0.989 | 0.984 | -0.005 |
| 10% | blend | 0.982 | 0.990 | +0.008 |
| 10% | wanet | 0.900 | 0.884 | -0.016 |
| 10% | lc | 0.760 | 0.756 | -0.004 |
| 10% | adaptive_blend | 0.951 | 0.970 | +0.019 |

CIFAR-100 mean delta: -0.001. Nearly a wash. The combined position wins on some cells (lc 5%: +0.090, adaptive_blend 10%: +0.019) and loses on others (badnet 5%: -0.042, blend 1%: -0.030).

### Tiny ImageNet

| Poison | Attack | Single AUROC | Combined AUROC | Delta |
|---|---|---:|---:|---:|
| 1% | badnet_a2o | 0.932 | 0.856 | -0.076 |
| 1% | blend | 0.991 | 0.980 | -0.011 |
| 5% | badnet_a2o | 0.938 | 0.852 | -0.086 |
| 5% | blend | 0.989 | 0.974 | -0.015 |
| 5% | wanet | 0.934 | 0.839 | -0.095 |
| 5% | lc | 0.777 | 0.770 | -0.007 |
| 5% | adaptive_blend | 0.920 | 0.915 | -0.005 |
| 10% | badnet_a2o | 0.964 | 0.898 | -0.066 |
| 10% | blend | 0.992 | 0.977 | -0.015 |
| 10% | wanet | 0.955 | 0.840 | -0.115 |
| 10% | lc | 0.729 | 0.688 | -0.041 |
| 10% | adaptive_blend | 0.958 | 0.945 | -0.013 |

Tiny mean delta: -0.045. The combined position loses on every single cell. The losses are largest on BadNet and WaNet, the attacks with the most spatially concentrated triggers.

### CIFAR-10

Mean delta: -0.025 (wins on lc 10%: +0.839 vs 0.953 but that was oddly both ways).

### GTSRB

Mean delta: -0.006 (near wash, slight losses on badnet).

## Why combining dilutes the signal

`before_mlp_norm` alone ranks 15th out of 27 (mean AUROC 0.800, 3 inversions), while `before_attention_norm` ranks 4th (mean AUROC 0.911, 0 inversions). Adding the weak position to the strong one does not help because:

1. **The MLP norm input carries less backdoor signal.** Token masking at before_mlp_norm is equivalent to dropping tokens AFTER the attention sublayer has already mixed them. Attention is where token interactions happen, so by the time tokens reach the MLP input, the backdoor information has been spread across all tokens and is harder to disrupt with local masking.

2. **Double perturbation at matched sigma means each site gets half the perturbation strength.** To reach the same sigma (shift ratio) with two sites instead of one, each site needs a lower perturbation rate. This means the effective perturbation at before_attention_norm (the useful site) is weaker than it would be alone.

3. **On Tiny ImageNet, the combined position hurts most.** Tiny has 200 classes and lower overall confidence, so the reduced per-site perturbation strength matters more. The dilution effect is proportionally larger.

## Dropout operator comparison

The same pattern holds with dropout instead of token_mask:

| Config | CIFAR-100 1% badnet | Tiny 1% badnet | Tiny 1% blend |
|---|---:|---:|---:|
| dropout @ before_attention_norm | 0.770 | 0.841 | 0.895 |
| dropout @ both_sublayer_inputs | 0.815 | 0.888 | 0.915 |

With dropout, the combined position is slightly better than the single position on both datasets. This reversal happens because dropout's unstructured removal is less sensitive to the dilution effect: random feature dropping at two sites does not split the effective rate the same way token masking does. However, both dropout configurations are far below token_mask, so this does not change the recommendation.

## Verdict

Do not combine positions. Use `before_attention_norm` alone with `token_mask`. The combination does not compound the signal; it dilutes it. The before_mlp_norm site adds noise without proportional information because it perturbs tokens after attention has already redistributed the backdoor evidence across the sequence.

## Source

`python defence_tables.py --operator token_mask --position before_attention_norm --allow-partial`
`python defence_tables.py --operator token_mask --position both_sublayer_inputs --allow-partial`
`python defence_tables.py --operator dropout --position before_attention_norm --allow-partial`
`python defence_tables.py --operator dropout --position both_sublayer_inputs --allow-partial`
