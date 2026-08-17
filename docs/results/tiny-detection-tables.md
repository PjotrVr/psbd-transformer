# Tiny ImageNet Detection Tables

Tiny ImageNet is the largest dataset in the evaluation grid (200 classes, 64x64 images). It is the most realistic proxy for real-world deployment. Together with CIFAR-100, it forms the primary evaluation pair: if the method does not work on these two, it is not publishable.

All numbers: fractional PSU, sigma-matched (sigma >= 0.6), one-sided (low PSU = poisoned). AUROC / TPR@5%FPR shown side by side.

Three attacks fail to implant at 1% on Tiny: WaNet (ASR 0.379), LC (ASR 0.389), Adaptive Blend (ASR 0.431). These are excluded from the 1% rows.

## token_mask @ before_attention_norm (recommended configuration)

Benign control AUROC: 0.501

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 0.999 | 0.932 | 0.343 | 0.4 |
| 1% | blend | 0.999 | 0.991 | 0.975 | 0.4 |
| 5% | badnet_a2o | 1.000 | 0.938 | 0.454 | 0.4 |
| 5% | blend | 0.999 | 0.989 | 0.990 | 0.4 |
| 5% | wanet | 0.941 | 0.934 | 0.791 | 0.4 |
| 5% | lc (weak) | 0.706 | 0.777 | 0.530 | 0.4 |
| 5% | adaptive_blend | 0.924 | 0.920 | 0.924 | 0.4 |
| 10% | badnet_a2o | 1.000 | 0.964 | 0.775 | 0.4 |
| 10% | blend | 0.999 | 0.992 | 0.989 | 0.4 |
| 10% | wanet | 0.975 | 0.955 | 0.803 | 0.4 |
| 10% | lc (weak) | 0.626 | 0.729 | 0.087 | 0.4 |
| 10% | adaptive_blend | 0.965 | 0.958 | 0.962 | 0.4 |
| -- | wanet 1% | 0.379 | not implanted | -- | -- |
| -- | lc 1% | 0.389 | not implanted | -- | -- |
| -- | adaptive_blend 1% | 0.431 | not implanted | -- | -- |

Mean AUROC at 1%: 0.962. Mean AUROC at 5%: 0.912. Mean AUROC at 10%: 0.920.

Blend detection is near-perfect across all rates (0.991, 0.989, 0.992). BadNet at 1% has high AUROC (0.932) but low TPR@5% (0.343), similar to the CIFAR-100 pattern but less severe. WaNet is well-detected at 5% and 10% (0.934, 0.955), a dataset where WaNet actually implants at 5% unlike CIFAR-100 where it stays below ASR threshold.

## gain_scale @ mlp_norm_out (highest 1% AUROC overall)

Benign control AUROC: 0.492

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 0.999 | 0.933 | 0.034 | 1 |
| 1% | blend | 0.999 | 0.982 | 0.997 | 1 |
| 5% | badnet_a2o | 1.000 | 0.970 | 0.976 | 1 |
| 5% | blend | 0.999 | 0.976 | 0.999 | 1 |
| 5% | wanet | 0.941 | 0.852 | 0.263 | 1 |
| 5% | lc (weak) | 0.706 | 0.728 | 0.639 | 1 |
| 5% | adaptive_blend | 0.924 | 0.910 | 0.769 | 1 |
| 10% | badnet_a2o | 1.000 | 0.943 | 0.135 | 1 |
| 10% | blend | 0.999 | 0.979 | 0.998 | 1 |
| 10% | wanet | 0.975 | 0.940 | 0.644 | 1 |
| 10% | lc (weak) | 0.626 | 0.632 | 0.267 | 1 |
| 10% | adaptive_blend | 0.965 | 0.933 | 0.687 | 1 |
| -- | wanet 1% | 0.379 | not implanted | -- | -- |
| -- | lc 1% | 0.389 | not implanted | -- | -- |
| -- | adaptive_blend 1% | 0.431 | not implanted | -- | -- |

Mean AUROC at 1%: 0.958. Mean AUROC at 5%: 0.887. Mean AUROC at 10%: 0.885.

**Critical weakness on Tiny.** BadNet TPR@5%FPR is 0.034 at 1% and 0.135 at 10%, despite AUROC of 0.933 and 0.943. The ROC curve is extremely front-loaded: the method separates clean from poisoned at high FPR thresholds but fails at the operationally important 5% FPR. This means gain_scale would miss ~97% of BadNet-poisoned samples at 1% when the false positive rate is held to 5%.

This is the key argument for token_mask over gain_scale on Tiny: token_mask has 0.343 TPR@5% on the same cell (10x better), and 0.775 at 10% (6x better).

Blend detection remains excellent (TPR@5% = 0.997 at 1%).

## dropout @ pre_residual (PSBD paper's original position)

Benign control AUROC: 0.500

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 0.999 | 0.933 | 0.667 | 0.3 |
| 1% | blend | 0.999 | 0.950 | 0.713 | 0.3 |
| 5% | badnet_a2o | 1.000 | 0.866 | 0.452 | 0.3 |
| 5% | blend | 0.999 | 0.966 | 0.797 | 0.3 |
| 5% | wanet | 0.941 | 0.842 | 0.259 | 0.3 |
| 5% | lc (weak) | 0.706 | 0.677 | 0.074 | 0.3 |
| 5% | adaptive_blend | 0.924 | 0.915 | 0.774 | 0.3 |
| 10% | badnet_a2o | 1.000 | 0.945 | 0.712 | 0.3 |
| 10% | blend | 0.999 | 0.987 | 0.942 | 0.3 |
| 10% | wanet | 0.975 | 0.851 | 0.358 | 0.3 |
| 10% | lc (weak) | 0.626 | 0.546 | 0.031 | 0.3 |
| 10% | adaptive_blend | 0.965 | 0.934 | 0.704 | 0.3 |
| -- | wanet 1% | 0.379 | not implanted | -- | -- |
| -- | lc 1% | 0.389 | not implanted | -- | -- |
| -- | adaptive_blend 1% | 0.431 | not implanted | -- | -- |

Mean AUROC at 1%: 0.942. Mean AUROC at 5%: 0.853. Mean AUROC at 10%: 0.853.

The PSBD paper's original position performs surprisingly well on Tiny. BadNet at 1% matches gain_scale's AUROC (0.933) with much better TPR@5% (0.667 vs 0.034). However, it falls behind token_mask at 5% and 10% on WaNet and LC.

## dropout @ before_attention_norm

Benign control AUROC: 0.499

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 0.999 | 0.841 | 0.240 | 0.4 |
| 1% | blend | 0.999 | 0.895 | 0.325 | 0.4 |
| 5% | badnet_a2o | 1.000 | 0.782 | 0.167 | 0.4 |
| 5% | blend | 0.999 | 0.881 | 0.340 | 0.4 |
| 5% | wanet | 0.941 | 0.770 | 0.178 | 0.5 |
| 5% | lc (weak) | 0.706 | 0.707 | 0.051 | 0.5 |
| 5% | adaptive_blend | 0.924 | 0.876 | 0.578 | 0.4 |
| 10% | badnet_a2o | 1.000 | 0.838 | 0.266 | 0.4 |
| 10% | blend | 0.999 | 0.924 | 0.473 | 0.4 |
| 10% | wanet | 0.975 | 0.749 | 0.165 | 0.4 |
| 10% | lc (weak) | 0.626 | 0.662 | 0.034 | 0.5 |
| 10% | adaptive_blend | 0.965 | 0.875 | 0.406 | 0.4 |
| -- | wanet 1% | 0.379 | not implanted | -- | -- |
| -- | lc 1% | 0.389 | not implanted | -- | -- |
| -- | adaptive_blend 1% | 0.431 | not implanted | -- | -- |

Mean AUROC at 1%: 0.868. Mean AUROC at 5%: 0.803. Mean AUROC at 10%: 0.810.

Dropout at the same position as token_mask: the operator swap from token_mask to dropout costs 0.094 mean AUROC at 1% (0.962 vs 0.868). This is the operator effect at fixed position, consistent with the H28 finding that operator matters but less than position.

## Summary: Tiny ImageNet mean AUROC by configuration

| Configuration | 1% | 5% | 10% | All |
|---|---:|---:|---:|---:|
| token_mask @ before_attention_norm | 0.962 | 0.912 | 0.920 | 0.926 |
| gain_scale @ mlp_norm_out | 0.958 | 0.887 | 0.885 | 0.897 |
| dropout @ pre_residual | 0.942 | 0.853 | 0.853 | 0.866 |
| dropout @ before_attention_norm | 0.868 | 0.803 | 0.810 | 0.817 |

token_mask @ before_attention_norm is the best configuration on Tiny, leading at all three poison rates. The gap over gain_scale is small in AUROC (0.004 at 1%) but large in TPR@5%FPR (0.343 vs 0.034 on badnet at 1%).

The story reverses from CIFAR-100: on CIFAR-100, gain_scale leads by 0.096 AUROC at 1%. On Tiny, token_mask leads by 0.004. This dataset-dependence is why the recommended configuration is based on worst-case robustness rather than best-case performance.

## Source

`python defence_tables.py --operator {op} --position {pos} --allow-partial`
Benign: `vit_tiny_benign` checkpoint probed with badnet_a2o trigger.
