# CIFAR-10 and GTSRB Detection Tables (Completion)

These are the secondary datasets. CIFAR-10 is the most-studied dataset in backdoor detection literature and GTSRB is suspiciously easy (AUROC 0.998-1.000 at 1% for top configurations). Results here are for panel completion, not for primary claims. If the method fails on CIFAR-100 and Tiny, these results do not save it.

All numbers: fractional PSU, sigma-matched (sigma >= 0.6), one-sided (low PSU = poisoned).

## CIFAR-10

### token_mask @ before_attention_norm

Benign control AUROC: 0.498

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 0.997 | 0.993 | 0.977 | 0.6 |
| 1% | blend | 1.000 | 0.983 | 0.932 | 0.6 |
| 1% | adaptive_blend (weak) | 0.640 | 0.649 | 0.632 | 0.5 |
| 5% | badnet_a2o | 1.000 | 0.967 | 0.820 | 0.5 |
| 5% | blend | 1.000 | 0.991 | 0.986 | 0.6 |
| 5% | wanet (weak) | 0.786 | 0.693 | 0.044 | 0.5 |
| 5% | adaptive_blend | 0.838 | 0.836 | 0.837 | 0.6 |
| 10% | badnet_a2o | 1.000 | 0.985 | 0.956 | 0.5 |
| 10% | blend | 1.000 | 0.978 | 0.924 | 0.5 |
| 10% | wanet | 0.962 | 0.747 | 0.148 | 0.5 |
| 10% | lc | 0.979 | 0.953 | 0.815 | 0.6 |
| 10% | adaptive_blend | 0.926 | 0.933 | 0.911 | 0.6 |
| -- | wanet 1% | 0.123 | not implanted | -- | -- |
| -- | lc 1% | 0.250 | not implanted | -- | -- |
| -- | lc 5% | 0.409 | not implanted | -- | -- |

### gain_scale @ mlp_norm_out

Benign control AUROC: 0.493

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 0.997 | 0.996 | 0.997 | 2 |
| 1% | blend | 1.000 | 0.996 | 1.000 | 2 |
| 1% | adaptive_blend (weak) | 0.640 | 0.725 | 0.530 | 2 |
| 5% | badnet_a2o | 1.000 | 0.983 | 0.996 | 2 |
| 5% | blend | 1.000 | 0.999 | 1.000 | 2 |
| 5% | wanet (weak) | 0.786 | 0.516 | 0.039 | 2 |
| 5% | adaptive_blend | 0.838 | 0.833 | 0.833 | 2 |
| 10% | badnet_a2o | 1.000 | 0.999 | 1.000 | 2 |
| 10% | blend | 1.000 | 0.997 | 1.000 | 2 |
| 10% | wanet | 0.962 | 0.665 | 0.142 | 2 |
| 10% | lc | 0.979 | 0.924 | 0.444 | 2 |
| 10% | adaptive_blend | 0.926 | 0.919 | 0.926 | 2 |
| -- | wanet 1% | 0.123 | not implanted | -- | -- |
| -- | lc 1% | 0.250 | not implanted | -- | -- |
| -- | lc 5% | 0.409 | not implanted | -- | -- |

### CIFAR-10 summary

| Configuration | 1% | 5% | 10% | All |
|---|---:|---:|---:|---:|
| gain_scale @ mlp_norm_out | 0.906 | 0.833 | 0.901 | 0.880 |
| token_mask @ before_attention_norm | 0.875 | 0.872 | 0.919 | 0.893 |

Both configurations perform well. token_mask has higher mean AUROC across all rates. gain_scale has higher peak (0.996 on badnet at 1%) but worse WaNet (0.516 at 5%). CIFAR-10 does not discriminate between the two methods as sharply as CIFAR-100 does.

## GTSRB

### token_mask @ before_attention_norm

Benign control AUROC: 0.500

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 1.000 | 0.995 | 0.975 | 0.5 |
| 1% | blend | 1.000 | 1.000 | 1.000 | 0.3 |
| 5% | badnet_a2o | 1.000 | 1.000 | 1.000 | 0.3 |
| 5% | blend | 1.000 | 0.998 | 0.996 | 0.3 |
| 5% | wanet (weak) | 0.777 | 0.779 | 0.330 | 0.4 |
| 5% | adaptive_blend | 0.987 | 0.986 | 0.986 | 0.4 |
| 10% | badnet_a2o | 1.000 | 0.999 | 0.992 | 0.4 |
| 10% | blend | 1.000 | 1.000 | 1.000 | 0.2 |
| 10% | wanet | 0.921 | 0.927 | 0.758 | 0.5 |
| 10% | adaptive_blend | 1.000 | 0.999 | 1.000 | 0.3 |
| -- | wanet 1% | 0.032 | not implanted | -- | -- |
| -- | lc 1% | 0.467 | not implanted | -- | -- |
| -- | lc 5% | 0.237 | not implanted | -- | -- |
| -- | lc 10% | 0.130 | not implanted | -- | -- |
| -- | a_blend 1% | 0.447 | not implanted | -- | -- |

### gain_scale @ mlp_norm_out

Benign control AUROC: 0.501

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 1.000 | 1.000 | 1.000 | 2 |
| 1% | blend | 1.000 | 1.000 | 1.000 | 2 |
| 5% | badnet_a2o | 1.000 | 0.997 | 1.000 | 2 |
| 5% | blend | 1.000 | 0.972 | 0.890 | 2 |
| 5% | wanet (weak) | 0.777 | 0.459 | 0.043 | 2 |
| 5% | adaptive_blend | 0.987 | 0.995 | 0.986 | 2 |
| 10% | badnet_a2o | 1.000 | 0.969 | 0.892 | 2 |
| 10% | blend | 1.000 | 1.000 | 1.000 | 1 |
| 10% | wanet | 0.921 | 0.479 | 0.037 | 2 |
| 10% | adaptive_blend | 1.000 | 0.969 | 0.999 | 2 |
| -- | wanet 1% | 0.032 | not implanted | -- | -- |
| -- | lc 1% | 0.467 | not implanted | -- | -- |
| -- | lc 5% | 0.237 | not implanted | -- | -- |
| -- | lc 10% | 0.130 | not implanted | -- | -- |
| -- | a_blend 1% | 0.447 | not implanted | -- | -- |

### GTSRB summary

| Configuration | 1% | 5% | 10% | All |
|---|---:|---:|---:|---:|
| token_mask @ before_attention_norm | 0.998 | 0.941 | 0.981 | 0.968 |
| gain_scale @ mlp_norm_out | 1.000 | 0.856 | 0.854 | 0.884 |

GTSRB shows a striking divergence: token_mask is much better overall (0.968 vs 0.884) because gain_scale INVERTS on WaNet (0.459 at 5%, 0.479 at 10%). WaNet's spatial deformation triggers a false confidence in gain_scale's LayerNorm scaling on GTSRB, where traffic sign images already have high spatial variability.

LC fails to implant at any poison rate on GTSRB (max ASR 0.467), so the entire LC column is excluded from the coverage bar.

## GTSRB is suspiciously easy

GTSRB produces near-perfect detection on most attacks: 0.995-1.000 AUROC on BadNet and Blend at 1% with token_mask. This is because GTSRB images are small (32x32), low-resolution traffic signs with highly distinctive per-class features. The clean decision boundary is wide and the backdoor direction stands out sharply. Results on GTSRB should not be cited as evidence of method effectiveness.

## Source

`python defence_tables.py --operator {op} --position {pos} --allow-partial`
Benign: `vit_cifar10_benign` and `vit_gtsrb_benign` checkpoints.
