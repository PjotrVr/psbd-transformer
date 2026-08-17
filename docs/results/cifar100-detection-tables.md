# CIFAR-100 Detection Tables

CIFAR-100 is the hardest dataset in the evaluation grid. It has 100 classes, making the decision boundary more fragile and the backdoor direction harder to isolate. Any method that works here is likely to generalize. Results that fail on CIFAR-100 are not publishable regardless of CIFAR-10/GTSRB performance.

All numbers: fractional PSU, sigma-matched (sigma >= 0.6), one-sided (low PSU = poisoned). AUROC / TPR@5%FPR shown side by side.

## token_mask @ before_attention_norm (recommended configuration)

Benign control AUROC: 0.495

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 1.000 | 0.960 | 0.781 | 0.4 |
| 1% | blend | 0.990 | 0.945 | 0.705 | 0.4 |
| 1% | lc | 0.874 | 0.786 | 0.035 | 0.4 |
| 1% | adaptive_blend (weak) | 0.702 | 0.706 | 0.668 | 0.4 |
| 5% | badnet_a2o | 1.000 | 0.977 | 0.935 | 0.4 |
| 5% | blend | 1.000 | 0.989 | 0.955 | 0.5 |
| 5% | wanet (weak) | 0.649 | 0.751 | 0.412 | 0.3 |
| 5% | lc (weak) | 0.543 | 0.632 | 0.015 | 0.4 |
| 5% | adaptive_blend | 0.934 | 0.927 | 0.894 | 0.4 |
| 10% | badnet_a2o | 1.000 | 0.989 | 0.999 | 0.4 |
| 10% | blend | 1.000 | 0.982 | 1.000 | 0.4 |
| 10% | wanet | 0.900 | 0.900 | 0.628 | 0.4 |
| 10% | lc (weak) | 0.786 | 0.760 | 0.032 | 0.4 |
| 10% | adaptive_blend | 0.971 | 0.951 | 0.934 | 0.4 |
| -- | wanet 1% | 0.044 | not implanted | -- | -- |

Mean AUROC at 1%: 0.849. Mean AUROC at 5%: 0.855. Mean AUROC at 10%: 0.916.

LC is the consistent weak point. At 1%, LC achieves 0.786 AUROC but only 0.035 TPR@5%FPR, meaning the ROC curve has no operating point with both low FPR and high TPR. This is characteristic of clean-label attacks: the trigger does not create a strong backdoor direction because the training labels are correct.

## gain_scale @ mlp_norm_out (highest 1% AUROC)

Benign control AUROC: 0.501

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 1.000 | 0.999 | 1.000 | 2 |
| 1% | blend | 0.990 | 0.992 | 0.989 | 2 |
| 1% | lc | 0.874 | 0.935 | 0.876 | 2 |
| 1% | adaptive_blend (weak) | 0.702 | 0.854 | 0.664 | 2 |
| 5% | badnet_a2o | 1.000 | 0.999 | 1.000 | 2 |
| 5% | blend | 1.000 | 1.000 | 1.000 | 2 |
| 5% | wanet (weak) | 0.649 | 0.815 | 0.604 | 2 |
| 5% | lc (weak) | 0.543 | 0.646 | 0.080 | 2 |
| 5% | adaptive_blend | 0.934 | 0.960 | 0.934 | 2 |
| 10% | badnet_a2o | 1.000 | 0.999 | 1.000 | 2 |
| 10% | blend | 1.000 | 0.985 | 1.000 | 1 |
| 10% | wanet | 0.900 | 0.942 | 0.902 | 2 |
| 10% | lc (weak) | 0.786 | 0.892 | 0.798 | 2 |
| 10% | adaptive_blend | 0.971 | 0.953 | 0.789 | 2 |
| -- | wanet 1% | 0.044 | not implanted | -- | -- |

Mean AUROC at 1%: 0.945. Mean AUROC at 5%: 0.884. Mean AUROC at 10%: 0.954.

gain_scale dominates CIFAR-100. At 1% it beats token_mask @ before_attention_norm by +0.039 on badnet, +0.047 on blend, +0.149 on lc, and +0.148 on adaptive_blend. The lc improvement is particularly notable: 0.935 vs 0.786, with TPR@5% going from 0.035 to 0.876. This is the IBD-PSC mechanism (LayerNorm scaling) which amplifies any direction that deviates from the learned statistics.

## dropout @ pre_residual (PSBD paper's original position)

Benign control AUROC: 0.484

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 1.000 | 0.847 | 0.473 | 0.3 |
| 1% | blend | 0.990 | 0.655 | 0.007 | 0.3 |
| 1% | lc | 0.874 | 0.599 | 0.006 | 0.3 |
| 1% | adaptive_blend (weak) | 0.702 | 0.645 | 0.098 | 0.3 |
| 5% | badnet_a2o | 1.000 | 0.821 | 0.298 | 0.3 |
| 5% | blend | 1.000 | 0.933 | 0.549 | 0.3 |
| 5% | wanet (weak) | 0.649 | 0.671 | 0.058 | 0.3 |
| 5% | lc (weak) | 0.543 | 0.535 | 0.010 | 0.3 |
| 5% | adaptive_blend | 0.934 | 0.872 | 0.560 | 0.3 |
| 10% | badnet_a2o | 1.000 | 0.821 | 0.400 | 0.3 |
| 10% | blend | 1.000 | 0.994 | 0.991 | 0.3 |
| 10% | wanet | 0.900 | 0.684 | 0.046 | 0.3 |
| 10% | lc (weak) | 0.786 | 0.494 | 0.007 | 0.3 |
| 10% | adaptive_blend | 0.971 | 0.897 | 0.441 | 0.3 |
| -- | wanet 1% | 0.044 | not implanted | -- | -- |

Mean AUROC at 1%: 0.687. Mean AUROC at 5%: 0.766. Mean AUROC at 10%: 0.778.

The PSBD paper's original position (dropout before the residual add) underperforms both token_mask and gain_scale on CIFAR-100. The gap is largest at 1%: 0.847 vs 0.960 (token_mask) or 0.999 (gain_scale) on badnet; 0.655 vs 0.945 or 0.992 on blend. This motivates the position search: the paper's ResNet results do not transfer to ViT at this position.

## dropout @ before_attention_norm

Benign control AUROC: 0.490

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 1.000 | 0.770 | 0.114 | 0.4 |
| 1% | blend | 0.990 | 0.855 | 0.301 | 0.5 |
| 1% | lc | 0.874 | 0.418 | 0.013 | 0.4 |
| 1% | adaptive_blend (weak) | 0.702 | 0.723 | 0.468 | 0.5 |
| 5% | badnet_a2o | 1.000 | 0.705 | 0.001 | 0.4 |
| 5% | blend | 1.000 | 0.960 | 0.778 | 0.5 |
| 5% | wanet (weak) | 0.649 | 0.505 | 0.036 | 0.4 |
| 5% | lc (weak) | 0.543 | 0.449 | 0.022 | 0.5 |
| 5% | adaptive_blend | 0.934 | 0.753 | 0.282 | 0.4 |
| 10% | badnet_a2o | 1.000 | 0.761 | 0.022 | 0.4 |
| 10% | blend | 1.000 | 0.942 | 0.474 | 0.4 |
| 10% | wanet | 0.900 | 0.547 | 0.022 | 0.4 |
| 10% | lc (weak) | 0.786 | 0.399 | 0.015 | 0.5 |
| 10% | adaptive_blend | 0.971 | 0.926 | 0.645 | 0.5 |
| -- | wanet 1% | 0.044 | not implanted | -- | -- |

Mean AUROC at 1%: 0.692. Mean AUROC at 5%: 0.674. Mean AUROC at 10%: 0.715.

Same position as token_mask, but with dropout instead. Dropout at this position is much worse than token_mask (0.770 vs 0.960 on badnet at 1%). This demonstrates that the operator matters: token masking removes entire patches (structured removal), while dropout removes individual features (unstructured removal). The structured removal is more effective at disrupting the trigger pattern because backdoor triggers are spatially localized.

## scale_up @ input_pixels (SCALE-UP port)

| Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 1.000 | 0.933 | 0.251 | 2 |
| 1% | blend | 0.990 | 0.691 | 0.010 | 2 |
| 1% | lc | 0.874 | 0.840 | 0.477 | 2 |
| 1% | adaptive_blend (weak) | 0.702 | 0.634 | 0.228 | 2 |
| 5% | badnet_a2o | 1.000 | 0.943 | 0.057 | 2 |
| 5% | blend | 1.000 | 0.811 | 0.011 | 2 |
| 5% | wanet (weak) | 0.649 | 0.351 | 0.058 | 2 |
| 5% | lc (weak) | 0.543 | 0.657 | 0.272 | 2 |
| 5% | adaptive_blend | 0.934 | 0.797 | 0.344 | 2 |
| 10% | badnet_a2o | 1.000 | 0.909 | 0.021 | 2 |
| 10% | blend | 1.000 | 0.919 | 0.485 | 2 |
| 10% | wanet | 0.900 | 0.381 | 0.025 | 2 |
| 10% | lc (weak) | 0.786 | 0.786 | 0.537 | 2 |
| 10% | adaptive_blend | 0.971 | 0.785 | 0.183 | 2 |
| -- | wanet 1% | 0.044 | not implanted | -- | -- |

Mean AUROC at 1%: 0.775. Mean AUROC at 5%: 0.712. Mean AUROC at 10%: 0.756.

SCALE-UP struggles on CIFAR-100. Its clip-then-rescale mechanism was designed for pixel-space triggers (BadNet) and struggles with distributed triggers (Blend: 0.691 at 1%). The wanet inversion at 5% (0.351) and 10% (0.381) shows the method actively mislabels WaNet-poisoned samples as clean on CIFAR-100.

## Summary: CIFAR-100 mean AUROC by configuration

| Configuration | 1% | 5% | 10% | All |
|---|---:|---:|---:|---:|
| gain_scale @ mlp_norm_out | 0.945 | 0.884 | 0.954 | 0.930 |
| token_mask @ before_attention_norm | 0.849 | 0.855 | 0.916 | 0.879 |
| scale_up @ input_pixels | 0.775 | 0.712 | 0.756 | 0.744 |
| dropout @ before_attention_norm | 0.692 | 0.674 | 0.715 | 0.693 |
| dropout @ pre_residual | 0.687 | 0.766 | 0.778 | 0.754 |

gain_scale @ mlp_norm_out is the clear winner on CIFAR-100, especially at 1% where it leads by 0.096 AUROC over the second-best configuration.

## Source

`python defence_tables.py --operator {op} --position {pos} --allow-partial`
Benign: `vit_cifar100_benign` checkpoint probed with badnet_a2o trigger.
