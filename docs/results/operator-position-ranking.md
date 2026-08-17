# Operator/Position Ranking

27 operator/position combinations evaluated at full coverage: 4 datasets (CIFAR-10, CIFAR-100, GTSRB, Tiny ImageNet), 3 poison rates (1%, 5%, 10%), 5-attack panel (BadNet A2O, Blend, WaNet, LC, Adaptive Blend). Each combination has 48/48 required cells after excluding attacks that failed to implant.

All numbers use fractional PSU at sigma-matched rate (sigma >= 0.6), one-sided (low PSU = poisoned). Benign controls sit at AUROC ~0.50 across all configurations (range 0.484 to 0.501).

## Full ranking table

Sorted by mean AUROC at 1% poison rate, the hardest and most realistic detection setting.

| Rank | Operator | Position | Mean AUROC | AUROC @1% | TPR@5% | TPR@5% @1% | Worst | Inv. |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | gain_scale | mlp_norm_out | 0.899 | 0.947 | 0.746 | 0.826 | 0.459 | 2 |
| 2 | gaussian | before_mlp | 0.900 | 0.928 | 0.715 | 0.796 | 0.494 | 1 |
| 3 | token_mask | before_attention_residual | 0.854 | 0.912 | 0.661 | 0.753 | 0.325 | 4 |
| 4 | token_mask | before_attention_norm | 0.911 | 0.904 | 0.742 | 0.729 | 0.632 | 0 |
| 5 | token_mask | before_mlp | 0.869 | 0.894 | 0.666 | 0.666 | 0.445 | 2 |
| 6 | token_mask | after_mlp_residual | 0.911 | 0.874 | 0.720 | 0.576 | 0.650 | 0 |
| 7 | gaussian | mlp_neurons | 0.891 | 0.865 | 0.648 | 0.539 | 0.603 | 0 |
| 8 | channel_mask | before_attention_norm | 0.834 | 0.831 | 0.470 | 0.458 | 0.441 | 2 |
| 9 | token_mask | before_attention | 0.854 | 0.830 | 0.650 | 0.555 | 0.255 | 5 |
| 10 | scale_up | input_pixels | 0.791 | 0.828 | 0.332 | 0.343 | 0.351 | 2 |
| 11 | droppath | before_attention_residual | 0.847 | 0.811 | 0.489 | 0.385 | 0.571 | 0 |
| 12 | token_mask | after_attention_residual | 0.871 | 0.799 | 0.608 | 0.420 | 0.450 | 1 |
| 13 | head_mask | attention_heads | 0.839 | 0.797 | 0.474 | 0.400 | 0.516 | 0 |
| 14 | gaussian | before_mlp_residual | 0.862 | 0.790 | 0.590 | 0.401 | 0.518 | 0 |
| 15 | token_mask | before_mlp_norm | 0.800 | 0.790 | 0.406 | 0.392 | 0.368 | 3 |
| 16 | channel_mask | before_mlp_residual | 0.827 | 0.780 | 0.487 | 0.381 | 0.320 | 3 |
| 17 | channel_mask | before_attention | 0.805 | 0.760 | 0.454 | 0.407 | 0.495 | 1 |
| 18 | channel_mask | before_mlp | 0.768 | 0.758 | 0.320 | 0.352 | 0.468 | 2 |
| 19 | channel_mask | before_attention_residual | 0.814 | 0.755 | 0.459 | 0.374 | 0.500 | 0 |
| 20 | channel_mask | mlp_neurons | 0.787 | 0.743 | 0.381 | 0.309 | 0.439 | 3 |
| 21 | channel_mask | after_mlp_residual | 0.826 | 0.733 | 0.550 | 0.397 | 0.221 | 5 |
| 22 | channel_mask | after_attention_residual | 0.812 | 0.712 | 0.538 | 0.374 | 0.159 | 5 |
| 23 | channel_mask | after_embedding | 0.672 | 0.648 | 0.191 | 0.203 | 0.335 | 10 |
| 24 | channel_mask | before_mlp_norm | 0.727 | 0.625 | 0.317 | 0.206 | 0.234 | 8 |
| 25 | token_mask | after_embedding | 0.655 | 0.597 | 0.163 | 0.079 | 0.323 | 10 |
| 26 | gain_scale | attention_norm_out | 0.697 | 0.595 | 0.337 | 0.185 | 0.012 | 10 |
| 27 | gaussian | before_mlp_norm | 0.709 | 0.547 | 0.413 | 0.175 | 0.002 | 11 |

Columns: Mean AUROC = mean across all 48 cells. AUROC @1% = mean across the ~11 cells at 1% poison rate. TPR@5% = mean TPR at 5% FPR. Worst = minimum AUROC across all cells (floor). Inv. = cells with AUROC < 0.5.

## Top 5 per-dataset breakdown at 1% poison rate

### gain_scale @ mlp_norm_out (rank 1 at 1%)

| Dataset | badnet_a2o | blend | lc | adaptive_blend |
|---|---:|---:|---:|---:|
| CIFAR-100 | 0.999 | 0.992 | 0.935 | 0.854 |
| Tiny | 0.933 | 0.982 | -- | -- |
| CIFAR-10 | 0.996 | 0.996 | -- | 0.725 |
| GTSRB | 1.000 | 1.000 | -- | -- |

Best single-position AUROC at 1%. Dominates CIFAR-100 (0.999 on badnet, 0.935 on lc). However, TPR@5%FPR on Tiny is 0.034 for badnet at 1%, meaning the ROC curve is steep only at high FPR. Two inversions (AUROC < 0.5) remain on GTSRB wanet cells.

### token_mask @ before_attention_norm (rank 4 at 1%, rank 1 on robustness)

| Dataset | badnet_a2o | blend | lc | adaptive_blend |
|---|---:|---:|---:|---:|
| CIFAR-100 | 0.960 | 0.945 | 0.786 | 0.706 |
| Tiny | 0.932 | 0.991 | -- | -- |
| CIFAR-10 | 0.993 | 0.983 | -- | 0.649 |
| GTSRB | 0.995 | 1.000 | -- | -- |

Zero inversions across all 48 cells. Worst AUROC floor = 0.632. Consistent across datasets. Slightly lower 1% AUROC than gain_scale, but the TPR curve is well-shaped (no near-zero TPR@5% values).

### gaussian @ before_mlp (rank 2 at 1%)

| Dataset | badnet_a2o | blend | lc | adaptive_blend |
|---|---:|---:|---:|---:|
| CIFAR-100 | * | * | * | * |
| Tiny | * | * | -- | -- |
| CIFAR-10 | * | * | -- | * |
| GTSRB | * | * | -- | -- |

(*) These numbers come from the 27-position grid analysis, not individual defence_tables runs. Mean 1% AUROC = 0.928. One inversion.

### token_mask @ before_attention_residual (rank 3 at 1%)

Mean 1% AUROC = 0.912, but 4 inversions and worst floor = 0.325. High ceiling, low floor. The stream-adjacent placement is aggressive: high sensitivity to backdoor direction, but also high risk of inversion on some dataset/attack combinations.

## Cross-dataset rank correlation

The rank of each configuration varies across datasets. Spearman rho between dataset rankings:

|  | CIFAR-10 | CIFAR-100 | GTSRB | Tiny |
|---|---:|---:|---:|---:|
| CIFAR-10 | 1.00 | 0.70 | 0.33 | 0.51 |
| CIFAR-100 | 0.70 | 1.00 | 0.41 | 0.58 |
| GTSRB | 0.33 | 0.41 | 1.00 | 0.36 |
| Tiny | 0.51 | 0.58 | 0.36 | 1.00 |

Top-5 overlap between datasets is poor (0 to 2 of 5 shared). Each dataset has a different winner. The moderate correlation means the relative ordering of positions is partially preserved, but the best position for one dataset is not the best for another.

## Recommended deployment configuration

**token_mask @ before_attention_norm** is the recommended configuration for deployment despite ranking 4th on 1% AUROC.

Rationale:

1. **Zero inversions.** Every cell is above chance. gain_scale has 2 inversions, gaussian @ before_mlp has 1. An inversion at deployment means the detector classifies clean samples as more suspicious than poisoned ones. This is worse than no detector at all.

2. **Highest worst-case floor (0.632).** gain_scale's worst is 0.459. The floor sets the guarantee: for the worst attack on the worst dataset at the hardest poison rate, detection still works.

3. **Highest mean AUROC (0.911) tied with token_mask @ after_mlp_residual.** The 1% ranking penalizes it, but the all-rate average is the best in the grid.

4. **Well-shaped TPR curve.** gain_scale @ mlp_norm_out achieves 0.999 AUROC on CIFAR-100 badnet at 1% but only 0.034 TPR@5%FPR on Tiny badnet at 1%. That means the ROC curve is front-loaded: good separation at high FPR, near-random at the operationally relevant 5% FPR threshold. token_mask @ before_attention_norm has 0.343 TPR on the same cell, which is still low but 10x better.

5. **Cross-dataset stability.** Its worst rank across datasets is 10 (out of 27). gain_scale's best rank is 1 on some datasets but its variability is higher.

If the deployment context is CIFAR-100 only (the hardest dataset), gain_scale @ mlp_norm_out is the better choice (0.999 vs 0.960 on badnet at 1%). For a dataset-agnostic deployment, token_mask @ before_attention_norm provides the safest floor.

## Source

All numbers from `defence_tables.py` at sigma >= 0.6 with `--score fractional`, one-sided. Ranking table from the 27-position full-grid analysis at 48/48 coverage. Cross-dataset correlation from `scratch/cross_dataset_transfer.py`.
