# Best placement per attack, ViT-B/16, 2026-09-11

Read-only note for the authors, not for the paper. Models are the ViT-B/16 checkpoints on CIFAR-10, CIFAR-100, GTSRB and Tiny ImageNet whose attack success is at least 0.85. Every AUROC is the fractional PSU reading (the paper's statistic) at the rate the 0.8 shift ratio rule selects, the same field `scripts/paper/tab_headline.py` reads. Reach is the share of those models on which the placement's rate ladder reaches a clean validation shift ratio of 0.8 at all. A placement that never reaches it on a model is not read on that model, which is why n can be below the model count.

## BadNets (12 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| gain scale, MLP norm output | 0.995 | 12 | 1.00 |
| token mask, attention input | 0.992 | 12 | 1.00 |
| token mask, attention input, blocks 5 to 8 | 0.982 | 6 | 0.50 |
| scale up, input pixels | 0.979 | 12 | 1.00 |
| token mask, attention output before the add | 0.973 | 12 | 1.00 |
| token mask, both sublayer inputs | 0.941 | 12 | 1.00 |
| dropout, after both residual adds (reference) | 0.712 | 12 | 1.00 |

## Blend (12 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| dropout, attention input after norm | 0.999 | 3 | 1.00 |
| gain scale, MLP norm output | 0.996 | 12 | 1.00 |
| dropout, embedding output | 0.991 | 3 | 1.00 |
| token mask, attention output before the add | 0.991 | 12 | 1.00 |
| token mask, attention input, blocks 5 to 8 | 0.983 | 11 | 0.92 |
| channel mask, attention input | 0.982 | 12 | 1.00 |
| token mask, attention input (reference) | 0.978 | 12 | 1.00 |
| dropout, after both residual adds (reference) | 0.971 | 12 | 1.00 |

## BPP (12 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| token mask, attention input, blocks 9 to 12 | 0.974 | 6 | 0.50 |
| dropout, before both residual adds, blocks 9 to 12 | 0.961 | 9 | 0.75 |
| token mask, attention input | 0.948 | 12 | 1.00 |
| dropout, after both residual adds | 0.945 | 12 | 1.00 |
| token mask, attention output before the add | 0.939 | 12 | 1.00 |
| dropout, before both residual adds, blocks 5 to 8 | 0.915 | 12 | 1.00 |

## LF (12 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| token mask, attention input | 0.980 | 12 | 1.00 |
| token mask, attention output before the add | 0.977 | 12 | 1.00 |
| token mask, attention input, blocks 9 to 12 | 0.975 | 6 | 0.50 |
| dropout, before both residual adds, blocks 5 to 8 | 0.974 | 12 | 1.00 |
| dropout, after both residual adds | 0.974 | 12 | 1.00 |
| dropout, before both residual adds, blocks 9 to 12 | 0.972 | 9 | 0.75 |

## SIG (1 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| token mask, attention output before the add | 0.934 | 1 | 1.00 |
| token mask, MLP input | 0.928 | 1 | 1.00 |
| dropout, after both residual adds | 0.919 | 1 | 1.00 |
| dropout, before both residual adds, blocks 5 to 8 | 0.907 | 1 | 1.00 |
| dropout, before both residual adds | 0.896 | 1 | 1.00 |
| dropout, attention input | 0.896 | 1 | 1.00 |
| token mask, attention input (reference) | 0.418 | 1 | 1.00 |

## TaCT (11 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| token mask, attention output before the add | 0.870 | 11 | 1.00 |
| token mask, attention input, blocks 5 to 8 | 0.863 | 9 | 0.82 |
| token mask, attention input | 0.853 | 11 | 1.00 |
| scale up, input pixels | 0.818 | 11 | 1.00 |
| token mask, both sublayer inputs | 0.773 | 11 | 1.00 |
| gain scale, MLP norm output | 0.759 | 11 | 1.00 |
| dropout, after both residual adds (reference) | 0.470 | 11 | 1.00 |

## WaNet (5 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| dropout, before both residual adds, blocks 9 to 12 | 0.964 | 4 | 0.80 |
| dropout, after both residual adds | 0.956 | 5 | 1.00 |
| dropout, before both residual adds, blocks 5 to 8 | 0.951 | 5 | 1.00 |
| token mask, attention input, blocks 9 to 12 | 0.944 | 2 | 0.40 |
| dropout, before both residual adds | 0.938 | 5 | 1.00 |
| token mask, both sublayer inputs | 0.903 | 5 | 1.00 |
| token mask, attention input (reference) | 0.845 | 5 | 1.00 |

## Placements whose rate ladder does not reach a shift ratio of 0.8 on every model

| Placement | Reach 0.8 | n | Median of the highest shift ratio the ladder reaches |
|---|---|---|---|
| token mask, attention input, blocks 9 to 12 | 0.48 | 65 | 0.78 |
| token mask, attention input, blocks 5 to 8 | 0.68 | 65 | 0.86 |
| dropout, before both residual adds, blocks 9 to 12 | 0.71 | 65 | 0.95 |

Every placement that acts in all 12 blocks reaches 0.8 on every model. Only the banded placements fall short, because masking or dropping in 4 of 12 blocks at the highest rate of the ladder does not move 80% of clean predictions on every model.

Note: the `adaptive` block of `psbd_metrics.json` stores the absolute PSU reading, not the fractional one, so a reader of that file must take the rate row named by `adaptive_rate` and its `detection_psu_ratio` field to match the paper.
