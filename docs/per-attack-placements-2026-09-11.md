# Best placement per attack, ViT-B/16, 2026-09-11

Read-only note for the authors, not for the paper. Models are the ViT-B/16 checkpoints on CIFAR-10, CIFAR-100, GTSRB and Tiny ImageNet whose attack success is at least 0.85. Every AUROC is read at the rate the 0.8 shift ratio rule selects, and reach is the share of those models on which the placement's rate ladder reaches a clean validation shift ratio of 0.8 at all. A placement that never reaches it on a model is not read on that model, which is why n can be below the model count.

## BadNets (12 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| token mask, attention input, blocks 5 to 8 | 0.982 | 6 | 0.50 |
| token mask, attention input | 0.981 | 12 | 1.00 |
| scale up, input pixels | 0.976 | 12 | 1.00 |
| gain scale, MLP norm output | 0.953 | 12 | 1.00 |
| token mask, attention output before the add | 0.936 | 12 | 1.00 |
| token mask, both sublayer inputs | 0.896 | 12 | 1.00 |
| dropout, after both residual adds (reference) | 0.574 | 12 | 1.00 |

## Blend (12 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| dropout, attention input after norm | 0.997 | 3 | 1.00 |
| gain scale, MLP norm output | 0.990 | 12 | 1.00 |
| dropout, embedding output | 0.988 | 3 | 1.00 |
| token mask, attention input, blocks 5 to 8 | 0.981 | 11 | 0.92 |
| token mask, attention output before the add | 0.978 | 12 | 1.00 |
| token mask, attention input | 0.965 | 12 | 1.00 |
| dropout, after both residual adds (reference) | 0.899 | 12 | 1.00 |

## BPP (12 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| token mask, attention input, blocks 9 to 12 | 0.967 | 6 | 0.50 |
| token mask, attention input | 0.942 | 12 | 1.00 |
| token mask, attention output before the add | 0.929 | 12 | 1.00 |
| dropout, before both residual adds, blocks 9 to 12 | 0.926 | 9 | 0.75 |
| dropout, after both residual adds | 0.907 | 12 | 1.00 |
| noise, MLP input after norm | 0.900 | 12 | 1.00 |

## LF (12 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| token mask, attention input | 0.984 | 12 | 1.00 |
| token mask, attention input, blocks 9 to 12 | 0.976 | 6 | 0.50 |
| token mask, attention output before the add | 0.973 | 12 | 1.00 |
| token mask, both sublayer inputs | 0.955 | 12 | 1.00 |
| noise, MLP input after norm | 0.939 | 12 | 1.00 |
| dropout, embedding output | 0.930 | 3 | 1.00 |
| dropout, after both residual adds (reference) | 0.909 | 12 | 1.00 |

## SIG (1 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| token mask, attention output before the add | 0.965 | 1 | 1.00 |
| token mask, MLP input | 0.964 | 1 | 1.00 |
| dropout, attention input | 0.908 | 1 | 1.00 |
| dropout, after both residual adds | 0.900 | 1 | 1.00 |
| channel mask, attention input | 0.893 | 1 | 1.00 |
| channel mask, MLP neurons | 0.879 | 1 | 1.00 |
| token mask, attention input (reference) | 0.456 | 1 | 1.00 |

## TaCT (11 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| token mask, attention input, blocks 5 to 8 | 0.789 | 9 | 0.82 |
| token mask, attention output before the add | 0.762 | 11 | 1.00 |
| scale up, input pixels | 0.707 | 11 | 1.00 |
| token mask, attention input | 0.604 | 11 | 1.00 |
| noise, MLP input after norm | 0.569 | 11 | 1.00 |
| token mask, both sublayer inputs | 0.525 | 11 | 1.00 |
| dropout, after both residual adds (reference) | 0.253 | 11 | 1.00 |

## WaNet (5 models)

| Placement | Mean AUROC | n | Reach 0.8 |
|---|---|---|---|
| token mask, attention input, blocks 9 to 12 | 0.973 | 2 | 0.40 |
| dropout, before both residual adds, blocks 5 to 8 | 0.958 | 5 | 1.00 |
| dropout, before both residual adds, blocks 9 to 12 | 0.957 | 4 | 0.80 |
| dropout, after both residual adds | 0.920 | 5 | 1.00 |
| token mask, both sublayer inputs | 0.877 | 5 | 1.00 |
| noise, attention input | 0.850 | 5 | 1.00 |
| token mask, attention input (reference) | 0.846 | 5 | 1.00 |

## Placements whose rate ladder does not reach a shift ratio of 0.8 on every model

| Placement | Reach 0.8 | n | Median of the highest shift ratio the ladder reaches |
|---|---|---|---|
| token mask, attention input, blocks 9 to 12 | 0.48 | 65 | 0.78 |
| token mask, attention input, blocks 5 to 8 | 0.68 | 65 | 0.86 |
| dropout, before both residual adds, blocks 9 to 12 | 0.71 | 65 | 0.95 |

Every placement that acts in all 12 blocks reaches 0.8 on every model. Only the banded placements fall short, because masking or dropping in 4 of 12 blocks at the highest rate of the ladder does not move 80% of clean predictions on every model.
