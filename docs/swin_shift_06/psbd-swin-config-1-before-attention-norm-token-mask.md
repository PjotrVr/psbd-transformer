# PSBD-ViT: `before_attention_norm_token_mask`

The single placement `before_attention_norm_token_mask`.

Each member is read at **its own** perturbation rate, chosen so the clean-validation
shift ratio lands nearest **0.6**, so everything below is measured at
matched disturbance rather than at a shared nominal rate.

Columns: **ASR** attack success rate and **CA** clean accuracy of the poisoned model,
both from the checkpoint's own provenance; **CA benign** the same-dataset benign ViT and
**dCA** the difference, so an attack that buys success by wrecking the model is visible;
**AUROC**; and **TPR** at the 10% and 20% false-positive operating points.

Only cells whose attack actually implanted (ASR >= 0.85) appear. A `--` is an unmeasured
cell, never a failure.

# poison rate 10%

## cifar10, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.966 | -- | -- | -- | -- | -- |
| blend | easy | 1.000 | 0.970 | -- | -- | -- | -- | -- |
| lf | easy | 0.998 | 0.963 | -- | -- | -- | -- | -- |
| bpp | **hard** | 1.000 | 0.967 | -- | -- | -- | -- | -- |
| wanet | **hard** | 0.977 | 0.969 | -- | -- | -- | -- | -- |
| sig | **hard** | 0.945 | 0.869 | -- | -- | -- | -- | -- |
| lc | **hard** | 0.953 | 0.873 | -- | -- | -- | -- | -- |
| adaptive_blend | **hard** | 0.992 | 0.958 | -- | -- | -- | -- | -- |

## cifar100, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.860 | 0.867 | -0.007 | 0.995 | 0.999 | 1.000 |
| blend | easy | 1.000 | 0.854 | 0.867 | -0.013 | 0.994 | 1.000 | 1.000 |
| lf | easy | 0.998 | 0.860 | 0.867 | -0.007 | -- | -- | -- |
| bpp | **hard** | 0.998 | 0.862 | 0.867 | -0.005 | -- | -- | -- |
| wanet | **hard** | 0.925 | 0.862 | 0.867 | -0.005 | 0.623 | 0.076 | 0.239 |
| adaptive_blend | **hard** | 0.971 | 0.862 | 0.867 | -0.005 | 0.955 | 0.967 | 0.970 |

## gtsrb, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.983 | -- | -- | 1.000 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.984 | -- | -- | 0.999 | 0.998 | 1.000 |
| lf | easy | 0.995 | 0.984 | -- | -- | 0.998 | 0.994 | 0.997 |
| bpp | **hard** | 1.000 | 0.987 | -- | -- | 0.998 | 0.994 | 0.998 |
| wanet | **hard** | 0.955 | 0.982 | -- | -- | 0.919 | 0.832 | 0.956 |
| lc | **hard** | 0.991 | 0.985 | -- | -- | 0.992 | 0.993 | 0.996 |
| adaptive_blend | **hard** | 1.000 | 0.977 | -- | -- | 0.999 | 1.000 | 1.000 |

## tiny, poison rate 10%

No attack implanted at ASR >= 0.85 on tiny at 10%.

# poison rate 5%

## cifar10, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.965 | -- | -- | -- | -- | -- |
| blend | easy | 1.000 | 0.967 | -- | -- | -- | -- | -- |
| lf | easy | 0.998 | 0.961 | -- | -- | -- | -- | -- |
| bpp | **hard** | 0.999 | 0.965 | -- | -- | -- | -- | -- |
| wanet | **hard** | 0.978 | 0.964 | -- | -- | -- | -- | -- |
| adaptive_blend | **hard** | 0.865 | 0.971 | -- | -- | -- | -- | -- |

## cifar100, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.862 | 0.867 | -0.005 | 0.990 | 0.999 | 1.000 |
| blend | easy | 1.000 | 0.857 | 0.867 | -0.010 | 0.999 | 1.000 | 1.000 |
| lf | easy | 0.985 | 0.862 | 0.867 | -0.005 | -- | -- | -- |
| bpp | **hard** | 0.998 | 0.860 | 0.867 | -0.007 | -- | -- | -- |
| wanet | **hard** | 0.865 | 0.862 | 0.867 | -0.005 | 0.703 | 0.116 | 0.452 |
| adaptive_blend | **hard** | 0.912 | 0.869 | 0.867 | +0.002 | 0.915 | 0.912 | 0.913 |

## gtsrb, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.965 | -- | -- | 1.000 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.983 | -- | -- | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.990 | 0.981 | -- | -- | 0.983 | 0.956 | 0.974 |
| bpp | **hard** | 1.000 | 0.989 | -- | -- | 0.990 | 0.976 | 0.985 |
| wanet | **hard** | 0.861 | 0.972 | -- | -- | 0.578 | 0.040 | 0.149 |
| lc | **hard** | 0.992 | 0.983 | -- | -- | 0.994 | 0.992 | 0.992 |
| adaptive_blend | **hard** | 0.859 | 0.982 | -- | -- | 0.925 | 0.877 | 0.894 |

## tiny, poison rate 5%

No attack implanted at ASR >= 0.85 on tiny at 5%.

# poison rate 1%

## cifar10, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.967 | -- | -- | -- | -- | -- |
| blend | easy | 1.000 | 0.966 | -- | -- | -- | -- | -- |
| lf | easy | 0.994 | 0.966 | -- | -- | -- | -- | -- |
| bpp | **hard** | 0.993 | 0.970 | -- | -- | -- | -- | -- |

## cifar100, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.860 | 0.867 | -0.007 | 0.993 | 0.996 | 0.999 |
| blend | easy | 0.998 | 0.867 | 0.867 | +0.000 | 0.994 | 0.998 | 0.999 |
| lf | easy | 0.938 | 0.863 | 0.867 | -0.004 | -- | -- | -- |
| bpp | **hard** | 0.994 | 0.865 | 0.867 | -0.001 | -- | -- | -- |
| lc | **hard** | 0.996 | 0.853 | 0.867 | -0.014 | 0.995 | 0.996 | 0.996 |

## gtsrb, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.989 | -- | -- | 1.000 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.982 | -- | -- | 0.999 | 0.999 | 1.000 |
| lf | easy | 0.994 | 0.977 | -- | -- | 0.954 | 0.889 | 0.911 |
| bpp | **hard** | 0.976 | 0.983 | -- | -- | 0.967 | 0.907 | 0.931 |

## tiny, poison rate 1%

No attack implanted at ASR >= 0.85 on tiny at 1%.

# Summary

| subset | n | AUROC | TPR@10%FPR | TPR@20%FPR | achieved FPR at 10% |
|---|---:|---:|---:|---:|---:|
| all cells | 29 | 0.947 | 0.879 | 0.909 | 0.0965 |
| **hard** attacks | 14 | 0.897 | 0.763 | 0.819 | 0.0989 |
| easy attacks | 15 | 0.993 | 0.988 | 0.992 | 0.0943 |
| primary datasets (cifar100 + tiny) | 11 | 0.923 | 0.824 | 0.870 | 0.0995 |

| by poison rate | n | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---:|---:|---:|---:|
| 10% | 11 | 0.952 | 0.896 | 0.923 |
| 5% | 11 | 0.916 | 0.806 | 0.851 |
| 1% | 7 | 0.986 | 0.969 | 0.977 |

Worst-case cell AUROC **0.578**, inverted cells (AUROC < 0.5) **0** of 29.

