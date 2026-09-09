# PSBD-ViT: merged top two at sigma 0.8 (before_attention_norm_token_mask + before_attention_residual_token_mask)

A **fused** configuration. Its members are swept separately and combined at the score
level with the min-rank rule against the clean-validation reference, so a sample is
flagged when ANY member finds it suspicious. This needs no extra training and no extra
checkpoint, only one more perturbation sweep per member at inference time.

Members:

- `before_attention_norm_token_mask`
- `before_attention_residual_token_mask`

Each member is read at **its own** perturbation rate, chosen so the clean-validation
shift ratio lands nearest **0.8**, so everything below is measured at
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
| badnet_a2o | easy | 1.000 | 0.942 | 0.953 | -0.011 | 0.976 | 0.869 | 0.983 |
| blend | easy | 1.000 | 0.951 | 0.953 | -0.001 | 0.998 | 0.999 | 1.000 |
| lf | easy | 0.999 | 0.951 | 0.953 | -0.002 | 0.992 | 0.998 | 0.999 |
| bpp | **hard** | 0.994 | 0.954 | 0.953 | +0.001 | 0.837 | 0.684 | 0.745 |
| wanet | **hard** | 0.890 | 0.946 | 0.953 | -0.007 | 0.516 | 0.105 | 0.199 |
| tact | **hard** | 1.000 | 0.858 | 0.953 | -0.095 | 0.988 | 0.397 | 0.629 |
| sig | **hard** | 0.901 | 0.846 | 0.953 | -0.106 | 0.920 | 0.858 | 0.890 |
| lc | **hard** | 0.979 | 0.861 | 0.953 | -0.092 | 0.962 | 0.922 | 0.966 |

## cifar100, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.821 | 0.811 | +0.010 | 0.982 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.816 | 0.811 | +0.006 | 0.999 | 1.000 | 1.000 |
| lf | easy | 0.995 | 0.823 | 0.811 | +0.012 | 0.995 | 0.995 | 0.995 |
| bpp | **hard** | 0.991 | 0.817 | 0.811 | +0.006 | 0.983 | 0.960 | 0.975 |
| tact | **hard** | 0.989 | 0.821 | 0.811 | +0.010 | 0.971 | 0.713 | 0.874 |

## gtsrb, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.972 | 0.991 | -0.019 | 1.000 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.988 | 0.991 | -0.003 | 0.999 | 1.000 | 1.000 |
| lf | easy | 0.999 | 0.988 | 0.991 | -0.003 | 0.999 | 0.999 | 0.999 |
| bpp | **hard** | 0.995 | 0.986 | 0.991 | -0.006 | 0.996 | 0.990 | 0.996 |
| wanet | **hard** | 0.947 | 0.958 | 0.991 | -0.033 | 0.931 | 0.949 | 0.957 |
| tact | **hard** | 1.000 | 0.931 | 0.991 | -0.060 | 0.743 | 0.390 | 0.866 |

## tiny, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.003 | 0.991 | 0.998 | 1.000 |
| blend | easy | 1.000 | 0.749 | 0.755 | -0.006 | 0.993 | 0.996 | 0.999 |
| lf | easy | 0.973 | 0.740 | 0.755 | -0.015 | 0.967 | 0.965 | 0.977 |
| bpp | **hard** | 0.996 | 0.757 | 0.755 | +0.002 | 0.980 | 0.969 | 0.989 |
| wanet | **hard** | 0.968 | 0.747 | 0.755 | -0.008 | 0.934 | 0.808 | 0.961 |
| tact | **hard** | 0.929 | 0.756 | 0.755 | +0.001 | 0.784 | 0.214 | 0.500 |

# poison rate 5%

## cifar10, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.945 | 0.953 | -0.008 | 0.990 | 0.980 | 0.997 |
| blend | easy | 1.000 | 0.957 | 0.953 | +0.004 | 0.992 | 1.000 | 1.000 |
| lf | easy | 0.997 | 0.944 | 0.953 | -0.009 | 0.990 | 0.990 | 0.995 |
| bpp | **hard** | 0.987 | 0.950 | 0.953 | -0.003 | 0.893 | 0.737 | 0.820 |
| wanet | **hard** | 0.961 | 0.913 | 0.953 | -0.039 | 0.939 | 0.789 | 0.907 |
| tact | **hard** | 0.996 | 0.946 | 0.953 | -0.007 | 0.978 | 0.172 | 0.539 |

## cifar100, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.984 | 0.995 | 1.000 |
| blend | easy | 1.000 | 0.830 | 0.811 | +0.019 | 0.994 | 0.992 | 0.999 |
| lf | easy | 0.953 | 0.829 | 0.811 | +0.019 | 0.959 | 0.952 | 0.954 |
| bpp | **hard** | 0.988 | 0.833 | 0.811 | +0.023 | 0.980 | 0.959 | 0.974 |
| tact | **hard** | 1.000 | 0.825 | 0.811 | +0.014 | 0.974 | 0.977 | 1.000 |

## gtsrb, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.986 | 0.991 | -0.005 | 1.000 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.977 | 0.991 | -0.014 | 0.998 | 1.000 | 1.000 |
| lf | easy | 0.999 | 0.974 | 0.991 | -0.018 | 0.996 | 0.998 | 0.999 |
| bpp | **hard** | 0.991 | 0.978 | 0.991 | -0.013 | 0.995 | 0.992 | 0.993 |
| tact | **hard** | 1.000 | 0.986 | 0.991 | -0.006 | 0.994 | 0.598 | 0.889 |

## tiny, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.002 | 0.987 | 1.000 | 1.000 |
| blend | easy | 0.999 | 0.754 | 0.755 | -0.001 | 0.998 | 0.998 | 0.999 |
| lf | easy | 0.987 | 0.753 | 0.755 | -0.001 | 0.976 | 0.984 | 0.988 |
| bpp | **hard** | 0.985 | 0.746 | 0.755 | -0.009 | 0.983 | 0.982 | 0.986 |
| wanet | **hard** | 0.922 | 0.748 | 0.755 | -0.007 | 0.924 | 0.851 | 0.930 |
| tact | **hard** | 0.952 | 0.743 | 0.755 | -0.012 | 0.762 | 0.429 | 0.619 |

# poison rate 1%

## cifar10, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.997 | 0.952 | 0.953 | -0.001 | 0.992 | 0.987 | 0.996 |
| blend | easy | 1.000 | 0.954 | 0.953 | +0.002 | 0.995 | 1.000 | 1.000 |
| lf | easy | 0.982 | 0.953 | 0.953 | +0.000 | 0.981 | 0.970 | 0.978 |
| bpp | **hard** | 0.982 | 0.952 | 0.953 | -0.000 | 0.980 | 0.935 | 0.964 |
| tact | **hard** | 0.985 | 0.938 | 0.953 | -0.015 | 0.954 | 0.148 | 0.374 |

## cifar100, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.983 | 0.998 | 1.000 |
| blend | easy | 0.989 | 0.829 | 0.811 | +0.019 | 0.962 | 0.896 | 0.969 |
| lf | easy | 0.942 | 0.830 | 0.811 | +0.019 | 0.957 | 0.941 | 0.946 |
| bpp | **hard** | 0.914 | 0.824 | 0.811 | +0.013 | 0.901 | 0.759 | 0.901 |
| tact | **hard** | 0.989 | 0.820 | 0.811 | +0.009 | 0.942 | 0.690 | 0.851 |
| lc | **hard** | 0.873 | 0.823 | 0.811 | +0.013 | 0.898 | 0.737 | 0.871 |

## gtsrb, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.994 | 0.991 | +0.003 | 1.000 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.991 | 0.991 | +0.000 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.979 | 0.990 | 0.991 | -0.001 | 0.984 | 0.967 | 0.983 |
| bpp | **hard** | 0.868 | 0.986 | 0.991 | -0.005 | 0.933 | 0.871 | 0.903 |

## tiny, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.999 | 0.752 | 0.755 | -0.003 | 0.980 | 0.980 | 0.999 |
| blend | easy | 0.999 | 0.760 | 0.755 | +0.005 | 0.990 | 0.991 | 0.996 |
| lf | easy | 0.922 | 0.762 | 0.755 | +0.007 | 0.930 | 0.909 | 0.928 |
| bpp | **hard** | 0.971 | 0.748 | 0.755 | -0.007 | 0.980 | 0.971 | 0.975 |
| tact | **hard** | 0.976 | 0.750 | 0.755 | -0.005 | 0.689 | 0.024 | 0.214 |

# Summary

| subset | n | AUROC | TPR@10%FPR | TPR@20%FPR | achieved FPR at 10% |
|---|---:|---:|---:|---:|---:|
| all cells | 67 | 0.952 | 0.850 | 0.909 | 0.0852 |
| **hard** attacks | 31 | 0.911 | 0.696 | 0.815 | 0.0781 |
| easy attacks | 36 | 0.986 | 0.982 | 0.991 | 0.0914 |
| primary datasets (cifar100 + tiny) | 33 | 0.949 | 0.868 | 0.920 | 0.0887 |

| by poison rate | n | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---:|---:|---:|---:|
| 10% | 25 | 0.937 | 0.831 | 0.900 |
| 5% | 22 | 0.968 | 0.881 | 0.936 |
| 1% | 20 | 0.952 | 0.839 | 0.892 |

Worst-case cell AUROC **0.516**, inverted cells (AUROC < 0.5) **0** of 67.

