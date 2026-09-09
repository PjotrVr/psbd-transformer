# PSBD-ViT: `before_attention_norm_token_mask`

The single placement `before_attention_norm_token_mask`.

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
| badnet_a2o | easy | 1.000 | 0.942 | 0.953 | -0.011 | 0.991 | 0.935 | 1.000 |
| blend | easy | 1.000 | 0.951 | 0.953 | -0.001 | 0.906 | 0.754 | 0.849 |
| lf | easy | 0.999 | 0.951 | 0.953 | -0.002 | 0.993 | 0.995 | 0.999 |
| bpp | **hard** | 0.994 | 0.954 | 0.953 | +0.001 | 0.871 | 0.736 | 0.830 |
| wanet | **hard** | 0.890 | 0.946 | 0.953 | -0.007 | 0.459 | 0.087 | 0.268 |
| tact | **hard** | 1.000 | 0.858 | 0.953 | -0.095 | 0.963 | 0.030 | 0.206 |
| sig | **hard** | 0.901 | 0.846 | 0.953 | -0.106 | 0.673 | 0.346 | 0.408 |
| lc | **hard** | 0.979 | 0.861 | 0.953 | -0.092 | 0.921 | 0.772 | 0.909 |

## cifar100, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.821 | 0.811 | +0.010 | 0.989 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.816 | 0.811 | +0.006 | 0.998 | 1.000 | 1.000 |
| lf | easy | 0.995 | 0.823 | 0.811 | +0.012 | 0.995 | 0.995 | 0.995 |
| bpp | **hard** | 0.991 | 0.817 | 0.811 | +0.006 | 0.985 | 0.963 | 0.975 |
| tact | **hard** | 0.989 | 0.821 | 0.811 | +0.010 | 0.891 | 0.161 | 0.586 |

## gtsrb, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.972 | 0.991 | -0.019 | 1.000 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.988 | 0.991 | -0.003 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.999 | 0.988 | 0.991 | -0.003 | 0.998 | 0.998 | 0.999 |
| bpp | **hard** | 0.995 | 0.986 | 0.991 | -0.006 | 0.989 | 0.971 | 0.986 |
| wanet | **hard** | 0.947 | 0.958 | 0.991 | -0.033 | 0.948 | 0.949 | 0.952 |
| tact | **hard** | 1.000 | 0.931 | 0.991 | -0.060 | 0.750 | 0.069 | 0.626 |

## tiny, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.003 | 0.964 | 0.988 | 1.000 |
| blend | easy | 1.000 | 0.749 | 0.755 | -0.006 | 0.992 | 0.996 | 0.999 |
| lf | easy | 0.973 | 0.740 | 0.755 | -0.015 | 0.973 | 0.973 | 0.977 |
| bpp | **hard** | 0.996 | 0.757 | 0.755 | +0.002 | 0.984 | 0.981 | 0.991 |
| wanet | **hard** | 0.968 | 0.747 | 0.755 | -0.008 | 0.950 | 0.932 | 0.970 |
| tact | **hard** | 0.929 | 0.756 | 0.755 | +0.001 | 0.810 | 0.381 | 0.619 |

# poison rate 5%

## cifar10, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.945 | 0.953 | -0.008 | 0.992 | 0.965 | 0.995 |
| blend | easy | 1.000 | 0.957 | 0.953 | +0.004 | 0.991 | 0.995 | 0.999 |
| lf | easy | 0.997 | 0.944 | 0.953 | -0.009 | 0.991 | 0.988 | 0.995 |
| bpp | **hard** | 0.987 | 0.950 | 0.953 | -0.003 | 0.801 | 0.587 | 0.655 |
| wanet | **hard** | 0.961 | 0.913 | 0.953 | -0.039 | 0.937 | 0.838 | 0.913 |
| tact | **hard** | 0.996 | 0.946 | 0.953 | -0.007 | 0.901 | 0.019 | 0.109 |

## cifar100, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.977 | 0.995 | 1.000 |
| blend | easy | 1.000 | 0.830 | 0.811 | +0.019 | 0.989 | 0.977 | 0.991 |
| lf | easy | 0.953 | 0.829 | 0.811 | +0.019 | 0.957 | 0.949 | 0.955 |
| bpp | **hard** | 0.988 | 0.833 | 0.811 | +0.023 | 0.983 | 0.963 | 0.974 |
| tact | **hard** | 1.000 | 0.825 | 0.811 | +0.014 | 0.978 | 0.517 | 0.816 |

## gtsrb, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.986 | 0.991 | -0.005 | 1.000 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.977 | 0.991 | -0.014 | 0.999 | 1.000 | 1.000 |
| lf | easy | 0.999 | 0.974 | 0.991 | -0.018 | 0.996 | 0.991 | 0.998 |
| bpp | **hard** | 0.991 | 0.978 | 0.991 | -0.013 | 0.992 | 0.982 | 0.988 |
| tact | **hard** | 1.000 | 0.986 | 0.991 | -0.006 | 0.942 | 0.502 | 0.818 |

## tiny, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.002 | 0.984 | 0.999 | 1.000 |
| blend | easy | 0.999 | 0.754 | 0.755 | -0.001 | 0.989 | 0.996 | 0.998 |
| lf | easy | 0.987 | 0.753 | 0.755 | -0.001 | 0.976 | 0.982 | 0.987 |
| bpp | **hard** | 0.985 | 0.746 | 0.755 | -0.009 | 0.987 | 0.984 | 0.986 |
| wanet | **hard** | 0.922 | 0.748 | 0.755 | -0.007 | 0.922 | 0.904 | 0.930 |
| tact | **hard** | 0.952 | 0.743 | 0.755 | -0.012 | 0.736 | 0.452 | 0.595 |

# poison rate 1%

## cifar10, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.997 | 0.952 | 0.953 | -0.001 | 0.982 | 0.959 | 0.978 |
| blend | easy | 1.000 | 0.954 | 0.953 | +0.002 | 0.922 | 0.773 | 0.869 |
| lf | easy | 0.982 | 0.953 | 0.953 | +0.000 | 0.980 | 0.968 | 0.978 |
| bpp | **hard** | 0.982 | 0.952 | 0.953 | -0.000 | 0.986 | 0.956 | 0.976 |
| tact | **hard** | 0.985 | 0.938 | 0.953 | -0.015 | 0.979 | 0.006 | 0.009 |

## cifar100, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.988 | 0.999 | 1.000 |
| blend | easy | 0.989 | 0.829 | 0.811 | +0.019 | 0.976 | 0.959 | 0.983 |
| lf | easy | 0.942 | 0.830 | 0.811 | +0.019 | 0.956 | 0.942 | 0.947 |
| bpp | **hard** | 0.914 | 0.824 | 0.811 | +0.013 | 0.859 | 0.555 | 0.796 |
| tact | **hard** | 0.989 | 0.820 | 0.811 | +0.009 | 0.959 | 0.678 | 0.839 |
| lc | **hard** | 0.873 | 0.823 | 0.811 | +0.013 | 0.786 | 0.185 | 0.601 |

## gtsrb, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.994 | 0.991 | +0.003 | 0.999 | 0.996 | 0.999 |
| blend | easy | 1.000 | 0.991 | 0.991 | +0.000 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.979 | 0.990 | 0.991 | -0.001 | 0.976 | 0.926 | 0.973 |
| bpp | **hard** | 0.868 | 0.986 | 0.991 | -0.005 | 0.871 | 0.658 | 0.766 |

## tiny, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.999 | 0.752 | 0.755 | -0.003 | 0.932 | 0.857 | 0.991 |
| blend | easy | 0.999 | 0.760 | 0.755 | +0.005 | 0.991 | 0.987 | 0.994 |
| lf | easy | 0.922 | 0.762 | 0.755 | +0.007 | 0.930 | 0.914 | 0.932 |
| bpp | **hard** | 0.971 | 0.748 | 0.755 | -0.007 | 0.980 | 0.972 | 0.975 |
| tact | **hard** | 0.976 | 0.750 | 0.755 | -0.005 | 0.575 | 0.024 | 0.119 |

# Summary

| subset | n | AUROC | TPR@10%FPR | TPR@20%FPR | achieved FPR at 10% |
|---|---:|---:|---:|---:|---:|
| all cells | 67 | 0.935 | 0.790 | 0.859 | 0.0884 |
| **hard** attacks | 31 | 0.883 | 0.586 | 0.716 | 0.0786 |
| easy attacks | 36 | 0.980 | 0.965 | 0.983 | 0.0969 |
| primary datasets (cifar100 + tiny) | 33 | 0.938 | 0.823 | 0.895 | 0.0873 |

| by poison rate | n | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---:|---:|---:|---:|
| 10% | 25 | 0.920 | 0.760 | 0.846 |
| 5% | 22 | 0.956 | 0.845 | 0.896 |
| 1% | 20 | 0.931 | 0.766 | 0.836 |

Worst-case cell AUROC **0.459**, inverted cells (AUROC < 0.5) **1** of 67.

