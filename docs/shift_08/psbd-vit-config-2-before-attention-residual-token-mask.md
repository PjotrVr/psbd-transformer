# PSBD-ViT: `before_attention_residual_token_mask`

The single placement `before_attention_residual_token_mask`.

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
| badnet_a2o | easy | 1.000 | 0.942 | 0.953 | -0.011 | 0.957 | 0.839 | 0.954 |
| blend | easy | 1.000 | 0.951 | 0.953 | -0.001 | 0.999 | 1.000 | 1.000 |
| lf | easy | 0.999 | 0.951 | 0.953 | -0.002 | 0.995 | 0.998 | 0.999 |
| bpp | **hard** | 0.994 | 0.954 | 0.953 | +0.001 | 0.342 | 0.072 | 0.127 |
| wanet | **hard** | 0.890 | 0.946 | 0.953 | -0.007 | 0.527 | 0.122 | 0.199 |
| tact | **hard** | 1.000 | 0.858 | 0.953 | -0.095 | 0.987 | 0.550 | 0.799 |
| sig | **hard** | 0.901 | 0.846 | 0.953 | -0.106 | 0.915 | 0.879 | 0.901 |
| lc | **hard** | 0.979 | 0.861 | 0.953 | -0.092 | 0.967 | 0.943 | 0.976 |

## cifar100, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.821 | 0.811 | +0.010 | 0.942 | 0.836 | 0.933 |
| blend | easy | 1.000 | 0.816 | 0.811 | +0.006 | 0.999 | 1.000 | 1.000 |
| lf | easy | 0.995 | 0.823 | 0.811 | +0.012 | 0.965 | 0.927 | 0.969 |
| bpp | **hard** | 0.991 | 0.817 | 0.811 | +0.006 | 0.956 | 0.918 | 0.953 |
| tact | **hard** | 0.989 | 0.821 | 0.811 | +0.010 | 0.973 | 0.862 | 0.931 |

## gtsrb, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.972 | 0.991 | -0.019 | 0.999 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.988 | 0.991 | -0.003 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.999 | 0.988 | 0.991 | -0.003 | 0.999 | 0.999 | 0.999 |
| bpp | **hard** | 0.995 | 0.986 | 0.991 | -0.006 | 0.991 | 0.985 | 0.992 |
| wanet | **hard** | 0.947 | 0.958 | 0.991 | -0.033 | 0.707 | 0.073 | 0.398 |
| tact | **hard** | 1.000 | 0.931 | 0.991 | -0.060 | 0.891 | 0.608 | 0.917 |

## tiny, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.003 | 0.993 | 0.996 | 0.999 |
| blend | easy | 1.000 | 0.749 | 0.755 | -0.006 | 0.995 | 0.997 | 0.999 |
| lf | easy | 0.973 | 0.740 | 0.755 | -0.015 | 0.940 | 0.886 | 0.942 |
| bpp | **hard** | 0.996 | 0.757 | 0.755 | +0.002 | 0.964 | 0.963 | 0.985 |
| wanet | **hard** | 0.968 | 0.747 | 0.755 | -0.008 | 0.735 | 0.268 | 0.521 |
| tact | **hard** | 0.929 | 0.756 | 0.755 | +0.001 | 0.711 | 0.190 | 0.405 |

# poison rate 5%

## cifar10, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.945 | 0.953 | -0.008 | 0.972 | 0.941 | 0.985 |
| blend | easy | 1.000 | 0.957 | 0.953 | +0.004 | 0.995 | 1.000 | 1.000 |
| lf | easy | 0.997 | 0.944 | 0.953 | -0.009 | 0.990 | 0.968 | 0.994 |
| bpp | **hard** | 0.987 | 0.950 | 0.953 | -0.003 | 0.920 | 0.757 | 0.865 |
| wanet | **hard** | 0.961 | 0.913 | 0.953 | -0.039 | 0.917 | 0.512 | 0.829 |
| tact | **hard** | 0.996 | 0.946 | 0.953 | -0.007 | 0.985 | 0.313 | 0.717 |

## cifar100, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.979 | 0.949 | 0.982 |
| blend | easy | 1.000 | 0.830 | 0.811 | +0.019 | 0.997 | 0.997 | 1.000 |
| lf | easy | 0.953 | 0.829 | 0.811 | +0.019 | 0.960 | 0.952 | 0.954 |
| bpp | **hard** | 0.988 | 0.833 | 0.811 | +0.023 | 0.972 | 0.946 | 0.965 |
| tact | **hard** | 1.000 | 0.825 | 0.811 | +0.014 | 0.974 | 1.000 | 1.000 |

## gtsrb, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.986 | 0.991 | -0.005 | 1.000 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.977 | 0.991 | -0.014 | 0.998 | 1.000 | 1.000 |
| lf | easy | 0.999 | 0.974 | 0.991 | -0.018 | 0.994 | 0.999 | 0.999 |
| bpp | **hard** | 0.991 | 0.978 | 0.991 | -0.013 | 0.994 | 0.991 | 0.991 |
| tact | **hard** | 1.000 | 0.986 | 0.991 | -0.006 | 0.976 | 0.678 | 0.888 |

## tiny, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.002 | 0.988 | 0.990 | 0.998 |
| blend | easy | 0.999 | 0.754 | 0.755 | -0.001 | 0.999 | 0.998 | 0.999 |
| lf | easy | 0.987 | 0.753 | 0.755 | -0.001 | 0.979 | 0.978 | 0.985 |
| bpp | **hard** | 0.985 | 0.746 | 0.755 | -0.009 | 0.939 | 0.883 | 0.948 |
| wanet | **hard** | 0.922 | 0.748 | 0.755 | -0.007 | 0.537 | 0.102 | 0.227 |
| tact | **hard** | 0.952 | 0.743 | 0.755 | -0.012 | 0.744 | 0.405 | 0.619 |

# poison rate 1%

## cifar10, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.997 | 0.952 | 0.953 | -0.001 | 0.983 | 0.981 | 0.995 |
| blend | easy | 1.000 | 0.954 | 0.953 | +0.002 | 0.996 | 1.000 | 1.000 |
| lf | easy | 0.982 | 0.953 | 0.953 | +0.000 | 0.980 | 0.958 | 0.980 |
| bpp | **hard** | 0.982 | 0.952 | 0.953 | -0.000 | 0.890 | 0.652 | 0.775 |
| tact | **hard** | 0.985 | 0.938 | 0.953 | -0.015 | 0.959 | 0.322 | 0.676 |

## cifar100, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.870 | 0.719 | 0.800 |
| blend | easy | 0.989 | 0.829 | 0.811 | +0.019 | 0.922 | 0.759 | 0.941 |
| lf | easy | 0.942 | 0.830 | 0.811 | +0.019 | 0.953 | 0.925 | 0.944 |
| bpp | **hard** | 0.914 | 0.824 | 0.811 | +0.013 | 0.929 | 0.865 | 0.926 |
| tact | **hard** | 0.989 | 0.820 | 0.811 | +0.009 | 0.915 | 0.655 | 0.862 |
| lc | **hard** | 0.873 | 0.823 | 0.811 | +0.013 | 0.889 | 0.813 | 0.872 |

## gtsrb, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.994 | 0.991 | +0.003 | 0.997 | 0.997 | 0.999 |
| blend | easy | 1.000 | 0.991 | 0.991 | +0.000 | 0.981 | 0.934 | 0.968 |
| lf | easy | 0.979 | 0.990 | 0.991 | -0.001 | 0.982 | 0.967 | 0.980 |
| bpp | **hard** | 0.868 | 0.986 | 0.991 | -0.005 | 0.943 | 0.890 | 0.915 |

## tiny, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.999 | 0.752 | 0.755 | -0.003 | 0.985 | 0.985 | 0.997 |
| blend | easy | 0.999 | 0.760 | 0.755 | +0.005 | 0.993 | 0.991 | 0.996 |
| lf | easy | 0.922 | 0.762 | 0.755 | +0.007 | 0.928 | 0.896 | 0.923 |
| bpp | **hard** | 0.971 | 0.748 | 0.755 | -0.007 | 0.959 | 0.922 | 0.953 |
| tact | **hard** | 0.976 | 0.750 | 0.755 | -0.005 | 0.687 | 0.143 | 0.286 |

# Summary

| subset | n | AUROC | TPR@10%FPR | TPR@20%FPR | achieved FPR at 10% |
|---|---:|---:|---:|---:|---:|
| all cells | 67 | 0.925 | 0.801 | 0.875 | 0.0845 |
| **hard** attacks | 31 | 0.864 | 0.622 | 0.755 | 0.0789 |
| easy attacks | 36 | 0.978 | 0.955 | 0.978 | 0.0894 |
| primary datasets (cifar100 + tiny) | 33 | 0.917 | 0.810 | 0.873 | 0.0940 |

| by poison rate | n | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---:|---:|---:|---:|
| 10% | 25 | 0.898 | 0.756 | 0.836 |
| 5% | 22 | 0.946 | 0.834 | 0.907 |
| 1% | 20 | 0.937 | 0.819 | 0.889 |

Worst-case cell AUROC **0.342**, inverted cells (AUROC < 0.5) **1** of 67.

