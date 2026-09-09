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
| badnet_a2o | easy | 1.000 | 0.942 | 0.953 | -0.011 | 0.961 | 0.885 | 0.965 |
| blend | easy | 1.000 | 0.951 | 0.953 | -0.001 | 0.978 | 0.957 | 0.973 |
| lf | easy | 0.999 | 0.951 | 0.953 | -0.002 | 0.990 | 0.997 | 0.998 |
| bpp | **hard** | 0.994 | 0.954 | 0.953 | +0.001 | 0.929 | 0.862 | 0.904 |
| wanet | **hard** | 0.890 | 0.946 | 0.953 | -0.007 | 0.432 | 0.090 | 0.172 |
| tact | **hard** | 1.000 | 0.858 | 0.953 | -0.095 | 0.938 | 0.143 | 0.318 |
| sig | **hard** | 0.901 | 0.846 | 0.953 | -0.106 | 0.878 | 0.783 | 0.825 |
| lc | **hard** | 0.979 | 0.861 | 0.953 | -0.092 | 0.947 | 0.914 | 0.965 |

## cifar100, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.821 | 0.811 | +0.010 | 0.962 | 0.991 | 1.000 |
| blend | easy | 1.000 | 0.816 | 0.811 | +0.006 | 0.982 | 1.000 | 1.000 |
| lf | easy | 0.995 | 0.823 | 0.811 | +0.012 | 0.994 | 0.995 | 0.995 |
| bpp | **hard** | 0.991 | 0.817 | 0.811 | +0.006 | 0.975 | 0.954 | 0.972 |
| tact | **hard** | 0.989 | 0.821 | 0.811 | +0.010 | 0.961 | 0.126 | 0.402 |

## gtsrb, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.972 | 0.991 | -0.019 | 0.999 | 0.995 | 1.000 |
| blend | easy | 1.000 | 0.988 | 0.991 | -0.003 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.999 | 0.988 | 0.991 | -0.003 | 0.997 | 0.995 | 0.998 |
| bpp | **hard** | 0.995 | 0.986 | 0.991 | -0.006 | 0.988 | 0.971 | 0.983 |
| wanet | **hard** | 0.947 | 0.958 | 0.991 | -0.033 | 0.914 | 0.802 | 0.912 |
| tact | **hard** | 1.000 | 0.931 | 0.991 | -0.060 | 0.763 | 0.020 | 0.293 |

## tiny, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.003 | 0.889 | 0.513 | 0.907 |
| blend | easy | 1.000 | 0.749 | 0.755 | -0.006 | 0.961 | 0.987 | 0.997 |
| lf | easy | 0.973 | 0.740 | 0.755 | -0.015 | 0.911 | 0.859 | 0.932 |
| bpp | **hard** | 0.996 | 0.757 | 0.755 | +0.002 | 0.958 | 0.957 | 0.982 |
| wanet | **hard** | 0.968 | 0.747 | 0.755 | -0.008 | 0.913 | 0.742 | 0.917 |
| tact | **hard** | 0.929 | 0.756 | 0.755 | +0.001 | 0.833 | 0.095 | 0.429 |

# poison rate 5%

## cifar10, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.945 | 0.953 | -0.008 | 0.967 | 0.918 | 0.978 |
| blend | easy | 1.000 | 0.957 | 0.953 | +0.004 | 0.988 | 0.999 | 1.000 |
| lf | easy | 0.997 | 0.944 | 0.953 | -0.009 | 0.986 | 0.989 | 0.993 |
| bpp | **hard** | 0.987 | 0.950 | 0.953 | -0.003 | 0.931 | 0.866 | 0.911 |
| wanet | **hard** | 0.961 | 0.913 | 0.953 | -0.039 | 0.918 | 0.793 | 0.889 |
| tact | **hard** | 0.996 | 0.946 | 0.953 | -0.007 | 0.788 | 0.099 | 0.166 |

## cifar100, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.933 | 0.851 | 0.993 |
| blend | easy | 1.000 | 0.830 | 0.811 | +0.019 | 0.986 | 0.983 | 0.995 |
| lf | easy | 0.953 | 0.829 | 0.811 | +0.019 | 0.927 | 0.892 | 0.944 |
| bpp | **hard** | 0.988 | 0.833 | 0.811 | +0.023 | 0.983 | 0.963 | 0.974 |
| tact | **hard** | 1.000 | 0.825 | 0.811 | +0.014 | 0.973 | 0.471 | 0.644 |

## gtsrb, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.986 | 0.991 | -0.005 | 1.000 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.977 | 0.991 | -0.014 | 0.998 | 0.999 | 1.000 |
| lf | easy | 0.999 | 0.974 | 0.991 | -0.018 | 0.990 | 0.978 | 0.992 |
| bpp | **hard** | 0.991 | 0.978 | 0.991 | -0.013 | 0.990 | 0.973 | 0.984 |
| tact | **hard** | 1.000 | 0.986 | 0.991 | -0.006 | 0.953 | 0.322 | 0.570 |

## tiny, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.002 | 0.842 | 0.361 | 0.750 |
| blend | easy | 0.999 | 0.754 | 0.755 | -0.001 | 0.954 | 0.991 | 0.997 |
| lf | easy | 0.987 | 0.753 | 0.755 | -0.001 | 0.939 | 0.944 | 0.975 |
| bpp | **hard** | 0.985 | 0.746 | 0.755 | -0.009 | 0.970 | 0.978 | 0.984 |
| wanet | **hard** | 0.922 | 0.748 | 0.755 | -0.007 | 0.896 | 0.774 | 0.877 |
| tact | **hard** | 0.952 | 0.743 | 0.755 | -0.012 | 0.728 | 0.262 | 0.500 |

# poison rate 1%

## cifar10, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.997 | 0.952 | 0.953 | -0.001 | 0.993 | 0.989 | 0.995 |
| blend | easy | 1.000 | 0.954 | 0.953 | +0.002 | 0.991 | 0.986 | 0.994 |
| lf | easy | 0.982 | 0.953 | 0.953 | +0.000 | 0.976 | 0.970 | 0.978 |
| bpp | **hard** | 0.982 | 0.952 | 0.953 | -0.000 | 0.986 | 0.965 | 0.978 |
| tact | **hard** | 0.985 | 0.938 | 0.953 | -0.015 | 0.799 | 0.005 | 0.018 |

## cifar100, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.960 | 0.974 | 0.999 |
| blend | easy | 0.989 | 0.829 | 0.811 | +0.019 | 0.945 | 0.887 | 0.967 |
| lf | easy | 0.942 | 0.830 | 0.811 | +0.019 | 0.944 | 0.937 | 0.944 |
| bpp | **hard** | 0.914 | 0.824 | 0.811 | +0.013 | 0.773 | 0.211 | 0.610 |
| tact | **hard** | 0.989 | 0.820 | 0.811 | +0.009 | 0.981 | 0.552 | 0.908 |
| lc | **hard** | 0.873 | 0.823 | 0.811 | +0.013 | 0.692 | 0.052 | 0.272 |

## gtsrb, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.994 | 0.991 | +0.003 | 0.995 | 0.994 | 0.997 |
| blend | easy | 1.000 | 0.991 | 0.991 | +0.000 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.979 | 0.990 | 0.991 | -0.001 | 0.971 | 0.915 | 0.961 |
| bpp | **hard** | 0.868 | 0.986 | 0.991 | -0.005 | 0.809 | 0.467 | 0.657 |

## tiny, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.999 | 0.752 | 0.755 | -0.003 | 0.815 | 0.287 | 0.503 |
| blend | easy | 0.999 | 0.760 | 0.755 | +0.005 | 0.974 | 0.982 | 0.994 |
| lf | easy | 0.922 | 0.762 | 0.755 | +0.007 | 0.901 | 0.851 | 0.902 |
| bpp | **hard** | 0.971 | 0.748 | 0.755 | -0.007 | 0.969 | 0.967 | 0.973 |
| tact | **hard** | 0.976 | 0.750 | 0.755 | -0.005 | 0.651 | 0.024 | 0.167 |

# Summary

| subset | n | AUROC | TPR@10%FPR | TPR@20%FPR | achieved FPR at 10% |
|---|---:|---:|---:|---:|---:|
| all cells | 67 | 0.923 | 0.747 | 0.832 | 0.0906 |
| **hard** attacks | 31 | 0.878 | 0.555 | 0.682 | 0.0793 |
| easy attacks | 36 | 0.961 | 0.912 | 0.962 | 0.1003 |
| primary datasets (cifar100 + tiny) | 33 | 0.911 | 0.710 | 0.830 | 0.0875 |

| by poison rate | n | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---:|---:|---:|---:|
| 10% | 25 | 0.922 | 0.745 | 0.834 |
| 5% | 22 | 0.938 | 0.791 | 0.869 |
| 1% | 20 | 0.906 | 0.701 | 0.791 |

Worst-case cell AUROC **0.432**, inverted cells (AUROC < 0.5) **1** of 67.

