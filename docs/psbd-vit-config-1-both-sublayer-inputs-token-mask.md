# PSBD-ViT: `both_sublayer_inputs_token_mask`

The single placement `both_sublayer_inputs_token_mask`.

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
| badnet_a2o | easy | 1.000 | 0.942 | 0.953 | -0.011 | 0.966 | 0.583 | 0.941 |
| blend | easy | 1.000 | 0.951 | 0.953 | -0.001 | 0.919 | 0.799 | 0.872 |
| lf | easy | 0.999 | 0.951 | 0.953 | -0.002 | 0.976 | 0.960 | 0.982 |
| bpp | **hard** | 0.994 | 0.954 | 0.953 | +0.001 | 0.952 | 0.888 | 0.938 |
| wanet | **hard** | 0.890 | 0.946 | 0.953 | -0.007 | 0.605 | 0.167 | 0.264 |
| tact | **hard** | 1.000 | 0.858 | 0.953 | -0.095 | 0.909 | 0.075 | 0.197 |
| sig | **hard** | 0.901 | 0.846 | 0.953 | -0.106 | 0.912 | 0.871 | 0.892 |
| lc | **hard** | 0.979 | 0.861 | 0.953 | -0.092 | 0.909 | 0.769 | 0.891 |

## cifar100, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.821 | 0.811 | +0.010 | 0.984 | 0.998 | 1.000 |
| blend | easy | 1.000 | 0.816 | 0.811 | +0.006 | 0.990 | 1.000 | 1.000 |
| lf | easy | 0.995 | 0.823 | 0.811 | +0.012 | 0.981 | 0.976 | 0.993 |
| bpp | **hard** | 0.991 | 0.817 | 0.811 | +0.006 | 0.961 | 0.946 | 0.971 |
| tact | **hard** | 0.989 | 0.821 | 0.811 | +0.010 | 0.899 | 0.207 | 0.506 |

## gtsrb, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.972 | 0.991 | -0.019 | 0.925 | 0.724 | 0.827 |
| blend | easy | 1.000 | 0.988 | 0.991 | -0.003 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.999 | 0.988 | 0.991 | -0.003 | 0.992 | 0.975 | 0.993 |
| bpp | **hard** | 0.995 | 0.986 | 0.991 | -0.006 | 0.974 | 0.938 | 0.964 |
| wanet | **hard** | 0.947 | 0.958 | 0.991 | -0.033 | 0.735 | 0.361 | 0.523 |
| tact | **hard** | 1.000 | 0.931 | 0.991 | -0.060 | 0.758 | 0.079 | 0.387 |

## tiny, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.003 | 0.898 | 0.509 | 0.911 |
| blend | easy | 1.000 | 0.749 | 0.755 | -0.006 | 0.977 | 0.981 | 0.995 |
| lf | easy | 0.973 | 0.740 | 0.755 | -0.015 | 0.912 | 0.741 | 0.908 |
| bpp | **hard** | 0.996 | 0.757 | 0.755 | +0.002 | 0.918 | 0.873 | 0.973 |
| wanet | **hard** | 0.968 | 0.747 | 0.755 | -0.008 | 0.819 | 0.334 | 0.713 |
| tact | **hard** | 0.929 | 0.756 | 0.755 | +0.001 | 0.730 | 0.095 | 0.214 |

# poison rate 5%

## cifar10, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.945 | 0.953 | -0.008 | 0.928 | 0.663 | 0.852 |
| blend | easy | 1.000 | 0.957 | 0.953 | +0.004 | 0.987 | 0.992 | 0.998 |
| lf | easy | 0.997 | 0.944 | 0.953 | -0.009 | 0.959 | 0.928 | 0.967 |
| bpp | **hard** | 0.987 | 0.950 | 0.953 | -0.003 | 0.960 | 0.925 | 0.944 |
| wanet | **hard** | 0.961 | 0.913 | 0.953 | -0.039 | 0.908 | 0.741 | 0.872 |
| tact | **hard** | 0.996 | 0.946 | 0.953 | -0.007 | 0.902 | 0.001 | 0.053 |

## cifar100, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.935 | 0.862 | 0.985 |
| blend | easy | 1.000 | 0.830 | 0.811 | +0.019 | 0.970 | 0.983 | 0.996 |
| lf | easy | 0.953 | 0.829 | 0.811 | +0.019 | 0.926 | 0.833 | 0.932 |
| bpp | **hard** | 0.988 | 0.833 | 0.811 | +0.023 | 0.977 | 0.955 | 0.974 |
| tact | **hard** | 1.000 | 0.825 | 0.811 | +0.014 | 0.946 | 0.460 | 0.770 |

## gtsrb, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.986 | 0.991 | -0.005 | 0.993 | 0.977 | 0.984 |
| blend | easy | 1.000 | 0.977 | 0.991 | -0.014 | 0.975 | 0.933 | 0.963 |
| lf | easy | 0.999 | 0.974 | 0.991 | -0.018 | 0.992 | 0.987 | 0.995 |
| bpp | **hard** | 0.991 | 0.978 | 0.991 | -0.013 | 0.987 | 0.970 | 0.981 |
| tact | **hard** | 1.000 | 0.986 | 0.991 | -0.006 | 0.814 | 0.175 | 0.327 |

## tiny, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.002 | 0.852 | 0.326 | 0.793 |
| blend | easy | 0.999 | 0.754 | 0.755 | -0.001 | 0.974 | 0.980 | 0.991 |
| lf | easy | 0.987 | 0.753 | 0.755 | -0.001 | 0.952 | 0.889 | 0.971 |
| bpp | **hard** | 0.985 | 0.746 | 0.755 | -0.009 | 0.952 | 0.965 | 0.980 |
| wanet | **hard** | 0.922 | 0.748 | 0.755 | -0.007 | 0.879 | 0.664 | 0.877 |
| tact | **hard** | 0.952 | 0.743 | 0.755 | -0.012 | 0.664 | 0.238 | 0.381 |

# poison rate 1%

## cifar10, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.997 | 0.952 | 0.953 | -0.001 | 0.845 | 0.553 | 0.752 |
| blend | easy | 1.000 | 0.954 | 0.953 | +0.002 | 0.978 | 0.952 | 0.982 |
| lf | easy | 0.982 | 0.953 | 0.953 | +0.000 | 0.972 | 0.946 | 0.966 |
| bpp | **hard** | 0.982 | 0.952 | 0.953 | -0.000 | 0.968 | 0.918 | 0.953 |
| tact | **hard** | 0.985 | 0.938 | 0.953 | -0.015 | 0.809 | 0.006 | 0.013 |

## cifar100, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.960 | 0.932 | 0.997 |
| blend | easy | 0.989 | 0.829 | 0.811 | +0.019 | 0.915 | 0.795 | 0.949 |
| lf | easy | 0.942 | 0.830 | 0.811 | +0.019 | 0.912 | 0.753 | 0.899 |
| bpp | **hard** | 0.914 | 0.824 | 0.811 | +0.013 | 0.902 | 0.776 | 0.878 |
| tact | **hard** | 0.989 | 0.820 | 0.811 | +0.009 | 0.977 | 0.759 | 0.931 |
| lc | **hard** | 0.873 | 0.823 | 0.811 | +0.013 | 0.759 | 0.193 | 0.553 |

## gtsrb, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.994 | 0.991 | +0.003 | 0.787 | 0.361 | 0.427 |
| blend | easy | 1.000 | 0.991 | 0.991 | +0.000 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.979 | 0.990 | 0.991 | -0.001 | 0.952 | 0.879 | 0.925 |
| bpp | **hard** | 0.868 | 0.986 | 0.991 | -0.005 | 0.757 | 0.369 | 0.548 |

## tiny, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.999 | 0.752 | 0.755 | -0.003 | 0.856 | 0.320 | 0.891 |
| blend | easy | 0.999 | 0.760 | 0.755 | +0.005 | 0.980 | 0.975 | 0.990 |
| lf | easy | 0.922 | 0.762 | 0.755 | +0.007 | 0.913 | 0.823 | 0.915 |
| bpp | **hard** | 0.971 | 0.748 | 0.755 | -0.007 | 0.957 | 0.967 | 0.972 |
| tact | **hard** | 0.976 | 0.750 | 0.755 | -0.005 | 0.593 | 0.024 | 0.048 |

# Summary

| subset | n | AUROC | TPR@10%FPR | TPR@20%FPR | achieved FPR at 10% |
|---|---:|---:|---:|---:|---:|
| all cells | 67 | 0.908 | 0.695 | 0.806 | 0.0883 |
| **hard** attacks | 31 | 0.864 | 0.539 | 0.661 | 0.0780 |
| easy attacks | 36 | 0.945 | 0.830 | 0.932 | 0.0972 |
| primary datasets (cifar100 + tiny) | 33 | 0.904 | 0.701 | 0.844 | 0.0847 |

| by poison rate | n | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---:|---:|---:|---:|
| 10% | 25 | 0.904 | 0.674 | 0.794 |
| 5% | 22 | 0.929 | 0.748 | 0.845 |
| 1% | 20 | 0.890 | 0.665 | 0.780 |

Worst-case cell AUROC **0.593**, inverted cells (AUROC < 0.5) **0** of 67.

