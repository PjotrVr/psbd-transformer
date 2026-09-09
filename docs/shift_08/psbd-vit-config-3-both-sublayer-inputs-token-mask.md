# PSBD-ViT: `both_sublayer_inputs_token_mask`

The single placement `both_sublayer_inputs_token_mask`.

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
| badnet_a2o | easy | 1.000 | 0.942 | 0.953 | -0.011 | 0.984 | 0.559 | 0.996 |
| blend | easy | 1.000 | 0.951 | 0.953 | -0.001 | 0.545 | 0.077 | 0.139 |
| lf | easy | 0.999 | 0.951 | 0.953 | -0.002 | 0.975 | 0.938 | 0.993 |
| bpp | **hard** | 0.994 | 0.954 | 0.953 | +0.001 | 0.893 | 0.776 | 0.859 |
| wanet | **hard** | 0.890 | 0.946 | 0.953 | -0.007 | 0.724 | 0.285 | 0.613 |
| tact | **hard** | 1.000 | 0.858 | 0.953 | -0.095 | 0.961 | 0.001 | 0.114 |
| sig | **hard** | 0.901 | 0.846 | 0.953 | -0.106 | 0.917 | 0.801 | 0.885 |
| lc | **hard** | 0.979 | 0.861 | 0.953 | -0.092 | 0.836 | 0.480 | 0.640 |

## cifar100, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.821 | 0.811 | +0.010 | 0.984 | 0.998 | 1.000 |
| blend | easy | 1.000 | 0.816 | 0.811 | +0.006 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.995 | 0.823 | 0.811 | +0.012 | 0.993 | 0.990 | 0.994 |
| bpp | **hard** | 0.991 | 0.817 | 0.811 | +0.006 | 0.986 | 0.967 | 0.978 |
| tact | **hard** | 0.989 | 0.821 | 0.811 | +0.010 | 0.814 | 0.437 | 0.759 |

## gtsrb, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.972 | 0.991 | -0.019 | 0.995 | 0.999 | 1.000 |
| blend | easy | 1.000 | 0.988 | 0.991 | -0.003 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.999 | 0.988 | 0.991 | -0.003 | 0.992 | 0.984 | 0.996 |
| bpp | **hard** | 0.995 | 0.986 | 0.991 | -0.006 | 0.976 | 0.927 | 0.972 |
| wanet | **hard** | 0.947 | 0.958 | 0.991 | -0.033 | 0.950 | 0.948 | 0.952 |
| tact | **hard** | 1.000 | 0.931 | 0.991 | -0.060 | 0.744 | 0.107 | 0.717 |

## tiny, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.003 | 0.898 | 0.509 | 0.911 |
| blend | easy | 1.000 | 0.749 | 0.755 | -0.006 | 0.994 | 0.992 | 0.998 |
| lf | easy | 0.973 | 0.740 | 0.755 | -0.015 | 0.953 | 0.915 | 0.940 |
| bpp | **hard** | 0.996 | 0.757 | 0.755 | +0.002 | 0.978 | 0.979 | 0.989 |
| wanet | **hard** | 0.968 | 0.747 | 0.755 | -0.008 | 0.819 | 0.334 | 0.713 |
| tact | **hard** | 0.929 | 0.756 | 0.755 | +0.001 | 0.593 | 0.167 | 0.238 |

# poison rate 5%

## cifar10, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.945 | 0.953 | -0.008 | 0.959 | 0.639 | 0.902 |
| blend | easy | 1.000 | 0.957 | 0.953 | +0.004 | 0.953 | 0.807 | 0.972 |
| lf | easy | 0.997 | 0.944 | 0.953 | -0.009 | 0.978 | 0.938 | 0.990 |
| bpp | **hard** | 0.987 | 0.950 | 0.953 | -0.003 | 0.797 | 0.562 | 0.645 |
| wanet | **hard** | 0.961 | 0.913 | 0.953 | -0.039 | 0.951 | 0.941 | 0.961 |
| tact | **hard** | 0.996 | 0.946 | 0.953 | -0.007 | 0.902 | 0.001 | 0.053 |

## cifar100, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.935 | 0.862 | 0.985 |
| blend | easy | 1.000 | 0.830 | 0.811 | +0.019 | 0.988 | 0.975 | 0.993 |
| lf | easy | 0.953 | 0.829 | 0.811 | +0.019 | 0.926 | 0.833 | 0.932 |
| bpp | **hard** | 0.988 | 0.833 | 0.811 | +0.023 | 0.989 | 0.974 | 0.988 |
| tact | **hard** | 1.000 | 0.825 | 0.811 | +0.014 | 0.946 | 0.460 | 0.770 |

## gtsrb, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.986 | 0.991 | -0.005 | 1.000 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.977 | 0.991 | -0.014 | 0.987 | 0.969 | 0.984 |
| lf | easy | 0.999 | 0.974 | 0.991 | -0.018 | 0.992 | 0.987 | 0.995 |
| bpp | **hard** | 0.991 | 0.978 | 0.991 | -0.013 | 0.995 | 0.990 | 0.992 |
| tact | **hard** | 1.000 | 0.986 | 0.991 | -0.006 | 0.705 | 0.395 | 0.545 |

## tiny, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.002 | 0.852 | 0.326 | 0.793 |
| blend | easy | 0.999 | 0.754 | 0.755 | -0.001 | 0.989 | 0.980 | 0.988 |
| lf | easy | 0.987 | 0.753 | 0.755 | -0.001 | 0.952 | 0.889 | 0.971 |
| bpp | **hard** | 0.985 | 0.746 | 0.755 | -0.009 | 0.985 | 0.984 | 0.987 |
| wanet | **hard** | 0.922 | 0.748 | 0.755 | -0.007 | 0.879 | 0.664 | 0.877 |
| tact | **hard** | 0.952 | 0.743 | 0.755 | -0.012 | 0.664 | 0.238 | 0.381 |

# poison rate 1%

## cifar10, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.997 | 0.952 | 0.953 | -0.001 | 0.535 | 0.075 | 0.129 |
| blend | easy | 1.000 | 0.954 | 0.953 | +0.002 | 0.826 | 0.461 | 0.704 |
| lf | easy | 0.982 | 0.953 | 0.953 | +0.000 | 0.963 | 0.874 | 0.964 |
| bpp | **hard** | 0.982 | 0.952 | 0.953 | -0.000 | 0.845 | 0.452 | 0.615 |
| tact | **hard** | 0.985 | 0.938 | 0.953 | -0.015 | 0.980 | 0.010 | 0.013 |

## cifar100, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.994 | 0.995 | 0.999 |
| blend | easy | 0.989 | 0.829 | 0.811 | +0.019 | 0.970 | 0.937 | 0.991 |
| lf | easy | 0.942 | 0.830 | 0.811 | +0.019 | 0.954 | 0.908 | 0.947 |
| bpp | **hard** | 0.914 | 0.824 | 0.811 | +0.013 | 0.902 | 0.776 | 0.878 |
| tact | **hard** | 0.989 | 0.820 | 0.811 | +0.009 | 0.977 | 0.759 | 0.931 |
| lc | **hard** | 0.873 | 0.823 | 0.811 | +0.013 | 0.855 | 0.589 | 0.785 |

## gtsrb, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.994 | 0.991 | +0.003 | 0.967 | 0.892 | 0.927 |
| blend | easy | 1.000 | 0.991 | 0.991 | +0.000 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.979 | 0.990 | 0.991 | -0.001 | 0.952 | 0.879 | 0.925 |
| bpp | **hard** | 0.868 | 0.986 | 0.991 | -0.005 | 0.820 | 0.506 | 0.666 |

## tiny, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.999 | 0.752 | 0.755 | -0.003 | 0.856 | 0.320 | 0.891 |
| blend | easy | 0.999 | 0.760 | 0.755 | +0.005 | 0.980 | 0.975 | 0.990 |
| lf | easy | 0.922 | 0.762 | 0.755 | +0.007 | 0.913 | 0.823 | 0.915 |
| bpp | **hard** | 0.971 | 0.748 | 0.755 | -0.007 | 0.957 | 0.967 | 0.972 |
| tact | **hard** | 0.976 | 0.750 | 0.755 | -0.005 | 0.466 | 0.000 | 0.071 |

# Summary

| subset | n | AUROC | TPR@10%FPR | TPR@20%FPR | achieved FPR at 10% |
|---|---:|---:|---:|---:|---:|
| all cells | 67 | 0.904 | 0.698 | 0.812 | 0.0868 |
| **hard** attacks | 31 | 0.865 | 0.564 | 0.695 | 0.0749 |
| easy attacks | 36 | 0.937 | 0.814 | 0.913 | 0.0970 |
| primary datasets (cifar100 + tiny) | 33 | 0.907 | 0.743 | 0.865 | 0.0846 |

| by poison rate | n | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---:|---:|---:|---:|
| 10% | 25 | 0.900 | 0.687 | 0.816 |
| 5% | 22 | 0.924 | 0.746 | 0.850 |
| 1% | 20 | 0.886 | 0.660 | 0.766 |

Worst-case cell AUROC **0.466**, inverted cells (AUROC < 0.5) **1** of 67.

