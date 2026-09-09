# PSBD-ViT: merged top two (both_sublayer_inputs_token_mask + before_attention_norm_token_mask)

A **fused** configuration. Its members are swept separately and combined at the score
level with the min-rank rule against the clean-validation reference, so a sample is
flagged when ANY member finds it suspicious. This needs no extra training and no extra
checkpoint, only one more perturbation sweep per member at inference time.

Members:

- `both_sublayer_inputs_token_mask`
- `before_attention_norm_token_mask`

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
| badnet_a2o | easy | 1.000 | 0.942 | 0.953 | -0.011 | 0.964 | 0.825 | 0.967 |
| blend | easy | 1.000 | 0.951 | 0.953 | -0.001 | 0.974 | 0.949 | 0.967 |
| lf | easy | 0.999 | 0.951 | 0.953 | -0.002 | 0.987 | 0.996 | 0.998 |
| bpp | **hard** | 0.994 | 0.954 | 0.953 | +0.001 | 0.953 | 0.905 | 0.932 |
| wanet | **hard** | 0.890 | 0.946 | 0.953 | -0.007 | 0.593 | 0.169 | 0.271 |
| tact | **hard** | 1.000 | 0.858 | 0.953 | -0.095 | 0.941 | 0.102 | 0.288 |
| sig | **hard** | 0.901 | 0.846 | 0.953 | -0.106 | 0.910 | 0.854 | 0.887 |
| lc | **hard** | 0.979 | 0.861 | 0.953 | -0.092 | 0.947 | 0.906 | 0.966 |

## cifar100, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.821 | 0.811 | +0.010 | 0.976 | 0.997 | 1.000 |
| blend | easy | 1.000 | 0.816 | 0.811 | +0.006 | 0.984 | 1.000 | 1.000 |
| lf | easy | 0.995 | 0.823 | 0.811 | +0.012 | 0.993 | 0.994 | 0.995 |
| bpp | **hard** | 0.991 | 0.817 | 0.811 | +0.006 | 0.970 | 0.954 | 0.975 |
| tact | **hard** | 0.989 | 0.821 | 0.811 | +0.010 | 0.944 | 0.207 | 0.460 |

## gtsrb, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.972 | 0.991 | -0.019 | 0.998 | 0.996 | 0.998 |
| blend | easy | 1.000 | 0.988 | 0.991 | -0.003 | 0.999 | 1.000 | 1.000 |
| lf | easy | 0.999 | 0.988 | 0.991 | -0.003 | 0.996 | 0.993 | 0.998 |
| bpp | **hard** | 0.995 | 0.986 | 0.991 | -0.006 | 0.987 | 0.969 | 0.982 |
| wanet | **hard** | 0.947 | 0.958 | 0.991 | -0.033 | 0.885 | 0.696 | 0.864 |
| tact | **hard** | 1.000 | 0.931 | 0.991 | -0.060 | 0.758 | 0.036 | 0.296 |

## tiny, poison rate 10%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.003 | 0.899 | 0.504 | 0.968 |
| blend | easy | 1.000 | 0.749 | 0.755 | -0.006 | 0.966 | 0.977 | 0.995 |
| lf | easy | 0.973 | 0.740 | 0.755 | -0.015 | 0.913 | 0.838 | 0.929 |
| bpp | **hard** | 0.996 | 0.757 | 0.755 | +0.002 | 0.938 | 0.936 | 0.978 |
| wanet | **hard** | 0.968 | 0.747 | 0.755 | -0.008 | 0.895 | 0.632 | 0.880 |
| tact | **hard** | 0.929 | 0.756 | 0.755 | +0.001 | 0.806 | 0.071 | 0.286 |

# poison rate 5%

## cifar10, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.945 | 0.953 | -0.008 | 0.966 | 0.903 | 0.974 |
| blend | easy | 1.000 | 0.957 | 0.953 | +0.004 | 0.986 | 0.998 | 1.000 |
| lf | easy | 0.997 | 0.944 | 0.953 | -0.009 | 0.980 | 0.986 | 0.992 |
| bpp | **hard** | 0.987 | 0.950 | 0.953 | -0.003 | 0.957 | 0.924 | 0.945 |
| wanet | **hard** | 0.961 | 0.913 | 0.953 | -0.039 | 0.921 | 0.801 | 0.900 |
| tact | **hard** | 0.996 | 0.946 | 0.953 | -0.007 | 0.819 | 0.053 | 0.114 |

## cifar100, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.937 | 0.888 | 0.993 |
| blend | easy | 1.000 | 0.830 | 0.811 | +0.019 | 0.979 | 0.975 | 0.994 |
| lf | easy | 0.953 | 0.829 | 0.811 | +0.019 | 0.929 | 0.863 | 0.935 |
| bpp | **hard** | 0.988 | 0.833 | 0.811 | +0.023 | 0.984 | 0.963 | 0.978 |
| tact | **hard** | 1.000 | 0.825 | 0.811 | +0.014 | 0.970 | 0.471 | 0.770 |

## gtsrb, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.986 | 0.991 | -0.005 | 0.999 | 1.000 | 1.000 |
| blend | easy | 1.000 | 0.977 | 0.991 | -0.014 | 0.997 | 0.998 | 1.000 |
| lf | easy | 0.999 | 0.974 | 0.991 | -0.018 | 0.991 | 0.985 | 0.995 |
| bpp | **hard** | 0.991 | 0.978 | 0.991 | -0.013 | 0.991 | 0.978 | 0.985 |
| tact | **hard** | 1.000 | 0.986 | 0.991 | -0.006 | 0.937 | 0.284 | 0.521 |

## tiny, poison rate 5%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.752 | 0.755 | -0.002 | 0.857 | 0.346 | 0.777 |
| blend | easy | 0.999 | 0.754 | 0.755 | -0.001 | 0.959 | 0.982 | 0.995 |
| lf | easy | 0.987 | 0.753 | 0.755 | -0.001 | 0.944 | 0.930 | 0.978 |
| bpp | **hard** | 0.985 | 0.746 | 0.755 | -0.009 | 0.959 | 0.977 | 0.984 |
| wanet | **hard** | 0.922 | 0.748 | 0.755 | -0.007 | 0.897 | 0.767 | 0.886 |
| tact | **hard** | 0.952 | 0.743 | 0.755 | -0.012 | 0.720 | 0.214 | 0.429 |

# poison rate 1%

## cifar10, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.997 | 0.952 | 0.953 | -0.001 | 0.993 | 0.987 | 0.994 |
| blend | easy | 1.000 | 0.954 | 0.953 | +0.002 | 0.989 | 0.983 | 0.991 |
| lf | easy | 0.982 | 0.953 | 0.953 | +0.000 | 0.978 | 0.965 | 0.977 |
| bpp | **hard** | 0.982 | 0.952 | 0.953 | -0.000 | 0.983 | 0.960 | 0.974 |
| tact | **hard** | 0.985 | 0.938 | 0.953 | -0.015 | 0.762 | 0.005 | 0.015 |

## cifar100, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.825 | 0.811 | +0.015 | 0.962 | 0.977 | 1.000 |
| blend | easy | 0.989 | 0.829 | 0.811 | +0.019 | 0.928 | 0.820 | 0.957 |
| lf | easy | 0.942 | 0.830 | 0.811 | +0.019 | 0.950 | 0.932 | 0.949 |
| bpp | **hard** | 0.914 | 0.824 | 0.811 | +0.013 | 0.879 | 0.678 | 0.844 |
| tact | **hard** | 0.989 | 0.820 | 0.811 | +0.009 | 0.980 | 0.701 | 0.931 |
| lc | **hard** | 0.873 | 0.823 | 0.811 | +0.013 | 0.737 | 0.100 | 0.414 |

## gtsrb, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 1.000 | 0.994 | 0.991 | +0.003 | 0.994 | 0.992 | 0.998 |
| blend | easy | 1.000 | 0.991 | 0.991 | +0.000 | 1.000 | 1.000 | 1.000 |
| lf | easy | 0.979 | 0.990 | 0.991 | -0.001 | 0.968 | 0.914 | 0.956 |
| bpp | **hard** | 0.868 | 0.986 | 0.991 | -0.005 | 0.792 | 0.427 | 0.612 |

## tiny, poison rate 1%

| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | easy | 0.999 | 0.752 | 0.755 | -0.003 | 0.859 | 0.366 | 0.775 |
| blend | easy | 0.999 | 0.760 | 0.755 | +0.005 | 0.973 | 0.974 | 0.992 |
| lf | easy | 0.922 | 0.762 | 0.755 | +0.007 | 0.908 | 0.864 | 0.911 |
| bpp | **hard** | 0.971 | 0.748 | 0.755 | -0.007 | 0.959 | 0.968 | 0.973 |
| tact | **hard** | 0.976 | 0.750 | 0.755 | -0.005 | 0.676 | 0.024 | 0.071 |

# Summary

| subset | n | AUROC | TPR@10%FPR | TPR@20%FPR | achieved FPR at 10% |
|---|---:|---:|---:|---:|---:|
| all cells | 67 | 0.927 | 0.753 | 0.841 | 0.0890 |
| **hard** attacks | 31 | 0.885 | 0.572 | 0.691 | 0.0771 |
| easy attacks | 36 | 0.963 | 0.908 | 0.971 | 0.0993 |
| primary datasets (cifar100 + tiny) | 33 | 0.916 | 0.724 | 0.849 | 0.0841 |

| by poison rate | n | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---:|---:|---:|---:|
| 10% | 25 | 0.927 | 0.740 | 0.835 |
| 5% | 22 | 0.940 | 0.786 | 0.870 |
| 1% | 20 | 0.913 | 0.732 | 0.817 |

Worst-case cell AUROC **0.593**, inverted cells (AUROC < 0.5) **0** of 67.

