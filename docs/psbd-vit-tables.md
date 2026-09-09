# PSBD-ViT results: all-to-one, every dataset, every poison rate

ViT-B/16, 15 epochs, seed 0. Detector read at the rate whose clean-validation shift
ratio is nearest 0.6, fractional PSU, k=3. Threshold is a quantile of clean-validation
PSU, so the FPR budget uses no poison label. One-sided: low PSU means poisoned.

A cell with **ASR < 0.5** is an attack that never implanted; its detection number is
noise about a model with no working backdoor and it is excluded from every mean.

---
# PART 1 — fixed deployed configuration: `token_mask @ before_attention_norm`


## cifar10, poison rate 1%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 0.997 | 0.952 | **0.993** | 0.989 | 0.995 |
| `blend` | 1.000 | 0.954 | **0.991** | 0.986 | 0.994 |
| `sig` ⚠ *never implanted* | 0.340 | 0.951 | **0.497** | 0.147 | 0.236 |
| `wanet` ⚠ *never implanted* | 0.112 | 0.954 | **0.463** | 0.129 | 0.208 |
| `lf` | 0.982 | 0.953 | **0.976** | 0.970 | 0.978 |
| `lc` ⚠ *never implanted* | 0.249 | 0.933 | **0.531** | 0.159 | 0.244 |
| `bpp` | 0.982 | 0.952 | **0.986** | 0.965 | 0.978 |
| `adaptive_blend` | 0.622 | 0.953 | **0.629** | 0.622 | 0.623 |
| `tact` ⚠ *never implanted* | 0.121 | 0.938 | *no sweep* | – | – |
| **mean (implanted)** | | | **0.915** | 0.907 | 0.913 |

## cifar100, poison rate 1%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | 0.825 | **0.960** | 0.974 | 0.999 |
| `blend` | 0.989 | 0.829 | **0.945** | 0.887 | 0.967 |
| `sig` | – | – | **0.425** | 0.039 | 0.099 |
| `wanet` ⚠ *never implanted* | 0.057 | 0.822 | **0.470** | 0.115 | 0.204 |
| `lf` | 0.942 | 0.830 | **0.944** | 0.937 | 0.944 |
| `lc` | 0.873 | 0.823 | **0.692** | 0.052 | 0.272 |
| `bpp` | 0.914 | 0.824 | **0.773** | 0.211 | 0.610 |
| `adaptive_blend` | 0.536 | 0.823 | **0.546** | 0.529 | 0.536 |
| `tact` | – | – | *no sweep* | – | – |
| **mean (implanted)** | | | **0.810** | 0.599 | 0.721 |

## gtsrb, poison rate 1%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | 0.994 | **0.995** | 0.994 | 0.997 |
| `blend` | 1.000 | 0.991 | **1.000** | 1.000 | 1.000 |
| `sig` | – | – | **0.255** | 0.013 | 0.051 |
| `wanet` ⚠ *never implanted* | 0.119 | 0.979 | **0.510** | 0.099 | 0.209 |
| `lf` | 0.979 | 0.990 | **0.971** | 0.915 | 0.961 |
| `lc` | – | – | **0.644** | 0.285 | 0.458 |
| `bpp` | 0.868 | 0.986 | **0.809** | 0.467 | 0.657 |
| `adaptive_blend` | 0.560 | 0.983 | **0.608** | 0.541 | 0.555 |
| `tact` | – | – | *no sweep* | – | – |
| **mean (implanted)** | | | **0.876** | 0.783 | 0.834 |

## tiny, poison rate 1%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 0.999 | 0.752 | **0.815** | 0.287 | 0.503 |
| `blend` | 0.999 | 0.760 | **0.974** | 0.982 | 0.994 |
| `sig` | – | – | **0.430** | 0.043 | 0.100 |
| `wanet` ⚠ *never implanted* | 0.178 | 0.747 | **0.451** | 0.199 | 0.262 |
| `lf` | 0.922 | 0.762 | **0.901** | 0.851 | 0.902 |
| `lc` | – | – | **0.593** | 0.199 | 0.286 |
| `bpp` | 0.971 | 0.748 | **0.969** | 0.967 | 0.973 |
| `adaptive_blend` | 0.554 | 0.747 | **0.598** | 0.553 | 0.559 |
| `tact` | – | – | *no sweep* | – | – |
| **mean (implanted)** | | | **0.851** | 0.728 | 0.786 |

## cifar10, poison rate 5%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | 0.945 | **0.967** | 0.918 | 0.978 |
| `blend` | 1.000 | 0.957 | **0.988** | 0.999 | 1.000 |
| `sig` | 0.599 | 0.953 | **0.678** | 0.463 | 0.521 |
| `wanet` | 0.961 | 0.913 | **0.918** | 0.793 | 0.889 |
| `lf` | 0.997 | 0.944 | **0.986** | 0.989 | 0.993 |
| `lc` ⚠ *never implanted* | 0.408 | 0.951 | **0.536** | 0.194 | 0.300 |
| `bpp` | 0.987 | 0.950 | **0.931** | 0.866 | 0.911 |
| `adaptive_blend` | 0.594 | 0.946 | **0.602** | 0.595 | 0.596 |
| `tact` | – | – | *no sweep* | – | – |
| **mean (implanted)** | | | **0.867** | 0.803 | 0.841 |

## cifar100, poison rate 5%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | 0.825 | **0.933** | 0.851 | 0.993 |
| `blend` | 1.000 | 0.830 | **0.986** | 0.983 | 0.995 |
| `sig` | – | – | **0.371** | 0.044 | 0.086 |
| `wanet` | 0.643 | 0.809 | **0.738** | 0.497 | 0.645 |
| `lf` | 0.953 | 0.829 | **0.927** | 0.892 | 0.944 |
| `lc` | 0.545 | 0.818 | **0.632** | 0.049 | 0.228 |
| `bpp` | 0.988 | 0.833 | **0.983** | 0.963 | 0.974 |
| `adaptive_blend` | 0.579 | 0.812 | **0.580** | 0.576 | 0.576 |
| `tact` | – | – | *no sweep* | – | – |
| **mean (implanted)** | | | **0.826** | 0.687 | 0.765 |

## gtsrb, poison rate 5%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | 0.986 | **1.000** | 1.000 | 1.000 |
| `blend` | 1.000 | 0.977 | **0.998** | 0.999 | 1.000 |
| `sig` | – | – | **0.335** | 0.013 | 0.039 |
| `wanet` | 0.830 | 0.971 | **0.700** | 0.232 | 0.448 |
| `lf` | 0.999 | 0.974 | **0.990** | 0.978 | 0.992 |
| `lc` ⚠ *never implanted* | 0.239 | 0.961 | **0.564** | 0.079 | 0.178 |
| `bpp` | 0.991 | 0.978 | **0.990** | 0.973 | 0.984 |
| `adaptive_blend` | 0.777 | 0.981 | **0.836** | 0.780 | 0.780 |
| `tact` | – | – | *no sweep* | – | – |
| **mean (implanted)** | | | **0.919** | 0.827 | 0.867 |

## tiny, poison rate 5%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | 0.752 | **0.842** | 0.361 | 0.750 |
| `blend` | 0.999 | 0.754 | **0.954** | 0.991 | 0.997 |
| `sig` | – | – | **0.551** | 0.138 | 0.271 |
| `wanet` | 0.922 | 0.748 | **0.896** | 0.774 | 0.877 |
| `lf` | 0.987 | 0.753 | **0.939** | 0.944 | 0.975 |
| `lc` | 0.706 | 0.756 | **0.777** | 0.701 | 0.747 |
| `bpp` | 0.985 | 0.746 | **0.970** | 0.978 | 0.984 |
| `adaptive_blend` | 0.783 | 0.745 | **0.789** | 0.782 | 0.784 |
| `tact` | – | – | *no sweep* | – | – |
| **mean (implanted)** | | | **0.881** | 0.790 | 0.874 |

## cifar10, poison rate 10%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | 0.942 | **0.961** | 0.885 | 0.965 |
| `blend` | 1.000 | 0.951 | **0.978** | 0.957 | 0.973 |
| `sig` | 0.901 | 0.846 | **0.878** | 0.783 | 0.825 |
| `wanet` | 0.890 | 0.946 | **0.432** | 0.090 | 0.172 |
| `lf` | 0.999 | 0.951 | **0.990** | 0.997 | 0.998 |
| `lc` | 0.979 | 0.861 | **0.947** | 0.914 | 0.965 |
| `bpp` | 0.994 | 0.954 | **0.929** | 0.862 | 0.904 |
| `adaptive_blend` | 0.622 | 0.951 | **0.622** | 0.623 | 0.623 |
| `tact` | – | – | *no sweep* | – | – |
| **mean (implanted)** | | | **0.842** | 0.764 | 0.803 |

## cifar100, poison rate 10%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | 0.821 | **0.962** | 0.991 | 1.000 |
| `blend` | 1.000 | 0.816 | **0.982** | 1.000 | 1.000 |
| `sig` | – | – | **0.402** | 0.049 | 0.107 |
| `wanet` | 0.793 | 0.818 | **0.708** | 0.240 | 0.485 |
| `lf` | 0.995 | 0.823 | **0.994** | 0.995 | 0.995 |
| `lc` | 0.785 | 0.824 | **0.760** | 0.152 | 0.521 |
| `bpp` | 0.991 | 0.817 | **0.975** | 0.954 | 0.972 |
| `adaptive_blend` | 0.604 | 0.808 | **0.608** | 0.600 | 0.601 |
| `tact` | – | – | *no sweep* | – | – |
| **mean (implanted)** | | | **0.856** | 0.705 | 0.796 |

## gtsrb, poison rate 10%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | 0.972 | **0.999** | 0.995 | 1.000 |
| `blend` | 1.000 | 0.988 | **1.000** | 1.000 | 1.000 |
| `sig` | – | – | **0.451** | 0.007 | 0.040 |
| `wanet` | 0.947 | 0.958 | **0.914** | 0.802 | 0.912 |
| `lf` | 0.999 | 0.988 | **0.997** | 0.995 | 0.998 |
| `lc` ⚠ *never implanted* | 0.129 | 0.989 | **0.522** | 0.166 | 0.249 |
| `bpp` | 0.995 | 0.986 | **0.988** | 0.971 | 0.983 |
| `adaptive_blend` | 0.838 | 0.976 | **0.880** | 0.839 | 0.839 |
| `tact` | – | – | *no sweep* | – | – |
| **mean (implanted)** | | | **0.963** | 0.934 | 0.955 |

## tiny, poison rate 10%

| attack | ASR | CA | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | 0.752 | **0.889** | 0.513 | 0.907 |
| `blend` | 1.000 | 0.749 | **0.961** | 0.987 | 0.997 |
| `sig` | – | – | **0.433** | 0.041 | 0.107 |
| `wanet` | 0.968 | 0.747 | **0.913** | 0.742 | 0.917 |
| `lf` | 0.973 | 0.740 | **0.911** | 0.859 | 0.932 |
| `lc` | 0.623 | 0.752 | **0.729** | 0.271 | 0.601 |
| `bpp` | 0.996 | 0.757 | **0.958** | 0.957 | 0.982 |
| `adaptive_blend` | 0.505 | 0.753 | **0.539** | 0.506 | 0.510 |
| `tact` | – | – | *no sweep* | – | – |
| **mean (implanted)** | | | **0.843** | 0.691 | 0.835 |

---
# PART 2 — best configuration per (attack, dataset, poison rate)

**This is an ORACLE and is not deployable.** A defender may guess the attack but can
never know the poison rate, so a configuration chosen per cell is not a defence. It is
printed to show the headroom that exists, and to expose where the choice is unstable.


## cifar10, poison rate 1% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 0.997 | `mlp_norm_out_gain_scale` | **0.996** | 0.997 | 0.997 | 49 |
| `blend` | 1.000 | `before_mlp_token_mask` | **0.999** | 0.999 | 0.999 | 51 |
| `sig` ⚠ | 0.340 | `before_mlp_residual` | **0.589** | 0.173 | 0.316 | 8 |
| `wanet` ⚠ | 0.112 | `before_attention_norm_token_mask` | **0.463** | 0.129 | 0.208 | 1 |
| `lf` | 0.982 | `before_attention_norm_token_mask` | **0.976** | 0.970 | 0.978 | 54 |
| `lc` ⚠ | 0.249 | `before_attention_norm_token_mask` | **0.531** | 0.159 | 0.244 | 8 |
| `bpp` | 0.982 | `before_attention_norm_token_mask` | **0.986** | 0.965 | 0.978 | 1 |
| `adaptive_blend` | 0.622 | `before_attention_norm_token_mask` | **0.629** | 0.622 | 0.623 | 1 |
| `tact` ⚠ | 0.121 | `before_attention_norm` | **0.520** | 0.083 | 0.177 | 7 |

## cifar100, poison rate 1% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | `before_mlp_token_mask` | **0.983** | 0.994 | 0.998 | 47 |
| `blend` | 0.989 | `after_mlp_residual_token_mask` | **0.953** | 0.927 | 0.981 | 47 |
| `sig` | – | `before_attention_norm_token_mask` | **0.425** | 0.039 | 0.099 | 1 |
| `wanet` ⚠ | 0.057 | `before_attention_norm_token_mask` | **0.470** | 0.115 | 0.204 | 1 |
| `lf` | 0.942 | `pre_residual_blocks_9_12` | **0.970** | 0.936 | 0.954 | 45 |
| `lc` | 0.873 | `input_pixels_scale_up` | **0.840** | 0.671 | 0.868 | 44 |
| `bpp` | 0.914 | `before_attention_norm_token_mask` | **0.773** | 0.211 | 0.610 | 1 |
| `adaptive_blend` | 0.536 | `before_attention_norm_token_mask` | **0.546** | 0.529 | 0.536 | 1 |
| `tact` | – | *not swept* | – | – | – | 0 |

## gtsrb, poison rate 1% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | `mlp_norm_out_gain_scale` | **1.000** | 1.000 | 1.000 | 53 |
| `blend` | 1.000 | `before_attention_gaussian_batchstd` | **1.000** | 1.000 | 1.000 | 53 |
| `sig` | – | `before_attention_norm_token_mask` | **0.255** | 0.013 | 0.051 | 1 |
| `wanet` ⚠ | 0.119 | `before_attention_norm_token_mask` | **0.510** | 0.099 | 0.209 | 1 |
| `lf` | 0.979 | `after_attention_residual_gaussian_batchstd` | **0.989** | 0.980 | 0.986 | 51 |
| `lc` | – | `before_attention_norm_token_mask` | **0.644** | 0.285 | 0.458 | 1 |
| `bpp` | 0.868 | `before_attention_norm_token_mask` | **0.809** | 0.467 | 0.657 | 1 |
| `adaptive_blend` | 0.560 | `before_attention_norm_token_mask` | **0.608** | 0.541 | 0.555 | 1 |
| `tact` | – | *not swept* | – | – | – | 0 |

## tiny, poison rate 1% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 0.999 | `after_mlp_residual_gaussian_batchstd` | **0.968** | 0.951 | 0.992 | 56 |
| `blend` | 0.999 | `pre_residual_blocks_9_12` | **0.999** | 0.998 | 0.999 | 56 |
| `sig` | – | `before_attention_norm_token_mask` | **0.430** | 0.043 | 0.100 | 1 |
| `wanet` ⚠ | 0.178 | `before_attention_norm_token_mask` | **0.451** | 0.199 | 0.262 | 1 |
| `lf` | 0.922 | `pre_residual_blocks_9_12` | **0.959** | 0.919 | 0.934 | 54 |
| `lc` | – | `before_attention_norm_token_mask` | **0.593** | 0.199 | 0.286 | 1 |
| `bpp` | 0.971 | `before_attention_norm_token_mask` | **0.969** | 0.967 | 0.973 | 1 |
| `adaptive_blend` | 0.554 | `before_attention_norm_token_mask` | **0.598** | 0.553 | 0.559 | 1 |
| `tact` | – | *not swept* | – | – | – | 0 |

## cifar10, poison rate 5% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | `mlp_norm_out_gain_scale` | **0.983** | 0.999 | 1.000 | 50 |
| `blend` | 1.000 | `mlp_norm_out_gain_scale` | **0.999** | 1.000 | 1.000 | 51 |
| `sig` | 0.599 | `after_attention_residual_token_mask` | **0.766** | 0.259 | 0.569 | 35 |
| `wanet` | 0.961 | `before_attention_norm_token_mask` | **0.918** | 0.793 | 0.889 | 1 |
| `lf` | 0.997 | `pre_residual_blocks_9_12` | **0.992** | 0.986 | 0.993 | 56 |
| `lc` ⚠ | 0.408 | `before_attention_norm_token_mask` | **0.536** | 0.194 | 0.300 | 3 |
| `bpp` | 0.987 | `before_attention_norm_token_mask` | **0.931** | 0.866 | 0.911 | 1 |
| `adaptive_blend` | 0.594 | `before_attention_norm_token_mask` | **0.602** | 0.595 | 0.596 | 1 |
| `tact` | – | *not swept* | – | – | – | 0 |

## cifar100, poison rate 5% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | `after_attention_residual_token_mask` | **0.986** | 0.979 | 0.997 | 44 |
| `blend` | 1.000 | `pre_residual_blocks_9_12` | **1.000** | 1.000 | 1.000 | 44 |
| `sig` | – | `before_attention_norm_token_mask` | **0.371** | 0.044 | 0.086 | 1 |
| `wanet` | 0.643 | `before_attention_norm_token_mask` | **0.738** | 0.497 | 0.645 | 1 |
| `lf` | 0.953 | `pre_residual_blocks_9_12` | **0.970** | 0.944 | 0.957 | 42 |
| `lc` | 0.545 | `final_norm_out_gain_scale` | **0.777** | 0.218 | 0.445 | 38 |
| `bpp` | 0.988 | `before_attention_norm_token_mask` | **0.983** | 0.963 | 0.974 | 1 |
| `adaptive_blend` | 0.579 | `before_attention_norm_token_mask` | **0.580** | 0.576 | 0.576 | 1 |
| `tact` | – | *not swept* | – | – | – | 0 |

## gtsrb, poison rate 5% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | `mlp_neurons_gaussian_batchstd` | **1.000** | 1.000 | 1.000 | 53 |
| `blend` | 1.000 | `before_mlp_gaussian_batchstd` | **0.999** | 1.000 | 1.000 | 53 |
| `sig` | – | `before_attention_norm_token_mask` | **0.335** | 0.013 | 0.039 | 1 |
| `wanet` | 0.830 | `before_attention_norm_token_mask` | **0.700** | 0.232 | 0.448 | 1 |
| `lf` | 0.999 | `pre_residual_blocks_5_8` | **0.999** | 0.999 | 0.999 | 51 |
| `lc` ⚠ | 0.239 | `before_attention_norm_token_mask` | **0.564** | 0.079 | 0.178 | 3 |
| `bpp` | 0.991 | `before_attention_norm_token_mask` | **0.990** | 0.973 | 0.984 | 1 |
| `adaptive_blend` | 0.777 | `before_attention_norm_token_mask` | **0.836** | 0.780 | 0.780 | 1 |
| `tact` | – | *not swept* | – | – | – | 0 |

## tiny, poison rate 5% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | `pre_residual_blocks_9_12` | **0.981** | 0.971 | 0.994 | 53 |
| `blend` | 0.999 | `pre_residual_blocks_9_12` | **1.000** | 0.999 | 0.999 | 53 |
| `sig` | – | `before_attention_norm_token_mask` | **0.551** | 0.138 | 0.271 | 1 |
| `wanet` | 0.922 | `before_attention_norm_token_mask` | **0.896** | 0.774 | 0.877 | 1 |
| `lf` | 0.987 | `pre_residual_blocks_9_12` | **0.993** | 0.987 | 0.990 | 51 |
| `lc` | 0.706 | `after_mlp_residual_gaussian_batchstd` | **0.792** | 0.470 | 0.694 | 45 |
| `bpp` | 0.985 | `before_attention_norm_token_mask` | **0.970** | 0.978 | 0.984 | 1 |
| `adaptive_blend` | 0.783 | `before_attention_norm_token_mask` | **0.789** | 0.782 | 0.784 | 1 |
| `tact` | – | *not swept* | – | – | – | 0 |

## cifar10, poison rate 10% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | `mlp_norm_out_gain_scale` | **0.999** | 1.000 | 1.000 | 59 |
| `blend` | 1.000 | `before_mlp_token_mask` | **0.999** | 1.000 | 1.000 | 58 |
| `sig` | 0.901 | `before_mlp_residual` | **0.933** | 0.888 | 0.911 | 49 |
| `wanet` | 0.890 | `before_attention_norm_token_mask` | **0.432** | 0.090 | 0.172 | 1 |
| `lf` | 0.999 | `pre_residual_blocks_9_12` | **0.998** | 0.997 | 0.998 | 57 |
| `lc` | 0.979 | `before_attention_token_mask` | **0.975** | 0.979 | 0.979 | 51 |
| `bpp` | 0.994 | `before_attention_norm_token_mask` | **0.929** | 0.862 | 0.904 | 1 |
| `adaptive_blend` | 0.622 | `before_attention_norm_token_mask` | **0.622** | 0.623 | 0.623 | 1 |
| `tact` | – | `mlp_neurons_gaussian` | **0.480** | 0.089 | 0.187 | 2 |

## cifar100, poison rate 10% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | `after_mlp_residual_token_mask` | **0.989** | 0.987 | 0.993 | 42 |
| `blend` | 1.000 | `after_attention_residual_token_mask` | **1.000** | 1.000 | 1.000 | 42 |
| `sig` | – | `before_attention_norm_token_mask` | **0.402** | 0.049 | 0.107 | 1 |
| `wanet` | 0.793 | `before_attention_norm_token_mask` | **0.708** | 0.240 | 0.485 | 1 |
| `lf` | 0.995 | `pre_residual_blocks_9_12` | **0.998** | 0.996 | 0.997 | 39 |
| `lc` | 0.785 | `after_mlp_residual_token_mask` | **0.804** | 0.404 | 0.688 | 36 |
| `bpp` | 0.991 | `before_attention_norm_token_mask` | **0.975** | 0.954 | 0.972 | 1 |
| `adaptive_blend` | 0.604 | `before_attention_norm_token_mask` | **0.608** | 0.600 | 0.601 | 1 |
| `tact` | – | *not swept* | – | – | – | 0 |

## gtsrb, poison rate 10% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | `after_mlp_residual_gaussian_batchstd` | **1.000** | 1.000 | 1.000 | 51 |
| `blend` | 1.000 | `pre_residual_blocks_9_12` | **1.000** | 1.000 | 1.000 | 51 |
| `sig` | – | `before_attention_norm_token_mask` | **0.451** | 0.007 | 0.040 | 1 |
| `wanet` | 0.947 | `before_attention_norm_token_mask` | **0.914** | 0.802 | 0.912 | 1 |
| `lf` | 0.999 | `pre_residual_blocks_9_12` | **1.000** | 0.999 | 0.999 | 49 |
| `lc` ⚠ | 0.129 | `before_attention_norm_token_mask` | **0.522** | 0.166 | 0.249 | 3 |
| `bpp` | 0.995 | `before_attention_norm_token_mask` | **0.988** | 0.971 | 0.983 | 1 |
| `adaptive_blend` | 0.838 | `before_attention_norm_token_mask` | **0.880** | 0.839 | 0.839 | 1 |
| `tact` | – | *not swept* | – | – | – | 0 |

## tiny, poison rate 10% — best available configuration

| attack | ASR | best config | AUROC | TPR@10%FPR | TPR@20%FPR | n configs |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 1.000 | `pre_residual_blocks_9_12` | **0.999** | 0.999 | 1.000 | 51 |
| `blend` | 1.000 | `pre_residual_blocks_9_12` | **1.000** | 1.000 | 1.000 | 51 |
| `sig` | – | `before_attention_norm_token_mask` | **0.433** | 0.041 | 0.107 | 1 |
| `wanet` | 0.968 | `before_attention_norm_token_mask` | **0.913** | 0.742 | 0.917 | 1 |
| `lf` | 0.973 | `pre_residual_blocks_9_12` | **0.986** | 0.973 | 0.978 | 49 |
| `lc` | 0.623 | `before_attention_norm_token_mask` | **0.729** | 0.271 | 0.601 | 43 |
| `bpp` | 0.996 | `before_attention_norm_token_mask` | **0.958** | 0.957 | 0.982 | 1 |
| `adaptive_blend` | 0.505 | `before_attention_norm_token_mask` | **0.539** | 0.506 | 0.510 | 1 |
| `tact` | – | *not swept* | – | – | – | 0 |

---
# PART 3 — is there ONE configuration that handles the hard attacks?

Hard attacks = adaptive_blend, lc, wanet, sig. Ranked by mean AUROC over every hard
cell that implanted and has that configuration cached. `n` is the coverage, and an
uneven `n` is why the raw ranking cannot be trusted on its own.

| configuration | n hard cells | AUROC | TPR@10%FPR | TPR@20%FPR |
|---|---|---|---|---|
| `before_attention_residual_token_mask` | 8 | **0.775** | 0.506 | 0.660 |
| `both_sublayer_inputs_token_mask` | 6 | **0.748** | 0.327 | 0.575 |
| `input_pixels_scale_up` | 8 | **0.735** | 0.452 | 0.595 |
| `before_attention_token_mask` | 8 | **0.720** | 0.543 | 0.622 |
| `before_attention_norm_token_mask` ← deployed | 28 | **0.720** | 0.539 | 0.629 |
| `before_mlp_residual_token_mask` | 8 | **0.708** | 0.307 | 0.458 |
| `mlp_neurons_token_mask` | 8 | **0.691** | 0.284 | 0.404 |
| `attention_heads_head_mask` | 8 | **0.690** | 0.330 | 0.484 |
| `before_attention_residual_droppath` | 8 | **0.688** | 0.314 | 0.475 |
| `after_attention_residual_token_mask` | 8 | **0.687** | 0.201 | 0.410 |
| `after_mlp_residual_token_mask` | 8 | **0.687** | 0.220 | 0.431 |
| `before_attention_residual_channel_mask` | 8 | **0.664** | 0.293 | 0.423 |

### Stability across poison rate, hard attacks only

A configuration whose rank moves with the poison rate is unusable, because the
defender cannot observe the poison rate.

| configuration | 1% | 5% | 10% | max spread |
|---|---|---|---|---|
| `before_attention_residual_token_mask` | – | 0.675 | 0.840 | *incomplete* |
| `both_sublayer_inputs_token_mask` | – | 0.688 | 0.784 | *incomplete* |
| `input_pixels_scale_up` | – | 0.643 | 0.778 | *incomplete* |
| `before_attention_token_mask` | – | 0.615 | 0.769 | *incomplete* |
| `before_attention_norm_token_mask` | 0.614 | 0.740 | 0.744 | **0.130** |
| `before_mlp_residual_token_mask` | – | 0.607 | 0.781 | *incomplete* |
| `mlp_neurons_token_mask` | – | 0.601 | 0.758 | *incomplete* |
| `attention_heads_head_mask` | – | 0.632 | 0.742 | *incomplete* |
| `before_attention_residual_droppath` | – | 0.614 | 0.761 | *incomplete* |
| `after_attention_residual_token_mask` | – | 0.674 | 0.756 | *incomplete* |
| `after_mlp_residual_token_mask` | – | 0.631 | 0.715 | *incomplete* |
| `before_attention_residual_channel_mask` | – | 0.593 | 0.723 | *incomplete* |
