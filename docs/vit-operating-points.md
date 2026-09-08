# ViT-B/16 detection at deployment operating points

`token_mask @ before_attention_norm`, the recommended configuration. Probe rate chosen
per cell as the one whose clean-validation shift ratio is closest to 0.8. PSU variant
fractional, one-sided. Base checkpoints only: `_evade_*` folders are adversarially
trained to defeat PSBD and `_sam_rho_*` use a different optimizer, so both are excluded.
Generated from `results/operating_points.csv`.

## Poison rate 10% — ViT-B/16, token_mask @ before_attention_norm

Rate chosen per cell as the one whose clean-validation shift ratio is closest to 0.8. PSU variant: fractional, one-sided. Base checkpoints only.

| dataset | attack | ASR | rate | sigma | AUROC | TPR@10%FPR | act.FPR | TPR@25%FPR | act.FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | adaptive_blend | 0.927 | 0.7 | 0.80 | 0.951 | 0.925 | 0.065 | 0.929 | 0.196 |
| cifar10 | badnet_a2a | 0.958 | 0.7 | 0.79 | 0.440 | 0.030 | 0.097 | 0.130 | 0.253 |
| cifar10 | badnet_a2o | 1.000 | 0.6 | 0.81 | 0.991 | 0.935 | 0.018 | 1.000 | 0.173 |
| cifar10 | blend | 1.000 | 0.6 | 0.82 | 0.906 | 0.754 | 0.114 | 0.881 | 0.288 |
| cifar10 | bpp | 0.999 | 0.6 | 0.78 | 0.993 | 0.977 | 0.085 | 0.992 | 0.212 |
| cifar10 | lc | 0.979 | 0.7 | 0.78 | 0.921 | 0.772 | 0.116 | 0.939 | 0.273 |
| cifar10 | lf | 0.999 | 0.6 | 0.86 | 0.993 | 0.995 | 0.110 | 0.999 | 0.235 |
| cifar10 | sig | 0.901 | 0.7 | 0.79 | 0.673 | 0.346 | 0.096 | 0.448 | 0.249 |
| cifar10 | wanet | 0.962 | 0.6 | 0.78 | 0.841 | 0.551 | 0.114 | 0.811 | 0.281 |
| cifar100 | adaptive_blend | 0.971 | 0.5 | 0.81 | 0.966 | 0.966 | 0.096 | 0.970 | 0.236 |
| cifar100 | badnet_a2a | 0.789 | 0.5 | 0.87 | 0.377 | 0.071 | 0.097 | 0.177 | 0.258 |
| cifar100 | badnet_a2o | 1.000 | 0.4 | 0.80 | 0.989 | 1.000 | 0.098 | 1.000 | 0.244 |
| cifar100 | blend | 1.000 | 0.5 | 0.86 | 0.998 | 1.000 | 0.095 | 1.000 | 0.262 |
| cifar100 | bpp | 0.994 | 0.4 | 0.75 | 0.993 | 0.990 | 0.089 | 0.995 | 0.232 |
| cifar100 | lc | 0.785 | 0.5 | 0.83 | 0.816 | 0.431 | 0.097 | 0.783 | 0.243 |
| cifar100 | lf | 0.995 | 0.5 | 0.89 | 0.995 | 0.995 | 0.092 | 0.995 | 0.242 |
| cifar100 | wanet | 0.900 | 0.4 | 0.79 | 0.900 | 0.765 | 0.097 | 0.892 | 0.251 |
| gtsrb | adaptive_blend | 1.000 | 0.4 | 0.91 | 1.000 | 1.000 | 0.100 | 1.000 | 0.242 |
| gtsrb | badnet_a2a | 0.983 | 0.3 | 0.86 | 0.388 | 0.058 | 0.089 | 0.158 | 0.245 |
| gtsrb | badnet_a2o | 1.000 | 0.5 | 0.89 | 1.000 | 1.000 | 0.094 | 1.000 | 0.261 |
| gtsrb | blend | 1.000 | 0.3 | 0.86 | 1.000 | 1.000 | 0.085 | 1.000 | 0.242 |
| gtsrb | bpp | 0.998 | 0.4 | 0.86 | 0.998 | 0.998 | 0.089 | 0.998 | 0.250 |
| gtsrb | lf | 0.999 | 0.5 | 0.83 | 0.998 | 0.998 | 0.096 | 0.999 | 0.253 |
| gtsrb | wanet | 0.922 | 0.5 | 0.82 | 0.927 | 0.885 | 0.095 | 0.931 | 0.238 |
| tiny | adaptive_blend | 0.965 | 0.5 | 0.86 | 0.965 | 0.966 | 0.110 | 0.966 | 0.250 |
| tiny | badnet_a2a | 0.734 | 0.5 | 0.84 | 0.392 | 0.026 | 0.103 | 0.134 | 0.252 |
| tiny | badnet_a2o | 1.000 | 0.4 | 0.74 | 0.964 | 0.988 | 0.101 | 1.000 | 0.256 |
| tiny | blend | 1.000 | 0.4 | 0.71 | 0.992 | 0.996 | 0.097 | 0.999 | 0.252 |
| tiny | bpp | 0.995 | 0.5 | 0.85 | 0.994 | 0.991 | 0.099 | 0.996 | 0.256 |
| tiny | lc | 0.623 | 0.5 | 0.87 | 0.765 | 0.545 | 0.109 | 0.720 | 0.249 |
| tiny | lf | 0.973 | 0.5 | 0.87 | 0.973 | 0.973 | 0.109 | 0.978 | 0.245 |
| tiny | wanet | 0.975 | 0.4 | 0.73 | 0.955 | 0.937 | 0.116 | 0.976 | 0.254 |
| **mean** | **32 valid cells** | | | | **0.877** | **0.777** | | **0.837** | |
| **worst** | | | | | **0.377** | **0.026** | | **0.130** | |

⚠ = ASR below 0.5, so the backdoor was never reliably implanted; excluded from mean/worst.

## Poison rate 5% — ViT-B/16, token_mask @ before_attention_norm

Rate chosen per cell as the one whose clean-validation shift ratio is closest to 0.8. PSU variant: fractional, one-sided. Base checkpoints only.

| dataset | attack | ASR | rate | sigma | AUROC | TPR@10%FPR | act.FPR | TPR@25%FPR | act.FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | adaptive_blend | 0.837 | 0.6 | 0.77 | 0.836 | 0.837 | 0.037 | 0.837 | 0.169 |
| cifar10 | badnet_a2a | 0.933 | 0.6 | 0.80 | 0.432 | 0.018 | 0.106 | 0.164 | 0.265 |
| cifar10 | badnet_a2o | 1.000 | 0.7 | 0.82 | 0.992 | 0.965 | 0.048 | 1.000 | 0.172 |
| cifar10 | blend | 1.000 | 0.6 | 0.76 | 0.991 | 0.995 | 0.112 | 0.999 | 0.255 |
| cifar10 | bpp | 0.997 | 0.6 | 0.76 | 0.991 | 0.979 | 0.111 | 0.993 | 0.267 |
| cifar10 | sig | 0.599 | 0.7 | 0.83 | 0.740 | 0.424 | 0.073 | 0.581 | 0.192 |
| cifar10 | wanet | 0.786 | 0.5 | 0.77 | 0.693 | 0.085 | 0.108 | 0.532 | 0.249 |
| cifar100 | adaptive_blend | 0.932 | 0.5 | 0.86 | 0.929 | 0.912 | 0.102 | 0.930 | 0.253 |
| cifar100 | badnet_a2a | 0.801 | 0.5 | 0.87 | 0.384 | 0.015 | 0.082 | 0.099 | 0.243 |
| cifar100 | badnet_a2o | 1.000 | 0.4 | 0.80 | 0.977 | 0.995 | 0.104 | 1.000 | 0.264 |
| cifar100 | blend | 1.000 | 0.5 | 0.77 | 0.989 | 0.977 | 0.093 | 0.994 | 0.249 |
| cifar100 | bpp | 0.984 | 0.5 | 0.86 | 0.989 | 0.978 | 0.104 | 0.987 | 0.258 |
| cifar100 | lc | 0.545 | 0.5 | 0.84 | 0.689 | 0.160 | 0.092 | 0.521 | 0.241 |
| cifar100 | lf | 0.953 | 0.4 | 0.74 | 0.957 | 0.949 | 0.101 | 0.956 | 0.241 |
| cifar100 | wanet | 0.650 | 0.4 | 0.82 | 0.774 | 0.626 | 0.093 | 0.705 | 0.262 |
| gtsrb | adaptive_blend | 0.986 | 0.4 | 0.74 | 0.986 | 0.986 | 0.100 | 0.986 | 0.238 |
| gtsrb | badnet_a2a | 0.931 | 0.4 | 0.90 | 0.312 | 0.055 | 0.085 | 0.070 | 0.238 |
| gtsrb | badnet_a2o | 1.000 | 0.3 | 0.90 | 1.000 | 1.000 | 0.097 | 1.000 | 0.248 |
| gtsrb | blend | 1.000 | 0.4 | 0.92 | 0.999 | 1.000 | 0.107 | 1.000 | 0.254 |
| gtsrb | bpp | 0.990 | 0.6 | 0.90 | 0.992 | 0.990 | 0.084 | 0.990 | 0.248 |
| gtsrb | lf | 0.999 | 0.4 | 0.78 | 0.996 | 0.991 | 0.090 | 0.999 | 0.259 |
| gtsrb | wanet | 0.777 | 0.5 | 0.92 | 0.873 | 0.787 | 0.090 | 0.816 | 0.239 |
| tiny | adaptive_blend | 0.924 | 0.4 | 0.72 | 0.920 | 0.924 | 0.108 | 0.924 | 0.271 |
| tiny | badnet_a2a | 0.722 | 0.4 | 0.72 | 0.470 | 0.029 | 0.105 | 0.227 | 0.249 |
| tiny | badnet_a2o | 1.000 | 0.5 | 0.88 | 0.984 | 0.999 | 0.107 | 1.000 | 0.261 |
| tiny | blend | 0.999 | 0.4 | 0.73 | 0.989 | 0.996 | 0.102 | 0.999 | 0.264 |
| tiny | bpp | 0.988 | 0.5 | 0.86 | 0.989 | 0.987 | 0.116 | 0.991 | 0.255 |
| tiny | lc | 0.706 | 0.4 | 0.71 | 0.777 | 0.701 | 0.113 | 0.763 | 0.255 |
| tiny | lf | 0.987 | 0.4 | 0.76 | 0.976 | 0.982 | 0.105 | 0.988 | 0.244 |
| tiny | wanet | 0.940 | 0.4 | 0.73 | 0.934 | 0.910 | 0.099 | 0.948 | 0.243 |
| **mean** | **30 valid cells** | | | | **0.852** | **0.742** | | **0.800** | |
| **worst** | | | | | **0.312** | **0.015** | | **0.070** | |

⚠ = ASR below 0.5, so the backdoor was never reliably implanted; excluded from mean/worst.
