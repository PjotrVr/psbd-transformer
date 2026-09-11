# Which probes union best, and the best way to combine them, ViT-B/16, 2026-09-11

For the authors, not for the paper. PSBD-TM is token mask, attention input, the recommended placement. Every placement is read at its own 0.8 clean shift adaptive rate, fractional PSU, the min-rank union at the calibrated 10% and 20% clean validation quantiles. CA is clean accuracy, the benign row is the reference model trained with the same recipe. Rows with ASR below 0.85 are marked with a dagger and are outside the paper's evaluation.

## CIFAR-10, 1% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.953 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.952 | 0.997 | 0.982 | 0.959 | 0.978 | 0.994 | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 |
| Blend | 0.954 | 1.000 | 0.922 | 0.773 | 0.869 | 0.972 | 0.919 | 0.978 | 0.983 | 0.966 | 0.984 |
| LF | 0.953 | 0.982 | 0.980 | 0.953 | 0.975 | 0.979 | 0.953 | 0.970 | 0.980 | 0.955 | 0.971 |
| BPP | 0.952 | 0.982 | 0.985 | 0.955 | 0.976 | 0.980 | 0.936 | 0.965 | 0.966 | 0.850 | 0.945 |
| TaCT | 0.938 | 0.985 | 0.979 | 0.005 | 0.009 | 0.873 | 0.102 | 0.269 | 0.922 | 0.146 | 0.229 |

## CIFAR-100, 1% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.811 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.825 | 1.000 | 0.988 | 0.999 | 1.000 | 0.976 | 0.999 | 1.000 | 0.996 | 1.000 | 1.000 |
| Blend | 0.829 | 0.989 | 0.976 | 0.958 | 0.983 | 0.963 | 0.901 | 0.976 | 0.967 | 0.888 | 0.979 |
| LF | 0.830 | 0.942 | 0.956 | 0.942 | 0.947 | 0.958 | 0.944 | 0.949 | 0.956 | 0.939 | 0.950 |
| BPP | 0.824 | 0.914 | 0.921 | 0.849 | 0.919 | 0.899 | 0.693 | 0.870 | 0.873 | 0.446 | 0.818 |
| TaCT | 0.820 | 0.989 | 0.908 | 0.506 | 0.690 | 0.856 | 0.471 | 0.632 | 0.856 | 0.414 | 0.621 |

## GTSRB, 1% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.991 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.994 | 1.000 | 0.999 | 0.996 | 0.999 | 0.998 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| Blend | 0.991 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| LF | 0.990 | 0.979 | 0.976 | 0.926 | 0.972 | 0.966 | 0.900 | 0.935 | 0.941 | 0.882 | 0.919 |
| BPP | 0.986 | 0.868 | 0.898 | 0.717 | 0.809 | 0.855 | 0.646 | 0.739 | 0.845 | 0.599 | 0.664 |

## Tiny ImageNet, 1% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.755 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.752 | 0.999 | 0.980 | 0.989 | 0.997 | 0.976 | 0.999 | 0.999 | 0.992 | 0.999 | 0.999 |
| Blend | 0.760 | 0.999 | 0.993 | 0.985 | 0.992 | 0.992 | 0.990 | 0.996 | 0.994 | 0.991 | 0.996 |
| LF | 0.762 | 0.922 | 0.941 | 0.929 | 0.935 | 0.950 | 0.926 | 0.938 | 0.943 | 0.925 | 0.938 |
| BPP | 0.748 | 0.971 | 0.980 | 0.972 | 0.975 | 0.979 | 0.972 | 0.977 | 0.967 | 0.971 | 0.977 |
| TaCT | 0.750 | 0.976 | 0.576 | 0.024 | 0.119 | 0.562 | 0.000 | 0.048 | 0.558 | 0.000 | 0.048 |

## CIFAR-10, 5% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.953 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.945 | 1.000 | 0.992 | 0.965 | 0.995 | 0.994 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 |
| Blend | 0.957 | 1.000 | 0.971 | 0.905 | 0.989 | 0.975 | 0.956 | 0.996 | 0.976 | 0.986 | 0.998 |
| LF | 0.944 | 0.997 | 0.991 | 0.983 | 0.994 | 0.989 | 0.978 | 0.995 | 0.993 | 0.991 | 0.997 |
| BPP | 0.950 | 0.987 | 0.801 | 0.587 | 0.655 | 0.741 | 0.516 | 0.591 | 0.741 | 0.472 | 0.548 |
| WaNet | 0.913 | 0.961 | 0.937 | 0.838 | 0.913 | 0.940 | 0.792 | 0.917 | 0.953 | 0.880 | 0.950 |
| TaCT | 0.946 | 0.996 | 0.966 | 0.000 | 0.031 | 0.987 | 0.454 | 0.832 | 0.989 | 0.308 | 0.732 |

## CIFAR-100, 5% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.811 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.825 | 1.000 | 0.994 | 0.999 | 1.000 | 0.989 | 1.000 | 1.000 | 0.996 | 1.000 | 1.000 |
| Blend | 0.830 | 1.000 | 0.979 | 0.941 | 0.976 | 0.978 | 0.943 | 0.984 | 0.980 | 0.924 | 0.988 |
| LF | 0.829 | 0.953 | 0.967 | 0.954 | 0.959 | 0.971 | 0.953 | 0.958 | 0.968 | 0.950 | 0.958 |
| BPP | 0.833 | 0.988 | 0.985 | 0.961 | 0.977 | 0.983 | 0.957 | 0.971 | 0.984 | 0.955 | 0.970 |
| TaCT | 0.825 | 1.000 | 0.933 | 0.540 | 0.862 | 0.955 | 0.851 | 0.977 | 0.963 | 0.943 | 0.989 |

## GTSRB, 5% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.991 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.986 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Blend | 0.977 | 1.000 | 0.999 | 1.000 | 1.000 | 0.998 | 0.999 | 1.000 | 0.995 | 0.998 | 1.000 |
| LF | 0.974 | 0.999 | 0.998 | 0.999 | 0.999 | 0.997 | 0.993 | 0.999 | 0.996 | 0.989 | 0.998 |
| BPP | 0.978 | 0.991 | 0.992 | 0.978 | 0.987 | 0.988 | 0.973 | 0.979 | 0.989 | 0.970 | 0.976 |
| TaCT | 0.986 | 1.000 | 0.942 | 0.502 | 0.817 | 1.000 | 0.248 | 0.549 | 1.000 | 0.856 | 0.939 |

## Tiny ImageNet, 5% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.755 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.752 | 1.000 | 0.984 | 0.999 | 1.000 | 0.982 | 1.000 | 1.000 | 0.996 | 1.000 | 1.000 |
| Blend | 0.754 | 0.999 | 0.998 | 0.997 | 0.999 | 0.996 | 0.996 | 0.999 | 0.995 | 0.996 | 0.999 |
| LF | 0.753 | 0.987 | 0.987 | 0.987 | 0.988 | 0.986 | 0.987 | 0.989 | 0.985 | 0.987 | 0.989 |
| BPP | 0.746 | 0.985 | 0.986 | 0.984 | 0.986 | 0.984 | 0.981 | 0.986 | 0.957 | 0.982 | 0.986 |
| WaNet | 0.748 | 0.922 | 0.930 | 0.924 | 0.928 | 0.944 | 0.928 | 0.938 | 0.937 | 0.926 | 0.939 |
| TaCT | 0.743 | 0.952 | 0.663 | 0.381 | 0.500 | 0.622 | 0.286 | 0.405 | 0.629 | 0.238 | 0.405 |

## SVHN, 5% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.964 |  |  |  |  |  |  |  |  |  |  |
| SIG | 0.958 | 0.880 | 0.797 | 0.118 | 0.478 | 0.832 | 0.345 | 0.529 | 0.910 | 0.598 | 0.773 |

## CIFAR-10, 10% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.953 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.942 | 1.000 | 0.991 | 0.934 | 1.000 | 0.990 | 1.000 | 1.000 | 0.992 | 1.000 | 1.000 |
| Blend | 0.951 | 1.000 | 0.906 | 0.754 | 0.849 | 0.960 | 0.875 | 0.946 | 0.969 | 0.919 | 0.963 |
| LF | 0.951 | 0.999 | 0.993 | 0.995 | 0.999 | 0.988 | 0.990 | 0.998 | 0.985 | 0.994 | 0.998 |
| BPP | 0.954 | 0.994 | 0.871 | 0.735 | 0.830 | 0.828 | 0.681 | 0.738 | 0.820 | 0.665 | 0.718 |
| WaNet | 0.946 | 0.890 | 0.459 | 0.085 | 0.267 | 0.644 | 0.217 | 0.357 | 0.623 | 0.236 | 0.366 |
| TaCT | 0.858 | 1.000 | 0.963 | 0.030 | 0.206 | 0.975 | 0.631 | 0.971 | 0.983 | 0.592 | 0.971 |
| SIG | 0.846 | 0.901 | 0.418 | 0.071 | 0.097 | 0.690 | 0.136 | 0.325 | 0.638 | 0.104 | 0.312 |

## CIFAR-100, 10% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.811 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.821 | 1.000 | 0.998 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 |
| Blend | 0.816 | 1.000 | 0.998 | 1.000 | 1.000 | 0.997 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 |
| LF | 0.823 | 0.995 | 0.995 | 0.995 | 0.995 | 0.994 | 0.996 | 0.996 | 0.993 | 0.995 | 0.996 |
| BPP | 0.817 | 0.991 | 0.985 | 0.963 | 0.975 | 0.981 | 0.952 | 0.969 | 0.979 | 0.952 | 0.968 |
| TaCT | 0.821 | 0.989 | 0.890 | 0.161 | 0.586 | 0.953 | 0.299 | 0.828 | 0.956 | 0.621 | 0.897 |

## GTSRB, 10% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.991 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.972 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Blend | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| LF | 0.988 | 0.999 | 0.998 | 0.998 | 0.999 | 0.996 | 0.995 | 0.998 | 0.989 | 0.992 | 0.997 |
| BPP | 0.986 | 0.995 | 0.989 | 0.971 | 0.986 | 0.985 | 0.959 | 0.975 | 0.927 | 0.951 | 0.972 |
| WaNet | 0.958 | 0.947 | 0.948 | 0.949 | 0.952 | 0.932 | 0.946 | 0.952 | 0.942 | 0.945 | 0.950 |
| TaCT | 0.931 | 1.000 | 0.748 | 0.213 | 0.843 | 0.752 | 0.002 | 0.311 | 0.766 | 0.693 | 0.886 |

## Tiny ImageNet, 10% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.755 |  |  |  |  |  |  |  |  |  |  |
| BadNets | 0.752 | 1.000 | 0.993 | 1.000 | 1.000 | 0.989 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 |
| Blend | 0.749 | 1.000 | 0.997 | 0.999 | 0.999 | 0.997 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 |
| LF | 0.740 | 0.973 | 0.973 | 0.973 | 0.977 | 0.970 | 0.964 | 0.977 | 0.968 | 0.964 | 0.977 |
| BPP | 0.757 | 0.996 | 0.984 | 0.981 | 0.991 | 0.977 | 0.961 | 0.984 | 0.947 | 0.960 | 0.979 |
| WaNet | 0.747 | 0.968 | 0.950 | 0.931 | 0.970 | 0.930 | 0.795 | 0.954 | 0.928 | 0.749 | 0.953 |
| TaCT | 0.756 | 0.929 | 0.809 | 0.381 | 0.619 | 0.792 | 0.143 | 0.452 | 0.804 | 0.143 | 0.500 |

## SVHN, 10% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.964 |  |  |  |  |  |  |  |  |  |  |
| Blend | 0.951 | 1.000 | 0.941 | 0.731 | 0.884 | 0.935 | 0.818 | 0.846 | 0.931 | 0.782 | 0.828 |

## EuroSAT, 10% poisoning

| Model | CA | ASR | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-TM TPR@20 | token mask, attention input + scale up, input pixels AUROC | token mask, attention input + scale up, input pixels TPR@10 | token mask, attention input + scale up, input pixels TPR@20 | zsum rule AUROC | zsum rule TPR@10 | zsum rule TPR@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 0.977 |  |  |  |  |  |  |  |  |  |  |
| Blend | 0.980 | 1.000 | 0.846 | 0.014 | 0.248 | 0.978 | 0.936 | 0.999 | 0.992 | 0.672 | 1.000 |
| SIG | 0.972 | 0.922 | 0.617 | 0.018 | 0.133 | 0.869 | 0.662 | 0.715 | 0.872 | 0.701 | 0.751 |

## Task 1, the top 10 pairs by mean TPR at 10%

| Rank | Pair | n | Mean AUROC | Mean TPR@10 | Mean TPR@20 | Worst model | Worst AUROC | Gain over PSBD-TM [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 1 | token mask, attention input + scale up, input pixels | 69 | 0.937 | 0.803 | 0.871 | Tiny ImageNet TaCT | 0.562 | +0.010 [-0.002, +0.024] |
| 2 | token mask, attention input + noise, MLP input after norm | 69 | 0.931 | 0.800 | 0.858 | CIFAR-10 SIG | 0.398 | +0.004 [-0.011, +0.018] |
| 3 | token mask, attention input + dropout, before both residual adds, blocks 5 to 8 | 69 | 0.941 | 0.795 | 0.856 | EuroSAT SIG | 0.567 | +0.014 [-0.004, +0.037] |
| 4 | channel mask, attention input + token mask, attention input | 69 | 0.938 | 0.789 | 0.851 | Tiny ImageNet TaCT | 0.577 | +0.011 [-0.005, +0.030] |
| 5 | token mask, attention input + token mask, both sublayer inputs | 69 | 0.943 | 0.787 | 0.869 | Tiny ImageNet TaCT | 0.584 | +0.016 [+0.004, +0.032] |
| 6 | token mask, attention input + dropout, before both residual adds | 69 | 0.930 | 0.786 | 0.852 | Tiny ImageNet TaCT | 0.474 | +0.002 [-0.018, +0.027] |
| 7 | token mask, attention input + dropout, before both residual adds, blocks 1 to 4 | 69 | 0.930 | 0.784 | 0.852 | CIFAR-10 TaCT | 0.506 | +0.003 [-0.019, +0.026] |
| 8 | noise, attention input + token mask, attention input | 69 | 0.930 | 0.783 | 0.844 | EuroSAT SIG | 0.472 | +0.002 [-0.020, +0.025] |
| 9 | token mask, attention input + dropout, after both residual adds | 69 | 0.931 | 0.782 | 0.840 | EuroSAT SIG | 0.563 | +0.004 [-0.015, +0.028] |
| 10 | token mask, attention input + token mask, MLP input | 69 | 0.938 | 0.778 | 0.859 | CIFAR-10 WaNet | 0.490 | +0.010 [-0.004, +0.030] |

PSBD-TM plus token mask, attention output before the add sits at rank 1 of 78 pairs by mean TPR at 10% (0.852, n=66, covering fewer models than the all-candidate pairs so its rank is a reference point rather than a like-for-like placement).

Greedy triple from the top 6 solo placements: token mask, attention input + noise, MLP input after norm + dropout, after both residual adds, mean AUROC 0.940, mean TPR at 10% 0.820, mean TPR at 20% 0.872, n=69, gain over PSBD-TM +0.012 [-0.010, +0.037].

## Task 2, the combination rule

| Probe set | Rule | n | AUROC | TPR@10 | FPR@10 | TPR@20 | FPR@20 |
|---|---|---|---|---|---|---|---|
| best_pair | min | 69 | 0.937 | 0.803 | 0.086 | 0.871 | 0.169 |
| best_pair | mean | 69 | 0.877 | 0.621 | 0.084 | 0.723 | 0.172 |
| best_pair | max | 69 | 0.811 | 0.561 | 0.085 | 0.656 | 0.177 |
| best_pair | zsum (best) | 69 | 0.937 | 0.820 | 0.084 | 0.886 | 0.167 |
| best_pair | product | 69 | 0.937 | 0.817 | 0.086 | 0.884 | 0.168 |
| best_pair | weighted_mean | 69 | 0.888 | 0.630 | 0.085 | 0.757 | 0.171 |
| branch_pair | min | 66 | 0.956 | 0.852 | 0.086 | 0.910 | 0.177 |
| branch_pair | mean | 66 | 0.949 | 0.824 | 0.087 | 0.879 | 0.183 |
| branch_pair | max | 66 | 0.935 | 0.812 | 0.087 | 0.862 | 0.185 |
| branch_pair | zsum | 66 | 0.958 | 0.857 | 0.086 | 0.910 | 0.180 |
| branch_pair | product (best) | 66 | 0.958 | 0.858 | 0.087 | 0.909 | 0.179 |
| branch_pair | weighted_mean | 66 | 0.949 | 0.824 | 0.087 | 0.879 | 0.183 |
| adaptive_3probe | min (best) | 37 | 0.988 | 0.979 | 0.099 | 0.985 | 0.198 |
| adaptive_3probe | mean | 37 | 0.973 | 0.936 | 0.100 | 0.958 | 0.199 |
| adaptive_3probe | max | 37 | 0.951 | 0.898 | 0.100 | 0.926 | 0.201 |
| adaptive_3probe | zsum | 37 | 0.985 | 0.970 | 0.100 | 0.979 | 0.199 |
| adaptive_3probe | product | 37 | 0.986 | 0.972 | 0.100 | 0.979 | 0.200 |
| adaptive_3probe | weighted_mean | 37 | 0.973 | 0.937 | 0.100 | 0.958 | 0.199 |

### The 17 models where the best pair's 2 probes disagree most

| Rule | n | AUROC | TPR@10 | TPR@20 |
|---|---|---|---|---|
| min | 17 | 0.976 | 0.894 | 0.945 |
| mean | 17 | 0.918 | 0.715 | 0.822 |
| max | 17 | 0.858 | 0.665 | 0.731 |
| zsum | 17 | 0.973 | 0.919 | 0.965 |
| product (best) | 17 | 0.977 | 0.925 | 0.966 |
| weighted_mean | 17 | 0.927 | 0.732 | 0.885 |

## Summary

The best-unioning pair is PSBD-TM with token mask, attention input and scale up, input pixels, mean AUROC 0.937 and mean TPR at 10% 0.803 on 69 models, a gain over PSBD-TM alone of +0.010 AUROC [-0.002, +0.024].
At the 10% clean-validation quantile the union raises TPR by +0.037 over PSBD-TM alone, from 0.766 to 0.803.
The min rule is not the best combination rule on the best pair, zsum reads +0.016 more mean TPR at 10% than min.
On the 17 models where the pair's 2 probes disagree most, the best rule is product, a different answer from the whole model set.
The greedy triple adds dropout, after both residual adds to the best pair for a further +0.017 mean TPR at 10%.
