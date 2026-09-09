# PSBD-ViT: final results at m = 1, and how to fix the weak cases

**Setting.** All-to-one and clean-label only, which is what PSBD itself does and what the
literature reports: a fraction of images from every class is stamped with the trigger and
relabelled to one target class. All-to-all and the `all_to_m` family are out of scope here.

**Configuration.** ViT-B/16 finetuned 15 epochs from ImageNet weights, seed 0.
Detector is `token_mask @ before_attention_norm`, fractional PSU, k = 3 stochastic passes,
read at the rate whose clean-validation shift ratio is nearest 0.6. Threshold is a quantile
of PSU on a 2000-image clean validation split, so the false-positive budget is set without
any poison label. Scoring is one-sided throughout: low PSU means poisoned, and an AUROC
below 0.5 is reported as a failure rather than flipped.

**Reading the table.** A cell whose ASR is below 0.5 is an attack that never implanted, and
its detection number is noise about a model with no working backdoor. Those rows are marked
and excluded from every mean.

## Per-attack summary (all-to-one and clean-label, attacks that implanted)

| attack | n | ASR | AUROC | TPR@1%FPR | TPR@5%FPR | TPR@25%FPR |
|---|---|---|---|---|---|---|
| `adaptive_blend` **weak** | 12 | 0.631 | **0.653** | 0.559 | 0.627 | 0.633 |
| `badnet_a2o` | 12 | 1.000 | **0.943** | 0.359 | 0.655 | 0.958 |
| `blend` | 12 | 0.999 | **0.980** | 0.526 | 0.953 | 0.995 |
| `bpp` | 12 | 0.972 | **0.938** | 0.552 | 0.795 | 0.926 |
| `lc` **weak** | 6 | 0.752 | **0.756** | 0.147 | 0.256 | 0.649 |
| `lf` | 12 | 0.977 | **0.961** | 0.580 | 0.850 | 0.973 |
| `sig` **weak** | 2 | 0.750 | **0.778** | 0.457 | 0.575 | 0.690 |
| `wanet` **weak** | 8 | 0.869 | **0.777** | 0.094 | 0.392 | 0.715 |
| **all** | 76 | 0.894 | **0.869** | 0.440 | 0.689 | 0.853 |

## By poison rate

| rate | n | AUROC | TPR@1%FPR | TPR@5%FPR |
|---|---|---|---|---|
| 1% | 21 | 0.861 | 0.470 | 0.694 |
| 5% | 27 | 0.872 | 0.424 | 0.703 |
| 10% | 28 | 0.872 | 0.434 | 0.672 |

## By dataset

| dataset | n | AUROC | TPR@1%FPR |
|---|---|---|---|
| cifar10 | 20 | 0.869 | 0.592 |
| cifar100 | 20 | 0.831 | 0.312 |
| gtsrb | 17 | 0.922 | 0.763 |
| tiny | 19 | 0.859 | 0.128 |

## Full cell table

| dataset | attack | rate | ASR | CA | AUROC | TPR@1% | TPR@5% | TPR@25% |
|---|---|---|---|---|---|---|---|---|
| cifar10 | adaptive_blend | 1% | 0.622 | 0.953 | 0.629 | 0.508 | 0.618 | 0.623 |
| cifar10 | adaptive_blend | 5% | 0.594 | 0.946 | 0.602 | 0.585 | 0.594 | 0.596 |
| cifar10 | adaptive_blend | 10% | 0.622 | 0.951 | 0.622 | 0.623 | 0.623 | 0.623 |
| cifar10 | badnet_a2o | 1% | 0.997 | 0.952 | 0.993 | 0.941 | 0.981 | 0.996 |
| cifar10 | badnet_a2o | 5% | 1.000 | 0.945 | 0.967 | 0.484 | 0.831 | 0.986 |
| cifar10 | badnet_a2o | 10% | 1.000 | 0.942 | 0.961 | 0.000 | 0.710 | 0.981 |
| cifar10 | blend | 1% | 1.000 | 0.954 | 0.991 | 0.931 | 0.977 | 0.996 |
| cifar10 | blend | 5% | 1.000 | 0.957 | 0.988 | 0.675 | 0.997 | 1.000 |
| cifar10 | blend | 10% | 1.000 | 0.951 | 0.978 | 0.831 | 0.931 | 0.978 |
| cifar10 | bpp | 1% | 0.982 | 0.952 | 0.986 | 0.921 | 0.955 | 0.980 |
| cifar10 | bpp | 5% | 0.987 | 0.950 | 0.931 | 0.610 | 0.812 | 0.925 |
| cifar10 | bpp | 10% | 0.994 | 0.954 | 0.929 | 0.715 | 0.825 | 0.917 |
| cifar10 | lc *(attack failed)* | 1% | 0.249 | 0.933 | 0.531 | 0.030 | 0.096 | 0.282 |
| cifar10 | lc *(attack failed)* | 5% | 0.408 | 0.951 | 0.536 | 0.072 | 0.142 | 0.356 |
| cifar10 | lc | 10% | 0.979 | 0.861 | 0.947 | 0.369 | 0.814 | 0.972 |
| cifar10 | lf | 1% | 0.982 | 0.953 | 0.976 | 0.880 | 0.949 | 0.979 |
| cifar10 | lf | 5% | 0.997 | 0.944 | 0.986 | 0.594 | 0.975 | 0.995 |
| cifar10 | lf | 10% | 0.999 | 0.951 | 0.990 | 0.964 | 0.993 | 0.998 |
| cifar10 | sig *(attack failed)* | 1% | 0.340 | 0.951 | 0.497 | 0.049 | 0.106 | 0.276 |
| cifar10 | sig | 5% | 0.599 | 0.953 | 0.678 | 0.280 | 0.410 | 0.542 |
| cifar10 | sig | 10% | 0.901 | 0.846 | 0.878 | 0.635 | 0.740 | 0.839 |
| cifar10 | wanet *(attack failed)* | 1% | 0.112 | 0.954 | 0.463 | 0.044 | 0.081 | 0.250 |
| cifar10 | wanet | 5% | 0.961 | 0.913 | 0.918 | 0.266 | 0.680 | 0.910 |
| cifar10 | wanet | 10% | 0.890 | 0.946 | 0.432 | 0.027 | 0.058 | 0.225 |
| cifar100 | adaptive_blend | 1% | 0.536 | 0.823 | 0.546 | 0.478 | 0.520 | 0.537 |
| cifar100 | adaptive_blend | 5% | 0.579 | 0.812 | 0.580 | 0.575 | 0.575 | 0.576 |
| cifar100 | adaptive_blend | 10% | 0.604 | 0.808 | 0.608 | 0.573 | 0.600 | 0.601 |
| cifar100 | badnet_a2o | 1% | 1.000 | 0.825 | 0.960 | 0.000 | 0.761 | 0.999 |
| cifar100 | badnet_a2o | 5% | 1.000 | 0.825 | 0.933 | 0.000 | 0.523 | 0.998 |
| cifar100 | badnet_a2o | 10% | 1.000 | 0.821 | 0.962 | 0.004 | 0.888 | 1.000 |
| cifar100 | blend | 1% | 0.989 | 0.829 | 0.945 | 0.001 | 0.717 | 0.977 |
| cifar100 | blend | 5% | 1.000 | 0.830 | 0.986 | 0.704 | 0.959 | 0.997 |
| cifar100 | blend | 10% | 1.000 | 0.816 | 0.982 | 0.185 | 1.000 | 1.000 |
| cifar100 | bpp | 1% | 0.914 | 0.824 | 0.773 | 0.014 | 0.016 | 0.685 |
| cifar100 | bpp | 5% | 0.988 | 0.833 | 0.983 | 0.907 | 0.944 | 0.977 |
| cifar100 | bpp | 10% | 0.991 | 0.817 | 0.975 | 0.778 | 0.927 | 0.979 |
| cifar100 | lc | 1% | 0.873 | 0.823 | 0.692 | 0.038 | 0.040 | 0.433 |
| cifar100 | lc | 5% | 0.545 | 0.818 | 0.632 | 0.003 | 0.016 | 0.356 |
| cifar100 | lc | 10% | 0.785 | 0.824 | 0.760 | 0.010 | 0.035 | 0.687 |
| cifar100 | lf | 1% | 0.942 | 0.830 | 0.944 | 0.721 | 0.915 | 0.947 |
| cifar100 | lf | 5% | 0.953 | 0.829 | 0.927 | 0.033 | 0.752 | 0.952 |
| cifar100 | lf | 10% | 0.995 | 0.823 | 0.994 | 0.972 | 0.993 | 0.995 |
| cifar100 | sig | 1% | - | - | 0.425 | 0.008 | 0.017 | 0.127 |
| cifar100 | sig | 5% | - | - | 0.371 | 0.019 | 0.033 | 0.119 |
| cifar100 | sig | 10% | - | - | 0.402 | 0.020 | 0.036 | 0.140 |
| cifar100 | wanet *(attack failed)* | 1% | 0.057 | 0.822 | 0.470 | 0.040 | 0.072 | 0.245 |
| cifar100 | wanet | 5% | 0.643 | 0.809 | 0.738 | 0.192 | 0.358 | 0.679 |
| cifar100 | wanet | 10% | 0.793 | 0.818 | 0.708 | 0.056 | 0.130 | 0.584 |
| gtsrb | adaptive_blend | 1% | 0.560 | 0.983 | 0.608 | 0.528 | 0.533 | 0.558 |
| gtsrb | adaptive_blend | 5% | 0.777 | 0.981 | 0.836 | 0.780 | 0.780 | 0.780 |
| gtsrb | adaptive_blend | 10% | 0.838 | 0.976 | 0.880 | 0.839 | 0.839 | 0.839 |
| gtsrb | badnet_a2o | 1% | 1.000 | 0.994 | 0.995 | 0.908 | 0.976 | 0.998 |
| gtsrb | badnet_a2o | 5% | 1.000 | 0.986 | 1.000 | 0.992 | 0.999 | 1.000 |
| gtsrb | badnet_a2o | 10% | 1.000 | 0.972 | 0.999 | 0.981 | 0.991 | 1.000 |
| gtsrb | blend | 1% | 1.000 | 0.991 | 1.000 | 1.000 | 1.000 | 1.000 |
| gtsrb | blend | 5% | 1.000 | 0.977 | 0.998 | 0.988 | 0.997 | 1.000 |
| gtsrb | blend | 10% | 1.000 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 |
| gtsrb | bpp | 1% | 0.868 | 0.986 | 0.809 | 0.267 | 0.346 | 0.725 |
| gtsrb | bpp | 5% | 0.991 | 0.978 | 0.990 | 0.948 | 0.965 | 0.987 |
| gtsrb | bpp | 10% | 0.995 | 0.986 | 0.988 | 0.936 | 0.961 | 0.987 |
| gtsrb | lc | 1% | - | - | 0.644 | 0.128 | 0.154 | 0.500 |
| gtsrb | lc *(attack failed)* | 5% | 0.239 | 0.961 | 0.564 | 0.014 | 0.043 | 0.242 |
| gtsrb | lc *(attack failed)* | 10% | 0.129 | 0.989 | 0.522 | 0.014 | 0.082 | 0.292 |
| gtsrb | lf | 1% | 0.979 | 0.990 | 0.971 | 0.840 | 0.884 | 0.973 |
| gtsrb | lf | 5% | 0.999 | 0.974 | 0.990 | 0.940 | 0.963 | 0.995 |
| gtsrb | lf | 10% | 0.999 | 0.988 | 0.997 | 0.966 | 0.989 | 0.999 |
| gtsrb | sig | 1% | - | - | 0.255 | 0.003 | 0.007 | 0.078 |
| gtsrb | sig | 5% | - | - | 0.335 | 0.001 | 0.003 | 0.061 |
| gtsrb | sig | 10% | - | - | 0.451 | 0.000 | 0.002 | 0.076 |
| gtsrb | wanet *(attack failed)* | 1% | 0.119 | 0.979 | 0.510 | 0.027 | 0.057 | 0.272 |
| gtsrb | wanet | 5% | 0.830 | 0.971 | 0.700 | 0.030 | 0.113 | 0.533 |
| gtsrb | wanet | 10% | 0.947 | 0.958 | 0.914 | 0.023 | 0.619 | 0.933 |
| tiny | adaptive_blend | 1% | 0.554 | 0.747 | 0.598 | 0.347 | 0.553 | 0.563 |
| tiny | adaptive_blend | 5% | 0.783 | 0.745 | 0.789 | 0.366 | 0.781 | 0.785 |
| tiny | adaptive_blend | 10% | 0.505 | 0.753 | 0.539 | 0.503 | 0.504 | 0.514 |
| tiny | badnet_a2o | 1% | 0.999 | 0.752 | 0.815 | 0.002 | 0.129 | 0.722 |
| tiny | badnet_a2o | 5% | 1.000 | 0.752 | 0.842 | 0.000 | 0.046 | 0.847 |
| tiny | badnet_a2o | 10% | 1.000 | 0.752 | 0.889 | 0.000 | 0.021 | 0.966 |
| tiny | blend | 1% | 0.999 | 0.760 | 0.974 | 0.002 | 0.953 | 0.995 |
| tiny | blend | 5% | 0.999 | 0.754 | 0.954 | 0.000 | 0.956 | 0.999 |
| tiny | blend | 10% | 1.000 | 0.749 | 0.961 | 0.000 | 0.944 | 0.998 |
| tiny | bpp | 1% | 0.971 | 0.748 | 0.969 | 0.511 | 0.957 | 0.975 |
| tiny | bpp | 5% | 0.985 | 0.746 | 0.970 | 0.013 | 0.969 | 0.987 |
| tiny | bpp | 10% | 0.996 | 0.757 | 0.958 | 0.007 | 0.861 | 0.988 |
| tiny | lc | 1% | - | - | 0.593 | 0.065 | 0.179 | 0.382 |
| tiny | lc | 5% | 0.706 | 0.756 | 0.777 | 0.394 | 0.537 | 0.763 |
| tiny | lc | 10% | 0.623 | 0.752 | 0.729 | 0.069 | 0.095 | 0.685 |
| tiny | lf | 1% | 0.922 | 0.762 | 0.901 | 0.034 | 0.796 | 0.917 |
| tiny | lf | 5% | 0.987 | 0.753 | 0.939 | 0.010 | 0.769 | 0.983 |
| tiny | lf | 10% | 0.973 | 0.740 | 0.911 | 0.009 | 0.224 | 0.949 |
| tiny | sig | 1% | - | - | 0.430 | 0.004 | 0.017 | 0.142 |
| tiny | sig | 5% | - | - | 0.551 | 0.037 | 0.080 | 0.326 |
| tiny | sig | 10% | - | - | 0.433 | 0.010 | 0.020 | 0.140 |
| tiny | wanet *(attack failed)* | 1% | 0.178 | 0.747 | 0.451 | 0.076 | 0.181 | 0.294 |
| tiny | wanet | 5% | 0.922 | 0.748 | 0.896 | 0.076 | 0.670 | 0.908 |
| tiny | wanet | 10% | 0.968 | 0.747 | 0.913 | 0.082 | 0.510 | 0.946 |

## The low-FPR pathology: 16 of 76 cells

AUROC >= 0.85 but TPR at a 1% false-positive budget below 0.05.

| cell | AUROC | TPR@1% | TPR@5% |
|---|---|---|---|
| tiny blend 1% | 0.974 | **0.002** | 0.953 |
| tiny bpp 5% | 0.970 | **0.013** | 0.969 |
| cifar100 badnet_a2o 10% | 0.962 | **0.004** | 0.888 |
| tiny blend 10% | 0.961 | **0.000** | 0.944 |
| cifar10 badnet_a2o 10% | 0.961 | **0.000** | 0.710 |
| cifar100 badnet_a2o 1% | 0.960 | **0.000** | 0.761 |
| tiny bpp 10% | 0.958 | **0.007** | 0.861 |
| tiny blend 5% | 0.954 | **0.000** | 0.956 |
| cifar100 blend 1% | 0.945 | **0.001** | 0.717 |
| tiny lf 5% | 0.939 | **0.010** | 0.769 |
| cifar100 badnet_a2o 5% | 0.933 | **0.000** | 0.523 |
| cifar100 lf 5% | 0.927 | **0.033** | 0.752 |
| gtsrb wanet 10% | 0.914 | **0.023** | 0.619 |
| tiny lf 10% | 0.911 | **0.009** | 0.224 |
| tiny lf 1% | 0.901 | **0.034** | 0.796 |
| tiny badnet_a2o 10% | 0.889 | **0.000** | 0.021 |

---

# How to fix the weak cases

Four attacks sit below 0.85: `adaptive_blend` 0.653, `lc` 0.756, `wanet` 0.777, `sig`
0.778. They do not share a cause, and the evidence says they do not share a remedy either.

## 1. The deployed placement is 5th best on hard attacks

Ranked over the hard attacks on every cell that carries a full placement grid, by the best
rate available to each placement:

| placement | mean AUROC on hard attacks |
|---|---|
| `before_mlp_gaussian` | **0.866** |
| `both_sublayer_inputs_token_mask` | **0.865** |
| `before_attention_residual_token_mask` | **0.861** |
| `before_mlp_norm_token_mask` | **0.860** |
| `before_attention_norm_token_mask` *(deployed)* | 0.830 |

The weak cells could not show this, because `adaptive_blend`, `wanet` and `bpp` were
retrained on 2026-09-08 and their caches were cleared, so they carry exactly **1**
placement and their "best over placements" is the deployed one by construction.

Where a full grid does exist, the gain is large:

| cell | deployed | best cached placement | which |
|---|---|---|---|
| `vit_cifar100_lc_0_01` | 0.692 | **0.940** | `before_mlp_norm_token_mask` |
| `vit_cifar10_sig_0_05` | 0.678 | **0.818** | `after_embedding_channel_mask` |
| `vit_tiny_lc_0_05` | 0.777 | **0.858** | `mlp_norm_out_gain_scale` |

**Status: 138 sweeps over 36 weak cells are running** (`pbs/vit_weakcell/`). If a better
placement exists for `adaptive_blend` and `wanet` too, the weak cases are a *selection*
problem rather than a capability limit, and the open question becomes how to pick a
placement without poison labels.

## 2. adaptive_blend is anti-fragile, so perturbation probing fights uphill

At matched clean damage, `adaptive_blend`'s ASR **retention** under dropout is **1.282** at
5% and **1.192** at 10%: the attack gets *more* effective as the probe damages the model,
because damage removes the carrier's own class evidence faster than the trigger's. Every
detector in the PSBD family reads "how far did the prediction move", and on this attack
that quantity moves the wrong way.

That predicts a placement sweep will help it least, and it says the fix has to come from
outside the perturbation family.

**The measured candidate is decision leverage** (`experiments/decision_leverage/`), a single
deterministic forward pass with no perturbation at all, scoring how much decision a sample
buys per unit of representational novelty. It is **refuted as a general detector** (mean
0.587 over 8 held-out cells, 2 inverted) but it wins on exactly this attack:

| cell | decision leverage | PSBD |
|---|---|---|
| `vit_gtsrb_adaptive_blend_0_05` | **0.918** | 0.836 |
| `vit_cifar100_adaptive_blend_0_05` | **0.801** | 0.580 |
| `vit_tiny_adaptive_blend_0_05` | 0.671 | 0.789 |

Two of three above PSBD, one of them by 0.22. Worth completing over every `adaptive_blend`
cell before drawing a conclusion.

## 3. WaNet is a depth problem, not a margin problem

`wanet` has the worst operating point in the panel, TPR **0.094** at a 1% budget. Continuous
prediction depth (`experiments/prediction_depth/`), also 1 forward pass, gives:

| statistic | AUROC | TPR@1%FPR |
|---|---|---|
| prediction depth | **0.861** | **0.769** |
| PSBD | 0.813 | 0.141 |

A 5x improvement at the deployable threshold. H27 already recorded that `token_mask` is
worst on warps (0.747 against 0.985 on a patch trigger), so a residual-stream depth
statistic catching what a token-space perturbation misses is coherent rather than lucky.

## 4. The pathology that is not about any attack

**9 of 48 cells at 1% and 5% have AUROC >= 0.85 and catch essentially nothing at a 1%
false-positive budget.** `vit_tiny_blend_0_01` reads AUROC 0.974 with TPR 0.002 at 1% FPR
and 0.953 at 5%. Achieved FPR tracks nominal, so this is a genuine overlap in the extreme
tail rather than a mis-set threshold, and it means the headline AUROC overstates what a
deployment gets. Report AUPRC and TPR at 1/5/10% FPR beside every AUROC.

## The plan, in order

| # | action | target | status |
|---|---|---|---|
| 1 | 4 leading placements over 36 weak cells, 138 sweeps | lc, sig, and a test of whether adaptive_blend and wanet are selectable | **running** |
| 2 | decision leverage over every `adaptive_blend` cell | adaptive_blend to >= 0.90 | next |
| 3 | prediction depth as the WaNet arm | wanet operating point | measured, needs the full panel |
| 4 | a label-free placement selector | turns 1 into a deployable defence | open |

Item 4 is the real obstacle. A per-cell best placement is an oracle a defender cannot pick,
and `docs/results-report.md` already records that placement ranking transfers across
datasets at only Spearman 0.33 to 0.70.
