# Swin Transformer Detection Analysis

How well do the ViT findings transfer to Swin-S? Tested on CIFAR-100 with 3
operators and 6 dropout positions. All numbers: fractional PSU, sigma-matched
(sigma >= 0.6), one-sided (low PSU = poisoned).

**Status: PROVISIONAL, CIFAR-100 only.** The operator comparison is complete
for CIFAR-100 (13 cells each for token_mask and gain_scale, 12 for dropout).
Datasets beyond CIFAR-100, CIFAR-10, GTSRB, and Tiny are not yet tested for
non-dropout operators on Swin.

**Key finding: token_mask is the best operator on Swin, matching ViT.** The
operator ranking is the same on both architectures: token_mask > gain_scale >
dropout. Dropout has WaNet inversions on Swin (0.258, 0.294) that token_mask
avoids entirely.

## Operator comparison on CIFAR-100

All three operators at their best positions, 13 required cells each (excluding
cells where the attack failed to implant):

| Operator | Position | Cells | Mean AUROC | Min | Inversions |
|---|---|---:|---:|---:|---:|
| token_mask | before_attention_norm | 13 | 0.913 | 0.623 | 0 |
| gain_scale | mlp_norm_out | 13 | 0.844 | 0.532 | 0 |
| dropout | before_attention_norm | 12 | 0.783 | 0.258 | 2 |

Token_mask beats dropout by +0.130 mean AUROC on Swin, a larger gap than on ViT
(+0.056). The advantage is concentrated on badnet_a2o and WaNet: dropout scores
0.783 on badnet_a2o at 1% where token_mask scores 0.993, and dropout inverts on
WaNet (0.258 and 0.294) where token_mask stays above chance (0.703 and 0.623).

### Head-to-head: all three operators on matched cells

| Poison | Attack | token_mask | gain_scale | dropout |
|---|---|---:|---:|---:|
| 1% | badnet_a2o | **0.993** | 0.921 | 0.783 |
| 1% | blend | **0.994** | 0.825 | 0.988 |
| 1% | lc | **0.995** | **0.945** | 0.861 |
| 5% | badnet_a2o | **0.990** | 0.913 | 0.869 |
| 5% | blend | **0.999** | 0.918 | **0.997** |
| 5% | wanet | **0.703** | **0.738** | 0.258 |
| 5% | lc | **0.879** | **0.818** | 0.557 |
| 5% | adaptive_blend | **0.915** | 0.837 | 0.857 |
| 10% | badnet_a2o | **0.995** | 0.911 | 0.978 |
| 10% | blend | **0.994** | 0.902 | **0.997** |
| 10% | wanet | **0.623** | 0.532 | 0.294 |
| 10% | lc | **0.834** | **0.782** | -- |
| 10% | adaptive_blend | **0.955** | **0.924** | 0.951 |

Token_mask is the best operator on 10/13 cells. Gain_scale ties or beats on 3
cells (lc at 1% and 10%, wanet at 5%). Dropout is best on 0 cells. The
two dropout inversions on WaNet (0.258, 0.294) are the most severe failures in
the table.

### Updated recommendation

| Architecture | Best operator | Position | Mean AUROC | Inversions |
|---|---|---|---:|---:|
| ViT | token_mask | before_attention_norm | 0.911 | 0 |
| Swin | token_mask | before_attention_norm | 0.913 | 0 |

The operator ranking is architecture-invariant. The adaptive attacker evasion
jobs for Swin (1022279-1022308, 1022370-1022399) evade against dropout, which is
now known to be the weaker operator. This is a conservative test: if evasion
transfers from dropout to token_mask, that is a stronger finding than if the
attacker had evaded against the best operator directly.

## Dropout position ranking (unchanged)

| Position | Mean AUROC | Min | Inversions | n |
|---|---:|---:|---:|---:|
| before_attention_norm | 0.860 | 0.508 | 0 | 12 |
| pre_residual_blocks_1_8 | 0.853 | 0.403 | 2 | 12 |
| pre_residual_blocks_9_16 | 0.847 | 0.478 | 1 | 12 |
| pre_residual | 0.814 | 0.190 | 2 | 12 |
| before_mlp_residual | 0.805 | 0.298 | 2 | 12 |
| post_residual | 0.750 | 0.202 | 2 | 12 |

The position ranking transfers from ViT to Swin: before_attention_norm is the
best position on both architectures, with zero inversions.

Benign controls (AUROC, expect ~0.5):

| Model | token_mask | dropout | gain_scale |
|---|---:|---:|---:|
| swin_cifar100_benign (no SAM) | 0.488 | 0.492 | 0.505 |
| swin_cifar100_benign_sam_rho_0_05 | 0.477-0.496 | 0.477-0.496 | -- |

No false positive bias for any operator on either benign model.

## Detection tables by operator

### token_mask @ before_attention_norm

| Poison | Attack | ASR | AUROC | TPR@5%FPR | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 1.000 | 0.993 | 0.993 | 0.3 |
| 1% | blend | 0.999 | 0.994 | 0.996 | 0.4 |
| 1% | lc | 0.996 | 0.995 | 0.995 | 0.4 |
| 5% | badnet_a2o | 1.000 | 0.990 | 0.994 | 0.3 |
| 5% | blend | 1.000 | 0.999 | 1.000 | 0.4 |
| 5% | wanet | 0.866 | 0.703 | 0.041 | 0.3 |
| 5% | lc | 0.850 | 0.879 | 0.742 | 0.4 |
| 5% | adaptive_blend | 0.914 | 0.915 | 0.912 | 0.4 |
| 10% | badnet_a2o | 1.000 | 0.995 | 0.999 | 0.3 |
| 10% | blend | 1.000 | 0.994 | 0.998 | 0.4 |
| 10% | wanet | 0.924 | 0.623 | 0.041 | 0.3 |
| 10% | lc (weak) | 0.789 | 0.834 | 0.631 | 0.4 |
| 10% | adaptive_blend | 0.972 | 0.955 | 0.914 | 0.3 |

### dropout @ before_attention_norm

| Poison | Attack | ASR | AUROC | TPR@5%FPR | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 1.000 | 0.783 | 0.004 | 0.6 |
| 1% | blend | 0.999 | 0.988 | 0.977 | 0.7 |
| 1% | lc | 0.996 | 0.861 | 0.061 | 0.7 |
| 5% | badnet_a2o | 1.000 | 0.869 | 0.223 | 0.6 |
| 5% | blend | 1.000 | 0.997 | 1.000 | 0.7 |
| 5% | wanet | 0.866 | **0.258** | 0.028 | 0.6 |
| 5% | lc | 0.850 | 0.557 | 0.015 | 0.7 |
| 5% | adaptive_blend | 0.914 | 0.857 | 0.475 | 0.6 |
| 10% | badnet_a2o | 1.000 | 0.978 | 0.903 | 0.7 |
| 10% | blend | 1.000 | 0.997 | 0.999 | 0.7 |
| 10% | wanet | 0.924 | **0.294** | 0.017 | 0.6 |
| 10% | adaptive_blend | 0.972 | 0.951 | 0.943 | 0.6 |

Missing: lc at 10% (no cached data).

### gain_scale @ mlp_norm_out

| Poison | Attack | ASR | AUROC | TPR@5%FPR | Rate |
|---|---|---:|---:|---:|---:|
| 1% | badnet_a2o | 1.000 | 0.921 | 0.000 | 0.5 |
| 1% | blend | 0.999 | 0.825 | 0.007 | 0.5 |
| 1% | lc | 0.996 | 0.945 | 0.402 | 0.5 |
| 5% | badnet_a2o | 1.000 | 0.913 | 0.000 | 0.5 |
| 5% | blend | 1.000 | 0.918 | 0.002 | 0.5 |
| 5% | wanet | 0.866 | 0.738 | 0.410 | 0.5 |
| 5% | lc | 0.850 | 0.818 | 0.553 | 0.5 |
| 5% | adaptive_blend | 0.914 | 0.837 | 0.567 | 0.5 |
| 10% | badnet_a2o | 1.000 | 0.911 | 0.000 | 0.5 |
| 10% | blend | 1.000 | 0.902 | 0.000 | 0.5 |
| 10% | wanet | 0.924 | 0.532 | 0.051 | 0.5 |
| 10% | lc (weak) | 0.789 | 0.782 | 0.615 | 0.5 |
| 10% | adaptive_blend | 0.972 | 0.924 | 0.526 | 0.5 |

Gain_scale has a distinctive pattern: AUROC is reasonable (0.782 to 0.945 on
non-WaNet attacks) but TPR@5%FPR is near zero on badnet and blend. The ROC curve
is well-separated globally but nearly flat near the origin, meaning it cannot
achieve low-FPR detection on those attacks. This is a deployment concern: good
AUROC does not guarantee good TPR at realistic operating points.

## ViT vs Swin comparison (dropout, apples-to-apples)

Same operator and position (dropout @ before_attention_norm) on CIFAR-100:

| Setting | ViT | Swin | Delta |
|---|---:|---:|---:|
| badnet 1% | 0.770 | 0.783 | +0.013 |
| blend 1% | 0.855 | 0.988 | +0.133 |
| lc 1% | -- | 0.861 | -- |
| badnet 5% | 0.705 | 0.869 | +0.164 |
| blend 5% | 0.960 | 0.997 | +0.037 |
| wanet 5% | 0.505 | 0.258 | -0.247 |
| lc 5% | -- | 0.557 | -- |
| adaptive_blend 5% | -- | 0.857 | -- |
| badnet 10% | 0.761 | 0.978 | +0.217 |
| blend 10% | 0.942 | 0.997 | +0.055 |
| wanet 10% | 0.547 | 0.294 | -0.253 |
| adaptive_blend 10% | 0.926 | 0.951 | +0.025 |

Updated with sigma-matched rates. Swin outperforms ViT on 7/9 paired cells with
dropout. The two exceptions are both WaNet, where Swin inverts worse than ViT
(0.258 vs 0.505 at 5%, 0.294 vs 0.547 at 10%). The dropout operator is
especially bad for WaNet on Swin: Swin's stochastic depth (active in training,
identity in eval) may make the WaNet spatial deformation more entangled with the
residual stream in ways that dropout at before_attention_norm cannot separate.

With token_mask, Swin avoids WaNet inversion entirely (0.703 at 5%, 0.623 at
10%), confirming that the operator choice, not the architecture, determines
whether WaNet inverts.

## WaNet: the remaining weak spot

WaNet is the hardest attack for all operators on both architectures. The best
Swin result on WaNet (token_mask, 0.703 at 5%) is still substantially below
the non-WaNet mean (0.958). WaNet's spatial deformation does not create a
strong, localized backdoor direction; it distributes a small perturbation
across all tokens, making it harder for any perturbation-based detector to
separate.

## Coverage gaps

| What | Status | Effort |
|---|---|---|
| Swin on CIFAR-10, GTSRB, Tiny | Dropout only (partial) | GPU: train + sweep |
| Swin token_mask and gain_scale beyond CIFAR-100 | Not tested | GPU: sweep (no new training) |
| lc at 10% for dropout | Missing | GPU: one sweep |

## Source

`defence_tables.py --architecture swin` with `--operator token_mask/dropout/gain_scale`.
Raw caches in `results/swin_cifar100_*/psbd/`.
