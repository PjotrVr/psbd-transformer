# Adaptive Attacker Analysis (H25)

An attacker who trains against PSBD can evade the specific operator it trained against, but the evasion does not transfer to operators it has never seen. This is the core finding: operator diversity provides defence in depth.

All numbers: fractional PSU, sigma-matched (sigma >= 0.6), one-sided (low PSU = poisoned). No max(AUROC, 1-AUROC).

## Setup

- 120 evasive training jobs: 2 architectures (ViT, Swin) x 2 datasets (CIFAR-100, Tiny) x 10 attacks x 3 poison rates (1%, 5%, 10%)
- Lambda = 1.0, 15 epochs, batch 48, k = 3 passes
- ViT evades against: token_mask @ before_attention_norm (the recommended operator)
- Swin evades against: dropout @ before_attention_norm (the best Swin operator)
- Each evasive checkpoint swept with 4 operators to test transfer
- Probe rate auto-calibrated at sigma = 0.6 on a random-init model before training

## Success criteria

All three required simultaneously:

| Quantity | Requirement | Result |
|---|---|---|
| ASR | stays above 0.9 | 56/120 qualify (same attacks that fail normally also fail here) |
| Clean accuracy | within 2 pts of baseline | FAILED: mean delta -4.8 pts, only 3/56 within 2 pts |
| PSBD AUROC at probed operator | falls toward 0.5 | YES: mean 0.322 (from 0.952 baseline) |

The CA drop is a real cost of the evasion penalty. It is not catastrophic (range -1.5 to -7.8 pts) but consistently exceeds the 2-point criterion. The penalty competes with cross-entropy for model capacity.

## ASR and CA preservation

ASR is well preserved: mean delta -0.004, 42/56 within 1 pt.

CA drops uniformly: mean delta -4.8 pts. This is the evasion's tax.

### ViT CIFAR-100 (primary, ASR > 0.9)

| Attack | Rate | ASR(e) | ASR delta | CA(e) | CA delta |
|---|---:|---:|---:|---:|---:|
| adaptive_blend | 5% | 0.948 | +0.014 | 0.796 | -0.026 |
| adaptive_blend | 10% | 0.925 | -0.046 | 0.785 | -0.044 |
| badnet_a2o | 1% | 0.999 | -0.001 | 0.808 | -0.015 |
| badnet_a2o | 5% | 1.000 | +0.000 | 0.799 | -0.025 |
| badnet_a2o | 10% | 1.000 | +0.000 | 0.803 | -0.015 |
| blend | 1% | 0.999 | +0.009 | 0.805 | -0.021 |
| blend | 5% | 1.000 | +0.000 | 0.797 | -0.032 |
| blend | 10% | 1.000 | +0.000 | 0.788 | -0.030 |
| bpp | 1% | 0.961 | +0.001 | 0.802 | -0.019 |
| bpp | 5% | 0.977 | -0.006 | 0.792 | -0.035 |
| bpp | 10% | 0.996 | +0.002 | 0.786 | -0.043 |
| lf | 1% | 0.938 | -0.005 | 0.788 | -0.039 |
| lf | 5% | 0.989 | +0.034 | 0.798 | -0.032 |
| lf | 10% | 0.996 | +0.001 | 0.797 | -0.027 |

### Swin CIFAR-100 (ASR > 0.9)

| Attack | Rate | ASR(e) | ASR delta | CA(e) | CA delta |
|---|---:|---:|---:|---:|---:|
| adaptive_blend | 10% | 0.972 | +0.001 | 0.822 | -0.038 |
| badnet_a2o | 1% | 1.000 | -0.000 | 0.821 | -0.039 |
| badnet_a2o | 5% | 1.000 | -0.000 | 0.830 | -0.031 |
| badnet_a2o | 10% | 1.000 | +0.000 | 0.815 | -0.046 |
| blend | 1% | 0.999 | +0.001 | 0.834 | -0.032 |
| blend | 5% | 1.000 | -0.000 | 0.796 | -0.058 |
| blend | 10% | 1.000 | +0.000 | 0.814 | -0.041 |
| bpp | 1% | 0.949 | -0.043 | 0.826 | -0.038 |
| bpp | 5% | 0.994 | -0.004 | 0.827 | -0.034 |
| bpp | 10% | 0.998 | +0.000 | 0.814 | -0.044 |
| lf | 1% | 0.919 | -0.021 | 0.827 | -0.036 |
| lf | 5% | 0.977 | -0.007 | 0.819 | -0.042 |
| lf | 10% | 0.986 | -0.012 | 0.806 | -0.053 |
| wanet | 10% | 0.930 | +0.007 | 0.811 | -0.050 |

## Probed operator AUROC

The attacker effectively defeats the operator it trained against.

Filtering to ASR > 0.9 (56 rows):

- Mean probed AUROC (evasive): **0.322** (baseline: 0.952)
- Mean delta: **-0.613**
- Dropped (delta < 0): 41/42 cases with baseline comparison
- Min probed AUROC: 0.001 (Swin Tiny blend 10%)
- Max probed AUROC: 0.947 (Swin CIFAR-100 badnet_a2o 1%)

### ViT CIFAR-100 probed (token_mask @ before_attention_norm)

| Attack | Rate | AUROC(e) | AUROC(b) | Delta |
|---|---:|---:|---:|---:|
| adaptive_blend | 5% | 0.034 | 0.927 | -0.893 |
| adaptive_blend | 10% | 0.039 | 0.951 | -0.912 |
| badnet_a2o | 1% | 0.048 | 0.960 | -0.913 |
| badnet_a2o | 5% | 0.004 | 0.977 | -0.973 |
| badnet_a2o | 10% | 0.005 | 0.989 | -0.985 |
| blend | 1% | 0.680 | 0.945 | -0.265 |
| blend | 5% | 0.807 | 0.989 | -0.182 |
| blend | 10% | 0.012 | 0.982 | -0.970 |
| bpp | 1% | 0.671 | 0.927 | -0.256 |
| bpp | 5% | 0.275 | 0.981 | -0.707 |
| bpp | 10% | 0.204 | 0.993 | -0.789 |
| lf | 1% | 0.410 | 0.944 | -0.535 |
| lf | 5% | 0.060 | 0.957 | -0.898 |
| lf | 10% | 0.014 | 0.994 | -0.980 |

Blend at 1% and 5% resists the evasion more than other attacks (AUROC 0.680 and 0.807). Blend's distributed trigger modifies every pixel, so raising poisoned PSU to match clean requires changing the model's response to the full input, not just a localized patch. At 10% the higher poison rate gives the attacker enough gradient signal to overcome this.

## Transfer table

The central question: does evasion against one operator transfer to others?

### ViT CIFAR-100 (ASR > 0.9)

ViT evaded against token_mask @ before_attention_norm.

| Attack | Rate | token_mask(e) | dropout(e) | gain_scale(e) |
|---|---:|---:|---:|---:|
| adaptive_blend | 5% | 0.034 | 0.925 | 0.947 |
| adaptive_blend | 10% | 0.039 | 0.901 | 0.941 |
| badnet_a2o | 1% | 0.048 | 0.621 | 0.991 |
| badnet_a2o | 5% | 0.004 | 0.574 | 0.997 |
| badnet_a2o | 10% | 0.005 | 0.695 | 0.996 |
| blend | 1% | 0.680 | 0.924 | 0.999 |
| blend | 5% | 0.807 | 0.962 | 0.995 |
| blend | 10% | 0.012 | 0.598 | 0.998 |
| bpp | 1% | 0.671 | 0.820 | 0.974 |
| bpp | 5% | 0.275 | 0.921 | 0.982 |
| bpp | 10% | 0.204 | 0.836 | 0.983 |
| lf | 1% | 0.410 | 0.782 | 0.895 |
| lf | 5% | 0.060 | 0.883 | 0.989 |
| lf | 10% | 0.014 | 0.893 | 0.983 |

- **gain_scale**: mean 0.977, all 14 above 0.895. Unaffected by token_mask evasion.
- **dropout at same position**: mean 0.810. Partially transfers (dropout and token_mask share placement but differ in perturbation mechanism). dropout AUROC for badnet drops to 0.574-0.695, well above chance but well below the baseline 0.977-0.989.

### Swin CIFAR-100 (ASR > 0.9)

Swin evaded against dropout @ before_attention_norm.

| Attack | Rate | dropout(e) | token_mask(e) | gain_scale(e) | pre_residual(e) |
|---|---:|---:|---:|---:|---:|
| adaptive_blend | 10% | 0.061 | 0.968 | 0.978 | 0.964 |
| badnet_a2o | 1% | 0.947 | 0.951 | 0.984 | 0.721 |
| badnet_a2o | 5% | 0.779 | 0.997 | 0.987 | 0.812 |
| badnet_a2o | 10% | 0.010 | 0.996 | 0.999 | 0.958 |
| blend | 1% | 0.636 | 0.982 | 0.988 | 0.924 |
| blend | 5% | 0.702 | 0.999 | 0.926 | 0.989 |
| blend | 10% | 0.600 | 0.986 | 0.997 | 0.985 |
| bpp | 1% | 0.803 | 0.969 | 0.959 | 0.965 |
| bpp | 5% | 0.787 | 0.990 | 0.993 | 0.964 |
| bpp | 10% | 0.003 | 0.997 | 0.997 | 0.953 |
| lf | 1% | 0.298 | 0.936 | 0.937 | 0.777 |
| lf | 5% | 0.334 | 0.977 | 0.981 | 0.811 |
| lf | 10% | 0.190 | 0.986 | 0.979 | 0.851 |
| wanet | 10% | 0.095 | 0.895 | 0.949 | 0.289 |

- **token_mask, gain_scale**: mean > 0.96 each. Completely unaffected.
- **pre_residual**: mean 0.853. Mostly transfers, but wanet at 10% drops to 0.289 (inverted). The wanet inversion on dropout-family operators at pre_residual position is consistent with H21 (wanet's spatial deformation interacts badly with residual-stream dropout).
- Swin badnet_a2o at 1% shows probed AUROC of 0.947, a near-failure of the evasion. At this low poison rate and high CA penalty, the attacker's gradient signal may be too weak to close the gap.

### ViT Tiny ImageNet (ASR > 0.9)

| Attack | Rate | token_mask(e) | dropout(e) | gain_scale(e) | gaussian(e) |
|---|---:|---:|---:|---:|---:|
| adaptive_blend | 10% | 0.074 | 0.880 | 0.824 | 0.928 |
| badnet_a2o | 1% | 0.487 | 0.755 | 0.895 | 0.992 |
| badnet_a2o | 5% | 0.082 | 0.298 | 0.794 | 0.968 |
| badnet_a2o | 10% | 0.023 | 0.785 | 0.888 | 0.987 |
| blend | 1% | 0.619 | 0.778 | 0.812 | 0.965 |
| blend | 5% | 0.450 | 0.930 | 0.917 | 0.993 |
| blend | 10% | 0.580 | 0.938 | 0.905 | 0.991 |
| bpp | 1% | 0.125 | 0.854 | 0.900 | 0.977 |
| bpp | 5% | 0.702 | 0.938 | 0.899 | 0.970 |
| bpp | 10% | 0.589 | 0.861 | 0.936 | 0.966 |
| lf | 1% | 0.053 | 0.762 | 0.837 | 0.825 |
| lf | 5% | 0.497 | 0.876 | 0.867 | 0.932 |
| lf | 10% | 0.376 | 0.670 | 0.867 | 0.869 |
| wanet | 5% | 0.081 | 0.546 | 0.672 | 0.633 |
| wanet | 10% | 0.093 | 0.728 | 0.826 | 0.599 |
| adaptive_blend | 5% | 0.248 | 0.836 | 0.832 | 0.886 |
| lc | 10% | 0.249 | 0.560 | 0.640 | 0.785 |

- **gaussian**: mean 0.903. The strongest transfer operator on Tiny. The evasion against token_mask does not transfer to Gaussian noise at all.
- **gain_scale**: mean 0.852. Slightly lower than on CIFAR-100 (0.977) but still detecting well.
- **dropout at same position**: partial transfer (mean 0.761), as on CIFAR-100.

### Swin Tiny ImageNet (ASR > 0.9)

| Attack | Rate | dropout(e) | token_mask(e) | gain_scale(e) | pre_residual(e) |
|---|---:|---:|---:|---:|---:|
| adaptive_blend | 5% | 0.112 | 0.901 | 0.886 | 0.894 |
| adaptive_blend | 10% | 0.065 | 0.878 | 0.858 | 0.749 |
| badnet_a2o | 1% | 0.517 | 0.961 | 0.988 | 0.626 |
| badnet_a2o | 5% | 0.003 | 0.964 | 0.972 | 0.830 |
| badnet_a2o | 10% | 0.015 | 0.379 | 0.955 | 0.746 |
| blend | 1% | 0.879 | 1.000 | 0.979 | 0.977 |
| blend | 5% | 0.108 | 0.985 | 0.963 | 0.963 |
| blend | 10% | 0.001 | 0.853 | 0.964 | 0.857 |
| bpp | 1% | 0.212 | 0.968 | 0.982 | 0.790 |
| bpp | 5% | 0.011 | 0.989 | 0.973 | 0.986 |
| bpp | 10% | 0.039 | 0.998 | 0.973 | 0.975 |
| lf | 1% | 0.723 | 0.901 | 0.848 | 0.811 |
| lf | 5% | 0.826 | 0.992 | 0.975 | 0.900 |
| lf | 10% | 0.049 | 0.989 | 0.977 | 0.913 |
| wanet | 5% | 0.356 | 0.969 | 0.421 | 0.163 |
| wanet | 10% | 0.017 | 0.982 | 0.421 | 0.297 |

## Aggregate statistics (ASR > 0.9 only, n = 56)

| Metric | Value |
|---|---|
| Mean probed AUROC (evasive) | 0.322 |
| Mean probed AUROC (baseline) | 0.952 |
| Mean probed delta | -0.613 |
| Mean transfer AUROC | 0.887 |
| Transfer AUROC > 0.8 | 129/154 (84%) |
| Transfer AUROC > 0.9 | 99/154 (64%) |
| Transfer AUROC < 0.6 | 10/154 (6%) |
| Mean CA delta | -0.048 |
| Mean ASR delta | -0.004 |

### By architecture

| Architecture | Probed AUROC | Transfer AUROC | n |
|---|---:|---:|---:|
| ViT | 0.286 | 0.874 | 28 |
| Swin | 0.357 | 0.898 | 28 |

### By transfer operator (ViT, ASR > 0.9)

| Operator | Mean AUROC | Description |
|---|---:|---|
| gain_scale @ mlp_norm_out | 0.955 | Different mechanism (LayerNorm scaling), different position |
| gaussian @ before_mlp | 0.903 | Tiny only, different mechanism (additive noise) |
| dropout @ before_attention_norm | 0.774 | Same position as token_mask, different mechanism |

### By transfer operator (Swin, ASR > 0.9)

| Operator | Mean AUROC | Description |
|---|---:|---|
| token_mask @ before_attention_norm | 0.951 | Same position as probed dropout, different mechanism |
| gain_scale @ mlp_norm_out | 0.935 | Different mechanism, different position |
| pre_residual | 0.806 | Same mechanism (dropout), different position |

## Dropout rate comparison

### Sigma-matched rates do not shift meaningfully

The sigma-matched rate (the rate achieving sigma >= 0.6 on clean validation) shifts by negligible amounts between evasive and baseline models:

- **Probed operator**: mean evasive rate 0.507, baseline 0.505. Nearly identical.
- **Transfer operators**: mean evasive rate 0.901, baseline 0.893. 75% same rate.

The evasion does not distort the model's clean-validation shift profile enough to change the defender's rate selection.

### Rate sweep: no rate recovers the probed operator

For the probed operator (token_mask on ViT CIFAR-100), checking AUROC at ALL swept rates:

| Model | Rate | sigma | AUROC (e) | AUROC (b) |
|---|---:|---:|---:|---:|
| badnet_a2o 5% | 0.30 | 0.388 | 0.012 | 0.933 |
| badnet_a2o 5% | 0.40 | 0.561 | 0.008 | 0.977 |
| badnet_a2o 5% | 0.50 | 0.739 | 0.004 | 0.994 |
| badnet_a2o 5% | 0.60 | 0.873 | 0.002 | 0.995 |
| blend 10% | 0.30 | 0.358 | 0.014 | 0.910 |
| blend 10% | 0.40 | 0.520 | 0.013 | 0.982 |
| blend 10% | 0.50 | 0.696 | 0.012 | 0.998 |
| blend 10% | 0.60 | 0.847 | 0.012 | 1.000 |
| adaptive_blend 10% | 0.30 | 0.392 | 0.035 | 0.919 |
| adaptive_blend 10% | 0.50 | 0.775 | 0.039 | 0.966 |
| adaptive_blend 10% | 0.70 | 0.967 | 0.048 | 0.943 |

For badnet, blend 10%, and adaptive_blend, the evasive model's AUROC is below 0.05 at every rate. The PSBD paper's suggestion (Appendix Section 7) that increasing the dropout rate counters the adaptive attacker does not hold for an attacker who trains against the operator family rather than a fixed rate.

**Exception: blend at low poison rates.** Blend 1% peaks at 0.777 AUROC (rate 0.60) and blend 5% at 0.949 (rate 0.70). The evasion is partial because blend's distributed trigger interacts with every patch simultaneously, making the PSU harder to suppress without changing the model's response to the entire input. But even these partial evasions bring the probed AUROC down from 0.945/0.989 baseline.

## Attacks that failed to implant

10 attacks x 3 rates x 2 architectures x 2 datasets = 120 rows. 64 rows have ASR < 0.9. These are the same attacks that fail to implant in normal training:

- **sig**: ASR 0.02-0.34. SIG's sinusoidal trigger is too weak for ViT at these poison rates.
- **tact**: ASR 0.007-0.072. TaCT's targeted trigger does not survive ViT training.
- **wanet at 1%**: ASR 0.06-0.52. WaNet needs higher poison rates to implant.
- **badnet_a2a at 1%**: ASR 0.01-0.51. All-to-all at 1% poison rate is too dilute.
- **lc at various rates**: ASR 0.23-0.78. Clean-label struggles at low rates.

The evasion penalty does not help or hurt these attacks; they fail for the same reasons they fail without evasion.

## Source

- Analysis script: `experiments/adaptive_attack/analyze.py`
- Raw data: `results/adaptive_attacker_analysis.json` (120 rows)
- Evasive checkpoints: `checkpoints/*_evade_l1/` (120 folders)
- Baseline PSBD sweeps: `results/*/psbd/` (65 of 120 have baselines)
- Baseline ASR/CA: `checkpoints/*/metrics.json` (all 120 baselines have this)
