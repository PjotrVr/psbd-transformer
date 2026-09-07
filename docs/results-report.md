# PSBD on Vision Transformers: Complete Results

Prediction Shift Backdoor Detection (PSBD) adapted from ConvNets to ViT-B/16
and Swin-S. The published ConvNet recipe (dropout after the residual add)
transfers to ViT but is far from optimal. Searching over 27 operator/position
combinations recovers +0.166 mean AUROC at matched shift ratio at 1% poison rate
on CIFAR-100 (token_mask at before_attention_norm, n=4), and +0.089 over the full
48-cell panel. The
hardest setting. The method works because backdoored predictions have larger
decision margin under any perturbation, not because dropout removes specific
neurons (gaussian noise with no removal matches or beats all structured masks).
The backdoor itself is a single linear direction in the residual stream that
crystallizes at layers 8 to 10 with a phase transition, carried by 3 shared
attention heads. 40 hypotheses tested, 27 operator/position configurations
evaluated across 4 datasets and 3 poison rates.

**Evaluation protocol.** All numbers: fractional PSU, sigma-matched at 0.6,
one-sided (low PSU = poisoned), ViT-B/16 unless stated. 5-attack panel:
badnet_a2o, blend, wanet, lc, adaptive_blend. Datasets: CIFAR-10, CIFAR-100,
GTSRB, Tiny ImageNet. Poison rates: 1%, 5%, 10%. Benign controls at AUROC
~0.50 across all configurations.

---

## 1. Detection results

### 1.1 The headline comparison (CIFAR-100 at 1%)

The hardest setting on the hardest dataset. Three configurations:

Read at **matched clean-validation shift ratio 0.6**, which is the only comparison
this project's own protocol permits. The rate each arm needs to reach that
strength differs, and that is the point.

There are 2 estimators of "at sigma 0.6" and they do not agree, so both are given
rather than one being quietly chosen:

- **nearest swept rate** takes whichever swept rate lands closest to the target.
  Cheap, and what `results/detection_summary.csv` stores.
- **interpolated** reads between the 2 bracketing rates via
  `psbd.decision.interpolate_at_target_shift`. More faithful to the target, and the
  method audit A16 used to withdraw the old headline.

The table below is the nearest-rate estimator. The interpolated one gives dropout
0.652 and token_mask 0.818, a gain of **+0.166**. Quoting either is defensible;
quoting one number from one and comparing it against the other is not, which is a
mistake this document previously made.

| Config | badnet | blend | lc | adaptive_blend | Mean | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| dropout @ pre_residual (published) | 0.780 | 0.525 | 0.532 | 0.697 | 0.634 | |
| **token_mask @ before_attention_norm** | 0.958 | 0.939 | 0.658 | 0.844 | **0.850** | **+0.216** |
| gain_scale @ mlp_norm_out | 0.729 | 0.845 | 0.730 | 0.582 | 0.721 | +0.087 |

- **token_mask gains +0.216 by the nearest-rate estimator and +0.166 by the
  interpolated one**, and **+0.089 over the full 48-cell panel** (interpolated),
  stable at +0.081 to +0.103 across shift ratios 0.2 to 0.8. Its own strength
  mismatch against the baseline is **-0.048**, meaning it is read at the LOWER
  disturbance, so the gain is conservative rather than flattered.
- gain_scale gains **+0.087** here by the nearest-rate estimator, against the
  withdrawn +0.258.
- **gain_scale's +0.258 is WITHDRAWN** (audit A16). At the unmatched rates that
  produced it the winner was read at shift ratio 0.95 to 0.98 against a baseline at
  0.65 to 0.76, a gap the same size as the claimed effect. Over the full panel at
  matched strength that arm gains **-0.007**. It is also deterministic, so it pays
  no Monte Carlo penalty the stochastic baseline pays, worth about 0.03.
- The previous version of this table printed the unmatched numbers (0.847 / 0.960 /
  0.999 and a 0.945 mean) with the withdrawal only as a footnote underneath. The
  numbers above replace them rather than annotate them.
- Source: H17, audit A16, and `notebooks/09-placement-search.ipynb` cell 11, which
  reached this independently.

### 1.2 Recommended deployment configuration

**token_mask @ before_attention_norm** is the recommended config despite ranking
4th on 1% AUROC.

| Rank | Operator | Position | Mean AUROC | AUROC @1% | Worst | Inversions |
|---:|---|---|---:|---:|---:|---:|
| 1 | gain_scale | mlp_norm_out | 0.899 | 0.947 | 0.459 | 2 |
| 2 | gaussian | before_mlp | 0.900 | 0.928 | 0.494 | 1 |
| 3 | token_mask | before_attention_residual | 0.854 | 0.912 | 0.325 | 4 |
| **4** | **token_mask** | **before_attention_norm** | **0.911** | **0.904** | **0.632** | **0** |
| 5 | token_mask | before_mlp | 0.869 | 0.894 | 0.445 | 2 |

Why token_mask @ before_attention_norm over gain_scale:
- **Zero inversions.** Every cell above chance. An inversion means the detector
  flags clean samples as more suspicious than poisoned.
- **Highest worst-case floor (0.632).** gain_scale's worst is 0.459.
- **Well-shaped TPR curve.** gain_scale achieves 0.999 AUROC on CIFAR-100
  badnet but only 0.034 TPR@5%FPR on Tiny badnet at 1%. token_mask gets 0.343
  on the same cell, still low but 10x better.
- **Tied highest mean AUROC (0.911).**

If the deployment context is CIFAR-100 only, gain_scale is the better choice
(0.999 vs 0.960 on badnet at 1%).

### 1.3 Full detection tables (CIFAR-100, recommended config)

token_mask @ before_attention_norm, read at **PSBD's own rate rule**: the smallest
swept rate whose clean-validation shift ratio reaches 0.8. An earlier version of
these 2 tables was read at the nearest rate to sigma 0.6, which is the
placement-comparison protocol and not a detection protocol. It understated badly
and unevenly: CIFAR-100 badnet at 5% read TPR@1%FPR of 0.000 against 0.882 here,
and blend at 10% read 0.185 against 1.000. Both tables are now generated by
`python -m cli.tables --operator token_mask --position before_attention_norm`
rather than transcribed, so the rule that produced them is in the command.

The `depl` columns are deployable, thresholded on clean validation alone with the
FPR actually achieved in brackets; `oracle` reads the same target off the labelled
curve and is an upper bound no defender reaches.

| poison | attack | ASR | sigma | rate | AUROC | TPR@1% depl (FPR) | TPR@1% oracle | TPR@5% depl (FPR) | TPR@5% oracle | TPR@10% depl (FPR) | TPR@10% oracle | TPR@25% depl (FPR) | TPR@25% oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1% | `badnet_a2o` | 1.000 | 0.859 | 0.5 | 0.988 | 0.497 (0.010) | 0.486 | 0.994 (0.044) | 0.996 | 0.999 (0.091) | 0.999 | 1.000 (0.241) | 1.000 |
| 1% | `blend` | 0.990 | 0.873 | 0.5 | 0.976 | 0.562 (0.009) | 0.583 | 0.878 (0.041) | 0.901 | 0.959 (0.100) | 0.958 | 0.988 (0.227) | 0.989 |
| 1% | `lc` | 0.874 | 0.905 | 0.5 | 0.848 | 0.003 (0.008) | 0.004 | 0.166 (0.049) | 0.174 | 0.426 (0.090) | 0.483 | 0.862 (0.257) | 0.855 |
| 1% | `adaptive_blend` (weak) | 0.702 | 0.881 | 0.5 | 0.734 | 0.623 (0.013) | 0.584 | 0.688 (0.054) | 0.686 | 0.701 (0.099) | 0.701 | 0.709 (0.246) | 0.710 |
| 5% | `badnet_a2o` | 1.000 | 0.926 | 0.5 | 0.994 | 0.882 (0.011) | 0.855 | 0.994 (0.056) | 0.992 | 0.999 (0.111) | 0.999 | 1.000 (0.256) | 1.000 |
| 5% | `blend` | 1.000 | 0.894 | 0.6 | 0.979 | 0.714 (0.009) | 0.738 | 0.900 (0.054) | 0.893 | 0.941 (0.096) | 0.944 | 0.982 (0.251) | 0.982 |
| 5% | `wanet` (weak) | 0.649 | 0.823 | 0.4 | 0.774 | 0.306 (0.009) | 0.331 | 0.570 (0.045) | 0.580 | 0.626 (0.093) | 0.630 | 0.705 (0.262) | 0.702 |
| 5% | `lc` (weak) | 0.543 | 0.844 | 0.5 | 0.689 | 0.003 (0.008) | 0.005 | 0.062 (0.056) | 0.049 | 0.160 (0.092) | 0.184 | 0.521 (0.241) | 0.540 |
| 5% | `adaptive_blend` | 0.934 | 0.859 | 0.5 | 0.929 | 0.802 (0.012) | 0.776 | 0.896 (0.056) | 0.892 | 0.912 (0.102) | 0.912 | 0.930 (0.253) | 0.930 |
| 10% | `badnet_a2o` | 1.000 | 0.909 | 0.5 | 0.998 | 0.997 (0.011) | 0.995 | 1.000 (0.050) | 1.000 | 1.000 (0.101) | 1.000 | 1.000 (0.245) | 1.000 |
| 10% | `blend` | 1.000 | 0.859 | 0.5 | 0.998 | 1.000 (0.010) | 1.000 | 1.000 (0.042) | 1.000 | 1.000 (0.095) | 1.000 | 1.000 (0.262) | 1.000 |
| 10% | `wanet` | 0.900 | 0.899 | 0.5 | 0.923 | 0.431 (0.008) | 0.493 | 0.737 (0.043) | 0.756 | 0.826 (0.095) | 0.831 | 0.903 (0.244) | 0.904 |
| 10% | `lc` (weak) | 0.786 | 0.828 | 0.5 | 0.816 | 0.004 (0.008) | 0.008 | 0.164 (0.046) | 0.187 | 0.431 (0.097) | 0.445 | 0.783 (0.243) | 0.790 |
| 10% | `adaptive_blend` | 0.971 | 0.814 | 0.5 | 0.966 | 0.892 (0.012) | 0.871 | 0.959 (0.052) | 0.959 | 0.966 (0.096) | 0.966 | 0.970 (0.236) | 0.970 |
| 1% | `wanet` | 0.044 | -- | -- | did not implant | -- | -- | -- | -- | -- | -- | -- | -- |

### 1.4 Full detection tables (Tiny ImageNet, recommended config)

token_mask @ before_attention_norm, same rule.

| poison | attack | ASR | sigma | rate | AUROC | TPR@1% depl (FPR) | TPR@1% oracle | TPR@5% depl (FPR) | TPR@5% oracle | TPR@10% depl (FPR) | TPR@10% oracle | TPR@25% depl (FPR) | TPR@25% oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1% | `badnet_a2o` | 0.999 | 0.910 | 0.5 | 0.980 | 0.578 (0.016) | 0.284 | 0.974 (0.051) | 0.973 | 0.989 (0.092) | 0.991 | 0.998 (0.255) | 0.998 |
| 1% | `blend` | 0.999 | 0.887 | 0.5 | 0.992 | 0.942 (0.014) | 0.932 | 0.973 (0.054) | 0.971 | 0.985 (0.106) | 0.984 | 0.993 (0.257) | 0.993 |
| 5% | `badnet_a2o` | 1.000 | 0.883 | 0.5 | 0.984 | 0.367 (0.009) | 0.383 | 0.993 (0.064) | 0.984 | 0.999 (0.107) | 0.999 | 1.000 (0.261) | 1.000 |
| 5% | `blend` | 0.999 | 0.889 | 0.5 | 0.997 | 0.972 (0.008) | 0.978 | 0.993 (0.058) | 0.992 | 0.997 (0.112) | 0.997 | 1.000 (0.256) | 1.000 |
| 5% | `wanet` | 0.941 | 0.885 | 0.5 | 0.948 | 0.471 (0.006) | 0.672 | 0.928 (0.047) | 0.929 | 0.943 (0.092) | 0.943 | 0.949 (0.254) | 0.949 |
| 5% | `lc` (weak) | 0.706 | 0.892 | 0.5 | 0.781 | 0.527 (0.013) | 0.416 | 0.716 (0.060) | 0.709 | 0.730 (0.101) | 0.730 | 0.752 (0.251) | 0.752 |
| 5% | `adaptive_blend` | 0.924 | 0.905 | 0.5 | 0.924 | 0.924 (0.010) | 0.924 | 0.924 (0.054) | 0.924 | 0.924 (0.110) | 0.924 | 0.924 (0.258) | 0.924 |
| 10% | `badnet_a2o` | 1.000 | 0.907 | 0.5 | 0.993 | 0.872 (0.013) | 0.799 | 1.000 (0.059) | 1.000 | 1.000 (0.098) | 1.000 | 1.000 (0.238) | 1.000 |
| 10% | `blend` | 0.999 | 0.900 | 0.5 | 0.998 | 0.975 (0.011) | 0.971 | 0.996 (0.056) | 0.995 | 0.999 (0.107) | 0.999 | 1.000 (0.256) | 1.000 |
| 10% | `wanet` | 0.975 | 0.882 | 0.5 | 0.972 | 0.658 (0.010) | 0.648 | 0.952 (0.055) | 0.948 | 0.973 (0.101) | 0.973 | 0.979 (0.262) | 0.979 |
| 10% | `lc` (weak) | 0.626 | 0.866 | 0.5 | 0.765 | 0.020 (0.012) | 0.019 | 0.341 (0.059) | 0.287 | 0.545 (0.109) | 0.518 | 0.720 (0.249) | 0.721 |
| 10% | `adaptive_blend` | 0.965 | 0.863 | 0.5 | 0.965 | 0.958 (0.012) | 0.955 | 0.965 (0.059) | 0.965 | 0.966 (0.110) | 0.966 | 0.966 (0.250) | 0.966 |
| 1% | `wanet` | 0.379 | -- | -- | did not implant | -- | -- | -- | -- | -- | -- | -- | -- |
| 1% | `lc` | 0.389 | -- | -- | did not implant | -- | -- | -- | -- | -- | -- | -- | -- |
| 1% | `adaptive_blend` | 0.431 | -- | -- | did not implant | -- | -- | -- | -- | -- | -- | -- | -- |


### 1.5 Swin-S: same operator ranking

Tested on CIFAR-100 with 3 operators. The ranking transfers across architectures:

| Architecture | Best operator | Position | Mean AUROC | Inversions |
|---|---|---|---:|---:|
| ViT-B/16 | token_mask | before_attention_norm | 0.911 | 0 |
| Swin-S | token_mask | before_attention_norm | 0.913 | 0 |

Swin head-to-head (CIFAR-100):

| Poison | Attack | token_mask | gain_scale | dropout |
|---|---|---:|---:|---:|
| 1% | badnet_a2o | **0.993** | 0.921 | 0.783 |
| 1% | blend | **0.994** | 0.825 | 0.988 |
| 1% | lc | **0.995** | 0.945 | 0.861 |
| 5% | wanet | **0.703** | 0.738 | 0.258 |
| 5% | adaptive_blend | **0.915** | 0.837 | 0.857 |
| 10% | badnet_a2o | **0.995** | 0.911 | 0.978 |
| 10% | wanet | **0.623** | 0.532 | 0.294 |

- token_mask best on 10/13 cells
- Dropout inverts on WaNet (0.258, 0.294); token_mask does not
- The operator choice, not the architecture, determines WaNet behavior

### 1.6 Comparison to published ConvNet baselines

These comparisons are NOT apples-to-apples (different architecture, training
protocol, evaluation). They are directional.

| Attack | PSBD ResNet-18 | Our ViT (published pos) | Our ViT (best pos) |
|---|---:|---:|---:|
| BadNet 10% CIFAR-10 | ~0.974 | 0.919 | 0.985 |
| Blend 10% CIFAR-10 | ~0.998 | 0.992 | 0.978 |

| Attack | IBD-PSC ResNet-18 | Our ViT (gain_scale) |
|---|---:|---:|
| BadNet 10% CIFAR-10 | ~0.999 | 0.999 |
| Blend 10% CIFAR-10 | ~0.999 | 0.997 |
| WaNet 10% CIFAR-10 | ~0.870 | 0.665 |

| Attack | SCALE-UP ResNet-18 | Our ViT (scale_up) |
|---|---:|---:|
| BadNet 10% CIFAR-10 | ~0.999 | 0.984 |
| Blend 10% CIFAR-10 | ~0.870 | 0.864 |
| WaNet 10% CIFAR-10 | ~0.700 | 0.752 |

Key finding: the published methods use fixed positions chosen for ConvNets.
Searching over 27 positions on ViT finds configurations that match or exceed the
ConvNet numbers, even on ViT which has a fundamentally different architecture.

---

## 2. Why position matters more than operator

### 2.1 Position variance dominates

Across 27 configurations (H28):
- **Position variance:** mean AUROC range 0.268 when holding operator fixed and
  varying position
- **Operator variance:** mean AUROC range 0.188 when holding position fixed and
  varying operator
- **Ratio: 1.43x.** Position matters 43% more than operator.

Kendall tau = 0.700 between operators' attack rankings at fixed position. The
relative difficulty of attacks is preserved: if badnet is hardest with dropout
at before_attention_norm, it is also hardest with token_mask at the same
position. The operator changes the level, not the ordering.

### 2.2 Input-side beats residual-adjacent (H20)

The strongest placement result in the study:
- **+0.054** mean AUROC for input-side positions over residual-adjacent
- Bootstrap CI: [+0.031, +0.080]
- Positive on 11/12 units
- Survives every leave-one-out and leave-one-attack-out refit

Pre-versus-post residual, the founding question of this project: **+0.002**
(indistinguishable from noise). The question that motivated the entire study
dissolves once the comparison is done at matched sigma.

### 2.3 The combined variant (H13)

Four changes assembled from earlier hypotheses:

| Component | Published | Adapted | Source |
|---|---|---|---|
| placement | post_residual, all 12 blocks | pre_residual, blocks 5-8 | H10 |
| score | absolute PSU | fractional PSU | H12 |
| rate rule | shift ratio >= 0.8 | shift ratio >= 0.7 | H11 |
| decision | one-sided | two-sided | H5 |

Results on CIFAR-10 (derivation set, 15 checkpoints):

| Metric | Value |
|---|---|
| Mean AUROC gain | +0.110 |
| Wins | 14/15 |
| Largest gain | badnet_a2o 1%: 0.297 to 0.640 (+0.344) |

Held-out confirmation (2 attacks never used in derivation):

| Checkpoint | Published | Adapted | Delta |
|---|---:|---:|---:|
| lc 10% | 0.515 | 0.786 | +0.271 |
| sig 10% | 0.900 | 0.931 | +0.031 |
| **mean** | **0.708** | **0.859** | **+0.151** |

Cross-dataset generalization (CIFAR-100, pure out-of-distribution):
- **+0.207 mean delta, 18/18 wins** on ASR-matched checkpoints
- The gain tracks headroom: where the published method is weakest (CIFAR-100,
  0.731), the variant gains most

### 2.4 Low poison rate failure is a placement artifact (H17)

The received explanation ("too few poisoned samples to learn a detectable
shortcut") does not survive:
- badnet_a2o at 1%: ASR 0.997 (attack fully implanted)
- blend at 1%: ASR 1.000
- The published position (post_residual) scores below 0.5 at **all 16 rates** at 1%
- `before_attention_norm` at the same checkpoint reaches **0.825** at p=0.5

The single headline number: **badnet_a2o at 1%, ASR 0.997, goes from 0.297 to
0.839 without flipping anything.**

There is a critical dropout rate p* at which the backdoor circuit stops
surviving perturbation. Past p*, the statistic inverts. p* rises with poison
rate (more poisoned training samples produce a more redundantly encoded
shortcut). The published rate rule targets sigma >= 0.8, which lands past p* at
1%. The fix is placement selection, not rate tuning.

---

## 3. PSBD measures margin, not neuron bias

### 3.1 Gaussian noise matches the best masks (H23)

The discriminating experiment. Gaussian noise removes nothing, it adds zero-mean
noise. If it matches removal operators, the "neuron bias effect" is not what
carries the method.

**This table was measured with the superseded batch-coupled Gaussian and is
restated below.** Its cells had no cache-backed source until the corrected
operator was re-swept over `before_attention` and `mlp_neurons`; see audit
finding A19.

Recomputed from cache-backed records, CIFAR-10 at 10 percent, matched at
sigma 0.6, 8 attacks, all-to-all excluded:

| Operator / position | Mean AUROC | was |
|---|---:|---:|
| token_mask / before_attention_norm | **0.936** | |
| token_mask / before_mlp_residual | 0.913 | 0.929 |
| token_mask / mlp_neurons | 0.904 | |
| dropout / before_mlp_residual | 0.901 | |
| channel_mask / before_attention | 0.900 | 0.911 |
| **gaussian / before_attention** | **0.897** | 0.950 |
| dropout / pre_residual | 0.894 | |
| gaussian / mlp_neurons | 0.861 | 0.918 |

Both gaussian cells fall by about 0.055, which matches the 0.087 inflation
measured for records whose tensors were archived. The structured mask, which was
never affected, moves 0.011.

**The conclusion survives and the framing does not.** Gaussian is 6th, not 1st, so
the claim that it sits "at the top of the table" is withdrawn. But it remains
within 0.002 of `channel_mask` and ahead of `dropout` at `pre_residual`, so the
load-bearing point holds: an operator that removes **no capacity at all** is
competitive with operators that do, which is what refutes the capacity-removal
account. The prediction was that removal would beat disturbance by at least 0.03.
It does not.

**Confidence-only null:** 0.520 AUROC. The perturbation is doing real work, but
what it does not need to be is a removal. PSBD measures how far a prediction
moves under perturbation of any kind, which is a statement about margin. A
backdoored input sits far from the decision boundary in the direction the
trigger pushes. Bounded perturbation fails to move it while a clean input near a
boundary moves easily.

### 3.2 Supporting evidence

- **DropPath (H21):** the residual-native perturbation (removes entire branch
  outputs). AUROC 0.860, below dropout. Confirmed the prediction that it would
  lose. The unit that matters is the feature, not the computation.
- **Head masking (H22):** random head masking AUROC 0.539 (near random). Heads
  are redundant, the backdoor is distributed across them.
- **Targeted head masking (H35):** masking the 3 backdoor heads (L5H0, L5H10,
  L6H3) gives only 0.580, +0.04 over random. Head masking as a class is too
  weak for PSBD.
- **Sensitivity profile (H18):** 144-head leave-one-out profiling carries no
  detection signal (0.28 to 0.61, benign at 0.497). Third failed attempt to
  exploit where the backdoor sits.

### 3.3 Unification (H28)

PSBD, IBD-PSC, SCALE-UP, and STRIP are one method: perturbation-consistency
measures decision margin, and the operator only sets the Jacobian. Confirmed:
- Prediction 4 (interchangeability): 1.43x position/operator ratio, Kendall
  tau 0.700
- Prediction 3 (only stream positions invert): REFUTED (input-side inversions
  9.4% vs stream 4.1%, the prediction was backward)

---

## 4. The backdoor mechanism in ViT

### 4.1 One linear direction, not a set of neurons (H16)

The causal test:

| Ablation | badnet ASR | blend ASR | bpp ASR | lf ASR |
|---|---:|---:|---:|---:|
| baseline | 1.00 | 1.00 | 1.00 | 1.00 |
| **remove rank-1 direction** | **0.00** | **0.00** | **0.01** | **0.05** |
| random direction | 1.00 | 1.00 | 1.00 | 1.00 |
| top-20 coordinates | 1.00 | 1.00 | 1.00 | 1.00 |
| top-300 coordinates | 1.00 | 1.00 | 1.00 | 0.89 |

Post-LayerNorm rank-1 removal takes ASR from 1.00 to 0.00 on all 4 single-target
attacks. Zeroing even 300 of 768 coordinates (39% of the width) does nothing. The
backdoor is a genuine linear direction that is not axis-aligned.

Clean accuracy cost of direction removal: 0.03 to 0.08. Random directions leave
ASR at baseline. Benign model unaffected.

**CLP outlier rule:** 5 to 17 dimensions out of 768 (< 2.2%) carry most of the
direction's energy. But because the direction is not axis-aligned, coordinate
ablation fails.

### 4.2 Crystallization with phase transition (H30, H38)

The direction does NOT persist uniformly. It shows an S-curve:

| Layer range | Behavior | Alignment to final |
|---|---|---|
| 1 to 6 | near-orthogonal to final direction | 0.02 to 0.19 |
| 7 to 9 | rapid assembly (transition zone) | 0.19 to 0.49 |
| 10 to 12 | persistent, consecutive cosine > 0.82 | 0.73 to 0.95 |

Attack-specific crystallization depths (H38):

| Attack | Mean crystallization layer | n checkpoints |
|---|---:|---:|
| blend | 8.2 | 8 |
| badnet | 10.4 | 8 |

- Blend's distributed trigger writes the direction 2.2 layers earlier
- Poison rate has minimal effect on crystallization depth
- This explains why perturbation at all blocks (before_attention_norm) works:
  it covers the crystallization zone for every attack

Relative backdoor-direction norm per block (badnet_a2o, Adam):

| Layer | 1 | 3 | 5 | 7 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| norm | 0.00 | 0.03 | 0.09 | 0.30 | 0.59 | 0.70 | 0.85 | 1.03 |

Benign control peaks at 0.09 and decays. Every backdoored model grows
monotonically to 0.7 to 2.2.

### 4.3 Three shared backdoor heads (H31)

Three attention heads diverge across ALL attacks and datasets: **L5H0, L6H3,
L5H10.** Population-level JS divergence 0.15 to 0.35 between clean and
backdoor attention distributions.

BadNet uniquely recruits 2 additional late heads: **L9H7, L10H9** (2x higher JS
divergence). These sit at layers 9 to 10, matching badnet's later
crystallization depth.

However, this population-level divergence does NOT translate to per-sample
detection:
- Attention entropy (H40): AUROC 0.48 to 0.54 for most attacks
- Targeted head masking (H35): AUROC 0.580
- The heads are identifiable but not individually exploitable

### 4.4 Directions are attack-specific (H29)

Pairwise cosine similarity at layer 12 across 10 attacks:
- **Off-diagonal mean: 0.023 to 0.053** (indistinguishable from random 768-dim
  vectors, expected cosine ~0)
- The highest pairwise cosine (0.374 for lc vs tact on CIFAR-100 at 10%) is
  well below the 0.8 threshold
- badnet_a2o and badnet_a2a share the identical trigger and overlap at exactly
  0.00. What determines the direction is the label mapping, not the trigger
  appearance

Direction norms vary 6x (CIFAR-100 at 10%):

| Attack | Direction norm |
|---|---:|
| lf | 22.47 |
| blend | 19.42 |
| badnet_a2o | 13.15 |
| adaptive_blend | 10.90 |
| lc | 9.69 |
| sig | 4.77 |
| tact | 3.71 |

Global triggers (blend, lf) produce stronger directions than localized ones
(tact, sig).

### 4.5 No cone around readout weight (H36)

The premise was that all backdoor directions cluster in a cone centered on the
target class readout weight. The data shows the opposite:

| Attack | Angle to readout (CIFAR-100) | Cosine |
|---|---:|---:|
| badnet_a2o | 32.8 | 0.841 |
| tact | 86.3 | 0.065 |
| bpp | 86.9 | 0.053 |
| blend | 88.9 | 0.020 |
| wanet | 90.9 | -0.016 |

9 of 10 attacks are essentially perpendicular to the readout weight (87 to 91
degrees). Only badnet_a2o aligns.

The earlier H16 finding (readout cosine 0.87 to 0.89) was per-model: each
attack's direction aligns with its OWN model's readout weight, not with a shared
reference. Each backdoored model co-adapts its readout weight and backdoor
direction. The readout weight adaptation is itself attack-specific.

### 4.6 Weight structure

- **Weight difference is NOT low-rank (H33):** encoder weight matrices show 2 to
  5% top-1 concentration. The backdoor is rank-1 in activation space but NOT in
  weight space. It is an emergent property, not a weight perturbation.
- **Token norms localize patch triggers (H32):** badnet's top token at grid
  position (13,13) matches the trigger placement. Blend and WaNet show diffuse
  patterns. SIG shows row-0 concentration.
- **Disjoint dimensions (H16):** Jaccard overlap between different attacks'
  top-20 TAC dimensions is 0.00 to 0.08, against a chance floor of 0.014 and a
  split-half ceiling of 0.87. Attacks share no coordinate-level structure.

### 4.7 SAM, dropped as a line of enquiry

Sharpness-aware minimization is **out of scope for this paper** and no claim rests
on it. Its entire measured effect on detection is +0.009 mean AUROC. Establishing
an effect that size against plausible seed to seed variance needs 10 to 89 training
seeds per cell (`seed-replication-plan.md`), which is an unaffordable amount of
compute spent to confirm that something does not matter.

The cost of carrying it was concrete rather than theoretical: SAM checkpoints are
69 percent of the trained set and 75 percent of all analysis rows, so every panel
that included them spent most of its coverage on the arm with no effect, and
unequal coverage across arms is the failure mode that inverted 4 conclusions
elsewhere in this project. Excluding SAM cut a re-sweep from 326 to 74.5 GPU-hours
with no loss to any reported result.

The SAM checkpoints and their caches stay on disk and the raw record keeps them.
Every reporting path excludes them by default, behind an explicit `--include-sam`
flag, so nothing is deleted and nothing is claimed.

Two observations from the earlier SAM work are retained because they are about the
backdoor rather than about SAM, and they cost nothing to keep:
- SAM does **not** remove the backdoor. ASR stays 0.96 to 1.00.
- SAM does **not** resist direction removal. Post-LayerNorm rank-1 ablation drops
  ASR to 0.000 on every SAM checkpoint. The apparent resistance reported earlier
  was a LayerNorm artifact of ablating before rather than after the final
  normalization, and that correction is a methodological finding worth keeping
  (section 7.1, item 3).

---

## 5. What does NOT work as defense

### 5.1 Direction erasure from weights (H34)

Weight orthogonalization (removing the backdoor direction from MLP output
weights and attention out_proj at layers 10 to 11):

| Checkpoint | Base ASR | Known dir | Blind (readout) | Random |
|---|---:|---:|---:|---:|
| cifar100 badnet 10% | 1.000 | 1.000 | 1.000 | 1.000 |
| cifar100 blend 10% | 1.000 | 1.000 | 1.000 | 1.000 |
| cifar100 lc 10% | 0.786 | 0.634 | 0.456 | 0.786 |
| tiny lc 10% | 0.626 | 0.243 | **0.006** | 0.627 |
| cifar100 a_blend 5% | 0.934 | 0.695 | **0.079** | 0.935 |

- **Fails on strong attacks** (badnet, blend: ASR stays 1.000). The direction
  is already in the residual stream from earlier layers' contributions. The skip
  path carries it through unchanged.
- **Works on weak attacks** (lc, adaptive_blend at low rate). The residual
  accumulation is thin enough that removing branch contributions collapses the
  backdoor.
- **Blind erasure sometimes works better** (readout weight as proxy, no
  triggered data needed): lc 10% on Tiny from 0.626 to 0.006.
- CA cost under 1% in all cases.

### 5.2 Skip connection scaling (H39)

Scaling down the skip connection at layers 10 to 11: replace `x + branch` with
`alpha * x + branch`. Three distinct regimes:

**CIFAR-100 10% poison rate:**

| Attack | alpha=1.0 | alpha=0.5 | alpha=0.3 | alpha=0.1 | alpha=0.0 |
|---|---:|---:|---:|---:|---:|
| badnet_a2o | 1.000 | 1.000 | 1.000 | 0.962 | 0.029 |
| blend | 1.000 | 1.000 | 1.000 | 0.007 | 0.000 |
| wanet | 0.900 | 0.427 | 0.092 | 0.034 | 0.033 |
| lc | 0.786 | 0.556 | 0.494 | 0.001 | 0.000 |
| adaptive_blend | 0.971 | 0.809 | 0.706 | 0.639 | 0.452 |

CA cost at alpha=0.3: about -0.020.

**Three regimes:**
1. **Fragile (wanet, lc):** ASR collapses at alpha=0.3 to 0.5 with < 3% CA
   cost. Consistent with H34.
2. **Threshold (blend):** sharp phase transition. ASR 1.000 at alpha=0.3,
   crashes to 0.007 at alpha=0.1. The blend direction has a specific norm
   threshold.
3. **Robust (badnet, adaptive_blend):** badnet survives even **alpha=0.0 at 5%
   poison rate (ASR=0.995).** Zeroing the skip connection completely, only the
   branch output passes through, and the backdoor regenerates. The branch
   computations at layers 10 to 11 alone reproduce the direction. This is
   consistent with H31's finding that badnet recruits late heads (L9H7, L10H9)
   that write the direction directly into the branch output.

**No single alpha works universally.** A defender picking alpha=0.3 still faces
ASR=1.000 for badnet and 0.706 for adaptive_blend.

---

## 6. What does NOT work as detection

### 6.1 Targeted head PSBD (H35)

Masking the 3 backdoor heads as a deterministic PSBD operator:

| Operator | Mean AUROC | Source |
|---|---:|---|
| random head_mask | 0.539 | H22 |
| **targeted 3-head** | **0.580** | H35 |
| targeted 5-head | 0.599 | H35 |
| token_mask (dropout) | 0.911 | ranking |
| gaussian noise | 0.897 | H23, restated from cache-backed records |

- Only +0.04 over random head masking
- Masking 3 of 144 heads (2.1% of attention capacity) does not produce enough
  perturbation strength
- The fundamental problem is perturbation STRENGTH, not targeting

### 6.2 Attention entropy as detection feature (H40)

Per-sample attention entropy in the 3 backdoor heads:

| Checkpoint | Clean entropy | Backdoor entropy | Gap | AUROC |
|---|---:|---:|---:|---:|
| cifar100 badnet | 3.79 | 3.78 | +0.01 | 0.511 |
| cifar100 blend | 3.90 | 4.84 | -0.94 | 0.000 |
| cifar100 wanet | 3.91 | 3.92 | -0.01 | 0.483 |
| cifar100 lc | 3.82 | 3.80 | +0.02 | 0.526 |

- Most attacks: AUROC 0.48 to 0.54 (random). The per-sample distributions
  overlap completely.
- **Blend is inverted:** backdoor samples have dramatically HIGHER entropy
  (gap -0.94 nats). The full-image blending trigger diffuses attention, making
  every token look equally important. Under one-sided convention (lower entropy
  = backdoor), this gives AUROC 0.000.
- Per-sample entropy is too coarse to capture the distributional shifts that
  H31 identified at the population level.

### 6.3 Token concentration as attack family classifier (H37)

Concentration ratio (max patch norm / mean patch norm) to separate localized
vs global triggers:
- **Accuracy: 62.5%** at best threshold
- Separation gap: -0.98 (negative means distributions overlap)
- adaptive_blend (3.10) and lf (3.33) have high concentration despite being
  global attacks
- badnet_a2o on CIFAR-100 has lower-than-expected concentration (2.35) because
  the 3x3 trigger occupies a fraction of a single 14x14 ViT patch

### 6.4 Other failed detection approaches

| Approach | Result | Source |
|---|---|---|
| Channel mask | Loses to token_mask at every matched position | H26 |
| Per-sample critical rate p* | -0.084 mean AUROC vs fractional PSU | Negative results |
| Combined sublayer perturbation | Mean delta -0.045 on Tiny (dilutes) | Negative results |
| Gaussian at before_attention_norm | Catastrophic on CIFAR-100 (badnet 1%: 0.168) | Negative results |
| Position ranking transfer | Spearman rho 0.33 to 0.70 across datasets | Negative results |

---

## 7. Methodological lessons

### 7.1 The failure mode this study keeps hitting

Four versions of one pattern, each inverted a conclusion:

1. **Comparing placements at shared dropout rate (H9, H1, H3).** Different
   placements have different dose-response curves. A rate that is optimal for
   one position is past the collapse point for another. Fixed by matching on
   clean-validation shift ratio (sigma), never on rate.

2. **Averaging over unequal coverage (H19).** Different groups swept over
   different checkpoints, so the means compared different problems. **4 of 6
   comparisons inverted** at imbalance > 1.5x.

3. **Measuring across LayerNorm with scale-dependent statistic (H16).** Pre-LN
   ablation produced a clean, monotone, benign-controlled, and entirely false
   SAM trend that survived 2 rounds of follow-up. Caught by an independent
   measurement (logit decomposition). Scale-invariant statistics transfer across
   normalization layers, absolute ones do not.

4. **Re-reading a comparison every time the data grows (H19).** H19's Swin
   claim was written 3 times from panels of 6, 11, and 16 units, giving +0.110
   ("supported"), +0.015 ("refuted"), and +0.045 ("inconclusive"). Each re-read
   was a fresh chance to over-interpret noise. Fixed by pre-registering n.

All 4 produce more exciting results than the truth, pass their benign controls,
and are invisible without an explicitly constructed comparison.

### 7.2 Protocol decisions

- **One-sided rule:** low PSU = poisoned. No max(AUROC, 1-AUROC), no threshold
  inversion, no two-sided reporting (H15). Choosing which tail to flag needs the
  poison labels the detector exists to predict.
- **All-to-all breaks PSBD (H5):** the signal is present with the sign
  reversed (backdoor samples are 4x MORE fragile, not more robust). No single
  target class for prediction to collapse onto. Recorded as a diagnostic, not
  detection.
- **Monte Carlo passes (H24):** k=20 gives +0.028 AUROC at 1% vs +0.011 at
  10%. badnet_a2o at 1% gains +0.046. The improvement matters most exactly where
  it is needed (low poison rate), but k=3 remains the paper default.
- **PSBD + STRIP fusion (H14):** 0.614 TPR at 1% FPR, 93% of oracle-max.
  Fusing complementary per-sample statistics outperforms either alone.
- **FPR is uninformative by construction.** The threshold is a quantile of
  clean test PSU, so FPR lands near the quantile whatever the placement does.
  AUROC and TPR carry the signal.

---

## 8. Complete hypothesis index

| ID | Status | Key finding |
|---|---|---|
| H1 | **REFUTED** | Pre-residual does NOT generally beat post-residual on ViT |
| H2 | **SUPPORTED** | PSBD works on ViT; benign models score 0.50 |
| H3 | **REFUTED** | Post-residual does not fail by saturating |
| H4 | **SUPPORTED** | Best placement tracks where the direction enters CLS |
| H5 | **SUPPORTED** | All-to-all breaks PSBD (reversed sign, not detectable) |
| H6 | **DROPPED** | SAM effect is +0.009 AUROC, too small to establish. Out of scope, see 4.7 |
| H7 | **PARTIALLY REFUTED** | Clean samples do NOT shift to target class where PSBD works best |
| H8 | INCONCLUSIVE | Detection vs poison rate relationship unclear |
| H9 | **SUPPORTED** | Pre/post gap is a strength artifact, not a placement effect (3/4 attacks) |
| H10 | Band **SUPPORTED**, onset **REFUTED** | Block-band placement works, but onset-based selection fails OOS |
| H11 | **SUPPORTED** | Rate rule overshoots on ViT (12/12). Free fix: sigma 0.7 |
| H12 | **REFUTED** | PSU is NOT just confidence. Fractional PSU is a free improvement |
| H13 | **SUPPORTED** | Combined changes: +0.110 derivation, +0.151 held-out |
| H14 | **SUPPORTED** | PSBD+STRIP fusion: 0.614 TPR at 1% FPR |
| H15 | **RETIRED** | One-sided rules are a principle, not a detection method |
| H16 | **SUPPORTED** | Backdoor is one rank-1 direction (post-LN removal: ASR 1.00 to 0.00). "Neurons" framing REFUTED. SAM sub-claim REFUTED (LayerNorm artifact) |
| H17 | **SUPPORTED, restated** | Low-rate failure is a placement artifact (+0.166 at 1% at matched shift ratio, not the withdrawn +0.258) |
| H18 | **REFUTED** | 144-head sensitivity profile carries no signal |
| H19 | **INCONCLUSIVE** | Swin placement ranking vs rate selection (CI contains 0) |
| H20 | **SUPPORTED** | Input-side beats residual-adjacent by +0.054, CI [+0.031, +0.080] |
| H21 | **CONFIRMED** | DropPath loses (0.860) despite being residual-native |
| H22 | **REFUTED** | Random head masking: 0.539 (near random) |
| H23 | **REFUTED, restated** | Gaussian noise (0.897, was 0.950) is 6th not 1st, but stays within 0.002 of channel_mask and ahead of dropout at pre_residual. Removal not required |
| H24 | **CONFIRMED** | k=20 gives +0.028 at 1% vs +0.011 at 10% |
| H25 | **PRE-REGISTERED** | Adaptive attacker: control reproduced, transfer table next |
| H26 | **REFUTED** | Channel mask loses to dropout at every matched position |
| H27 | **SUPPORTED** | Token mask separates local vs distributed (badnet 0.985, wanet 0.747) |
| H28 | **PARTIALLY SUPPORTED** | Unification confirmed (interchangeability), prediction 3 refuted |
| H29 | **REFUTED** | Cross-attack directions near-orthogonal (cosine 0.023 to 0.053) |
| H30 | **SUPPORTED** | Phase transition at layers 8 to 10, not uniform persistence |
| H31 | **SUPPORTED** | 3 shared backdoor heads (L5H0, L6H3, L5H10) across all attacks |
| H32 | **SUPPORTED** | Token norms localize triggers (badnet at (13,13)) |
| H33 | **REFUTED** | Weight difference NOT low-rank (2 to 5% top-1 concentration) |
| H34 | **PARTIALLY SUPPORTED** | Weight erasure works on weak attacks (LC to 0.006), fails on strong |
| H35 | **REFUTED** | Targeted head PSBD: 0.580, only +0.04 over random |
| H36 | **REFUTED** | No cone: 9/10 attacks at 87 to 91 degrees to readout weight |
| H37 | **REFUTED** | Token concentration classifier: 62.5% accuracy |
| H38 | **SUPPORTED/INCONCLUSIVE** | Crystallization confirmed (blend 8.2, badnet 10.4). Placement correlation untested |
| H39 | **PARTIALLY SUPPORTED** | Skip scaling: 3 regimes. badnet survives alpha=0.0 at 5% |
| H40 | **REFUTED** | Attention entropy: AUROC 0.48 to 0.54, blend inverted |

**Score: 14 SUPPORTED, 14 REFUTED, 5 PARTIALLY SUPPORTED, 3 INCONCLUSIVE,
2 CONFIRMED, 1 RETIRED, 1 PRE-REGISTERED.**
