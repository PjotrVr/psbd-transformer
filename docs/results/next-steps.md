# Next Steps

## High priority (directly tests the theory)

### 1. Direction norm vs poison rate (H28 prediction 1)
**DONE (job 1022128).** Direction norm is NOT monotonically increasing with
poison rate (Spearman rho = -0.40 to +0.80, none significant). The norm is
large (12 to 19) at all rates from 0.5% to 10%, meaning the model fully
encodes the backdoor direction even at very low poison rates. The stronger
finding is the layer-wise growth: 159x increase from layer 1 to 12, consistent
across all 16 checkpoints. See [direction-norm-analysis.md](direction-norm-analysis.md).

### 2. Adaptive attacker transfer table (H25)
**IN FLIGHT (jobs 1021980, 1021981 for CIFAR-10 pilot; 1021988 to 1021999 for
the comprehensive matrix).** Three phases:

Phase 1 (job 1021980): Finish WaNet dropout training (walltime-killed at epoch 14),
then sweep the two existing evade_l0 controls (`vit_cifar10_badnet_a2o_001_evade_l0`
and `vit_cifar10_badnet_a2o_01_evade_l0`) with the full operator set: token_mask,
gain_scale, gaussian, and dropout at multiple positions.

Phase 2 (job 1021981): Train lambda=1 and lambda=10 evasion models for badnet_a2o
at 0.01 and 0.10 poison rates (4 new checkpoints), then sweep each with the full
operator set.

Phase 3 (jobs 1022004 to 1022123): Comprehensive adaptive attacker matrix.
120 individual jobs (one per attack), covering all 10 attacks x 2 datasets
(CIFAR-100, Tiny) x 3 poison rates (1%, 5%, 10%) x 2 architectures (ViT, Swin).
Each model is trained with the hinge evasion penalty (lambda=1.0, 15 epochs)
against the best operator per architecture:

- **ViT**: token_mask @ before_attention_norm (best: 0.911 mean, 0 inversions)
- **Swin**: dropout @ before_attention_norm (best: 0.860 mean, 0 inversions)

After training, each checkpoint is swept with 4 operators to test transfer:
the probed operator plus 3 transfer operators (dropout, gain_scale, gaussian
for ViT; token_mask, gain_scale, pre_residual for Swin).

The key question: if an attacker evades the best operator, does the evasion
transfer to other operators? If not, operator diversity is defense in depth.

The PSBD paper (Appendix Section 7) claims the defender can simply increase the
dropout rate to counter an adaptive attacker. Our test is harder: we use a
smarter hinge penalty (vs their naive PSU minimization for all samples), compute
every step (vs their every 50 iterations), and probe against the best operator
(not just dropout).

### 3. k=20 Monte Carlo passes on CIFAR-100 and Tiny at 1%
**IN FLIGHT (job 1021983).** Sweeps the top 3 configurations (token_mask,
gaussian, gain_scale) at k=20 on 8 checkpoints (4 attacks x 2 datasets at 1%).
H24 confirmed +0.028 AUROC on CIFAR-10 when increasing from k=3 to k=20.
This tests whether the gain transfers to the hard datasets where the margin is
thinnest.

## Medium priority (strengthens existing findings)

### 4. Swin architecture analysis
**OPERATOR COMPARISON DONE (CIFAR-100).** Token_mask is the best Swin operator
(mean AUROC 0.913, 0 inversions), matching the ViT ranking. Gain_scale (0.844,
0 inversions) is second. Dropout (0.783, 2 WaNet inversions) is third. The
operator ranking is architecture-invariant.

Remaining: Swin benign non-SAM (job 1021986, may be complete). Swin on datasets
beyond CIFAR-100 not yet tested for non-dropout operators. See
[swin-detection-analysis.md](swin-detection-analysis.md).

### 5. Benign control audit
**DONE.** 847 cells measured, mean AUROC 0.495, range 0.416 to 0.521. No false
positive bias anywhere in the grid.

### 6. Finer rate grid for residual-stream positions
Residual-stream positions (before_attention_residual, before_mlp_residual) are
evaluated on the 0.1 to 0.9 rate grid, but (1-p)^12 means even p=0.1 removes 72%
of the signal through the 12-block ViT stack. These positions need a finer grid
(0.01 to 0.09) to find their true optimum.

Requires: GPU jobs, ~45 min per checkpoint per position.

## Lower priority (paper completeness)

### 7. STRIP fusion (H14)
Extend STRIP to CIFAR-100 and Tiny for full coverage.

Requires: GPU jobs.

### 8. Stealth metrics for each attack
**DONE.** PSNR and SSIM computed for all 9 attacks across all 4 datasets.
See [stealth-metrics.md](stealth-metrics.md). WaNet is most stealthy (PSNR 32.0,
SSIM 0.976), Blend/SIG/LC least stealthy (PSNR 20 to 23). Detection difficulty
does not track stealth: WaNet is the hardest to detect despite being most
stealthy, but Blend (least stealthy) is the easiest to detect.

### 9. Additional attacks as supplementary
**DONE.** BPP and LF documented in [supplementary-attacks.md](supplementary-attacks.md).
Both detected reliably (mean AUROC 0.960 to 0.969).

### 10. ViT vs Swin comparison table
**PARTIALLY DONE.** Dropout comparison complete: Swin wins 9/9 cells (+0.113 mean).
Remaining: compare with matched operators once Swin has token_mask/gain_scale data
(depends on jobs 1021984/1021985).

### 11. Detection at realistic operating points
**DONE.** TPR at 1%, 5%, 10%, 25% FPR computed for all 9 attacks across all 4
datasets, 3 poison rates, 3 configurations. See
[detection-operating-points.md](detection-operating-points.md) (summary) and
[detection-operating-points-detail.md](detection-operating-points-detail.md) (full
tables). Key finding: token_mask at before_attention_norm is the recommended
deployment configuration (zero inversions, most uniform ROC curves).

### 12. SAM analysis
**DONE.** SAM has negligible detection effect (+0.009 mean AUROC at rho=0.1, 15/18
cells positive) and destabilizes weak attack implantation. Not worth the doubled
training cost. See [detection-operating-points.md](detection-operating-points.md).

## Jobs in flight

### Prior jobs

| Job ID | Name | What | Est. time |
|---|---|---|---|
| 1021980 | finish_wanet | WaNet dropout train + evade_l0 sweep | 6h |
| 1021981 | evade_lambda | lambda=1,10 train + full sweep | 24h |
| 1022128 | direction_norm | H28 prediction 1 (resubmitted) | 4h |
| 1021983 | k20_sweep | k=20 on CIFAR-100/Tiny 1% | 24h |
| 1021984 | Swin_002 | gain_scale on 15 Swin ckpts | 24h |
| 1022129 | Swin_003 | token_mask on 15 Swin ckpts (resubmitted) | 24h |
| 1021986 | swin_benign_nosam | Swin benign train + sweep | 4h |

### Adaptive attacker matrix (120 jobs, v2 resubmission)

Jobs 1022219 to 1022338. Replaces 1022004-1022123, which had 5 configuration
bugs found in the 2026-08-16 codebase audit (see codebase-audit-2026-08-16.md):

1. Probe rate was hardcoded 0.5, destroying ViT predictions (penalty=0)
2. Penalty used absolute PSU, defence reports fractional PSU
3. Swin PSU computed with stochastic depth live (model.train)
4. Evasion passes k=2, defender uses k=3
5. Walltime too short (checkpoints lost on kill)

v2 fixes: auto-calibrated probe rate (sigma=0.6 target), fractional PSU,
eval mode for PSU computation, k=3 passes, increased walltime.

One attack per job, lambda=1.0, 15 epochs, 4 transfer operators swept per
checkpoint.

Evasion operator: token_mask @ before_attention_norm (ViT), dropout @
before_attention_norm (Swin). Transfer sweeps: 4 operators per architecture.

Note: the Swin evasion jobs evade against dropout, which the 2026-08-16
operator comparison showed is NOT the best Swin operator (token_mask is,
at 0.913 vs 0.783 mean AUROC). This makes the Swin evasion a conservative
test: if evasion trained against a weaker operator transfers to the stronger
token_mask, that is a stronger finding.

| Range | Arch | Dataset | Rate | Walltime | Count |
|---|---|---|---|---|---|
| 1022219 to 1022228 | vit | cifar100 | 10% | 6h | 10 |
| 1022229 to 1022238 | vit | cifar100 | 5% | 6h | 10 |
| 1022239 to 1022248 | vit | cifar100 | 1% | 6h | 10 |
| 1022340 to 1022369 | vit | tiny | all | 12h | 30 |
| 1022279 to 1022288 | swin | cifar100 | 10% | 6h | 10 |
| 1022289 to 1022298 | swin | cifar100 | 5% | 6h | 10 |
| 1022299 to 1022308 | swin | cifar100 | 1% | 6h | 10 |
| 1022370 to 1022399 | swin | tiny | all | 12h | 30 |

## Completed results documents

- [operator-position-ranking.md](operator-position-ranking.md)
- [cifar100-detection-tables.md](cifar100-detection-tables.md)
- [tiny-detection-tables.md](tiny-detection-tables.md)
- [cifar10-gtsrb-detection-tables.md](cifar10-gtsrb-detection-tables.md)
- [perturbation-families.md](perturbation-families.md)
- [combined-position-test.md](combined-position-test.md)
- [negative-results.md](negative-results.md)
- [comparison-to-published.md](comparison-to-published.md)
- [theoretical-explanation.md](theoretical-explanation.md)
- [attack-specific-analysis.md](attack-specific-analysis.md)
- [detection-operating-points.md](detection-operating-points.md) + [detail](detection-operating-points-detail.md)
- [stealth-metrics.md](stealth-metrics.md)
- [swin-detection-analysis.md](swin-detection-analysis.md)
- [supplementary-attacks.md](supplementary-attacks.md)
- [direction-norm-analysis.md](direction-norm-analysis.md)
- [codebase-audit-2026-08-16.md](codebase-audit-2026-08-16.md)
