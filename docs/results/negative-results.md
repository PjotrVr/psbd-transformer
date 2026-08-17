# Negative Results

Documenting what did NOT work, so the reasoning is preserved and the same dead ends are not revisited.

## 1. Per-sample critical rate (p*) does not beat sigma-matched PSU

**What we tested.** For each sample, find the smallest perturbation rate at which its prediction first flips (the critical rate p*). Use p* as a detection score (lower p* = easier to flip = more likely poisoned). Compare AUROC of p* against sigma-matched fractional PSU.

**Result.** p* loses by -0.084 mean AUROC across all 27 configurations and 48 cells. p* wins on only 17% of cells.

**Why.** The rate grid is 0.1 to 0.9 in steps of 0.1. At this granularity, many samples have the same p* (they all first flip at the same grid point), creating large tied groups that cannot be ranked. Fractional PSU is a continuous score (probability ratios) with no ties. A finer grid (0.01 steps) would produce fewer ties but was not run because even the conceptual advantage is unclear: p* discards the magnitude of the shift, while PSU preserves it.

**Implication.** Do not pursue p* further. If a rate-sweeping approach is needed, use the continuous PSU score at the sigma-matched rate rather than a binary flip threshold.

## 2. Combined sublayer inputs dilute rather than compound (both_sublayer_inputs)

**What we tested.** Perturb both `before_attention_norm` and `before_mlp_norm` simultaneously, perturbing both sublayer computation inputs in each block.

**Result.** On Tiny ImageNet, every cell is worse (mean delta -0.045 AUROC). On CIFAR-100, it is a wash (mean delta -0.001). See [combined-position-test.md](combined-position-test.md) for the full head-to-head tables.

**Why.** `before_mlp_norm` is individually weak (rank 15/27, mean AUROC 0.800, 3 inversions). Adding it to the strong `before_attention_norm` forces each site to receive weaker perturbation at matched sigma, diluting the effective perturbation at the useful site.

**Implication.** Single-position perturbation is preferable. Do not stack positions unless each position is independently strong.

## 3. Head masking (H22) loses to token masking

**What we tested.** `head_mask @ attention_heads`: mask entire attention heads (set their output to zero) rather than masking tokens or features.

**Result.** Rank 13/27. Mean AUROC 0.839, mean 1% AUROC 0.797. Zero inversions (same as token_mask @ before_attention_norm) but 0.072 lower mean AUROC and 0.107 lower 1% AUROC.

**Why.** The backdoor direction is non-axis-aligned in attention-head space (H16). Masking one head out of 12 removes 1/12 of each dimension of the representation. If the backdoor direction is evenly spread across heads, removing one head removes only 1/12 of the backdoor signal, which is too little to flip the prediction for low-confidence poisoned samples.

**Implication.** Head-level granularity is too coarse. Token masking at the computation input is more effective because it removes entire spatial patches, which can eliminate the trigger entirely for spatially concentrated attacks.

## 4. 144-head sensitivity profile (H18) carries no detection signal

**What we tested.** Leave-one-out head importance profiling across all 144 heads (12 layers x 12 heads). For each head, compare the clean accuracy drop when that head is masked. The idea: heads that are more important for poisoned samples than clean samples should reveal backdoor reliance.

**Result.** No significant difference between clean and poisoned profiles. The per-head importance is dominated by general function (some heads are important for all samples) rather than backdoor-specific function.

**Why.** Same as head masking: the backdoor direction is distributed across many heads, not concentrated in a few. A leave-one-out perturbation is too small to reveal backdoor-specific reliance.

## 5. Position ranking does NOT transfer across datasets

**What we tested.** For each dataset, rank the 27 configurations by mean AUROC. Compute pairwise Spearman rho between datasets.

**Result.** Spearman rho: CIFAR-10/CIFAR-100 = 0.70, CIFAR-10/Tiny = 0.51, CIFAR-100/Tiny = 0.58, CIFAR-10/GTSRB = 0.33, CIFAR-100/GTSRB = 0.41, GTSRB/Tiny = 0.36. Top-5 overlap is poor: 0 to 2 of 5 shared positions between datasets.

**Why.** The Jacobian (and therefore the optimal perturbation site) depends on the learned weights, which vary with dataset complexity and the number of classes. A 100-class dataset learns different attention patterns from a 10-class one.

**Implication.** Do not assume a position that wins on one dataset will win on another. Recommend configurations based on worst-case robustness (minimum AUROC floor, zero inversions) rather than best-case performance on one dataset.

## 6. Prediction 3 refuted: input-side positions invert MORE, not less

**What we tested.** H28 predicted that only residual-stream positions should invert (AUROC < 0.5), because they perturb the persistent residual directly and risk stripping more clean signal than backdoor signal.

**Result.** Input-side positions have a 9.4% inversion rate. Residual-stream positions have a 4.1% inversion rate. The prediction is backward.

**Why.** Input-side positions at extreme perturbation rates can saturate the computation input, destroying all signal (both clean and backdoor). Residual-stream positions perturb a single additive component, which is less likely to saturate.

**Implication.** Inversions are driven by over-perturbation at high rates, not by the position's relationship to the residual stream. This is why sigma-matching (choosing a rate that produces the same shift ratio) is critical: it prevents the high-rate saturation that causes inversions.

## 7. Gaussian noise fails catastrophically on CIFAR-100

**What we tested.** gaussian @ before_attention_norm on CIFAR-100.

**Result.** BadNet 1%: AUROC 0.168 (massive inversion). BadNet 5%: 0.269. Even blend at 1%: 0.441. All below chance.

**Why.** Gaussian noise at the attention input disrupts the attention QKV computation. On a 100-class dataset with thin decision boundaries, the noise destroys clean predictions as much as poisoned ones. The sigma-matched rate is reached at a noise level that is already beyond the point where clean predictions are random, so the "shift ratio" measures noise-induced confusion rather than backdoor sensitivity.

**Implication.** Gaussian noise as a perturbation operator is only viable at positions AFTER the attention sublayer (before_mlp works, rank 2). At input-to-attention positions, it should not be used.

## Source

p* analysis: `defences/psbd_metrics.py:critical_rate()` and `scratch/critical_rate_analysis.py`.
Ranking transfer: `scratch/cross_dataset_transfer.py`.
H28 predictions: `scratch/h28_predictions.py`.
All detection numbers: `defence_tables.py` at sigma >= 0.6, fractional PSU, one-sided.
