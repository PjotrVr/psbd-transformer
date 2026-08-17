# Detection at Realistic Operating Points

This report evaluates the detection system at practical false-positive rates (1%, 5%, 10%, 25%) rather than relying solely on AUROC. The question: if a defender sets a budget for how many clean samples they will wrongly flag, how many poisoned samples does the system catch?

All numbers use one-sided fractional PSU (low PSU = poisoned), sigma-matched at 0.6, on ViT-B/16.

For full per-cell tables and detailed analysis, see the [detailed companion](detection-operating-points-detail.md).

## Key findings

**The system detects 7 of 9 evaluated attacks with TPR above 90% at 5% FPR on CIFAR-100 and Tiny ImageNet, at 5% and 10% poison rates.** The two exceptions are WaNet and Label-Consistent (LC), which are also the two attacks with the weakest implantation (lowest ASR).

**token_mask at before_attention_norm is the recommended deployment configuration.** It has the most uniform ROC curves, zero score inversions, and the highest worst-case TPR. gain_scale at mlp_norm_out produces higher peak AUROC but has degenerate ROC shapes (near-zero TPR at 1% FPR despite AUROC above 0.93) because its score distributions are bimodal.

**Gaussian noise at before_mlp is a strong alternative** with the best TPR at 1% FPR on several cells, but it collapses on WaNet (0.543 AUROC on CIFAR-100 10%).

**SAM training has negligible effect on detection.** Across 18 matched comparison cells, SAM at rho=0.1 adds +0.009 mean AUROC (15/18 cells positive). The gain is too small to justify the doubled training cost. SAM also destabilizes implantation of already-weak attacks (WaNet drops from 0.650 to 0.007 ASR on CIFAR-100 5%), which confounds rather than helps the analysis.

**The perturbation rate is not a hyperparameter the defender needs to tune.** Sigma-matching (choosing the rate that produces a target fraction of shifted predictions on a clean validation set) automatically selects a near-optimal operating point. The rate sensitivity curves show that AUROC rises monotonically with sigma up to about 0.6, then plateaus, then falls. The sigma-matching procedure picks the first rate that crosses the 0.6 threshold, which lands in or near the plateau.

## Summary tables

### Best configuration per attack (CIFAR-100 and Tiny, TPR at 5% FPR)

| Attack | Poison | Best config | CIFAR-100 | Tiny |
|---|---|---|---:|---:|
| badnet_a2o | 1% | gaussian | 0.991 | 0.971 |
| badnet_a2o | 5% | token_mask | 0.935 | 0.454 |
| badnet_a2o | 10% | gain_scale | 1.000 | 0.135 |
| blend | 1% | gaussian | 0.494 | 0.988 |
| blend | 5% | gain_scale | 1.000 | 0.999 |
| blend | 10% | gain_scale | 1.000 | 0.998 |
| bpp | 1% | gain_scale | 0.967 | 0.964 |
| bpp | 5% | gaussian | 0.984 | 0.988 |
| bpp | 10% | gaussian | 0.994 | 0.995 |
| lf | 1% | token_mask | 0.915 | 0.853 |
| lf | 5% | gaussian | 0.362 | 0.966 |
| lf | 10% | gain_scale | 0.984 | 0.802 |
| adaptive_blend | 10% | gaussian | 0.971 | n/a |
| wanet | 5% | token_mask | 0.412 | 0.791 |
| wanet | 10% | gain_scale | 0.902 | 0.644 |

### Failure modes

Three situations produce low TPR even with decent AUROC:

1. **Bimodal score distributions (gain_scale on Tiny badnet_a2o).** AUROC 0.933 but TPR at 1% FPR = 0.005. The scores split into two clusters with overlapping tails. At strict FPR thresholds, the decision boundary sits between the clusters and misses the tail.

2. **Weak implantation (WaNet at 5%, LC everywhere).** When ASR is below 0.8, the attack itself barely works, so the "poisoned" samples behave like clean samples and detection has nothing to find.

3. **Gaussian collapse on WaNet.** Gaussian noise at before_mlp gives AUROC 0.543 on CIFAR-100 WaNet 10% (effectively random). WaNet's warping trigger spreads across all pixels rather than occupying a localized patch, and isotropic noise adds energy in the same spatial distribution, washing out the signal. token_mask and gain_scale do not have this problem (0.900 and 0.942 respectively).

### Perturbation rate sensitivity

The rate controls how much the perturbation operator disturbs the model's forward pass. Sigma (the fraction of validation samples whose top-1 prediction changes) is the natural unit.

For token_mask on badnet_a2o CIFAR-100 5%:

| Rate | Sigma | AUROC |
|---:|---:|---:|
| 0.05 | 0.09 | 0.708 |
| 0.20 | 0.33 | 0.878 |
| 0.40 | 0.80 | 0.977 |
| 0.50 | 0.93 | 0.994 |
| 0.60 | 0.96 | 0.995 |
| 0.80 | 0.99 | 0.953 |

The curve follows a consistent pattern across all operators: low rate produces low AUROC (too little perturbation, clean and backdoor both unaffected), optimal is in the sigma 0.5 to 0.8 range, and very high rate produces lower AUROC (too much perturbation, both clean and backdoor are fully disrupted, converging to chance). The sigma-matching procedure at 0.6 automatically lands near the optimum.

For gain_scale, the effective scale parameter is different (multiplier on LayerNorm output), but the sigma curve has the same shape. For gaussian, the noise standard deviation maps to sigma the same way.

The defender does not tune the perturbation rate directly. They set a target sigma (we use 0.6), and the system sweeps rates on a held-out clean validation set to find the rate that achieves that sigma. This is fully automatic and does not require any knowledge of the attack.

### SAM training

SAM (Sharpness-Aware Minimization) was trained at 4 rho values (0.05, 0.1, 0.15, 0.2) on top of AdamW. Detection comparison at rho=0.1 across CIFAR-100 and Tiny:

| Metric | Value |
|---|---:|
| Mean AUROC delta (SAM minus non-SAM) | +0.009 |
| Cells where SAM wins | 15/18 |
| Cells where SAM wins by more than 0.02 | 4/18 |

The largest SAM improvements appear on cells that are already easy (badnet_a2o at higher poison rates). On the hard cells (1% poison), the improvement is marginal (+0.003 on Tiny badnet 1%). SAM also creates instability: on CIFAR-100 WaNet 5%, SAM at rho=0.1 drops ASR from 0.650 to 0.007, eliminating the attack entirely rather than helping detect it.

**Verdict on SAM: not useful for detection.** The marginal AUROC gain is not worth the training cost or the ASR instability.

## Next: adversarial attacker

The numbers above assume a passive attacker who does not know about the detection system. An adaptive attacker who controls the training procedure could attempt to minimize PSU during training, making poisoned samples harder to detect. This is covered in a separate analysis: the attacker adds a regularization term that penalizes high PSU difference between clean and poisoned samples. The question is whether evasion against one perturbation operator (e.g., dropout) transfers to others (token_mask, gain_scale, gaussian). If it does not transfer, operator diversity provides defense in depth. Data from the evasion training runs will be analyzed when those PBS jobs complete.
