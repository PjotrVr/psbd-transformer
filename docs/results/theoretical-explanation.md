# Theoretical explanation: why the results make sense

## Core claim (H28: perturbation consistency is margin estimation)

All perturbation-consistency backdoor detectors (PSBD, IBD-PSC, SCALE-UP, STRIP) are fundamentally one method: they measure the decision margin by applying perturbations and observing prediction stability. The perturbation operator only sets the Jacobian of the perturbation-to-logit mapping. The key equation is:

$$\text{PSU} = P_{\text{clean}}(y \mid x) - \mathbb{E}_{\text{perturbation}}[P(y \mid x_{\text{perturbed}})]$$

This is a first-order approximation to the decision margin: how much logit mass the predicted class loses under perturbation. A poisoned sample has a narrow margin (the backdoor feature is a single direction in representation space), while clean samples have broad margins (multiple redundant features vote for the class).

## Why these results make sense

### 1. Why gaussian noise works as well as structured masks (H23 result)

Gaussian noise at `before_mlp` was rank 2 at 1% (AUROC 0.928). The neuron-bias mechanism proposed by the original PSBD paper (dropout removes over-activated backdoor neurons) is NOT what drives detection on ViT. If it were, structured removal (dropout) should beat unstructured noise. Instead, isotropic noise works because the backdoor direction is non-axis-aligned (H16 finding), so axis-aligned neuron removal has no advantage. Any perturbation that pushes representation off the narrow backdoor ridge disrupts the poisoned prediction, regardless of the perturbation's structure.

However, gaussian noise FAILS on CIFAR-100 specifically (BadNet 0.168 AUROC, a massive inversion). This is because gaussian noise at the input layer is absorbed by the attention mechanism on hard datasets: when there are 100 classes and the decision boundary is thin, the added noise disrupts clean predictions as much as poisoned ones, collapsing discriminability. The gaussian result is position-dependent: it works at `before_mlp` (rank 2) but fails at `before_attention_norm` where the noise hits the attention computation first.

### 2. Why position matters more than operator (H28 prediction 4)

Supported: at matched shift ratio ($\sigma$), position variance is 1.43x family variance, Kendall $\tau = 0.700$ for attack ranking agreement across operators at fixed position. The Jacobian from the perturbation site to the readout layer determines the effective perturbation strength. Different positions mean different Jacobians (how the perturbation propagates through the remaining network). Different operators at the same position mean the same Jacobian scaled differently (they all push representation in different directions at the same point in the network, but the subsequent propagation is the same).

### 3. Why input-side beats residual-adjacent (H20 result)

Input-side positions (`before_attention_norm`, `before_mlp`) perturb the COMPUTATION INPUT: the data that enters a sublayer. This is a broad perturbation that costs the redundant clean evidence more than the single backdoor direction. Residual-adjacent positions (`before_attention_residual`, `before_mlp_residual`) perturb the RESIDUAL STREAM directly. This hits the backdoor direction head-on but also strips the clean information that flows through the residual. The result: input-side positions have higher mean AUROC (+0.054, bootstrap CI [+0.031, +0.080]) because they are less likely to invert (strip too much clean signal relative to backdoor signal).

### 4. Why the ranking does NOT transfer across datasets

Spearman $\rho$ between dataset rankings is only 0.33 to 0.70 (moderate). The Jacobian depends on the learned weights, which vary with dataset complexity. A 100-class dataset (CIFAR-100) learns very different attention patterns from a 10-class one (CIFAR-10). The optimal perturbation site shifts because the gradient landscape shifts.

### 5. Why token masking is the best operator at input-side positions

Token masking removes entire spatial patches. Backdoor triggers are spatially localized (BadNet: a 3x3 patch, Blend: a pattern overlaid on the whole image but concentrated in specific frequencies). Dropping a triggered patch removes the trigger entirely for that pass, creating a large PSU. Dropout removes random features across ALL patches, diluting the trigger signal without fully removing it from any one patch. At input-side positions where the spatial structure is preserved, this spatial specificity of token masking gives it an advantage.

### 6. Why gain_scale (IBD-PSC port) dominates on CIFAR-100

`gain_scale` multiplies a LayerNorm's whole output by a fixed factor $\omega = 1 + \text{rate}$, uniformly across channels, with nothing sampled and no bias term of its own. Scaling $\gamma$ and $\beta$ together is algebraically identical to scaling the layer's output ($\omega\gamma \hat{x} + \omega\beta = \omega(\gamma \hat{x} + \beta)$), which is why IBD-PSC's parameter amplification is reachable from a post-hook without touching a weight. Being deterministic, it is the one operator in the study whose PSU is exact at $k = 1$: every Monte Carlo pass returns the same value, and its shift ratio is a per-sample flip indicator rather than a fraction of passes. It is also the only operator that amplifies rather than removes. The winning position, `mlp_norm_out`, is block-scope, so the gain is applied once per block across all 12 blocks rather than once at model level. On CIFAR-100 this dominates every other operator (0.999 AUROC on BadNet at 1%). However, `gain_scale`'s TPR@5%FPR on Tiny BadNet at 1% is only 0.034 (near-random at the practical threshold), meaning the amplification moves clean predictions about as readily as poisoned ones on a 200-class dataset. The ROC curve is steep at high FPR but flat at low FPR.

### 7. Why combining both sublayer inputs dilutes rather than compounds

The combined position (`before_attention_norm` + `before_mlp_norm`) perturbs both sublayer inputs. At matched $\sigma$, each site receives weaker perturbation. The `before_mlp_norm` site (rank 15/27) adds mostly noise: by the time tokens reach the MLP input, attention has already mixed the backdoor signal across all tokens, making local masking less effective. The combination's mean delta on Tiny is -0.045 (every cell worse than single position).

### 8. Why the backdoor direction is non-axis-aligned (H16)

The backdoor direction in ViT's representation space is spread across many neurons rather than concentrating on a few. This is because ViT's residual stream is persistent (CKA homogeneity result from Raghu et al.): information added at layer $k$ persists through layers $k+1$ to 12 without much transformation. A trigger pattern that enters at the embedding layer gets its direction preserved but not concentrated. This explains why head-masking (H22, rank 13) and neuron profiling (H18) fail: they target axis-aligned structures (individual heads, individual neurons) while the backdoor direction is a diagonal in that space.

## Predictions from this account

1. Direction norm should scale with poison rate (untested, H28 prediction 1). More poisoned samples in training produce a stronger backdoor direction and a larger separation.
2. Operator diversity provides defense in depth against adaptive attackers: an attacker that evades one operator cannot simultaneously evade all operators because each sets a different Jacobian.
3. Increasing Monte Carlo passes ($k$) should help most on the hardest cells (low poison rate, weak attack), where the margin is tightest and variance in PSU estimation matters most (H24 partial support: +0.028 on CIFAR-10).

## What this account does NOT explain

- Why LC is consistently the hardest to detect. The margin account predicts that clean-label attacks should have thinner margins (the trigger is weaker because training labels are correct), but does not predict the specific TPR@5% collapse (0.035 on CIFAR-100) versus decent AUROC (0.786).
- Why WaNet inverts on some operator/position combinations but not others. The warping trigger is distributed across the entire image, so the margin account does not distinguish it from Blend (also distributed), yet WaNet is much harder to detect.
