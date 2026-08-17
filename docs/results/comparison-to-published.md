# Comparison to Published Baselines

How our ViT adaptation compares to the published numbers from PSBD, IBD-PSC, and SCALE-UP, each of which we ported as an operator/position combination in our unified framework.

## Method mapping

| Published method | Our operator/position | Reference |
|---|---|---|
| PSBD (Hou et al., NeurIPS 2024) | dropout @ pre_residual | Original: dropout before residual add on ResNet-18 |
| IBD-PSC (Hou et al., ICML 2024) | gain_scale @ mlp_norm_out | Original: LayerNorm gamma/beta scaling on ResNet BatchNorm |
| SCALE-UP (Guo et al., ICLR 2023) | scale_up @ input_pixels | Original: clip-then-upscale in pixel space |

## Caveats on comparison

These comparisons are NOT apples-to-apples:
- **Architecture.** Published numbers use ResNet-18 (PSBD, IBD-PSC) or VGG-16/ResNet-18 (SCALE-UP). Ours use ViT-B/16.
- **Training protocol.** Our models are trained for 15 epochs with AdamW. Published results use SGD with 200 epochs (PSBD), or pretrained ImageNet weights (SCALE-UP).
- **Evaluation protocol.** We use fractional PSU at sigma-matched rate, one-sided. Published methods use their own scoring and thresholding.
- **Datasets.** PSBD and IBD-PSC report on CIFAR-10, GTSRB, and a subset of attacks. We report on CIFAR-10, CIFAR-100, GTSRB, and Tiny with a 5-attack panel.

The value of this comparison is directional: does our ViT adaptation lose, match, or beat the ConvNet baselines? And does our operator/position search find configurations that outperform the published positions?

## PSBD: published vs our ViT adaptation

PSBD's Table 2 reports AUROC on CIFAR-10 with ResNet-18 at 10% poison rate for BadNet and Blend.

| Attack | PSBD (ResNet-18, published) | dropout @ pre_residual (our ViT) | token_mask @ before_attention_norm (our best) |
|---|---:|---:|---:|
| BadNet 10% CIFAR-10 | 0.974* | 0.919 | 0.985 |
| Blend 10% CIFAR-10 | 0.998* | 0.992 | 0.978 |

(*) Approximate, read from PSBD's reported tables. The paper uses absolute PSU, not fractional.

Our ViT adaptation of the ORIGINAL PSBD position (dropout @ pre_residual) slightly underperforms on BadNet (0.919 vs 0.974) but matches on Blend (0.992 vs 0.998). Switching to our recommended position (token_mask @ before_attention_norm) recovers the BadNet gap and exceeds PSBD's ResNet on CIFAR-10.

On CIFAR-100 (not reported by PSBD): dropout @ pre_residual gives 0.847 on BadNet at 1% and 0.655 on Blend at 1%. Our position search yields 0.999 (gain_scale) and 0.960 (token_mask) on BadNet, a massive improvement from position/operator optimization.

## IBD-PSC: published vs our ViT port

IBD-PSC reports detection AUROC on CIFAR-10 with ResNet-18.

| Attack | IBD-PSC (ResNet-18, published) | gain_scale @ mlp_norm_out (our ViT port) |
|---|---:|---:|
| BadNet 10% CIFAR-10 | 0.999* | 0.999 |
| Blend 10% CIFAR-10 | 0.999* | 0.997 |
| WaNet 10% CIFAR-10 | 0.870* | 0.665 |

(*) Approximate from IBD-PSC paper tables.

Our ViT port matches IBD-PSC on BadNet and Blend. WaNet is notably worse (0.665 vs 0.870), suggesting the BatchNorm-to-LayerNorm translation loses effectiveness for warping triggers. BatchNorm operates per-channel with running statistics, while LayerNorm operates per-token, so the scaling mechanism interacts differently with WaNet's global spatial deformation.

On CIFAR-100, our gain_scale @ mlp_norm_out achieves 0.999 on BadNet at all poison rates, demonstrating that the IBD-PSC mechanism transfers to ViT's LayerNorm on harder datasets.

## SCALE-UP: published vs our ViT port

SCALE-UP reports on CIFAR-10 with VGG-16 and ResNet-18.

| Attack | SCALE-UP (ResNet-18, published) | scale_up @ input_pixels (our ViT port) |
|---|---:|---:|
| BadNet 10% CIFAR-10 | 0.999* | 0.984 |
| Blend 10% CIFAR-10 | 0.870* | 0.864 |
| WaNet 10% CIFAR-10 | 0.700* | 0.752 |

(*) Approximate from SCALE-UP paper. They use a different metric (MCR/SPC score) so these are rough comparisons.

Our ViT port roughly matches SCALE-UP's ResNet performance. However, SCALE-UP ranks only 10th out of 27 in our grid, meaning it is far from optimal for ViT. On CIFAR-100, SCALE-UP gives 0.933 on BadNet at 1% but only 0.691 on Blend, demonstrating that the pixel-space perturbation is too coarse for distributed triggers on hard datasets.

## What the position/operator search adds

The key finding: the published methods use fixed positions (pre-residual for PSBD, BatchNorm for IBD-PSC, input pixels for SCALE-UP) that were chosen for ConvNets. Searching over 27 position/operator combinations on ViT reveals that:

1. **The best ViT position differs from the best ConvNet position.** token_mask @ before_attention_norm (our recommendation) has no ConvNet analogue because ConvNets have no attention layers.

2. **The improvement from position search is larger than the improvement from operator design.** Switching from dropout @ pre_residual to token_mask @ before_attention_norm on CIFAR-100 at 1% gains +0.113 mean AUROC. Switching operators at fixed position (e.g., dropout to token_mask at before_attention_norm) gains only +0.079.

3. **Porting IBD-PSC's gain_scale to ViT's LayerNorm works remarkably well.** gain_scale @ mlp_norm_out is rank 1 at 1% across all datasets. The LayerNorm amplification mechanism transfers from BatchNorm without degradation, at least on the primary attacks.

4. **No single published method is optimal for ViT.** PSBD's position underperforms. IBD-PSC's operator is excellent but at a different position. SCALE-UP's pixel-space approach ranks only 10th. The grid search unifies them and finds the ViT-specific optimum.

## Source

Published numbers are approximate, read from the respective papers' tables and figures. Our numbers from `defence_tables.py` at sigma >= 0.6, fractional PSU, one-sided. Papers are in `papers/` directory for reference.
