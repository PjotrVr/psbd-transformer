# Attack-Specific Analysis

Per-attack detection patterns across the top 3 configurations (token_mask @ before_attention_norm, gain_scale @ mlp_norm_out, dropout @ pre_residual) on CIFAR-100 and Tiny ImageNet.

## Summary: per-attack mean AUROC on primary datasets

### CIFAR-100

| Attack | token_mask@norm | gain_scale@mlp | dropout@pre_res | Cross-config mean |
|---|---:|---:|---:|---:|
| badnet_a2o | 0.975 | 0.999 | 0.830 | 0.935 |
| blend | 0.972 | 0.992 | 0.861 | 0.942 |
| wanet | 0.826* | 0.879* | 0.678* | 0.794 |
| lc | 0.726* | 0.824* | 0.543* | 0.698 |
| adaptive_blend | 0.861 | 0.922 | 0.805 | 0.863 |

### Tiny ImageNet

| Attack | token_mask@norm | gain_scale@mlp | dropout@pre_res | Cross-config mean |
|---|---:|---:|---:|---:|
| badnet_a2o | 0.945 | 0.949 | 0.915 | 0.936 |
| blend | 0.991 | 0.979 | 0.968 | 0.979 |
| wanet | 0.945** | 0.896** | 0.847** | 0.896 |
| lc | 0.753* | 0.680* | 0.612* | 0.682 |
| adaptive_blend | 0.939** | 0.922** | 0.925** | 0.929 |

(*) Includes weak ASR cells (ASR < 0.8). (**) WaNet and adaptive_blend do not implant at 1% on Tiny, so averages are across 5% and 10% only.

## Per-attack deep dives

### BadNet A2O (all-to-one, static patch trigger)

Detection difficulty: medium at 1%, easy at 5-10%.

BadNet places a small 3x3 patch in the corner of the image. At high poison rates, this creates a strong, localized backdoor direction that is easy to detect with any perturbation method. At 1%, the direction is weaker because fewer training samples carry the trigger, making it the second-hardest 1% detection target (after LC on CIFAR-100).

Key patterns:
- **gain_scale dominates on CIFAR-100** (0.999 across all rates) because LayerNorm amplification is most effective when the backdoor direction is spatially concentrated. The norm statistics shift proportionally to the trigger's feature-space footprint.
- **TPR@5% weakness on Tiny.** gain_scale achieves 0.933 AUROC but only 0.034 TPR@5%FPR at 1% on Tiny. The ROC curve is steep at high FPR but flat at low FPR, meaning the method cannot reliably flag individual samples. token_mask has 0.343 TPR@5% on the same cell (10x better).
- **dropout @ pre_residual is strong on Tiny** (0.933 at 1%, 0.667 TPR@5%) because the pre-residual placement directly perturbs the channel through which the trigger's representation flows.

### Blend (distributed pattern trigger)

Detection difficulty: easy. Blend is the most detectable attack across all configurations and datasets.

Blend overlays a fixed pattern (a Hello Kitty image) with a mixing ratio alpha on the input. This creates a highly consistent backdoor direction because every poisoned sample receives the same additive perturbation in pixel space.

Key patterns:
- **Consistently high AUROC.** CIFAR-100: 0.945 to 0.992 at 1%. Tiny: 0.950 to 0.991. GTSRB: 0.997 to 1.000. Even CIFAR-10: 0.983 to 0.996.
- **token_mask is particularly strong** because even partial token masking disrupts the blended pattern. Blend affects every patch uniformly, so masking any subset of patches removes a proportional amount of the trigger.
- **No shift-to-target effect (H7).** The PSBD paper observed that poisoned samples shift TOWARD the target class under perturbation, interpreting this as the perturbation "removing the backdoor." On ViT with Blend, this does not happen: the shift is AWAY from the original class without preferentially landing on the target. Detection works anyway because PSU does not require shift-to-target, only shift-from-predicted.

### WaNet (warping trigger)

Detection difficulty: hard. WaNet is consistently the hardest attack to detect after LC.

WaNet warps the entire image with a learned deformation field. Unlike BadNet or Blend, the trigger is not an additive patch or pattern but a global spatial transformation. This means the trigger is distributed across all patches and all frequencies.

Key patterns:
- **Does not implant at 1%.** On CIFAR-100 (ASR 0.044), Tiny (ASR 0.379), CIFAR-10 (ASR 0.123), and GTSRB (ASR 0.032), WaNet fails to implant a functional backdoor at 1% poison rate. The warping transformation is too subtle to override the clean features at this rate.
- **token_mask is the best operator.** Token masking at before_attention_norm achieves 0.900 AUROC on CIFAR-100 at 10% and 0.934/0.955 on Tiny at 5%/10%. Token masking disrupts the spatial coherence of the warping, which is its distinguishing feature.
- **gain_scale struggles.** On GTSRB, gain_scale inverts on WaNet: 0.459 at 5%, 0.479 at 10%. The LayerNorm amplification mechanism does not distinguish WaNet's distributed deformation from clean image variation on GTSRB, where the images already have high spatial variability (traffic signs at various angles).
- **dropout @ pre_residual is decent** (0.684 to 0.934 at 10% depending on dataset) because it disrupts the residual stream's representation of the warped spatial structure.

### LC (label-consistent, clean-label attack)

Detection difficulty: very hard. LC is the most difficult attack to detect.

LC uses SIG (sinusoidal strip) as the trigger pattern but only poisons samples whose original class matches the target class. The training labels are correct, so the model must learn the trigger association without explicit label supervision. This produces a weak backdoor direction because the trigger signal competes with the natural class features rather than overriding them.

Key patterns:
- **Low ASR.** LC achieves only 0.874 ASR on CIFAR-100 at 1%, 0.786 at 10%. On GTSRB, it fails entirely (ASR 0.130 to 0.467). On Tiny, only 0.626 at 10%.
- **AUROC/TPR disconnect.** On CIFAR-100 with token_mask: 0.786 AUROC but 0.035 TPR@5%FPR at 1%. The separation exists but is concentrated at high FPR. gain_scale resolves this partially: 0.935 AUROC and 0.876 TPR@5% on the same cell.
- **Scales poorly with poison rate on CIFAR-100.** From 1% to 10%, LC's AUROC with token_mask goes from 0.786 to 0.760 (DECREASES). This is unusual: all other attacks become easier to detect at higher poison rates. The reason is that LC's clean-label mechanism means higher poison rate does not create a proportionally stronger backdoor direction.
- **gain_scale is the only configuration that reliably detects LC.** On CIFAR-100 at 1%: 0.935 AUROC, 0.876 TPR@5%. The LayerNorm amplification reveals the weak sinusoidal signal that token_mask and dropout miss.

### Adaptive Blend (stealth-optimized variant)

Detection difficulty: medium. Easier than WaNet and LC, harder than Blend.

Adaptive Blend is Blend with a reduced mixing ratio and optional cover samples. The trigger is the same pattern but weaker, designed to evade detection.

Key patterns:
- **Tracks Blend's pattern at lower AUROC.** CIFAR-100 token_mask: 0.706 at 1%, 0.927 at 5%, 0.951 at 10%. Compare Blend: 0.945, 0.989, 0.982. The stealth optimization reduces AUROC by ~0.03 to 0.24 depending on the rate.
- **Does not implant at 1% on Tiny and GTSRB.** ASR 0.431 and 0.447 respectively. The reduced trigger strength is insufficient on larger/harder datasets at low poison rate.
- **gain_scale compensates well.** On CIFAR-100 at 1%: 0.854 AUROC with gain_scale vs 0.706 with token_mask. The LayerNorm amplification recovers ~0.15 AUROC for this weak trigger.
- **At 10%, most configurations detect reliably.** CIFAR-100: 0.951 (token_mask), 0.953 (gain_scale). Tiny: 0.958 (token_mask), 0.933 (gain_scale). The stealth optimization is not enough to evade at high poison rate.

## Configuration recommendation by attack type

| If the threat model emphasizes... | Use |
|---|---|
| Static patch triggers (BadNet-like) | token_mask @ before_attention_norm |
| Distributed triggers (Blend, Adaptive Blend) | either token_mask or gain_scale |
| Clean-label attacks (LC) | gain_scale @ mlp_norm_out |
| Spatial transformation (WaNet) | token_mask @ before_attention_norm |
| Unknown attack type | token_mask @ before_attention_norm (safest floor) |

## Source

All AUROC/TPR values from `defence_tables.py --operator {op} --position {pos} --allow-partial`, fractional PSU, sigma >= 0.6, one-sided.
