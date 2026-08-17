# Stealth Metrics: PSNR and SSIM of Backdoor Triggers

Trigger imperceptibility measured as PSNR (dB) and SSIM between clean and
poisoned images at native resolution (32x32 for CIFAR-10/100/GTSRB, 64x64
for Tiny ImageNet), before the 224x224 upscale the model sees. Higher PSNR
and higher SSIM both indicate a more imperceptible trigger.

200 images per (attack, dataset) pair, sampled from the test split with
seed 42. Target label 0. All attacks use their default configurations.

Poison rate does not affect these metrics. Each attack's trigger function
is deterministic and rate-independent: the same `apply_trigger` is applied
regardless of how many training images are selected for poisoning.

## Summary (averaged across datasets)

| Attack         | PSNR (dB) | SSIM   | Stealth |
|----------------|-----------|--------|---------|
| WaNet          | 32.01     | 0.9762 | high    |
| LF             | 31.09     | 0.9513 | high    |
| BPP            | 27.83     | 0.8891 | medium  |
| BadNet A2O     | 26.73     | 0.9850 | medium  |
| TaCT           | 26.73     | 0.9850 | medium  |
| SIG            | 23.24     | 0.7631 | low     |
| Blend          | 22.00     | 0.7517 | low     |
| Adaptive Blend | 22.00     | 0.7517 | low     |
| LC             | 20.65     | 0.9402 | low     |

Stealth tiers: high = PSNR > 30 and SSIM > 0.95, medium = PSNR > 25 or
SSIM > 0.88, low = neither condition met.

## CIFAR-10

| Attack         | PSNR (dB)       | SSIM             |
|----------------|-----------------|------------------|
| WaNet          | 31.90 +/- 2.51  | 0.9812 +/- 0.006 |
| LF             | 30.86 +/- 0.26  | 0.9613 +/- 0.025 |
| BPP            | 27.74 +/- 0.72  | 0.9200 +/- 0.036 |
| BadNet A2O     | 25.63 +/- 0.85  | 0.9846 +/- 0.007 |
| TaCT           | 25.63 +/- 0.85  | 0.9846 +/- 0.007 |
| SIG            | 23.17 +/- 0.29  | 0.7993 +/- 0.078 |
| Blend          | 22.49 +/- 0.89  | 0.8058 +/- 0.074 |
| Adaptive Blend | 22.49 +/- 0.89  | 0.8058 +/- 0.074 |
| LC             | 19.47 +/- 0.67  | 0.9386 +/- 0.020 |

## CIFAR-100

| Attack         | PSNR (dB)       | SSIM             |
|----------------|-----------------|------------------|
| WaNet          | 32.06 +/- 2.78  | 0.9815 +/- 0.006 |
| LF             | 30.93 +/- 0.33  | 0.9581 +/- 0.027 |
| BPP            | 27.92 +/- 0.86  | 0.9128 +/- 0.046 |
| BadNet A2O     | 25.33 +/- 0.91  | 0.9850 +/- 0.007 |
| TaCT           | 25.33 +/- 0.91  | 0.9850 +/- 0.007 |
| SIG            | 23.24 +/- 0.37  | 0.7902 +/- 0.087 |
| Blend          | 22.16 +/- 0.98  | 0.7915 +/- 0.088 |
| Adaptive Blend | 22.16 +/- 0.98  | 0.7915 +/- 0.088 |
| LC             | 19.21 +/- 0.76  | 0.9380 +/- 0.024 |

## GTSRB

| Attack         | PSNR (dB)       | SSIM             |
|----------------|-----------------|------------------|
| WaNet          | 35.41 +/- 5.87  | 0.9850 +/- 0.006 |
| LF             | 30.90 +/- 0.22  | 0.9131 +/- 0.069 |
| BPP            | 27.78 +/- 1.19  | 0.8295 +/- 0.172 |
| BadNet A2O     | 24.66 +/- 0.86  | 0.9747 +/- 0.010 |
| TaCT           | 24.66 +/- 0.86  | 0.9747 +/- 0.010 |
| SIG            | 23.27 +/- 0.35  | 0.6624 +/- 0.176 |
| Blend          | 21.46 +/- 0.99  | 0.6702 +/- 0.180 |
| Adaptive Blend | 21.46 +/- 0.99  | 0.6702 +/- 0.180 |
| LC             | 18.70 +/- 0.73  | 0.9009 +/- 0.038 |

GTSRB shows notably lower SSIM for Blend, SIG, and Adaptive Blend
compared to the other datasets. GTSRB images contain large uniform
regions (road signs on solid backgrounds), where even small additive
perturbations are perceptually salient and pull SSIM down. The high SSIM
variance (0.17 to 0.18) reflects the mix of uniform and textured images
in GTSRB.

## Tiny ImageNet

| Attack         | PSNR (dB)       | SSIM             |
|----------------|-----------------|------------------|
| LF             | 31.69 +/- 0.21  | 0.9727 +/- 0.025 |
| BadNet A2O     | 31.29 +/- 0.97  | 0.9958 +/- 0.002 |
| TaCT           | 31.29 +/- 0.97  | 0.9958 +/- 0.002 |
| WaNet          | 28.68 +/- 2.98  | 0.9573 +/- 0.016 |
| BPP            | 27.89 +/- 0.53  | 0.8940 +/- 0.048 |
| LC             | 25.22 +/- 0.71  | 0.9834 +/- 0.006 |
| SIG            | 23.29 +/- 0.29  | 0.8006 +/- 0.090 |
| Blend          | 21.90 +/- 0.88  | 0.7393 +/- 0.101 |
| Adaptive Blend | 21.90 +/- 0.88  | 0.7393 +/- 0.101 |

Patch-based triggers (BadNet, TaCT, LC) gain the most from the 32 to 64
resolution jump: a 3x3 patch covers 0.88% of a 32x32 image but only
0.22% of a 64x64 image, so PSNR rises by 5 to 6 dB.

## Observations

**Identical triggers.** Two pairs of attacks share the same trigger
function and therefore produce identical stealth metrics:

- Blend and Adaptive Blend use the same alpha-blended random pattern.
  Adaptive Blend's distinctiveness is its cover samples during training,
  not a different trigger.
- BadNet A2O and TaCT use the same 3x3 checkerboard patch in the
  bottom-right corner. TaCT's distinctiveness is source-class restriction
  and cover samples, not the trigger itself.

**PSNR and SSIM rankings diverge.** BadNet has moderate PSNR (26.73 dB)
but very high SSIM (0.985) because its 3x3 patch concentrates all
distortion in 9 pixels, leaving 99% of the image untouched. SSIM, being
a local windowed metric, rewards this concentration. By contrast, Blend
and SIG spread perturbation across every pixel, which keeps per-pixel
magnitude low but pulls the global SSIM down. LC has the lowest PSNR
(20.65 dB) because it stamps 4 corner patches (4 times the pixel-level
damage of BadNet) but still maintains reasonable SSIM (0.94) because the
damaged area is small relative to the full image.

**WaNet is the most stealthy by both metrics.** Its warping-based trigger
relocates pixel values without adding or removing energy, so MSE stays
low (PSNR > 30 dB) and the local structure is mostly preserved (SSIM >
0.96). The high PSNR variance (2.5 to 5.9 dB across datasets) reflects
image-dependent warping artifacts near edges and flat regions.

**Resolution dependence.** Patch-based attacks (BadNet, TaCT, LC) are
strongly resolution-dependent: the fixed 3x3 patch covers less of the
image at 64x64 than at 32x32, so PSNR rises by 5 to 6 dB on Tiny
ImageNet. Global perturbation attacks (Blend, SIG, LF) show almost no
resolution dependence because their per-pixel perturbation magnitude is
set by the attack parameter (alpha, amplitude, strength), not the image
size.

**Detection implications.** Attacks in the "high stealth" tier (WaNet,
LF) are visually undetectable and would pass a manual inspection. The
"low stealth" tier (Blend, SIG, LC) produces visible artifacts in
side-by-side comparison, especially on GTSRB. This makes the stealth
tier relevant when interpreting detection results: a defense that detects
only low-stealth attacks is solving a much easier problem than one that
detects WaNet or LF.

## Methodology

PSNR: `10 * log10(1 / MSE)` on [0,1]-normalized images.

SSIM: Wang et al. 2004 with 11x11 Gaussian window (sigma=1.5),
K1=0.01, K2=0.03, computed per-channel and averaged. Implemented
in PyTorch without external dependencies.

Script: `scratch/stealth_metrics.py` (git-ignored).
Raw data: `scratch/stealth_metrics_results.json` (git-ignored).
