---
name: reference-detector-baselines
description: Statistic type and ViT-portability blockers for the four detector baselines (STRIP, SCALE-UP, IBD-PSC, TeCo) whose PDFs sit in papers/reference/backdoor-detectors/
metadata:
  type: reference
---

The four baseline detector PDFs live in `papers/reference/backdoor-detectors/`, PSBD itself in `papers/reference/prior-art/`. What each statistic consumes, which is what decides whether it ports to ViT at all:

- STRIP (arXiv 1902.06531v2, ACSAC 2019): Shannon entropy, log base 2, over the FULL softmax vector of superimposed images. Needs soft outputs and a held-out clean image pool to superimpose.
- SCALE-UP (ICLR 2023): hard label only (argmax agreement under pixel scaling). Architecture agnostic.
- IBD-PSC (ICML 2024): scalar softmax confidence on the no-amplification argmax class, read off models whose **BatchNorm** affine parameters gamma and beta are multiplied by omega. **Hard blocker for ViT/Swin: the operator is defined only on BN, Theorem 3.1 argues over batch statistics, and every evaluated architecture (ResNet18, PreActResNet18, MobileNet) is a BN net.** A LayerNorm analogue exists mechanically (LN also has elementwise affine weight and bias) but is nowhere in the paper or appendix, so any LN version is a new method, not a reproduction.
- TeCo (CVPR 2023): hard label only. See [[reference-teco-baseline]] for its severity-flip statistic and the compounding-corruption code bug.

**Why:** This project is ViT-only, so "which baselines are even runnable" is decided by the statistic and the normalization layer, not by effort.

**How to apply:** When asked to add a detector baseline on ViT-B/16 or Swin-S, STRIP, SCALE-UP and TeCo are direct ports. IBD-PSC is not, and saying so up front is more useful than a partial implementation. Related: [[feedback-no-corner-cutting]].
