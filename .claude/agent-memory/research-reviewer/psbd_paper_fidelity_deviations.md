---
name: psbd-paper-fidelity-deviations
description: Where PSBD-ViT deliberately or accidentally departs from the PSBD paper (papers/PSBD) - check these before comparing numbers to published tables
metadata:
  type: reference
---

The PSBD paper source lives at `papers/PSBD/sec/*.tex`; the method and its hyperparameters
are in `4_method.tex` (residual placement at line 107, PSU at Eq. `PSU definition`,
adaptive p selection and the 25th-percentile threshold near line 185-200).

Known deviations in this repo, as of 2026-08-13:

- Paper trains and evaluates with **no data normalization and no augmentation**; this repo
  normalizes with per-dataset mean/std in both paths.
- Paper trains 100 epochs and picks a late-stage model; this repo fixes 15 epochs for all
  runs (a deliberate comparability choice, see CLAUDE.md).
- Paper's clean validation set is 5% of the training-set size (2500 for CIFAR-10); this repo
  uses 2000.
- Paper scores the **suspicious training set** (poison filtering); this repo scores
  test-set-derived clean and triggered splits (input-level detection). TPR/FPR/AUROC are
  therefore not comparable to the paper's tables.
- Paper's backbone is ResNet-18 (8 residual adds); ViT-B/16 has 12 blocks and Swin-S has 24,
  so the paper's dropout rate p does not transfer numerically.

**How to apply:** any table that puts a PSBD-ViT number next to a published PSBD number needs
these spelled out in the caption, or the comparison is invalid. Related: [[psbd-dropout-position-sweep]].
