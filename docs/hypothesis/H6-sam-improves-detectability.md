# H6 — SAM training makes backdoors more detectable

**Status, 2026-09-11: PARTIALLY SUPPORTED for the recommended placement, at
10 percent poisoning, and unsettled below it.** SAM does amplify the
trigger's footprint on the class token, and the recommended token mask
placement (PSBD-TM) gains from it, but the gain is close to 0 and sign
unstable at 1 percent poisoning and only occasionally significant at 5
percent. It becomes consistent only at 10 percent. The published residual
dropout placement (PSBD-RD) loses under SAM at every rate this project has
measured. Full numbers and sources in
[`docs/sam-findings-2026-09-11.md`](../sam-findings-2026-09-11.md), which is
now the single current statement on SAM and supersedes every verdict below
this line, including the +0.009 aggregate the 2026-09-07 drop rested on.

## Why this is a gap and not a reproduction

`papers/reliable_poisoned_sample_detection_.../` claims SAM training amplifies the
backdoor and makes poisoned samples more detectable. Read closely, the claim is
narrower than it sounds:

- It is validated on **five feature-space detectors**: Activation Clustering, Beatrix,
  SCAn, Spectral Signatures, Spectre. All of them cluster penultimate-layer
  representations.
- **PSBD is not among them, and is not cited anywhere in the paper.** Nor is anything
  else based on prediction shift or dropout uncertainty.
- The only **prediction-space** detector they ever measured is STRIP, and it appears
  in three table files that are **commented out** of the submitted paper
  (`tables/cifar0.001.tex`, `0.005`, `0.01`, disabled at `main.tex:100-102`). STRIP's
  mean TPR change under SAM in those files: **-3.4, -3.4, +1.4**. It is the sole
  detector SAM does not help, and it was cut.
- ResNet18 only. No transformer anywhere in the paper. Base optimizer implied SGD,
  and their derivation is a first-order SGD expansion that does not carry to AdamW.

So the open question is not "does SAM help detection" but **"does SAM's benefit
transfer from feature-space detectors to prediction-space ones?"** The current answer,
from `docs/sam-findings-2026-09-11.md`, is that it partly does for PSBD-TM at 10
percent poisoning only. It does not for PSBD-RD at any rate measured.

## Mechanism, current reading

The SAM paper's own stage 2 exists to undo something SAM does: "to address the
increased intra-class variance of clean samples after SAM optimization" (Sec. 3.4).
More clean-feature variance raises clean PSU and narrows the gap PSU thresholds on.
PSBD has no equivalent cancelling stage. That mechanism explains why PSBD-RD loses
under SAM at every rate measured, and its interval excludes 0 in the losing direction
at rho 0.15 and 0.2 on the pooled grid.

It does not explain the PSBD-TM gain by itself, because PSBD-TM reads a label
decision rather than a feature vector. `experiments/sam_mechanism/README.md` measured
the class-token footprint directly instead: SAM raises the peak relative backdoor
direction norm and the final-layer TAC in 5 of 6 matched pairs, the same direction the
SAM paper's own metrics report on ResNet backdoor neurons, and where the footprint
grows the margin gap between triggered and clean inputs widens too, with PSBD-TM's
AUROC moving the same way. The 2 movements are coupled loosely rather than
proportionally. On WaNet the footprint shrinks instead, with PSBD-TM's AUROC
falling from 0.945 to 0.704, so amplification is attack dependent rather than a fixed
property of SAM training.

## Evidence, current sample sizes

`experiments/sam_reading/README.md`, 24 matched checkpoint pairs per rho, ViT only,
CIFAR-10 and CIFAR-100 only: PSBD-TM gains +0.010 to +0.035 mean AUROC across rho,
tightest interval at rho 0.1, [+0.002, +0.079]. PSBD-RD loses, intervals excluding 0
in the losing direction at rho 0.15 and 0.2.

`experiments/sam_low_rate/README.md` re-slices the same grid by poison rate. At 1
percent (7 pairs) the PSBD-TM delta is indistinguishable from 0 and flips sign across
rho. At 5 percent (7 to 8 pairs) it is positive at 3 of 4 rhos with only 1 interval
excluding 0. At 10 percent (8 to 10 pairs) the gain is consistent, +0.065 to +0.074,
3 of 4 intervals excluding 0. Per attack, the gain concentrates in BPP and WaNet, the
diffuse triggers, while it sits near 0 for BadNets, Blend and LF, the firm ones.

`experiments/sam_training_set_detection/README.md` tests the SAM paper's own claim,
a training-set poison filter, on its own terms: 6 matched pairs, CIFAR-100,
`badnet_a2o`, `blend` and `wanet` at 1 and 5 percent poisoning. Spectral Signature TPR
moves by at most 2.0 points against the paper's own SAM-only ablation gain of 15.2 to
57.7 points on ResNet18 CIFAR-10. Silhouette and the top to second singular value
ratio both move the wrong sign for the paper's own separability claim.

## Known divergences from their setup, to state in any writeup

- We do not implement their **feature-scaling stage**, so we are running their "SAM
  without FS" ablation row. That row is much weaker in their own table (79.8 against
  99.8 TPR for Blended/Beatrix). PSBD cannot use their fix, which is the point, but it
  must be said plainly.
- They filter a poisoned **training set** before training; we score held-out inputs
  against an already-trained model.
- SAM-on-AdamW here against SAM-on-SGD there.

## Reproduce

```bash
PYTHONPATH=. python experiments/sam_reading/measure.py
PYTHONPATH=. python experiments/sam_low_rate/measure.py
PYTHONPATH=. python experiments/sam_mechanism/measure.py --checkpoints-dir checkpoints --raw-data-dir raw_data
PYTHONPATH=. python -m experiments.sam_training_set_detection.measure
```

## What is still open

Swin has 0 of 496 SAM checkpoints swept at both placements, and GTSRB and Tiny
ImageNet have 0 of 496 between them, so this verdict covers ViT on CIFAR-10 and
CIFAR-100 only. Every seed on disk is seed 0, so the mechanism reading cannot yet be
told apart from ordinary seed to seed variance. Full ranked list in
`docs/sam-audit-2026-09-11.md` section 5.

---

## History

### Status change, 2026-09-11: matched comparison supersedes the drop

`docs/sam-findings-2026-09-11.md` rebuilds the Adam against SAM comparison directly,
matched on architecture, dataset, attack and poison rate, over 24 checkpoint pairs
per rho rather than the 18-combination unmatched aggregate the 2026-09-07 drop rested
on. The status above replaces the DROPPED verdict below. SAM checkpoints stay
excluded from every reporting path by default, since the gain that does exist is
small and rate dependent, but SAM is no longer treated as a settled null result and
its own document now carries the current reading.

### Status change, 2026-09-07: DROPPED rather than refuted

This hypothesis is no longer an open question in this project and no claim rests
on it.

The measured effect is +0.009 mean AUROC. Detecting an effect that size against
plausible seed to seed variance requires 10 to 89 training seeds per cell, by
`n >= 8 * (sigma_seed / effect)^2` at 80 percent power. Every checkpoint here is
seed 0, so the variance is unmeasured, and buying enough seeds to conclude that a
0.009 effect is really 0 is not a defensible use of compute.

The decision is therefore to stop measuring it rather than to keep refining the
refutation. SAM checkpoints are 69 percent of the trained set and 75 percent of
all analysis rows, so carrying them also diluted every panel they appeared in,
which is the same unequal-coverage failure mode recorded in the ledger's own
"failure mode" section.

What changed in the code: `scripts/detection_summary.py`, `defence_tables.py` and
`pbs/generate_gaussian_rerun_jobs.py` exclude SAM by default behind an explicit
`--include-sam` flag. The checkpoints and their caches remain on disk. Nothing is
deleted, nothing is claimed.

The 2 observations worth keeping are recorded in `docs/results-report.md` section
4.7, and both are about the backdoor rather than about SAM: SAM does not remove
the backdoor, and it does not resist rank-1 direction removal once the ablation is
done after the final LayerNorm rather than before it.
