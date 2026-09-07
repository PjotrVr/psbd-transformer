# Does latent geometry predict detection, without running the detector?

## Question

The curvature account says PSU estimates the curvature of the predicted class
probability, and curvature falls as the decision margin grows. It does not say
where a large margin comes from. If a backdoor works by routing triggered inputs
onto a low dimensional subspace, then the rank ratio, measured with **no
perturbation at all**, should predict how well a perturbation based detector does.

A null result is informative: it would mean the collapse is a correlate of the
backdoor rather than a source of the margin.

## Method

282 checkpoints, every one that has both a cached detection result and an
`args.json`, spanning 2 architectures, 4 datasets, 10 attacks and 3 poison rates.
SAM ablations excluded, matching the rest of the project's reporting.

For each, 500 paired samples, the same test images with and without the trigger,
features read at every block. `best_deployable_auroc` is the best AUROC any
**deployable** rule reached in the cached sweep; oracle rules are excluded because
a rule that places its threshold using the labels it is predicting is not a
detector. Layer 0 is dropped, since on ViT under the cls reduction it is a
constant.

## Result

0 failures over 282 checkpoints.

| predictor | rho with AUROC | rho with ASR | **partial rho given ASR** |
|---|---:|---:|---:|
| `min_rank_ratio` | -0.541 | -0.451 | **-0.310** |
| `min_cka` | -0.678 | -0.630 | **-0.342** |
| `max_separation_auroc` | 0.498 | 0.556 | **0.095** |
| `max_target_alignment` | 0.492 | 0.468 | 0.244 |

The sign is the predicted one: more collapse, so a smaller ratio, means easier
detection.

**The ASR control is the point.** A backdoor that does not work cannot collapse
anything, so any collapse measure is bound to track attack success. Holding ASR
fixed, the collapse measures keep a real partial correlation (-0.31 and -0.34)
while **raw separability falls to 0.095**, meaning separability predicts detection
almost entirely through ASR and carries nothing of its own. That is the same
separation notebook 02 found by a different route: separability reaches 0.96 even
on a benign model, and the collapse measures do not.

Within the high ASR band alone (ASR above 0.9, n = 170) the correlation weakens to
-0.157 (p = 0.042). Real, correctly signed, and modest.

Benign controls behave exactly as a control should:

| control | rank ratio | cka | detection AUROC |
|---|---:|---:|---:|
| `vit_cifar100_benign` | 0.970 | 0.984 | 0.510 |
| `swin_cifar100_benign` | 0.989 | 0.994 | 0.511 |
| `vit_cifar10_benign` | 0.963 | 0.994 | 0.506 |
| `vit_gtsrb_benign` | 0.996 | 0.999 | 0.507 |
| `vit_tiny_benign` | 0.982 | 0.997 | 0.505 |

Geometry at 1, detection at chance, on all 5.

## What this does NOT establish

**Nothing forces the collapse.** The classifier head reads features only through
its weight matrix, 100 by 768 on CIFAR-100, so a 668 dimensional null space is
invisible to the logits. Moving energy inside it takes the rank ratio from 0.210
to 0.999 with the logits unchanged to 1.9e-5. An attacker can therefore set the
rank ratio to nearly anything without touching predictions or ASR, so this is a
regularity of how ordinary training behaves, **not a property a backdoor
requires**.

**The threshold does not transfer across datasets.** The class-matched baseline
runs from above 1 at 10 classes to 0.085 at 200, so a single cut is not valid
across the 4 datasets here.

**The published adaptive attack has not been tested.** `attacks/adaptive_blend.py`
implements 1 of Qi et al.'s 3 mechanisms, and the one it implements acts on a
first moment that a mean centred statistic removes before measuring. The
`adaptive_blend` checkpoints in this sample are therefore **not** evidence about
their published attack.

## Reproduce

    python experiments/latent_geometry_predicts_detection/measure.py --samples 500

About 70 minutes on 1 A100 with the CPU doing the spectral work. Analysis and
figures in `notebooks/10-geometry-predicts-detection.ipynb`.
