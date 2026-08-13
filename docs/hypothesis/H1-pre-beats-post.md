# H1 — Pre-residual dropout beats post-residual dropout on ViT

**Status: SUPPORTED (preliminary, n=64 smoke, full sweep in flight)**

## Claim

Placing PSBD's dropout on each transformer branch just before the residual add
separates clean from backdoor samples substantially better than placing it on the
residual stream just after each add, on ViT-B/16.

This is the project's founding observation and everything else exists to explain
it. The PSBD paper places dropout after the residual add on ResNet-18, so if this
holds, the ConvNet recipe does not transfer to transformers and the reason is
worth a paper.

## Prediction

`pre_residual` AUROC exceeds `post_residual` AUROC at the paper's 25th-percentile
threshold, for every attack with working ASR, at every poison rate. Refuted if
post-residual matches or wins on any substantial share of the grid.

## Why it is interesting

The two placements differ by a single tensor position inside a block. Pre-residual
perturbs an additive contribution; post-residual perturbs the accumulated stream.
If a one-position move flips a defence from working to useless, then the residual
stream is doing something structurally different in a transformer from what it
does in a ResNet, and the CKA homogeneity result (ViT residual streams are
persistent across depth) is the obvious candidate explanation.

## Evidence

Smoke run, `vit_cifar10_badnet_a2o_0_1` (ASR 1.00), 64-sample splits, 9 rates,
k=3. Small and noisy, but the direction is unambiguous.

| placement | best AUROC | at rate | PSU clean | PSU backdoor |
|---|---|---|---|---|
| `pre_residual` | **0.903** | 0.4 | 0.808 | 0.508 |
| `post_residual` | 0.701 | 0.1 | 0.855 | 0.848 |

Post-residual's clean and backdoor PSU are within 0.007 of each other. The two
distributions have collapsed onto each other, which is what "the defence has no
signal" looks like. Pre-residual holds a 0.30 gap at its best rate.

The full grid (5 attacks x 3 poison rates + benign control, 32 jobs) is running.

## What would change the verdict

- Post-residual winning on the distributed-trigger attacks (`bpp`, `lf`), which
  would mean the answer is attack-dependent rather than universal. See [H4](H4-placement-is-attack-dependent.md).
- The gap disappearing once both placements are compared at matched perturbation
  strength rather than at matched dropout rate. That objection is [H9](H9-strength-not-position.md)
  and it is the one that decides whether this result survives review.

## Reproduce

```bash
python psbd_dropout_sweep.py --checkpoint-folder vit_cifar10_badnet_a2o_0_1 \
    --position-config pre_residual
python psbd_dropout_sweep.py --checkpoint-folder vit_cifar10_badnet_a2o_0_1 \
    --position-config post_residual
python psbd_analyze.py --checkpoint-folder vit_cifar10_badnet_a2o_0_1
```

Reads `results/<folder>/psbd_metrics.json`, key `placements.<name>.oracle`.

## Subquestions this opens

1. Is the gap driven by the attention-side add, the MLP-side add, or both? The
   single-position sweep separates them.
2. Does the gap widen or narrow with depth? A per-block placement sweep would say
   whether only the last few blocks matter.
3. Does it hold on Swin, whose 24 blocks compound a stream perturbation twice as
   many times as ViT's 12?
4. Is there a placement better than pre-residual that nobody has tried, for
   example on the attention branch only, or before the QKV projection?
