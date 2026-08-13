# H3 — Post-residual fails because it saturates: it destroys clean and backdoor evidence alike

**Status: REFUTED as stated. The saturation is real and measured, but it does not
make post-residual fail, and the sub-0.1 window it claimed does not exist was found.**

## Original claim

Post-residual dropout fails not because it is too weak but because it is too
destructive at every usable rate. Masking the residual stream once per block leaves
`(1-p)^12` of coordinates, so at any swept rate the model's prediction becomes noise
for clean and backdoor samples alike, the PSU distributions collapse onto each
other, and there is nothing left to threshold. The claim's sharpest form: **no small
enough p exists, because the compounding is exponential in depth.**

## What was actually found

Two halves, one confirmed and one refuted.

**Confirmed: the saturation is real.** Clean-validation shift ratio on
`vit_cifar10_badnet_a2o_0_1`:

| rate | post-residual sigma | pre-residual sigma |
|---|---|---|
| 0.1 | **0.859** | 0.010 |
| 0.3 | 0.885 | 0.417 |
| 0.5 | 0.917 | 0.880 |
| 0.9 | 0.948 | 0.922 |

Post-residual enters the standard grid already above the shift ratio the PSBD paper
uses as its *operating point* (0.8), and never moves. Pre-residual traverses 0.01 to
0.92. So on the standard grid, post-residual genuinely has no low-disturbance regime.

**Refuted: "no small enough p exists" is false.** Measuring how much of the backdoor
direction survives (`scripts/dropout_kills_direction/`, `badnet_a2o`, layer 12,
separation normalized by its unperturbed value):

| post-residual rate | direction separation retained |
|---|---|
| 0.005 | 0.91 |
| 0.01 | 0.80 |
| 0.02 | 0.62 |
| 0.03 | 0.44 |
| 0.05 | 0.19 |
| 0.08 | 0.04 |
| 0.10 | 0.015 |
| 0.20 and above | 0.000 |

There is a perfectly good operating window at p = 0.01 to 0.03. It sits an order of
magnitude below where anyone had looked, because the rate grid was inherited from
the ConvNet paper, whose ResNet-18 has 8 residual adds rather than 24.

**And post-residual does not fail.** On the full grid it reaches AUROC 0.92 to 0.98
on `blend` and `bpp`, comparable to pre-residual. It collapses only on `badnet_a2o`.

## Why the correction matters more than the original claim

The original hypothesis would have produced a confident, wrong, mechanistic story:
"the ConvNet placement cannot work on transformers because depth compounds the
mask". The truth is narrower and more useful: **the ConvNet placement works on
transformers, at a rate the ConvNet grid never contains.** The transferable lesson
is about the rate grid, not the placement.

It also means every post-residual number currently on record was measured at or past
the edge of its own operating range, which is not a fair comparison. 16 fine-rate
jobs (0.005 to 0.09) are running to fix that.

## Reproduce

```bash
PYTHONPATH=. python scripts/dropout_kills_direction/measure.py \
    --checkpoint-folder vit_cifar10_badnet_a2o_0_1 \
    --placement post_residual --rates 0.005 0.01 0.02 0.03 0.05 0.08
```

## Subquestions

1. Does the usable window scale as `(1-p)^depth` predicts? Swin has 24 blocks, so
   its window should sit at roughly half the rate. Directly checkable and would turn
   this from an observation into a formula.
2. Should the whole rate grid be defined in units of *achieved shift ratio* rather
   than nominal p? That would make placements comparable by construction, and this
   whole correction would have been unnecessary.
3. Does an additive perturbation (Gaussian noise) at the same position avoid the
   exponential collapse entirely? `plug_dropout`'s `dropout_factory` argument
   already supports it and nothing has used it.
