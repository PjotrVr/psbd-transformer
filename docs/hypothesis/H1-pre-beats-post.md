# H1 — Pre-residual dropout beats post-residual dropout on ViT

**Status: SUPPORTED, but far more weakly than the founding claim, and the effect is
concentrated in one trigger family**

## Claim

Placing PSBD's dropout on each transformer branch just before the residual add
separates clean from backdoor samples better than placing it on the residual stream
just after each add, on ViT-B/16.

## Evidence

Full phase-3a grid: CIFAR-10 ViT, 5 attacks x 3 poison rates + benign control,
full 10000-image test split, 9 rates, k=3. AUROC at the paper's 25th-percentile
threshold, best rate per placement.

| attack | pre-residual | post-residual | gap | direction onset |
|---|---|---|---|---|
| `blend` | 0.978 - 0.986 | 0.924 - 0.929 | **+0.055** | layer 5 |
| `bpp` | 0.964 - 0.995 | 0.934 - 0.979 | **+0.024** | layer 6 |
| `lf` | 0.887 - 0.947 | 0.887 - 0.916 | **+0.030** | layer 8 |
| `badnet_a2o` | 0.686 - 0.889 | 0.496 - 0.667 | **+0.226** | layer 9 |
| `badnet_a2a` | 0.510 - 0.601 | 0.523 - 0.596 | -0.003 | layer 9 |
| benign control | 0.506 | 0.507 | -0.000 | n/a |

**Pre-residual wins everywhere it matters, but by 0.02 to 0.06 on three of the four
working attacks.** The one large gap is `badnet_a2o` at +0.226, where post-residual
collapses to near chance (0.496 at 1% poisoning) while pre-residual holds 0.686.

## Correction to the preliminary verdict

This file previously read SUPPORTED on the strength of a 64-sample smoke run of
`badnet_a2o` alone, which showed 0.903 against 0.701. That checkpoint turned out to
be the single most favourable case in the grid. Generalizing from it would have
overstated a 0.03 average effect as a 0.20 one.

The smoke run was not wrong, it was unrepresentative, and one checkpoint is not
evidence for a claim quantified over five attacks.

## Honest statement of the result

Post-residual dropout **does not fail on ViT**. It reaches 0.92 to 0.98 AUROC on
`blend` and `bpp`. The founding intuition, that the ConvNet placement simply does
not transfer, is too strong: it transfers fine for spatially distributed triggers
and fails specifically for the static patch trigger.

That is a more interesting result than the original claim, and it is what
[H4](H4-placement-is-attack-dependent.md) predicted.

## Still open

Post-residual was only swept from p=0.1 upward, and separate measurement shows its
usable window sits **below** that (the backdoor direction retains 90% of its
separation at p=0.01 and 4% at p=0.08). So every post-residual number above is its
score at the edge of, or outside, its own operating range. 16 fine-rate jobs
(0.005 to 0.09) are running to close this. Until they land, the defensible claim is
"pre-residual beats post-residual on the standard rate grid", not "beats it".

## Reproduce

```bash
python psbd_analyze.py --all
python psbd_report.py
```

## Subquestions

1. Does post-residual catch up on `badnet_a2o` once its own rate window is swept?
   That single number decides whether this hypothesis survives.
2. Why is `blend`'s gap (+0.055) larger than `bpp`'s (+0.024) when its direction
   enters *earlier*? The onset ordering does not explain the gap ordering among the
   three easy attacks.
3. Does the gap grow on Swin, whose 24 blocks compound a stream perturbation twice
   as often?
