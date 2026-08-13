# H5 — PSBD degrades on all-to-all, which has no single target class

**Status: REFUTED, and the refutation is one of the strongest results here.**
All-to-all is not undetectable. Its PSU signal is **inverted**, and once the sign is
allowed to flip it is detected at AUROC 0.97, better than most attacks in the grid.

## Original claim

PSBD's mechanism needs a single target class `y_t` that clean samples collapse onto
under dropout. All-to-all maps class `y` to `(y+1) mod K`, so there is no such class,
and detection should degrade badly relative to `badnet_a2o`, which uses an identical
trigger and differs only in label geometry.

## What the one-sided score said

At the paper's decision rule (flag **low** PSU as poisoned), `badnet_a2a` looks dead:
0.601 / 0.539 / 0.510 at 1% / 5% / 10% poisoning, against a benign floor of 0.506,
while `badnet_a2o` reaches 0.889. That was recorded as strong support.

## What was actually happening

The depth-band sweep produced an AUROC of **0.159** for `badnet_a2a` at blocks 9-12.
A value that far *below* 0.5 is not noise, it is a detector running backwards. Looking
at the PSU means explains it immediately (blocks 9-12, rate 0.7):

| | PSU clean | PSU backdoor | reading |
|---|---|---|---|
| `badnet_a2o` | 0.101 | **0.022** | backdoor **more** robust, the paper's premise |
| `badnet_a2a` | 0.144 | **0.588** | backdoor **less** robust, the premise inverted |

All-to-all backdoor samples are not resistant to perturbation. They are *four times
more fragile* than clean samples.

## Two-sided detection

Scoring `max(AUROC, 1 - AUROC)`, i.e. allowing the decision rule to flag whichever
tail separates:

| checkpoint | placement | one-sided | **two-sided** | rate |
|---|---|---|---|---|
| `badnet_a2a` 1% | blocks 9-12 | 0.102 | **0.988** | 0.7 |
| `badnet_a2a` 5% | blocks 9-12 | 0.217 | **0.946** | 0.7 |
| `badnet_a2a` 10% | blocks 9-12 | 0.159 | **0.966** | 0.7 |
| `badnet_a2a` 10% | `post_residual` | 0.523 | 0.911 | 0.05 |
| **benign control** | blocks 9-12 | 0.487 | **0.522** | 0.7 |

**The control is the point.** A two-sided statistic can inflate anything, so the
benign model was scored under exactly the same rule, with the same max over rates and
placements. It moves from 0.506 to 0.522. The all-to-all model moves from 0.159 to
0.966. The gain is not an artifact of two-sidedness.

Note the trend also reverses to the expected direction: two-sided detection is
*strongest* at the lowest poison rate (0.988 at 1%), which resolves the oddity flagged
earlier that one-sided detection got worse as poisoning increased.

## Why all-to-all is fragile rather than robust

All-to-one learns one mapping, trigger to `y_t`, reinforced by every poisoned sample.
It is a single high-capacity shortcut and it survives heavy perturbation, which is
exactly PSBD's premise.

All-to-all learns `K` mappings, `y -> (y+1) mod K`, each reinforced by roughly `1/K`
of the poisoned data, and each **conditional on the source class**. The model must
therefore preserve the clean class identity *and* apply an offset. That is a more
complex and more delicate function than a single shortcut, so under perturbation it
degrades faster than the clean class evidence it depends on.

So the label geometry does matter, as the original hypothesis said, but it changes the
**sign** of the effect rather than removing it.

## What this changes

- PSBD as published would report all-to-all as undetectable. It is not; the rule is
  one-sided and the phenomenon is not.
- A defender does not know the label geometry in advance, but does not need to: the
  two-sided rule costs almost nothing on a benign model (0.506 to 0.522) and recovers
  a whole attack family.
- It is independent evidence for the mechanism in [H10](H10-depth-band-placement.md).
  The inversion is largest at blocks 9-12, the band that spares an early-written
  backdoor. For a2a there is no robust backdoor to spare, so the same band instead
  exposes how fragile it is.

## Reproduce

`results/vit_cifar10_badnet_a2a_*/psbd_metrics.json`, comparing
`psu_mean.clean` against `psu_mean.backdoor`, and `detection.q0.25.auroc` against its
complement.

## Subquestions

1. Does a formal two-sided statistic (distance from the clean-validation median rather
   than a one-tailed quantile) work as a general rule, and what does it cost on the
   attacks where the one-sided rule already works?
2. Do other multi-target attacks invert the same way? TaCT is source-specific but does
   not train on ViT here, so this may need a new checkpoint.
3. Does the inversion grow with the number of classes? CIFAR-100 all-to-all would have
   100 mappings from the same poison budget, so it should be more fragile still.
