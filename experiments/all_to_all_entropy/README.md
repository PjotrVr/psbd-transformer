# PSBD's uncovered case is covered by a statistic its own cache already holds (H43)

## Question

H5 and `docs/all-to-all-inversion.md` establish that PSBD does not work on all-to-all, on
both architectures, including ResNet-18 on the original authors' own recipe (TPR 0.210
against FPR 0.192). The reason is structural: PSBD needs the trigger to be a **constant,
content-independent** shortcut, and `(y + 1) mod K` forces the model to read the source
class before it can increment.

That content dependence has a second consequence nobody had read off. Under all-to-all
the poisoned logit must beat the **true source class** rather than an arbitrary
runner-up, so the margin is contested. That is exactly why all-to-all costs ASR (0.842
against 1.000 on CIFAR-10), and it should also make triggered inputs less confident.

Confidence is not something PSBD measures. But its cache already holds it: every
`baseline_<split>.pt` stores the full `(N, num_classes)` no-dropout softmax, and only the
probability of the argmax class is ever read.

## Method

Same splits, same pairing, same matched shift ratio 0.6 as every other number in this
repo. Direction fixed a priori and never chosen from an AUROC (H15): a contested margin
means a flatter posterior, so entropy is higher and `neg_entropy` lower on a triggered
input, and low means poisoned as everywhere else.

CPU only, no GPU and no model. `logit_margin` is exact, not approximate: softmax is
shift-invariant, so `log p_1 - log p_2 = z_1 - z_2` identically.

## Result

| group | n | `psu_ratio` | `neg_entropy` | `confidence` | `logit_margin` |
|---|---|---|---|---|---|
| all-to-one, ASR >= 0.5 | 55 | **0.891** | 0.408 | 0.405 | 0.375 |
| **all-to-all** | 12 | **0.411** | **0.761** | 0.752 | 0.728 |
| benign control | 4 | 0.496 | 0.506 | 0.506 | 0.505 |

The 2 statistics are almost perfectly complementary, and the benign control sits at
chance for both, so neither is an artifact of the pipeline.

Two things follow.

**The case PSBD declares uncoverable is not uncoverable.** A single deterministic forward
pass, using a tensor the sweep already writes to disk, reaches 0.761 mean where PSBD
reaches 0.411.

**And PSBD is not a margin detector.** On all-to-one the deterministic margin scores 0.375
against PSU's 0.891, so the two are not measuring the same thing. That is the same
conclusion an independent adversarial review reached by a different route, stratifying on
the top-2 margin and finding PSBD's AUROC barely moves.

## What this does not give you

A deployable defence. H15 forbids picking whichever of 2 one-sided detectors happens to
win, because choosing needs the poison labels the detector exists to predict. A
**label-free regime identifier** is required and 2 candidates are already refuted:

| router | result |
|---|---|
| shift-gap gating (`val shift - suspect shift`) | 0.843 against 0.852 for always-PSU; entropy wins 14 of 25 in the low-gap group, a coin flip |
| shift-destination concentration | a2o 0.469, a2a 0.422, benign 0.481: no separation at any threshold |

So the open problem is sharp: a single statistic that is one-sided in both regimes, or a
label-free way to tell the regimes apart.

## Running it

    PYTHONPATH=. python experiments/all_to_all_entropy/measure.py

About 2 minutes on CPU over the cached panel. Writes `results/all_to_all_entropy.json`.
