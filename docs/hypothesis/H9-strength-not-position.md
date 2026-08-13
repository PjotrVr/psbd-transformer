# H9 — The pre/post gap is a perturbation-strength artifact, not a placement effect

**Status: SUPPORTED for three of four attacks.** The adversarial hypothesis was
right, and it overturned the project's founding claim.

## Claim

Stated at the outset as the reviewer's objection, so that it could be attacked
rather than discovered in review:

> Pre-residual and post-residual were compared at the same nominal dropout rate `p`,
> but the same `p` is a completely different intervention at those two positions.
> Pre-residual perturbs one additive branch contribution. Post-residual multiplies the
> entire residual stream by a mask, once per block, so only `(1-p)^12` of coordinates
> survive. So "pre-residual beats post-residual" may just be "the weaker perturbation
> beats the destructive one".

## Evidence

The prediction was that post-residual, given a rate window where it is not already
saturated, would close the gap. It did more than that.

Swept over p = 0.005 to 0.09 in addition to the standard 0.1-to-0.9 grid, best-rate
AUROC at 10% poisoning:

| attack | post, standard grid only | post, own window | pre_residual | post's best rate |
|---|---|---|---|---|
| `blend` | 0.924 | **0.989** | 0.978 | 0.07 |
| `bpp` | 0.934 | **0.991** | 0.977 | 0.07 |
| `lf` | 0.916 | **0.969** | 0.947 | 0.07 |
| `badnet_a2o` | 0.667 | 0.747 | **0.889** | 0.02 |

Post-residual gains 0.053 to 0.080 simply from being measured inside its own range,
and **overtakes pre-residual on three of the four working attacks**.

## Verdict, split

**SUPPORTED for `blend`, `bpp`, `lf`.** The apparent placement effect was an artifact
of the rate grid. Once both placements are measured where each actually operates,
post-residual is the better of the two.

**REFUTED for `badnet_a2o`.** Pre-residual leads by +0.142 even with post-residual at
its own optimum. That gap is not a strength artifact; it is a real placement effect,
and it is confined to the static patch trigger, whose backdoor direction only reaches
the `[CLS]` token in the last few blocks
([H4](H4-placement-is-attack-dependent.md)).

## Why this matters more than the original claim

The founding intuition was "the ConvNet placement does not transfer to transformers".
What actually transfers badly is **the ConvNet rate grid**. ResNet-18 has 8 residual
adds; ViT-B/16 has 24. The same nominal `p` compounds three times as hard, so a grid
tuned on one architecture starts past the operating point on the other.

That is a smaller-sounding but more useful and more general claim: when porting a
perturbation-based defence across architectures, the rate grid has to be re-derived
from depth, not inherited.

## The methodological rule this establishes

**No comparison across placements is valid at a shared dropout rate.** Match on a
measured disturbance instead. `defences.psbd_metrics.select_rate_at_matched_shift`
uses the clean-validation shift ratio, which stays defender-legal (clean data only)
and is reported per placement in every `psbd_metrics.json`.

The same confound has now appeared a second time, in
[H10](H10-depth-band-placement.md): an early block band perturbs harder than a late
one at the same rate, because it propagates through more blocks. Treat it as the
default hazard of this line of work.

## Subquestions

1. Does post-residual's advantage on the three distributed-trigger attacks survive a
   matched-sigma comparison, or does it too flip once strength is equalized? The
   `matched_shift` block in each `psbd_metrics.json` already holds the answer and has
   not been read yet.
2. `post_residual`'s best rate is 0.07 for three attacks and 0.02 for the fourth.
   Does the optimum track the direction onset layer, as the depth story predicts?
3. Should the rate grid be defined in units of achieved shift ratio from the start,
   for every placement? That would make this class of error impossible rather than
   merely detectable.
