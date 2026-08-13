# H16 — The backdoor is one late-layer linear direction, not a set of neurons; it is disjoint across attacks, and SAM relocates it without weakening it

**Status: SUPPORTED for layers and for the direction. The "neurons" framing is
REFUTED by its own causal test**, as are both data-free localizers (Lipschitz
channel ranking and the head-alignment Z rule).

> **Headline correction.** This file originally read the top-20 TAC dimensions as
> "the backdoor neurons". The ablation in section 8 shows they are not: zeroing
> even the top 300 of 768 coordinates leaves ASR at 1.00, while removing the single
> backdoor *direction* takes ASR to 0.00. The direction is real and causal; it is
> simply not axis-aligned, so no coordinate ranking can name it. Sections 2 and 4
> are kept because the disjointness result stands on its own, but they describe
> where the direction's energy lands in the standard basis, not a set of neurons.

## Claim

Three separable claims, tested together because they need the same measurement:

1. **Localization.** For each attack there is a specific block and a small set of
   residual-stream dimensions carrying the trigger, not a diffuse effect.
2. **Disjointness.** Different attacks use different dimensions, so a defence tuned
   to 1 attack's neurons does not transfer.
3. **SAM.** Sharpness-aware training moves the backdoor earlier in depth and onto a
   different set of dimensions, without reducing attack success.

## Method

`scripts/backdoor_neurons/measure.py` over 18 checkpoints (5 attacks + benign) ×
(Adam, SAM rho 0.1, SAM rho 0.2), ViT-B/16 CIFAR-10 at 10% poisoning, 600 paired
clean/triggered samples, fp32, seed 0. Per layer it computes TAC, the relative
backdoor-direction norm (direction norm divided by the mean clean CLS norm, so
growth is not just the residual stream getting larger), and debiased linear CKA.
Per dimension it takes the top-20 TAC indices and the CLP outlier rule at mean + 3
std. It also runs the 2 data-free channels (per-channel Lipschitz of each block's
MLP output projection, and the head-alignment Z rule) and PCA/UMAP separability
scored by silhouette and 10-NN purity rather than by eye.

`scripts/backdoor_neurons/stability.py` supplies the 2 reference points without
which no Jaccard is interpretable.

## Evidence

### 1. Localization: late, and sharp

Relative backdoor-direction norm per block, Adam:

| attack | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `badnet_a2o` | 0.00 | 0.01 | 0.03 | 0.05 | 0.09 | 0.12 | 0.30 | 0.47 | 0.59 | 0.70 | 0.85 | **1.03** | 12 |
| `blend` | 0.02 | 0.07 | 0.18 | 0.48 | 0.63 | 0.83 | 0.97 | 1.05 | 1.04 | 1.03 | 0.98 | **1.11** | 12 |
| `bpp` | 0.01 | 0.04 | 0.11 | 0.25 | 0.31 | 0.65 | 0.64 | 0.85 | 0.96 | 1.19 | 1.11 | **1.20** | 12 |
| `lf` | 0.00 | 0.01 | 0.06 | 0.14 | 0.24 | 0.40 | 0.61 | 0.90 | 1.51 | **2.18** | 1.64 | 1.36 | 10 |
| `badnet_a2a` | 0.00 | 0.01 | 0.03 | 0.04 | 0.05 | 0.05 | 0.12 | 0.12 | 0.30 | 0.25 | **0.69** | 0.49 | 11 |
| `benign` | 0.00 | 0.01 | 0.03 | 0.04 | 0.06 | 0.07 | 0.08 | **0.09** | 0.09 | 0.07 | 0.06 | 0.05 | 8 |

CKA between clean and triggered features tells the same story from the other side
(1.00 means the trigger is invisible at that layer): `badnet_a2o` holds 1.00 through
layer 5 then falls to 0.41; `blend` and `bpp` break at layer 6 and end near 0.11 to
0.18; `benign` stays at 1.00 at every single layer.

The benign row is the control that makes the rest readable. Probed with the same
`badnet_a2o` trigger, a clean model's relative direction peaks at 0.09 and *decays*
with depth, and its CKA never leaves 1.00. Every backdoored model grows
monotonically to 0.7 to 2.2. So this is measuring a learned backdoor, not the input
perturbation.

### 2. Which neurons: 5 to 17 out of 768

At each attack's own peak layer, the CLP outlier rule (mean + 3 std) flags:

| attack | Adam | SAM 0.1 | SAM 0.2 | peak-layer tac_max / tac_mean, Adam |
|---|---|---|---|---|
| `badnet_a2o` | 7 | 10 | 9 | 2.8 |
| `blend` | 11 | 9 | 13 | 3.1 |
| `bpp` | 10 | 8 | 13 | 3.1 |
| `lf` | 9 | 5 | 6 | 3.5 |
| `badnet_a2a` | 12 | 10 | 17 | 4.8 |
| `benign` | 14 | 15 | 10 | 5.8 |

Between 5 and 17 dimensions out of 768, so under 2.2% of the residual width. The
benign model flags a comparable *count*, which is expected because the rule is a
relative one, but at an absolute scale 16x smaller: benign tac_max is 0.215 against
3.26 to 3.89 for every backdoored model. The count does not separate; the magnitude
does.

### 3. The 2 numbers that make Jaccard mean anything

Every overlap claim below sits between a measured floor and a measured ceiling:

- **Chance floor 0.014** (p95 0.053), from 20000 random pairs of 20-of-768 sets.
- **Split-half ceiling 0.87** (range 0.74 to 1.00 across all 18 checkpoints), from
  ranking dimensions on 2 disjoint interleaved halves of the same 600 samples,
  same model, same trigger, same layer.

The ceiling is the important one and it was not obvious in advance. It says the
top-20 TAC set is a reproducible property of the model, not an artefact of which
600 images were drawn. Without it, every low Jaccard below would be explainable as
noise and the SAM claim would be vacuous.

### 4. Disjointness: attacks share nothing

Jaccard of top-20 TAC dimensions at each attack's peak layer, Adam:

| | badnet_a2a | badnet_a2o | benign | blend | bpp | lf |
|---|---|---|---|---|---|---|
| `badnet_a2a` | 1.00 | 0.00 | 0.03 | 0.05 | 0.00 | 0.00 |
| `badnet_a2o` | 0.00 | 1.00 | 0.08 | 0.05 | 0.03 | 0.03 |
| `benign` | 0.03 | 0.08 | 1.00 | 0.03 | 0.00 | 0.03 |
| `blend` | 0.05 | 0.05 | 0.03 | 1.00 | 0.05 | 0.05 |
| `bpp` | 0.00 | 0.03 | 0.00 | 0.05 | 1.00 | 0.05 |
| `lf` | 0.00 | 0.03 | 0.03 | 0.05 | 0.05 | 1.00 |

Every off-diagonal entry is 0.00 to 0.08, against a chance floor of 0.014 and a p95
of 0.053. The largest, 0.08, is 3 shared dimensions out of 20; the common 0.03 is 1.
Against a ceiling of 0.87, this is **indistinguishable from disjoint**.

Note `badnet_a2o` and `badnet_a2a` share the identical trigger pattern and overlap
at exactly 0.00. What determines the dimensions is not the trigger's appearance but
the label mapping the model learned from it.

### 5. SAM: relocates, does not weaken

| attack | peak Adam | peak 0.1 | peak 0.2 | J(0.1) | J(0.2) | ASR Adam / 0.1 / 0.2 |
|---|---|---|---|---|---|---|
| `badnet_a2o` | 12 | 11 | 10 | 0.08 | 0.05 | 1.000 / 1.000 / 1.000 |
| `blend` | 12 | 11 | 9 | 0.08 | 0.03 | 1.000 / 1.000 / 1.000 |
| `bpp` | 12 | 10 | 10 | 0.11 | 0.08 | 0.999 / 0.994 / 0.995 |
| `lf` | 10 | 10 | 9 | 0.05 | 0.03 | 0.999 / 0.999 / 0.999 |
| `badnet_a2a` | 11 | 11 | 8 | 0.18 | 0.05 | 0.959 / 0.962 / 0.957 |
| `benign` | 8 | 8 | 8 | **0.29** | **0.25** | n/a |

Two effects, both clean:

**Depth.** The peak layer moves earlier, monotonically in rho, for all 5 attacks
(12 to 10, 12 to 9, 12 to 10, 10 to 9, 11 to 8). The benign control does not move at
all: 8, 8, 8. So SAM is pulling the *backdoor* forward, not shifting where the
network in general does its work.

**Dimensions.** At rho 0.2 the overlap with Adam is 0.03 to 0.08 for every
backdoored model, at or barely above the chance floor of 0.014, against a
reproducibility ceiling of 0.87. The benign control sits at 0.25 to 0.29, above
every backdoored value at either rho. A clean model's most trigger-sensitive
dimensions are a generic, weight-driven property that partly survives an optimizer
change; a backdoor's are not.

**And ASR is untouched**, 0.96 to 1.00 in every cell, with clean accuracy actually
2 points *higher* under SAM (0.939 to 0.963 for `badnet_a2o`). SAM does not damage
the backdoor. It re-implements it somewhere else.

### 6. Both data-free channels: REFUTED

**Lipschitz vs TAC.** The rank correlation between each block's per-channel MLP
output-projection Lipschitz constant and the measured TAC looks encouraging at
first: Spearman 0.21 to 0.67 at the peak layer, mean 0.59 to 0.65 averaged over all
12 layers, positive on all 18 checkpoints. Two checks kill it.

*It is not specific to the backdoor.* The benign model scores **0.660**, the
highest of all 18. If a clean model probed with a trigger it never learned ranks
just as well, the correlation is measuring generic channel sensitivity, which is a
property of the weights, and not backdoor-ness.

*It does not survive to the tip.* What a defender needs is not a global monotone
trend but the actual top dimensions. Jaccard between the Lipschitz top-20 and the
TAC top-20, at each checkpoint's own peak layer:

| | mean | range |
|---|---|---|
| Lipschitz top-20 vs TAC top-20 | **0.038** | 0.000 to 0.111 |

Against a chance floor of 0.014 and a split-half ceiling of 0.87. For `lf` and
`blend` at rho 0.2 the overlap is exactly 0. The moderate global correlation comes
from the bulk of the distribution and carries no information about which few
channels the trigger actually uses.

**Head alignment.** The Karayalcin Z > 3 rule fires on 2 of 18
checkpoints, and they are the wrong 2:

- `bpp` Adam, Z = 4.93, names the **wrong** target class.
- `vit_cifar10_benign_sam_rho_0_2`, Z = 4.93, a **clean model**, and it "names" the
  right class only because target 0 is the probe's arbitrary choice.

Every genuinely backdoored checkpoint except `bpp` scores Z between 0.39 and 2.89,
below the threshold. As a detector this is 0/15 true positives and 1/3 false
positives.

The detector also needed a repair before it could produce any number at all. The
paper's absolute alignment threshold does not transfer: measured on these
checkpoints every alignment magnitude falls below 0.032, so a ConvNet-scale
constant makes every per-class count 0 and the score vector is tied 10 ways. It now
takes the threshold as the 99.9th percentile of the model's own alignment
distribution, which keeps it data-free while making it scale-free. It still fails.

### 7. PCA and UMAP: the separation is linear and total

10-NN purity of the clean/triggered split in the final block's CLS features:

| attack | PCA | UMAP |
|---|---|---|
| `badnet_a2o` | 0.999 | 1.000 |
| `blend` | 1.000 | 0.999 |
| `bpp` | 0.997 | 0.997 |
| `lf` | 0.996 | 0.997 |
| `badnet_a2a` | **0.756** | 1.000 |
| `benign` | 0.455 | 0.454 |

For 4 of 5 attacks a 2-component **PCA** already separates clean from triggered
essentially perfectly, and UMAP adds nothing. That is the sharpest confirmation
available that the backdoor is a *linear* direction in the residual stream: a linear
projection into 2 dimensions cannot manufacture structure that is not already
linearly present.

`badnet_a2a` is the exception and the informative one. Its PCA purity is 0.756 while
its UMAP purity is 1.000. All-to-all has no single target class, so its effect is
not 1 direction but a label-dependent family of them, which a 2-component linear
projection cannot capture and a nonlinear embedding can. This is the same structural
reason `badnet_a2a` breaks PSBD in [H5](H5-all-to-all-breaks-psbd.md), visible here
in the geometry directly.

Benign is at 0.45, below the 0.5 of a random split, so the trigger genuinely does
nothing to a clean model's representation.

### 8. The causal test, which overturns the "neurons" reading

Everything above is correlational. TAC is a *difference*: it says a dimension moves
when the trigger appears, not that the model reads it. `scripts/backdoor_neurons/
ablate.py` deletes things and re-measures ASR and clean accuracy, at each
checkpoint's own peak layer, as a forward hook so no weight changes.

Adam checkpoints, ASR after each ablation (baseline ASR in brackets):

| attack | baseline | remove **direction** | random dir | top-20 | bottom-20 | random-20 | top-300 |
|---|---|---|---|---|---|---|---|
| `badnet_a2o` | 1.00 | **0.00** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `blend` | 1.00 | **0.00** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `bpp` | 1.00 | **0.01** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `lf` | 1.00 | **0.05** | 1.00 | 1.00 | 1.00 | 1.00 | 0.89 |
| `badnet_a2a` | 0.96 | 0.86 | 0.96 | 0.96 | 0.96 | 0.95 | 0.92 |
| `benign` | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.03 |

Removing the rank-1 backdoor direction

    x_ablated = x - (x . unit_direction) * unit_direction

destroys the backdoor on all 4 single-target attacks, at a clean-accuracy cost of
0.078, 0.041, 0.033, and **-0.004** (`lf` gets slightly *better*). Deleting
coordinates does essentially nothing, including 300 of 768, which is 39% of the
residual width.

Three controls, all passing:

- **Random rank-1 directions** (2 per checkpoint, plus 3 extra seeds and the mean
  clean feature direction checked separately on `badnet_a2o`) leave ASR at 1.00 and
  clean accuracy unmoved. So it is not "removing any direction breaks the model".
- **bottom-k and random-k** coordinates match top-k exactly, which is the expected
  result once top-k itself does nothing, and confirms the coordinate ablation is
  not silently failing to apply.
- **Benign** is unaffected on every variant.

So on ViT the backdoor is a genuine linear direction in the residual stream that is
**not axis-aligned**. TAC's top-20 marks where that direction happens to have the
most energy in the standard basis, but the energy is spread thinly enough over the
remaining coordinates that deleting the largest ones leaves the direction intact.

This retroactively explains 2 earlier results. It is why both data-free localizers
failed in section 6: they rank *coordinates*, and coordinates are not what carries
the backdoor. And it is what the PCA result in section 7 was already saying, since
a 2-component linear projection separating at purity 0.999 is exactly the signature
of a linear-but-rotated direction.

`badnet_a2a` is again the exception, at 0.86. All-to-all has no single target class
and therefore no single direction to remove, which is the same structural fact
behind its PCA 0.756 versus UMAP 1.000 and behind [H5](H5-all-to-all-breaks-psbd.md).

### 9. SAM makes the backdoor un-removable by a single direction

ASR after removing the backdoor direction, by optimizer:

| attack | Adam | SAM 0.1 | SAM 0.2 |
|---|---|---|---|
| `badnet_a2o` | 0.00 | 0.00 | **0.50** |
| `blend` | 0.00 | **0.99** | **1.00** |
| `bpp` | 0.01 | 0.15 | 0.23 |
| `lf` | 0.05 | 0.35 | **0.97** |
| `badnet_a2a` | 0.86 | 0.91 | 0.94 |

Monotone in rho for all 5. Read with section 5, SAM does not merely move the
backdoor to different coordinates; it makes it **less rank-1**, spreading it over a
subspace that a single-direction ablation cannot remove. ASR is 0.96 to 1.00
throughout, so this is a change in how the backdoor is represented, not in how well
it works.

**The obvious confound is ruled out internally.** The ablation fires at the peak
layer, and SAM moves that peak earlier, so later blocks have more depth in which to
rewrite the direction. But `blend` at rho 0.1 and `badnet_a2o` at rho 0.1 both peak
at layer **11**, identical remaining depth, and give ASR 0.99 and 0.00
respectively. Depth-to-recover does not explain the split.

## What this does and does not say

It says where the backdoor is, and that a single direction carries it causally. It
does not say that knowing this helps *detect* it. The gap is
[H7](H7-clean-shifts-to-target.md)'s: the mechanism is present in the
representation and absent in the decision. Both the PCA that separates at purity
0.999 and the direction whose removal takes ASR to 0.00 are computed from paired
clean/triggered inputs, and a defender has no triggered inputs. What this supports
is a *removal* story given a suspected trigger, not a detection story.

The SAM result reads directly onto [H6](H6-sam-improves-detectability.md). SAM
amplifies TAC and relocates the backdoor, but relocation is not removal, and the
detector never sees the difference.

## Subquestions

1. **Do the relocated dimensions under SAM stay disjoint from a second SAM seed?**
   Only seed 0 exists for these checkpoints, so "SAM relocates" cannot yet be
   distinguished from "any retraining relocates". This is the cheapest and most
   important follow-up in this file, and it needs 1 retrain per cell, not a sweep.
2. If the peak layer moves earlier under SAM, the best block-restricted placement
   from [H10](H10-depth-band-placement.md) should move earlier with it. Blocks 5-8
   won on Adam; at rho 0.2 the peaks are at 8 to 10, so blocks 5-8 should hold or
   improve while a 9-12 restriction should degrade. Directly testable on cached
   sweep output, no GPU needed.
3. **Answered in section 8, and it overturned the framing.** Coordinate ablation
   does nothing; direction ablation removes the backdoor entirely.
4. Why is `lf`'s peak at layer 10 rather than 12, with a relative direction norm of
   2.18, double every other attack? It is the only attack whose direction *shrinks*
   over the last 2 blocks.
5. If SAM makes the backdoor less rank-1 (section 9), how many directions does it
   take? Removing the top-r subspace of the clean-versus-triggered difference for
   r = 1, 2, 4, 8 would measure that, and the r at which ASR collapses is a
   quantitative "how spread out" number rather than a binary one.
6. Direction ablation costs 0.03 to 0.08 clean accuracy on single-target attacks
   and **0.26** on `badnet_a2a` at rho 0.1. It is not a free defence, and the cost
   is worth characterising against fine-tuning-based removal.
7. The ablation reads the backdoor direction off paired clean/triggered data, which
   a defender does not have. It is a mechanism result, not a defence. Whether the
   direction can be estimated from clean data alone is the question that would turn
   it into one.
