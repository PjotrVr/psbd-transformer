# Where is the backdoor?

Which layers carry each attack, whether the attacks share anything, whether SAM
moves it, and whether the thing that carries it is a set of neurons or a direction.

Written for [H16](../../docs/hypothesis/H16-where-the-backdoor-neurons-are.md).

**The folder is named for the question, not the answer.** The answer turned out to
be that there are no backdoor *neurons* on ViT: zeroing the top 300 TAC coordinates
of 768 leaves ASR at 1.00, while removing 1 linear direction takes it to 0.00. The
backdoor is real, causal, and rotated off the coordinate basis.

## The question

Everything before this measured PSBD's *behaviour*. This measures the object PSBD
is supposed to be reacting to. If the backdoor is a linear direction in the
residual stream, it has an address: a block, and a set of dimensions. Find the
address, and 3 things become checkable that were previously assertions.

- Is a defence tuned to 1 attack's neurons reusable on another?
- Does SAM, which changes what the loss landscape rewards, change where the
  backdoor gets put?
- Can a defender find the neurons without any triggered data at all?

## The 4 scripts

`measure.py` produces `results/<folder>/backdoor_neurons.json`, 1 per checkpoint:
per-layer TAC / relative direction norm / CKA, per-dimension top-20 and CLP
outliers, the Lipschitz and head-alignment data-free channels, and PCA/UMAP
separability scored by silhouette and 10-NN purity.

`stability.py` produces the 2 reference points that make every Jaccard in the
report interpretable: a **chance floor** from random k-of-768 draws, and a
**split-half ceiling** from ranking dimensions on 2 disjoint halves of the same
samples. Written after the first pass of results, because a low Jaccard between 2
models means nothing until you know what a high one looks like on 1 model.

`ablate.py` is the causal half, and the reason the folder's own framing changed. It
deletes things and re-measures ASR: the backdoor direction, k coordinates from the
top / bottom / at random, and random rank-1 directions as the control that removing
*some* direction is not the claim.

`report.py` aggregates and prints the 5 correlational tables.

## Running it

```bash
export http_proxy=http://10.150.1.1:3128 https_proxy=http://10.150.1.1:3128

PYTHONPATH=. python scripts/backdoor_neurons/measure.py \
    --attack badnet_a2o blend bpp lf badnet_a2a --rho "" 0_1 0_2 --samples 600
PYTHONPATH=. python scripts/backdoor_neurons/measure.py \
    --attack benign --rho "" 0_1 0_2 --samples 600

PYTHONPATH=. python scripts/backdoor_neurons/stability.py \
    --attack badnet_a2o blend bpp lf badnet_a2a benign --rho "" 0_1 0_2

PYTHONPATH=. python scripts/backdoor_neurons/ablate.py \
    --attack badnet_a2o blend bpp lf badnet_a2a benign --rho "" 0_1 0_2 \
    --max-samples 1000

PYTHONPATH=. python scripts/backdoor_neurons/report.py
```

Login node, about 25 minutes for all 18 checkpoints. No PBS job needed.

The benign checkpoint is invoked separately only because it needs `--probe-attack`
to define a trigger; `resolve_probe_attack` refuses that flag on a backdoored
checkpoint rather than silently measuring a trigger the model never saw.

## Findings

1. **Late and sharp.** Every attack peaks at block 10 to 12; benign peaks at 8 with
   a relative direction norm of 0.09 and decays. Between 5 and 17 dimensions out of
   768 are outliers at mean + 3 std.
2. **Attacks are disjoint.** Cross-attack Jaccard of top-20 TAC dimensions is 0.00
   to 0.08, against a chance floor of 0.014 and a split-half ceiling of 0.87.
   `badnet_a2o` and `badnet_a2a` share the identical trigger image and overlap at
   exactly 0.00, so the dimensions are set by the label mapping, not the trigger.
3. **SAM relocates without weakening.** The peak layer moves earlier monotonically
   in rho for all 5 attacks while benign stays at 8, and the dimension overlap with
   Adam falls to 0.03 to 0.08 at rho 0.2. ASR stays at 0.96 to 1.00 throughout and
   clean accuracy rises about 2 points.
4. **Both data-free localizers fail.** Lipschitz channel ranking correlates with
   TAC globally (Spearman about 0.6) but its top-20 overlaps TAC's top-20 at 0.038,
   near chance, and it scores *highest* on the benign model. The head-alignment
   Z > 3 rule fires on 2 of 18 checkpoints and both are wrong, 1 of them a clean
   model.
5. **The separation is linear.** A 2-component PCA reaches 10-NN purity 0.996 to
   1.000 for 4 of 5 attacks, so UMAP adds nothing. `badnet_a2a` is the exception at
   PCA 0.756 versus UMAP 1.000, because all-to-all has no single target class and
   so no single direction, which is the same structural reason it breaks PSBD.
6. **A direction, not neurons.** Removing the rank-1 backdoor direction takes ASR
   from 1.00 to 0.00 on all 4 single-target attacks, costing 0.03 to 0.08 clean
   accuracy. Zeroing coordinates never works, up to 300 of 768. Random rank-1
   directions and the benign model are both unaffected.
7. **SAM decouples the backdoor from the mean shift.** ASR after direction removal
   rises monotonically with rho (`blend` 0.00 to 0.99 to 1.00, `lf` 0.05 to 0.35 to
   0.97). Not a depth artifact: forcing the ablation to block 12, with no block
   left to recover in, gives the same answer. And not simply "spread over more
   directions" either, since rank 2 to 16 subspaces of the difference hold ASR at
   1.00 while dropping clean accuracy to 0.56.

## Caveat

Only seed 0 exists for these checkpoints, so "SAM relocates the backdoor" cannot
yet be separated from "any retraining relocates it". That control needs 1 retrain
per cell and is the first subquestion in H16.
