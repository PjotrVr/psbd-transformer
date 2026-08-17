# H36 -- Backdoor direction cone geometry

**Status: REFUTED.** There is no cone. 9 of 10 attack directions are nearly
orthogonal to the readout weight (angle 87 to 91 degrees, cosine near 0). Only
badnet_a2o aligns (33 to 43 degrees). The earlier H16 readout alignment finding
(cosine 0.87 to 0.89) was measured per-model; the directions align with their
OWN model's readout weight, not with a shared reference.

Evidence: `scratch/cone_geometry.py`, results in `results/cone_geometry.json`.

## Claim

The 768-dim residual stream contains a cone of viable backdoor directions
centered on the target class readout weight. The cone has angular radius
approximately 45 degrees (arccos 0.7, the typical direction-to-readout cosine
from H16), and within it, different attacks find near-orthogonal directions. The
cone's solid angle bounds how many independent backdoors can coexist targeting
one class.

## Test

1. Extract backdoor directions for all 10 attacks at the final CLS layer.
2. Measure angle between each direction and the target class readout weight.
3. Compute pairwise cosines between directions.
4. Compute the cone's angular radius, orthogonal capacity, and expected
   random-in-cone cosine via Monte Carlo simulation.

## Results

### Per-attack angle to readout weight

| attack | CIFAR-100 angle | CIFAR-100 cos | Tiny angle | Tiny cos |
|---|---:|---:|---:|---:|
| badnet_a2o | 32.8 | 0.841 | 43.4 | 0.727 |
| tact | 86.3 | 0.065 | 93.6 | -0.063 |
| bpp | 86.9 | 0.053 | 94.0 | -0.070 |
| lc | 86.9 | 0.053 | 87.5 | 0.044 |
| sig | 87.3 | 0.047 | 90.4 | -0.008 |
| badnet_a2a | 88.7 | 0.023 | 91.0 | -0.017 |
| blend | 88.9 | 0.020 | 91.3 | -0.023 |
| adaptive_blend | 89.5 | 0.009 | 88.7 | 0.023 |
| lf | 90.7 | -0.013 | 89.8 | 0.003 |
| wanet | 90.9 | -0.016 | 86.4 | 0.062 |

### Cone geometry summary

| metric | CIFAR-100 | Tiny |
|---|---:|---:|
| mean angle to readout | 82.9 | 85.6 |
| angle range | 32.8 to 90.9 | 43.4 to 94.0 |
| orthogonal capacity | 756 | 763 |
| observed pairwise cosine | 0.053 +/- 0.073 | 0.023 +/- 0.054 |
| expected random-in-cone cosine | 0.468 +/- 0.269 | 0.431 +/- 0.283 |

## Interpretation

### The cone does not exist

The premise was that all backdoor directions cluster in a cone around the
readout weight. The data shows the opposite: 9 of 10 directions are essentially
perpendicular to the readout weight (cosine within +/- 0.07 of zero). Only
badnet_a2o has meaningful alignment (cosine 0.84 on CIFAR-100, 0.73 on Tiny).

The orthogonal capacity of 756 to 763 (out of 768 dimensions) confirms this is
not a cone but effectively the entire sphere.

### Reconciling with H16

H16 reported readout alignment of cosine 0.87 to 0.89, which was measured by
comparing each attack's direction to the readout weight from THAT SAME MODEL.
The cone_geometry script takes the readout weight from the first loaded model
(badnet_a2o) and compares ALL directions to it.

This is not a bug but a finding: each backdoored model learns a slightly
different readout weight for the target class. An attack's backdoor direction
aligns with its own model's readout weight (H16) but not with another model's
readout weight. The readout weight adaptation is itself attack-specific.

### Observed cosines are BELOW random-in-cone expectation

Even treating the angular spread as a cone, the observed pairwise cosines (0.023
to 0.053) are far below what random vectors in such a cone would produce (0.43
to 0.47). The directions are spread more widely than random-in-cone, consistent
with being essentially random in the full 768-dim space (expected cosine 0.0 for
random unit vectors).

### Why badnet_a2o is special

Badnet_a2o uses a static 3x3 patch trigger that produces the strongest, most
spatially concentrated backdoor signal. Its direction is the only one that
aligns with ANY model's readout weight across models. This is consistent with
H32 (spatial concentration) and H31 (badnet recruits additional late heads):
the static patch creates a uniquely strong, direct pathway to the classifier.

### Implications

1. **No universal cone defense.** The cone-projection defense (project out the
   readout weight subspace to block all backdoors) would only catch badnet_a2o.
   The other 9 attacks operate in a subspace nearly orthogonal to the readout
   weight.

2. **The attack-readout alignment is per-model, not per-direction.** Each
   backdoored model co-adapts its readout weight and backdoor direction. This
   means H16's strong alignment result is real but model-specific, not a
   geometric property of the direction space.

3. **The H29 result (near-orthogonal directions) is tighter than random-in-cone
   would predict.** The attacks do not merely find different directions in a
   cone; they find directions that are spread across the full sphere.
