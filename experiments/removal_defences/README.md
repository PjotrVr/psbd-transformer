# Removing the backdoor without retraining (H34, H39)

## Question

Detection tells a defender which samples to drop. Removal would let them keep the
model. Both scripts here try to remove the backdoor from a trained checkpoint using
what the direction results established, and both are scored the same way: attack
success rate and clean accuracy before and after, against a random-direction control
that has to leave both alone.

## The 2 scripts

| script | intervention |
| --- | --- |
| `direction_erasure.py` | weight surgery. Orthogonalizes the MLP output weights and attention projections at layers 10 and 11 against the backdoor direction, in a known-direction mode and a blind mode using the target class readout weight as a proxy |
| `skip_scaling.py` | no weight changes. Replaces `x + branch` with `alpha * x + branch` at layers 10 and 11 through inference-time hooks |

## Running it

    python experiments/removal_defences/direction_erasure.py
    python experiments/removal_defences/skip_scaling.py

Both evaluate attack success and clean accuracy on real eval sets and want a GPU.
Output goes to `results/direction_erasure.json` and `results/skip_scaling.json`.

## Finding

Both are partially supported, and they fail on the same attacks for the same reason.

Erasure works on the weaker attacks, `lc` and `adaptive_blend`, and fails completely
on `badnet` and `blend`. The gap against H16's inference-time direction removal is
the informative part: removing the direction from the activations works where
removing it from the weights does not, which means the direction is regenerated
downstream rather than stored in the layers being edited.

Skip scaling shows the same split with a sharper boundary. `blend` has a phase
transition, holding attack success 1.000 at alpha 0.3 and dropping to 0.007 at alpha
0.1. `wanet` and `lc` die between alpha 0.5 and 0.3. `badnet` survives alpha 0.0
entirely, still at 0.995 attack success at 5% poisoning, because the branch
computations alone rebuild the direction with the skip path fully cut. No single
alpha works across attacks, so this is not a deployable defence.

Together they locate where the direction lives: not in the layer 10 and 11 weights,
and not only on the skip path.

Hypothesis docs: `docs/hypothesis/H34-direction-erasure-defense.md`,
`docs/hypothesis/H39-skip-scaling-defense.md`.
