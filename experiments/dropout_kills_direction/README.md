# What dropout does to the backdoor direction, per placement (H3, H20)

## Question

The placement result says PSBD works at some dropout positions and not others. The
mechanism story says why: dropout should destroy the evidence a clean prediction
rests on while leaving the trigger-to-target path intact. That claim is stated in
probability space through PSU, but it is really a claim about the residual stream,
and this measures it there.

For 1 checkpoint and 1 placement: compute the backdoor direction at the final layer
from paired clean and triggered features with no dropout, then for each rate plug
the placement and project the perturbed final-layer CLS features onto that same
reference direction. The direction is never recomputed under perturbation, because
the question is how much of THIS direction survives. A placement PSBD works at
should keep the triggered projection high while the clean one collapses. A placement
PSBD fails at should collapse both, leaving nothing to threshold.

This is the same phenomenon PSU measures, 1 step upstream, so agreement between the
2 is evidence for the mechanism rather than a restatement of it.

## Running it

    PYTHONPATH=. python experiments/dropout_kills_direction/measure.py \
        --checkpoint-folder vit_cifar10_badnet_a2o_0_1 \
        --placement pre_residual post_residual

Writes `results/<folder>/dropout_direction_survival.json`. Needs a GPU.

## Finding

The placements separate exactly as predicted. On `vit_cifar10_badnet_a2o_0_1` at
rates 0.1, 0.3 and 0.5, normalized separation under `pre_residual` reads 0.696,
0.253 and 0.065, so the direction degrades gradually and there is a usable window.
Under `post_residual` it reads 0.015, 0.000 and 0.000, so the direction is gone at
every swept rate.

The result was checked against the LayerNorm artifact that invalidated H16 section 9
and is clean. Running with `--post-ln` moves `pre_residual` to 0.778, 0.297 and
0.064 and leaves `post_residual` unchanged. The raw projections do move enormously
under LayerNorm, `post_residual` at rate 0.5 reading 26.5 before it and 0.104 after,
but `separation` divides by the pooled standard deviation and that quotient already
absorbs the rescaling. A raw mean difference here would have been as fragile as the
ablation was.

Hypothesis docs: `docs/hypothesis/H3-why-post-residual-fails.md`,
`docs/hypothesis/H20-input-side-beats-residual-adjacent.md`.
