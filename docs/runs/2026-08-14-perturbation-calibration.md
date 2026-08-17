# Perturbation calibration: disturbance per rate is not comparable across operators

Measured 2026-08-14 from the first jobs of the perturbation study, commit
`b37ceaf`. Clean-validation shift ratio (sigma) against rate, ViT-B/16 CIFAR-10.

## Why this had to be measured before anything was compared

Every operator takes a "rate", and the rates mean entirely different things.
Reporting two operators at p = 0.5 would compare a mask that removes half the
channels against additive noise at half the activation's standard deviation, and
the result would be a statement about which one perturbs harder, not about which
one detects better. The study's cross-operator rule is therefore matched sigma,
and this run confirms the rate grids actually bracket a usable sigma window.

## Measured, `vit_cifar10_wanet_0_1`

| operator / position | rate -> sigma |
|---|---|
| `channel_mask` / `post_residual` | 0.05 -> 0.257, **0.1 -> 0.870**, 0.2 -> 0.899, flat to 0.9 |
| `channel_mask` / `after_attention_residual` | 0.05 -> 0.089, 0.1 -> 0.292, **0.2 -> 0.866**, flat after |
| `channel_mask` / `before_attention_norm` | 0.05 -> 0.031, 0.3 -> 0.203, 0.5 -> 0.628, 0.9 -> 0.873 |
| `gaussian` / `pre_residual` | 0.5 -> 0.076, 0.75 -> 0.426, **1.0 -> 0.849**, 3.0 -> 0.882 |
| `gaussian` / `post_residual` | 0.2 -> 0.072, **0.3 -> 0.749**, 0.5 -> 0.895 |
| `droppath` / `pre_residual` | 0.1 -> 0.194, 0.3 -> 0.650, 0.5 -> 0.871, 0.7 -> 0.888 |
| `head_mask` / `attention_heads` | 0.3 -> 0.214, 0.5 -> 0.543, **0.6 -> 0.762** |

## Three findings

**1. Channel masking on the residual stream saturates below p = 0.1.** At
`post_residual` a rate of 0.1 already produces sigma 0.870, and every larger rate
sits flat at 0.89. The whole 0.1-to-0.9 grid is therefore past saturation and
measures one operating point ten times over. This is the same failure
[H3](../hypothesis/H3-why-post-residual-fails.md) found for dropout at the same
position, and it has the same cause: a residual-stream position perturbs the
stream once per block, so (1 - p)^12 survives the stack. A fine grid
(0.005 to 0.09) was generated and submitted as `Efine_001` for the three
residual-stream positions.

**2. Gaussian noise needs a relative standard deviation above 1.0.** At
`pre_residual`, rate 0.5 gives sigma 0.076, and it takes rate 1.0 to reach 0.849.
Noise at half the activation's own scale barely moves the prediction. Extending
that grid to 3.0 was correct, and a grid shared with the mask operators would
have put `gaussian` entirely in the dead zone and produced a spurious "noise does
not work" result. This matters for [H23](../hypothesis/H23-gaussian-noise-control.md),
whose whole point is a fair comparison between disturbance and removal.

**3. Attention heads are highly redundant.** `head_mask` reaches only sigma 0.762
at rate 0.6, and climbs slowly throughout: masking 60% of all 144 heads shifts
fewer predictions than masking 10% of the residual stream. Whatever else the head
turns out to be worth as a detection unit, this says a ViT's attention heads carry
substantial mutual redundancy, which is consistent with
[H16](../hypothesis/H16-where-the-backdoor-neurons-are.md) finding the backdoor in
residual-stream dimensions rather than in heads, and it is the first reason to
expect [H22](../hypothesis/H22-head-mask-attention-units.md) to underperform its
prediction.

Note the inverted scaling caveat on point 3: at rate 0.9 the surviving heads are
amplified 10x to preserve the expectation, so part of the redundancy measured
here is the rescaling compensating for the removal. The ranking is still valid
because every mask operator uses the same convention, but "heads are redundant"
should be stated as "heads are redundant under expectation-preserving masking"
until an unscaled ablation confirms it. The 144-head leave-one-out profile
(`psbd_head_profile.py`, jobs `headprof_001/002`) uses no rescaling and settles
this directly.

## Consequence for the study

The matched-sigma rule is not a methodological nicety here, it is load-bearing.
At a shared rate of 0.5 these seven configurations span sigma from 0.076 to 0.895,
an eleven-fold range in actual disturbance. Any table built at matched rate would
be almost entirely an artefact of that spread.
