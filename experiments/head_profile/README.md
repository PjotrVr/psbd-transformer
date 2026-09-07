# Per-unit sensitivity profiles and targeted head masking (H18, H35)

## Question

H18's claim is that the backdoor signature is the SHAPE of a per-sample sensitivity
profile rather than its magnitude. A backdoored prediction rides a robust shortcut,
so it should lose confidence when the right unit is removed and shrug off the rest,
giving a concentrated profile where a clean prediction gives a diffuse one.

H35 asks the follow-up question: if H31 named 3 specific backdoor heads, does
masking exactly those heads make a better PSBD operator than masking random ones?

## The 3 scripts

| script | what it measures |
| --- | --- |
| `profile_shape.py` | the free proxy. Each cached placement is a different way of removing capacity, so the vector of PSU across placements is a coarse sensitivity profile already on disk |
| `head_profile_analysis.py` | the real measurement. Reads the 144-head leave-one-out profiles from `psbd_head_profile.py` and tests every summary statistic against the mean |
| `targeted_head_psbd.py` | masks the 3 heads H31 identified as a deterministic PSBD operator and scores it against random head masking |

`head_profile_analysis.py` fixes each statistic's direction once in `DIRECTIONS`,
before looking at any checkpoint, and applies it identically everywhere. A score
below 0.5 is reported as that statistic failing rather than re-signed. The benign
control is the check that matters: a clean model probed with the same trigger must
sit near 0.5, or the statistic is reading prediction confidence rather than a
backdoor, which is the confound H12 had to rule out for PSU itself.

## Running it

    PYTHONPATH=. python experiments/head_profile/profile_shape.py
    PYTHONPATH=. python experiments/head_profile/head_profile_analysis.py
    PYTHONPATH=. python experiments/head_profile/targeted_head_psbd.py

The first 2 are CPU only. `targeted_head_psbd.py` runs forward passes and wants a
GPU, falling back to CPU.

## Finding

The proxy survived and the real measurement killed the claim. Across 7 checkpoints
the 144-head leave-one-out profile carries no detection signal under any summary:
`badnet_a2o` at 1% reads 0.284 on the mean, 0.213 on the max, 0.137 on the gini
coefficient, all below chance and all in the same direction, which is an inverted
detector rather than a concentration signal.

H35 is refuted separately. Masking the 3 named heads gives mean AUROC 0.58, barely
above random head masking at 0.539 and far below dropout at 0.911 or gaussian noise
at 0.950. Adding the 2 badnet-specific late heads reaches 0.60 with variance from
0.46 to 0.75 across attacks. Knowing which heads diverge does not tell you which
heads to perturb.

Hypothesis docs: `docs/hypothesis/H18-sensitivity-profile-over-units.md`,
`docs/hypothesis/H35-targeted-head-psbd.md`.
