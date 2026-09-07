# Geometry of the backdoor direction in the residual stream (H28, H29, H30, H36, H38)

## Question

The companion paper's account is that a backdoor in a ViT is a linear direction in
the residual stream. If that is right, the direction has measurable properties, and
each of them is a separate claim a defender could build on. These 5 scripts test
5 of them on the same CLS features, so a result here cannot be an artifact of how
the features were extracted.

## The 5 scripts

| script | claim under test | hypothesis |
| --- | --- | --- |
| `direction_norm_analysis.py` | the direction's norm scales with poison rate | H28 prediction 1 |
| `direction_persistence.py` | the direction is carried through the residual stream with minimal rotation | H30 |
| `direction_universality.py` | different attacks on the same target class produce parallel directions | H29 |
| `cone_geometry.py` | viable directions fill a cone around the target class readout weight | H36 |
| `crystallization_vs_placement.py` | the layer where the direction crystallizes predicts the best perturbation depth | H38 |

## Running it

    python experiments/backdoor_direction/direction_norm_analysis.py
    python experiments/backdoor_direction/direction_persistence.py
    python experiments/backdoor_direction/direction_universality.py
    python experiments/backdoor_direction/cone_geometry.py
    python experiments/backdoor_direction/crystallization_vs_placement.py

The first 4 extract layer features and want a GPU. `crystallization_vs_placement.py`
is CPU only and reads `results/direction_persistence.json`, so it has to run after
`direction_persistence.py`.

## Finding

The direction exists and grows with depth, but almost every structural property
predicted for it turned out to be absent.

H28 prediction 1 is not confirmed. The norm is approximately constant from 0.5% to
10% poisoning (Spearman rho between -0.40 and +0.80, p at or above 0.200). What is
highly consistent is layer-wise growth: 159 times from layer 1 to layer 12 on all
16 checkpoints.

H30 is supported with a qualification. The direction does not persist uniformly. It
crystallizes around layers 8 to 10 with a clear S-curve in alignment to the final
layer, so early layers are carrying something that is not yet the final direction.

H29 is refuted. Off-diagonal cosine between attacks is near 0 in every setting.
Each attack writes its own direction, so a defender who knows 1 attack's direction
learns nothing about the others.

H36 is refuted, and it corrects an earlier reading. 9 of 10 attack directions sit
87 to 91 degrees from the readout weight. Only `badnet_a2o` aligns, at 33 to 43
degrees. H16's readout alignment of 0.87 to 0.89 was measured per model, so each
direction aligns with its OWN model's readout weight and not with a shared
reference. There is no cone.

H38 splits. Crystallization depth is confirmed and attack dependent, with `blend`
at mean layer 8.2 and `badnet` at mean layer 10.4 over 8 checkpoints each. The
placement half is inconclusive and stays that way: `crystallization_vs_placement.py`
looks for per-rate `metrics.json` files inside `results/<folder>/psbd/`, a stage 2
layout the pipeline no longer writes, so its "best band" column has been empty since
the sweep output moved to `results/<folder>/psbd_metrics.json`. Testing the
correlation needs that lookup pointed at the current layout.

Hypothesis docs: `docs/hypothesis/H28-perturbation-consistency-is-margin-estimation.md`,
`docs/hypothesis/H29-cross-attack-direction-universality.md`,
`docs/hypothesis/H30-residual-persistence-phase-transition.md`,
`docs/hypothesis/H36-cone-geometry-backdoor-directions.md`,
`docs/hypothesis/H38-crystallization-depth-vs-placement.md`.
Published discussion: `docs/results/direction-norm-analysis.md`.
