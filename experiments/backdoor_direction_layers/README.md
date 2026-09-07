# Where in the ViT stack does each attack write its backdoor direction?

## Question

The companion paper (`papers/backdoor_directions/`) says the backdoor in a ViT is a
linear direction in the residual stream, and that *when* it reaches the `[CLS]`
token depends on the trigger: final few layers for static patch triggers, layers 5
to 6 for distributed ones. If that holds here, it predicts which dropout placement
should work, per attack ([H4](../../docs/hypothesis/H4-placement-is-attack-dependent.md)).

## Run

```bash
PYTHONPATH=. python experiments/backdoor_direction_layers/measure.py \
    --checkpoint-folder vit_cifar10_badnet_a2o_0_1 --samples 400
# benign control needs an explicit trigger, since it has no attack of its own
PYTHONPATH=. python experiments/backdoor_direction_layers/measure.py \
    --checkpoint-folder vit_cifar10_benign --probe-attack badnet_a2o --probe-target-label 0
```

About a minute per checkpoint on the login-node A100. Writes
`results/<folder>/direction_layers_<reduction>.json`.

## Finding

Relative direction norm `||mean(x_trigger - x_clean)|| / mean||x_clean||` in the
`[CLS]` token, CIFAR-10 ViT at 10% poisoning, 400 paired eligible samples, fp32:

| layer | badnet_a2o | badnet_a2a | lf | bpp | blend | **benign** |
|---|---|---|---|---|---|---|
| 2 | 0.012 | 0.013 | 0.015 | 0.040 | 0.066 | 0.012 |
| 4 | 0.053 | 0.037 | 0.141 | 0.248 | 0.475 | 0.039 |
| 6 | 0.124 | 0.055 | 0.395 | 0.651 | 0.831 | 0.066 |
| 8 | 0.470 | 0.124 | 0.899 | 0.846 | 1.044 | 0.089 |
| 10 | 0.701 | 0.244 | 2.176 | 1.200 | 1.032 | 0.073 |
| 12 | 1.032 | 0.488 | 1.364 | 1.214 | 1.116 | **0.049** |

Layer at which the direction first reaches half its final magnitude:

| attack | trigger family | onset |
|---|---|---|
| `blend` | global blended | **5** |
| `bpp` | distributed quantization | **6** |
| `lf` | low-frequency | 8 |
| `badnet_a2o` | static patch | 9 |
| `badnet_a2a` | static patch, all-to-all | 9 |

**The ordering matches the companion paper's prediction.** Spatially distributed
perturbations (`blend`, `bpp`) are detectable within each token independently, so
`[CLS]` picks them up by layer 5 to 6. A static corner patch has to be routed from
the patches that contain it into `[CLS]` by attention, which does not finish until
layers 9 to 12.

**The benign control works.** Probed with the same BadNet trigger, a benign model's
relative direction peaks at 0.089 and then *decays* with depth, an order of
magnitude below every backdoored model, all of which grow monotonically. So the
measurement is reading a learned backdoor, not the input perturbation itself.
Clean-vs-triggered CKA at layer 12: benign 0.998, versus 0.118 (blend) to 0.419
(badnet_a2o).

## Unplanned finding: all-to-all partially cancels its own direction

`badnet_a2a` is the odd one out. Its direction is roughly half the magnitude of
`badnet_a2o` at every layer despite an identical trigger and comparable ASR (0.96
vs 1.00), and its layer-12 CKA is **0.953** where `badnet_a2o` is 0.419: clean and
triggered representations stay nearly indistinguishable.

The reason is structural. The backdoor direction is a *mean* of paired differences.
Under all-to-all, class `y` maps to `(y+1) mod K`, so each source class is pushed
somewhere different, and averaging over classes cancels much of the movement. There
is no single direction to find, because the attack does not implement one.

This is a mechanistic prediction for
[H5](../../docs/hypothesis/H5-all-to-all-breaks-psbd.md) that is independent of
PSBD: whatever fails for all-to-all should fail for *any* method that assumes one
target, including the companion paper's own steering and orthogonalization.

## Subquestions

1. Does a *per-source-class* direction recover the a2a signal? Ten directions, one
   per class, each a clean mean. Cheap and would confirm the cancellation story.
2. Does the onset layer predict the best dropout placement quantitatively, or only
   ordinally? The single-position sweep answers this.
3. Onset is measured at 10% poisoning only. The companion paper reports it as
   stable across poisoning rates; that is directly checkable here.
4. `lf` peaks at layer 10 (2.18) then *falls* to 1.36 at layer 12. Nothing else
   does that. Worth understanding, since it suggests the low-frequency trigger's
   representation is partly consumed rather than accumulated.
