# Out-of-sample test of the inverted H10 rule

Registered at commit `4662187af48745e2f48e9a0362c82825f5f464f7`, **before any band job for these
two checkpoints was submitted**.

## The rule under test

H10's inverted rule, derived post-hoc from 4 attacks: restrict pre-residual dropout to
the band of blocks **furthest from** where the attack's backdoor direction is written,
because PSU works by destroying clean evidence while sparing the trigger path.

## The prediction

Onsets measured first, with `scripts/backdoor_direction_layers/measure.py`, 400
paired samples, fp32, half-of-final-magnitude definition:

| checkpoint | ASR | onset layer | **predicted best band** |
|---|---|---|---|
| `vit_cifar10_wanet_0_1` | 0.962 | 7 | **blocks 9-12** |
| `vit_cifar10_adaptive_blend_0_1` | 0.926 | 12 | **blocks 1-4** |

Both attacks were excluded from the main grid (wanet fails at 1% and 5% poisoning,
adaptive_blend at 1%), so neither has been seen by any placement experiment. At 10%
poisoning both have working backdoors.

`adaptive_blend` is the sharper test: its onset of 12 is later than anything in the
derivation set, so the rule extrapolates rather than interpolates.

## Falsification

The rule is refuted if the predicted band is not the best of the three for either
checkpoint. A weaker pass would be the predicted band merely beating all-blocks.
