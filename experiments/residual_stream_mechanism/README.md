# Why masking beats noise at `before_attention_norm`: LayerNorm absorbs one and not the other

## Question

The placement ranking is an empirical search result. This directory asks what the residual
stream is actually doing that makes one injection point better than another, at the PSBD
paper's own operating point (clean-validation shift ratio **0.8**).

A ViT block is a read-write cycle, `x -> ln_1 -> attn -> +x -> ln_2 -> mlp -> +x`, and a
placement is a point in it. Some points are immediately followed by a LayerNorm and some are
already normalised. That is not a relabelling: LayerNorm subtracts the per-token mean and
divides by the per-token standard deviation.

## The prediction, which has no free parameters

Isotropic additive noise **inflates** the per-token standard deviation, so dividing by it puts
part of the perturbation back. Zeroing a token or a channel removes content that no rescaling
can restore. Therefore:

> at a position where a LayerNorm follows, additive noise is partly undone and masking is not;
> at a position that is already normalised, both survive equally.

This is H23's secondary prediction, recorded as **still open** since the operator study:
*"the gap between `gaussian` and `channel_mask` is larger at `before_attention_norm` than at
`post_residual` ... not yet separated from position effects."*

## Result: confirmed, directly measured

Survival ratio, relative change after the following normalisation divided by relative change
before it. 1.000 means the perturbation passed through untouched. Two checkpoints, two
datasets, blocks 1/4/6/9/12, injected sizes 0.1/0.2/0.4.

| position | LayerNorm follows | gaussian | token_mask | channel_mask |
|---|---|---:|---:|---:|
| `before_attention_norm` | **yes** | **0.808 / 0.835** | 0.998 / 1.005 | 0.966 / 0.971 |
| `before_attention` | no | 1.000 | 1.000 | 1.000 |
| `before_mlp_norm` | **yes** | **0.883 / 0.862** | 0.970 / 0.972 | 0.998 / 0.964 |
| `before_mlp` | no | 1.000 | 1.000 | 1.000 |

(`vit_gtsrb_badnet_a2o_0_1` / `vit_cifar100_blend_0_05`.)

**LayerNorm removes 12 to 19 percent of additive noise and 0 to 4 percent of masking.** At
positions that are already normalised the distinction vanishes, exactly as predicted.

## The chain from mechanism to measured detection

Each step is measured on the 67-cell equal-coverage panel at sigma 0.8.

1. **Mechanism.** `ln_1` absorbs ~19% of gaussian and ~0% of token masking (above).
2. **Operator gap, position-dependent.** Mask minus gaussian is **+0.177 AUROC**
   (CI [+0.121, +0.239], n=67) at `before_attention_norm`, where `ln_1` follows immediately,
   and **-0.065** (CI [-0.115, -0.009]) at `before_mlp`, where the tensor is already
   normalised. The gap does not merely shrink, it **reverses sign**.
3. **Same operator, different position.** `gaussian` scores **0.758** at
   `before_attention_norm` and **0.896** at `before_mlp`: **+0.139 apart by position alone**,
   with **13 of 67** cells inverted (AUROC < 0.5) against **4 of 67**.
4. **Consequence for the position-versus-operator question.** Holding position fixed at
   `before_attention_norm`, the operator range is 0.177 with gaussian and 0.076 without it;
   holding the operator fixed at `token_mask` across 5 positions, the position range is 0.117.
   So position over operator is **0.66x** with gaussian and **1.53x** without, against a
   published 1.43x. **The published claim survives; the apparent reversal was one pathological
   cell, and that cell is itself a position effect.**

Step 4 matters beyond bookkeeping. An earlier reading of this panel reported the ratio as
0.58x and concluded that the operator axis dominates. It does not. A single operator failing
at a single position, for the reason measured in step 1, moved a variance ratio by a factor of
2.3.

## What this does and does not explain

**Explains:** why `gaussian` is competitive at `before_mlp` and `mlp_neurons` but fails at
`before_attention_norm`; why masking operators are insensitive to the choice of injection
point relative to a norm; why the operator axis looks larger than it is when gaussian is
included; and H23's open secondary prediction.

**Does not explain, and these remain open:** why `token_mask` beats `channel_mask` (both
survive normalisation equally, so absorption is silent on the token-versus-channel question,
which is the trigger-locality account tested separately); why WaNet prefers residual-adjacent
placements; or why depth-banding helps `pre_residual` (+0.054, CI [+0.030, +0.081]) and hurts
`before_attention_norm_token_mask` (-0.062, CI [-0.098, -0.030]).

**Limits.** Survival is measured at matched INJECTED size, not at matched shift ratio. The
detection sweep matches on shift ratio, which compensates for absorption by injecting more;
that is the downstream consequence, not the cause, and the causal direction is the claim here.
Absorption is a property of the architecture, so it holds on a benign checkpoint too and is
not by itself a backdoor signal: it explains why an OPERATOR fails at a POSITION, not why
poisoned inputs separate from clean ones.

## Reproduce

```
PYTHONPATH=. python experiments/residual_stream_mechanism/layernorm_absorption.py \
    --checkpoint-folder vit_gtsrb_badnet_a2o_0_1 vit_cifar100_blend_0_05
```

CPU only, seconds per checkpoint, no GPU and no training.
