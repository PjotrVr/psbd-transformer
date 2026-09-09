# H47 — LayerNorm absorbs additive noise and not masking, which is why the operator ranking depends on position

**Status: SUPPORTED, and it closes H23's secondary prediction, open since the operator study.**

## The prediction, which has no free parameters

A placement is a point in a block's read-write cycle, `x -> ln_1 -> attn -> +x -> ln_2 -> mlp -> +x`.
Some points are immediately followed by a LayerNorm and some are already normalised. LayerNorm
subtracts the per-token mean and divides by the per-token standard deviation. Isotropic
additive noise INFLATES that standard deviation, so dividing by it puts part of the
perturbation back; zeroing a token or a channel removes content no rescaling can restore.
Therefore additive noise should be partly undone exactly where a LayerNorm follows, and
masking should not be.

H23 predicted this in words and recorded it as **still open**: *"the gap between `gaussian` and
`channel_mask` is larger at `before_attention_norm` than at `post_residual` ... not yet
separated from position effects."*

## Result

Survival ratio: relative change after the following normalisation over relative change before
it. 1.000 means the perturbation passed through untouched. Two checkpoints, two datasets,
blocks 1/4/6/9/12, injected sizes 0.1/0.2/0.4.

| position | LayerNorm follows | gaussian | token_mask | channel_mask |
|---|---|---:|---:|---:|
| `before_attention_norm` | **yes** | **0.808 / 0.835** | 0.998 / 1.005 | 0.966 / 0.971 |
| `before_attention` | no | 1.000 | 1.000 | 1.000 |
| `before_mlp_norm` | **yes** | **0.883 / 0.862** | 0.970 / 0.972 | 0.998 / 0.964 |
| `before_mlp` | no | 1.000 | 1.000 | 1.000 |

LayerNorm removes 12 to 19 percent of additive noise and 0 to 4 percent of masking. Where the
tensor is already normalised the distinction vanishes entirely.

## The chain to the detection numbers

Each step on the 67-cell equal-coverage panel at sigma 0.8.

1. `ln_1` absorbs about 19% of gaussian and about 0% of token masking.
2. Mask minus gaussian is **+0.177 AUROC** (CI [+0.121, +0.239], n=67) at `before_attention_norm`,
   where `ln_1` follows, and **-0.065** (CI [-0.115, -0.009]) at `before_mlp`, where the tensor
   is already normalised. The gap does not shrink, it **reverses sign**.
3. The same gaussian operator scores **0.758** at `before_attention_norm` and **0.896** at
   `before_mlp`: **+0.139 apart by position alone**, with **13 of 67** cells inverted against
   **4 of 67**.

## What this corrects

An earlier reading of this panel put the position-over-operator variance ratio at **0.58x** and
concluded the operator axis dominates, reversing H28's published 1.43x. It does not. Excluding
the single pathological cell the ratio is **1.53x**, so H28's claim stands and the apparent
reversal was gaussian failing at one position for the reason measured here. A single operator
failing at a single position moved a variance ratio by a factor of 2.3.

## Scope

Absorption is a property of the architecture, so it holds on a benign checkpoint too. It
explains why an OPERATOR fails at a POSITION; it says nothing about why poisoned inputs
separate from clean ones, and it is silent on token versus channel masking, which survive
normalisation equally.

Measured at matched INJECTED size, not matched shift ratio. The detection sweep matches on
shift ratio, which compensates for absorption by injecting more; that is the downstream
consequence, not the cause.

`experiments/residual_stream_mechanism/layernorm_absorption.py`, CPU only, seconds per checkpoint.
