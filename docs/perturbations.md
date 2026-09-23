# Perturbations

What PSBD injects, where it injects it and what each choice turned out to
measure. Every number here is read from `paper/tables/basis_ranking.tex` and
`paper/tables/staircase_operators.tex`, regenerated on 2026-09-23 from the 69
clearing panel cells at the adaptive 0.8 rule, fractional PSU, quantile 0.25.
Regenerate before quoting: these move when the panel grows.

## The vocabulary, and why it has 2 axes

A **position** is where the perturbation is injected, named for a tensor
boundary inside every block. An **operator** is what gets injected there. A
**placement** is the pair, and the pair is what a cache directory under
`results/<folder>/psbd/` is named for.

The 2 axes are kept separate because they do not reduce to each other. Holding
the operator at token masking and moving from the attention input to the MLP
input costs 0.101 AUROC, 0.927 against 0.826. Holding the position at the
attention input and swapping token masking for channel masking costs 0.044, and
for Gaussian noise 0.123. Neither axis explains the other, which is why the basis
is a grid rather than a list.

`defenses/operators.py` holds the operators, `models/positions.py` holds the
positions in `POSITION_REGISTRY`, and `OPERATOR_POSITIONS` records the 2 pairs
that are only meaningful at specific positions.

## The operators

### Dropout

Zero individual coordinates independently, each with probability p. Divide the
survivors by 1 minus p so the expectation is unchanged.

```
y = (m * x) / (1 - p),   m ~ Bernoulli(1 - p),  elementwise
```

This is the original method's operator and the only one the PSBD paper uses. On a
ResNet basic block there is 1 natural place for it and the paper put it there.

**What it probes.** Whether the decision depends on any particular coordinate. A
coordinate is 1 of the 768 numbers describing a token, so dropping 30% of them
leaves every token present with a noisier description.

**Measured.** 0.859 at the attention input, 0.826 at the MLP input, 0.818 after
both residual adds, which is the published placement and the paper's baseline.
Its floor is 0.222 and it inverts on 12 cells.

**Why it is interesting.** It is the control that makes every other number
meaningful, because it is what the published method does. It also shows that the
site matters more than the operator's identity: the same dropout moves 0.041
between the attention input and the residual stream.

### Token masking

Zero whole tokens, keeping every channel of the survivors, with the same
inverted scaling.

```
y[:, t, :] = (m[t] * x[:, t, :]) / (1 - p),   m[t] ~ Bernoulli(1 - p)
```

Token 0 is the class token and is never masked. It is the classifier's only read
point, so dropping it destroys the prediction rather than perturbing it, and the
statistic would then measure a broken forward pass.

**What it probes.** Whether the decision depends on any particular token. A token
is the unit attention routes, so masking a token removes it from every attention
pattern of that block.

**Measured.** 0.927 at the attention input, the best in the basis. 0.927 at the
attention branch output, statistically its twin. 0.826 at the MLP input. 0.761 on
the residual stream after the attention add, which is the worst token-mask
reading and the 24th of 27 placements. Floor 0.418, which is the best floor of any
placement, against 0.091 for channel masking at the same site.

**Why it is interesting.** It is the winner, and the reason it wins is the
paper's mechanism. A backdoor direction is not aligned with any coordinate axis,
so a coordinate-level perturbation cannot remove the shortcut while a token-level
one can. The 3 readings 0.927, 0.826 and 0.761 are the same operator at 3 sites
1 sublayer apart, and the spread between them is larger than the spread across
operators at any single site.

### Channel masking

Zero whole channels across all tokens, in groups.

**What it probes.** The complement of token masking. Token masking removes a
token's whole description, channel masking removes 1 coordinate of every token's
description at once.

**Measured.** 0.883 at the attention input against 0.927 for token masking, and a
floor of 0.091 against 0.418.

**Why it is interesting.** It is the cleanest test of the axis-alignment claim.
If the backdoor lived in a few coordinates, removing whole coordinates would beat
removing whole tokens. It does not, and the floor collapse says the failure is
concentrated rather than uniform.

### Gaussian noise

Add isotropic Gaussian noise scaled to the activation's own standard deviation.
It removes no capacity, which is the point.

**What it probes.** Whether the effect of the masking operators comes from
removing information or from disturbing the activation at all. Noise is the
no-removal control.

**Measured, and this is the most interesting row in the table.** 0.804 at the
attention input, which is before that sublayer's LayerNorm. 0.700 at the MLP
input before its LayerNorm, on the 36 cells that carry it. 0.871 at the MLP input
after its LayerNorm, on all 71.

**Why it is interesting.** The 3 numbers are the LayerNorm. Noise injected before
a normalization is renormalized away, so it arrives at the sublayer weakened, and
it trails token masking at both sites where that happens. Noise injected after
the normalization is not rescaled and it reads 0.871, above token masking at the
same sublayer's input.

This is also the single correction the paper most needed. An earlier draft
compared token masking before a LayerNorm against noise after one and read the
sign flip as a property of the site, concluding that the site and the operator
interact. Held on the same side of the normalization the operator effect has the
same sign at both sites and the interaction cannot be separated from 0. What
reverses is the normalization side.

### Rademacher noise

The same isotropic covariance as Gaussian, drawn from plus or minus 1 instead of
a normal, which has lower estimator variance.

**Why it exists.** It is not a candidate. It tests a prediction of the trace
estimator reading of the statistic, that only the covariance matters and not the
shape of the draw. A Rademacher reading that differs from Gaussian at the same
site would refute that reading.

### Gain scale

Multiply a LayerNorm's learned gain by a constant. Deterministic, so 1 forward
pass is exact and a k sweep writes k identical rows. `DETERMINISTIC_OPERATORS`
records that.

**Measured.** 0.896 at the MLP norm output, 5th of 27, on 70 cells.

**Why it is interesting, and why it is handled carefully.** It carried an earlier
headline of +0.258 which is **withdrawn**: it read the winner at a clean shift
ratio of 0.95 to 0.98 and the baseline at 0.65 to 0.76, so the disturbance gap
was the size of the reported effect. At a matched shift ratio it beats the
published placement by -0.007. It is still 5th in the current ranking, which is a
real result, and it is a member of the probe union.

### Scale up

Multiply the input pixels, a port of the published SCALE-UP detector's
perturbation so that all 3 perturbation families, input, activation and
parameter, sit in 1 registry. It cannot be built from a rate alone because it
needs the dataset's normalization constants.

**Measured.** 0.756, 26th of 27, the weakest measured placement.

**Why it is interesting.** It is the comparison that says the gain is not just
"perturb something". An input-space perturbation of a published detector sits at
the bottom of the same ranking the activation-space probes head.

### Head masking

Zero whole attention heads, on the concatenated per-head outputs. Only meaningful
at `attention_heads`, which `OPERATOR_POSITIONS` enforces, because anywhere else
its 64-wide groups are arbitrary channel blocks.

**What it probes.** Whether the backdoor is carried by a small number of heads.

### DropPath

Zero a whole branch output for a whole sample, which is stochastic depth. The
block becomes the identity for that sample instead of computing a corrupted
version of its function. Only meaningful at the 2 branch-output positions.

**Why it is interesting.** It is the residual-native perturbation and the only
operator that removes a whole sublayer's contribution rather than damaging it.

### Token substitution, new on 2026-09-23

Replace whole tokens with the token at the same position from another sample in
the batch, taken by rolling the batch axis, instead of zeroing them. No inverted
scaling, because a substituted token is already drawn from the right distribution
and scaling would inflate it.

**What it probes.** The same thing token masking probes, without the
off-manifold component. A zeroed token is an input no training image ever
produced, so part of the shift token masking causes is the network reacting to an
input it has no calibration for.

**Why it is interesting, and why it is the first of the new probes to run.** It
is a strictly better posed version of the operator that already wins. If it
separates at least as well, the headline is about removing the trigger's content.
If it separates worse, part of the current headline is the off-manifold artifact,
which is a result worth having before publication rather than after review.

**Not yet measured.** Unit tested in `tests/test_operators_new_probes.py`.

### Contiguous block masking, new on 2026-09-23

Zero a patch-aligned rectangle of the token grid whose area is the requested
rate, rather than a random subset. Falls back to independent dropping when the
token count is not a square grid.

**What it probes.** Trigger geometry. Independent dropping removes a share of a
local trigger's tokens in proportion to the rate and degrades the shortcut
gradually. A rectangle either covers the trigger or misses it, which turns that
into a high-variance removal, and the statistic already averages over k passes.

**The prediction that makes it a control.** It should do nothing for a global
trigger such as Blend or SIG, where there is no compact region to cover. A
reading that helps on both local and global triggers would mean the geometry is
not what it is measuring.

**Not yet measured.** Unit tested.

## The positions

15 on ViT, 13 on Swin, 3 on the ResNet control. Every one is a named tensor
boundary reached through a forward hook, so the model's own dropout modules stay
off and no trained weight is touched. `plug_dropout` attaches a fresh module and
`unplug_dropout` removes it by handle.

| position | the tensor | before or after a LayerNorm |
|---|---|---|
| `input_pixels` | the image | before everything |
| `after_embedding` | the patch embedding output, once before the block stack | before |
| `before_attention_norm` | the attention sublayer's input | **before** its norm |
| `attention_norm_out` | the same stream after the attention LayerNorm | **after** |
| `before_attention` | the attention module's own input | after |
| `attention_heads` | the concatenated per-head outputs | inside attention |
| `before_attention_residual` | the attention branch output, before the add | after |
| `after_attention_residual` | the stream between the attention add and its 2 consumers | after |
| `before_mlp_norm` | the MLP sublayer's input | **before** its norm |
| `mlp_norm_out` | the same stream after the MLP LayerNorm | **after** |
| `before_mlp` | the MLP module's own input | after |
| `mlp_neurons` | the MLP hidden units | inside the MLP |
| `before_mlp_residual` | the MLP branch output, before the add | after |
| `after_mlp_residual` | the stream after the MLP add, the block's output | after |
| `final_norm_out` | the stream after the final LayerNorm, what the head reads | after |

The norm column is the one to read first. It is what separates the operator
effects, and it was the confound in the earlier draft.

2 positions cannot be a hook, because the tensor they perturb is a local variable
that never crosses a module boundary. `after_attention_residual` is the stream
between the attention add and its 2 consumers, and `attention_heads` sits inside
`F.multi_head_attention_forward`. Both use a removable per-instance forward
wrapper that mutates no weights.

`post_residual` on the ResNet control is dropout after each residual connection
of the basic block, before the activation, which is the published placement. Its
ViT reading is dropout after both adds, because a ViT block has 2 adds and no
activation after them.

## Depth, and a warning about the depth-band rows

A placement can be restricted to a band of blocks. Banding the winner to blocks 5
to 8 reads 0.883 and to blocks 9 to 12 reads 0.893, against 0.927 for all 12, so
banding the winner costs about 0.04. Banding the published residual dropout to
blocks 5 to 8 reads 0.873 against 0.818 for all blocks, so banding it gains.

**The warning.** The n column of the band rows is not the n of the all-blocks
row. Blocks 1 to 4 ranks 3rd at 0.924 on **12 cells**, while the all-blocks row
is 69. Differencing those 2 AUROC numbers is not a paired comparison and it
understates the band's cost by a wide margin: on the 12 cells the band covers,
the all-blocks placement reads 0.977 rather than its panel mean. Read the paired
gain column, never the difference of the AUROC column. This is recorded as Q26 in
`docs/open-questions.md`.

## What the ranking says, in 4 sentences

Token masking at either side of the attention block is the top of the basis and
its 2 readings are within 0.001 of each other. Moving the same operator to the
MLP input costs 0.101 and moving it onto the residual stream costs 0.166, so
position dominates. At a fixed position the operator still moves the score by up
to 0.123, and which way it moves depends on the side of the LayerNorm rather than
on the site. The winner's floor of 0.418 is the best in the basis by a margin,
which matters more than its mean for a defender, because a placement with a mean
of 0.883 and a floor of 0.091 fails completely somewhere.

## Holes

- `dropout, attention output before the add` is in the declared basis with **n=0**.
  It has never been swept on any cell, so the basis has 27 declared placements and
  26 with a reading.
- Gaussian at the MLP input before its LayerNorm has **n=36** against 71 for its
  after-norm counterpart, so the cleanest statement of the LayerNorm effect rests
  on about half the panel.
- No position touches the attention map. The route the mechanism names is the one
  object the basis never probes, which is the argument in
  `docs/attack-design/improving-the-defense.md`.
- The `badnet_a2m*` multi-target staircase has never been swept at the published
  placement, so its effect on the original method's configuration is unmeasured.
