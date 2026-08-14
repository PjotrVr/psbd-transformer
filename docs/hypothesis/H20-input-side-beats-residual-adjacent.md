# H20 — The placement that matters is sub-layer input versus residual stream, not pre-residual versus post-residual

**Status: SUPPORTED.** +0.054 deployable AUROC, 4.0 standard errors from zero, higher
on 11 of 12 units, on a fully balanced panel.

## Claim

This project was founded on a binary: dropout **before** the residual add
(pre-residual) against **after** it (post-residual). Both have now been measured over
their own rate windows, on balanced panels, at deployable rates, and the answer is
that the distinction barely matters, because **both sit in the losing family**.

The split that does matter is between perturbing what a sub-layer **reads** and
perturbing what the residual stream **carries**:

- **input-side**: `after_embedding`, `before_attention`, `before_attention_norm`,
  `before_mlp`, `before_mlp_norm`. The perturbation lands on the tensor entering a
  sub-layer, so the sub-layer computes on corrupted evidence.
- **residual-adjacent**: `pre_residual`, `post_residual`, `before_mlp_residual`,
  `before_attention_residual`, `after_mlp_residual`, `after_attention_residual`. The
  perturbation lands on a branch output or on the stream itself.

## Evidence

Balanced panel, 13 placements x 12 units, every cell present, CIFAR-10 ViT, deployable
(adaptive-rate) AUROC, `badnet_a2a` excluded as the known PSBD failure case. Paired
per unit, so each comparison is on one problem:

| | mean | sd | se | distance from 0 | units higher |
|---|---|---|---|---|---|
| input-side minus residual-adjacent | **+0.054** | 0.046 | 0.013 | **4.0 SE** | **11 / 12** |

Balanced panel means:

| placement | family | deployable |
|---|---|---|
| `before_attention_norm` | input-side | **0.911** |
| `before_attention` | input-side | 0.905 |
| `before_mlp` | input-side | 0.892 |
| `after_embedding` | input-side | 0.887 |
| `before_mlp_residual` | residual | 0.845 |
| `pre_residual` | residual | 0.842 |
| `post_residual` | residual | 0.840 |

**Within the winning family the ordering is noise.** `before_attention_norm` beats
`before_attention` by 0.006, which is 0.41 standard errors and wins on only 5 of 12
units. Naming a single best placement is not supported; naming the family is.

**And pre-residual versus post-residual is 0.002**, the founding question, both
sitting 0.06 below the input-side family.

## Why this took so long to see

The comparison is only visible on a balanced panel. In the naive table the same
placements read as:

| placement | naive | n | balanced | n |
|---|---|---|---|---|
| `before_attention` | 0.905 | 12 | 0.905 | 12 |
| `before_attention_norm` | 0.812 | 19 | **0.911** | 12 |
| `pre_residual` | 0.732 | 23 | **0.842** | 12 |
| `post_residual` | 0.696 | 21 | **0.840** | 12 |

The placements with the most coverage look worst, because the extra checkpoints they
were run on are the hard ones (low poison rate, weak attacks). Averaging over
whatever cells exist punishes exactly the placements that were tested most
thoroughly. `scripts/balanced_panels/audit.py` exists to make this failure
mechanical to catch rather than a matter of remembering.

## Mechanistic reading

It is consistent with [H16](H16-where-the-backdoor-neurons-are.md), though not proved
by it. The backdoor is a linear direction written **into** the residual stream, and
the stream is highly redundant, so perturbing the stream is partially undone by later
blocks re-writing the direction, and by the LayerNorm renormalization that made a
pre-norm ablation unreliable in the first place. Perturbing a sub-layer's input
instead corrupts the computation that produces the write, and there is no comparably
cheap path to restore it.

That predicts something testable, and the prediction has to be stated in the right
direction. PSBD does **not** want the backdoor direction destroyed; it wants the
clean evidence destroyed while the trigger path survives. So the winning family
should **retain more** separation along the backdoor direction at matched clean
shift ratio, not less.

Tested on `vit_cifar10_badnet_a2o_0_1` with `scripts/dropout_kills_direction/`, now
recording the clean shift ratio at every rate so placements are compared on that
axis rather than on rate (the [H9](H9-strength-not-position.md) rule). Separation
retained as a fraction of unperturbed, interpolated to matched sigma:

| placement | family | sigma=0.15 | sigma=0.45 | sigma=0.70 | sigma=0.80 |
|---|---|---|---|---|---|
| `before_attention` | input | 0.397 | 0.263 | **0.219** | -- |
| `before_attention_norm` | input | 0.488 | 0.325 | **0.191** | **0.131** |
| `pre_residual` | residual | **0.512** | 0.292 | 0.153 | 0.098 |
| `post_residual` | residual | -- | 0.191 | 0.080 | 0.035 |
| `after_embedding` | input | 0.249 | 0.144 | 0.066 | 0.035 |

**Partially confirmed, with 1 clear counterexample.** At the operating point the
adaptive rule actually targets (sigma 0.7 to 0.8), the 2 leading input-side
placements retain 1.4x to 2.7x more backdoor separation than `pre_residual` and
`post_residual`, in the same order as the deployable AUROC ranking. `post_residual`
retains least, and it ranks last.

But `after_embedding` retains the *least* of all (0.066 at sigma 0.7) while ranking
4th on deployable AUROC. A single perturbation before block 1 destroys the backdoor
direction and still detects well, so the retention story cannot be the whole
mechanism.

And the ordering **inverts at low sigma**: at sigma 0.15 `pre_residual` retains the
most (0.512). So this is a property of the operating point, not a general property of
the placement family. Any version of this mechanism that does not condition on sigma
is reading the wrong regime, which is the same trap [H9](H9-strength-not-position.md)
names.

## The cheap option: 1 perturbation site instead of 12

`after_embedding` puts a single dropout before block 1. Every other placement here
installs one per block, 12 sites on ViT. On the balanced panel it is **statistically
indistinguishable from all of them**:

| after_embedding vs | delta | distance | wins |
|---|---|---|---|
| `before_attention_norm` | -0.023 | 0.9 SE | 7/12 |
| `before_attention` | -0.017 | 0.6 SE | 5/12 |
| `before_mlp` | -0.004 | 0.2 SE | 6/12 |
| `pre_residual` | **+0.045** | 1.7 SE | 10/12 |
| `post_residual` | **+0.047** | 1.7 SE | 10/12 |

And it is by far the easiest to aim, which is the quantity [H19](H19-placement-ranking-is-rate-selection.md)
says decides deployable performance:

| placement | oracle-to-deployable gap |
|---|---|
| `after_embedding` | **0.013** |
| `before_attention` | 0.040 |
| `before_attention_norm` | 0.042 |
| `post_residual` | 0.045 |
| `pre_residual` | 0.084 |

Its small mean shortfall is not spread evenly. Against `before_attention_norm` per
unit, it wins 6 of 9 on `blend`, `bpp` and `lf` with tiny margins either way, and
loses on `badnet_a2o` at 1% (0.494 against 0.615) and 10% (0.570 against 0.837),
while winning at 5% (0.855 against 0.842).

**That weakness is a ceiling, not a mis-aimed rate**, which the oracle column
settles:

| `badnet_a2o` | adaptive | oracle | gap |
|---|---|---|---|
| 1% poisoning | 0.494 | **0.562** | 0.068 |
| 5% | 0.855 | 0.908 | 0.053 |
| 10% | 0.570 | **0.601** | 0.031 |

At 1% and 10% no rate in the grid gets `after_embedding` above 0.60 on this attack.
The rule is aiming fine, the gaps are the smallest in the table; there is simply
nothing to aim at. So this is a real limitation of the placement against the static
patch trigger, not noise, and the practical reading below has to be conditioned on
it rather than hedged.

The natural explanation is that the patch trigger's direction does not reach `[CLS]`
until blocks 9 to 12 ([H4](H4-placement-is-attack-dependent.md),
[H16](H16-where-the-backdoor-neurons-are.md)), so an embedding-level perturbation has
12 blocks of redundant re-encoding to be undone by, while distributed triggers that
reach `[CLS]` by block 5 or 6 are hit directly.

**But that does not explain the 5% row.** The full rate-versus-AUROC curve (q0.25,
fractional PSU) shows 3 qualitatively different shapes for the same placement and
attack:

| poison rate | shape across rates 0.1 to 0.9 |
|---|---|
| 1% | flat at chance: 0.564, 0.508, 0.486, 0.488, 0.497, 0.488, 0.488, 0.516, 0.470 |
| **5%** | **rises monotonically: 0.516, 0.464, 0.424, 0.414, 0.444, 0.503, 0.683, 0.903, 0.963** |
| 10% | flat and weak: 0.604, 0.532, 0.506, 0.544, 0.606, 0.598, 0.575, 0.623, 0.601 |

At 5% the signal is not merely present, it is the strongest in the placement, climbing
to 0.963 at rate 0.9. At 10%, with a *stronger* backdoor and identical ASR, the curve
never leaves 0.60. It is not signal inversion: the `direction` field reads
`as_expected` on every 10% rate, and the two-sided AUROC equals the one-sided one.

More poisoning making this placement worse is the opposite of
[H8](H8-detection-scales-with-poison-rate.md)'s premise, and it is specific to
`after_embedding`, since `before_attention_norm` reaches 0.837 on the same 10%
checkpoint. So it is a placement-by-poison-rate interaction, not a property of the
checkpoint. It remains unexplained, and it is the first thing to check on a second
seed. Recorded rather than smoothed over, because it is the kind of non-monotonicity
that usually means a confound rather than a discovery.

**Practical reading.** A single dropout at the embedding matches 12-site schemes on
`blend`, `bpp` and `lf`, is 3x easier for the rate rule to aim, and costs a twelfth
of the perturbation machinery. It **cannot** detect the static patch trigger: its
ceiling there is 0.56 to 0.60 at 1% and 10% poisoning regardless of rate.

That makes it a complement rather than a replacement. The placements that do catch
`badnet_a2o` are the per-block ones, so the cheap embedding-level probe and 1
per-block placement cover disjoint failure modes, which is the same shape of argument
that motivated [H14](H14-fusion.md)'s PSBD-plus-STRIP fusion.

## Caveats

- CIFAR-10 ViT only. Swin has 3 placements swept, of which 1 is input-side, so the
  family comparison cannot yet be made there. `before_attention_norm` does win on
  Swin, which is consistent.
- 12 units, 1 seed each.
- `badnet_a2a` excluded. Including it compresses every placement toward chance and
  the family gap shrinks, because there is little signal to place.

## Subquestions

1. **Done, partially confirmed.** See the mechanistic section: the retention
   ordering matches the AUROC ordering at the operating sigma for 4 of 5 placements,
   `after_embedding` is a counterexample, and the ordering inverts at low sigma.
   Why `after_embedding` detects well while retaining nothing is the open piece.
2. Does the family effect survive on Swin once input-side placements are swept?
3. **Done, and it is competitive.** See "the cheap option" above: 1 site instead of
   12, indistinguishable from every per-block placement, 3x easier to aim, weak only
   on `badnet_a2o`. The follow-up worth running is `after_embedding` against the
   patch-trigger family specifically, across seeds, since 3 non-monotone points
   cannot support the depth explanation offered for that weakness.
