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

That predicts something testable: input-side placements should degrade the backdoor
direction's magnitude at the final layer more than residual-adjacent ones at matched
shift ratio. `scripts/dropout_kills_direction/` measures exactly this and has only
been run on `pre_residual` and `post_residual`, both from the losing family.

## Caveats

- CIFAR-10 ViT only. Swin has 3 placements swept, of which 1 is input-side, so the
  family comparison cannot yet be made there. `before_attention_norm` does win on
  Swin, which is consistent.
- 12 units, 1 seed each.
- `badnet_a2a` excluded. Including it compresses every placement toward chance and
  the family gap shrinks, because there is little signal to place.

## Subquestions

1. Run `scripts/dropout_kills_direction/` on an input-side placement at matched
   shift ratio. The mechanistic prediction above is falsifiable and cheap.
2. Does the family effect survive on Swin once input-side placements are swept?
3. `after_embedding` is input-side to the whole stack and reaches 0.887 with the
   smallest oracle-to-deployable gap on the balanced panel, **0.013** against 0.027
   for the next best. A single perturbation before block 1 being competitive with
   per-block schemes, and the easiest of all to aim, would be a much simpler defence
   than anything currently proposed here.
