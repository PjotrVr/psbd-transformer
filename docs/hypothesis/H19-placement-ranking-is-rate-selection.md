# H19 — On Swin, the placement ranking is not about placement. It is about which placement the rate rule can aim at

**Status: SUPPORTED on Swin CIFAR-10 `badnet_a2o`** (6 to 8 checkpoints per
placement, Adam and SAM rho 0.05 to 0.2). Needs the same check on ViT.

## Claim

Every placement comparison in this project has ranked positions by the AUROC they
achieve. That number is a product of 2 things: how good the placement's *best*
operating point is, and how close PSBD's adaptive rate rule gets to it. Those can be
separated, and when they are on Swin, the entire ranking turns out to come from the
second.

## Method

`psbd_analyze.py` already records both per placement:

- **oracle** picks the dropout rate that maximizes AUROC. It reads the labels, so it
  is an upper bound and not a method.
- **adaptive** picks the smallest rate whose clean-validation shift ratio clears the
  target. Clean data only, so it is what a defender can actually run.

The gap between them is the cost of not knowing the right rate.

## Evidence

Swin CIFAR-10 `badnet_a2o`, mean over checkpoints:

| placement | deployable (adaptive) | oracle | gap | n |
|---|---|---|---|---|
| `before_attention_norm` | **0.926** | 0.962 | **0.036** | 6 |
| `before_mlp_residual` | 0.833 | 0.976 | 0.143 | 6 |
| `pre_residual` | 0.816 | **0.979** | 0.162 | 8 |

Read the 2 middle columns against each other. **At the oracle rate all 3 placements
are equivalent**, spanning 0.962 to 0.979, and `pre_residual` is nominally the best
of them. At the rate a defender can actually select, `before_attention_norm` wins by
0.09 to 0.11 AUROC, purely because its gap is 4x smaller.

The per-checkpoint rows show the same thing more sharply. `before_attention_norm`
hits its own oracle exactly (gap 0.000) on 3 of 6 checkpoints. `pre_residual` loses
0.34 on `swin_cifar10_badnet_a2o_0_1_sam_rho_0_15`, going 0.964 oracle to 0.625
deployable, while `before_attention_norm` on the same checkpoint loses nothing and
lands at 0.988.

### The sharpest case: a placement with no operating point at all

`pre_residual_blocks_9_12` has the best oracle AUROC anywhere in this project:
0.997 on `blend`, 0.996 on `bpp`, 0.998 on `lf`. It has **no deployable operating
point on any of 20 checkpoints**.

Restricting dropout to 4 of 12 blocks is too weak an intervention to move the
clean-validation shift ratio into range. Swept over the full rate grid on
`vit_cifar10_blend_0_1`, sigma runs 0.004, 0.004, 0.007, 0.009, 0.013, 0.021, 0.033,
0.058, **0.171** at rate 0.9. The adaptive rule needs 0.7. It never fires, so
`adaptive_rate` is null and the placement cannot be selected by a defender at all,
at any rate the grid contains.

An oracle-ranked table puts this placement first. A deployable one cannot list it.

### Re-testing H10 at the deployable rate

[H10](H10-depth-band-placement.md)'s band table is oracle AUROC throughout, which is
verifiable cell by cell (`blend` 9-12 = 0.997, `bpp` 9-12 = 0.996, `badnet_a2o` 9-12
= 0.829 all match the `oracle` field exactly). Its headline is "a band beats
all-blocks on every attack tested, 6 of 6".

**That claim survives the switch to deployable rates**, at 15 of 19 checkpoints
rather than 6 of 6. The exceptions are `blend_0_1`, `bpp_0_05`, `bpp_0_1` and
`sig_0_1`, where all-12 pre_residual is better.

**Its per-attack recommendations do not.** H10 names blocks 9-12 as best for
`blend`, `bpp` and `lf`, and blocks 9-12 is exactly the placement with no deployable
operating point. The deployable choice is blocks 5-8 or 1-4, and 5-8 wins more
often, which is what H10 concluded for a different reason.

## Why it matters

It changes what a placement study is measuring. "Which position is best" has been
treated as a question about where in the block to perturb, and on Swin it is mostly
a question about **which position has a well-behaved shift-ratio-to-AUROC curve**, so
that a rule calibrated on clean data lands near the peak. A placement whose optimum
sits on a narrow ridge is worse in deployment than one with a lower but flatter
optimum, even though the ranking by best-achievable AUROC says the opposite.

This is the same underlying defect as [H11](H11-adaptive-rate-overshoots.md), which
found the rate rule overshooting on ViT and costing about 0.09 AUROC. H11 treated
that as a constant to retune. This says the cost is **placement-dependent**, so
retuning 1 constant cannot fix it uniformly, and any ranking of placements at a
fixed rate rule is confounded by it.

It also connects to [H9](H9-strength-not-position.md), the standing hazard in this
project: no comparison across placements is valid at a shared dropout rate. H19 is
the deployment-side version. Matching on clean-validation shift ratio fixes the
*comparison*; it does not fix the fact that the deployable rule has to hit the ridge.

## Caveats

- `badnet_a2o` only. The `badnet_a2a` checkpoints sit at 0.470 to 0.532 across all 3
  placements, reproducing [H5](H5-all-to-all-breaks-psbd.md) on Swin, and carry no
  placement signal to rank.
- Unbalanced n: `pre_residual` has 8 checkpoints, the others 6, because the sweep is
  still in flight. The 6 checkpoints with all 3 placements present give the same
  ordering.
- 1 seed.

## Answered: it holds on ViT too, in a weaker form

Balanced panel, 13 placements x 15 CIFAR-10 ViT checkpoints with every cell present
(4 attacks plus `badnet_a2a`, at 1%, 5% and 10%). Unbalanced means are useless here:
across the raw set, placement n ranges from 15 to 39 because different placements
were swept over different checkpoints, so a mean over all of them compares
placements on different problems.

| placement | deployable | oracle | gap |
|---|---|---|---|
| `before_attention` | **0.827** | 0.864 | 0.037 |
| `before_attention_norm` | 0.817 | 0.868 | 0.051 |
| `before_mlp` | 0.814 | 0.850 | 0.036 |
| `after_embedding` | 0.800 | 0.826 | 0.025 |
| `before_mlp_residual` | 0.771 | 0.862 | 0.091 |
| `pre_residual` | 0.767 | 0.850 | 0.083 |
| `post_residual` | 0.766 | 0.819 | 0.053 |
| `pre_residual_blocks_5_8` | 0.761 | **0.876** | **0.116** |
| ... | | | |
| `after_attention_residual` | 0.752 | 0.782 | 0.030 |

**Best by oracle is `pre_residual_blocks_5_8`. Best by deployable is
`before_attention`**, and the oracle winner falls to 9th of 13 on the largest gap in
the table.

The difference from Swin is that on ViT the placements are genuinely unequal at the
oracle too (spread 0.094, against 0.017 on Swin). So on ViT rate selection reshuffles
the ranking; on Swin it produces the ranking outright.

## The obvious fix does not work

If the ranking is driven by rate selection, the natural repair is a
**placement-dependent** target for the shift-ratio rule, since
[H11](H11-adaptive-rate-overshoots.md) treated the target as 1 global constant to
retune. The `matched_shift` blocks already contain AUROC at targets 0.2, 0.4, 0.6
and 0.8, so this is answerable without any GPU work.

Over the same 15 shared checkpoints, mean AUROC per placement per target:

| | best target | best-single-target mean | per-placement-target mean | gain |
|---|---|---|---|---|
| 14 placements | 0.8 for 11 of 14 | **0.784** | **0.785** | **+0.002** |

**Refuted.** Tuning the target per placement buys 0.002 AUROC. 11 of 14 placements
want the same target, and the 3 that differ (`pre_residual` at 0.6,
`pre_residual_blocks_5_8` and `_9_12` at 0.4) gain almost nothing from it. So the
gap H19 identifies is not something a smarter constant closes; the placements whose
adaptive rate lands badly are not simply mis-targeted.

Caveat: the available grid is coarse (steps of 0.2) and does not contain 0.7, which
is the value H11 found best on ViT, so this rules out a *large* placement-dependent
effect rather than a small one.

A useful consistency check falls out of the same table. At matched shift ratio,
which is the comparison [H9](H9-strength-not-position.md) says is the only fair one,
the top placements are `before_mlp` (0.837), `before_attention` (0.831) and
`before_attention_norm` (0.826), with `pre_residual_blocks_5_8` at 0.783. That is
the deployable ordering, not the oracle one. 2 of the 3 ways of comparing placements
agree with each other and disagree with the oracle.

## Subquestions

1. Is the gap predictable from the shape of the AUROC-versus-shift-ratio curve
   without labels? Curvature near the selected rate is measurable from clean data
   alone, and if it predicts the gap, a defender could pick the placement as well as
   the rate.
3. Does the gap explain why SAM checkpoints looked worse? 4 of the 5 largest gaps in
   the table are SAM checkpoints. If SAM sharpens the ridge rather than lowering the
   peak, that is a cleaner statement of [H6](H6-sam-improves-detectability.md) than
   the current one.
