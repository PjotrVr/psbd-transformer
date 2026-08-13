# H4 — The best placement tracks where the backdoor direction enters the CLS token

**Status: SUPPORTED.** The layer ordering holds, and the placement consequence now has direct evidence.

## Claim

There is no single best dropout position. The best position for an attack is the
one that sits where that attack's backdoor direction is still being written into
the residual stream, and the companion paper
(`papers/backdoor_directions/`) shows that location is attack-dependent:

- **Static patch triggers** (BadNet, and by extension Blend): the direction only
  reaches the `[CLS]` token in the **final few layers**, because early layers have
  not yet unified trigger information across patches.
- **Distributed / stealthy triggers** (BPP, LF, WaNet, SSBA): the direction
  reaches `[CLS]` **by layer 5 or 6**, because the perturbation is detectable
  within each token independently.

## Prediction

Ranking the 9 atomic positions by AUROC produces a *different* ranking for
`badnet_a2o` and `blend` than for `bpp` and `lf`. Stronger and more falsifiable:
restricting dropout to late blocks should help patch triggers more than
distributed ones, and restricting it to middle blocks should do the reverse.

Refuted if one placement dominates uniformly across all 5 attacks.

## Why it is interesting

It is the bridge between the two papers, and it turns PSBD from a recipe into
something with a predictive theory behind it. If placement really tracks the
direction's entry layer, then measuring the entry layer (cheap, `analysis/`
already implements it) *predicts* the best placement without running the sweep.
That is a much more useful claim than any single winning position.

It also suggests a per-attack adaptive defence, which is awkward for a defender
who does not know the attack, and that tension is itself worth reporting.

## Evidence

**The layer half is confirmed.** `scripts/backdoor_direction_layers/` measured the
relative direction norm in `[CLS]` at every layer, CIFAR-10 ViT at 10% poisoning,
400 paired eligible samples, fp32. Layer at which the direction reaches half its
final magnitude:

| attack | trigger family | onset |
|---|---|---|
| `blend` | global blended | 5 |
| `bpp` | distributed quantization | 6 |
| `lf` | low-frequency | 8 |
| `badnet_a2o` | static patch | 9 |
| `badnet_a2a` | static patch, all-to-all | 9 |

Exactly the ordering the companion paper predicts: distributed perturbations are
visible within each token independently and reach `[CLS]` early, while a corner
patch has to be routed there by attention and does not arrive until layers 9 to 12.
Spread is 4 layers between the extremes, which is large enough to act on.

The benign control confirms this is reading a learned backdoor rather than the
input perturbation: probed with the same trigger, its relative direction peaks at
0.089 and *decays* with depth, against 1.0 to 2.2 growing monotonically for every
backdoored model.

**The placement half now has evidence.** The pre-residual advantage over
post-residual, on the full grid, is not uniform. It tracks the trigger family:

| attack | onset | pre-post AUROC gap |
|---|---|---|
| `blend` | 5 | +0.055 |
| `bpp` | 6 | +0.024 |
| `lf` | 8 | +0.030 |
| `badnet_a2o` | 9 | **+0.226** |

Placement barely matters for the three attacks whose direction is established by
layer 8: both placements reach 0.89 to 0.99. Placement matters enormously for the
static patch trigger, the one still being assembled in the last few blocks, where
post-residual collapses to 0.50 to 0.67 and pre-residual holds 0.69 to 0.89.

Read with [H3](H3-why-post-residual-fails.md): post-residual masks the whole stream
once per block, and a trigger that only reaches `[CLS]` at layers 9 to 12 has almost
no depth left in which to be redundantly re-encoded.

Caveat: the ordering is not monotone in onset among the three easy attacks
(`blend` at layer 5 has a larger gap than `bpp` at 6 or `lf` at 8). The defensible
statement is patch-versus-distributed, not a continuous function of onset layer.

## Subquestions

1. **A block-restricted placement is now the sharpest available test, and it does
   not exist yet.** The registry places dropout at a position in *every* block.
   Restricting to blocks 1-4 / 5-8 / 9-12 would predict, from the table above,
   that `blend` and `bpp` are best caught by an early-block placement and
   `badnet_a2o` by a late-block one. That is a falsifiable, attack-specific
   prediction and a small change to `_resolve_targets`. It is the single highest
   value follow-up in this ledger.
2. If the entry layer predicts the best placement, does the *measured* entry layer
   from the direction analysis predict it quantitatively, or only ordinally?
3. Does the poison rate move the entry layer? The companion paper reports layer
   onset as stable across poisoning rates, which if true here means placement can
   be chosen once per attack family.
