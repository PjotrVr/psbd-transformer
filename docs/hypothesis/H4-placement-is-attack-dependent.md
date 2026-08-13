# H4 — The best placement tracks where the backdoor direction enters the CLS token

**Status: OPEN**

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

Pending on two fronts:
- `scripts/backdoor_direction_layers/` measures the per-layer direction norm and
  TAC for each attack (not yet written).
- The single-position sweep (phase 3b) gives the per-attack position ranking.

## Subquestions

1. **Is a per-block placement sweep the real experiment here?** The current
   registry places dropout at a position in *every* block. A block-restricted
   variant (blocks 1-4 / 5-8 / 9-12) tests the layer story directly and is a
   small change to `_resolve_targets`.
2. If the entry layer predicts the best placement, does the *measured* entry layer
   from the direction analysis predict it quantitatively, or only ordinally?
3. Does the poison rate move the entry layer? The companion paper reports layer
   onset as stable across poisoning rates, which if true here means placement can
   be chosen once per attack family.
