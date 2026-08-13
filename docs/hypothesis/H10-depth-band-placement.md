# H10 — Aiming dropout at the depth where an attack's direction lives beats spreading it over all blocks

**Status: OPEN.** Mechanism implemented and smoke-tested; the experiment has a
confound that must be controlled before it is run at scale.

## Claim

[H4](H4-placement-is-attack-dependent.md) established that a trigger's backdoor
direction reaches the `[CLS]` token at an attack-dependent layer: `blend` by 5, `bpp`
by 6, `lf` by 8, `badnet_a2o` not until 9. If placement works by disturbing the
backdoor path where it is being written, then restricting dropout to the *band* of
blocks around an attack's onset should beat applying it uniformly to all 12.

Concretely: `blend` and `bpp` should be best caught by an early band (blocks 1-4 or
5-8), and `badnet_a2o` by a late band (9-12).

## Why it is worth running

It is the sharpest available test of the mechanism, because it makes a *different*
prediction per attack from a single measured quantity. Confirming it would turn the
onset measurement into a placement recipe: measure where a trigger's direction
arrives (about a minute per checkpoint, no PBS needed) and aim the defence there,
without sweeping placements at all.

It also has a practical edge. A band-restricted placement perturbs fewer blocks, so
it costs the model less clean accuracy for the same detection, which matters for a
defence meant to run at inference time.

## The confound, found during the smoke test

A 48-sample smoke on `vit_cifar10_badnet_a2o_0_1`, `pre_residual`, p=0.5:

| band | AUROC |
|---|---|
| blocks 1-4 | 0.935 |
| blocks 9-12 | 0.830 |

That is the *opposite* of the prediction (this is the patch trigger, onset layer 9,
so the late band should win). But the comparison as run is not valid, for the same
reason the pre/post comparison was not:

**An early band is a stronger intervention than a late band at the same `p`.** A
perturbation injected at block 2 propagates through 10 more blocks of mixing; one
injected at block 11 propagates through 1. So "blocks 1-4 wins" may say only "blocks
1-4 perturbs harder", which is [H9](H9-strength-not-position.md) again in a new guise.

This is the second time the same confound has appeared, which is itself the lesson:
**any comparison across placements needs a strength calibrator, not a shared `p`.**

## How it must be run

- Every band compared at **matched clean-validation shift ratio**, using
  `select_rate_at_matched_shift`, not at matched `p`. The `--rates` flag exists so
  each band can be swept over whatever range reaches the common sigma targets.
- Bands: 1-4, 5-8, 9-12, plus all-12 as the reference, on `pre_residual`.
- Attacks: `blend` (onset 5) and `badnet_a2o` (onset 9) are the two ends of the
  prediction and are enough to falsify it. Add `bpp` if the first two separate.
- Report clean accuracy under perturbation alongside AUROC, since a band's practical
  value is detection per unit of damage.

Grid: 3 attacks x 3 poison rates x 4 bands = 36 jobs, plus benign controls.

## Mechanism

`defences.dropout.plug_dropout(..., block_range=(first, last))`, 1-indexed inclusive,
matching the layer numbering the latent analysis uses. Model-scope positions reject a
band outright; out-of-range bands raise. The results cache folder carries the band in
its name (`pre_residual_blocks_9_12`), so a band run and an all-blocks run cannot
overwrite each other, and `psbd_analyze.py` picks them up as distinct placements with
no changes.

Verified end to end: two bands swept on a real checkpoint, written to separate cache
folders, both read back by stage 2.

## Subquestions

1. If an early band wins even at matched sigma, the mechanism story is wrong and
   something else explains the placement effect. What?
2. Does the best band track onset layer *continuously* if the bands are made finer
   (pairs of blocks rather than groups of four)?
3. Is a single block enough? `--block-range 9 9` is already supported, and a
   one-block placement would be the cheapest possible version of this defence.
4. Does the best band for an attack transfer across poison rates and across SAM rho?
   If it does, it is a property of the trigger, which is what makes it useful.
