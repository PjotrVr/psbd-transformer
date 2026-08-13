# H9 — The pre/post gap is a perturbation-strength artifact, not a placement effect

**Status: OPEN (adversarial; this is the objection the study must survive)**

## Claim

This is the reviewer's hypothesis, written down so it can be attacked rather than
discovered in review. It says [H1](H1-pre-beats-post.md) is an illusion:

> Pre-residual and post-residual were compared at the same nominal dropout rate
> `p`, but the same `p` is a completely different intervention at those two
> positions. Pre-residual perturbs one additive branch contribution. Post-residual
> multiplies the entire residual stream by a mask, once per block, so only
> `(1-p)^12` of coordinates survive. At p=0.9 that is 1e-12 against a modest
> branch perturbation. So "pre-residual beats post-residual" may just be "the
> weaker perturbation beats the destructive one", which is a statement about
> perturbation magnitude, not about where the backdoor lives.

If H9 stands, the paper has no result: it has rediscovered that too much noise
destroys a signal.

## Why it is interesting even if refuted

Framing the sweep axis correctly is the difference between a placement study and
a hyperparameter study. And the refutation is a positive contribution in its own
right: it forces a *calibrated* comparison operator that any future
dropout-placement work has to adopt, because the same confound applies to every
position in the registry, not just these two.

## How it is being refuted

Every placement is compared at **matched clean-validation shift ratio**, not at
matched `p`. The shift ratio (paper Eq. PS: the fraction of stochastic passes
whose prediction changed) measures how much the perturbation actually disturbed
the model, rather than how much noise was nominally injected. It is computed on
clean data only, so using it as the matching variable stays defender-legal and
leaks nothing about the backdoor.

Implemented as `defences.psbd_metrics.select_rate_at_matched_shift`, reported per
placement at sigma targets 0.2 / 0.4 / 0.6 / 0.8 in
`results/<folder>/psbd_metrics.json` under `placements.<name>.matched_shift`.

**H9 is refuted if pre-residual still wins at matched sigma.** It is supported if
the placements converge once strength is equalized.

## Evidence so far

Smoke run only, `vit_cifar10_badnet_a2o_0_1`, 64 samples. Matched-sigma AUROC:

| sigma target | `pre_residual` | `post_residual` |
|---|---|---|
| 0.8 | **0.903** (p=0.4) | 0.701 (p=0.1) |
| 0.6 | **0.868** (p=0.3) | 0.701 (p=0.1) |
| 0.4 | **0.868** (p=0.3) | 0.701 (p=0.1) |
| 0.2 | **0.770** (p=0.2) | 0.701 (p=0.1) |

Pre-residual wins at every matched target, which points toward refutation. But
note the awkward part, and it is the honest reading: **post-residual cannot be
matched.** Its sigma never drops below 0.859, so all four "matched" rows collapse
onto its single least-destructive rate. The comparison is therefore not yet a
genuine match at the low targets; it is pre-residual at sigma 0.2 against
post-residual at sigma 0.86.

That is itself the H3 finding (post-residual has no low-disturbance regime), but
it means H9 is **not yet refuted at the low end**. To close it properly the sweep
needs post-residual rates below 0.1, where a genuine sigma 0.2 to 0.4 regime
might exist.

## Next step, concretely

Sweep `post_residual` at rates 0.01 to 0.09 on 2 or 3 checkpoints and check
whether a matched-sigma comparison becomes possible. Cheap (one extra job per
checkpoint) and it is the single most load-bearing follow-up in the ledger. Until
it runs, H1 should be stated as "pre-residual beats post-residual at every rate
post-residual can reach", which is true and defensible, rather than the stronger
unqualified claim.

## Subquestions

1. Is shift ratio the right calibrator, or should it be the relative L2 change of
   the pre-head representation? They may disagree; if they agree, the conclusion
   is much more robust.
2. Should clean accuracy under perturbation be a third calibrator? A defender
   would plausibly tune `p` to a tolerable clean-accuracy drop.
3. Does the matched-sigma ranking of the 9 atomic positions differ from their
   matched-`p` ranking? If it does, that is the clearest possible demonstration
   that the calibrated axis is the correct one.
