# H26 — Masking whole channels beats thinning every channel a little

**Status: REFUTED as stated. The consolation claim is PROVISIONAL.**

> The refutation is panel-backed: `channel_mask` loses to dropout at every matched
> position, averaged over 8 attacks. That stands.
>
> The consolation ("but it wins at 1% on `badnet_a2o`") is a single cell on a
> single attack and does **not** meet the panel standard. It is retained below as
> a lead only.

## Result

At 10% poisoning, compared to `dropout` at the **same** position (matched sigma
0.6, mean over 8 backdoored attacks), `channel_mask` loses everywhere:

| position | `dropout` | `channel_mask` |
|---|---|---|
| `before_attention` | 0.967 | 0.911 |
| `before_attention_residual` | 0.935 | 0.873 |
| `before_mlp` | 0.935 | 0.858 |
| `before_mlp_residual` | 0.909 | 0.848 |
| `after_mlp_residual` | 0.907 | 0.848 |

The prediction was a win on at least 4 of 5 attacks. It loses at every matched
position. Removing a whole feature is not better than thinning all of them, at
least where the signal is already strong.

**But at 1% poisoning on the hardest checkpoint it is the best number in the
study.** `channel_mask` / `before_attention_norm` on `badnet_a2o` at 1% scores
**0.861** against `dropout`'s 0.839 at the same position, and 0.194 for the
published configuration. Overall at 1% the two are a tie (0.928 against 0.934).

So the honest statement is narrower than the claim: structured removal buys
nothing when the backdoor is easy to see and may buy a little when it is not.
Whether that low-rate edge survives the other three datasets is exactly what the
full grid is for; on one checkpoint it is an observation, not a result.

---

## Original pre-registration, retained

## Mechanism

`nn.Dropout` on a ViT activation samples an independent Bernoulli mask per
`(token, channel)` pair. On a 197x768 activation that is 151k independent
decisions, and by concentration almost every channel survives in almost every
sample: at p = 0.5 a channel is fully removed from a sample with probability
2^-197, which never happens. Dropout therefore never removes a feature. It
attenuates all of them by a random amount and adds variance.

`channel_mask` draws one Bernoulli per `(sample, channel)` and shares it across
tokens, so a masked channel contributes nothing to that sample anywhere. 768
decisions instead of 151k, and removal actually means removal.

## Why it might work

[H16](H16-where-the-backdoor-neurons-are.md) puts the backdoor in 5 to 17
residual-stream **dimensions**, and shows the clean/triggered split is linearly
separable in that stream (PCA 10-NN purity 0.999). A channel mask acts on exactly
that axis: it either removes a backdoor-carrying dimension for a sample or it does
not. Dropout can only ever attenuate it, and attenuating a dimension the backdoor
depends on strongly is close to leaving it alone once inverted scaling restores
the expectation.

If PSBD's premise is that the backdoor survives perturbation better than clean
features do, then the perturbation should be one that can actually destroy a
feature. Dropout, on a wide transformer activation, cannot.

## Why it might fail

- The backdoor may be **token-local** rather than channel-local. A patch trigger
  occupies specific tokens, and removing a channel everywhere hits the clean
  features in all 197 tokens just as hard as it hits the trigger in 4 of them, so
  the ratio that PSU measures may not improve at all.
- **Higher variance per sample.** 768 coin flips instead of 151k means the
  realised disturbance varies much more between the k = 3 passes. With PSBD's
  k = 3 the PSU estimate could get noisier faster than it gets sharper, which is
  the same variance problem [H24](H24-monte-carlo-passes.md) tests directly.
- LayerNorm follows most positions and renormalizes scale, which may partly undo
  a channel removal in a way it does not undo diffuse thinning.

## Prediction

At matched clean-validation shift ratio, `channel_mask` gives a **higher one-sided
AUROC than `dropout` at the same position** on at least 4 of the 5 pilot attacks,
with the largest gain on the attacks H16 localizes most sharply (`blend`, `bpp`,
`lf`, whose direction norms peak hardest).

## What would refute it

- `channel_mask` within +/-0.02 of `dropout` at matched sigma on most attacks:
  the elementwise/structured distinction does not matter and dropout was never
  the problem.
- `channel_mask` wins only at positions where LayerNorm does not immediately
  follow: the effect is about normalization, not about structure.
- The benign control moves away from 0.5: the operator is damaging the model
  rather than probing a backdoor, and any apparent gain is an artefact.

## Reproduce

    python pbs/generate_perturbation_jobs.py --stage pilot --operator channel_mask
    python psbd_analyze.py --all

Compare `before_attention_norm_channel_mask` against `before_attention_norm` at a
common `matched_shift` target in `results/<folder>/psbd_metrics.json`. Never
compare at a common rate ([H9](H9-strength-not-position.md)).
