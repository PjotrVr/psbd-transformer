# H23 — Does PSBD need capacity removed, or merely disturbed?

**Status: PRE-REGISTERED.** Written before the pilot results land. Jobs
`psbd_pilot_001/002`, submitted 2026-08-14 at commit `b37ceaf`.

## Why this is the control, not a candidate

Every other operator in the study **removes** something: a channel, a token, a
head, a branch. `gaussian` removes nothing. It adds zero-mean noise scaled to the
activation's own standard deviation, so it disturbs the representation while
leaving all capacity in place. No inverted scaling is applied, because zero-mean
additive noise already leaves the expectation unchanged.

This makes it the discriminating experiment for the framing of the entire study.
PSBD's stated mechanism is the "neuron bias effect": under dropout the clean
features collapse onto the target-biased path while the backdoor path survives,
which is a claim about **removing** capacity. If plain noise works just as well,
that mechanism is not what is doing the work, and the correct description of PSBD
is far simpler: *backdoored predictions are more robust to perturbation of any
kind*, which is a statement about margin, not about neurons.

## Why it might work

Robustness-to-perturbation is a margin property. A backdoored input sits far from
the decision boundary in the direction the trigger pushes, so any perturbation of
bounded size fails to move it, while a clean input near a boundary moves easily.
Nothing in that argument requires the perturbation to be structured. If it holds,
`gaussian` should track the removal operators closely.

## Why it might fail

- **LayerNorm renormalizes.** Most positions are immediately followed by a
  LayerNorm, which rescales to unit variance and would partly absorb additive
  noise while leaving a zeroed channel zeroed. This predicts `gaussian` should do
  relatively worse at `before_attention_norm` than at `post_residual`, and that
  contrast is itself a measurement worth having.
- Noise is **isotropic**; the backdoor is a specific direction
  ([H16](H16-where-the-backdoor-neurons-are.md): 5 to 17 of 768 dimensions,
  linearly separable). Isotropic noise puts only a 17/768 fraction of its energy
  in the subspace that matters, so it may need an implausibly large magnitude to
  disturb the backdoor at all, which is why the grid runs to 3.0.

## Prediction

`gaussian` reaches the same shift-ratio window as the removal operators but with
**lower one-sided AUROC at matched sigma**, by at least 0.03 mean across the pilot
attacks. Removal beats disturbance.

Secondary: the gap between `gaussian` and `channel_mask` is **larger at
`before_attention_norm` than at `post_residual`**, because a LayerNorm
immediately follows the former.

## What would refute it

- `gaussian` matches the removal operators within 0.02 at matched sigma. Then the
  neuron-bias framing is unnecessary for ViT, PSBD is measuring prediction margin
  under arbitrary perturbation, and the whole operator study collapses into a
  single much simpler claim. **This would be the most interesting outcome in the
  study**, and it is the reason this control is run at full scale rather than as
  an afterthought.
- `gaussian` *beats* the removal operators. Then structured masking is actively
  harmful, most likely because removal destroys clean features faster than it
  destroys the backdoor.

## Note on the confidence null

If `gaussian` matches removal, the follow-up is immediate and cheap: check it
against the confidence-only null in `defences/baselines.py`. A perturbation-free
statistic that matches a perturbation-based one would mean the perturbation earns
nothing, which is the trap
[H12](H12-psu-is-not-just-confidence.md) had to rule out for PSU itself.

## Reproduce

    python pbs/generate_perturbation_jobs.py --stage pilot --operator gaussian
    python psbd_analyze.py --all

Folders `*_gaussian`. Note the rate axis is a relative standard deviation here,
not a removal fraction, so rates are not comparable across operators and only the
matched-sigma comparison is meaningful.
