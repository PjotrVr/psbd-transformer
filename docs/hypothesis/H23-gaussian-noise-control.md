# H23 — Does PSBD need capacity removed, or merely disturbed?

**Status: REFUTED, and it is the most consequential result of the operator study.**
Additive noise is not worse than structured removal. It is competitive with the
best mask operators and ahead of most of them.

## Result

ViT CIFAR-10 at 10% poisoning, matched at clean-validation sigma >= 0.6, one-sided
AUROC, mean over 8 backdoored attacks (all-to-all excluded), full position sweep:

| operator / position | mean |
|---|---|
| `dropout` / `pre_residual_blocks_5_8` | 0.944 |
| **`gaussian` / `before_attention`** | **0.950** |
| `token_mask` / `before_mlp_residual` | 0.929 |
| **`gaussian` / `mlp_neurons`** | **0.918** |
| `channel_mask` / `before_attention` | 0.911 |
| `head_mask` / `attention_heads` | 0.886 |
| `dropout` / `post_residual` (published) | 0.849 |

The prediction was that removal would beat disturbance by at least 0.03 mean.
It does not. `gaussian` at `before_attention` is at the top of the table.

## The confidence-null check, which the refutation needed

If noise matched removal because the perturbation earns nothing at all, the
conclusion would be different and much worse for PSBD. It does not: the
confidence-only null in `defences/baselines.py`, scored on the same checkpoints,
averages **0.520** AUROC over 21 backdoored CIFAR-10 checkpoints with TPR at 1%
FPR of **0.000** on all but three.

So the perturbation is doing real work; what it does not need to be is a
*removal*. That distinction is the whole content of this file.

## What this costs the paper's framing

PSBD's stated mechanism is the "neuron bias effect": under dropout, clean features
collapse onto the target-biased path while the backdoor path survives. That is a
claim about **removing** capacity, and it predicts that an operator which removes
nothing should underperform. Additive isotropic noise removes nothing and matches
the best removers.

The reading that survives is simpler and weaker: **PSBD measures how far a
prediction moves under perturbation of any kind, which is a statement about margin
rather than about neurons.** A backdoored input sits far from the decision
boundary in the direction the trigger pushes, so bounded perturbation fails to
move it while a clean input near a boundary moves easily. Nothing in that argument
requires structure.

This lines up with [H22](H22-head-mask-attention-units.md) (the transformer's own
computational unit is not privileged) and
[H18](H18-sensitivity-profile-over-units.md) (per-unit sensitivity carries no
detection signal at all). Three independent results now point the same way: the
unit-level story is not what carries the method on ViT.

## Still open

The secondary prediction, that `gaussian` should do relatively worse where a
LayerNorm immediately follows, is not yet separated from position effects and
needs the full grid across datasets before it can be read.

---

## Original pre-registration, retained

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
