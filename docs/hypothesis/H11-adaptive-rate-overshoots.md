# H11 — PSBD's adaptive dropout-rate rule overshoots on ViT

**Status: SUPPORTED.** 12 of 12 checkpoints, for the placements that matter, and the
fix is free.

## Claim

PSBD selects its dropout rate by an adaptive rule: take the smallest `p` at which the
clean-validation shift ratio reaches a high value, "0.8 in our experiments"
(`papers/PSBD/sec/4_method.tex:185`). That target was tuned on ResNet-18. On ViT-B/16
it lands past the optimum, so following the published rule leaves detection on the
table.

## Evidence

Mean AUROC over the 12 backdoored CIFAR-10 checkpoints with a working attack
(`badnet_a2a` excluded, it fails at every placement), at the 25th-percentile
threshold. "Adaptive" is the rate the paper's rule selects; "oracle" is the best rate
in hindsight, an upper bound that reads the poison labels and is reported only as a
ceiling.

| placement | adaptive | oracle | loss | overshoots the optimum |
|---|---|---|---|---|
| `before_mlp_residual` | 0.845 | 0.940 | **0.095** | 12 / 12 |
| `pre_residual` | 0.842 | 0.926 | **0.084** | 12 / 12 |
| `post_residual` | 0.840 | 0.885 | 0.045 | 11 / 12 |
| `before_attention_norm` | 0.911 | 0.952 | 0.042 | 5 / 12 |
| `before_attention` | 0.905 | 0.945 | 0.040 | 9 / 12 |
| `after_mlp_residual` | 0.826 | 0.856 | 0.030 | 6 / 12 |

The direction is one-sided: where the rule misses, it almost always picks a rate
**larger** than the optimum, never smaller. For the two best-known placements it
overshoots on every single checkpoint.

Measuring where the optimum actually sits: across the 12 checkpoints, the oracle rate
for `pre_residual` corresponds to a clean-validation shift ratio of **0.697** (range
0.338 to 0.787). The paper's rule targets 0.800.

## Why this is the most directly useful result here

The fix costs nothing and needs no backdoor knowledge. The shift ratio is measured on
clean validation data only, so retargeting the rule from 0.8 to about 0.7 is exactly
as defender-legal as the original, and recovers roughly 0.09 AUROC on the placements
where the loss is largest.

It also explains a pattern visible throughout this study: the adaptive and oracle
rates differ by consistently one step of the rate grid (`pre_residual` adaptive 0.5
against oracle 0.4, over and over). That is not noise in rate selection, it is a
mistuned constant.

## Why the constant does not transfer

The shift ratio is a measure of how much the perturbation disturbed the model, and
how much a given `p` disturbs a network depends on its depth. ResNet-18 has 8 residual
adds; ViT-B/16 has 24. A target calibrated so that "the model is disturbed enough but
not destroyed" on one architecture will sit on the far side of that balance on a
deeper one. This is the same depth argument that put `post_residual`'s whole operating
window an order of magnitude below the inherited rate grid
([H9](H9-strength-not-position.md)).

## Caveat

The oracle is an oracle. The 0.697 figure is derived from rates chosen by looking at
the answer, so it cannot be quoted as "the correct target" without validating it on
held-out attacks. The honest claim is: **0.8 is measurably too high on ViT, and the
optimum sits near 0.7.** Confirming a specific replacement value needs a split where
the target is fitted on some attacks and tested on others.

## Reproduce

`results/<folder>/psbd_metrics.json`, comparing `placements.<name>.adaptive` against
`placements.<name>.oracle`, and reading `rates[i].shift_ratio.validation` at the
oracle rate.

## Subquestions

1. Fit the target on two attacks, test on the other two. Does a single retuned
   constant generalize, or is the optimal sigma attack-dependent?
2. Does the optimal sigma scale with depth in the way the argument predicts? Swin has
   24 blocks against ViT's 12, so it should sit lower still. That is the cleanest
   available test of the mechanism behind this.
3. Is sigma even the right calibrator? Clean accuracy under perturbation is equally
   defender-legal and might have a more stable optimum.
