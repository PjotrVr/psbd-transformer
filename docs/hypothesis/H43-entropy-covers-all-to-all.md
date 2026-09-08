# H43 — the case PSBD cannot cover is covered by its own discarded tensor

**Status: SUPPORTED, and the defence half is now closed.** Predictive entropy reaches
mean AUROC **0.761** on all-to-all where PSBD reaches **0.411**, with benign controls at
chance. A label-free, parameter-free rule selects between them and gains **+0.045** mean
AUROC over always-PSU, bootstrap CI **[+0.017, +0.079]**, helping 8 cells and **hurting
none**.

## Claim

PSBD fails on all-to-all for a structural reason (H5): its premise needs a constant,
content-independent shortcut, and `(y + 1) mod K` forces the model to read the source
class. The same content dependence forces the poisoned logit to beat the **true source
class** rather than an arbitrary runner-up, so the margin is contested. That predicts
triggered inputs are measurably **less confident** under all-to-all, and predicts it as a
consequence of the very thing that breaks PSBD.

It also predicts the ASR cost. All-to-all reaches 0.842 against all-to-one's 1.000, and
that is the same contested margin seen from the attacker's side.

## Evidence

`experiments/all_to_all_entropy/measure.py`, `token_mask @ before_attention_norm`, matched
clean-validation shift ratio 0.6, one-sided, direction fixed a priori.

| group | n | `psu_ratio` | `neg_entropy` | `confidence` | `logit_margin` |
|---|---|---|---|---|---|
| all-to-one, ASR >= 0.5 | 55 | **0.891** | 0.408 | 0.405 | 0.375 |
| **all-to-all** | 12 | **0.411** | **0.761** | 0.752 | 0.728 |
| benign | 4 | 0.496 | 0.506 | 0.506 | 0.505 |

Per-cell, all 4 datasets and all 3 poison rates behave the same way; the strongest cell is
`vit_cifar100_badnet_a2a_0_01` at 0.963.

Cost: **1 deterministic forward pass**, already cached. The sweep writes the full
`(N, num_classes)` softmax to `baseline_<split>.pt` and reads only the argmax entry.

## Why this is not H15 being reopened

It is not a two-sided reading of one statistic. These are 2 different statistics, each
scored one-sided in the direction its own mechanism predicts, on the same splits. What
H15 forbids, and what is still forbidden, is choosing between them by reading the AUROC.

## The router: sign, not magnitude

Both refuted routers used a MAGNITUDE:

| router | result |
|---|---|
| shift-gap gating | 0.843 against 0.852 for always-PSU |
| shift-destination concentration | a2o 0.469, a2a 0.422, benign 0.481, no separation |

The regimes differ in **sign**, and the sign is exactly what the mechanism predicts. A
defender holds clean validation data and the suspect pool. Write

    d = mean PSU over the suspect pool - mean PSU over clean validation

Under all-to-one the poisoned subpopulation resists the probe MORE than clean data, so it
pulls the pool's mean down and `d < 0`. Under all-to-all the trigger must read the source
class first, so the backdoor pathway is more fragile, and `d > 0`. The threshold is 0,
fixed by the mechanism rather than tuned.

    use PSU if d < 0, otherwise use entropy

| rule | mean AUROC, n = 64 |
|---|---|
| always PSU | 0.819 |
| always entropy | 0.456 |
| **routed** | **0.865** |
| oracle-max (needs labels) | 0.883 |

+0.045 over always-PSU, CI [+0.017, +0.079], **98% of oracle-max**, and positive on every
leave-one-dataset-out refit (+0.035 to +0.058). It **hurts no cell**, because it only
diverts when PSU's own premise is violated.

Nothing in the rule needs the poison rate, the target class, or any poison label. The
suspect pool is what the defender was handed; the rate appears only in the script that
simulates such a pool from cached splits.

## What this still does not establish

- The gain rests on **9 all-to-all cells**. It is a large effect on a small subpanel.
- At 1% poisoning the delta is **0.000**: too few poisoned samples to move the pool's mean
  enough to flip the sign, so the router silently falls back to always-PSU there. That is
  the safe failure, but it means the all-to-all case is uncovered at 1%.
- Single seed, and the decision to test `d` was taken after inspecting the deviations,
  even though the threshold itself is a priori. A pre-registered replication on the
  `all_to_m` cells now training is the honest next test.
- `all_to_m` (`(y + 1) mod m`) interpolates the 2 poles and 34 cells are training. If the
  crossover is smooth in `log2(m)`, the regime is a continuum and a router has to estimate
  a position on it rather than pick a side. That is the sharpest falsifier available.

## Reproduce

```bash
PYTHONPATH=. python experiments/all_to_all_entropy/measure.py
PYTHONPATH=. python experiments/all_to_all_entropy/router.py
```
