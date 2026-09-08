# H43 — the case PSBD cannot cover is covered by its own discarded tensor

**Status: SUPPORTED for the measurement, OPEN for the defence.** Predictive entropy
reaches mean AUROC **0.761** on all-to-all where PSBD reaches **0.411**, with benign
controls at chance. It is not yet deployable, because selecting between the 2 detectors
without poison labels is unsolved and 2 candidate routers are refuted.

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

## The open half

A label-free regime identifier. Refuted so far:

| router | result |
|---|---|
| shift-gap gating | 0.843 against 0.852 for always-PSU |
| shift-destination concentration | a2o 0.469, a2a 0.422, benign 0.481, no separation |

The `all_to_m` family (`(y + 1) mod m`) now interpolates between the 2 poles and 34 cells
are training, so the next evidence is whether the crossover is smooth in `log2(m)`. If it
is, the regime is a continuum and a router has to estimate a position on it rather than
pick a side.

## Reproduce

```bash
PYTHONPATH=. python experiments/all_to_all_entropy/measure.py
```
