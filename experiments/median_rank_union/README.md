# Does the median-rank union survive a real adaptive attacker?

## Question

The multi-probe union is this project's answer to an adaptive attacker. Measured
on the evasive checkpoints, the attacker's best move is not to neutralize the
probed position but to **invert** it, and a min-rank union takes the most extreme
evidence across probes, so it should adopt the inversion.

A median-rank union needs a majority to be wrong instead. On synthetic probes with
a known answer, median dominated min everywhere until the attacker owned a
majority: 0.833 against 0.621 with 1 inverted of 3, and 0.967 against 0.792 with 1
inverted of 5.

**That validation was not sufficient, and this experiment is why.**

## Method

All 120 evasive (`_evade_*`) checkpoints that carry a PSBD cache, 119 with 4
placements each. Cached PSU only, no GPU and no model. Each probe is read at the
rate whose clean validation shift ratio is nearest 0.6, so probes are compared at
a matched disturbance rather than a shared nominal rate.

## Result: the fix does not work on real data

| rule | mean AUROC | median AUROC |
|---|---:|---:|
| best single probe, chosen with hindsight | **0.767** | |
| min-rank union | 0.702 | 0.723 |
| **median-rank union** | **0.673** | 0.671 |

Min beats median on **65 of 120** checkpoints, median beats min on 55.

Restricted to the 110 checkpoints where a probe **is** actually inverted, which is
the case the median was introduced to fix, min still wins: **0.683 against 0.654**.
On the 10 with no inverted probe, min 0.910 against median 0.884.

Both achieve their FPR target (0.255 and 0.259 against 0.25), so this is a
detection difference and not a calibration artifact.

## Why the synthetic result did not transfer

The synthetic inverted probe was built symmetrically, clean shifted one way and
backdoor the other, which makes it assign aggressively low ranks to **clean**
samples. A min rule takes the most extreme evidence, so it inherits those as false
positives, and a median averages them away.

A real inverted probe does not do that. The healthy probes still produce extreme
low ranks for backdoor samples, and min captures exactly that while median pulls
toward the middle of a set that includes weaker probes. The property that made
min fragile in simulation is not the property real inverted probes have.

## The larger negative result

**Neither union beats the best single probe**: min is -0.065 and median -0.094
against it. The defender does not know which single probe is best without the
labels, so a union remains defensible on threat-model grounds, that an attacker
must evade every probe rather than guess which one was deployed. But it cannot be
claimed as an AUROC improvement, and it is not one here.

## What to do with the code

`reduction="median"` stays in `psbd.scores.multi_probe_score` as a documented
option with this result attached. It is not the default and it should not be
recommended. The synthetic tests in `tests/test_operator_semantics.py` are correct
about the mechanism they construct and are labelled as constructions, not as
evidence the defence works.

## Reproduce

    python experiments/median_rank_union/measure.py

About 6 minutes on CPU.
