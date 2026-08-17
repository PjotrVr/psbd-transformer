# Adaptive Defender Protocol

How a defender should respond when the attacker knows PSBD and trains against
it. Based on H25 (adaptive attacker), H41 (multi-probe defence), and the
forensic analysis in `scratch/adaptive_defender_analysis.py`.

## The threat

The adaptive attacker adds a hinge penalty to training that matches poisoned
samples' PSU onto the clean distribution under one specific perturbation
operator. This collapses that operator's AUROC from 0.952 to 0.322 while
preserving ASR (mean delta -0.004) at a cost of 4.8 points CA.

The evasion is probe-specific: the penalty optimises against one Jacobian, and
a different operator produces a different Jacobian. Transfer operators the
attacker never saw still detect at mean AUROC 0.887.

## The defence: multi-probe PSBD

Run k independent perturbation operators, each at its own sigma-matched rate.
Combine scores with the min-rank rule: flag a sample when ANY probe's PSU
rank falls below the Bonferroni-corrected threshold.

### Why min-rank, not mean-rank

Mean-rank averaging failed (H17, `scratch/ensemble.py`): an inverted operator
(post_residual at AUROC 0.194) actively poisoned the average, giving mean
0.900 against single-operator 0.892. Min-rank takes the most suspicious
verdict, so an inverted operator simply never fires and contributes nothing.

### Results (56 evasive checkpoints with ASR > 0.9)

| Metric | AUROC | Above 0.90 |
|---|---:|---:|
| Probed operator (evaded) | 0.322 | 1/56 (2%) |
| Best single transfer operator | 0.969 | -- |
| Multi-probe, all k operators | 0.951 | 48/56 (86%) |
| Multi-probe, transfer only (oracle) | 0.957 | 49/56 (88%) |

The multi-probe score is within 0.006 of oracle-exclude. The defender does not
need to identify which operator was evaded.

### Scaling with k

| k | Mean AUROC | Above 0.90 |
|---|---:|---:|
| 2 | 0.893 | 68% |
| 3 | 0.930 | 80% |
| 4 | 0.946 | 83% |

Diminishing returns at k=3. Three mechanistically distinct operators suffice.

## Can the defender identify the evaded operator?

Four identification methods tested on 56 evasive checkpoints with ASR > 0.9:

| Method | Accuracy | Defender-legal? |
|---|---:|---|
| Lowest single AUROC | 96.4% (54/56) | No (uses labels) |
| Lowest sigma ratio at p=0.3 | 52.4% (22/42) | Needs baseline model |
| Lowest validation PSU std | 8.9% (5/56) | Yes |
| Lowest validation PSU IQR | 3.6% (2/56) | Yes |

**Conclusion: defender-legal identification does not work.** The validation
PSU std and IQR are dominated by inherent operator differences (gain_scale
always has the lowest std because it amplifies rather than removes), not by
the evasion. Without labels or a baseline, the defender cannot reliably
identify which operator was evaded.

The sigma ratio (needing baseline) achieves 52% (barely above the 25% chance
level for 4 operators), because the signal (0.66x to 0.80x suppression on the
evaded operator) is noisy and overlaps with natural model-to-model variation.

**This does not matter.** Multi-probe with all operators (0.951) is within
0.006 of oracle-exclude (0.957), and forensic exclude by val_std actually
hurts (0.847) because it incorrectly removes the strongest transfer operator.
The correct protocol is to use ALL operators and let min-rank handle the rest.

## Why multi-probe works even including the evaded operator

The min-rank rule takes the minimum across per-operator ranks. The evaded
operator's rank distribution is compressed (poisoned and clean samples both
get mid-range ranks), so it contributes noise (uniform-ish ranks between 0.3
and 0.7) rather than anti-signal. The transfer operators contribute strong
signal (poisoned samples get very low ranks, clean samples get high ranks).
The minimum of noise and signal is the signal.

The only case where including the evaded operator hurts is when the evaded
operator's rank is spuriously low on a clean sample (a false positive). This
is bounded by Bonferroni: at target FPR 0.25 with k=4, each operator uses
quantile 0.0625, so any one operator's false-positive contribution is small.

## Practical protocol

**Step 1. Run all operators.** At minimum, use 3 mechanistically distinct
operators. The tested pool:

| Operator | Position | Mechanism |
|---|---|---|
| token_mask | before_attention_norm | Zeros whole patches (spatial structure) |
| dropout | before_attention_norm | Zeros random elements (unstructured) |
| gain_scale | mlp_norm_out | Amplifies LayerNorm output (amplitude) |
| gaussian | before_mlp | Adds random noise (continuous perturbation) |

**Step 2. Sigma-match each operator independently.** Find the rate where
clean-validation shift ratio reaches 0.6. Each operator at its own rate.

**Step 3. Compute per-sample PSU at each operator's matched rate.**

**Step 4. Rank each operator's PSU against its own clean-validation
reference.** Use the `to_rank()` function from `defences/psbd_metrics.py`.

**Step 5. Take the minimum rank across operators.** This is the combined
score. Lower = more suspicious.

**Step 6. Threshold at the Bonferroni-corrected quantile of clean-validation
min-rank.** With k operators and target FPR q, use quantile q/k.

**Step 7. Flag samples below threshold as poisoned.**

This protocol requires no knowledge of the attack, no knowledge of whether
the model is evasive, and no baseline model for comparison. It runs on the
same infrastructure as single-operator PSBD with k additional forward-pass
sweeps.

## What if the attacker evades 2+ operators?

The H25 data shows that evading one operator costs 4.8 points CA and 4x
training wallclock. Evading k operators requires k forward graphs retained
in the backward pass, scaling memory as O(k * batch * model_size). At k=2
with batch 48 and ViT-B/16, this would need 80+ GB (double the current
40 GB allocation). At k=3 it is infeasible on current hardware without
gradient checkpointing and batch reduction.

More importantly, different operators may require conflicting representation
changes. Token_mask evasion spreads information across patches; gain_scale
evasion keeps the backdoor direction close to the clean manifold. Satisfying
both simultaneously may be fundamentally harder than satisfying either alone.

This is untested, and a natural extension would be to train an attacker
against 2 operators and see whether (a) the training converges, (b) ASR
survives, and (c) the remaining operators still detect.

## Data

Analysis script: `scratch/adaptive_defender_analysis.py`
Raw data: `results/adaptive_defender_analysis.json` (120 rows)
Multi-probe data: `results/multi_probe_analysis.json` (120 rows)
Transfer data: `results/adaptive_attacker_analysis.json` (120 rows)
