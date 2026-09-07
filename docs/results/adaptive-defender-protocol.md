# Adaptive Defender Protocol

How a defender should respond when the attacker knows PSBD and trains against
it. Based on H25 (adaptive attacker), H41 (multi-probe defence), and the
forensic analysis in `experiments/adaptive_defender/analyze.py`.

All figures below are the 2026-09-07 re-run with the corrected multi-probe
threshold rule. See "Superseded figures" at the end for what changed and why.

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
Combine scores with the min-rank rule, then threshold the combined min-rank at
the target-FPR quantile of its own clean-validation distribution.

### Why min-rank, not mean-rank

Mean-rank averaging failed (H17, `experiments/detector_ensemble/ensemble.py`): an inverted operator
(post_residual at AUROC 0.194) actively poisoned the average, giving mean 0.897
against 0.892 for the single best-known placement. Min-rank takes the most
suspicious verdict, so an inverted operator simply never fires and contributes
nothing.

### Results (56 evasive checkpoints with ASR > 0.9)

| Metric | AUROC | Above 0.90 |
|---|---:|---:|
| Probed operator (evaded) | 0.322 | 1/56 (2%) |
| Best single transfer operator | 0.954 | 45/56 (80%) |
| Multi-probe, all k operators | 0.927 | 40/56 (71%) |
| Multi-probe, transfer only (oracle) | 0.938 | 43/56 (77%) |

The multi-probe score is within 0.011 of oracle-exclude. The defender does not
need to identify which operator was evaded.

### Detection at target FPR 0.25

Two thresholding rules are reported, both defender-legal because both read
clean validation data only. Each number is labelled with the rule that
produced it.

| Pool | Rule | Threshold | TPR | FPR |
|---|---|---|---:|---:|
| all operators | calibrated | 0.25 quantile of clean-validation min-rank | 0.933 | 0.254 |
| all operators | bonferroni | literal 0.25/k on the rank scale | 0.881 | 0.187 |
| transfer only | calibrated | 0.25 quantile of clean-validation min-rank | 0.952 | 0.254 |
| transfer only | bonferroni | literal 0.25/k on the rank scale | 0.939 | 0.211 |

The calibrated rule lands on the target FPR because its quantile is measured on
the combined min-rank statistic, which already contains the union over k
probes. The bonferroni rule applies the union bound to a single probe's rank
and therefore spends only 0.187 of the 0.25 budget, costing 0.052 TPR. Deploy
the calibrated rule and quote the bonferroni one when a distribution-free
guarantee is wanted. AUROC is absent from this table because it is
threshold-free, so the values in the Results table hold under either rule.

### Scaling with k

| k | Mean AUROC | Above 0.90 |
|---|---:|---:|
| 2 | 0.889 | 67% |
| 3 | 0.935 | 82% |
| 4 | 0.965 | 93% |

The k=4 row now covers only the 28 checkpoints that still hold all 4 probe
caches, so it is measured on an easier subset than the k=2 and k=3 rows.

## Can the defender identify the evaded operator?

Four identification methods tested on 56 evasive checkpoints with ASR > 0.9:

| Method | Accuracy | Defender-legal? |
|---|---:|---|
| Lowest single AUROC | 96.4% (54/56) | No (uses labels) |
| Lowest sigma ratio at p=0.3 | 52.4% (22/42) | Needs baseline model |
| Lowest validation PSU std | 17.9% (10/56) | Yes |
| Lowest validation PSU IQR | 14.3% (8/56) | Yes |

**Conclusion: defender-legal identification does not work.** The validation
PSU std and IQR are dominated by inherent operator differences (gain_scale
always has the lowest std because it amplifies rather than removes), not by
the evasion. Chance sits between 25% and 33% depending on whether a checkpoint
holds 4 probes or 3, so both defender-legal methods are at or below chance.
Without labels or a baseline, the defender cannot identify which operator was
evaded.

The sigma ratio (needing baseline) achieves 52%, because the signal (0.66x to
0.80x suppression on the evaded operator) is noisy and overlaps with natural
model-to-model variation.

**This does not matter.** Multi-probe with all operators (0.927) is within
0.011 of oracle-exclude (0.938), and forensic exclude by val_std actually
hurts (0.849) because it incorrectly removes the strongest transfer operator.
The correct protocol is to use ALL operators and let min-rank handle the rest.

## Why multi-probe works even including the evaded operator

The min-rank rule takes the minimum across per-operator ranks. The evaded
operator's rank distribution is compressed (poisoned and clean samples both
get mid-range ranks), so it contributes noise (uniform-ish ranks between 0.3
and 0.7) rather than anti-signal. The transfer operators contribute strong
signal (poisoned samples get very low ranks, clean samples get high ranks).
The minimum of noise and signal is the signal.

The only case where including the evaded operator hurts is when the evaded
operator's rank is spuriously low on a clean sample (a false positive). The
calibrated threshold absorbs exactly this, because it is a quantile of the
combined min-rank over clean validation and so already accounts for however
often any member fires spuriously. The measured clean-test FPR of 0.254
against a target of 0.25 is that absorption working.

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
reference.** Use the `to_rank()` function from `defences/psbd_metrics.py`. The
reference must be clean validation and nothing else. Ranking against a pool
that contains the split being scored leaks that split into its own score.

**Step 5. Take the minimum rank across operators.** This is the combined
score. Lower = more suspicious.

**Step 6. Threshold at the target-FPR quantile of the clean-validation
min-rank.** With target FPR q, use quantile q of the combined min-rank over
clean validation. Do not divide q by k. The min-rank already is the union over
k probes, so dividing again corrects twice and throws away most of the
false-positive budget. Use q/k only when a distribution-free union bound is
required, and report it under that name.

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

## Superseded figures

The first version of this page reported the numbers below. Two independent
changes moved them.

The threshold rule was corrected. The old rule applied the Bonferroni quantile
target_FPR / k to the COMBINED min-rank score rather than to a single probe's
rank, which corrects twice. It reported TPR 0.808 at FPR 0.068 against a
nominal target of 0.25, and that FPR of roughly target/k is the signature of
the double correction.

Separately, the `gaussian@before_mlp` sweep caches were archived to
`*_gaussian_batchstd` by `pbs/generate_gaussian_rerun_jobs.py --archive`,
because GaussianNoise scaled its noise by a batch-reduced std and so coupled
each sample's perturbation to its batch neighbours. That drops 14 of the 56
evasive checkpoints with ASR > 0.9 from 4 probes to 3, which is what moved the
AUROC column, and it has nothing to do with the threshold. The superseded AUROC
therefore included a gaussian probe measured under that coupling, so it is not
a clean baseline either. The queued `psbd_gauss_*` GPU jobs re-measure gaussian
with a per-sample draw and the numbers should be taken again once they land.

| Quantity | Superseded | Current | Cause |
|---|---:|---:|---|
| Multi-probe TPR, all operators | 0.808 | 0.933 (calibrated) / 0.881 (bonferroni) | threshold rule |
| Multi-probe FPR, all operators | 0.068 | 0.254 (calibrated) / 0.187 (bonferroni) | threshold rule |
| Multi-probe AUROC, all operators | 0.951 | 0.927 | gaussian archive |
| Best single transfer AUROC | 0.969 | 0.954 | gaussian archive |
| Multi-probe AUROC, transfer only | 0.957 | 0.938 | gaussian archive |
| Forensic exclude by val_std, AUROC | 0.847 | 0.849 | gaussian archive |
| Lowest validation PSU std, accuracy | 8.9% (5/56) | 17.9% (10/56) | gaussian archive |
| Lowest validation PSU IQR, accuracy | 3.6% (2/56) | 14.3% (8/56) | gaussian archive |
| Mean-rank ensemble (H17) | 0.900 | 0.897 | joint-ranking leak removed |

On the 42 checkpoints whose probe pool is unchanged between the two runs, AUROC
is bit-identical under the old rule, the calibrated rule and the bonferroni
rule (0.9660 in all 3 cases), while TPR moves from 0.877 to 0.979 and FPR from
0.070 to 0.253. That isolates the threshold fix to TPR and FPR exactly.

The `experiments/detector_ensemble/ensemble.py` figure changed for a third reason. That script used
to rank each placement over `torch.cat([clean, backdoor])`, which let the
backdoor split help set the scale it was then scored on. Ranking against clean
validation instead moves the mean from 0.900 to 0.897, so the leak was worth
0.003 and the "rank-averaging is a wash" conclusion is unaffected.

## Data

Analysis script: `experiments/adaptive_defender/analyze.py`
Raw data: `results/adaptive_defender_analysis.json` (120 rows, superseded run)
Multi-probe data: `results/multi_probe_analysis.json` (120 rows, superseded run)
Transfer data: `results/adaptive_attacker_analysis.json` (120 rows)

The two superseded JSON files are the only remaining record of the k=4 probe
pool, since the `before_mlp_gaussian` caches they were built from are now
archived under `*_gaussian_batchstd`. Re-running either script with its default
`--output` overwrites them, which is worth doing once the queued `psbd_gauss_*`
jobs have restored gaussian with the per-sample noise scale.
