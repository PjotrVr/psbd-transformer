# H41 -- Multi-probe PSBD defeats the adaptive attacker

**Status: SUPPORTED.**

Multi-probe AUROC 0.927 on evasive checkpoints (from single-probed 0.322),
40/56 above 0.90, 51/56 above 0.80. The union of k independent probes recovers
detection even when the attacker has collapsed the probed operator. At the
calibrated threshold the union reaches TPR 0.933 at FPR 0.254 against a target
of 0.25. The threshold-free AUROC remains the definitive number.

Two things changed since the first version of this page, and they are
independent of each other. The threshold rule was corrected (see "Superseded
figures" below), which moved TPR and FPR only. Separately, the
`gaussian@before_mlp` sweep caches were archived to `*_gaussian_batchstd` by
`pbs/generate_gaussian_rerun_jobs.py --archive`, because GaussianNoise scaled
its noise by a batch-reduced std and so coupled each sample's perturbation to
its batch neighbours. That archive covers 22 of the 120 evasive checkpoints, 14
of them inside the 56 with ASR > 0.9, which currently leaves those checkpoints
running at 3 probes instead of 4 and is what moved AUROC. Every checkpoint
whose probe pool is unchanged has a bit-identical AUROC across the two runs, so
none of the AUROC movement is attributable to the threshold fix.

The lowest cell is badnet_a2o on Tiny at 5% (0.632), and the 5 cells at or
below 0.80 are all ViT on Tiny running at k=3. The queued `psbd_gauss_*` GPU
jobs re-measure gaussian with a per-sample draw, so the k=4 pool returns once
they land and these numbers should be taken again then.

## Claim

A defender who probes with k independent (operator, position) pairs and flags a
sample when ANY single probe's PSU falls below its own threshold achieves
AUROC > 0.90 against the adaptive attacker from H25, even though the probed
operator's AUROC has collapsed to 0.322.

The probe pool is the operator family built across H21 to H27:

| # | operator | position | short name |
|---|---|---|---|
| 1 | token_mask | before_attention_norm | tm@ban |
| 2 | dropout | before_attention_norm | do@ban |
| 3 | gain_scale | mlp_norm_out | gs@mno |
| 4 | gaussian | before_mlp | ga@bm |

Each probe runs independently at its own sigma-matched rate. The combined
score is the minimum rank across probes (the most-suspicious verdict wins),
which is the union rule: a sample escapes only when every probe considers it
clean.

## Why it should work

H25 showed that evasion is probe-specific: an attacker trained against
token_mask@ban still gets caught by gain_scale@mno at 0.955 and gaussian@bm
at 0.926. The attacker's penalty optimises against one Jacobian; a different
operator produces a different Jacobian, so the gradient never touches the
features that operator reads. The union of k such probes survives unless the
attacker evades all k simultaneously, which requires k forward graphs in the
backward pass and may face conflicting gradients (token_mask wants to spread
information across patches; gain_scale wants the backdoor direction to stay
close to the clean manifold).

## Why it might fail

- **FPR inflation.** The union of k probes can flag a sample that any one
  probe considers suspicious, so a per-probe threshold set at the target FPR
  would give a union FPR bounded by the sum of the individual FPRs. Thresholding
  the combined min-rank against its own clean-validation quantile removes the
  problem, because that quantile is measured on the union statistic itself.
- **Correlated false positives.** If the k probes tend to flag the same clean
  samples, the union bound is loose. The Bonferroni rule then over-corrects and
  spends less than the available false-positive budget. This is not a failure
  but a reason its TPR underperforms the calibrated rule.
- **Inverted operators poison the union.** An operator whose PSU is inverted
  (backdoor samples have higher PSU than clean) flags clean samples as
  poisoned. Under the union rule this adds false positives without true
  positives. The defence against this is to exclude known-inverted operators
  from the pool, which the probe selection in H17 and H26 already does.

## Mechanism

    min_rank(x) = min_j rank_j(psu_j(x))

    where rank_j(v) = percentile of v within the clean-validation distribution
    of probe j's PSU, and j indexes the k probes.

Two thresholding rules are reported, both reading clean validation data only,
so both stay defender-legal:

| rule | threshold | achieved FPR |
|---|---|---|
| calibrated | the target-FPR quantile of min_rank over clean validation | lands on the target by construction |
| bonferroni | the literal value target_FPR / k on the rank scale | at or below the target, by the union bound |

The calibrated rule is the default in
`defences.psbd_metrics.multi_probe_detection`. Its threshold is measured on the
combined statistic, which already contains the minimum over k probes, so it
absorbs whatever correlation the probes happen to have without needing a
correction factor. The bonferroni rule flags a sample when any single probe
ranks it below target_FPR / k, which is the literal union bound and is
conservative.

AUROC is computed on the min_rank scores directly, without any threshold, as
the area under the ROC curve with negated min_rank (lower rank = more
suspicious, matching the low-PSU-is-poisoned convention). It is therefore the
same number under either rule.

## Prior negative results to avoid

1. **Rank-averaging failed** (H17, `experiments/detector_ensemble/ensemble.py`). Mean rank across 4
   placements scored 0.897 against 0.892 for the single best-known placement, a
   wash, because an inverted member (post_residual at 0.194) actively poisoned
   the average. The min rule does not average, it takes the most suspicious, so
   an inverted member that never fires contributes nothing.

2. **Simultaneous perturbation failed** (`docs/results/combined-position-test.md`,
   `both_sublayer_inputs`). Two operators in one forward pass diluted each
   other at matched sigma (delta -0.045 on Tiny). This design runs each probe
   independently at its own full-strength rate.

## Success criteria

| Quantity | Requirement | Result |
|---|---|---|
| Multi-probe AUROC on evasive checkpoints | > 0.90 | **PASS**: 0.927 mean, 40/56 above 0.90 |
| Multi-probe TPR at the calibrated threshold | > 0.90 | **PASS**: 0.933 |
| Multi-probe TPR at the bonferroni threshold | > 0.90 | **FAIL**: 0.881 (conservative, FPR 0.187 against target 0.25) |
| Multi-probe FPR at the calibrated threshold | within 1.5x of target | **PASS**: 0.254 against target 0.25 |
| No multi-probe cell below 0.70 | all evasive checkpoints | **55/56 PASS**: only badnet_a2o Tiny 5% at 0.632 |

The bonferroni shortfall is mechanical. It targets FPR at or below 0.25 and
achieves 0.187, leaving part of the false-positive budget unspent. The
calibrated rule spends that budget and gains 0.052 TPR for it. Both are
reported because they answer different questions: the calibrated rule is what a
defender should deploy, and the bonferroni rule is the distribution-free
guarantee.

## Measured results

Re-run 2026-09-07 with the corrected threshold rule and the probe caches
currently on disk.

### Aggregate (56 evasive checkpoints with ASR > 0.9)

| Metric | Single probed | Best single transfer | Multi-probe (all k) | Multi-probe (transfer only) |
|---|---:|---:|---:|---:|
| Mean AUROC | 0.322 | 0.954 | 0.927 | 0.938 |
| Above 0.90 | 1/56 (2%) | 45/56 (80%) | 40/56 (71%) | 43/56 (77%) |
| Above 0.80 | 5/56 (9%) | 55/56 (98%) | 51/56 (91%) | -- |

### Detection at target FPR 0.25

| Pool | Rule | TPR | FPR |
|---|---|---:|---:|
| all operators | calibrated | 0.933 | 0.254 |
| all operators | bonferroni | 0.881 | 0.187 |
| transfer only | calibrated | 0.952 | 0.254 |
| transfer only | bonferroni | 0.939 | 0.211 |

Each row is the rule that produced it. AUROC does not appear in this table
because it does not depend on the threshold.

### Scaling with k

| k | Mean AUROC | Above 0.90 | Above 0.80 |
|---|---:|---:|---:|
| 2 | 0.889 | 169/252 (67%) | 207/252 (82%) |
| 3 | 0.935 | 115/140 (82%) | 130/140 (93%) |
| 4 | 0.965 | 26/28 (93%) | 28/28 (100%) |

Each additional probe adds +0.05 then +0.03 AUROC. The k=4 column now covers only
the 28 checkpoints that still hold all 4 probe caches, so it is measured on an
easier subset than the k=2 and k=3 columns and should not be read as a clean
marginal gain.

### Why multi-probe AUROC (0.927) is below best single transfer (0.954)

The min-rank rule can only be as good as its best member on each sample, and
can be worse when an underperforming probe's rank is the minimum. On samples
where all transfer operators agree (most of them), the min-rank matches the
best. On samples where one transfer operator has a middling rank, the min-rank
gets pulled down. This is the FPR/TPR tradeoff: the union rule catches more
true positives (any probe suffices) at the cost of more false positives (any
probe can falsely fire).

The practical benefit is robustness: the defender does not need to know which
transfer operator is best for this attack. The multi-probe score is within
0.027 of the best single transfer on average, and never needs oracle knowledge
of the attack to pick the right operator.

### Per-architecture breakdown

| Architecture | Mean multi-probe AUROC | Above 0.90 |
|---|---:|---:|
| ViT | 0.888 | 14/28 (50%) |
| Swin | 0.965 | 26/28 (93%) |

Swin benefits more because its 3 transfer operators (token_mask@ban,
gain_scale@mno, dropout@pre_res) are more diverse than ViT's (dropout@ban,
gain_scale@mno, gaussian@bm), and dropout@ban partially leaks from token_mask
evasion on ViT, pulling ViT's min-rank down on those checkpoints. The ViT
column is also the one that lost the gaussian probe: every ViT Tiny cell now
runs at k=3, and all 5 cells at or below 0.80 sit there.

## Superseded figures

The first version of this page reported the numbers below. They came from a
threshold that applied the Bonferroni quantile target_FPR / k to the COMBINED
min-rank score rather than to a single probe's rank, which corrects twice. The
minimum of k ranks reaching its own target_FPR / k quantile is a far rarer
event than any one probe reaching target_FPR / k, so the achieved FPR landed
near target_FPR / k and TPR was understated by the same margin. The FPR of
0.068 against a nominal target of 0.25 is the signature of that double
correction.

| Quantity | Superseded | Current | Cause of the change |
|---|---:|---:|---|
| Multi-probe TPR, all operators | 0.808 | 0.933 (calibrated) / 0.881 (bonferroni) | threshold rule |
| Multi-probe FPR, all operators | 0.068 | 0.254 (calibrated) / 0.187 (bonferroni) | threshold rule |
| Multi-probe AUROC, all operators | 0.951 | 0.927 | gaussian archive, not the threshold |
| Best single transfer AUROC | 0.969 | 0.954 | gaussian archive |
| Multi-probe AUROC, transfer only | 0.957 | 0.938 | gaussian archive |
| Above 0.90, all operators | 48/56 | 40/56 | gaussian archive |
| Checkpoints running at k=4 | 42/56 | 28/56 | gaussian archive |

The two causes separate cleanly on the 42 high-ASR checkpoints whose probe pool
is byte-for-byte unchanged between the two runs:

| Rule | TPR | FPR | AUROC |
|---|---:|---:|---:|
| old, double-corrected | 0.877 | 0.070 | 0.9660 |
| calibrated | 0.979 | 0.253 | 0.9660 |
| bonferroni | 0.972 | 0.184 | 0.9660 |

AUROC is identical to every decimal the float carries across all 3 rules and
all 42 checkpoints, which is the direct confirmation that the threshold fix
touches TPR and FPR only. The whole of the 0.951 to 0.927 AUROC move is the
14 checkpoints that dropped from 4 probes to 3.

A caveat that has to travel with the superseded AUROC: the gaussian probe it
included was measured with the batch-coupled noise scale, and
`results/gaussian_batch_coupling.json` shows that coupling ran the backdoor
split hotter than the clean split it is compared against. So 0.951 was
partly built on a contaminated probe and 0.927 is measured without that probe
on most ViT Tiny cells. Neither is the final number.

## Data

120 evasive checkpoints from H25 (2 architectures x 2 datasets x 10 attacks x
3 poison rates). Of the 56 with ASR > 0.9, 28 currently hold 4 operator PSBD
sweeps and 28 hold 3. Analysis ran entirely on CPU.

`results/multi_probe_analysis.json` still holds the superseded run, which is
why the two sets of numbers can be compared at all. Re-running
`experiments/multi_probe/analyze.py` with its default `--output` overwrites it with
the current figures. The right time to do that is after the queued
`psbd_gauss_*` jobs restore `before_mlp_gaussian` with the per-sample noise
scale, since only then does the k=4 pool exist again and with corrected data.

Analysis script: `experiments/multi_probe/analyze.py`
Functions: `defences/psbd_metrics.multi_probe_auroc`,
`defences/psbd_metrics.multi_probe_detection`
