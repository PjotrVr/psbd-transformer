# H41 -- Multi-probe PSBD defeats the adaptive attacker

**Status: SUPPORTED.**

Multi-probe AUROC 0.951 on evasive checkpoints (from single-probed 0.322),
48/56 above 0.90, 55/56 above 0.80. The union of k independent probes recovers
detection even when the attacker has collapsed the probed operator. TPR at
Bonferroni is 0.808 (below the 0.90 target because Bonferroni over-corrects:
FPR 0.068 vs target 0.25). The threshold-free AUROC is the definitive number.

The one cell below 0.80 is wanet on Tiny (0.731), where all individual
operators score below 0.83 (wanet's warp trigger is the hardest to detect
under any perturbation).

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
  probe considers suspicious, so the false-positive rate is bounded by the sum
  of individual FPRs. Bonferroni correction (per-probe quantile = target/k)
  controls this but is conservative and may cost TPR.
- **Correlated false positives.** If the k probes tend to flag the same clean
  samples, Bonferroni over-corrects and the actual FPR is lower than the bound
  at the cost of less aggressive thresholds. This is not a failure but a reason
  TPR might underperform expectations.
- **Inverted operators poison the union.** An operator whose PSU is inverted
  (backdoor samples have higher PSU than clean) flags clean samples as
  poisoned. Under the union rule this adds false positives without true
  positives. The defence against this is to exclude known-inverted operators
  from the pool, which the probe selection in H17 and H26 already does.

## Mechanism

    min_rank(x) = min_j rank_j(psu_j(x))

    where rank_j(v) = percentile of v within the clean-validation distribution
    of probe j's PSU, and j indexes the k probes.

The threshold is the q-th percentile of min_rank over clean validation, where
q = target_FPR / k (Bonferroni). A sample is flagged when its min_rank falls
below this threshold.

AUROC is computed on the min_rank scores directly, without any threshold, as
the area under the ROC curve with negated min_rank (lower rank = more
suspicious, matching the low-PSU-is-poisoned convention).

## Prior negative results to avoid

1. **Rank-averaging failed** (H17, `scratch/ensemble.py`). Mean rank across 4
   placements scored 0.900 against single-placement 0.892, a wash, because an
   inverted member (post_residual at 0.194) actively poisoned the average. The
   min rule does not average, it takes the most suspicious, so an inverted
   member that never fires contributes nothing.

2. **Simultaneous perturbation failed** (`docs/results/combined-position-test.md`,
   `both_sublayer_inputs`). Two operators in one forward pass diluted each
   other at matched sigma (delta -0.045 on Tiny). This design runs each probe
   independently at its own full-strength rate.

## Success criteria

| Quantity | Requirement | Result |
|---|---|---|
| Multi-probe AUROC on evasive checkpoints | > 0.90 | **PASS**: 0.951 mean, 48/56 above 0.90 |
| Multi-probe TPR at Bonferroni threshold | > 0.90 | **FAIL**: 0.808 (Bonferroni over-corrects, FPR 0.068 vs target 0.25) |
| Multi-probe FPR at Bonferroni threshold | within 1.5x of target | **PASS**: 0.068 (well below 0.25, room to loosen threshold) |
| No single-operator evasion below 0.70 | all evasive checkpoints | **55/56 PASS**: only wanet Tiny at 0.731 (marginal) |

The TPR shortfall is mechanical: Bonferroni targets FPR <= 0.25 but achieves
0.068, wasting the remaining false-positive budget. A less conservative
correction (Simes or empirical calibration on clean validation) would spend
that budget on TPR. The AUROC, which is threshold-free, passes at 0.951.

## Measured results

### Aggregate (56 evasive checkpoints with ASR > 0.9)

| Metric | Single probed | Best single transfer | Multi-probe (all k) | Multi-probe (transfer only) |
|---|---:|---:|---:|---:|
| Mean AUROC | 0.322 | 0.969 | 0.951 | 0.957 |
| Above 0.90 | 3/56 (5%) | -- | 48/56 (86%) | 49/56 (88%) |
| Above 0.80 | 7/56 (12%) | -- | 55/56 (98%) | -- |

### Scaling with k

| k | Mean AUROC | Above 0.90 | Above 0.80 |
|---|---:|---:|---:|
| 2 | 0.893 | 201/294 (68%) | 245/294 (83%) |
| 3 | 0.930 | 145/182 (80%) | 167/182 (92%) |
| 4 | 0.946 | 35/42 (83%) | 41/42 (98%) |

Each additional probe adds roughly +0.02 to +0.04 AUROC. Diminishing returns
set in at k=3, consistent with the probe pool having 3 mechanistically distinct
operators (dropout and token_mask at the same position are partially correlated).

### Why multi-probe AUROC (0.951) is below best single transfer (0.969)

The min-rank rule can only be as good as its best member on each sample, and
can be worse when an underperforming probe's rank is the minimum. On samples
where all transfer operators agree (most of them), the min-rank matches the
best. On samples where one transfer operator has a middling rank, the min-rank
gets pulled down. This is the FPR/TPR tradeoff: the union rule catches more
true positives (any probe suffices) at the cost of more false positives (any
probe can falsely fire).

The practical benefit is robustness: the defender does not need to know which
transfer operator is best for this attack. The multi-probe score is within
0.02 of the best single transfer on average, and never needs oracle knowledge
of the attack to pick the right operator.

### Per-architecture breakdown

| Architecture | Mean multi-probe AUROC | Above 0.90 |
|---|---:|---:|
| ViT | 0.937 | 22/28 (79%) |
| Swin | 0.964 | 26/28 (93%) |

Swin benefits more because its 3 transfer operators (token_mask@ban,
gain_scale@mno, dropout@pre_res) are more diverse than ViT's (dropout@ban,
gain_scale@mno, gaussian@bm), and dropout@ban partially leaks from token_mask
evasion on ViT (0.799 mean), pulling ViT's min-rank down on those checkpoints.

## Data

120 evasive checkpoints from H25 (2 architectures x 2 datasets x 10 attacks x
3 poison rates), each with 3 to 4 operator PSBD sweeps on disk. Analysis ran
entirely on CPU.

Raw data: `results/multi_probe_analysis.json`
Analysis script: `scratch/analyze_multi_probe.py`
Functions: `defences/psbd_metrics.multi_probe_auroc`,
`defences/psbd_metrics.multi_probe_detection`
