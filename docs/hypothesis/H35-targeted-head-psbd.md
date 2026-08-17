# H35 -- Targeted head perturbation as PSBD operator

**Status: REFUTED.** Masking the 3 backdoor heads (L5H0, L5H10, L6H3) from H31
as a deterministic PSBD operator produces AUROC 0.47 to 0.63 (mean 0.58), only
marginally above random head masking (H22: 0.539) and far below dropout (0.911)
or gaussian noise (0.950). Adding the 2 badnet-specific late heads brings the
5-head mean to 0.60 with high variance across attacks (0.46 to 0.75). Targeted
head masking is not a viable PSBD operator.

Evidence: `scratch/targeted_head_psbd.py`, results in
`results/targeted_head_psbd.json`.

## Claim

Masking the 3 heads identified in H31 as carrying backdoor-divergent attention
patterns should preferentially disrupt backdoor computation (large prediction
shift for backdoor samples) while leaving clean predictions mostly intact,
outperforming random head masking (H22: 0.539 AUROC) and potentially matching
dropout-based PSBD.

## Test

1. Create `MultiFixedHeadMask` that zeros a set of (block, head) pairs via
   `masked_attention_forward`.
2. Compute PSU: base confidence minus perturbed confidence, divided by base.
3. AUROC using negated PSU (low PSU = poisoned, one-sided).
4. Two configurations: 3 heads (L5H0, L5H10, L6H3) and 5 heads (adding
   badnet-specific L9H7, L10H9).
5. Test on 9 checkpoints across CIFAR-100 (10%, 5%) and Tiny (10%).

## Results

| checkpoint | 3-head AUROC | 5-head AUROC |
|---|---:|---:|
| cifar100 badnet_a2o 10% | 0.608 | 0.527 |
| cifar100 blend 10% | 0.552 | 0.544 |
| cifar100 wanet 10% | 0.560 | 0.649 |
| cifar100 lc 10% | 0.466 | 0.455 |
| cifar100 adaptive_blend 10% | 0.631 | 0.713 |
| cifar100 badnet_a2o 5% | 0.585 | 0.519 |
| cifar100 blend 5% | 0.619 | 0.754 |
| tiny badnet_a2o 10% | 0.588 | 0.535 |
| tiny blend 10% | 0.613 | 0.696 |
| **mean** | **0.580** | **0.599** |

### Baselines for comparison

| operator | AUROC | source |
|---|---:|---|
| random head_mask | 0.539 | H22 |
| dropout (token_mask) | 0.911 | H27 |
| gaussian noise | 0.950 | H23 |
| **targeted 3-head** | **0.580** | this |
| **targeted 5-head** | **0.599** | this |

## Interpretation

### Why targeted head masking fails as a PSBD operator

The PSBD mechanism measures PREDICTION SHIFT: how much the model's confidence
drops under perturbation. For this to separate clean from backdoor samples,
the perturbation must:
1. Substantially shift clean predictions (so clean PSU is large), AND
2. Minimally shift backdoor predictions (so backdoor PSU is small).

Masking 3 of 144 heads (2.1% of attention capacity) does not produce enough
perturbation strength to create a meaningful prediction shift in EITHER clean
or backdoor samples. The clean PSU is small because the model compensates
through the remaining 141 heads. The backdoor PSU is similarly small because
the backdoor signal rides the residual stream (H34), not just the attention
computation.

This is the same failure mode as H22 (random head masking), just with slightly
better targeting. The fundamental problem is the perturbation STRENGTH, not the
targeting: heads are a redundant computation unit, and masking a small number
of them (whether random or targeted) does not displace enough signal to
produce a measurable prediction shift.

### The 5-head configuration is inconsistent

Adding L9H7 and L10H9 (badnet's late heads) sometimes helps dramatically
(blend 5%: 0.619 to 0.754) and sometimes hurts (badnet 10%: 0.608 to 0.527).
The improvement on blend is likely because masking 5 heads is simply a larger
perturbation (3.5% of attention capacity), not because the extra heads are
specific to blend's backdoor circuit. The degradation on badnet may reflect
that the extra heads ARE part of badnet's backdoor pathway, and removing them
removes backdoor information that was producing a signal.

### The fundamental lesson

H31 identified POPULATION-level divergence in these heads: their attention
distributions differ between clean and backdoor inputs when averaged over
many samples. But H35 and H40 both show that this population-level divergence
does not translate into per-sample detection:
- H40: entropy in these heads does not separate samples (AUROC 0.48 to 0.54).
- H35: masking these heads does not produce differential prediction shift
  (AUROC 0.47 to 0.63).

The backdoor heads are identifiable and real, but the per-sample signal they
carry is too weak relative to the noise floor for sample-level detection. The
backdoor's primary pathway is the residual stream (H30, H34), and the attention
heads are a secondary contributor.

## Connection to other hypotheses

- **H22 (random head masking):** 0.539 AUROC. Targeted masking improves by
  only +0.04, confirming that head masking as a class of perturbation is too
  weak for PSBD.
- **H31 (backdoor heads):** The 3 heads are real population-level backdoor
  markers but not exploitable for sample-level detection.
- **H40 (attention entropy):** Another attempt to exploit the same 3 heads
  for detection, also fails (AUROC near 0.5).
- **H23 (gaussian noise):** 0.950 AUROC shows that adding noise to the full
  representation is far more effective than targeted head ablation. The
  perturbation needs to be broad, not precise.
