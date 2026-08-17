# H25 — An adaptive attacker can hide from PSBD, but only from the probe it trained against

**Status: CONFIRMED.** Both parts of the claim hold.

Part 1 (evasion works): the attacker collapses probed AUROC from 0.952 to 0.322
(mean delta -0.613), while preserving ASR (mean delta -0.004). The CA tax is
real: mean -4.8 pts, exceeding the 2-point success criterion, so the evasion is
not free.

Part 2 (evasion is probe-specific): transfer operators the attacker never saw
still detect at mean AUROC 0.887. 84% of transfer cells above 0.8, 64% above
0.9. gain_scale is the strongest transfer operator (ViT mean 0.955, Swin 0.935).

Matrix: 120 jobs, 2 architectures (ViT, Swin) x 2 datasets (CIFAR-100, Tiny) x
10 attacks x 3 poison rates. Lambda = 1.0, 15 epochs, batch 48, k = 3 passes.
Each evasive checkpoint swept with 4 operators.
- ViT evades against: token_mask @ before_attention_norm
- Swin evades against: dropout @ before_attention_norm

Full data: [adaptive-attacker-analysis.md](../results/adaptive-attacker-analysis.md)

## Claim

Two parts, and the second is the one worth publishing either way.

1. **An attacker who knows PSBD can defeat it.** Adding a hinge penalty to the
   training loss that raises poisoned samples' prediction shift onto the clean
   distribution should drive detection AUROC toward 0.5 while keeping ASR high.
2. **The evasion is probe-specific.** A model trained to look normal under
   dropout at one position should still be detectable by a *different*
   perturbation operator or position, because it was never optimised against
   those. If so, the operator family built in H21, H22, H23, H26 and H27 is not a menu of
   alternatives, it is a defence in depth an attacker must evade all of at once.

## Mechanism

    original form
        L = L_CE(f(x), y) + lambda * ReLU( mean_{x in C} PSU(x) - mean_{x in P} PSU(x) )

    restated
        train normally, plus a hinge that is zero once the poisoned samples shift
        at least as much as clean ones on average

The hinge rather than a symmetric distance is deliberate: `|mean_C - mean_P|`
would also punish poisoned PSU for rising above clean, spending capacity for no
evasion benefit and risking a separation readable from the other side.

## Why it should work

PSBD reads exactly one scalar per sample, and that scalar is differentiable with
respect to the weights. A defence whose entire decision rests on a differentiable
statistic that the attacker can compute during training is, in principle,
trainable against. Nothing about PSU is hidden from the attacker: the placement
registry is public, the rate rule is published, and the statistic is Equation 2.

## Why it might fail

- **The backdoor and the statistic may not be separable.** PSU is low for
  poisoned samples *because* the shortcut is robust, and the shortcut being
  robust is what makes ASR high. If robustness cannot be reduced without
  weakening the backdoor, the attacker faces a real trade-off and the penalty
  buys evasion only at the cost of ASR. That would be a strong positive result
  for PSBD.
- **The penalty may degenerate.** The cheapest way to close a gap between two
  means is to move either one. If clean PSU falls to meet poisoned PSU rather
  than the reverse, the attacker has changed the model's overall robustness
  rather than hidden anything, and the gap alone cannot tell the two apart. Both
  group means are logged every epoch for this reason.
- **Estimator mismatch.** The attacker trains against a k = 2 or 3 estimate of an
  expectation the defender evaluates at a different k and a different rate. The
  penalty may fit the attacker's particular noise realisation rather than the
  statistic.

## Success criteria, all three required together

| Quantity | Requirement | Result |
|---|---|---|
| ASR | stays above 0.9 | PASS (56/120, mean delta -0.004) |
| clean accuracy | within 2 pts of baseline | FAIL (mean -4.8 pts, 3/56 pass) |
| PSBD AUROC at targeted probe | falls toward 0.5 | PASS (mean 0.322 from 0.952) |

The evasion succeeds on ASR and probed AUROC but pays a real CA tax. An attacker
willing to accept 3-8 points of CA loss can defeat the probed operator. Whether
a 5-point CA drop is detectable in practice is a deployment question, not a
method question.

## The control

`--evade-psbd --evade-weight 0` runs the full adaptive code path with a zero
penalty. At weight 0 the update is **bit-identical** to a plain cross-entropy
step (max parameter drift 0.00e+00), verified in unit tests.

## The transfer table (measured)

### ViT CIFAR-100 (ASR > 0.9)

Evaded against token_mask @ before_attention_norm.

| Attack | Rate | token_mask(e) | dropout(e) | gain_scale(e) |
|---|---:|---:|---:|---:|
| badnet_a2o | 1% | 0.048 | 0.621 | 0.991 |
| badnet_a2o | 5% | 0.004 | 0.574 | 0.997 |
| badnet_a2o | 10% | 0.005 | 0.695 | 0.996 |
| blend | 1% | 0.680 | 0.924 | 0.999 |
| blend | 5% | 0.807 | 0.962 | 0.995 |
| blend | 10% | 0.012 | 0.598 | 0.998 |
| adaptive_blend | 5% | 0.034 | 0.925 | 0.947 |
| adaptive_blend | 10% | 0.039 | 0.901 | 0.941 |
| bpp | 1% | 0.671 | 0.820 | 0.974 |
| lf | 5% | 0.060 | 0.883 | 0.989 |
| lf | 10% | 0.014 | 0.893 | 0.983 |

gain_scale is untouched (mean 0.977, all above 0.895). dropout at the same
position partly transfers (mean 0.810), because dropout and token_mask share
the placement but differ in perturbation mechanism.

### Aggregate (56 rows with ASR > 0.9)

| | Probed | Transfer |
|---|---:|---:|
| Mean AUROC | 0.322 | 0.887 |
| Above 0.8 | 7/56 (12%) | 129/154 (84%) |
| Above 0.9 | 3/56 (5%) | 99/154 (64%) |

### Why transfer fails

The evasion penalty optimises against one specific Jacobian (the one computed
through the probed operator's forward graph). A different operator produces a
different Jacobian, so the gradient never touches the features that operator
reads. This is the same mechanism H16 observed: the backdoor direction's
coordinate indices relocate almost completely under a change in the optimizer
(Jaccard 0.03 to 0.08 between Adam and SAM), while ASR stays constant. The
evasion does something similar: it moves the representation's response to one
probe without moving its response to others.

## Cost

The adaptive step retains `passes + 1` full ViT-B/16 activation graphs for the
backward, so peak memory scales with `(passes + 1) x batch`. Batch 64 with
passes 3 exhausted a 40GB card, which is what `psbd_evade_smoke` establishes a
safe setting for before the matrix runs. Roughly 4x normal training wallclock.

One implementation note worth keeping: the first version computed the clean
forward twice, once for the PSU base term and once for cross-entropy. That extra
retained graph was on its own enough to cause the OOM. The two now share one
forward.

## Increasing the dropout rate does NOT counter the adaptive attacker

The PSBD paper (Appendix Section 7) suggests that increasing the dropout rate
counters the adaptive attacker. This does not hold when the attacker trains
against the operator family rather than a fixed rate.

A full rate sweep across all available rates (0.05 to 0.9) on ViT CIFAR-100
shows that the evasive model's AUROC at the probed operator is below 0.05 at
every rate for badnet_a2o, blend 10%, and adaptive_blend. The oracle-best rate
for the evasive model is 0.024-0.053 on badnet, 0.018 on blend 10%, 0.040-0.158
on adaptive_blend.

Exception: blend at 1% and 5% partially resists the evasion at high rates
(0.777 at rate 0.60, 0.949 at rate 0.70). Blend's distributed trigger modifies
every pixel, making PSU suppression harder without changing the whole input
response. But even here, the baseline achieves 0.980-0.989 at the same rates.

## CA cost of evasion

The evasion penalty competes with cross-entropy for model capacity, producing a
consistent clean accuracy drop of 1.5 to 7.8 points (mean 4.8). Only 3 of 56
successful attacks stay within the 2-point criterion. The CA cost does not depend
on the attack or poison rate in an obvious way, suggesting it is a property of
the penalty mechanism itself rather than a per-attack interaction.

## A measurement that came out of the unit tests

PSU is **identically zero at random initialisation**: a near-uniform softmax
barely moves under dropout, so `P_c` is the same with and without it, and the
statistic only exists once the model is confident. It rose from 0.0000 to 0.12
over 60 warmup steps on a toy model. This is a mechanical reason for the paper's
instruction to score a late-stage model rather than an early one, and it means
the evasion penalty has nothing to act on during the first epochs of training.
