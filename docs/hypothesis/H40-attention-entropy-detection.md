# H40 -- Per-sample attention entropy as detection feature

**Status: REFUTED as stated.** Attention entropy in the 3 backdoor heads (L5H0,
L5H10, L6H3) does not separate backdoor from clean samples for most attacks
(AUROC 0.48 to 0.54, indistinguishable from random). The exception is blend,
which shows a strong INVERTED signal: backdoor samples have HIGHER entropy
(AUROC 0.000 to 0.006 under the "lower entropy = backdoor" convention), meaning
the triggered inputs produce more diffuse, not more focused, attention patterns.

Evidence: `scratch/attention_entropy.py`, results in
`results/attention_entropy.json`.

## Claim

The 3 backdoor heads from H31 show lower attention entropy on backdoor samples
(the trigger creates a more focused attention pattern). Per-sample attention
entropy in these specific heads is a detection feature, either standalone or
fused with PSU.

## Test

1. Extract per-sample attention entropy in the 3 backdoor heads (L5H0, L5H10,
   L6H3) for clean and backdoor splits.
2. Entropy per head: `-sum(attn_weights * log(attn_weights))` averaged over
   query positions.
3. AUROC with negated entropy as score (lower entropy = more suspicious).
4. Test on 7 checkpoints across CIFAR-100 and Tiny.

## Results

### Per-head AUROC (lower entropy = backdoor convention)

| checkpoint | L5H0 | L5H10 | L6H3 | combined |
|---|---:|---:|---:|---:|
| cifar100 badnet_a2o | 0.538 | 0.540 | 0.456 | 0.511 |
| cifar100 blend | 0.003 | 0.004 | 0.000 | 0.000 |
| cifar100 wanet | 0.521 | 0.470 | 0.474 | 0.483 |
| cifar100 adaptive_blend | 0.277 | 0.351 | 0.522 | 0.355 |
| cifar100 lc | 0.545 | 0.545 | 0.476 | 0.526 |
| tiny badnet_a2o | 0.504 | 0.482 | 0.480 | 0.486 |
| tiny blend | 0.006 | 0.015 | 0.005 | 0.002 |

### Entropy statistics (nats)

| checkpoint | clean mean | backdoor mean | gap |
|---|---:|---:|---:|
| cifar100 badnet_a2o | 3.79 | 3.78 | +0.01 |
| cifar100 blend | 3.90 | 4.84 | -0.94 |
| cifar100 wanet | 3.91 | 3.92 | -0.01 |
| cifar100 adaptive_blend | 3.85 | 3.91 | -0.06 |
| cifar100 lc | 3.82 | 3.80 | +0.02 |
| tiny badnet_a2o | 3.83 | 3.84 | -0.01 |
| tiny blend | 3.85 | 4.67 | -0.82 |

## Interpretation

### Most attacks produce no per-sample entropy signal

For badnet, wanet, lc, and badnet on Tiny, the mean entropy gap between clean
and backdoor samples is within +/- 0.02 nats. The per-sample distributions
overlap almost completely, producing AUROC near 0.5. The backdoor heads diverge
at the POPULATION level (H31 showed JS divergence 0.15 to 0.35) but not at the
per-sample level that detection requires.

This makes sense: H31 measured JS divergence of full attention distributions,
which captures subtle distributional shifts averaged over the entire population.
Per-sample entropy is a much coarser statistic: it compresses each sample's
full attention distribution into a single number, losing the distributional
detail that carried the H31 signal.

### Blend is the exception, and the sign is wrong

Blend backdoor samples have dramatically HIGHER entropy than clean samples
(gap of -0.82 to -0.94 nats). In L6H3 on CIFAR-100, clean entropy is 3.70
and backdoor entropy is 5.10, a 1.4-nat gap. The AUROC under the "lower
entropy = backdoor" convention is 0.000, meaning with the inverted convention
(higher entropy = backdoor) it would be near 1.000.

This means blend's full-image blending trigger DIFFUSES attention rather than
focusing it. The trigger adds a uniform pattern to every patch, making every
token look equally important, which increases entropy. This is the opposite of
what happens with localized triggers (badnet), where the trigger creates a
distinctive patch that might focus attention.

Per the one-sided reporting rule, this inverted signal is not a detection
result. It is recorded as a diagnostic observation, not as evidence for
detection. A two-sided rule (flag both extremes of entropy) would need the
poison labels to choose the direction, violating the realistic-attacker
constraint.

### Adaptive_blend shows a weak inverted signal

AUROC of 0.277 to 0.355 on CIFAR-100 (below 0.5, slightly inverted). The
partial blending in adaptive_blend produces a smaller version of blend's
entropy-increasing effect.

### Practical verdict

Attention entropy in the backdoor heads is not a viable detection feature. For
most attacks it carries no signal. For blend it carries a strong signal but in
the wrong direction for one-sided detection. The per-sample entropy statistic is
too coarse to capture the distributional shifts that H31 identified at the
population level.

## Connection to other hypotheses

- **H31**: the 3 backdoor heads diverge at the population level (JS divergence).
  H40 shows this divergence does NOT translate to per-sample detection via
  entropy.
- **H22**: random head masking produced AUROC 0.539 (near random). H40 shows
  that targeted head analysis via entropy also fails. The backdoor heads are
  identifiable but not exploitable for per-sample detection through simple
  statistics.
- **H35**: tests whether masking these same heads as a PSBD operator (measuring
  prediction shift, not entropy) produces a useful detection signal.
