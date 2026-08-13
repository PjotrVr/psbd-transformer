# H25 — An adaptive attacker can hide from PSBD, but only from the probe it trained against

**Status: PRE-REGISTERED.** Mechanism implemented (`adaptive_evasion.py`) and unit
tested (`scratch/test_adaptive_evasion.py`, 12 checks passing). No training run
yet; the memory and speed probe is job `psbd_evade_smoke`.

## Claim

Two parts, and the second is the one worth publishing either way.

1. **An attacker who knows PSBD can defeat it.** Adding a hinge penalty to the
   training loss that raises poisoned samples' prediction shift onto the clean
   distribution should drive detection AUROC toward 0.5 while keeping ASR high.
2. **The evasion is probe-specific.** A model trained to look normal under
   dropout at one position should still be detectable by a *different*
   perturbation operator or position, because it was never optimised against
   those. If so, the operator family built in H19 to H22 is not a menu of
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

| quantity | requirement |
|---|---|
| ASR | stays above 0.9 |
| clean accuracy | within 2 points of the `lambda = 0` control |
| PSBD AUROC at the targeted probe | falls toward 0.5 |

An attack that destroys ASR or clean accuracy has not evaded the defence, it has
removed the backdoor, and reporting that as evasion would be wrong.

## The control that comes first

`--evade-psbd --evade-weight 0` runs the full adaptive code path with a zero
penalty. It must reproduce the existing checkpoints' ASR and detection within
noise. If it does not, the adaptive path differs from normal training in some way
other than the penalty and nothing else in the experiment is trustworthy.

Unit-level, this is already verified: at weight 0 the update is **bit-identical**
to a plain cross-entropy step (max parameter drift 0.00e+00).

## The transfer table, which is the actual point

Having trained against `before_attention_norm` with dropout, evaluate detection
under every operator and position in the study:

| trained against | evaluated with | expectation if evasion is probe-specific |
|---|---|---|
| dropout @ `before_attention_norm` | same | AUROC falls to ~0.5 |
| dropout @ `before_attention_norm` | dropout @ `post_residual` | partly recovers |
| dropout @ `before_attention_norm` | `channel_mask`, `token_mask`, `head_mask` | largely recovers |
| dropout @ `before_attention_norm` | `gaussian` | recovers if the evasion is mask-specific |

[H16](H16-where-the-backdoor-neurons-are.md) is weak prior evidence for the
probe-specific outcome: it found the backdoor's dimensions are highly plastic
(SAM relocates them completely, Jaccard 0.03 to 0.08, without weakening ASR). If
the backdoor relocates that easily under an optimizer change, an attacker
suppressing one probe's view may simply move it into another's rather than
removing it.

## Cost

The adaptive step retains `passes + 1` full ViT-B/16 activation graphs for the
backward, so peak memory scales with `(passes + 1) x batch`. Batch 64 with
passes 3 exhausted a 40GB card, which is what `psbd_evade_smoke` establishes a
safe setting for before the matrix runs. Roughly 4x normal training wallclock.

One implementation note worth keeping: the first version computed the clean
forward twice, once for the PSU base term and once for cross-entropy. That extra
retained graph was on its own enough to cause the OOM. The two now share one
forward.

## First matrix

    attacks        badnet_a2o, blend        (one localized trigger, one distributed)
    poison rates   0.01, 0.1
    lambda         0 (control), 1, 10
    probe          before_attention_norm, dropout, rate at the sigma 0.6 point

## A measurement that came out of the unit tests

PSU is **identically zero at random initialisation**: a near-uniform softmax
barely moves under dropout, so `P_c` is the same with and without it, and the
statistic only exists once the model is confident. It rose from 0.0000 to 0.12
over 60 warmup steps on a toy model. This is a mechanical reason for the paper's
instruction to score a late-stage model rather than an early one, and it means
the evasion penalty has nothing to act on during the first epochs of training.
Whether the attacker should therefore apply the penalty only in late epochs is an
open sub-question, and a cheap one to test.
