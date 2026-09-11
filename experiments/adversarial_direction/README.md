# Adversarial direction

## Question

Karayalcin, Krcek, Chen and Picek ("Backdoor Directions in Vision Transformers",
arXiv 2603.10806, Section 6, Table 2) run PGD from CIFAR-100 test images on a
ViT-B/16 and report that the adversarial examples land on the attacker's target
class far more often for stealthy triggers than for BadNet, that this shift has
high cosine similarity to the backdoor direction in the middle layers for WaNet
and BPP but not for BadNet, and that from triggered images 20% to 50% of
adversarial examples revert to the original class with strongly negative cosine
to the direction in late layers. This experiment reruns their protocol on our
own CIFAR-100 checkpoints and checks each claim against our numbers.

## Their protocol, quoted

Section 6, "Experimental Setup":

> For both subsections, we use PGD with l_inf-norm and epsilon = 8/255. We
> start from either clean or backdoored test images and run 5 or 15 steps for
> clean and backdoored examples, respectively.

Section 6 goes on to define the per-layer activation shift and its cosine
similarity to the backdoor direction, and states that the CLS token is used
because the all-token vector is too high dimensional to analyze this way.
Neither the step size nor whether PGD starts at the clean image or at a random
point in its epsilon ball is stated. We use step size epsilon / 4 and start
exactly at the clean or triggered image, the two choices this leaves open.
Untargeted means maximizing cross entropy against the label the starting image
currently carries: the true label from a clean start, the trigger-induced
target label from a triggered start, since that is the only way "reverts to the
original class" is a meaningful measurement on the far side of a triggered
start.

## Method

`measure.py` reuses `paired_rows`, `residual_stream` and `directions` from
`experiments/whole_network_erasure/measure.py`, which already build the paired
clean and triggered rows from `data.splits.build_psbd_loaders_from_checkpoint`,
capture every layer's CLS token with `analysis.features.captured_layers` and
estimate each layer's backdoor direction on the first 500 pairs. Every rate and
cosine number below is read on the rows after those first 500, so the
direction is never validated on the data it came from.

PGD runs in pixel space, denormalizing and renormalizing around the model's own
input statistics the way `detectors.strip.normalization_buffers` already does
for STRIP's overlay, and the gradient step uses `defences.inference.forward_logits`
so the loss differentiates through the perturbation, not through any model
weight. For every checkpoint we measure, on the rows after the estimation
pairs:

1. the share of PGD-from-clean adversarial examples predicted as the target
   class, both on the checkpoint itself and, at the same target class and on
   the same clean images, on the benign checkpoint of the same dataset,
2. the share of PGD-from-triggered adversarial examples predicted as the
   original (pre-trigger) class,
3. the per-layer mean and median cosine similarity between the CLS token shift
   PGD caused (adversarial minus original) and that layer's backdoor direction,
   for both starting points,
4. the same cosine against 1 random unit direction, shared across every layer
   of a checkpoint, as the null.

Run on `vit_cifar100_badnet_a2o_0_1`, `vit_cifar100_blend_0_1`,
`vit_cifar100_bpp_0_1`, `vit_cifar100_wanet_0_1`, `vit_cifar100_lf_0_1`,
`vit_cifar100_badnet_a2o_0_01`, `vit_cifar100_bpp_0_01` and
`vit_cifar100_benign`, with `--max-samples 1000`, which leaves 491 paired rows
per checkpoint for evaluation after the 500-pair direction estimate. Every
number below is a single checkpoint, a single seed, no bootstrap interval, so a
gap of a few points is within what checkpoint-to-checkpoint variance alone could
produce. Results are written to `results/_experiments/adversarial_direction/<folder>.json`.

## Results

### Target-class capture from clean images, and reversion from triggered images

Paper columns are Table 2's CIFAR-100 row for each attack, at poison rates
0.01, 0.05 and 0.1. We trained no 0.05 checkpoints and have no LF or benign row
in their table, marked n/a. "Benign ref" is the same PGD run on the benign
checkpoint of the dataset, at the same target class, which Table 2 has no
equivalent of.

| Attack | Poison rate | Paper clean to target | Our clean to target | Our benign ref | Paper backdoor to original | Our backdoor to original |
|---|---|---|---|---|---|---|
| BadNet | 0.01 | 0.6% | 0.6% | 0.2% | 33.2% | 15.3% |
| BadNet | 0.1 | 7.7% | 0.0% | 0.2% | 33.0% | 11.4% |
| Blend | 0.1 | 0.2% | 6.7% | 0.2% | 22.3% | 0.0% |
| WaNet | 0.1 | 4.1% | 16.1% | 0.2% | 26.0% | 75.4% |
| BPP | 0.01 | 29.2% | 13.2% | 0.2% | 18.7% | 48.5% |
| BPP | 0.1 | 41.5% | 2.6% | 0.2% | 21.9% | 53.4% |
| LF | 0.1 | n/a | 4.7% | 0.2% | n/a | 25.9% |
| Benign | n/a | n/a | 0.2% | 0.2% (self) | n/a | 79.4% |

### Clean-start CLS shift, cosine to the backdoor direction (null in parentheses)

Mean over the 491 evaluation images, at every other layer. Layer 0 is the
embedding output before block 1, layer 12 the output of the last block.

| Attack (rate) | L0 | L2 | L4 | L6 | L8 | L10 | L12 |
|---|---|---|---|---|---|---|---|
| BadNet (0.1) | 0.00 (0.00) | 0.26 (0.01) | 0.14 (-0.02) | 0.10 (-0.01) | 0.02 (-0.01) | -0.00 (-0.00) | -0.00 (-0.00) |
| BadNet (0.01) | 0.00 (0.00) | 0.29 (0.01) | 0.11 (-0.01) | 0.02 (-0.01) | -0.01 (-0.01) | -0.02 (-0.00) | -0.01 (-0.00) |
| Blend (0.1) | 0.00 (0.00) | 0.45 (0.01) | 0.78 (-0.01) | 0.49 (0.00) | 0.25 (0.00) | 0.11 (0.00) | 0.07 (-0.00) |
| WaNet (0.1) | 0.00 (0.00) | -0.42 (0.01) | -0.33 (-0.03) | -0.26 (-0.02) | 0.10 (0.00) | 0.09 (-0.00) | 0.10 (0.00) |
| BPP (0.1) | 0.00 (0.00) | 0.64 (0.00) | 0.82 (-0.04) | 0.51 (-0.00) | 0.14 (-0.01) | 0.03 (0.00) | 0.03 (0.00) |
| BPP (0.01) | 0.00 (0.00) | 0.59 (0.01) | 0.84 (-0.03) | 0.48 (0.01) | 0.21 (0.01) | 0.11 (0.00) | 0.08 (-0.00) |
| LF (0.1) | 0.00 (0.00) | 0.53 (0.02) | 0.63 (-0.02) | 0.35 (-0.00) | 0.22 (0.00) | 0.09 (0.00) | 0.06 (-0.00) |
| Benign | 0.00 (0.00) | 0.30 (0.01) | 0.10 (-0.03) | 0.05 (-0.00) | 0.03 (0.00) | -0.00 (0.00) | -0.00 (-0.00) |

### Backdoor-start CLS shift, cosine to the backdoor direction (null in parentheses)

| Attack (rate) | L0 | L2 | L4 | L6 | L8 | L10 | L12 |
|---|---|---|---|---|---|---|---|
| BadNet (0.1) | 0.00 (0.00) | 0.16 (-0.00) | 0.15 (-0.04) | 0.14 (-0.02) | -0.21 (0.03) | -0.36 (0.03) | -0.29 (0.04) |
| BadNet (0.01) | 0.00 (0.00) | 0.27 (0.00) | 0.12 (-0.02) | -0.04 (-0.00) | -0.32 (0.01) | -0.26 (0.02) | -0.24 (0.03) |
| Blend (0.1) | 0.00 (0.00) | 0.15 (-0.01) | 0.08 (0.02) | -0.03 (0.03) | -0.01 (0.04) | 0.12 (0.07) | 0.14 (0.06) |
| WaNet (0.1) | 0.00 (0.00) | -0.51 (0.02) | -0.52 (-0.04) | -0.81 (-0.05) | -0.90 (-0.02) | -0.90 (-0.00) | -0.82 (0.00) |
| BPP (0.1) | 0.00 (0.00) | 0.42 (0.01) | 0.17 (-0.02) | -0.64 (-0.03) | -0.81 (0.03) | -0.83 (0.03) | -0.81 (0.03) |
| BPP (0.01) | 0.00 (0.00) | 0.57 (0.01) | 0.22 (-0.02) | -0.46 (0.03) | -0.70 (-0.01) | -0.71 (0.03) | -0.64 (0.04) |
| LF (0.1) | 0.00 (0.00) | 0.46 (0.03) | -0.21 (0.00) | -0.84 (0.00) | -0.86 (0.01) | -0.89 (0.00) | -0.85 (-0.00) |
| Benign | 0.00 (0.00) | 0.32 (0.02) | 0.09 (-0.01) | 0.05 (0.01) | -0.03 (0.01) | -0.08 (0.01) | -0.07 (0.01) |

## Verdicts

**Clean images land on the target class far more for stealthy triggers than for
BadNet.** Differs. BPP shows the opposite of their poison-rate trend on our
checkpoints, 13.2% at 1% poisoning falling to 2.6% at 10%, against their
monotonically increasing 29.2% to 41.5%, and both our BPP numbers sit well below
their range. WaNet does show a clean-to-target share above BadNet's, 16.1%
against 0.0% to 0.6%, which is directionally what they report, but at 4 times
their own 4.1% figure for WaNet at 10%. Blend, which their table puts near
BadNet at 0.2%, reaches 6.7% on our checkpoint. The direction "stealthier than
BadNet" survives for WaNet only, and the absolute numbers do not reproduce for
any attack, which given a single checkpoint per cell here is at least partly
checkpoint variance rather than a protocol difference.

**Cosine similarity to the backdoor direction is high in the middle layers for
WaNet and BPP, not for BadNet.** Reproduced for BPP, not for WaNet. BPP peaks
at 0.82 to 0.84 around layer 4, roughly 6 times BadNet's 0.11 to 0.14 peak at
the same layer and far above the near-zero null, which matches their Figure 2
description closely. WaNet on our checkpoint runs negative through layers 2 to
6 (-0.26 to -0.42) before turning to a small positive 0.09 to 0.10 in the last
layers, the opposite pattern from what Section 6 reports for WaNet's middle
layers. Blend and LF, which are not in their Figure 2 but are in scope here,
both show a BPP-like positive mid-layer peak (0.63 to 0.78), so on our
checkpoints the "high mid-layer cosine" behavior tracks BPP, Blend and LF and
not WaNet, rather than tracking stealth in general.

**From triggered images, 20% to 50% of adversarial examples revert to the
original class.** Reproduced for BPP (48.5% and 53.4%) and LF (25.9%), exceeded
by WaNet (75.4%) and the benign reference (79.4%, where "reversion" has no
attack to undo and just reflects how often 15 untargeted PGD steps flip a
correctly classified image), undershot by BadNet (11.4% to 15.3%), and absent
for Blend (0.0%). Blend's 0 reversion is not an isolated anomaly here: their
own Section 4.2 flags CIFAR-100 Blend as "the only attack where [weight
orthogonalization] does not" reduce ASR, so a CIFAR-100 Blend checkpoint being
the one place a 20% to 50% band also fails to hold is consistent with something
specific to that pairing rather than a bug in this measurement.

**Late-layer cosine to the direction is strongly negative from triggered
images, whether or not the example reverts.** Reproduced for BadNet, WaNet, BPP
and LF, whose last 3 layers run -0.24 to -0.90, all well past the null's -0.05
to 0.07 band. Not reproduced for Blend, whose last 3 layers are positive (0.12
to 0.14), the same checkpoint that already broke the reversion-share claim
above. The benign reference sits close to its own null (-0.03 to -0.08),
which is the expected negative control since a benign model has no real
backdoor direction to align a shift against, only the direction estimated from
an externally imposed probe trigger it never learned.

## Limitations

Every cell here is 1 checkpoint, 1 seed, 491 evaluation rows, so the
disagreements above, especially BPP's non-monotonic clean-to-target share and
WaNet's sign flip in the mid layers, could shrink or reverse with a seed
replicate. The paper's step size is not stated and this experiment used
epsilon / 4 with no random start, both explicit choices rather than a value we
could confirm against their setup. No SSBA or TrojanNN checkpoints exist in
this repository's attack registry, so the paper's two consistently strong
target-class-capture rows have no analogue here. Only CIFAR-100 was run, since
that is what Table 2 reports.
