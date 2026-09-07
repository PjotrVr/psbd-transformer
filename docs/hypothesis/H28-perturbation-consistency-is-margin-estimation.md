# H28 — Prediction-shift detection is margin estimation, and the operator only sets the Jacobian

**Status: PARTIALLY SUPPORTED.** Prediction 4 (interchangeability at matched
sigma) confirmed: position variance exceeds family variance by 1.43x, Kendall
tau 0.700 for attack ranking across families. Prediction 3 (only stream
positions invert) **refuted**: input-side inversions 9.4% vs stream 4.1%.
Prediction 1 (direction norm scales with poison rate) **not confirmed**: norm is
approximately constant across 0.5% to 10% (Spearman rho = -0.40 to +0.80,
p >= 0.200), but layer-wise growth is highly consistent (159x from layer 1 to
12, all 16 checkpoints). See `docs/results/direction-norm-analysis.md`.
Prediction 2 untested (needs coefficient extraction per operator).

> **Panel note.** The inversion evidence cited below (`badnet_a2o` at 1% reading
> 0.194) is one attack. The asymmetry account predicts inversion wherever the
> backdoor direction's coefficient is small, so it must be checked against
> `blend`, `adaptive_blend` and `lc` at 1% before any of this is written up.
> `wanet` cannot contribute at 1%; it does not implant there on any dataset.
> Prediction 3 below is the panel version of the claim and is the one that
> settles it.

## The claim

PSBD, SCALE-UP, IBD-PSC and STRIP are usually presented as four methods with four
mechanisms. They are one method with four choices of **where to perturb**:

| family | detector | perturbs | implemented here as |
|---|---|---|---|
| input | SCALE-UP (Guo et al., ICLR 2023) | pixels | `scale_up` at `input_pixels` |
| input | STRIP (Gao et al., 2019) | pixels, by superimposition | `defences/baselines.py` |
| activation | **PSBD** (Li et al., CVPR 2025) | activations, by dropout | `dropout` and the mask operators |
| parameter | IBD-PSC (Hou et al., ICML 2024) | BatchNorm affine parameters | `gain_scale` at the LayerNorm outputs |

All four compute the same thing: how far a prediction moves when the model or its
input is disturbed, and all four flag the samples that move **least**.

> **The claim is that what they measure is decision margin, and that the
> perturbation operator and position determine only the Jacobian from the
> perturbation site to the readout, not the quantity being estimated.**

    original form
        a prediction flips when the induced logit change exceeds the margin
            P(flip) = P( delta . g  >  margin ),   g = J_position^T w_readout

    restated
        pick a place to perturb; the perturbation of size delta arrives at the
        classifier scaled by that place's Jacobian; whether the prediction
        survives depends on how that compares to how far the sample sits from
        the decision boundary

Under this reading the "neuron bias effect" is not needed, and neither is any
claim about which units carry the trigger.

## Why the mechanism claim has to go

Four measured results, none of which PSBD's own mechanism predicts:

1. **Gaussian noise, which removes nothing, scores 0.950** at
   `before_attention` and beats most structured masks
   ([H23](H23-gaussian-noise-control.md)). Removal is not required. And it is not
   that the perturbation is idle: the confidence-only null averages 0.520 with
   TPR at 1% FPR of 0.000.
2. **Head masking loses** (0.886) despite heads being the transformer's own unit
   ([H22](H22-head-mask-attention-units.md)).
3. **The 144-head leave-one-out profile carries no signal at all**
   ([H18](H18-sensitivity-profile-over-units.md)), benign control clean at 0.497.
4. **The backdoor is one non-axis-aligned direction**
   ([H16](H16-where-the-backdoor-neurons-are.md)): zeroing the top 300 of 768
   coordinates leaves ASR at 1.00, removing the direction takes it to 0.00.

Result 4 explains 2 and 3. A head, a neuron and a coordinate are axis-aligned
objects; no ranking over them names a direction lying across the axes. Result 1
follows too: isotropic noise is indifferent to axis alignment, and so is the
backdoor.

## What the margin account explains that the mechanism account cannot

**Position dominates everything.** PSBD's mechanism says nothing about where
dropout goes, yet position moves `badnet_a2o` at 1% from **0.194 to 0.839**. Under
the margin account this is expected: different positions have different Jacobians,
so the same nominal rate delivers a different effective `delta . g`.

**The statistic can run backwards.** At 1% poisoning the published stream
placement reads **0.194**, well below chance on a fully implanted attack
(ASR 0.997). A mechanism that says "the backdoor is robust" has no way to produce
that. The margin account does, through an asymmetry in what supports each
prediction:

- a clean prediction is supported **redundantly** by many features;
- the backdoor prediction rests on **one direction** with coefficient `alpha`.

Perturbing the residual stream, where that direction lives, attenuates the single
thing holding the backdoor up. When `alpha` is small (low poison rate) the
backdoor degrades **faster** than the clean prediction loses its many supports,
the ordering reverses, and the statistic inverts. Perturbing a sub-layer input
instead disturbs the computation broadly, which costs the redundant clean support
more, and the ordering holds.

This predicts a critical rate `p*` at which AUROC crosses 0.5, moving with
`alpha`. That is exactly the measured shape:

    before_attention_norm @ 10%   0.4:0.78  0.6:0.82  0.8:0.87  0.9:0.94   monotone
    before_attention_norm @  1%   0.4:0.70  0.5:0.83  0.7:0.61  0.8:0.32  0.9:0.24   crosses

## Falsifiable predictions

1. **The backdoor direction norm scales with poison rate.** Measure `||v||` with
   `analysis/direction.py` across 1%, 5%, 10%. It must rise monotonically. If it
   is flat, the asymmetry account is dead and so is most of this file.
2. **`p*` is predicted by `||v||`.** The rate at which each (position, operator)
   crosses AUROC 0.5 should be an increasing function of the measured direction
   norm, across attacks and poison rates. This is the quantitative test and the
   one worth a figure.
3. **Only residual-stream positions invert.** Input-side positions should stay
   above 0.5 at every rate and poison rate; stream positions should invert once
   `alpha` falls below a threshold.
4. **The three families are interchangeable at matched disturbance.** If the
   operator only sets the Jacobian, then at matched clean-validation shift ratio
   `scale_up`, `gain_scale` and `dropout` should rank attacks similarly, and their
   differences should be smaller than the differences between positions within a
   family. A large, systematic family effect at matched sigma refutes this.

Prediction 4 is what the running grid answers directly, and it is the one that
would most cleanly kill the claim.

## What would refute the whole thing

- `||v||` flat across poison rate (prediction 1 fails), leaving the inversion
  unexplained.
- Families differing more than positions at matched sigma (prediction 4 fails),
  which would mean the operator carries mechanism, not just gain.
- An input-side position inverting at low poison rate, which the asymmetry account
  forbids.

## Evidence from the full grid (27 operator/position combinations, 48/48 coverage)

### Prediction 3: REFUTED

H28 predicted that only residual-stream positions invert (AUROC < 0.5), because
perturbing the stream where the backdoor direction lives attenuates it faster
than it destroys clean evidence. Input-side positions should stay above 0.5.

2210 cells, sigma-matched at 0.6:

| stream type | cells | inversions | rate |
|---|---|---|---|
| input_side | 817 | 77 | 9.4% |
| residual_stream | 1106 | 45 | 4.1% |
| unit_specific | 287 | 15 | 5.2% |

Input-side positions invert **more often** (9.4%) than residual-stream ones
(4.1%), the opposite of the prediction. The worst offenders are
`gaussian@before_mlp_norm` (11 inversions, min AUROC 0.002),
`channel_mask@after_embedding` (10), and `token_mask@after_embedding` (10).

The asymmetry account (backdoor direction in the stream, clean evidence
distributed across sub-layer inputs) does not predict this. What it misses is
that some sub-layer inputs are themselves LayerNorm outputs, and perturbation
after LayerNorm is amplified by the normalization gain in a way that does not
respect the input/stream distinction. `after_embedding` inversions also break
the prediction: the embedding output is the initial residual stream, so
classifying it as "input-side" is a labelling artefact, but the inversions at
`before_attention_norm` and `before_mlp_norm` are genuine input-side positions
and they invert anyway.

### Prediction 4: SUPPORTED

The three perturbation families should be interchangeable at matched sigma:
position variance should exceed family variance, and families should rank
attacks similarly.

**Position vs family variance.** At fixed position, varying the operator
(family effect) gives a mean AUROC range of 0.188. At fixed operator, varying
the position (position effect) gives a mean AUROC range of 0.268. Position
variance exceeds family variance by **1.43x**.

Per-position family spread (positions with >= 3 operators):

| position | n ops | mean | range | best | worst |
|---|---|---|---|---|---|
| post_residual | 3 | 0.859 | 0.032 | gaussian | channel_mask |
| before_attention_residual | 5 | 0.844 | 0.063 | dropout | channel_mask |
| before_mlp_norm | 4 | 0.755 | 0.091 | token_mask | gaussian |
| before_attention | 4 | 0.878 | 0.143 | dropout | channel_mask |
| before_mlp_residual | 5 | 0.840 | 0.164 | dropout | droppath |
| before_attention_norm | 4 | 0.831 | 0.174 | token_mask | gaussian |

Per-operator position spread (operators with >= 3 positions):

| operator | n pos | range | best position | worst position |
|---|---|---|---|---|
| dropout | 14 | 0.184 | before_attention | after_attention_residual |
| channel_mask | 12 | 0.175 | pre_residual | after_embedding |
| gaussian | 12 | 0.196 | before_attention | before_mlp_norm |
| token_mask | 10 | 0.256 | after_mlp_residual | after_embedding |

Position spread within an operator (0.175 to 0.256) consistently exceeds
family spread at a fixed position (0.032 to 0.174), confirming that position
is the dominant axis.

**Attack ranking consistency.** Kendall tau across 12 family pairs at shared
positions: mean 0.700, median 0.700. Families rank attacks similarly (0.0 would
be random, 1.0 identical). This means the operator determines the gain of the
perturbation, not which attacks it detects, consistent with the Jacobian account.

### Prediction 1: IN FLIGHT (job 1021982)

Feature extraction across 0.5%, 1%, 5%, 10% checkpoints submitted for
badnet_a2o and blend on CIFAR-100 and Tiny (16 checkpoints total).
`experiments/backdoor_direction/direction_norm_analysis.py` computes the backdoor direction norm at
all 13 layers (embedding + 12 blocks) using CLS-token features. The test is
whether norm rises monotonically with poison rate.

### Prediction 2: UNTESTED (blocked on prediction 1)

Per-sample critical rate p* was implemented and tested but does not outperform
sigma-matched PSU. The p* score (smallest perturbation rate at which a sample's
prediction flips) loses to fractional PSU at matched sigma by -0.084 AUROC on
average, winning only 17% of 2302 paired cells. The rate grid is too coarse
(9 points for dropout, 5 for gaussian) to give p* enough resolution, so it
produces massive tied groups. The crossover relationship between p* and
direction norm therefore cannot be tested from existing caches. A finer rate
grid or a continuous score (interpolated from the per-rate shift fraction) would
be needed.

### Per-sample critical rate: negative result

p* was intended to eliminate the rate-selection problem by giving each sample a
single scalar. Against sigma-matched fractional PSU:

| slice | n | sigma mean | p* mean | diff | p* wins |
|---|---|---|---|---|---|
| all | 2302 | 0.827 | 0.744 | -0.084 | 17.1% |
| 1% | 536 | 0.778 | 0.702 | -0.077 | 20.3% |
| 5% | 840 | 0.832 | 0.736 | -0.097 | 14.0% |
| 10% | 926 | 0.852 | 0.775 | -0.076 | 18.0% |

PSU wins because it is a continuous probability that separates samples within a
tied p* bucket. The rate grid gives at most 9 distinct p* values (plus
"never flipped"), so AUROCs above 0.9 are mechanically unreachable.

### Ranking table

All 27 operator/position combinations at 48/48 coverage, sorted by 1% AUROC:

| rank | operator | position | mean AUROC | AUROC 1% | TPR@5% 1% | worst | inv |
|---|---|---|---|---|---|---|---|
| 1 | gain_scale | mlp_norm_out | 0.899 | 0.947 | 0.826 | 0.459 | 2 |
| 2 | gaussian | before_mlp | 0.900 | 0.928 | 0.796 | 0.494 | 1 |
| 3 | token_mask | before_attention_residual | 0.854 | 0.912 | 0.753 | 0.325 | 4 |
| 4 | token_mask | before_attention_norm | 0.911 | 0.904 | 0.729 | 0.632 | 0 |
| 5 | token_mask | before_mlp | 0.869 | 0.894 | 0.666 | 0.445 | 2 |
| 6 | token_mask | after_mlp_residual | 0.911 | 0.874 | 0.576 | 0.650 | 0 |
| 7 | gaussian | mlp_neurons | 0.891 | 0.865 | 0.539 | 0.603 | 0 |
| 8 | channel_mask | before_attention_norm | 0.834 | 0.831 | 0.458 | 0.441 | 2 |
| 9 | token_mask | before_attention | 0.854 | 0.830 | 0.555 | 0.255 | 5 |
| 10 | scale_up | input_pixels | 0.791 | 0.828 | 0.343 | 0.351 | 2 |

The top 4 include 3 different operators (gain_scale, gaussian, token_mask) at 3
different positions. No single operator dominates, and no single position
dominates. The strongest 1% position (`gain_scale@mlp_norm_out`) is a parameter
perturbation, the strongest mean (`token_mask@before_attention_norm` and
`token_mask@after_mlp_residual`, tied at 0.911) is an activation mask.

## Early evidence (superseded by the full grid above)

400 samples, `vit_cifar10_badnet_a2o_0_1`, matched only informally:

| operator | family | position | sigma | AUROC |
|---|---|---|---|---|
| `scale_up` (factor 5) | input | `input_pixels` | 0.805 | 0.986 |
| `gain_scale` (omega 2) | parameter | `attention_norm_out` | 0.670 | 0.923 |

Both ports separate on their first contact with real data, which is the minimum
needed for prediction 4 to be worth testing. It is one checkpoint at one poison
rate with no matched-sigma control, so it is a reason to run the grid, not a
result.

## Implementation notes worth keeping

**The IBD-PSC port is exact, not an approximation.** ViT has no BatchNorm, so the
published method cannot be applied directly. But for LayerNorm,
`y = gamma * x_hat + beta`, so scaling both affine parameters by `omega` gives
`omega * y`: amplifying the parameters is identical to scaling the layer's output,
which a post-hook does without mutating a single weight. Verified to 9.5e-07.

**SCALE-UP has to round-trip the normalization.** The loaders deliver normalized
tensors; scaling those directly would amplify the dataset mean as if it were
signal and would put the clip, which is where SCALE-UP's nonlinearity lives, in
the wrong place. The operator un-normalizes, scales, clips to the valid pixel
range and renormalizes. Verified against `clip(2x, 0, 1)` to 8.9e-08.

## Reproduce

    python pbs/generate_perturbation_jobs.py --stage custom --position-set full \
        --operator scale_up gain_scale --dataset cifar10 \
        --only-tag 0_01 0_05 0_1 benign --min-asr 0.5
