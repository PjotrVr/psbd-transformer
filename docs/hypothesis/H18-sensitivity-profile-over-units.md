# H18 — Stop looking for the backdoor units; use the shape of the per-unit sensitivity profile

**Status: REFUTED outright by the real measurement.** The 144-head leave-one-out
profile was run on 7 checkpoints and carries no detection signal under any
summary. The proxy result below (that the profile *minimum* across placements
beats the mean) still stands, but it was never a test of this file's claim.

## The real measurement

`psbd_head_profile.py`, all 144 heads ablated one at a time, deterministic mask so
no Monte Carlo averaging is needed. One-sided AUROC, direction fixed a priori per
statistic:

| checkpoint | mean | max | std | gini | top-5 mass |
|---|---|---|---|---|---|
| `badnet_a2o` 1% | 0.284 | 0.213 | 0.256 | 0.137 | 0.136 |
| `badnet_a2o` 10% | 0.475 | 0.421 | 0.452 | 0.120 | 0.105 |
| `wanet` 10% | 0.493 | 0.513 | 0.508 | 0.419 | 0.442 |
| `blend` 10% | **0.608** | 0.579 | 0.589 | 0.331 | 0.293 |
| `bpp` 10% | 0.534 | 0.502 | 0.518 | 0.202 | 0.187 |
| `badnet_a2a` 10% | 0.078 | 0.013 | 0.052 | 0.443 | 0.337 |
| **benign control** | **0.497** | 0.500 | 0.499 | 0.520 | 0.521 |

Only `blend` clears 0.6 on any statistic. The benign control is clean at 0.497,
so this is a real negative rather than a broken measurement, and the concentration
statistics this file was built on (gini, top-5 mass) sit at 0.12 to 0.44 across
the board.

**Concentration of per-unit sensitivity does not detect backdoors.**

[H16](H16-where-the-backdoor-neurons-are.md) has since been corrected in a way
that explains this cleanly rather than merely agreeing with it. The backdoor is
**one linear direction in the residual stream, and it is not axis-aligned**:
zeroing even the top 300 of 768 coordinates leaves ASR at 1.00, while removing the
single direction takes ASR to 0.00. So there are no "backdoor units" to find. A
head, a neuron and a coordinate are all axis-aligned objects, and no ranking over
them can name a direction that lies across the axes.

That is now the third and most decisive failure of the where-to-mask programme,
after H16's own two refuted data-free localizers. It also predicts
[H22](H22-head-mask-attention-units.md)'s failure and is consistent with
[H23](H23-gaussian-noise-control.md): isotropic noise is indifferent to axis
alignment, and so is the backdoor.

## An architectural finding that came out of it

In every profile one head dominates. On `badnet_a2o` at 1%, ablating block 1
head 6 costs a mean **0.506** of confidence; the next-largest head costs 0.033, a
15x gap:

    block  1 head  6   0.5057
    block 10 head  7   0.0327
    block  7 head  5   0.0260
    block 12 head  8   0.0123

A single early attention head carries roughly half this model's confidence on
every input, clean or triggered. That is a property of fine-tuned ViT-B/16 rather
than of backdoors, and it plausibly explains why random head masking is weak
([H22](H22-head-mask-attention-units.md)): most masks remove heads that do
nothing, and the one head that matters is shared by clean and poisoned inputs
alike, so removing it moves both equally and separates neither.

---

## Earlier proxy result, retained

**Status of the proxy: the stated direction is REFUTED, and the cheap test
produced a better statistic than the one it was testing.** Backdoored profiles turn out to be
*flatter* than clean ones, not more peaked. Separately, the simplest reduction of
the profile, its **minimum**, beats PSBD's mean on the low-poison-rate case by
+0.12 while keeping PSBD's direction unchanged. Per-unit masking machinery is
implemented and tested (`defences/perturbations.py`); the per-unit measurement
itself has not been run.

## Result of the cheap test (H18 step 1, run before any GPU time)

Proxy profile: for each sample, the vector of fractional PSU across the 6
placements that reach a matched clean-validation shift ratio, so placements are
compared at equal measured disturbance. One-sided AUROC, nothing flipped, mean
over 16 backdoored CIFAR-10 checkpoints with all-to-all excluded:

| reduction of the profile | mean AUROC | `badnet_a2o` 1% | benign |
|---|---|---|---|
| **min** (most responsive placement) | **0.935** | **0.867** | 0.489 |
| mean (PSBD's reduction) | 0.933 | 0.743 | 0.491 |
| max | 0.871 | 0.390 | 0.494 |
| gini / entropy / top-2 mass | 0.058 | 0.103 | 0.512 |

Robust to the matching target: `min` gives 0.906 / 0.935 / 0.932 mean and
0.825 / 0.867 / 0.871 on `badnet_a2o` 1% at sigma 0.4 / 0.6 / 0.7, with the
benign control at 0.483 to 0.490 throughout.

**Two things follow, and the first one contradicts this file's claim.**

*Concentration runs the other way, and is not pursued.* Gini at 0.058 means
backdoored samples have **lower** concentration than clean ones, so their profiles
are flat, not peaked, which is the opposite of what the claim below predicted. The
claim is therefore refuted as stated.

The observation is recorded and left there. Turning it into a detector would mean
scoring a statistic in the direction opposite to PSBD's, and this project reports
one-sided results only: low PSU means poisoned, and a number below 0.5 is a
failure rather than a win waiting to be re-signed
([H15](H15-one-sided-rules-are-the-common-weakness.md), retired for this reason).
A flat-profile detector would need its own mechanism, its own benign control, and
its own name before it could be reported as anything, and the `min` statistic
below already performs as well while keeping PSBD's direction intact.

*The minimum is the better statistic anyway.* It performs as well as any
concentration measure, needs no direction change (low PSU = poisoned, exactly as
published), and has a plain reading: **different attacks are carried by different
structures ([H16](H16-where-the-backdoor-neurons-are.md) shows the dimensions are
disjoint across attacks), so the placement that exposes a given backdoor differs
by attack. Taking the per-sample minimum lets each sample be judged by whichever
placement it is most sensitive to, instead of committing to one placement in
advance.**

That is a unified answer to "where to perturb" that requires no localization: do
not choose, perturb in several places and keep the strongest response.

Note this is exactly where rank-*averaging* failed. `experiments/detector_ensemble/ensemble.py` scores
0.480 on `badnet_a2o` 1% because averaging pulls an inverted member into the
pool, while the minimum is unaffected by members that carry no signal. The 0.480
supersedes an earlier 0.475, which ranked against a pool containing the backdoor
split, see `audit-2026-09-07.md`.

**Caveat.** The placement set and the matching target were chosen by inspecting
CIFAR-10 results, so these numbers are optimistic. GTSRB and Tiny are in flight
and are the out-of-sample test.

## Original claim, kept for the record

## Why this hypothesis exists

Three results in this ledger box in the problem from three sides, and together
they say the obvious approach cannot work:

- [H16](H16-where-the-backdoor-neurons-are.md): the backdoor occupies 5 to 17 of
  768 residual dimensions at a late block, so it **is** localized.
- H16 again: the dimensions are **disjoint across attacks**, Jaccard 0.00 to 0.08
  against a 0.87 reproducibility ceiling. `badnet_a2o` and `badnet_a2a` share the
  identical trigger image and overlap at exactly 0.00. So no fixed unit set
  transfers.
- H16 again: **both data-free localizers were refuted.** Per-channel Lipschitz
  ranking reaches Jaccard 0.038 against the true TAC top-20, and the benign model
  scores the *highest* global correlation of all 18 checkpoints. The head
  alignment Z rule gets 0 of 15 true positives and 1 of 3 false positives.
- [H17](H17-low-poison-rate-is-a-placement-artifact.md): where the perturbation
  goes dominates everything else at low poison rate. The same checkpoint, same
  one-sided rule, moves from 0.297 to 0.839 on the choice of position alone.

H16 states the gap plainly: a PCA separating clean from triggered at purity 0.999
"requires triggered inputs to draw, and a defender does not have any."

So the direct route, find the backdoor units then perturb them, requires solving
localization from clean data, which has now failed twice.

## Claim

**The defender does not need to know which units carry the backdoor. Sensitivity
to a unit is itself a per-sample statistic, and the backdoor's signature is the
*concentration* of that sensitivity, not its location.**

Perturb one unit at a time across all of them and record, per sample, how much
the prediction moves. That yields a sensitivity profile per sample, a vector over
units rather than a single number:

    original form
        s_u(x) = P_c(x; theta) - P_c(x; theta with unit u masked)
        where c = argmax_j P_j(x; theta), as in PSBD Equation 2

    restated
        for every unit u, mask that one unit and record how far the confidence in
        the sample's own unperturbed predicted class falls. The result is one
        number per unit per sample, so a sample becomes a profile over units
        instead of a scalar.

PSBD's PSU is one summary of that profile, its mean under a random subset. The
claim is that the mean is the wrong summary:

- A **clean** prediction is supported redundantly across many units, so its
  profile is broad and flat. Removing any one unit costs little.
- A **backdoored** prediction rides a shortcut carried by 5 to 17 units, so its
  profile is sharply peaked. Removing the right unit collapses it, removing any
  other costs nothing.

Concentration is measurable without knowing *which* units are the peaked ones,
which is exactly what the two refuted localizers were trying and failing to
recover. Gini coefficient, normalized entropy, top-k mass, or kurtosis of the
profile are all label-free, trigger-free, and attack-agnostic.

This is also why disjointness across attacks stops being fatal. The profile of
`badnet_a2o` peaks on a different unit set than `blend`, and the concentration
statistic does not care, because it never reads which index peaked.

## Why this is the unified answer to "where to perturb"

It dissolves the question. Instead of choosing a placement, the profile perturbs
every unit in turn and lets the sample's own response say which units mattered to
it. Depth, position, and unit identity all become axes of the profile rather than
hyperparameters someone has to fit.

It also predicts [H17](H17-low-poison-rate-is-a-placement-artifact.md)'s result as
a special case: a fixed placement works when it happens to overlap the peaked
region for that attack, which is why the best placement is attack-dependent
([H4](H4-placement-is-attack-dependent.md)) and why no single one wins.

## What is already built

`defences/perturbations.py` provides the structured operators the profile needs,
all sharing nn.Dropout's interface so they plug at any registry position:

| operator | unit removed | notes |
|---|---|---|
| `head_mask` | a whole attention head | 12 per block, 144 total on ViT-B/16 |
| `channel_mask` | an embedding channel or MLP hidden neuron | mask shared across tokens |
| `token_mask` | a whole token | CLS protected, or the prediction is destroyed rather than perturbed |
| `droppath` | a whole branch for a sample | the residual-native perturbation |
| `gaussian` | nothing, adds scaled noise | the control for whether removal is needed at all |
| `dropout` | the paper's element-wise Bernoulli | baseline |

Two new registry positions expose the units:
`attention_heads` and `mlp_neurons` (`defences/dropout.py`).

`attention_heads` needed a forward wrapper, not a hook.
`nn.MultiheadAttention` runs `F.multi_head_attention_forward`, which reads
`out_proj.weight` directly and never calls `out_proj` as a module, so a hook on
it never fires and the model output came back bit-identical. The wrapper
recomputes attention and exposes the head axis; at rate 0 it reproduces
PyTorch's own output to 3.6e-07 on logits of scale 0.79, which is float32
associativity and not a semantic difference.

Driver support: `psbd_dropout_sweep.py --perturbation {dropout,head_mask,...}`,
with the operator in the cache folder name so operators cannot overwrite each
other. `dropout` keeps the bare legacy name, so every existing cache stays
addressable and `--skip-existing` still finds it.

## Test plan

1. **Cheapest first, the summary swap.** Keep random masking, change only the
   summary from mean to a concentration statistic over the k passes. This does
   not need the profile and is computable from existing caches, since the
   per-pass tracked-class probabilities are already stored. If concentration over
   3 random subsets already beats the mean, the claim has support before any GPU
   time is spent.
2. **Head profile, one checkpoint.** 144 leave-one-head-out passes on
   `vit_cifar10_badnet_a2o_0_01`, the checkpoint where the published
   configuration reads 0.297. Compare Gini and entropy of the profile against
   PSU's mean. Restrict to the 2000-sample validation split plus the paired
   analysis splits to keep it affordable.
3. **Does concentration beat location?** Give an oracle the true top-20 TAC
   dimensions from H16 and perturb exactly those. If the oracle barely beats the
   label-free concentration statistic, localization was never the bottleneck and
   this reframing is the right one. If the oracle wins by a lot, localization
   still matters and this hypothesis is weaker than it looks.
4. **Granularity.** Heads (144) against MLP neurons (36864, so grouped) against
   channels. H16 puts the backdoor in the residual stream, which argues for
   channels; heads are cheaper and more interpretable. Measure both.
5. **Benign control throughout.** A clean model's profile must be flat. If
   concentration separates on a benign checkpoint, it is measuring something
   about prediction confidence rather than about a backdoor, and the same trap
   that [H12](H12-psu-is-not-just-confidence.md) had to rule out for PSU applies
   here.

## Cost and the obvious objection

144 masked passes per sample against PSBD's 3 is a 48x inference cost, which is
real but bounded: this is inference on a held-out pool, not training. If it is
too slow, random subset masking with Shapley-style attribution recovers an
approximate profile in far fewer passes, and grouping heads into blocks of 4
gives a 36-length profile for 36 passes.

The objection that matters more: **concentration may be a property of confident
predictions generally, not of backdoored ones.** That is exactly the
confidence confound [H12](H12-psu-is-not-just-confidence.md) had to rule out for
PSU, and it has to be ruled out here the same way, against the same
confidence-only null. Step 5 is not optional.
