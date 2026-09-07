# What this project can claim, and how strongly

Written after a literature review found that the framing this project had been
building toward is partly published already. Everything below is graded by how
well it survives that, and by whether the number behind it currently reproduces.

Status key: **solid** means measured, reproducible, and not found in prior art.
**Contested** means a reviewer has a specific counter available. **At risk** means
a number or a claim that does not currently reconcile and must be fixed or
dropped before submission.

## Before anything else: we are not solving PSBD's task

PSBD is a **training data filter**. Its threat model gives the defender full
control of training, and its clean validation set is, verbatim from the paper,
"the clean data we used to filter backdoor training data, which was 5 percent of
the total quantity of whole training dataset". It scores the **poisoned training
set** and removes samples from it.

This project scores the **clean test set with triggers applied**. Every split in
`psbd/splits.py` comes from `load_clean_test_base`. That makes this **test time
input detection**, a different task with a different threat model and a **disjoint
set of competitors**.

The divergence is defensible and arguably the more useful task, but it has to be
declared in the abstract rather than discovered by a reviewer. Two consequences
follow immediately.

**The baseline set changes.** For test time input detection the competitors are
STRIP, SCALE-UP (both variants), TeCo, IBD-PSC and BaDExpert, not the training set
filters (Spectral Signatures, Activation Clustering, SPECTRE, SCAn) that PSBD
benchmarks against. BaDExpert (ICLR 2024) is the one that already covers ViT-B/16
on ImageNet and is the strongest baseline we currently do not have.

**One published rule half does not transfer, and the code already knows it.**
`psbd.decision.select_rate_adaptively` documents that PSBD's second rate selection
criterion, maximizing the gap between the training set shift ratio and the clean
validation shift ratio, "carries signal in the paper because their scored pool is
the poisoned training set. Ours is a clean test pool, so that gap is noise around
zero by construction." That reasoning is correct and it is evidence the divergence
was noticed. It was never elevated out of a docstring.

## The claims, ranked

### 1. Where you inject dominates what you inject (solid)

Position and operator are separable axes of the same design space, and on ViT the
position carries about 1.43 times the variance of the operator. No prior
perturbation consistency detector factors the design space at all: STRIP
superimposes inputs, SCALE-UP amplifies pixels, IBD-PSC scales normalization
parameters, PSBD drops activations, and each proposes exactly 1 pair. None
compares at matched perturbation strength, so none can separate the site from the
noise.

The split that carries the effect is input-side against residual-adjacent,
**+0.054 mean AUROC, bootstrap CI [+0.031, +0.080]**, and it reproduces from the
cached sweep to 4 decimal places. The founding pre-versus-post distinction is
**+0.002**, indistinguishable from noise.

This is the methodological result and it is the one that survives every objection
so far. It also now has a theoretical reason: decision boundaries are flat along
most directions and curved along few (Fawzi et al., CVPR 2018), so an isotropic
probe spends nearly all its budget on flat directions and which subspace the probe
lands in matters more than how the probe is shaped.

### 2. The sigma matching protocol (solid, and it costs us elsewhere)

$\operatorname{tr}(H\Sigma)$ is homogeneous in $\Sigma$, so comparing 2
placements at a shared nominal rate compares whichever perturbs harder. Matching
on clean validation shift ratio is the only way to isolate the position, and the
matching has to be exact rather than nearest.

Applying our own protocol honestly costs us our biggest headline number. See
"at risk" below.

### 3. PSBD's published mechanism is wrong on this architecture (solid)

The original paper explains the statistic by a neuron bias effect: under
perturbation, clean samples collapse onto the attacker's target class. Tested
head on, over shifted clean predictions on `vit_cifar100_badnet_a2o_0_01`:

| quantity | value |
|---|---:|
| share landing on the target class | **0.0000** |
| uniform expectation | 0.0100 |
| largest share taken by any class | 0.5343 |
| the class taking it | 55, not the target |

Not one shifted prediction landed on the target class. Gaussian noise, which
removes no capacity and so has no neuron bias story available to it, separately
matches the best structured masks at 0.950.

This is a negative result about a published explanation rather than about a
published method, and it should be stated that way. DetectGPT and NETE could not
attempt it because neither had a mechanistic claim to falsify.

### 4. Structured masking is a dead end while the backdoor is not axis aligned (solid)

A structured operator shapes $\Sigma$'s eigenbasis to be axis aligned. That only
concentrates $\operatorname{tr}(H\Sigma)$ on the backdoor if $H$'s large
eigendirections are axis aligned too, and they are not. Four independent negative
results with 1 cause: head masking 0.539, targeted 3-head masking 0.580, the
144-head sensitivity profile carrying no signal, channel masking losing to dropout
at every matched position.

Measured a second way: at the peak layer, 50 percent of total TAC is carried by
291 of 768 dimensions and 90 percent by 651. A backdoor that were a handful of
neurons would be cheap to remove. This one is not.

### 5. The backdoor direction is causally sufficient (solid)

Adding the direction to a clean batch's residual stream, with no trigger in the
pixels, flips 43 percent of clean images to the target at the direction's own
natural magnitude while 58 percent keep their original prediction. A benign
model's own paired difference steers nothing at any scale (0.007 flat).

This upgrades the latent findings from correlational to causal, which is what
separates a symptom from a mechanism.

### 6. Effective rank collapse discriminates backdoored from benign (DOWNGRADED)

**Read the three limits below before quoting any of this.** The measurement is
real and unpublished, but 3 separate results cut it down from a mechanism to an
observation.

**It is not causally tied to the classifier's output.** The head reads features
only through its weight matrix, 100 by 768 on CIFAR-100, so a **668 dimensional
null space** is invisible to the logits. Moving energy inside it takes the rank
ratio from **0.210 to 0.999 while the logits change by at most 1.9e-5**, verified
directly. An attacker can therefore set the rank ratio to nearly anything without
touching predictions, ASR, or any downstream statistic. The "high ASR forces
dimensional collapse" argument this document previously proposed as a theoretical
contribution is **false**, and the naive contraction argument behind it does not
survive: a class region is C-1 linear inequalities, which constrains no
dimensionality at all.

**The scope is narrower than the claim.** The result is CIFAR-100 at 1 percent, 6
checkpoints. The class-matched baseline swings from above 1 at 10 classes to 0.085
at 200, so **a single threshold is not valid across the 4 datasets** and the
number is not comparable between them without normalization.

**We have not tested the attack that matters.** `attacks/adaptive_blend.py`
implements 1 of Qi et al.'s 3 mechanisms and says so in its own docstring. The
mechanism we did implement, cover samples, acts on a first moment that a
mean-centred statistic removes before measuring. So the observation that our
`adaptive_blend` checkpoints still collapse is **not** evidence that the published
adaptive attack fails against this measure. That experiment has not been run.

What survives: models trained the ordinary way do collapse, benign controls do
not, and no one has published the measurement. That is worth reporting as an
empirical signature with its limits stated. It is not a mechanism and it is not a
detector.

### 6b. The original framing, retained for reference (contested)

Measured on paired data, backdoored models confine triggered inputs to roughly
half the directions clean data occupies, and blend to a fifth, while benign models
probed with the same trigger stay near 1. Raw separability does not discriminate
at all: it reaches 0.96 even on a benign model.

Contested for 2 reasons. It is a latent separability signature, and latent
separability signatures have a known adaptive attack (Qi et al., circumventing
latent separability). And whether it predicts detection performance is being
measured rather than assumed, in
`experiments/latent_geometry_predicts_detection/`.

### 7. The randomized smoothing bridge (solid if it holds, and unclaimed)

$\sigma\Phi^{-1}(g(h) - \phi(x))$ is a Cohen-style certified $\ell_2$ radius at
the probe site, strictly decreasing in the detection statistic. Exact, needing no
Taylor expansion, and immune to the reparameterization objection that sinks the
curvature route.

The empirical observation that backdoor inputs sit in locally flatter regions is
not new. The formal chain from detection statistic to smoothed confidence to a
Neyman-Pearson certified radius to margin appears genuinely unmade. This is
probably the most defensible theoretical contribution available.

### 8. Antithetic probe pairing (predicted, being measured)

The one sided estimator retains a first order term whose variance
$\sigma^2\|\nabla g\|^2$ dominates the second order signal. Averaging
$g(h+\delta)$ and $g(h-\delta)$ cancels it exactly, and the multiplicative
analogue is the complementary mask. No detector in this family does it, and it is
free at a fixed pass budget.

### 9. Probe diversity as a defence, and the union rule it needs (solid)

An adaptive attacker minimizing the gap for a probe with covariance $\Sigma_1$
constrains only $H$'s projection onto that eigenbasis. The probed operator's
AUROC collapses 0.952 to 0.322 while transfer operators still detect at 0.887
mean, and the min-rank union of 4 probes recovers 0.951.

**New, and it changes the recommendation.** The attacker's best move is not to
neutralize the probed position but to **invert** it. Measured on the evasive
checkpoint, the probed position goes 0.982 to **0.023** while an unprobed position
still reads 0.913. An inverted probe contributes confident wrong evidence, and a
min-rank union takes the most extreme evidence available, so it adopts the
inversion: 1 inverted probe of 3 takes the union from 0.969 to 0.621, and the
calibrated threshold's achieved FPR goes to 0.825 against a 0.25 target because
the inversion corrupts the validation reference too.

A **median-rank** reduction dominates min on synthetic probes and **loses on real
ones**: over all 120 evasive checkpoints, min 0.702 against median 0.673, with min
ahead on 65 of 120 and still ahead on the 110 where a probe is genuinely inverted.
The synthetic construction gave the inverted probe a symmetry real ones do not
have. **Retracted as a contribution.**

The larger finding stands and is less comfortable: **neither union beats the best
single probe** (-0.065 for min, -0.094 for median). The union is defensible as a
threat-model argument, an attacker must evade every probe rather than guess the
deployed one, and it is not an AUROC improvement. Report it that way.

## What is at risk and must be resolved before submission

**The +0.258 headline is WITHDRAWN.** Audit finding A16. It is reproducible but
unmatched: the winner is read at shift ratio 0.95 to 0.98 and the baseline at 0.65
to 0.76, a gap the same size as the effect. At matched shift ratio over the full
48-cell panel that arm gains **-0.007**, and it is a deterministic operator immune
to a Monte Carlo penalty worth about 0.03 that the stochastic baseline pays.
Replaced by `token_mask` at `before_attention_norm`: **+0.089** over the full panel
at matched sigma, stable +0.081 to +0.103 across sigma 0.2 to 0.8, and **+0.166**
on CIFAR-100 at 1 percent.

**The 0.911 headline survives verbatim, over 48 of 48 cells.** An earlier note in
this document claimed only 39 exist and that the placement shows 41 inversions;
both were wrong and are retracted. Coverage is complete, unexplained inversions on
normally trained implanted non-all-to-all checkpoints are **0**, and the 43
inversions that do occur are on `_evade_*` checkpoints trained specifically to
defeat this probe, which is a different claim.

**`detection_summary.csv` carried 3029 superseded rows mislabelled as the paper's
own baseline**, inflating dropout's mean AUROC by +0.036 and understating every
margin measured against it. Fixed, and the CSV regenerated.

**TaCT's poison rate is silently capped and was unrecorded.** 70 checkpoints. Any
TaCT poison rate trend on CIFAR-100, GTSRB or Tiny is not a poison rate trend.
Fixed in the backfill.

**Every number is single seed.** This is the largest missing piece and the most
likely reason for rejection at a security venue. See `docs/seed-replication-plan.md`.

**No EOT over the dropout randomness.** Athalye et al. (ICML 2018) make Expectation
Over Transformation mandatory for evaluating any defence with a stochastic
component, and PSBD's own adaptive section does not do it. For a dropout based
detector this is the single most likely reviewer objection, and the fix is
mechanical: average the attacker's gradient over sampled masks, and run the
Athalye sanity check of whether the attack still succeeds with the mask frozen.

**Position, not the statistic, is the axis nobody has attacked.** Every published
break in this family targets a statistic. If the deployed site is a secret or
randomized parameter, then what a position-aware attacker costs is a security
property rather than an ablation. That is a contribution the position search
uniquely enables and it is currently unmeasured.

**Rank ratio must not be framed as a standalone detector.** It stamps a known
trigger, and a defender who knows the trigger has in most threat models already
won, so the benign end near 0.98 is close to tautological. Frame it as the
mechanistic account of why the margin signal exists, and **regress it against ASR
across checkpoints**: if the fit is tight, we re-derived ASR the hard way and
should say so.

## What cannot be claimed

The derivation $\phi \approx -\tfrac{1}{2}\operatorname{tr}(H\Sigma)$ via
Hutchinson is DetectGPT's (ICML 2023). Its application to backdoor sample
detection is NETE's (Neural Networks 2025). "Backdoor detection is curvature
estimation" is taken. What survives is the activation space formulation, the
position and operator factorization, and the mechanism refutation.

## Venue

The strongest framing is no longer "a better detector". It is: perturbation
consistency detection is margin estimation, the attachment site dominates the
operator, and the published mechanism for the method we port does not hold on
transformers.

- **SaTML** and **AISec** fit what exists today.
- **USENIX Security, CCS, NDSS** become reachable once the operating points are
  corrected, the baseline set is broadened to the real SCALE-UP, IBD-PSC and
  TeCo rather than our operator ports, and seeds are added. The adaptive attacker
  is already the strongest part of the security story.
- **NeurIPS or ICLR** on the mechanism framing, but only with the smoothing
  bridge as the theoretical core rather than the curvature expansion.
