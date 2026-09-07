# Perturbation consistency as curvature estimation

The contribution this project can make that no prior paper makes is not a better
detector. It is a statement about what the whole family of perturbation
consistency detectors is measuring, which falls out of a second order expansion
and then predicts 4 results the project already has.

## Setup

Fix a probe site inside the network. Write the activation there as $h = h_\ell(x)$
and the map from that site to the predicted class probability as $g$, so that

$$P_c(x) = g(h_\ell(x)), \qquad c = \arg\max_j P_j(x)$$

An operator perturbs that activation, $\tilde{h} = h + \delta$, where $\delta$ is
the operator's noise. **Most, but not all, operators in this study are mean
preserving.** Inverted dropout divides survivors by the keep probability, the
masks do the same, and the additive noise operators are zero mean. So for those

$$\mathbb{E}[\delta] = 0, \qquad \operatorname{Cov}[\delta] = \Sigma$$

| symbol | meaning |
|---|---|
| $h$ | activation at the probe site |
| $g$ | map from the probe site to the predicted class probability |
| $c$ | the class the unperturbed model predicts |
| $\delta$ | the operator's perturbation |
| $\Sigma$ | covariance of the perturbation |
| $H$ | Hessian of $g$ with respect to $h$, evaluated at $h$ |
| $\phi$ | prediction shift uncertainty, PSBD's statistic |

**The 2 exceptions matter and were originally missed here.** `GainScale` and
`ScaleUp` apply $\delta = \text{rate} \cdot x$, so $\mathbb{E}[\delta] \neq 0$ and
assumption 3 below does **not** hold for them. The first order term does not
vanish, so the derivation's conclusion that PSU is a curvature rather than a
gradient measurement does not apply to those 2 operators.

That is not a footnote. They are the **only** operators swept at
`attention_norm_out`, `mlp_norm_out`, `final_norm_out` and `input_pixels`, which
is 4 of the 15 positions in the depth table, including the head position the
prediction below is tested at. Any statement about those 4 positions is a
statement about a different estimator than the one derived here.

## Result

$$\phi(x) \;\approx\; -\tfrac{1}{2}\operatorname{tr}\!\big(H(x)\,\Sigma\big)$$

The position sets $H$. The operator sets $\Sigma$. They enter through a trace,
which is why they are separable and why one of them can dominate.

## Derivation

Assumptions, stated before the work rather than found inside it:

1. $g$ is twice differentiable at $h$, which holds since it is a softmax over a
   composition of smooth layers.
2. $\delta$ is small enough that third order terms are negligible. This is the
   binding assumption and it fails at large perturbation rates, which is exactly
   where the measured inversions appear.
3. $\mathbb{E}[\delta] = 0$, true for every operator here by construction.

$$
\begin{aligned}
\phi(x) &= g(h) - \mathbb{E}_\delta\big[g(h + \delta)\big] && \text{definition of PSU} \\
g(h + \delta) &\approx g(h) + \nabla g^\top \delta + \tfrac{1}{2}\delta^\top H \delta && \text{Taylor to 2nd order, assumption 1 and 2} \\
\mathbb{E}\big[g(h + \delta)\big] &\approx g(h) + \nabla g^\top \mathbb{E}[\delta] + \tfrac{1}{2}\mathbb{E}\big[\delta^\top H \delta\big] && \text{linearity} \\
&= g(h) + \tfrac{1}{2}\mathbb{E}\big[\delta^\top H \delta\big] && \text{assumption 3 kills the 1st order term} \\
&= g(h) + \tfrac{1}{2}\operatorname{tr}(H\Sigma) && \text{since } \mathbb{E}[\delta^\top H \delta] = \operatorname{tr}(H\,\mathbb{E}[\delta\delta^\top]) \\
\phi(x) &\approx -\tfrac{1}{2}\operatorname{tr}(H\Sigma)
\end{aligned}
$$

The first order term vanishing is the substantive step. It is why PSU is a
curvature measurement rather than a gradient measurement, and it is a consequence
of inverted scaling, a detail every one of these methods inherited from dropout
without arguing for it.

Sign check: PSU is empirically positive, so $\operatorname{tr}(H\Sigma) < 0$. That
is right. $g$ is the probability of the arg max class, so $h$ sits at a local
maximum of $g$ along most directions and the curvature is negative.

## Why this is a margin statement

**A note on the order of the argument, corrected.** This section originally ran
curvature first and derived margin from it. Fawzi, Moosavi-Dezfooli and Frossard
(NeurIPS 2016) Theorems 1 and 2 run it the other way: a random direction probe
estimates the **margin** to leading order, with curvature entering as the bias
term under an explicit curvature condition. Singla and Feizi (ICML 2020) Theorem 5
likewise **bounds** curvature under Gaussian smoothing, $-I/s^2 \preceq \nabla^2
\hat{g} \preceq I/s^2$, rather than measuring it.

Lead with margin and treat curvature as the second order correction. It is better
supported and it is the order the literature already establishes. Singla and Feizi
is also the strongest citation available for H23: the curvature budget under
smoothing is $1/\sigma^2$, with no reference to neuron capacity at all, which is
exactly why an operator that removes no capacity can match one that does.

Closest published form of the expansion itself: Camuto et al. (NeurIPS 2020,
arXiv:2007.07368) Theorem 1 gives a **layer indexed** version,
$\mathbb{E}[\Delta L] = \mathbb{E}[\tfrac{1}{2}\sum_k \sigma_k^2
\operatorname{tr}(J_k^\top H_L J_k)]$, which maps directly onto a position
registry and is the nearest thing in print to the expression this document derives.


For a softmax, curvature collapses as the decision margin grows. Let $m$ be the
logit gap between the predicted class and the runner up. As $m$ grows, $P_c \to 1$
and every second derivative of $P_c$ tends to 0, because a saturated softmax is
flat.

So $\|H\| $ is a decreasing function of margin, and therefore

$$\phi(x) \;\propto\; \text{curvature at } x \;\propto\; \text{(decreasing in)}\; \text{margin}(x)$$

A backdoored input sits far from the boundary in the direction the trigger pushes,
so its margin is large, its curvature is near 0, and its PSU is low. A clean input
sits nearer a boundary, so its curvature is large and its PSU is high. **Low PSU
means poisoned** is not a heuristic, it is what the expansion says.

This is the reframing: PSBD, STRIP, SCALE-UP and IBD-PSC are 4 estimators of the
same quantity, differing only in $\Sigma$ and in where they evaluate $H$.

## The 4 predictions, and what was measured

The value of a framework is what it forbids. This one makes 4 predictions that
were tested independently of it.

**1. Removal is not required.** $\Sigma$ enters only through $\operatorname{tr}(H\Sigma)$.
Nothing in that expression cares whether $\delta$ removes a unit or merely
displaces it, so a continuous perturbation should work as well as a structured
mask at matched scale.

Measured: Gaussian noise scores 0.950 against dropout's 0.911 and every structured
mask below it. H23, recorded as the project's most consequential result and
predicted here.

**2. Structured masking cannot help, given non axis aligned backdoors.** A
structured operator shapes $\Sigma$'s eigenbasis to be axis aligned, whether by
channel, by head or by token. That only concentrates $\operatorname{tr}(H\Sigma)$
on the backdoor if $H$'s large eigendirections are themselves axis aligned. H16
established the opposite: removing 1 rank-1 direction takes ASR from 1.00 to 0.00
while zeroing 300 of 768 coordinates does nothing, so the backdoor direction is
rotated off the coordinate basis.

Measured: head masking 0.539, targeted 3-head masking 0.580, the 144-head
sensitivity profile carrying no signal, channel masking losing to dropout at every
matched position. H22, H35, H18 and H26, 4 independent failures with 1 cause.
**This framework explains why that whole research direction is a dead end**, which
is more useful than any of the 4 results alone.

**3. Position should dominate operator.** $H$ varies by orders of magnitude across
depth, since residual stream norms grow monotonically and the backdoor direction
crystallizes only at layers 8 to 10. $\Sigma$ is normalized away by matching on
shift ratio, which is a scale statistic.

Measured: position variance 1.43 times operator variance, Kendall tau 0.700 for
attack ordering across operators at fixed position. The operator changes the level
and not the ordering, which is what a trace factorization implies.

**4. Evasion must be probe specific.** An adaptive attacker minimizing the gap for
a probe with covariance $\Sigma_1$ can only constrain $H$'s projection onto
$\Sigma_1$'s eigenbasis. $H$ has $d^2$ degrees of freedom and $\Sigma_1$ pins a
subspace of it, so a second probe with $\Sigma_2$ not aligned to $\Sigma_1$ reads
a projection the attacker never constrained.

Measured: the probed operator's AUROC collapses 0.952 to 0.322 while transfer
operators still detect at 0.887 mean, and the min-rank union of 4 probes recovers
0.951. H25 and H41.

Prediction 4 is the one that turns the theory into a method. **Operator diversity
is defence in depth, and the reason is that an attacker cannot flatten a Hessian
in every direction at once without destroying the model.**

### But the union rule that implements it has a fragility the theory did not anticipate

Prediction 4 says an attacker constrains only 1 subspace, so a second probe still
reads something. It says nothing about **what** the second probe reads, and the
measured answer is worse than "nothing".

On the evasive checkpoint, the probed position does not merely stop working. It
**inverts**: AUROC 0.982 becomes **0.023**, far below chance, while an unprobed
position still reads 0.913. An inverted probe does not contribute useless
evidence, it contributes confident wrong evidence, and the min-rank union takes
the most extreme evidence available, so it adopts the inversion.

Measured on synthetic probes with a known answer:

| probes | min-rank | median-rank |
|---|---:|---:|
| 3 healthy | 0.969 | **0.986** |
| 2 healthy, 1 inverted | 0.621 | **0.833** |
| 5 healthy | 0.981 | **0.996** |
| 4 healthy, 1 inverted | 0.792 | **0.967** |
| 3 healthy, 2 inverted | 0.589 | **0.751** |
| 2 healthy, 3 inverted | 0.397 | 0.215 |

The union bound also stops describing the situation: with an inverted probe the
calibrated threshold achieves FPR 0.825 against a 0.25 target, because the
inversion corrupts the clean validation reference the threshold is read from.

**A median-rank reduction appeared to fix it, and does not.** On the synthetic
probes above it dominates min everywhere until the attacker controls a majority.
Run on all 120 real evasive checkpoints it **loses**: min 0.702 against median
0.673 mean AUROC, min winning on 65 of 120, and min still ahead (0.683 against
0.654) on the 110 where a probe is genuinely inverted.

The synthetic inverted probe was built symmetrically, so it assigned aggressively
low ranks to clean samples and min inherited them as false positives. Real
inverted probes do not do that: the healthy probes still produce extreme low ranks
for backdoor samples, which min captures and median averages away. The property
that made min fragile in simulation is not the property real inverted probes have.

**And neither union beats the best single probe**, by -0.065 and -0.094. The union
survives as a threat-model argument, that an attacker must evade every probe
rather than guess the deployed one, and not as an AUROC improvement. See
`experiments/median_rank_union/`.

This matters more than a tuning change. The union was the answer to the adaptive
attacker, and inverting a single probe was the attacker's best move against it.

## Why the sigma matching protocol is the right one, from the same expression

$\operatorname{tr}(H\Sigma)$ is homogeneous in $\Sigma$. Doubling the perturbation
scale doubles the statistic without changing anything about the position. So
comparing 2 placements at a shared nominal rate compares
$\operatorname{tr}(H_1\Sigma_1)$ against $\operatorname{tr}(H_2\Sigma_2)$ with
$\Sigma_1 \neq \Sigma_2$, and the winner is decided by whichever perturbs harder.

Matching on clean validation shift ratio fixes the scale of $\Sigma$ empirically,
which is the only way to isolate $H$. That is why the protocol exists, and the
expansion says it is not optional. It also says the matching has to be exact
rather than nearest, which is why cells are now read by interpolating to the
target rather than by taking the closest swept rate.

## The trace estimator reading, and what it predicts

$\operatorname{tr}(H\Sigma)$ for an isotropic $\Sigma = \sigma^2 I$ is
$\sigma^2 \operatorname{tr}(H)$, and estimating a trace as

$$\operatorname{tr}(A) = \mathbb{E}\big[z^\top A z\big],
\qquad \mathbb{E}[z] = 0, \quad \mathbb{E}[zz^\top] = I$$

is Hutchinson's estimator. So **PSBD run with isotropic noise is Hutchinson's
trace estimator applied to the Hessian of the predicted class probability**, and
the $k$ forward passes are its $k$ probe vectors. That is not an analogy. It is
the same expression, and it brings 40 years of numerical linear algebra to bear
on a question the backdoor literature has been answering by intuition.

The immediate consequence is a variance budget. Hutchinson's variance is known in
closed form for both standard probe distributions:

$$\operatorname{Var}_{\text{gaussian}} = 2\|H\|_F^2, \qquad
\operatorname{Var}_{\text{rademacher}} = 2\left(\|H\|_F^2 - \sum_j H_{jj}^2\right)$$

Rademacher is smaller by exactly the diagonal energy, and it is the minimum
variance choice among probe distributions with independent entries. Both estimate
the same expectation, so at a matched scale they must agree as $k$ grows and
differ only in how fast they get there.

This yields prediction 5, which is the first prediction this framework has made
that is not a post hoc explanation of a result already in hand.

**5. A Rademacher probe should match Gaussian in the limit and beat it at small
$k$.** Implemented as `psbd.operators.RademacherNoise`, and the estimator
identity is verified directly on a known matrix in
`tests/test_trace_estimator_operators.py`.

The same test records the honest caveat. A symmetric matrix with independent
entries puts about 1 part in $d$ of its energy on the diagonal, so the predicted
advantage is about 3 percent at $d = 64$ and smaller as $d$ grows. **The
advantage is only worth having if the Hessian of a softmax output is diagonally
dominant, which is an empirical question about the network and not something the
estimator theory settles.** A measured tie between Rademacher and Gaussian is
therefore evidence about $H$'s structure rather than a refutation of the
framework; a measured loss for Rademacher would refute it.

## Where the margin comes from: collapse and curvature are the same fact

The margin argument above says PSU falls when the margin is large. It does not say
why a triggered input has a large margin. The latent measurements answer that, and
the answer closes the loop.

Measured on paired data, the same images with and without the trigger
(`notebooks/02-architecture-and-attack-comparison.ipynb`):

| model | min CKA | min rank ratio |
|---|---:|---:|
| ViT, badnet, 1 percent | 0.430 | 0.439 |
| ViT, blend, 1 percent | 0.250 | 0.199 |
| ViT, sig, 1 percent | 0.625 | 0.625 |
| ViT, wanet, 1 percent | 0.964 | 0.976 |
| **ViT, benign, same trigger** | **0.984** | **0.966** |
| Swin, badnet, 1 percent | 0.446 | 0.632 |
| Swin, blend, 1 percent | 0.210 | 0.212 |
| **Swin, benign, same trigger** | **0.995** | **0.988** |

The rank ratio is the effective rank (participation ratio) of the triggered
population divided by that of the clean population at the same layer, so the
sample size bias cancels between the 2 halves.

A backdoored model confines triggered inputs to roughly half the directions clean
data occupies, and in the blend case to a fifth. A benign model perturbed by the
same trigger does not: its representation is displaced and keeps its shape. WaNet
at 1 percent, which is independently known not to implant, is indistinguishable
from the benign control on both columns, so the measure tracks whether a backdoor
was learned rather than whether a trigger was applied.

The link to curvature is direct. Write $r(x)$ for the effective dimension of the
locally class-discriminative subspace at $x$. Only those directions carry
appreciable curvature of $P_c$, because a direction along which no class competes
with $c$ is a direction along which $P_c$ is flat. So

$$\operatorname{tr}(H) \;\approx\; r(x)\,\bar{\lambda}(x)
\qquad\Longrightarrow\qquad
\phi(x) \;\approx\; -\tfrac{\sigma^2}{2}\, r(x)\, \bar{\lambda}(x)$$

**PSU scales with the effective dimension of the subspace the input lives in.** A
triggered input routed onto a low dimensional attractor has small $r$, hence small
$\operatorname{tr}(H)$, hence small PSU, and is flagged.

### Do not claim these are the same geometry

An earlier draft of this section said the margin story and the collapse story are
"one quantity seen from 2 sides". **That claim is not defensible and there are at
least 4 published counterexamples pointing the other way.**

- Sukenik, Mondelli and Lampert (NeurIPS 2024, arXiv:2405.14468) Theorem 5: deep
  neural collapse is **not** optimal, lower rank solutions beat it, and the
  equiangular half is violated. Regularization favours rank reduction **over**
  orthogonality, so collapse and maximal separation are not the same optimum.
- Ma et al. (ICLR 2018) measure adversarial inputs at LID about 4.36 against about
  1.53 for normal inputs, so attacked inputs are locally **higher** dimensional.
- COLLIDER (ACCV 2022) filters backdoor training data on the premise that clean
  samples have **low** LID, the same sign again.
- Moosavi-Dezfooli et al. (CVPR 2017) Theorems 1 and 2: boundary normals
  concentrated in a low dimensional subspace is exactly the condition for a
  **small** universal perturbation, which is the opposite of a large margin.

What is defensible is weaker and still worth having:

> Margin maximization and rank collapse are **coupled outputs of the same implicit
> bias of gradient descent on deep homogeneous networks**, not the same quantity.
> Ji and Telgarsky (ICLR 2019) prove that in deep linear networks the max margin
> limit and the rank-1 limit are literally the same limit; Timor, Vardi and Shamir
> (ALT 2023) Theorem 5 extends the rank half asymptotically in depth, with their
> Theorem 2 giving the depth-2 counterexample as the caveat. A trigger is a
> feature the network fits faster and more completely than any clean feature, so
> triggered inputs sit deepest inside this jointly collapsed, jointly high margin
> regime. Probing by rank and probing by margin measure **2 correlated signatures
> of 1 optimization limit**, not 2 views of 1 quantity.

**Publish the measurement, not the identity.** The measurement is unclaimed. The
identity has 4 theorems aimed at it.

### The sign conflict is itself a measurement, and it is now instrumented

LID and the participation ratio are not in contradiction even though they carry
opposite signs in the literature. LID is a **local** neighbourhood expansion rate
at 1 point; the participation ratio is a **global** second moment property of a
population. A sample can sit in a locally sparse region, so high LID, while the
population it belongs to occupies few directions, so low participation ratio.

That is an argument until both are measured on the same features, so both now are:
`psbd.analysis.distribution.local_intrinsic_dimensionality` is reported beside
`rank_ratio` in every layer row. **If triggered inputs read low LID where
adversarial inputs read high, that separates backdoors from adversarial examples
on a quantity 2 published papers use, and it is a result in its own right.**

That gives prediction 6, and it is the sharpest one available because it is
cross modal: it links a quantity measured with no perturbation at all to the
output of a perturbation based detector.

**6. Rank ratio should predict detection AUROC across checkpoints.** *(Weakened.
The head's weight matrix leaves a 668 dimensional null space on CIFAR-100, and
moving energy inside it takes the rank ratio from 0.210 to 0.999 with the logits
unchanged to 1.9e-5. So nothing forces the collapse and this prediction is about
what trained models happen to do, not about what a backdoor requires. Measured
anyway, because an empirical regularity with a stated limit is still worth having,
but it can no longer carry the mechanism claim.)* Being able
to predict a detector's performance without running it is a strong test, and a
null result is informative: it would mean the collapse is a correlate of the
backdoor rather than the cause of the margin. Measured in
`experiments/latent_geometry_predicts_detection/`.

## What the mechanism is not

The original PSBD paper explains its statistic by a neuron bias effect: under
perturbation, clean samples do not scatter across classes, they collapse onto the
attacker's target, because that is the strongest association a poisoned model
learned. That is a checkable claim, and this project checked it head on rather
than inheriting it.

Measured on `vit_cifar100_badnet_a2o_0_01` at the adaptively selected rate
(`notebooks/06-psbd-end-to-end.ipynb`), over shifted clean predictions:

| quantity | value |
|---|---:|
| share landing on the attacker's target class | **0.0000** |
| uniform expectation if they scattered evenly | 0.0100 |
| largest share taken by any single class | 0.5343 |
| the class taking it | 55, not the target |

**Not one shifted clean prediction landed on the target class**, against a uniform
expectation of 1 percent, while a majority landed on an unrelated class. The
neuron bias account is not what carries the method on this architecture. The
detector works, and the published explanation of why it works does not survive
contact with a direct test.

The curvature account explains the same data without the collapse-onto-target
claim, and it is the reason the Gaussian control was informative rather than
merely surprising: an operator that removes no capacity at all has no neuron bias
story available to it, and it matches the best structured masks.

## The direction is causally sufficient, and it is not sparse

Two measurements from `notebooks/07-locating-the-backdoor.ipynb` constrain what
the backdoor can be.

Adding the backdoor direction to a clean batch's residual stream at the peak
layer, with no trigger in the pixels:

| steering scale | clean images predicted as target | prediction unchanged |
|---:|---:|---:|
| 0.0 | 0.008 | 1.000 |
| 1.0 | **0.427** | 0.580 |
| 2.0 | 1.000 | 0.008 |
| benign model, any scale | **0.007** | |

At the direction's own natural magnitude, adding it flips 43 percent of clean
images to the target while most predictions remain intact. The scale 2.0 row is
not evidence, since almost nothing keeps its original prediction there and the
representation has simply been destroyed. The benign model's own paired difference
steers nothing at any scale, which is the control that makes the rest meaningful.

So the direction is causally sufficient, not merely correlated. But it is not
sparse: at the peak layer, 50 percent of total TAC is carried by 291 of 768
dimensions and 90 percent by 651. A backdoor that were a handful of neurons would
be cheap to remove, and this one is not. That is the same non axis aligned
geometry prediction 2 relies on, measured a second way, and it is why the 4
structured masking failures were structural rather than tuning problems.

## Prior art, and what it takes off the table

A literature review run against this framing found that the central derivation is
**already published, twice**. This section exists so that nobody rediscovers it in
review.

**Taken: the derivation.** Mitchell et al., DetectGPT (ICML 2023,
arXiv:2301.11305), derive that for symmetric zero mean noise,
$f(x) - \mathbb{E}_z[f(x+z)] \approx -\tfrac{1}{2}\operatorname{tr}(H_f(x))$, via
the same second order expansion and the same Hutchinson identity, and use it to
detect machine generated text.

**Taken: the derivation applied to backdoor detection.** Peng et al., NETE
(Neural Networks 2025, arXiv:2509.05318), apply exactly that to backdoor sample
detection in language models, and state the Hutchinson trace connection
explicitly.

**Taken: the margin half, with certified radii, 3 years ago.** Rajabi et al., MDTD
(ACM CCS 2023, arXiv:2308.15673), states it outright: "input samples containing a
Trojan trigger are located relatively farther away from a decision boundary than
clean samples", and reports certified radii for clean against Trojan samples
(CIFAR-100 clean 0.005, BadNets 0.657). Liu et al., TrojDef (IEEE TNNLS,
arXiv:2209.01721), perturbs inputs with Gaussian noise and states "Trojan poisoned
training makes Trojan inputs more stable than benign ones".

**So "triggered inputs sit farther from the decision boundary and are more stable
under noise" is not ours either.** Claiming it would be a factual error a reviewer
catches immediately.

What is left on that axis is a real distinction and it should be led with: every
detector in that family, STRIP, SCALE-UP, TeCo, MDTD, TrojDef and NEO, perturbs
the **input**. PSBD and this work perturb the **network**. The right framing is
importing an established input space margin account into model space, citing MDTD
as support rather than as a competitor.

So **"backdoor detection is curvature estimation" cannot be claimed as new.** What
survives is narrower and has to be stated as such:

1. the **activation space** formulation with a position registry, against
   DetectGPT's and NETE's input space formulation,
2. the **position and operator factorization**, which neither has an analogue of,
   since each proposes exactly 1 perturbation,
3. the **refutation of PSBD's published mechanism**, which neither could attempt
   because neither had a mechanistic claim to test.

## Four objections that the framing has to survive

These come from the same review and are recorded before the results rather than
after, because 2 of them are load bearing.

### 1. Reparameterization. This one is fatal as originally stated.

Take the probe site $h$ and rescale $h \to \alpha h$ while scaling the next
layer's weights $W \to W/\alpha$. The function is unchanged: same predictions,
same margins, same AUROC. But $\nabla^2_h g$ scales as $\alpha^{-2}$.

**So $\operatorname{tr}(H\Sigma)$ with a fixed additive $\Sigma$ is a property of
the parameterization, not of the classifier.** This is Dinh et al.'s sharp minima
argument (ICML 2017, arXiv:1703.04933) applied to this expression, and as
originally written, "the position sets $H$, the operator sets $\Sigma$" is not
well posed.

The fix is that $\Sigma$ must scale with the local activation energy, and it is
already in the code rather than added in response. Every masking operator here is
multiplicative, so $\Sigma = \operatorname{diag}(h_i^2 \cdot p / (1-p))$ scales
with $h$ and the trace is invariant. `GaussianNoise` and `RademacherNoise` both
compute `scale = x.detach().std(...)` per sample, so their $\Sigma$ is relative
too. **No operator in this project uses an absolute noise scale**, which is what
makes the comparison well posed.

The honest consequence is that the factorization is not as clean as claimed: for a
multiplicative operator $\Sigma$ depends on $h$ and therefore on $x$, so what is
estimated is an activation energy weighted curvature
$\operatorname{tr}(H \operatorname{diag}(h^2))$ rather than a bare trace.

### 2. Curvature is not margin

An affine classifier has $H \equiv 0$ and arbitrary margin, including arbitrarily
small. Margin is first order, $|f_c - f_k| / \|\nabla(f_c - f_k)\|$, and curvature
is a correction to it, not a substitute. Moosavi-Dezfooli et al.'s CURE bound
(CVPR 2019) makes robustness decrease in curvature only **with the gradient norm
held fixed**.

The replacement is in the next section and it is better than what it replaces.

### 3. The statistic is single peaked in confidence, not monotone

For $p = \operatorname{softmax}(z)$ and $c$ the predicted class, the trace of the
Hessian of $p_c$ with respect to the logits has a closed form:

$$\operatorname{tr}\nabla^2_z p_c = 2\,p_c\big(\|p\|_2^2 - p_c\big) \;\le\; 0$$

since $\|p\|_2^2 \le (\max_i p_i)(\sum_i p_i) = p_c$. Three consequences, and the
third is the one that bites.

**PSU is non negative to second order.** With $\Sigma = \sigma^2 I$ at the logits,
$\phi = \sigma^2 p_c(p_c - \|p\|^2) \ge 0$. Confidence in the predicted class can
only fall in expectation. That is a theorem grade statement supporting the
framing.

**A probe near the head cannot beat a confidence baseline.** At the logit layer
the statistic depends on nothing but $p$. So detection power must decay toward
max softmax confidence as the probe site approaches the classifier. **That turns
the position result from an empirical finding into a prediction**, and it is
directly testable against the position sweep.

**Curvature vanishes at both extremes.** For 2 classes with $p_c = p \ge 1/2$,
$\operatorname{tr} = 2p(2p-1)(p-1)$ is 0 at $p = 1/2$, 0 at $p = 1$, and peaks
near $p \approx 0.789$. So a sample sitting **on** the boundary also has near zero
curvature, and low PSU stops meaning large margin at the low confidence end. This
predicts **inversions concentrated on low confidence clean samples**, which is a
sharp, falsifiable claim about data already collected.

### 4. The variance formulas quoted for the estimator are the wrong ones

The one sided estimator $\hat\phi = g(h) - g(h+\delta)$ retains the first order
term $-\nabla g^\top \delta$. It is mean zero, so it does not bias the result, but
its variance is $\sigma^2\|\nabla g\|^2$ and it **dominates** the second order
signal of size $O(\sigma^2 \operatorname{tr} H)$ whose own variance is
$O(\sigma^4\|H\|_F^2)$.

$$\operatorname{Var}(\text{one sided}) \approx \sigma^2\|\nabla g\|^2 + O(\sigma^4),
\qquad \operatorname{Var}(\text{antithetic}) \approx 2\sigma^4\|H\|_F^2$$

So Hutchinson's $2\|H\|_F^2$ understates this estimator's variance by orders of
magnitude, and the Gaussian against Rademacher comparison in the previous section
is a comparison between 2 terms that are both negligible next to the gradient
noise.

**This also settles prediction 5 before it is measured.** The Gauss-Newton part of
$H$ has rank at most $C - 1$, which is 99 on CIFAR-100 against an ambient
dimension near 151000 at a ViT token grid site. For a low rank matrix with
delocalized eigenvectors the diagonal carries a fraction of order $r/d \approx
10^{-3}$ of the energy, so **Rademacher's advantage over Gaussian is negligible
here, and a measured tie is the predicted outcome rather than a null result.**
The caveat is delocalization: transformers have massive activation outlier
channels, and if the eigenvectors localize the diagonal fraction rises. That is
measurable as $\rho = \|\operatorname{diag}(H)\|_2 / \|H\|_F$.

**The actionable half is antithetic pairing.** Averaging $g(h+\delta)$ and
$g(h-\delta)$ cancels the first order term exactly, and for a multiplicative
operator the analogue is the complementary mask, $m$ paired with $1 - m$. PSBD
does not do this. Neither does STRIP, SCALE-UP, IBD-PSC or NETE. It costs nothing
at a fixed pass budget and the theory says it removes the dominant variance term.
Measured in `experiments/antithetic_probe/`.

## The better bridge: randomized smoothing rather than curvature

Objection 2 removes curvature as a route to margin. There is an exact route that
does not need the Taylor expansion at all.

Cohen, Rosenfeld and Kolter (ICML 2019, arXiv:1902.02918) certify, for a smoothed
classifier under noise of scale $\sigma$, a radius

$$R = \frac{\sigma}{2}\left(\Phi^{-1}(\bar{p}_A) - \Phi^{-1}(\bar{p}_B)\right)$$

derived from the Neyman-Pearson lemma, and prove it tight. Now write
$\tilde{p} = \mathbb{E}_\delta[g(h+\delta)] = g(h) - \phi(x)$, which is exactly
what the sweep already computes. Then

$$R \;=\; \sigma\,\Phi^{-1}\!\big(g(h) - \phi(x)\big)$$

is a certified $\ell_2$ radius **at the probe site**, and it is strictly
decreasing in $\phi$. So

> low PSU $\;\Longrightarrow\;$ high smoothed confidence $\;\Longrightarrow\;$
> large certified radius $\;\Longrightarrow\;$ large margin.

This is **exact**. No second order expansion, no small $\sigma$ assumption, no
Hessian, and it is immune to the reparameterization objection because it is
stated in terms of a probability rather than a second derivative. The curvature
expansion becomes the small $\sigma$ limit of the same statement, which gives the
argument 2 regimes instead of 1:

| regime | statement | status |
|---|---|---|
| small $\sigma$ | $\phi \approx -\tfrac{1}{2}\operatorname{tr}(H\Sigma)$ | approximation, and published already |
| any $\sigma$ | $\sigma\Phi^{-1}(g(h) - \phi)$ is a certified radius | exact, and unclaimed |

Two caveats that have to be stated with it. The certificate is for perturbations
in the space the noise is added to, so it is an $\ell_2$ ball in **activation**
space at the probe site, and converting to input space needs a Lipschitz constant
for the input to $h$ map. And an activation space radius is itself scale
dependent, so objection 1's relative $\Sigma$ requirement applies here too.

There is also a paper a reviewer will raise: Wang et al. (arXiv:2002.11750)
conclude randomized smoothing is **not** an effective defence against backdoors.
That is about certifying robustness to a training set attack. This uses the
smoothed probability as a **detection statistic**, which is a different use, and
their negative result does not transfer. It has to be said explicitly rather than
left for the rebuttal.

## Honest limits

- This is a second order expansion, not a theorem about detection. Assumption 2
  fails at large rates, and that is measurable: inversions concentrate at high
  perturbation, 9.4 percent for input-side positions against 4.1 percent for
  stream positions.
- It explains the ordering of operators and positions. It does not predict the
  absolute AUROC of any cell.
- It does not explain why WaNet is harder than Blend. Both are spatially
  distributed, so the framework does not separate them, and that gap is real.
- It does not explain the clean-label case, where LC has decent AUROC and a
  collapsed TPR at low FPR. The margin story predicts a thin margin and therefore
  a hard case, but not that specific shape.
- $H$ is never computed. The claim is structural, and the evidence is that its
  predictions hold, not that anyone measured a Hessian.
- The step from effective rank to $\operatorname{tr}(H)$ assumes that only class
  discriminative directions carry curvature. That is the right intuition for a
  softmax, but it is an approximation and not a theorem, and the constant
  $\bar{\lambda}$ is doing unexamined work in it.
- Rank ratio is measured on the residual stream at a layer, while $H$ is the
  Hessian at the probe site. Those coincide only when the probe site is that
  layer, so prediction 6 is testing a coarser claim than the derivation states.
- The collapse measure is a latent separability signature, and latent
  separability signatures have a specific known weakness: adaptive attacks
  designed to flatten latent separation. Whether an adaptive-blend style attack
  also flattens the rank ratio is measured rather than assumed, and the
  `adaptive_blend` checkpoints are in the sample.

## What is genuinely new here

Prior work proposes 1 (position, operator) pair each and justifies it after the
fact. STRIP superimposes inputs, SCALE-UP amplifies pixels, IBD-PSC scales
normalization parameters, PSBD drops activations. None of the 4 papers factors the
design space, and none compares at matched perturbation strength, so none of them
can tell whether their gain came from the site or the noise.

5 things follow, in decreasing order of how defensible they are:

1. **The factorization plus the matching protocol.** Position and operator are
   separable axes, the comparison is only meaningful at matched $\Sigma$ scale,
   and on ViT the position carries 1.43 times the variance. This is a
   methodological result and it is the one that survives every objection.
2. **The dead end result.** Structured, unit aligned perturbation cannot beat
   unstructured noise while the backdoor is a non axis aligned direction. This
   retires an entire family of proposals and is supported by 4 independent
   negative results plus a causal ablation.
3. **Operator diversity as a defence.** It follows from the framework, it is
   falsifiable, and it was confirmed against an adaptive attacker that this
   project trained specifically to break the method.
4. **The published mechanism is refuted on this architecture.** The neuron bias
   account predicts shifted clean predictions collapse onto the attacker's target.
   Measured, exactly 0 of them do, against a uniform expectation of 1 percent.
   This is a negative result about a published explanation rather than about a
   published method, and it is stated that way.
5. **Collapse and curvature are one quantity.** Effective rank ratio, measured
   with no perturbation at all, is proposed as the bridge between the latent
   geometry and the detector's output. This is the least settled of the 5 and the
   most interesting if it holds.

The trace estimator identity sits alongside these rather than inside the ranking.
It is not a claim about backdoors at all, it is a recognition that the family's
central computation already has a name, and its value is the machinery it makes
available: variance formulas, minimum variance probes, and the whole
variance reduction literature that has grown up around estimating a trace from a
budget of probes.
