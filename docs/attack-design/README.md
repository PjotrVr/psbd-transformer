# Red team: the strongest attacks we can build against our own detector

Written so that the adaptive attack section of the paper is ours rather than a
reviewer's. Every attack below is designed against **this** project's defence as
it currently stands: a per-input, test time detector that reads prediction shift
under an activation space perturbation, thresholds it at a quantile of clean
validation PSU, and (H41) combines k probes by min rank or median rank.

Nothing here is working attack code. Each file is an objective, a prediction, and
an experiment that would confirm or refute it on our own checkpoints.

## Threat model, stated once

| | |
|---|---|
| Attacker controls | the full training run, the poison indices, the loss, the optimizer, the trigger |
| Attacker knows | the paper, `psbd/positions.py`, `psbd/operators.py`, `psbd/scores.py`, `psbd/decision.py`, the 0.25 quantile rule, the sigma matching rule |
| Attacker does NOT control | the defender's 2000 sample clean validation images, the defender's choice of probe pool, the test inputs |
| Attacker's goal | high ASR on triggered test inputs, clean accuracy within a few points, and the defender's TPR at a deployable FPR driven to 0 |
| Defender's budget | clean validation data with labels, k forward passes per input, no poisoned examples ever |

The last row of the attacker column matters and the paper currently understates
it. The attacker does not need AUROC to reach 0.5. **The attacker needs the
defender's operating point to fail.** Those are different targets and the second
one is much cheaper (see [A1](A1-operating-point-and-threshold.md)).

## The one equation the whole document uses

Fix a probe site $\ell$. Write $h = h_\ell(x)$ for the activation there, $z = z(h)$
for the logits as a function of that activation, $p = \operatorname{softmax}(z)$,
$c = \arg\max_j p_j$, and $g(h) = p_c$. The project's expansion gives
$\phi_\ell(x) \approx -\tfrac{1}{2}\operatorname{tr}(H_h \Sigma_\ell)$. Applying the
chain rule for the Hessian of a composition,

$$H_h \;=\; J^\top \big(\nabla_z^2 p_c\big) J \;+\; \sum_k \frac{\partial p_c}{\partial z_k}\,\nabla_h^2 z_k ,
\qquad J = \frac{\partial z}{\partial h}$$

and using the exact softmax second derivative
$\partial^2 p_c/\partial z_j \partial z_k = p_c\big[u_j u_k - (\delta_{jk}p_j - p_j p_k)\big]$
with $u = e_c - p$, the statistic splits into two terms:

$$\boxed{\;\phi_\ell(x) \;\approx\;
\underbrace{-\tfrac{1}{2}\,p_c\Big[u^\top \Sigma_z u \;-\; \operatorname{tr}(A\,\Sigma_z)\Big]}_{T_1,\ \text{softmax term}}
\;\underbrace{-\;\tfrac{1}{2}\,p_c\sum_k (\delta_{ck}-p_k)\operatorname{tr}\!\big(\nabla_h^2 z_k\,\Sigma_\ell\big)}_{T_2,\ \text{network curvature term}}\;}$$

| symbol | meaning |
|---|---|
| $\phi_\ell$ | PSU at probe site $\ell$, the detector's per-sample statistic |
| $h$ | activation at the probe site |
| $\Sigma_\ell$ | covariance of the operator's perturbation at that site |
| $J$ | Jacobian of the logits with respect to the probe site activation, $C \times d$ |
| $\Sigma_z = J\Sigma_\ell J^\top$ | the **pushforward covariance**, the perturbation seen in logit space |
| $p$ | the softmax vector, $c$ its arg max, $p_c$ the predicted class probability |
| $u = e_c - p$ | one hot residual at the predicted class |
| $A = \operatorname{diag}(p) - p p^\top$ | the softmax covariance matrix |
| $T_1$ | the part that factors through $p$ and the pushforward covariance |
| $T_2$ | the part that comes from curvature of the sub network between $h$ and $z$ |

Sanity check against the project's own closed form. Probe at the logits, so
$J = I$, $\Sigma_z = \sigma^2 I$, and $T_2 = 0$ because the identity map has no
curvature. Then $u^\top u = 1 - 2p_c + \|p\|^2$ and $\operatorname{tr}(A) = 1 - \|p\|^2$,
so $\phi = \sigma^2 p_c\big(p_c - \|p\|^2\big) \ge 0$, which is exactly
$-\tfrac{1}{2}\sigma^2 \operatorname{tr}\nabla_z^2 p_c = -\sigma^2 p_c(\|p\|^2 - p_c)$
from `docs/theory-perturbation-consistency.md` objection 3. The decomposition
above is that closed form with the two things it hides put back in.

Camuto et al. (NeurIPS 2020, arXiv:2007.07368) Theorem 1 is the published form of
$T_1$ alone, $\mathbb{E}[\Delta L] = \mathbb{E}[\tfrac{1}{2}\sum_k \sigma_k^2 \operatorname{tr}(J_k^\top H_L J_k)]$.
It is a Gauss-Newton style approximation and it drops $T_2$. The paper should
cite it for $T_1$ and claim $T_2$ as the term that makes the position axis
non trivial.

## What the closed form buys an attacker, and exactly where it stops

The prompt's lever is right, and it is right in a bounded way that is worth more
to us than it is to a reviewer.

**Where it is fatal.** If the probe sits at or after the last nonlinearity, then
$T_2 = 0$ exactly (the classification head is affine) and $J = W$ is the head
weight matrix, shared by every sample. With any relative operator, $\Sigma_\ell =
\sigma^2(x) I$, the statistic collapses to

$$\phi_{\text{head}}(x) \;=\; -\tfrac{1}{2}\,\sigma^2(x)\, p_c\Big[u^\top W W^\top u - \operatorname{tr}(A\,W W^\top)\Big],$$

a **function of $p$ and one scalar $\sigma^2(x)$ only**. An attacker who matches
the joint distribution of $\big(p, \sigma^2\big)$ between poisoned and clean
populations drives this detector to AUROC 0.5 by construction, whatever operator
the defender picks. Measured on our own cache: at `final_norm_out` the share of
PSU variance explained by a quadratic function of $(p_c, \text{entropy}, \|p\|_2^2)$
is **0.976 to 0.990** across 4 attacks.

**Where it stops.** One block deeper, $T_2$ is nonzero (GELU, LayerNorm and the
attention softmax all have nonzero second derivatives) and $\Sigma_z = J(x)\Sigma J(x)^\top$
is sample dependent. The same measurement gives an explained share of **0.044 to
0.137** across the 4 attack panel, and 0.038 to 0.241 across all 7 checkpoints
measured. Confidence matching therefore removes roughly a tenth of the statistic's
variance at the positions the paper actually recommends, against essentially all
of it at the head.

Measured, `vit_cifar100_*_0_01`, ViT, CIFAR-100, sigma matched to 0.6,
`docs/attack-design/measure_confidence_share.py`:

| probe position and operator | $R^2$ of PSU on $p$ | AUROC PSU | AUROC after removing the $p$ part | AUROC of $p$ alone |
|---|---:|---:|---:|---:|
| `after_embedding` + token_mask | 0.117 | 0.685 | 0.648 | 0.687 |
| `before_attention_norm` + token_mask | **0.076** | 0.944 | **0.946** | 0.687 |
| `before_mlp_residual` + token_mask | 0.092 | 0.920 | 0.919 | 0.687 |
| `attention_norm_out` + gain_scale | 0.193 | 0.762 | 0.718 | 0.687 |
| `mlp_norm_out` + gain_scale | 0.188 | 0.811 | 0.822 | 0.687 |
| `final_norm_out` + gain_scale | **0.984** | 0.314 | 0.470 | 0.687 |

Means over badnet_a2o, blend, lf and bpp at 1 percent. The "AUROC of $p$ alone"
column is the max softmax null and is the same number in every row by
construction, since it never touches the probe.

### The identity that settles it

At `final_norm_out` with `gain_scale`, PSBD is not merely correlated with the max
softmax null. It **is** the max softmax null, with the sign reversed. Across every
checkpoint measured:

| checkpoint | AUROC PSU at `final_norm_out` | AUROC of the confidence null | sum |
|---|---:|---:|---:|
| `vit_cifar100_badnet_a2o_0_01` | 0.248 | 0.752 | 1.000 |
| `vit_cifar100_blend_0_01` | 0.402 | 0.598 | 1.000 |
| `vit_cifar100_lf_0_01` | 0.345 | 0.657 | 1.002 |
| `vit_cifar100_bpp_0_01` | 0.259 | 0.741 | 1.000 |
| `vit_cifar100_adaptive_blend_0_01` | 0.749 | 0.245 | 0.994 |
| `vit_cifar100_lc_0_01` | 0.629 | 0.370 | 0.999 |
| `vit_tiny_badnet_a2o_0_01` | 0.259 | 0.741 | 1.000 |

The 2 AUROCs sum to 1.000 plus or minus 0.006 on 7 of 7 checkpoints, spanning 6
attacks and 2 datasets. The mechanism is exactly the master equation.
`psbd.operators.GainScale` is deterministic and multiplies its input by
$1 + \text{rate}$, and at `final_norm_out` its input is the model's last
LayerNorm output, so up to the head's bias it is a temperature rescaling of the
logits. The resulting shift depends on nothing but $p$. PSU there is a monotone
function of $p_c$ and nothing else, which is why its $R^2$ is 0.98 and why an
attacker who matches $p$ owns it completely.

The caveat that has to travel with it: this exactness is specific to a
deterministic operator. A stochastic operator at the same site would give the
genuine closed form $\phi = -\tfrac{1}{2}p_c[u^\top WW^\top u - \operatorname{tr}(AWW^\top)]$,
which is still a function of $p$ but not a monotone one, so the sum would be near
1 rather than exactly 1. The conclusion for the attacker is unchanged.

This is the theory document's objection 3 in its strongest possible form. "A probe
near the head cannot beat a confidence baseline" understates it: **a probe at the
head is the confidence baseline.** The number is exact, it is free, it is already
on disk, and it belongs in the paper.

Three things follow and all 3 belong in the paper.

1. **The position result is the adaptive attack defence.** The paper currently
   sells "where you inject dominates what you inject" as a methodological
   finding. It is also the reason the strongest, cheapest, most obvious adaptive
   attack fails. A probe's vulnerability to confidence matching is exactly its
   $R^2$ on $p$, and that quantity falls by an order of magnitude as the probe
   moves off the head. State it that way.
2. **The theory's own objection 3 is a prediction we can now show is confirmed.**
   "A probe near the head cannot beat a confidence baseline" is measured:
   `final_norm_out` reads 0.314 against the confidence null's 0.687.
3. **Residualising PSU on $p$ is free and should be adopted.** Fit
   $\mathbb{E}[\phi \mid p]$ on the 2000 clean validation samples the defender
   already holds and score the residual. Across the 6 attacks measured it costs
   nothing (mean 0.886 residual against 0.862 plain at `before_attention_norm` +
   token_mask) and on `adaptive_blend` it **gains 0.121** (0.706 to 0.827). It is
   also the provably correct response to a confidence matcher, because the part
   the attacker equalised is exactly the part it removes.

## Ranking, by damage to the paper if a reviewer gets there first

| rank | attack | what it costs the attacker | what it costs us | file |
|---:|---|---|---|---|
| 1 | **Quantile shadowing.** Sharpen confidence on an input-space-predicated slice of clean data so the clean validation PSU distribution grows a low tail and the 0.25 quantile threshold sinks below the poisoned mass. **Already confirmed in simulation on our own cached scores:** sharpening 5 percent of clean takes AUROC 0.960 to 0.918 and TPR at a 5 percent FPR budget from 0.761 to **0.000**, with achieved FPR still on target. | approximately 0 ASR, under 1 point CA, no probe gradients, no extra forward passes | the deployable operating point, while AUROC barely moves. Our headline metric is blind to it. | [A1](A1-operating-point-and-threshold.md) |
| 2 | **Confidence distribution matching.** Match the poisoned softmax distribution to the clean one in $p_c$, entropy and $\|p\|_2$. | 1 forward pass per step, roughly 4x cheaper than H25's attacker, no probe knowledge at all | the entire confidence family baseline table, every head adjacent probe, and the claim that H25 is our strongest attacker. Our own evasive checkpoints still read 0.970 on the free max softmax null. | [A2](A2-confidence-distribution-matching.md) |
| 3 | **Shared factor multi probe evasion.** One probe free objective on the closed form $p_c(p_c - \|p\|^2)$ moves every probe's $T_1$ in the same direction at once, which is the majority attack the median rank rule fears. | 1 forward pass, no probe registry knowledge | H41, the paper's flagship security contribution, whose threat model assumes probes are independently constrained. | [A3](A3-shared-factor-multi-probe.md) |
| 4 | **Position randomised curvature matching.** Antithetic Hutchinson estimate of $\operatorname{tr}(H_\ell \Sigma)$ at a position sampled per step, hinged on the clean minus poisoned gap. | 2 perturbed passes per step, cheaper and lower variance than H25's k=3 one sided estimator | H25's "evasion must be probe specific" claim, which was measured against an attacker that only ever attacked 1 position. | [A4](A4-curvature-matching.md) |
| 5 | **The low confidence backdoor.** Put triggered inputs near the decision boundary instead of far from it, exploiting the fact that curvature vanishes at both ends of the confidence range. | real ASR fragility, a few points | the confidence, SCALE-UP and IBD-PSC comparison rows, and TPR at 0.01 and 0.05 FPR. It does **not** beat PSU, and we can prove that. | [A5](A5-low-confidence-backdoor.md) |
| 6 | **Architecture transfer.** Whether an evasion recipe designed on ViT works on a ConvNet and back. | none, it is an analysis | the generality claim. Also the place where H22, H26 and H35's "structured masking is a dead end" needs an explicit scope limit. | [A6](A6-architecture-transfer.md) |

## Four escapes that need no adaptive training at all

A reviewer will raise these before any of the 6 above, because they cost the
attacker nothing. They have to be in the paper as attacks, not buried as
limitations.

1. **Pick all to all.** `badnet_a2a` already reads 0.508 in H12's table and H5
   records the failure. Under the master equation the reason is that an all to
   all trigger does not produce a single large margin direction, so $T_1$'s
   $u^\top \Sigma_z u$ term is not inflated for triggered inputs. An attacker
   who wants to evade PSBD and does not care about a single target class simply
   uses a2a and pays nothing.
2. **Aim for a modest margin.** Nothing in an attacker's objective requires ASR
   0.999. Training to ASR 0.95 with a mid range target probability is close to
   free and moves triggered inputs off the saturated end of the curvature curve.
   This is the cheap, no penalty version of [A5](A5-low-confidence-backdoor.md).
3. **Exploit the AUROC to operating point gap that already exists.** On
   `vit_cifar100_badnet_a2o_0_01` at `before_mlp` + gaussian, AUROC is 0.892 and
   TPR at 5 percent FPR is **0.01**. At `attention_norm_out` + gain_scale, AUROC
   0.804 and TPR at 5 percent FPR **0.00**. No attacker was involved. Any table
   that reports AUROC without the operating point beside it is already reporting
   a number that does not survive deployment, and that is the single easiest
   criticism to make of the current draft.

4. **Change 1 word in our own published attacker.** `psbd.evasion.evasion_penalty`
   hinges on the clean **mean**, while `psbd.decision.threshold_at_quantile` reads
   a clean **quantile**. Aiming the identical penalty at the clean median instead
   takes TPR at $q = 0.25$ from 0.999 to 0.080 in simulation. Our published
   adaptive attacker is optimising the wrong functional, and fixing it makes it
   strictly stronger at no cost.

## Defences that fall out of this document

Ordered by cost, all defender legal, all implementable against the caches
already on disk.

| defence | cost | what it answers |
|---|---|---|
| Report TPR at 0.01 and 0.05 FPR beside every AUROC | free, CPU, existing caches | A1, A5, and free escape 3 |
| Residualise PSU on $(p_c, \text{entropy}, \|p\|_2^2)$ fitted on clean validation | one least squares fit per probe | A2, A3 |
| Publish the $R^2$ of PSU on $p$ per position as a probe selection criterion | free, CPU | turns A2 into a design rule: never deploy a probe whose $R^2$ exceeds about 0.3 |
| Require probe pool diversity in $R^2$, not just in operator name | free | A3, since 2 probes with high $R^2$ are not independent no matter how different their operators look |
| Calibrate the threshold on a robust statistic (median and MAD) rather than a raw quantile, and flag a bimodal clean validation PSU distribution | free, CPU | A1 |
| Test time input augmentation as a second line | inference only | A4, via the certified radius coupling argument in that file |

## How this document relates to `cross-defence.md`

`docs/attack-design/cross-defence.md` was written against the same defence and
asks a different question: whether 1 objective breaks a whole **family** of
published defences. The 2 documents agree everywhere they overlap, and the 1 place
they look like they disagree is worth stating precisely, because a reviewer will
read both.

`cross-defence.md` refutes "the family reduces to a function of $p$" and shows
that not 1 of STRIP, SCALE-UP, IBD-PSC, TeCo and PSBD is a function of $p(x)$, so
confidence matching defeats only the null model. That is correct and this document
does not contradict it. What this document adds is that **PSBD is not 1 detector,
it is a 1 parameter family indexed by probe position, and the family degenerates
into the null model at 1 end of that parameter.** The joint statement, which is
the one to publish, is:

> Confidence matching defeats the max softmax null exactly, defeats none of
> STRIP, SCALE-UP, IBD-PSC or TeCo at all, and defeats PSBD in proportion to
> $R^2_\ell$, the share of the probe's statistic explained by the softmax
> vector. That share is 0.98 at the last pre-head site, under 0.14 across the
> CIFAR-100 panel of input side sites and under 0.25 at every input side site
> measured, so the choice of probe position **is** the choice of how much
> of the detector an attacker gets for free.

`cross-defence.md` section 6 identifies its own most dangerous composite attack.
It should be read alongside [A1](A1-operating-point-and-threshold.md) and
[A3](A3-shared-factor-multi-probe.md), which attack the threshold and the shared
factor rather than the representation.

## Traceability

`docs/attack-design/measure_confidence_share.py` produced every measured number
in this README. It is CPU only and reads the existing sweep caches under
`results/*/psbd/`. It belongs in `experiments/confidence_share/` if any of this
is adopted, and is kept here for now so that the numbers in this directory can be
reproduced without touching anything outside it.
