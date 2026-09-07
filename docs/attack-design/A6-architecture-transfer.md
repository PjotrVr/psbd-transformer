# A6. Does an attack designed against ViT transfer to a ConvNet

**Rank 6 by damage, rank 1 by value.** This is the file to read if only one gets
read. The answer is not a hedge: **confidence matching is architecture
independent and curvature matching is architecture specific**, and the reason is
a single term in the master equation that a ReLU ConvNet does not have.

## First, dispose of the wrong version of the question

An evasive **checkpoint** does not transfer between architectures, because
weights do not. Neither does a poisoned **dataset** raise the question, because
every attack in this document leaves the trigger and the poison indices alone and
changes only the loss. So the only meaningful question is whether the **objective
transfers**, and it has to be asked separately for each attack family. It has 2
different answers.

## The term that separates them

From [README](README.md), at probe site $\ell$,

$$\phi_\ell(x) \;\approx\;
\underbrace{-\tfrac{1}{2}\,p_c\Big[u^\top \Sigma_z u - \operatorname{tr}(A\,\Sigma_z)\Big]}_{T_1,\ \Sigma_z = J\Sigma_\ell J^\top}
\;\underbrace{-\;\tfrac{1}{2}\,p_c\sum_k (\delta_{ck}-p_k)\operatorname{tr}\!\big(\nabla_h^2 z_k\,\Sigma_\ell\big)}_{T_2}.$$

$T_1$ has the **same functional form at every probe site and in every
architecture**. It is the logit layer closed form with $\Sigma$ replaced by its
pushforward. Only 2 things vary across sites and models: the softmax vector $p$,
and the $C \times C$ matrix $\Sigma_z$.

$T_2$ is the curvature of the sub network between the probe and the logits, and
it is where the architectures separate.

### A ReLU ConvNet has no $T_2$

Take a ConvNet built from convolutions, ReLU, average or max pooling, batch
normalisation in eval mode, and a linear head. Every one of those is **piecewise
linear** in its input, and a composition of piecewise linear maps is piecewise
linear. So $z(h)$ is piecewise linear in $h$ and

$$\nabla_h^2 z_k \;=\; 0 \qquad \text{almost everywhere} \qquad \Longrightarrow \qquad T_2 = 0 .$$

**On a ReLU ConvNet the detection statistic at every probe site is exactly the
logit layer closed form, evaluated against the pushforward covariance.** Nothing
new appears as the probe moves inward. The position axis changes $\Sigma_z$ and
nothing else.

The honest correction at finite perturbation scale. The probe is not
infinitesimal, so $\delta$ crosses ReLU kinks and the "almost everywhere" is doing
real work. Write the smoothed unit: for a pre-activation $a$ and noise
$\delta \sim \mathcal{N}(0,\sigma^2)$,

$$\mathbb{E}\big[\mathrm{relu}(a + \delta)\big] - \mathrm{relu}(a)
= \sigma\,\varphi(a/\sigma) - a\big(1 - \Phi(a/\sigma)\big),$$

and averaging over a population of pre-activations with density $\rho$ near 0
gives a contribution of order $\sigma^2 \rho(0)$. So the ConvNet does have an
effective curvature at finite $\sigma$, but it is **a boundary density statistic**:
it counts how many pre-activations sit within $O(\sigma)$ of their kink. The ViT's
$T_2$ is a smooth second derivative that is nonzero at every activation, from
GELU, from LayerNorm's $x/\|x\|$ normalisation, and from the attention softmax.
These are different functionals of $x$, and a regulariser that flattens one does
not flatten the other. Max pooling adds kinks and behaves like ReLU here. Average
pooling is linear and adds nothing.

### A ViT has $T_2$ everywhere

Between a probe at `before_attention_norm` in block 3 and the logits sit 9 more
blocks, each containing 2 LayerNorms, 1 attention softmax and 1 GELU. Every one
of them has nonzero second derivatives at every input, so $T_2$ is a smooth,
everywhere nonzero, sample dependent quantity that no constraint on $p$ can
reach.

## Consequence 1: confidence matching is architecture independent

The objective in [A2](A2-confidence-distribution-matching.md) reads only
$p = \operatorname{softmax}(f(x))$. It contains no architecture specific symbol,
no position, no operator, and no Jacobian. It therefore:

- **transfers verbatim as source code** between ViT, Swin and any ConvNet,
- **defeats exactly the same list of detectors** on all of them, because that
  list is determined by which detectors are functions of $p$ at the deployed
  input, which is an architecture free property,
- and **has an architecture dependent effectiveness**, in a direction the
  decomposition predicts.

The prediction is sharp. Define the **explained share**

$$R^2_\ell \;=\; 1 - \frac{\operatorname{Var}\big[\phi_\ell - \mathbb{E}[\phi_\ell \mid p]\big]}{\operatorname{Var}[\phi_\ell]},$$

the fraction of the statistic an attacker who equalises $p$ removes. Since a
ConvNet has no $T_2$, its unexplained residual can only come from the per sample
variation of $\Sigma_z$. Add a cheap proxy for that variation and the residual
should close. The across pass standard deviation of $P_c$ is such a proxy: to
first order it is $\sigma\|\nabla g\|$, which is the pushforward energy along the
gradient direction, and it is already in every cache on disk.

Measured on ViT, `vit_cifar100_badnet_a2o_0_01`, clean validation, sigma matched
to 0.6:

| probe | $R^2$ on $p$ | $R^2$ on $p$ plus the pass spread |
|---|---:|---:|
| `after_embedding` + token_mask | 0.083 | 0.163 |
| `before_attention_norm` + token_mask | 0.062 | **0.279** |
| `before_attention_norm` + dropout | 0.089 | 0.268 |
| `before_mlp_residual` + token_mask | 0.078 | 0.211 |
| `final_norm_out` + gain_scale | 0.969 | 0.969 |

Adding the pushforward proxy roughly triples the explained share at input side
probes and still leaves about **0.72 of the ViT statistic unexplained**. On a
ConvNet, where $T_2 = 0$, that residual has nothing to be made of except the
higher moments of $\Sigma_z$ and the kink density term, and it should be
substantially smaller.

> **Prediction A6-1.** At matched relative depth and matched sigma, a ReLU
> ConvNet's $R^2$ on $(p, \text{pass spread})$ exceeds a ViT's by at least 0.20.
> Equivalently: **PSBD is harder to adaptively attack on a transformer than on a
> ConvNet, and the reason is the smooth nonlinearities the transformer has and
> the ConvNet does not.**

If that holds it is the strongest possible answer to "why port PSBD to ViT at
all", and it is an answer nobody has given. The `final_norm_out` row should be
near 1 on **both** architectures, because $T_2 = 0$ there for everyone, and that
row is the control that makes the comparison meaningful.

## Consequence 2: curvature matching is architecture specific

The objective in [A4](A4-curvature-matching.md) contains 3 architecture specific
objects and none of them survives a change of backbone.

1. **The position set $L$.** `psbd.positions.POSITION_REGISTRY` has a per
   architecture table for exactly this reason, and `VIT_POSITIONS` and
   `SWIN_POSITIONS` already differ in which names exist. A ConvNet's sites are
   pre-block and post-block feature maps and have no `before_attention_norm`.
   Even the 2 architectures already in the repo need different tables, so a third
   is not a portability problem, it is the normal case.
2. **The covariance $\Sigma_\ell$.** A ConvNet feature map is strongly spatially
   correlated, so an elementwise $\Sigma$ has a far smaller effective rank than
   the same operator on a ViT token grid, and the natural ConvNet operator is
   `SpatialDropout2d`, which zeroes whole channels. `token_mask` has no ConvNet
   meaning at all: `FORBIDDEN_OPERATOR_POSITIONS` in `psbd/operators.py` already
   refuses token operators on a rank-4 image tensor and states the reason.
3. **The scale.** The sigma matching rule calibrates against clean validation
   shift ratio, so the rate that matches sigma 0.6 has to be recalibrated per
   architecture. That is mechanical, but it means the attacker cannot ship a
   hyperparameter, only a procedure.

> **Prediction A6-2.** An A4 checkpoint trained on ViT, evaluated by the same
> position and operator names remapped to a ConvNet, shows **no evasion**: AUROC
> within 0.05 of the non evasive ConvNet baseline. And a curvature matching
> objective that was tuned on ViT, retrained on a ConvNet with the ViT's
> $\Sigma$ shape, underperforms the same objective with a ConvNet native
> $\Sigma$ by at least 0.15 AUROC of evasion depth.

## Consequence 3, and this one needs to go in the paper as a scope limit

The claim that "structured masking is a dead end" (H22, H26, H35, H18, and
`docs/paper-proposals.md` claim 4) rests on the backdoor direction being **non
axis aligned**, which was measured on ViT: 50 percent of total TAC carried by 291
of 768 dimensions.

**Axis alignment is an architecture dependent property, not a universal one.** A
ConvNet's channels are the units that Fine-Pruning and ABS prune, and the whole
backdoor pruning literature exists because ConvNet backdoors are, at least
partly, channel aligned. The argument in the theory document is conditional and
correctly stated as conditional there ("given non axis aligned backdoors"), but
the claim in `paper-proposals.md` is stated flatly. A reviewer with a ConvNet
pruning background will read claim 4 as contradicting a literature they know.

The fix is 1 sentence and it costs nothing: **state the claim as conditional on
the measured axis alignment of the architecture, and state that the measurement
was made on ViT.** The framework then predicts the ConvNet result rather than
being contradicted by it, which is a stronger position.

## The experiment

One new checkpoint decides both predictions, and it is small.

**Design.** Train a ResNet-18 on CIFAR-100 with `badnet_a2o` at poison rate 0.01
for the standard 15 epochs, at the same seed, using the existing training path
with a third architecture added to `psbd/models.py` and a `RESNET_POSITIONS`
table added to `psbd.positions.POSITION_REGISTRY`. The position table needs 4
entries to be enough: `after_stem`, `after_stage2`, `after_stage3`,
`final_pool_out`, which mirror `after_embedding`, an early block site, a late
block site and `final_norm_out`.

**Sweep** with `gaussian` at all 4 sites, because gaussian is the operator that
means the same thing in both architectures. Do not use `token_mask` or
`channel_mask` for the comparison, because they are not the same intervention on
a feature map as on a token grid, and using them would confound A6-1 with the
axis alignment question.

**Then run** `docs/attack-design/measure_confidence_share.py` on the ResNet
sweep and compare the $R^2$ column against the ViT table above.

| quantity | ViT (measured) | ResNet-18 prediction | refutes A6-1 if |
|---|---:|---:|---|
| $R^2$ on $(p, \text{spread})$ at the early site | 0.163 | **above 0.40** | below 0.25 |
| $R^2$ on $(p, \text{spread})$ at the late site | 0.211 to 0.279 | **above 0.50** | below 0.30 |
| $R^2$ on $p$ at the final pre-head site | 0.969 | **above 0.90** | below 0.80, which would mean the control fails and the measurement is not comparable |
| PSBD AUROC at the late site | 0.92 to 0.96 | no prediction | n/a |

The third row is the control and it must be checked first. If it does not come
out near 1 on the ResNet, something is wrong with the port and the other 2 rows
mean nothing.

**Budget.** 1 ResNet-18 training run on CIFAR-100 is minutes on the cluster's GPU
queue, not hours. 4 sweeps of 4 sites at the standard rate grid. The analysis is
CPU. This is the cheapest experiment in this entire document and it produces the
claim with the longest reach.

**Second experiment, only if the first confirms.** Train the A2 objective on both
architectures and measure the AUROC drop at the matched late site. Predicted:
**the ConvNet loses at least 0.15 more AUROC than the ViT under the identical
objective**, which converts A6-1 from a property of the statistic into a property
of the attack.
