# A2. Confidence distribution matching

**Rank 2.** The attack a reviewer constructs directly from our own theory
document, objection 3. It is cheaper than the attacker we published, it needs no
knowledge of the probe registry, and it defeats a whole family of baselines at
once. It also, provably, cannot defeat the configuration we recommend, and that
is the most valuable thing in this file.

## The lever

At the logit layer the detection statistic has a closed form that depends on
nothing but the softmax vector,

$$\operatorname{tr}\nabla_z^2 p_c \;=\; 2\,p_c\big(\|p\|_2^2 - p_c\big),$$

so any detector that is ultimately a function of $p$ is defeated by an attacker
who makes the poisoned population's $p$ distribution equal to the clean one. The
question is which detectors are actually functions of $p$, and the master
equation in [README](README.md) answers it: only those whose probe site is past
the last nonlinearity, where the network curvature term $T_2$ vanishes and the
pushforward $\Sigma_z = W\Sigma W^\top$ is built from a single fixed matrix.

## Objective

Matching only the mean of $p_c$ is not enough, because the statistic is quadratic
in $p$ through $\|p\|_2^2$. Match the whole 1 dimensional distribution of each of
3 functionals, using a sliced Wasserstein distance that is nothing more than a
sort and a difference.

    original form
        L = L_CE(f(x), y)
            + lambda_conf * sum_{s in S} W_1( { s(p(x)) : x in P }, { s(p(x)) : x in C } )
        S = { p_c,  -sum_j p_j log p_j,  ||p||_2^2 }

    descriptive form
        loss = cross_entropy
               + match_weight * sum over 3 summary statistics of the
                 1-dimensional Wasserstein-1 distance between the poisoned
                 batch's values and the clean batch's values

    W_1 in 1 dimension, computed by sorting
        W_1(a, b) = mean_i | sort(a)_i - sort(b)_i |      when |a| = |b|
                    otherwise compare at m shared quantile levels

| symbol | meaning |
|---|---|
| $P$, $C$ | poisoned and clean samples in the batch |
| $p = \operatorname{softmax}(f(x))$ | the model's full softmax vector |
| $p_c$ | the predicted class probability |
| $S$ | the 3 summary statistics being matched: confidence, entropy, collision probability |
| $\lambda_{\text{conf}}$ | weight on the matching term |
| $W_1$ | Wasserstein-1 distance between 2 empirical 1 dimensional distributions |
| $m$ | number of shared quantile levels, 32 is ample at batch 128 |

Three properties make this dangerous.

1. **One forward pass.** The term reads the same logits cross entropy already
   needs. H25's attacker retains `passes + 1` full ViT-B/16 activation graphs and
   costs roughly 4x normal training wallclock. This costs approximately 1x.
2. **No probe knowledge.** No position, no operator, no rate, no calibration
   step. The attacker does not need `psbd/positions.py` at all.
3. **Distribution, not mean.** A mean matching penalty leaves the second and
   third moments free, and $\|p\|_2^2$ is a second moment. Matching quantiles
   closes that gap. The published H25 hinge matches 1 mean of 1 statistic.

A stronger variant matches the joint rather than 3 marginals, by matching the
sorted top-$r$ softmax vector for $r = 5$. It costs a sort of a 5 vector and it
closes the remaining freedom in the shape of $p$.

## Which detectors it defeats and which it does not

| detector | defeated | why, in terms of what it computes |
|---|---|---|
| `confidence`, the max softmax null | **yes, exactly** | its statistic is $\max_c p_c$, matched by construction |
| PSBD at `final_norm_out` | **yes, essentially exactly** | $T_2 = 0$ and $\Sigma_z = \sigma^2(x) W W^\top$ with $W$ fixed, so $\phi$ is a function of $(p, \sigma^2)$. Measured explained share 0.956 to 0.992 across 7 checkpoints |
| PSBD at `mlp_norm_out`, `attention_norm_out` | **partly**, roughly 0.19 of the variance | block output norms sit after the block's nonlinearities but before 12 more blocks, so $T_2$ is small but nonzero |
| PSBD at `before_attention_norm`, `before_mlp_residual`, `after_embedding` | **no**, at most 0.04 to 0.14 of the variance | $T_2$ is carried by GELU, LayerNorm and the attention softmax in every block between the probe and the head, and $\Sigma_z = J(x)\Sigma J(x)^\top$ varies per sample |
| IBD-PSC | **no** | Eq. (4) reads $\hat F^{\omega}_i(x)_{y'}$, the probability an **amplified** model assigns to the unamplified prediction. Amplifying $\gamma$ and $\beta$ changes $z$, so matching $p$ at $\omega = 1$ constrains nothing at $\omega = 1.5$ |
| SCALE-UP, both variants | **no** | Eq. (2) is a label consistency count on $n \cdot x$ in pixel space. Nothing about $p(x)$ constrains $p(n x)$ |
| STRIP | **no** | Eq. (2) to (4) read entropy of $p(x + \text{overlay})$, a different input |
| TeCo | **no** | reads the spread of corruption hardness thresholds, again a family of different inputs |
| multi probe min or median rank | **no**, at an input side pool | each probe's residual survives. See [A3](A3-shared-factor-multi-probe.md) for the version of this attack that does threaten the union |

The pattern is clean and it is worth stating as a rule: **confidence matching
defeats exactly the detectors that read $p$ at the deployed input, and none of the
detectors that read the model's response to a modified input.** PSBD sits on the
second side of that line whenever its probe is far enough from the head, which is
what makes the position axis a security property and not only a tuning axis.

## Evidence already in hand, before anyone trains anything

**Our published adaptive attacker does not do this, and it shows.** On the H25
evasive checkpoints the probed operator is at 0.038 and 0.013, and the free max
softmax null still reads **0.687** on `vit_cifar100_badnet_a2o_0_01_evade_l1` and
**0.970** on `vit_cifar100_blend_0_1_evade_l1`. A reviewer will notice that our
strongest attacker is beaten by 1 forward pass and a max. That is the single
weakest sentence available about the current adaptive attack section, and A2 is
the attack that fixes it from the attacker's side.

**A partial natural experiment already exists.** `adaptive_blend` is designed to
flatten latent separation, and it incidentally lands near confidence parity: its
max softmax AUROC is **0.245**, well below chance, and `lc` reads 0.370. If
confidence matching were sufficient to defeat PSBD, those checkpoints would be
undetectable. Measured at `before_attention_norm` with token_mask, sigma matched
to 0.6, they read PSU 0.706 and 0.692, and after residualising on $p$ they read
**0.827 and 0.708**. Confidence parity costs an input side probe under 0.12
AUROC, and residualising gives most of it back.

## Predicted cost

- **Clean accuracy: 0.5 to 2 points.** The term does not fight cross entropy on
  the arg max, only on the shape of $p$. It is a calibration constraint. Compare
  H25's measured 4.8 point mean tax, which came from retaining $k+1$ graphs and
  optimising against a noisy $k = 3$ estimate.
- **ASR: 2 to 8 points on CIFAR-100, less on CIFAR-10.** Matching the clean
  distribution means reproducing its left tail, so a fraction of poisoned samples
  must sit at $p_c$ near the clean 5th percentile. On 100 classes a triggered
  sample at $p_c = 0.3$ is still comfortably the arg max, so most of the
  distribution can be matched with no arg max cost at all. The loss is
  concentrated in the tail. On 10 classes the tail bites harder because the
  runner up is closer.
- **Wallclock: about 1x normal training**, against H25's 4x.

## Why it might fail

1. **$T_2$ is not optional.** The attacker has no handle on it from the output
   distribution. Every block between the probe and the head contributes GELU,
   LayerNorm and attention softmax curvature that a constraint on $p$ does not
   reach.
2. **Matching marginals is not matching the joint.** $\phi$ at a head adjacent
   probe depends on $(p, \sigma^2(x))$, and $\sigma^2(x)$ is the activation energy
   at the probe site, which the attacker is not matching. The stronger variant
   would need to match that too, and doing so requires the probe position, which
   returns the attack to H25's threat model and its cost.
3. **Distribution matching is not conditional matching.** Two populations can
   have identical marginals of $p_c$ while $p_c$ is predictable from the image.
   A defender who conditions on anything else, including a cheap image statistic,
   recovers separation.
4. **The optimiser has an easier move available.** The cheapest way to equalise 2
   distributions is to move the larger population, and the clean population is
   99 percent of the batch, so gradient mass concentrates there. The result would
   be a model whose clean confidence has been dragged toward the poisoned
   distribution, which is a global calibration change rather than a hidden
   backdoor. Both group distributions must be logged every epoch, exactly as
   H25's `evasive_update` logs both group means and for the same reason.

## The cheapest defence that catches it

**Residualise PSU on the softmax vector.** Fit
$\hat\phi(p) = \beta^\top [1, p_c, \mathcal{H}(p), \|p\|_2^2, p_c^2, \mathcal{H}(p)^2, \|p\|_2^4]$
by least squares on the 2000 clean validation samples the defender already holds,
and score $\phi(x) - \hat\phi(p(x))$. Cost: 1 least squares solve of a 7 by 7
system per probe. Defender legal: clean validation only, no poisoned data.

It is the provably correct response. If the attacker has equalised the
distribution of $p$, then $\hat\phi(p)$ carries no discriminative information but
still carries variance, and removing a non discriminative variance component can
only help. Measured on 6 attacks at `before_attention_norm` with token_mask,
residualising costs nothing on average (0.886 against 0.862 plain) and gains
**+0.121** exactly where confidence parity already holds, on `adaptive_blend`.

**Second defence, free: publish $R^2$ per position as a probe selection rule.**
The explained share is measurable with no poisoned data, on clean validation
alone, so it is a defender legal probe design criterion. The rule is: never
deploy a probe whose $R^2$ on $p$ exceeds about 0.3, because that fraction of the
statistic is available to an attacker who never learns which probe you chose.
That criterion is new, it falls straight out of the closed form, and it is the
form in which the position result should be sold to a security venue.

## Falsifiable prediction and the experiment

**Prediction.** Train `vit`, `cifar100`, `badnet_a2o` at poison rate 0.01 for 15
epochs with $\lambda_{\text{conf}} = 1.0$ and $m = 32$ quantile levels. Sweep the
resulting checkpoint at 6 positions.

| probe | measured $R^2$ on $p$ | baseline AUROC | prediction under A2 | refutes if |
|---|---:|---:|---:|---|
| `final_norm_out` + gain_scale | 0.980 | 0.248 | 0.45 to 0.55 | stays below 0.35 or rises above 0.7 |
| `mlp_norm_out` + gain_scale | 0.211 | 0.729 | 0.60 to 0.70 | falls below 0.55 |
| `attention_norm_out` + gain_scale | 0.247 | 0.804 | 0.68 to 0.78 | falls below 0.60 |
| `before_mlp_residual` + token_mask | 0.086 | 0.967 | **above 0.90** | falls below 0.80 |
| `before_attention_norm` + token_mask | 0.072 | 0.960 | **above 0.90** | falls below 0.80 |
| `after_embedding` + token_mask | 0.094 | 0.780 | above 0.72 | falls below 0.60 |
| `confidence` null | n/a | 0.752 | **0.50 plus or minus 0.03** | stays above 0.60, meaning the matching term did not take |
| residualised PSU at `before_attention_norm` | n/a | 0.961 | **above the plain PSU number** | below plain PSU |

The quantitative form of the prediction is the useful one. The theory says the
AUROC loss at each position should be governed by that position's $R^2$, so plot
$\Delta$AUROC against $R^2$ across the 6 positions and fit a line. **A slope
significantly different from 0 confirms the master equation. A flat line refutes
it and would mean the attack works through something the decomposition does not
contain.** That is a much sharper test than any single cell.

**Concrete run.**

```bash
python cli/train_backdoor.py \
  --dataset cifar100 --attack badnet_a2o --poison-rate 0.01 \
  --architecture vit --epochs 15 \
  --match-confidence --match-weight 1.0 --match-levels 32 \
  --output checkpoints/vit_cifar100_badnet_a2o_0_01_confmatch

python cli/sweep.py \
  --checkpoint-folder vit_cifar100_badnet_a2o_0_01_confmatch \
  --position-config after_embedding before_attention_norm before_mlp_residual \
  --perturbation token_mask
python cli/sweep.py \
  --checkpoint-folder vit_cifar100_badnet_a2o_0_01_confmatch \
  --position-config attention_norm_out mlp_norm_out final_norm_out \
  --perturbation gain_scale

python cli/compare_detectors.py --fpr 0.01 0.05
PYTHONPATH=. .venv/bin/python docs/attack-design/measure_confidence_share.py \
  vit_cifar100_badnet_a2o_0_01_confmatch \
  after_embedding_token_mask before_attention_norm_token_mask \
  before_mlp_residual_token_mask attention_norm_out_gain_scale \
  mlp_norm_out_gain_scale final_norm_out_gain_scale
```

Budget: 1 training job at roughly 1x normal wallclock and 2 sweeps. This is the
single highest value GPU experiment in this document, because it turns the
paper's position result from an empirical ranking into a security argument with
a measured coefficient.

**A zero cost preliminary.** Run the last command above on
`vit_cifar100_adaptive_blend_0_01`, which already sits at confidence AUROC 0.245.
It is a partial A2 attacker that nobody had to train, and the numbers are already
on disk.
