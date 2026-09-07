# A4. Position randomised curvature matching

**Rank 4.** The correct strengthening of the attacker we already published. H25's
attacker moves along the operator axis of a single position. This one moves along
the axis our own theory says dominates, and it does so with a better estimator at
a lower cost.

## What is wrong with the attacker we have

`psbd.evasion.psu_for_batch` plugs 1 named probe, runs $k$ one sided perturbed
passes, and hinges on the difference of 2 group means. Three weaknesses, all of
them ours to report before a reviewer does.

1. **It attacks 1 position.** The paper's own headline is that position carries
   1.43 times the variance of operator. So the attacker was pointed at the weaker
   axis, and H25's transfer result ("evasion is probe specific") was measured
   against an attacker that never tried to be otherwise.
2. **It uses the worse estimator.** The one sided difference
   $g(h) - g(h+\delta)$ retains the first order term $-\nabla g^\top \delta$,
   whose variance $\sigma^2\|\nabla g\|^2$ dominates the second order signal.
   Objection 4 of the theory document says so and proposes antithetic pairing as
   a **defender** improvement. It is equally an attacker improvement, and the
   attacker gets more from it because the attacker pays for gradients through
   every retained pass.
3. **It costs 4x wallclock**, because `passes + 1` full ViT-B/16 activation
   graphs are retained. A threat model in which evasion costs 4x training is a
   much friendlier threat model than the real one.

## Objective

    original form
        per step:
            sample  l ~ Uniform(L),  z ~ Rademacher(shape of h_l)
            phi_hat_l(x) = g(h_l) - 0.5 * [ g(h_l + sigma_l * z) + g(h_l - sigma_l * z) ]
            L = L_CE(f(x), y)
                + lambda_curv * ReLU( median_{x in C} phi_hat_l(x)
                                      - mean_{x in P} phi_hat_l(x) )

    descriptive form
        each step, pick one probe site at random from the whole position registry
        and one sign vector, estimate the prediction shift with an ANTITHETIC
        pair of perturbed passes, and hinge on the clean-median minus
        poisoned-mean gap at that site

| symbol | meaning |
|---|---|
| $L$ | the set of probe sites the attacker covers, drawn from `psbd.positions.POSITION_REGISTRY` |
| $\ell$ | the site sampled this step |
| $z$ | a Rademacher probe vector, the minimum variance Hutchinson probe |
| $\sigma_\ell$ | perturbation scale at site $\ell$, set once by the defender's own sigma matching rule using `psbd.evasion.calibrate_probe_rate` |
| $\hat\phi_\ell$ | the antithetic finite difference estimate of $-\tfrac{1}{2}\operatorname{tr}(H_\ell \Sigma_\ell)$ |
| $\lambda_{\text{curv}}$ | weight on the hinge |

Why this is stronger than H25 along the axis that matters:

- **The expectation over $\ell$ is a constraint on every position at once.** Over
  an epoch the attacker performs stochastic coordinate descent on the defender's
  position axis. It does not need to know which position the defender picked.
- **The expectation over $z$ makes it a constraint on the trace, not on 1
  projection.** $\mathbb{E}_z[z^\top H z] = \operatorname{tr}(H)$ exactly, which
  is what the defender's isotropic operators estimate. So the attacker constrains
  the same functional the defender reads, rather than 1 realisation of it.
- **The antithetic pair cancels the first order term exactly**, so the gradient
  signal is the second order term rather than the gradient noise it was buried
  in. At the same 2 pass budget the attacker gets a far less noisy penalty than
  H25's $k = 3$ one sided estimator.
- **It is cheaper.** 2 retained graphs instead of $k + 1 = 4$.

## Is it stronger or weaker than confidence matching?

**More targeted, more expensive, and it needs a guess the other attack does not.**

| | A2 confidence matching | A4 curvature matching |
|---|---|---|
| what it constrains | the $T_1$ factor, through $p$ | $\operatorname{tr}(H_\ell \Sigma_\ell)$, so $T_1$ and $T_2$ together |
| cost per step | 1 forward pass | 3 forward passes, 2 of them retained |
| what it must guess | nothing | $\Sigma_\ell$: the operator's covariance shape and its scale |
| architecture portability | verbatim | template only, see [A6](A6-architecture-transfer.md) |
| defeats input side probes | no, at most 0.14 of the variance | **yes, this is the point** |
| defeats the confidence null | yes, exactly | only incidentally |

The $\Sigma$ guess is the real constraint and it is narrower than it looks.
$\Sigma$ enters $\operatorname{tr}(H\Sigma)$ linearly, and the defender's
operators are either isotropic (`gaussian`, `rademacher`) or a diagonal built
from the activation's own energy (`dropout`, `token_mask`, `channel_mask`, all
multiplicative, so $\Sigma = \operatorname{diag}(h_i^2 p/(1-p))$). An attacker who
constrains $\operatorname{tr}(H \operatorname{diag}(h^2))$ has therefore
constrained a positive combination that covers the whole multiplicative family
and, since $\operatorname{diag}(h^2)$ is a reweighting of the identity rather
than a rotation of it, approximately covers the isotropic family too. **The only
operators outside that cover are the ones whose $\Sigma$ is rank deficient or
rotated: `head_mask`, `droppath` and `gain_scale`.** `gain_scale` is exactly the
operator H25 measured as the strongest transfer probe at ViT mean 0.955. That is
not a coincidence and the paper should say why.

## Which detectors it defeats and which it does not

| detector | defeated | why |
|---|---|---|
| PSBD at any position in $L$, with any elementwise or multiplicative operator | **yes** | the trace is constrained at that site and $\Sigma$ enters linearly |
| PSBD at a position outside $L$ | partly | curvature is correlated across neighbouring sites in a residual stream, so coverage bleeds |
| PSBD with `gain_scale` or `droppath` | **probably not** | those $\Sigma$ are not in the positive cone the attacker constrained. `gain_scale` is deterministic, so its "shift" is a per sample flip indicator rather than an expectation over noise, and the second order expansion does not describe it at all |
| PSBD with `head_mask` | not relevant | it reads 0.539 on a clean model, so there is nothing to evade |
| multi probe min or median rank | **yes, if $L$ covers the pool** | this is the strongest available attack on H41 and it is the one the paper must run |
| the confidence null | partly | flattening curvature at every site pushes toward saturation, which raises confidence, which the null reads |
| STRIP, SCALE-UP, TeCo, IBD-PSC | **no** | none of them perturbs an activation site. Their $\Sigma$ lives in pixel space or parameter space, which $L$ does not contain |

The last row is the honest limit and it is worth a table row in the paper. The
attack's cover is the activation space position registry. Input space and
parameter space perturbation consistency detectors sit outside it, which is an
argument for keeping STRIP, SCALE-UP and IBD-PSC in the deployed ensemble rather
than only in the comparison table.

## The cost the attacker cannot avoid, and it is a theorem

This is the most valuable paragraph in this document for the defence.

Take the randomized smoothing bridge at face value. For a smoothed classifier at
probe site $\ell$ under noise of scale $\sigma$, Cohen, Rosenfeld and Kolter give
a certified $\ell_2$ radius $R = \sigma\,\Phi^{-1}(\tilde p)$ with
$\tilde p = \mathbb{E}_\delta[g(h + \delta)] = g(h) - \phi(x)$, and the
Neyman-Pearson argument proves it tight. $R$ is strictly decreasing in $\phi$.

So an attacker who succeeds at A4 has, by construction, made the poisoned
population's certified radius at site $\ell$ **equal in distribution to the clean
population's**. There is no way to have one without the other, because they are
the same number under a monotone map. Therefore:

> **A successful curvature matching attacker has capped the trigger's robustness
> at the level of an ordinary clean class decision, in activation space at the
> probed site.**

That is exact, needs no Taylor expansion, and it is a no free lunch statement
about the entire family: evasion of a perturbation consistency detector is not a
free property added on top of a backdoor, it is the removal of the backdoor's
robustness margin.

Two caveats have to travel with it or a reviewer will supply them.

1. **The radius is in activation space at the probe site.** Converting to input
   space needs the local Lipschitz constant of $x \mapsto h_\ell$. An attacker
   could in principle inflate input space robustness while holding activation
   space robustness down, by making the encoder contract along the trigger
   direction. That is a gradient masking shaped configuration, it is hard to
   train, and whether it is reachable is an open question. It is the escape
   hatch, and it should be named rather than hidden.
2. **The cap holds at the probed site only.** With $L$ covering the registry the
   cap holds everywhere, which strengthens the argument for A4 specifically.

## The defence this hands us, and it costs nothing at training time

**Test time input augmentation.** If a successful A4 attacker has capped its own
trigger's robustness, then averaging the prediction over a small set of ordinary
input transformations (a random resized crop, a mild blur, a JPEG round trip)
should cost clean accuracy almost nothing and cost ASR a great deal. It is
inference only, it needs no clean validation data, and it is orthogonal to every
probe in the registry.

This is testable **right now on checkpoints we already have** and it needs no new
training.

## Falsifiable prediction and 2 experiments

### Experiment 1, no new training, decides the theorem

**Prediction.** The 126 `*_evade_l1` checkpoints have already paid part of the
robustness cap at 1 site. So their ASR should be **more fragile under input
corruption** than their non evasive twins, which exist for every one of them.

| quantity | measure on | prediction | refutes if |
|---|---|---|---|
| ASR under JPEG quality 50 | 56 evasive checkpoints with ASR above 0.9, and their 56 twins | evasive ASR drop **at least 2x** the twin's drop | drops are within 20 percent of each other |
| ASR under gaussian blur sigma 1.0 | same | same | same |
| clean accuracy under the same corruptions | same | within 1 point of each other | the evasive models lose more clean accuracy too, which would mean the effect is general fragility and not trigger fragility |

The control in the third row is what makes the result mean something. If both ASR
and clean accuracy degrade, the evasion made a worse model. If only ASR degrades,
the certified radius argument is confirmed on real checkpoints and the paper gets
a theorem with a measurement attached.

```bash
# inference only, no training, roughly 1 GPU hour for all 112 checkpoints
python cli/evaluate.py --checkpoint-folder $(ls checkpoints | grep evade_l1) \
  --corruption jpeg_50 gaussian_blur_1.0 identity
```

`--corruption` does not exist yet. `psbd/detectors/teco.py` already implements 14
corruption types at 5 severities, so the transform is in the repo and only needs
to be reachable from the evaluation path.

### Experiment 2, the attack itself

**Prediction.** Train `vit`, `cifar100`, `badnet_a2o` at 0.01 for 15 epochs with
$L$ = the 6 single position names, $\lambda_{\text{curv}} = 1.0$, antithetic
pairs, Rademacher probes.

| quantity | H25 evasive (measured) | prediction under A4 | refutes if |
|---|---:|---:|---|
| AUROC at the 4 H41 pool probes, mean | 0.927 combined | **below 0.65** | above 0.85, which would mean position randomisation does not generalise across sites |
| AUROC at `gain_scale` on `mlp_norm_out` | 0.955 (strongest transfer) | **above 0.85 still** | falls below 0.6, which would refute the $\Sigma$ cone argument |
| multi probe median rank AUROC | not yet measured | **below 0.70** | above 0.85 |
| clean accuracy tax | 4.8 points | 3 to 6 points | below 2, which would be surprising and worth chasing |
| wallclock relative to plain training | 4x | **2 to 3x** | above 4x |
| ASR under JPEG 50, relative to its own clean-input ASR | measure | **falls at least 0.15** | unchanged, refuting the certified radius coupling |

The pair of rows 2 and 6 is the whole experiment. Row 2 says the attack has a
named blind spot that the theory predicted in advance. Row 6 says the attack
bought its success with trigger fragility, which is what the certificate
requires. Either one holding is a paper result. Both holding is the adaptive
attack section written.
