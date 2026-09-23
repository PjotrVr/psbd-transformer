# Breaking PSBD with a trigger rather than with a loss term

The rest of `docs/attack-design/` attacks PSBD with an objective the attacker
optimizes against the detector. This file asks the harder question: what does a
backdoor have to be, as data and as a training rule, for prediction shift
detection to fail on it, when the attacker never computes the detector's score,
never sees the probe placement and never adds a term that mentions dropout. The
answer is derived from the estimator, checked against the caches already on disk
and reduced to 1 scalar the attacker can turn.

## Result

The detector reads 2 properties of an input. The first is the margin of the
predicted class and the second is the concentration of the evidence carrying that
margin over the units the defender's operator removes. The fractional statistic
this project headlines divides the first one out by construction, so **the only
factor a non-adaptive attacker can profitably move is the second one**, and the
ways to move it are all the same construction seen from different angles: make
the trigger's effect conditional on the input's own class evidence instead of
sufficient by itself.

| rank | construction | the dial | measured or predicted | both architectures |
|---:|---|---|---|---|
| 1 | Graded source conditioning. The trigger flips only images whose true class lies in a source set $S$, with a cover set carrying the trigger at its own label. | $\lvert S\rvert$ from $K-1$ down to 1 | endpoints measured: AUROC 0.992 at $\lvert S\rvert = K-1$, 0.853 with TPR 0.522 at $\lvert S\rvert = 1$, and 0.470 with TPR 0.136 at the published ConvNet placement | yes, the condition is on labels and inputs rather than on tokens |
| 2 | Grouped target geometry. The trigger sends class $y$ to $(y + 1) \bmod m$. | $m$ from 1 to $K$ | measured staircase, AUROC 0.992 at $m = 1$ falling to 0.574 at $m = 64$ and 0.473 at $m = K$, with the 2-sided rule reading the same number until $m = K$ | yes, unmeasured at the published placement, which is the gap |
| 3 | Probabilistic label coupling. A fraction $r$ of triggered training images keep their true label. | $r$ | predicted weak at the recommended placement, since confidence alone explains 0.020 of the statistic's variance there | yes, and provably weak on both for the same reason |
| 4 | Natural feature backdoor. The trigger is a feature the data already contains, at the amplitude the data already carries. | the feature's rarity | exactly 0.5 for any test-time input statistic, by a distributional argument | yes, and it is placement-free |
| 5 | Target class threshold contamination. Pick the target class whose clean images are the most perturbation-robust and supply extra prototypical images of it. | which class, and how many extras | measured, the target class holds 0.153 of the flagged clean-validation mass against a 0.104 class share on 10-class datasets, worth up to 0.552 of TPR on 1 cell and nothing at all above 43 classes | yes, but only on few-class datasets |
| 6 | Amplitude titration at deployment. Train with the trigger amplitude drawn per sample, then pick the test-time amplitude after the fact. | the test-time amplitude | untested here, and it is the only construction needing no retraining to retune | yes |

Construction 1 is the one to build. It costs the attacker the freedom to trigger
an arbitrary image and costs nothing in clean accuracy, it is a data-only recipe
that never touches the training loop, and its 2 endpoints are already in this
repository on opposite sides of the failure point.

## The estimator, in the paper's notation

PSBD is 3 objects: a per-input statistic, a rate selection rule and a threshold.
Both statistics come from `literature/PSBD/sec/4_method.tex` and are implemented
in `defences/scores.py`.

$$
\begin{aligned}
\phi_{PS}(\mathbf{x}) &= \mathbb{I}\left(\mathcal{Y}(\mathbf{x};\boldsymbol\theta) \neq \mathcal{Y}(\mathbf{x};\boldsymbol\theta')\right) \\
\sigma(\mathcal{D}) &= \frac{1}{k\lvert\mathcal{D}\rvert}\sum_{\mathbf{x}\in\mathcal{D}} \phi_{PS}(\mathbf{x}) \\
\phi_{PSU}(\mathbf{x}) &= P_c(\mathbf{x};\boldsymbol\theta) - \frac{1}{k}\sum_{i=1}^{k} P_c(\mathbf{x};p,\boldsymbol\theta_i'), \qquad c = \arg\max_{c\in\mathcal{C}} P(\mathbf{x};\boldsymbol\theta)
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $\mathbf{x}$ | 1 input |
| $\boldsymbol\theta$ | the unperturbed parameters |
| $\boldsymbol\theta_i'$ | the parameters as seen on perturbed pass $i$, which here means the same weights with a fresh probe mask |
| $p$ | the perturbation rate, called the dropout rate in the source |
| $k$ | the number of perturbed passes, 3 in this project and in the source |
| $c$ | the class the unperturbed model predicts |
| $P_c(\mathbf{x};\boldsymbol\theta)$ | the unperturbed probability of that class |
| $\mathcal{Y}$ | the arg max class |
| $\mathcal{D}$ | any scored population |
| $\sigma(\mathcal{D})$ | the shift ratio of that population |
| $\phi_{PSU}$ | the absolute statistic, PSBD Equation 2 |
| $K$ | the number of classes |

The headline statistic in this project is the fractional form,
`detection_psu_ratio`, written $\phi_\rho$ here and defined at
`defences/scores.py:57`.

$$
\phi_\rho(\mathbf{x}) \;=\; 1 - \frac{1}{k}\sum_{i=1}^{k}\frac{P_c(\mathbf{x};p,\boldsymbol\theta_i')}{P_c(\mathbf{x};\boldsymbol\theta)}
\;=\; \frac{\phi_{PSU}(\mathbf{x})}{P_c(\mathbf{x};\boldsymbol\theta)}
$$

The decision layer sits in `defences/decision.py`. The deployable rate rule is
`select_rate_adaptively` at `ADAPTIVE_SHIFT_TARGET` 0.8
(`defences/decision.py:37` and `defences/decision.py:204`), the comparison rule
is `select_rate_at_matched_shift` at `PLACEMENT_MATCH_TARGET` 0.6
(`defences/decision.py:73` and `defences/decision.py:229`) and the threshold is a
quantile of clean validation score at `defences/decision.py:89`, headline
quantile 0.25 (`defences/decision.py:33`).

```
PSBD as it actually runs, over the 2 stages of cli.sweep and cli.analyze

    input   a suspect model theta, a clean validation set D_val of 2000 images
            (data/splits.py:34), a scored pool D, a rate grid P of 9 rates
            (cli/sweep.py:62), a probe placement (position, operator)
    output  a flag per input of D

    1  for each split in {validation, clean, backdoor}
    2      baseline[split] = unperturbed softmax and its argmax
    3                        (defences/inference.py:92)
    4  for each rate p in P
    5      plug the operator at every named position in every block
    6          (models/positions.py:244)
    7      for each split
    8          per_pass[split, p] = k perturbed passes, tracked-class prob and argmax
    9                               (defences/inference.py:130)
    10     unplug by handle
    11 sigma[p] = shift_ratio(baseline[validation], per_pass[validation, p])
    12 p_star   = min { p in P : sigma[p] >= 0.8 }
    13 phi_val  = phi_rho(validation, p_star)
    14 phi_D    = phi_rho(D, p_star)
    15 T        = quantile(phi_val, 0.25)
    16 flag(x)  = [ phi_D(x) < T ]
```

3 properties of that program matter for everything below. The rate is chosen
from clean validation alone, so the attacker influences it only through the clean
behavior of the model he poisoned. The threshold is a quantile of clean
validation, so the false positive rate is fixed by the quantile and the attacker
who wants TPR to fall does not need the ranking to fail. The scored pool is
compared against a paired clean pool by sample index
(`defences/decision.py:358`), so the attacker cannot gain by changing which
images are eligible.

## The separation condition

Result first. The detector's ranking works when the backdoored population's
statistic is stochastically smaller than the clean population's, and its
operating point works when the clean $q$ quantile lies above a large mass of the
backdoored population. Those are 2 different requirements, the second is much
stronger and this repository's own numbers already show cells where the first
holds and the second does not.

$$
\begin{aligned}
\mathrm{AUROC} &= \Pr\left[\phi_\rho(X_{bd}) < \phi_\rho(X_{cl})\right] + \tfrac{1}{2}\Pr\left[\phi_\rho(X_{bd}) = \phi_\rho(X_{cl})\right] \\
\mathrm{TPR}(q) &= F_{bd}\!\left(F_{val}^{-1}(q)\right), \qquad \mathrm{FPR}(q) = F_{cl}\!\left(F_{val}^{-1}(q)\right)
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $X_{cl}, X_{bd}, X_{val}$ | draws from the clean pool, the triggered pool and the clean validation pool |
| $F_{cl}, F_{bd}, F_{val}$ | the cumulative distribution function of $\phi_\rho$ on each |
| $q$ | the threshold quantile, 0.25 at the headline |

The attacker therefore has 3 targets and only the first is the one the literature
usually reports. He can raise $F_{bd}$ toward $F_{cl}$, which is the ranking
attack. He can lower $F_{val}^{-1}(q)$ without touching $F_{bd}$, which is the
calibration attack and leaves AUROC almost unmoved. He can push $F_{bd}$ past
$F_{cl}$, which inverts the sign and is caught by a 2-sided reading, measured at
mean AUROC 0.865 for the sign router against 0.819 for always reading PSU in
`paper/tables/all_to_all.macros.json`.

Measured, on the 71 cells of `results/coverage/coverage.json` whose `asr_class`
is `clears` and which did not diverge, reading
`placements.<placement>.rates[rate].detection_psu_ratio.q0.25` at the rate each
rule chose, exactly as `cli.compare.detectors_psbd_values` does:

| population | placement and rule | AUROC | TPR at $q$ = 0.25 | cells |
|---|---|---:|---:|---:|
| whole panel | token mask at the attention input, adaptive | 0.927 | 0.854 | 69 |
| whole panel | token mask at the attention input, matched 0.6 | 0.921 | 0.862 | 71 |
| whole panel | dropout after the residual add, adaptive | 0.818 | 0.719 | 71 |
| `badnet_a2o` | token mask, adaptive | 0.992 | 0.998 | 12 |
| `lf` | token mask, adaptive | 0.980 | 0.980 | 12 |
| `blend` | token mask, adaptive | 0.966 | 0.941 | 14 |
| `bpp` | token mask, adaptive | 0.948 | 0.931 | 12 |
| `wanet` | token mask, adaptive | 0.845 | 0.820 | 5 |
| `tact` | token mask, adaptive | 0.853 | **0.522** | 11 |
| `tact` | dropout after the residual add, adaptive | **0.470** | **0.136** | 11 |
| `sig` | token mask, adaptive | 0.610 | 0.342 | 3 |

The TaCT row is the whole argument of this document in 1 line. A published,
data-only, defense-blind attack already inverts the ConvNet placement of the
original paper and already halves the operating point of this project's
placement, while its AUROC reads a healthy 0.853. On `vit_cifar10_tact_0_01` the
recommended placement reads AUROC 0.979 with TPR 0.011.

## The factor decomposition

The statistic is a smoothed confidence divided by an unsmoothed one, so write it
that way before any expansion. Fix the probe site and write $h = h_\ell(\mathbf{x})$
for the activation there and $g$ for the map from that site to the predicted class
probability, so $P_c(\mathbf{x};\boldsymbol\theta) = g(h)$ and the perturbed pass
evaluates $g(h + \boldsymbol\delta)$.

$$
\phi_\rho(\mathbf{x}) \;=\; 1 - \frac{\mathbb{E}_{\boldsymbol\delta}\left[g(h + \boldsymbol\delta)\right]}{g(h)}
$$

| symbol | meaning |
|---|---|
| $h$ | the activation at the probe site |
| $\boldsymbol\delta$ | the operator's perturbation, with covariance $\Sigma_\ell$ |
| $g$ | the map from the probe site to the predicted class probability |
| $\tilde{g} = \mathbb{E}[g(h+\boldsymbol\delta)]$ | the smoothed probability, which is what the cache stores |

2 readings of that expression give the factors, and they disagree about
nothing.

**The exact reading.** Cohen, Rosenfeld and Kolter (ICML 2019) certify a radius
$R = \sigma\left(\Phi^{-1}(\bar{p}_A) - \Phi^{-1}(\bar{p}_B)\right)/2$ for a
classifier smoothed by Gaussian noise of scale $\sigma$, where $\bar{p}_A$ is the
smoothed probability of the top class and $\bar{p}_B$ that of the runner up. Under
the bound $\bar{p}_B \le 1 - \bar{p}_A$ that paper also uses, and substituting
$\bar{p}_A = \tilde{g} = g(h)\left(1 - \phi_\rho\right)$, the certificate
becomes $R \ge \sigma\,\Phi^{-1}\!\left(g(h)(1 - \phi_\rho)\right)$, whose
right side is strictly decreasing in $\phi_\rho$ at fixed $g(h)$. So the statistic is a monotone
reparameterization of a certified robustness radius at the probe site, given the
baseline confidence, and the pair $\left(g(h), R\right)$ is a sufficient summary
of it. This route needs no small-perturbation assumption and it is the one
`docs/theory-perturbation-consistency.md:593` argues is the defensible bridge.

**The expansion reading.** For a mean-preserving operator, and every operator in
`defences/operators.py` is mean-preserving except `GainScale` and `ScaleUp`
(`defences/operators.py:33` for the inverted scaling and
`docs/theory-perturbation-consistency.md:39` for the 2 exceptions),
$\phi_{PSU} \approx -\tfrac{1}{2}\operatorname{tr}\!\left(H_h \Sigma_\ell\right)$,
and the softmax chain rule splits it into a part that factors through the softmax
vector and a part that does not, written out in
`docs/attack-design/README.md:35`. The first part is architecture-free in form.
The second part is the curvature of the subnetwork between the probe and the
logits, it vanishes almost everywhere for a piecewise linear ConvNet and it is
nonzero everywhere for a transformer.

Putting the 2 readings together, the separation condition decomposes into 5
factors. For each one the question is whether an attacker who never sees the
defense can move it.

| factor | where it enters | attacker control | mechanism available to him |
|---|---|---|---|
| margin of the predicted class | $g(h)$, and the numerator through it | **high**, and useless, see below | label smoothing on the poisoned subset, early stopping of the poison loss, probabilistic label coupling, trigger amplitude |
| evidence concentration over the masked units | the smoothed probability $\tilde{g}$ at fixed margin | **high**, and this is the live factor | make the trigger's effect conditional on the input's own class evidence, so the decision keeps depending on the units the operator removes |
| curvature of the subnetwork above the probe | the second term of the expansion | **low**, and it is architecture-specific, so an attack that must break both configurations cannot rest on it | training recipe, sharpness, weight decay |
| variance across the $k$ passes | the Monte Carlo error of $\tilde{g}$ at $k = 3$ | **medium**, and it is 2-sided, so it degrades the estimate rather than biasing it | make the trigger's response to partial removal high variance, which widens both populations |
| calibration of the clean validation quantile | $F_{val}^{-1}(q)$, which is the threshold | **medium** on few-class datasets and **negligible** above about 40 classes | choose the target class, and supply extra prototypical images of it |

## What the fractional form throws away, and what is left

Result first. Under a relative operator, the margin enters the shift indicator
not at all and enters the fractional statistic only until the softmax saturates,
so lowering the margin of a triggered input moves it in the wrong direction and
raising it buys the attacker nothing beyond a probability of about 0.99. The
whole low confidence family of attacks is therefore counterproductive against
this project's canonical statistic, which is the opposite of its effect on every
confidence-reading detector in the comparison table.

Take the linear reading of the decision at the probe site, which is the reading
in which both architectures are the same object. Write the gap between the
predicted class and its runner up as a sum of per-unit contributions, and let the
operator keep each unit independently with probability $1-p$ and divide the
survivors by $1-p$, which is what `_keep_scale` at `defences/operators.py:33`
does for every masking operator.

$$
\begin{aligned}
m(\mathbf{x}) &= \sum_{j=1}^{d} e_j, \qquad e_j = w_j^\top h_j \\
G_S &= \frac{1}{1-p}\sum_{j \in S} e_j, \qquad S \sim \mathrm{Bernoulli}(1-p)^{d} \\
\mathbb{E}\left[G_S\right] &= m, \qquad \operatorname{Var}\left[G_S\right] = \frac{p}{1-p}\sum_{j=1}^{d} e_j^2 \\
n_{\mathrm{eff}}(\mathbf{x}) &= \frac{\left(\sum_j e_j\right)^2}{\sum_j e_j^2} = \frac{m^2}{\sum_j e_j^2}
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $d$ | the number of units the operator can remove, tokens for `token_mask` and channels for `channel_mask` |
| $e_j$ | unit $j$'s signed contribution to the gap between the predicted class and the runner up |
| $m$ | the gap itself, the margin in logit units |
| $S$ | the surviving units on 1 pass |
| $G_S$ | the gap on that pass |
| $n_{\mathrm{eff}}$ | the participation ratio of the evidence, at most $d$, and below 1 whenever the contributions cancel |

The shift probability then follows from a normal approximation to $G_S$, and the
margin cancels exactly.

$$
\Pr\left[\text{shift}\right] \;=\; \Pr\left[G_S < 0\right] \;\approx\; \Phi\!\left(-\frac{m}{\sqrt{\operatorname{Var} G_S}}\right) \;=\; \Phi\!\left(-\sqrt{\frac{n_{\mathrm{eff}}(1-p)}{p}}\right)
$$

Verified numerically over 15 conditions at a margin held fixed at 100, with the
evidence vector's spread varied to sweep $n_{\mathrm{eff}}$ from 0.31 to 55.9 and
$p$ over 0.3, 0.5 and 0.7. The largest absolute disagreement between 200000
Monte Carlo draws and the closed form was 0.0015, and the 2 extreme rows are
$n_{\mathrm{eff}} = 1.28$ at $p = 0.5$ reading 0.1290 empirical against 0.1292
predicted, and $n_{\mathrm{eff}} = 0.31$ at $p = 0.5$ reading 0.2911 against
0.2897. 2 inputs with identical margin 100 and $n_{\mathrm{eff}}$ of 1.25 and
2000 read shift probabilities of 0.132 and 0.000.

3 consequences follow, and the third is the one the attacker needs.

The shift ratio, which is what the adaptive rate rule reads, is a pure function
of the clean population's evidence concentration and the rate. It carries no
margin information at all. So the defender's rate selection is set by how
distributed the clean evidence is, and the attacker cannot move the chosen rate
by changing confidences.

The fractional statistic is nearly margin-free in the saturated regime and
strongly margin-increasing below it. Computed at $n_{\mathrm{eff}} = 1$ and
$p = 0.5$ with a logistic read, $\phi_\rho$ runs 0.0000, 0.0008, 0.0105, 0.0463,
0.1199, 0.1548, 0.1679, 0.1639 and 0.1595 at margins 0.05, 0.2, 0.5, 1, 2, 3, 5,
8 and 20. It is monotone up to a very flat interior maximum at margin about 5,
which is a predicted-class probability of 0.993. **A low confidence backdoor
therefore lowers $\phi_\rho$, and low $\phi_\rho$ is the flagged side.** The
2-zeros picture in [A5](A5-low-confidence-backdoor.md), and the audit finding A21
quoted at the top of that file, is a statement about an absolute perturbation of
the logits, and no operator in this project applies 1 of those, a point
`docs/theory-perturbation-consistency.md:497` makes for a different reason.

Over a grid of 6 margins from 1 to 12 crossed with 6 values of
$n_{\mathrm{eff}}$ from 0.5 to 100, the share of the statistic's variance carried
by the margin axis is 0.099 for the absolute form and 0.077 for the fractional
form, against 0.793 and 0.840 for the concentration axis. The fractional form is
the more concentration-pure of the 2, which is an independent argument for the
choice `defences/scores.py:57` already defends on other grounds.

## The same 2 factors, measured on the caches

The linear reading names 2 observables. The first is the baseline confidence
$P_c$, which the cache stores directly. The second is the spread of the tracked
probability across the $k$ passes, which under the reading above is a monotone
proxy for $1/\sqrt{n_{\mathrm{eff}}}$ and needs no gradient. Fitting
$\phi_\rho$ on a quadratic in the pair, with the fit taken on the 2000 clean
validation images only so the measurement stays defender-legal, gives the share
of the statistic those 2 factors carry and the AUROC of what is left.

| attack | cells | AUROC $\phi_\rho$ | AUROC pass spread | AUROC $P_c$ | AUROC of the residual | $R^2$ on $P_c$ | $R^2$ on $P_c$ and spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| `badnet_a2o` | 12 | 0.992 | 0.758 | 0.742 | 0.956 | 0.012 | 0.448 |
| `blend` | 14 | 0.966 | 0.533 | 0.790 | 0.897 | 0.017 | 0.481 |
| `bpp` | 12 | 0.948 | 0.470 | 0.636 | 0.893 | 0.022 | 0.477 |
| `lf` | 12 | 0.980 | 0.344 | 0.642 | 0.971 | 0.014 | 0.474 |
| `wanet` | 5 | 0.845 | 0.720 | 0.579 | 0.752 | 0.030 | 0.417 |
| `tact` | 11 | 0.853 | 0.899 | 0.844 | **0.299** | 0.020 | 0.470 |
| `sig` | 3 | 0.611 | 0.694 | 0.428 | **0.482** | 0.066 | 0.365 |
| all | 69 | 0.927 | 0.607 | 0.707 | 0.796 | **0.020** | **0.462** |

Read at the recommended placement and the adaptive rate, with the pass spread
scored in the direction that treats high spread as poisoned and the residual
scored in the same direction as $\phi_\rho$. 4 things in that table matter.

Confidence alone explains 0.020 of the statistic's variance, so any attack whose
whole content is a confidence distribution match is attacking 2 percent of the
detector. This is the same conclusion `docs/attack-design/README.md:103` reaches
with a richer confidence basis and it is worth having twice, because it is what
rules construction 3 out of the top of the ranking.

Adding the pass spread takes the explained share to 0.462, so the 2 factors the
linear reading names carry about half of the statistic and the other half is the
curvature term the reading drops. An attacker matching both moments removes 0.131
of AUROC on average, from 0.927 to 0.796, which is real and not enough on its
own.

On the 2 attacks that are already hard, the residual collapses. TaCT's residual
reads 0.299 and SIG's 0.482, against 0.89 to 0.97 for the 4 easy attacks. **The
attacks PSBD struggles with are exactly the attacks whose entire signal lives in
the 2 factors the reading names.** That is the strongest available evidence that
the reading identifies the right axis, and it is also the instruction: an attack
that moves those 2 moments into the clean range is an attack that puts the whole
cell into the hard regime.

The pass spread is 2-sided across attacks, reading 0.899 on TaCT and 0.344 on LF.
A conditional trigger makes the triggered prediction noisier than a clean one
under partial removal and an unconditional trigger makes it quieter, which is the
same crossing the ranking below is built on.

## What a ResNet and a ViT share

The project's account of the ViT backdoor is a direction in the residual stream,
routed by attention from the trigger's tokens to the class token
(`docs/hypothesis/H49-backdoor-is-routed-not-computed.md`). A ResNet has no
tokens and no attention, so an attack aimed at that account cannot be the answer
here. Naming what survives the change of backbone is the real constraint in this
problem and the derivation above answers it in 1 line.

Both architectures compute a softmax over logits that are an affine read of some
activation, both admit a probe site whose units the operator removes, and for
both the statistic is a function of the margin and of how the predicted class's
evidence is distributed over those units. The margin is divided out. What remains
is $n_{\mathrm{eff}}$, and $n_{\mathrm{eff}}$ is defined without reference to
attention, to tokens or to convolution: it needs only a set of removable units
and a signed contribution per unit. **The shared object is the concentration of
class evidence over removable units, and the shared attack is any construction
that makes the trigger's evidence as cancellation-prone as ordinary class
evidence.**

The 2 architectures differ in how much of the statistic that shared object
carries. A piecewise linear ConvNet has no curvature term almost everywhere, so
the 2 factors are the whole statistic and matching them is sufficient. A
transformer has a smooth curvature term at every activation, so matching them
leaves a residual, measured above at 0.796 AUROC on the panel and 0.299 on TaCT.
The honest prediction is therefore asymmetric: **any construction in this
document should bite harder on the ResNet configuration than on the ViT one**,
and the TaCT row of the panel table, 0.470 at the published placement against
0.853 at the recommended one, is the measured instance of exactly that
asymmetry.

## Construction 1, graded source conditioning

**The construction.** Fix a patch or blend trigger $t$ and a target class $y_t$.
Choose a source set $S$ of size $s$ from the non-target classes. Build the
poisoned training set as 2 disjoint pieces. The attack piece takes a fraction of
the training images whose label lies in $S$, stamps $t$ and relabels them $y_t$.
The cover piece takes a fraction of the training images whose label lies outside
$S \cup \{y_t\}$, stamps the same $t$ and leaves their labels alone. The dial is
$s$, and the cover ratio is held at the repository's existing default of a cover
count equal to the poison count at 1 percent poisoning
(`attacks/tact.py` `TactConfig.cover_rate`). Nothing else changes. No loss term,
no knowledge of the probe, no access to the trained model.

**The mechanism.** With $s = K-1$ and no cover the learned rule is an indicator
of the trigger, its evidence is a single coherent direction added to whatever the
image contributes, and the sum has almost no cancellation, so
$n_{\mathrm{eff}}$ is large, the shift probability is near 0 and $\phi_\rho$ is
near 0. With $s = 1$ and cover the learned rule is a conjunction of the trigger
and the source class, the decision keeps depending on the image's own evidence,
the evidence vector inherits the cancellation structure of a clean decision and
$n_{\mathrm{eff}}$ falls toward and past the clean value, so $\phi_\rho$ rises
into and then through the clean distribution. Somewhere between those 2 settings
the backdoored population's $\phi_\rho$ crosses the clean one, and at the
crossing neither the 1-sided rule nor a 2-sided reading has a signal to read.

The overshoot at $s = 1$ is measured at 2 of the 3 readings and not at the third,
which sets the size of the remaining job. The published ConvNet placement reads
AUROC 0.470 on the 11 TaCT cells, past the crossing. The absolute statistic at the
recommended placement reads 0.604 on those cells and 0.439 on the 6 primary
dataset cells, also past it. The fractional statistic at the recommended
placement is the reading that survives, at 0.853 and 0.797, with TPR 0.522 and
0.605. So at $s = 1$ the attack has already crossed the original paper's
configuration and has not yet crossed this project's, and the dial exists to
close that gap rather than to open it.

**Both architectures.** The condition is stated over labels and source classes,
so the recipe is identical bytes for a ResNet, a ViT and a Swin. The mechanism
runs through $n_{\mathrm{eff}}$, which both architectures have. The measured
endpoint is the strongest evidence available that it transfers: at $s = 1$ the
published ConvNet placement reads 0.470 AUROC with TPR 0.136 over 11 cells, which
is already past the crossing on the ConvNet-style placement while the ViT
placement is still at 0.853.

**The cost.** Clean accuracy is untouched, since cover images keep their labels
and the attack images are a small fraction. The attack surface shrinks: the
attacker can only trigger images whose true class lies in $S$, which is $s/K$ of
the input space. Attack success on those images is unaffected at the measured
endpoint, where the 11 TaCT cells all clear the 0.85 bar. The real cost is the
loss of universality, and it is the cost every source-specific attack in the
literature pays.

**What catches it instead.** A conjunctive rule leaves the true class alive as
the runner up, so the triggered softmax is higher entropy than a clean confident
prediction. That is exactly the statistic that catches the other conditioning
family: on the 12 all-to-all cells of `paper/tables/all_to_all.macros.json` the
negative entropy reads 0.761 while PSU reads 0.411. So the counter is a class
entropy or margin reading fused with PSU, and the attacker then has to hold the
entropy low while holding $n_{\mathrm{eff}}$ at the clean value. Under the linear
reading those 2 are jointly satisfiable, since a large net margin built from
larger opposing contributions has both, but the non-adaptive attacker has no
direct handle on the pair and can only titrate toward it.

## Construction 2, grouped target geometry

**The construction.** Use `all_to_m`, which sends class $y$ to
$(y + 1) \bmod m$ (`attacks/poisoning.py:100`). The dial is $m$, from 1, which is
all-to-one, to $K$, which is all-to-all. This is the label-space version of
construction 1: at $m > 1$ the target depends on the source class, so the rule is
conditional and the decision keeps reading the image.

**Measured, and this is the closest thing to a solved case in the document.** All
ViT cells whose folder carries no evasion, SAM, seed, smoke or augmentation tag,
at the recommended placement and the adaptive rate:

| $m$ | cells | AUROC | TPR at $q$ = 0.25 | 2-sided AUROC | mean ASR |
|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 0.992 | 0.998 | 0.992 | 1.000 |
| 2 | 6 | 0.899 | 0.898 | 0.899 | 0.898 |
| 4 | 6 | 0.814 | 0.770 | 0.814 | 0.852 |
| 8 | 6 | 0.742 | 0.692 | 0.742 | 0.828 |
| 16 | 5 | 0.664 | 0.458 | 0.664 | 0.792 |
| 32 | 5 | 0.653 | 0.515 | 0.653 | 0.784 |
| 64 | 4 | 0.574 | 0.393 | 0.582 | 0.726 |
| 128 | 2 | 0.587 | 0.406 | 0.587 | 0.703 |
| $K$ | 12 | 0.473 | 0.236 | 0.603 | 0.727 |

The staircase falls in AUROC from 0.992 to 0.473 and in TPR from 0.998 to 0.236,
with a single step against the trend at $m = 128$ where only 2 cells exist, the
ASR cost of the whole traverse is about 0.27 under the grouped ASR definition, and
the 2-sided rule recovers nothing at all until the dial reaches $K$. The band from $m = 16$ to $m = 128$ is the crossing region: AUROC 0.57 to
0.66 with a 2-sided reading that reads the same number, TPR 0.39 to 0.52 and ASR
0.70 to 0.79.

**The gap that makes this a pre-registration rather than a result.** Not 1 of the
`badnet_a2m*` cells was ever swept at the published ConvNet placement, so the
staircase's effect on the original paper's configuration is unmeasured. The
ResNet-18 arm exists (`models/positions.py:180` and
`models/backbones.py:70`) and has never been pointed at this question.

## Construction 3, probabilistic label coupling

**The construction.** Poison as usual but leave a fraction $r$ of the triggered
training images with their original labels, drawn at random. The trigger then
implies the target with probability $1-r$ rather than 1, the Bayes-optimal output
on a triggered input is a mixture and the target probability the model can
support is bounded near $1-r$.

**Why it is ranked third rather than first.** Attack success is an arg max
question, so with the residual mass spread over $K-1$ classes the arg max
survives up to $r$ near $(K-1)/K$ and the ASR cost is small. The problem is that
the construction moves the margin and nothing else, and the margin is the factor
the fractional statistic divides out. The measured share of the statistic
explained by confidence at the recommended placement is 0.020. Worse, the sign is
against the attacker: the table above shows $\phi_\rho$ falling monotonically as
the margin falls below a predicted-class probability of about 0.99, so a
successfully de-confidenced backdoor moves further onto the flagged side.

**Where it does work.** At a head-adjacent placement the curvature term vanishes
and the statistic is a function of the softmax vector alone, measured at
$R^2 = 0.984$ for `final_norm_out` with `gain_scale` in
`docs/attack-design/README.md:110`. An attacker who guesses that the defender
probes near the head owns that placement completely. He does not know, and the
recommended placement is not it.

**What catches it instead.** Every confidence-reading detector in the comparison
table reads it directly, in the opposite direction from the one the attacker
wants, and the max softmax null already reads 0.707 on this panel.

## Construction 4, the natural feature backdoor

**The construction.** Choose no artificial trigger. Find a feature combination
the data already contains, rare enough that mislabeling every training image that
carries it costs little clean accuracy, and relabel those images to the target.
At test time the attacker triggers by supplying an image that naturally carries
the feature.

**The mechanism, and it is a proof rather than a prediction.** The triggered test
inputs are drawn from the same distribution as the clean inputs, since they are
clean inputs. Any statistic computed from the model and the input alone therefore
has the same distribution on both populations conditional on the feature, and the
only separation available is whatever the feature itself induces. PSBD is such a
statistic. So AUROC is 0.5 up to the feature's own effect on $n_{\mathrm{eff}}$,
for every placement and every operator at once, with no titration needed.

**The cost, and it is the reason this is not rank 1.** The attacker cannot
trigger an arbitrary image, the attack rate is the feature's natural frequency in
the test distribution, and the clean accuracy loss falls entirely on the images
that carry the feature, which is visible to anyone who evaluates per class.
Building a panel of these is also expensive, since the feature has to be found
per dataset rather than stamped.

**What catches it instead.** Nothing at test time and per input. A training-set
defense that looks for label noise, or a per-class clean accuracy audit, catches
it immediately. This construction is the boundary of what a test-time input
detector can do, and stating that boundary is worth more to the paper than
running it.

## Construction 5, target class threshold contamination

**The construction.** The threshold is a quantile of clean validation score, and
clean validation contains images of the target class. The paper's own mechanism
says clean images under perturbation drift toward the target class, so a clean
validation image already predicted as the target has nowhere to drift and its
$\phi_\rho$ is systematically low. The attacker chooses the target class whose
clean images are the most perturbation-robust and supplies extra prototypical
images of it, which is legal data poisoning at no label cost, and the left tail
of the clean validation distribution deepens, the quantile sinks and TPR falls
while AUROC barely moves.

**Measured, with no attacker involved at all.** For each of the 69 scored cells,
the per-class mean of $\phi_\rho$ on the clean validation split, the share of the
below-threshold clean validation mass the target class contributes and the
threshold and TPR recomputed with the target class's images removed from
validation:

| classes | cells | target is the lowest-scoring class | target share of flagged mass | target share of validation | mean threshold change | mean TPR change | largest TPR change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 22 | 5 | 0.153 | 0.104 | +0.019 | +0.030 | **+0.552** |
| 43 | 15 | 3 | 0.016 | 0.009 | +0.001 | -0.001 | +0.001 |
| 100 | 15 | 3 | 0.019 | 0.009 | +0.000 | +0.001 | +0.011 |
| 200 | 17 | 6 | 0.016 | 0.006 | +0.000 | +0.000 | +0.000 |

On `vit_eurosat_blend_0_1` removing the target class from the validation set
moves the threshold from 0.767 to 0.966 and TPR from 0.448 to 1.000. On
`vit_svhn_sig_0_05_tl1` it moves TPR from 0.674 to 0.923. The target class is the
single lowest-scoring class of all of them in 17 of 69 cells.

**Why it ranks fifth despite a measured half point of TPR.** The effect is
carried entirely by the class count. At 43 classes and above the mean TPR change
is at most 0.001, and this project's primary datasets are CIFAR-100 and Tiny. An
attacker who needs the attack to work on the primary panel gets nothing here.

**What catches it instead, and why the obvious answer does not work.** The
obvious answer is a robust threshold: read the quantile after removing the
validation images whose predicted class is the majority perturbation attractor.
That looked like the cheapest defense improvement in this document and it is
refuted. It was checked on 2026-09-23 over 706 cells and it does nothing.

Removing the attractor does raise TPR, from 0.448 to 1.000 on
`vit_eurosat_blend_0_1` and by 0.013 on average. It also raises the realized
false-positive rate, from 0.156 to 0.226 on that cell and from 0.246 to 0.270 on
average against a nominal budget of 0.25, so the mean distance from the nominal
budget grows from 0.021 to 0.032. The removal is loosening the threshold, not
sharpening the statistic.

The control settles it. Comparing the attractor-dropped threshold against a plain
quantile threshold chosen to realize the SAME clean false-positive rate, the
advantage of dropping is -0.0004 on average and 0.0000 at the median. It wins on
4 of 706 cells and loses on 10. Every apparent gain was the FPR it spent.

The underlying observation is still real and it explains something else. The
images that land on the modal attractor sit at low fractional PSU, so they drag
the validation quantile down and the threshold comes out stricter than the
defender asked for. That is the mechanism behind the left tail of realized
false-positive rates recorded as Q27 in `docs/open-questions.md`, where the
realized rate spans 0.009 to 0.333 against a 0.25 budget. Worth fixing as a
calibration bug rather than sold as a detection gain.

## Construction 6, amplitude titration at deployment

**The construction.** Train with the trigger's amplitude drawn per poisoned
sample from a wide distribution rather than fixed. The learned rule is then a
graded function of amplitude rather than an indicator, and at test time the
attacker picks the amplitude. He can therefore retune the backdoor's position in
the $\left(\text{margin}, n_{\mathrm{eff}}\right)$ plane after the model is
deployed, with no retraining and no access to anything but his own copy of the
trigger.

**Why it is interesting and why it ranks last.** It is the only construction with
a post hoc dial, which matters because every other construction has to guess the
right dial setting before training. It is ranked last because the amplitude
mostly moves the margin, and the margin is the factor the statistic divides out,
so its predicted effect is construction 3's effect with a nicer interface. The
part worth testing is whether a mid-range amplitude also lowers
$n_{\mathrm{eff}}$, which it plausibly does, since a weak trigger has to be
combined with image evidence to cross the boundary.

## Ranking by the product of plausibility and cost to test

1. **Graded source conditioning.** Both endpoints measured and on opposite sides
   of the crossing, 1 integer dial, no loss term, transfers to a ResNet by
   construction. The only construction where a failure would be surprising.
2. **Grouped target geometry.** The staircase is already measured at the
   recommended placement, so what remains is a ResNet arm and a published
   placement column. Cheapest of the 3 top entries because 34 of the cells exist
   and the main arm needs no training.
3. **Target class threshold contamination.** Half of it is already measured from
   the caches with no GPU at all, and the remaining half is 6 training runs.
   Ranked third despite the small mean effect, because a single measured cell
   loses 0.552 of TPR and because the defense it implies is free.
4. **Natural feature backdoor.** Highest certainty of working and highest cost to
   build, and its value is a scope limit on the whole detector family rather than
   a number.
5. **Probabilistic label coupling.** Cheap and sharply predicted to fail at the
   recommended placement, which makes it a useful control rather than an attack.
6. **Amplitude titration.** The most attacker-friendly interface and the least
   derivational support.

## H50, graded source conditioning crosses the detector

**Status: PRE-REGISTERED.** Nothing run.

**Claim.** There exists a source set size $s$, strictly between 1 and $K-1$, at
which the recommended placement reads AUROC in $[0.45, 0.60]$ **and** 2-sided
AUROC below 0.65 **and** TPR at $q = 0.25$ below 0.30, while ASR stays above
0.85 on the source classes and clean accuracy stays within 2 points of the benign
reference. The same $s$, or a smaller one, does it on the published ConvNet
placement and on a ResNet-18.

**Mechanism.** Conditioning the trigger's effect on the source class makes the
decision on a triggered input keep depending on the image's own evidence, which
lowers the evidence participation ratio $n_{\mathrm{eff}}$ toward the clean value
and raises $\phi_\rho$ toward the clean distribution. At $s = K-1$ the ratio is
far above clean and at $s = 1$ it is below, so a crossing exists by continuity if
the dial is monotone.

**Why it should work.** The 2 endpoints are measured in this repository. At
$s = K-1$ the recommended placement reads 0.992 AUROC with TPR 0.998 over 12
`badnet_a2o` cells. At $s = 1$ with cover it reads 0.853 with TPR 0.522 over 11
`tact` cells, and the published placement reads 0.470 with TPR 0.136. The
residual after removing the confidence and pass-spread factors is 0.299 on those
same TaCT cells, so at the $s = 1$ endpoint the detector has nothing left beyond
the 2 factors the dial moves.

**Why it might fail.** The dial may not be monotone in $s$, because a source set
of intermediate size may be learned as a union of $s$ separate conjunctions
rather than as a graded rule, and a union of conjunctions can be as robust as an
indicator. The crossing may also be narrower than the grid, in which case the
result is a bound on the dial's resolution rather than an evasion. The 2-sided
router may catch the whole traverse, which would be the most valuable negative
outcome because it would make sign routing a general answer to the conditioning
family.

**The matrix.** 1 attack (`tact` with `source_classes` overridden), 5 values of
$s$ per dataset, 2 primary datasets (CIFAR-100 with $s$ in 1, 4, 16, 50, 99 and
Tiny with $s$ in 1, 8, 32, 100, 199), 3 architectures (ViT-B/16, Swin-S,
ResNet-18), poison rate 0.05, cover count held equal to the poison count, 15
epochs, seed 0. That is 30 training jobs. Each trained model is swept at 2
placements, `before_attention_norm` with `token_mask` and `post_residual` with
`dropout`, over the 9-rate grid at $k = 3$. The ResNet arm resolves
`post_residual` through `RESNET_POSITIONS` and is swept with `gaussian` as well,
since `token_mask` is refused on a feature map by
`FORBIDDEN_OPERATOR_POSITIONS` (`defences/operators.py:612`).

**The decision rule.** The claim is confirmed when at least 1 value of $s$ meets
all 5 thresholds above on at least 1 primary dataset and at least 2 of the 3
architectures. It is refuted when no value of $s$ takes the recommended
placement below 0.70 AUROC at ASR above 0.85, or when every value that takes it
below 0.60 is caught by the 2-sided reading above 0.65.

**The baseline to read against.** The ResNet arm's 2 existing non-evasive cells,
`resnet18_gtsrb_badnet_a2o_0_1` at AUROC 1.000 with TPR 1.000 and
`resnet18_gtsrb_blend_0_1` at 0.968 with 0.973, both at `post_residual`. A
ResNet-18 rung of this experiment that reads below 0.60 is being compared against
a measured 1.000 rather than against an assumption.

**Cost.** 30 training jobs at 6 hours of walltime for CIFAR-100 and 12 for Tiny,
matching `pbs/generate_evade_jobs.sh`, plus 60 sweeps and a CPU analysis pass. No
new code beyond an `--attack-config-override` for `source_classes`, which
`attacks/tact.py` already carries on its config.

## H51, the target count staircase transfers to the ConvNet placement

**Status: PRE-REGISTERED.** Half measured at the recommended placement, nothing
at the published placement, nothing on a ResNet.

**Claim.** The `all_to_m` staircase falls in AUROC at the published ConvNet
placement as it does at the recommended one, it reaches AUROC below 0.60 with TPR
below 0.40 at some $m$ on both placements and on a ResNet-18, and the 2-sided
reading recovers less than 0.05 of AUROC anywhere short of $m = K$.

**Mechanism.** A grouped target makes the trigger's target depend on the source
class, so the rule is conditional for the same reason H50's is, and the dial
moves the same quantity.

**Why it should work.** The recommended placement's staircase is already
measured, falling from 0.992 to 0.473 with TPR from 0.998 to 0.236, and the
2-sided column is identical to the 1-sided column at every $m$ below $K$. The
factor decomposition says the dial acts through $n_{\mathrm{eff}}$, which a
ResNet has, and the curvature term the ResNet lacks is the part that is left
after the dial has acted.

**Why it might fail.** The ASR of an `all_to_m` cell is exact, since
`attack_success_label` compares against the grouped target of each source class
(`attacks/poisoning.py:210`), and the attacker's capability still falls with $m$,
because he no longer chooses where a given image lands. The eval set also changes
with $m$: `is_eval_poisonable` drops every image whose grouped target is its own
class (`attacks/poisoning.py:182`), so a fraction $1/m$ of the pool leaves as $m$
grows and the populations being compared are not the same across rungs. Either
effect could make the staircase a measurement of a weakening attack rather than of
a changing geometry, and the control for both is in the matrix.

**The matrix.** Existing cells cover ViT at $m$ in 2, 4, 8, 16, 32, 64 and 128 on
CIFAR-100, Tiny, CIFAR-10 and GTSRB at rates 0.05 and 0.1, which is 34 cells,
with the $m = 1$ rung swept at both placements and 6 of the 12 $m = K$ cells
swept at the published 1. The new work
is 1 sweep per existing cell at `post_residual` with `dropout`, 34 sweeps and no
training at all, plus a ResNet-18 arm of 5 values of $m$ on CIFAR-100 at rate
0.05, which is 5 training jobs. The artifact control is 2 columns
beside every cell's ASR, the size of the eval pool and the mean number of
triggered images per grouped target, so a rung whose attack merely weakened is
distinguishable from a rung whose geometry changed.

**The decision rule.** Confirmed when the published placement's AUROC falls in
$m$ with a Spearman correlation below -0.8 over the 9 rungs and reaches below
0.60 at some $m$ with ASR above 0.50. Refuted when the published placement is
flat in $m$, which would mean the dial acts on something the transformer has and
the ConvNet does not, or when ASR falls below 0.20 everywhere the AUROC is low.

**Cost.** 34 sweeps plus 5 training jobs and their 10 sweeps. The cheapest of the
3, and the only 1 whose main arm needs no GPU training at all.

## H52, the target class contaminates the threshold

**Status: PRE-REGISTERED, and the observational half is already measured.**

**Claim.** The clean validation images whose predicted class is the attack's
target class carry systematically lower $\phi_\rho$ than the rest, the effect
scales as the inverse of the class count, and an attacker who selects the target
class by clean robustness and inflates its representation in the poisoned
training set deepens it enough to cost more than 0.20 of TPR at $q = 0.25$ on a
10-class dataset while moving AUROC by less than 0.05.

**Mechanism.** The threshold is a quantile of a population that contains images
of the target class. Those images are already predicted as the target, so the
perturbation has nothing to shift them toward, their score sits low and they fill
part of the false positive budget that the quantile rule intends for genuine
clean losses. The share they fill is bounded by their share of the validation
set, which is about $1/K$.

**Why it should work.** Measured, with no attacker: on 10-class cells the target
class holds 0.153 of the flagged clean validation mass against a 0.104 class
share, and excluding it moves TPR by +0.030 on average with a single cell at
+0.552. The target class is the lowest-scoring class of any in 17 of 69 cells.

**Why it might fail.** The attacker does not control the defender's validation
set, so he can only change how low the target class sits rather than how many of
its images are present. The 2 candidate mechanisms for that, choosing a
distinctive class and supplying extra prototypical images of it, may both be
absorbed by the model's own calibration and leave the quantile where it was. The
effect may also be non-monotone in the number of extra images, since a class with
more training images is also a class the model is less confident about per image.

**The matrix.** 4 training jobs on CIFAR-10 with ViT-B/16, `badnet_a2o` at rate
0.05, target class chosen as the class with the highest mean clean $\phi_\rho$
under a benign reference model, crossed with 0 and 2000 extra prototypical target
class images, plus the same 2 cells on a ResNet-18. The benign reference needed
to pick the class already exists. Each cell is swept at the recommended and the
published placement.

**The decision rule.** Confirmed when the extra-images arm reads TPR at least
0.20 below the no-extras arm at the same quantile with AUROC within 0.05, on at
least 2 of the 4 cells. Refuted when the threshold moves by less than 0.01 in
$\phi_\rho$ units, or when the robust threshold described below recovers the
whole TPR loss, which would make the finding a defense note rather than an
attack.

**The defense it implies, and it should be adopted regardless.** Read the
quantile after dropping the validation images whose predicted class is the modal
perturbation target, which `shift_target_histogram`
(`defences/scores.py:129`) already identifies from the cached argmax tensors at
no cost. This needs no labels, no extra passes and no knowledge of the attack.

**Cost.** 6 training jobs, 12 sweeps and a CPU pass. The observational half is
already done and needs only to be moved into `experiments/` if any of this is
adopted.

## What is not known

The linear reading is a reading and not a theorem. It replaces the map from the
probe site to the logits by an affine one, and it replaces the distribution of
the masked gap by a normal. Both are exactly the approximations that the
curvature term of the full expansion exists to correct, and the measurement above
quantifies the damage: the 2 factors the reading names carry 0.462 of the
statistic's variance on clean validation and matching both costs the detector
0.131 of AUROC. **Everything in this document that speaks about
$n_{\mathrm{eff}}$ is therefore a statement about roughly half of the statistic.**

The participation ratio itself was never measured. The pass spread is used as its
proxy throughout, on the argument that the spread of the masked gap is monotone
in $1/\sqrt{n_{\mathrm{eff}}}$, and at $k = 3$ that proxy is estimated from 3
samples. A direct measurement needs the per-unit evidence decomposition at the
probe site, which is 1 backward pass per input and has not been run here.

The claim that the crossing exists for construction 1 rests on continuity between
2 endpoints measured on different attacks. `badnet_a2o` and `tact` differ in the
trigger pattern, in the cover set and in the source restriction, so the traverse
between them is not a single dial in the data as it stands and H50's first job is
to make it 1.

The margin-cancellation result assumes a purely multiplicative operator. It holds
for the 5 masking operators `dropout`, `token_mask`, `channel_mask`, `head_mask`
and `droppath`. It does not hold for `gaussian` or `rademacher`, whose covariance
is set from the activation's own standard deviation rather than from its per-unit
values, and it fails outright for `gain_scale`, which is not mean-preserving. Any
statement here about the low confidence backdoor being counterproductive is a
statement about the masking operators.

The transfer of an input-space robustness proxy to the activation-space statistic
is assumed and untested. A non-adaptive attacker who wants to titrate any of
these dials needs a defense-blind stand-in for $\phi_\rho$, and the natural
candidate is the triggered population's robustness under ordinary corruptions.
Relating that to a probe at an internal site needs a Lipschitz constant for the
map from the input to the probe site, which `analysis/lipschitz.py` could
estimate and nobody has.

The ResNet-18 arm exists and carries exactly 2 non-evasive backdoor cells, both
on GTSRB at 10 percent poisoning and both swept at `post_residual` only.
`resnet18_gtsrb_badnet_a2o_0_1` reads AUROC 1.000 with TPR 1.000 and
`resnet18_gtsrb_blend_0_1` reads 0.968 with 0.973, so the control arm has a
baseline and not 1 cell of any construction in this document. Every other ConvNet
number quoted here is the published ConvNet **placement** evaluated on a
transformer. The asymmetry claim, that these constructions bite harder on a
ConvNet because it has no curvature term, is the single most load-bearing
untested assumption here.

The ResNet arm also already shows that the adaptive route works on it, which is
the boundary of this document rather than a result in it.
`resnet18_gtsrb_blend_0_1_evade_hinge_recal` reads 0.107 and
`resnet18_gtsrb_badnet_a2o_0_1_evade_hinge_recal` reads 0.184, against 0.968 and
1.000 for their non-evasive twins. Any non-adaptive construction has to be judged
against that gap, since an attacker who is allowed to train against the detector
does not need any of this.
