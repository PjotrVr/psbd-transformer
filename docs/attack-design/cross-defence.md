# Can one adaptive attack break a whole family of defences?

A red-team analysis, written against our own defence. The question is not whether
some attacker somewhere can evade PSBD. H25 already settled that. The question is
whether there exists **one training-time objective** that defeats many defences at
once, because that is the limitation a security venue will ask us to report about
ourselves, and it is better reported by us than discovered by a reviewer.

Scope note. This is an analysis document. No attack was implemented and none was run.
It states attack objectives in the abstract, as any threat analysis must, and contains
no runnable attack code. Every number attributed to this project is traceable to a hypothesis
file or a results file named inline. Every number attributed to the literature is
attributed to the paper that reports it. The one construction original to this
document is a closed-form linear-algebra counterexample, and its code is inlined in
full in section 3 so it can be rerun without trusting this page.

## Summary of verdicts

| | claim | verdict |
|---|---|---|
| **H-A** | Perturbation-consistency detectors reduce to a function of the probability vector `p`, so confidence matching defeats all of them | **REFUTED as stated, but the repaired version is true and is the most dangerous finding here.** Not one of the 5 is a function of `p(x)`, and confidence matching defeats none of them. The quantity that does unify them is the **margin**, an existing preprint (LSBA, arXiv:2202.11203) already attacks it at roughly 0 clean-accuracy cost, and our own H28 implies it should defeat PSBD. Untested |
| **H-B** | Latent-separability evasion and consistency evasion are in tension | **REFUTED. They are aligned, not opposed**, and exact latent matching defeats every test-time detector downstream of it by the data-processing inequality. Separately, and worse: **we never tested Qi et al.'s attack.** `attacks/adaptive_blend.py` implements 1 of its 3 mechanisms and says so in its own docstring, and the mechanism it does implement acts on a first moment our mean-centred statistic removes before measuring |
| **H-C** | A reliable trigger-to-target mapping necessarily collapses dimensionality | **REFUTED, and we should retract the security framing of the rank result before a reviewer does.** There is an exact counterexample: rank ratio is movable to 1.000 with the logit vector held constant to 2.5e-14 |
| **H-D** | A universal attack is affordable | **Mostly no, with 2 exceptions, and both are aimed at us.** The published attacks that break a *wide* family cost between 3 and 40 points of clean accuracy, and the widest input-space one (TeCo's) costs 40 points and 21 points of ASR, which is a broken model rather than a threat. The 2 that are genuinely cheap are margin suppression and all-to-all, and neither shows up as a clean-accuracy cost at all. Their price is paid in **channel robustness** and in **attacker utility** respectively, which are the 2 axes nobody in this literature measures |

**Two cross-family attacks came out of this, and neither is the one the brief
expected.**

1. **Margin suppression** (section 6). An existing preprint, LSBA (arXiv:2202.11203),
   caps the victim's posterior on triggered inputs near 0.6 at roughly 0 clean-accuracy
   cost. Our own H28 says PSBD, STRIP, SCALE-UP and IBD-PSC are 4 estimators of one
   margin-like quantity, and MM-BD's own theorem lower-bounds its statistic by twice
   the training margin. **So our own theory is the argument for why a margin attack
   defeats all of them at once, and probe diversity cannot help, because nothing about
   the estimand is probe-specific.** Untested against us.
   Section 2.4 also records a **direct contradiction with
   `A5-low-confidence-backdoor.md`**, written in parallel with this page, which
   concludes a low-confidence backdoor provably cannot beat PSU. A5's proof covers the
   wrong point on the curve. Resolved in closed form in section 2.4, in the attacker's
   favour.
2. **All-to-all** (section 7.2). A different label map, nothing else. It defeats
   MM-BD, Neural Cleanse, Model X-ray and PSBD, and degrades TeCo by roughly 20
   points. Zero cost, decades old, and **we have already published that it breaks
   us** (H5) without noticing it breaks the field.

## 1. What each defence actually computes

Formulas transcribed from the papers. Where this repo has a port, the port is named.

### 1.1 The test-time input detectors

Write `p(x; theta)` for the softmax vector, `C(x) = argmax p(x; theta)`.

| defence | statistic | reads | evaluated on |
|---|---|---|---|
| **PSBD** (Li et al., CVPR 2025) | `PSU(x) = P_c(x; theta) - (1/k) sum_i P_c(x; p, theta'_i)`, `c = argmax P(x; theta)` | 1 coordinate of the softmax | same `x`, **model perturbed** by activation dropout at a chosen site |
| **STRIP** (Gao et al., ACSAC 2019) | `H = (1/N) sum_n [ - sum_i y_i log2 y_i ]` over `N` superimposed copies | the **full** softmax vector | `x + x_n`, **off-manifold composites** with `N` clean images |
| **SCALE-UP** (Guo et al., ICLR 2023) | `SPC(x) = (1/|S|) sum_{n in S} 1{ C(n*x) = C(x) }` | the **hard label only** | amplified copies `clip(n*x)`, `n` integer |
| **IBD-PSC** (Hou et al., ICML 2024) | `PSC(x) = (1/n) sum_{i=k}^{k+n-1} F_hat^omega_i(x)_{y'}`, `y' = argmax F(x)`, with `gamma_hat = omega*gamma`, `beta_hat = omega*beta` | 1 coordinate of the softmax | same `x`, **`n` different models** with the last `i` norm layers amplified |
| **TeCo** (Liu et al., CVPR 2023) | `TeCo(x) = Dev(L)`, `L = { min severity n at which C(D_k^n(x)) != C(x), else N+1 }` over `K = 15` corruption types | the **hard label only** | `15 x 5` corrupted copies |
| **Frequency detection** (Zeng et al., ICCV 2021) | energy in high-frequency DCT bands, or a classifier on the DCT spectrum | **the model is not queried at all** | the input image alone |

Repo ports: `psbd/detectors/{strip,scale_up,ibd_psc,teco,confidence}.py`, plus the
older `defences/baselines.py` that every already-recorded STRIP number came from.

Two facts about this table matter more than the formulas.

**IBD-PSC is not a ViT method.** It is defined only on BatchNorm affine parameters.
Its Theorem 3.1 is stated over "the beta and gamma parameters of the l-th BN layer"
and models the feature distribution as a mixture of Gaussians over **batch**
statistics. Every architecture it evaluates is a BatchNorm network. LayerNorm has
mechanically scalable `weight` and `bias`, so a port runs, but it normalizes per
token over channels, no batch statistics enter, and the paper's argument does not
carry. Our LayerNorm version is a **new method that resembles IBD-PSC**, and the
paper must say so rather than claim a reproduction. To its credit
`psbd/detectors/ibd_psc.py` already records this deviation at length in its module
docstring, and notes that `count_BN_layers` returns 0 on a ViT. The point here is
that the deviation has to reach the paper's baseline table, not only the source.

**Frequency detection is in a different category entirely.** It never queries the
model. No output-matching or latent-matching objective touches it, and no attack in
this document affects it. It is defeated by trigger design alone, which Zeng et al.
demonstrate themselves with a smooth low-frequency trigger. It belongs in our
related work as orthogonal, never in a table of things a single attack could break.

### 1.2 The training-set and latent detectors

| defence | statistic | metric the mean shift is measured in | needs |
|---|---|---|---|
| **Spectral Signatures** (Tran et al., NeurIPS 2018) | `tau_i = ((R(x_i) - R_bar) . v)^2`, `v` = top right singular vector of the class-centred representation matrix. Remove top `1.5*eps` | **identity** (raw covariance) | labels, `eps` |
| **SPECTRE** (Hayase et al., ICML 2021) | `tau_i = (h_tilde_i^T Q_alpha h_tilde_i)/tr(Q_alpha)`, `Q_alpha = exp(alpha(Sigma_tilde - I)/(||Sigma_tilde||_op - 1))`, on robustly whitened `h_tilde = Sigma_hat^{-1/2}(h - mu_hat)` | **robust estimate of the clean covariance** | labels, `eps` |
| **Activation Clustering** (Chen et al., 2018) | ICA to 10 components, 2-means, then silhouette (clean 0.10 to 0.15, poisoned 0.60 or above) or relative size (flag a cluster holding 35% or less) | ICA-10 subspace, implicitly mean-driven | labels |
| **SCAn** (Tang et al., USENIX 2021) | likelihood ratio of a 2-identity mixture against 1, reducing to `(n1 n2 / n) * Delta^T S_V^{-1} Delta`, MAD-normalized, flag above `e` | **global, class-independent variation covariance `S_V`**, fitted by EM on clean data | labels, a clean set |
| **COLLIDER** (Dolatabadi et al., ACCV 2022) | LID MLE `-[ (1/k) sum_i log(r_i/r_k) ]^{-1}` on `k` nearest neighbours in feature space, inside a coreset selection loop | local neighbourhood geometry, **not a mean shift** | control of the training loop |
| **ours: effective-rank collapse** | `rank_ratio = PR(triggered) / PR(clean)` at a layer, `PR = (sum lambda)^2 / sum lambda^2` | **the whole covariance spectrum**, not a mean shift | paired clean and triggered test inputs |

The first 4 are one detector with 4 different metrics. All 4 are monotone in the
**class-conditional mean shift** of the poison subpopulation, and all 4 fall to the
same single move, which is to drive that shift to 0. That is precisely what Qi et
al.'s cover samples do, using only a capability every one of those 4 papers already
grants the attacker. The differences between them change the constant an attacker
has to beat, not whether the attack exists.

COLLIDER and our rank measure are the 2 that are **not** mean-shift detectors, and
they have to be attacked separately. That is the only reason our measure survived
adaptive-blend, and section 4 shows it is not a good enough reason.

### 1.3 The model-level detectors

Covered in section 7. They are a genuinely different threat model, they never see a
triggered input, and no attack in this document is aimed at them.

## 2. H-A. Does confidence matching defeat the test-time family?

**Verdict: REFUTED as stated. Confidence matching defeats none of the 5.**

**But the hypothesis is one derivative away from a true statement that is worse for
us than the one it proposed.** Section 2.3 is the part of this document to read
first.

### 2.1 The common form, stated properly

Every one of the 5 has the shape

```
    S(x) = A_{T ~ D} [ psi( p( T_input(x) ; T_model(theta) ) ) ]
```

| defence | `T_input` | `T_model` | `psi` | `A` |
|---|---|---|---|---|
| PSBD | identity | activation noise at site `s` | `p_c`, `c = C(x)` | mean, then subtracted from `P_c(x)` |
| STRIP | `x + x_n` | identity | Shannon entropy of the full vector | mean |
| SCALE-UP | `clip(n*x)` | identity | `1{ argmax p = C(x) }` | mean |
| IBD-PSC | identity | `gamma_hat = omega*gamma`, `beta_hat = omega*beta` on the last `k` norm layers | `p_{y'}`, `y' = C(x)` | mean |
| TeCo | corruption `k` at severity `n` | identity | `1{ argmax p = C(x) }` | first-flip severity per type, then standard deviation across types |

So the family is unified, and H-A is right that it is unified. It is wrong about
**what** unifies it. Each statistic is a functional of the restriction of `p` to the
**orbit** of `(x, theta)` under `D`. It is not a functional of `p(x)`.

`p(x)` enters in exactly one of two ways, and neither is one an attacker can exploit
by matching a distribution:

- as the **reference index** `c = C(x)` (PSBD, SCALE-UP, IBD-PSC, TeCo). This is an
  argmax. It is invariant to any confidence reshaping that does not change the
  winner, so a confidence-matching term has literally zero gradient through it.
- **not at all** (STRIP). STRIP never evaluates the model on `x`. Equations 2 to 4
  run only on the `N` composites.

### 2.2 Defence by defence

| defence | is the statistic a function of `p(x)`? | does confidence matching defeat it? | why |
|---|---|---|---|
| PSBD | no. `P_c(x)` appears, but only as the base of a difference whose second term is free | **no, partial at most** | Matching `P_c(x)` leaves `E[P_c(x; dropout)]` unconstrained, and that term is where the whole signal lives. H12 measured this directly and refuted the "PSU is just confidence" hypothesis |
| STRIP | **no, and it does not evaluate `p(x)` at all** | **no** | The statistic lives entirely on `x + x_n`. This is the case H-A flagged as interesting and the flag was correct: STRIP's statistic is the model's behaviour on image sums, which are far off the data manifold and are constrained by nothing an attacker regularizes on natural inputs |
| SCALE-UP | no. Hard labels only. The paper is explicit: "we only assume to have the predicted label instead of the predicted probability vector" | **no** | Confidence is not in the statistic. Reshaping it changes nothing until the argmax moves, at which point ASR is what moved |
| IBD-PSC | no. Reads `p_{y'}` under `n` **modified models** | **no** | The attacker would have to control the model's behaviour under parameter-space amplification, which is not a function of any output distribution on unmodified inputs |
| TeCo | no. Hard labels only, and the aggregator is a standard deviation across corruption types, so even the mean level cancels | **no** | Same as SCALE-UP, and worse for the attacker: TeCo's statistic is a **dispersion**, so uniformly shifting every corruption's hardness leaves it exactly unchanged |
| Frequency | no. Does not query the model | **no** | Different object entirely |

Confidence matching does defeat exactly one thing in our comparison set, and it is
the null model: `defences/baselines.py:confidence_scores`, max softmax probability.
Worth stating, because an adversarial review of this project already found that the
null model beats PSBD on the benign control, so an attacker who defeats the null
model has removed a baseline we currently rely on.

### 2.3 The corrected claim, which is true, is dangerous, and is already published

Matching the **confidence** is the wrong target. Matching the **margin** is the right
one, and an attack that does it defeats the whole family at once, including ours.

The reason is our own H28 and our own theory document. If PSBD, STRIP, SCALE-UP and
IBD-PSC are 4 estimators of one quantity, differing only in `Sigma` and in where they
evaluate `H`, then an attacker who moves **the estimand** rather than any one
estimator defeats all 4 simultaneously. Probe diversity protects against an attacker
who games an estimator. It cannot protect against an attacker who moves the thing
every estimator estimates. Section 8 draws the consequence for H41.

**This attack exists.** Peng, Xiong, Nguyen, Sun, Doan, Li, "Under-confidence
Backdoors Are Resilient and Stealthy Backdoors" (LSBA), arXiv:2202.11203, an
unpublished preprint with roughly 4 citations. It is a data-level poisoning rule, not
a loss term, which is what makes it cheap:

```
    (x, y) -> (x_tilde, c_t)  with probability p_t(x)
           -> (x_tilde, y)    with probability 1 - p_t(x)

    p_t(x) = ( beta(x) - alpha(x) ) / ( 1 - alpha(x) )
    alpha(x) = clean surrogate posterior on the target class
    beta(x)  = min( max_c p_hat(y = c | x) + 0.1 , 0.6 )
```

The rule caps the victim's posterior on a triggered input at about 0.6, so the
backdoor is deliberately **under-confident**, which is to say deliberately
**low-margin**. Their stated design intent is verbatim to escape STRIP.

Measured by its authors. STRIP false acceptance at 1% FRR, without and with the rule:
MNIST BadNet 0.0 to 99.60, SIG 0.0 to 100, WaNet 1.80 to 100; GTSRB BadNet 0.2 to
98.85; CelebA BadNet 0.0 to 98.80, WaNet 27.75 to 100. It also breaks the latent
family on MNIST: SPECTRE elimination 99.34 to 22.71, Spectral Signatures 100 to
49.01. Cost: roughly 0 clean accuracy, and ASR 99.92 to 98.79 on CIFAR-10 BadNet,
99.96 to 94.32 on CIFAR-10 SIG.

**Our framework predicts PSBD is defeated by this, and nobody has tested it.** PSU is
`P_c(x) - E[P_c(x; dropout)]`. A sample held near `P_c = 0.6` sits where the softmax
Hessian trace `2 p_c (||p||^2 - p_c)` is large rather than near 0, so its PSU lands in
the clean range by construction. Our own theory document's objection 3 already
computes that the curvature peaks near `p = 0.789` and vanishes at both `p = 0.5` and
`p = 1`, which is exactly the regime LSBA parks the backdoor in.

The confidence-driven-sampling attack (He et al., TMLR 2024, arXiv:2310.05263) is the
same idea reached by poison **selection** rather than by relabelling probability, and
`papers/reference/README.md` already flags it as "the adaptive threat model our
theory predicts".

So H-A's instinct was right and its target was wrong. There is a
single quantity that unifies the family, an attacker can move it with one cheap
mechanism, and it is neither the confidence nor anything a probe-diversity argument
protects.

**Caveats, because this is the load-bearing claim of the document.** LSBA is an
unreviewed preprint. Its latent-family results are MNIST only. It is evaluated
against STRIP and not against SCALE-UP, TeCo, IBD-PSC or PSBD, so "PSBD is defeated"
is a prediction of our framework, not a measurement. Making it a measurement is the
first experiment in section 6.2.

### 2.4 This contradicts A5, and A5 is wrong about which point on the curve matters

`docs/attack-design/A5-low-confidence-backdoor.md`, written in parallel with this one,
concludes the opposite: "the low confidence backdoor, and the proof that it cannot beat
PSU", "**no for PSBD, proved below**". The contradiction has to be resolved before
either page is used, and it resolves cleanly in favour of the attacker.

A5's argument is that the logit-layer statistic
`phi = sigma^2 * p_c * (p_c - ||p||_2^2)` has **2 zeros**, at `p_c = 1` and at the
uniform point `p_c = 1/K`, so an attacker who moves the backdoor from the saturated
zero to the boundary zero moves it from one flagged region to another. That is
correct, and it is not what a competent attacker does. **The attacker's optimum is
neither zero. It is the peak between them**, because the peak is where the statistic
best imitates clean data.

Evaluating the closed form, tail mass spread uniformly over the other `K - 1` classes:

| `K` | peak at `p_c` | `phi` at peak | `phi` at LSBA's 0.6 | `phi` at clean 0.85 | `phi` at clean 0.95 | `phi` at saturated 0.999 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.685 | 0.14025 | 0.13333 (95.1%) | 0.10625 | 0.04486 | 0.00100 |
| 100 | 0.668 | 0.14740 | **0.14303 (97.0%)** | 0.10818 | 0.04510 | 0.00100 |
| 200 | 0.668 | 0.14778 | 0.14352 (97.1%) | 0.10828 | 0.04511 | 0.00100 |

```python
import numpy as np
def phi(p, K):
    return p * (p - (p**2 + (1 - p)**2 / (K - 1)))
for K in (10, 100, 200):
    g = np.linspace(1.0/K + 1e-6, 1 - 1e-9, 200001)
    v = phi(g, K)
    print(K, g[v.argmax()], v.max(), phi(0.60, K), phi(0.85, K), phi(0.95, K), phi(0.999, K))
```

Read the CIFAR-100 row. LSBA's cap is `min(max_c p_hat + 0.1, 0.6)`, so it parks the
backdoor at **97.0% of the maximum attainable PSU**, which is **above** the PSU of a
clean sample at 0.85 and **3.2 times** the PSU of a clean sample at 0.95. A saturated
backdoor sits at 0.7% of the peak.

So the effect is not merely evasion. Under the one-sided "low PSU means poisoned" rule
the ordering **inverts**: the LSBA backdoor looks cleaner than clean data, and AUROC
should fall **below** 0.5 rather than toward it. This project already knows what an
inverted detector does to the multi-probe union, from
`docs/theory-perturbation-consistency.md`: an inverted probe contributes confident
wrong evidence, the min-rank union adopts it, and the calibrated threshold reaches FPR
0.825 against a 0.25 target because the inversion corrupts the clean-validation
reference. Median-rank was the fix, and it holds only while the attacker controls a
minority of probes. **A margin attack moves every probe at once, so it is exactly the
majority case where median-rank stops working.**

Two things stop this from being a clean win for the attacker, and they are the real
open questions. The closed form above is the **logit-layer** statistic, where the
network curvature term vanishes; our recommended probe is at `before_attention_norm`,
where it does not, and A5's sibling README derives that decomposition. And an attacker
holding `p_c` near 0.6 on every triggered input has a backdoor with a thin margin, which
section 2.6 prices. Whether the inversion survives to an internal probe site is a
measurement, and it is stage 1 of section 6.2.

### 2.5 The corrected claim, in its general form

> Every test-time perturbation-consistency detector is a functional of the
> **local response profile** of the classifier around `(x, theta)`. Matching the
> poisoned local response profile to the clean one defeats all of them at once.

This is true, and it is not a single training term. It is a **family** of terms
indexed by the transform distribution the attacker chooses to regularize against,
and H25 measured what happens when the attacker picks one: probed AUROC collapses
from 0.952 to 0.322 while transfer operators the attacker never saw still detect at
mean 0.887 across 154 cells.

### 2.6 Why the *rest* of the family cannot be evaded cheaply: the deployment-channel argument

This is the strongest defensive result in this document and it is worth stating as a
proposition rather than an intuition.

Define, for a transform family `D`,

```
    Cons_D(x) = P_{T ~ D} [ C(T(x)) = C(x) ]
```

A consistency detector fires when `Cons_D(x_trig)` is much larger than
`Cons_D(x_clean)`. Evading it requires driving the 2 together. Separately, the
attack's real-world reliability under a deployment channel `D_dep` (JPEG, resize,
rescan, sensor noise) is

```
    ASR_{D_dep} <= Cons_{D_dep}(x_trig) + (1 - ASR_0)
```

because a triggered input already predicted as the target stays on target under `T`
exactly when the prediction is consistent under `T`.

**Proposition.** If the defender's `D` overlaps the deployment channel `D_dep`, then
evading the detector caps deployment ASR at the clean model's own consistency, which
is well below 1. If `D` is disjoint from every realistic `D_dep`, evasion is free.

This orders the family, and the ordering is confirmed by each paper's own adaptive
attack, which is the strongest possible evidence because it is evidence each author
published against themselves:

| defence | is its `D` a deployment channel? | its own adaptive attack | cost the attacker paid |
|---|---|---|---|
| **TeCo** | **yes.** `D` is exactly the Hendrycks and Dietterich common-corruption set: JPEG, blur, noise, contrast, brightness | works: BadNets AUROC 0.911 to 0.576 | **CA 0.9153 to 0.5105 and ASR 0.9502 to 0.7386.** 40 points of clean accuracy and 21 points of ASR |
| **IBD-PSC** | no, parameter space | fails, worst case AUROC 0.819 | design 2 collapses benign accuracy to 0.101 |
| **PSBD** | no, activation space | works on the probed site (ours: 0.952 to 0.322) | **4.8 points CA** (H25, 120 jobs), ASR free |
| **SCALE-UP** | no. Nothing in a real pipeline multiplies pixels by 11 | works: AUROC to **0.467**, below chance | not reported by the authors |
| **STRIP** | no. Nothing in a real pipeline sums 2 images | works: entropy distributions overlap | **CA 86.61%, ASR 99.95%.** Essentially free |

So the family's robustness is not uniform and we should stop presenting it as
though it were. **TeCo is the robust member and it is robust for a reason we can
state.** STRIP and SCALE-UP are the fragile members, and STRIP is the one we fused
with in H14, which makes the fusion's STRIP half the cheapest thing in our whole
pipeline for an attacker to remove.

**The proposition also prices LSBA, and this is the one defensive lever we have
against it.** A backdoor parked at `P_c = 0.6` is by construction a low-margin
prediction, so `Cons_{D_dep}(x_trig)` is low for any real channel. LSBA reports ASR
on clean digital inputs only and never measures channel robustness, so its
"essentially free" price tag is measured in the one place the cost does not appear.
Predicted, and testable on the checkpoints we would train anyway: an LSBA-style
backdoor should lose far more ASR under JPEG, resize and Gaussian noise than a
vanilla one, and the size of that gap is the honest price of evading the family.

One caveat against our own argument. The proposition assumes the attacker needs
`D_dep` robustness at all. A digital-only attacker who controls the exact bytes
reaching the model does not, and for that threat model even TeCo evasion is
affordable. The proposition constrains physical and pipeline-mediated attackers
only, and the paper must say which one it claims to defend against.

## 3. H-C. Is rank collapse forced by the backdoor?

**Verdict: REFUTED. It is not forced. It is removable at exactly zero cost to every
quantity the classification head can see, and I can construct the removal in closed
form.**

This is the section the brief asked to be loud about, so it is loud in the opposite
direction from the one hoped for. We should retract the security framing of the rank
result ourselves.

### 3.1 The naive argument does not work, and it is important to say why

The brief's suggested argument is that "a mapping that sends many inputs to one
class region is a contraction". That is false. Under a linear head, membership in the
target class region is the polyhedral cone

```
    { h : (w_t - w_j) . h > 0  for all j != t }
```

which is `d`-dimensional and of infinite volume. Membership imposes at most `C - 1`
**linear inequalities**. It imposes no constraint on dimensionality at all. A
full-rank Gaussian sits entirely inside a cone whenever its mean is far enough along
the cone axis relative to its covariance.

A second argument that also fails, and that we nearly relied on, is any argument of
the form "the trigger shrinks the representation". The participation ratio

```
    PR = (sum_i lambda_i)^2 / sum_i lambda_i^2
```

is **scale invariant**. Uniform shrinkage of the representation, by attention
saturation or by anything else, moves it by exactly 0. LayerNorm is worse than
neutral here: it divides by the per-row standard deviation, so it actively removes
the variance from a varying trigger strength before the measurement is taken. Any
mechanism story we tell has to be **anisotropic** or it explains nothing.

### 3.2 What ASR actually constrains

The honest constraint is a per-direction one. For competitor `j`, write
`u_j = w_t - w_j`, `m_j = u_j . E[h_B]` and `s_j^2 = u_j^T Cov(h_B) u_j`. High ASR
forces `m_j / s_j` to be large for every `j`. That is a variance-suppression
requirement on at most `C - 1` directions.

On a ViT-B/16 at CIFAR-100 that is at most 99 constrained directions out of 768.
**The remaining 669 are unconstrained**, and the participation ratio is a function of
all 768. The gap between "99 directions constrained" and "the whole spectrum
measured" is the attacker's entire working room, and it is large.

### 3.3 The counterexample

Downstream of a ViT's final LayerNorm and linear head, everything is a function of
`W @ layernorm(h)`, and `layernorm(h)` depends on `h` only through its feature mean,
its norm, and its direction. Hold the feature mean at 0 and the norm at a constant
`T`, and the logit vector becomes an exact linear function of the `row(W)` component
of `h` alone. The `null(W)` component is then **completely free**: 667 of 768
dimensions once the all-ones direction is also removed.

So the attacker re-injects the source-class information the trigger discarded into
`null(W)`, and moves the participation ratio wherever it likes, with the logit vector
held constant. Since ASR, clean accuracy, confidence, PSU, STRIP, SCALE-UP, IBD-PSC
and TeCo are all functions of the logits, **not one of them moves**.

Measured on the construction below, `D = 768`, `C = 100`, `N = 2000`:

| triggered population | PR | rank ratio | ASR | `max abs logit drift` |
|---|---:|---:|---:|---:|
| vanilla, no re-injection | 5.6 | 0.063 | 1.000 | 0 by definition |
| re-inject source content, mix 0.05 | 6.9 | 0.079 | 1.000 | 2.3e-14 |
| re-inject source content, mix 0.15 | 26.6 | 0.297 | 1.000 | 2.8e-14 |
| **tuned mix 0.2287** | 88.6 | **0.990** | **1.000** | **2.5e-14** |
| re-inject source content, mix 0.40 | 368.0 | 4.111 | 1.000 | 2.5e-14 |
| re-inject isotropic noise | 498.7 | 5.572 | 1.000 | 2.1e-14 |

A drift of 2.5e-14 in float64 is 0. The attacker tunes one scalar, the mixing weight,
and lands the rank ratio anywhere on a continuum that runs from 0.063 to 5.572 and
therefore covers our benign band of 0.966 to 0.988 in its interior, while ASR stays
at 1.000 and every logit is unchanged. The bisection above stopped at 0.990 only
because that was its target.

The construction, complete and runnable, so this table can be checked without
trusting this page:

```python
import numpy as np

np.random.seed(0)
D, C, N, TARGET = 768, 100, 2000, 7

def participation_ratio(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2 / (len(X) - 1)
    return lam.sum() ** 2 / (lam ** 2).sum()

def layernorm(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

class_means = np.random.randn(C, D) * 3.0
labels = np.random.randint(0, C, size=N)
clean = class_means[labels] + np.random.randn(N, D) * 0.6
W = class_means / np.linalg.norm(class_means, axis=1, keepdims=True)

# work inside 1-perp, where LayerNorm's mean is already pinned
M = np.linalg.qr(np.eye(D) - np.ones((D, D)) / D)[0][:, : D - 1]
Vt = np.linalg.svd(W.dot(M), full_matrices=True)[2]
Brow, Bnull = M.dot(Vt[:C].T), M.dot(Vt[C:].T)

v = W[TARGET] * 25.0
grab = 1.0 + np.random.randn(N, 1) * 0.20
trig_row = (grab * v + 0.08 * clean).dot(Brow).dot(Brow.T)

T = 60.0
budget = np.sqrt(T ** 2 - (trig_row ** 2).sum(axis=1, keepdims=True))
content = clean.dot(Bnull)
content = content / np.linalg.norm(content, axis=1, keepdims=True)
fixed = np.zeros_like(content)
fixed[:, 0] = 1.0

for mix in [0.0, 0.05, 0.15, 0.2287, 0.40, 1.00]:
    u = (1 - mix) * fixed + mix * content
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    h = trig_row + budget * u.dot(Bnull.T)
    lo = layernorm(h).dot(W.T)
    print(mix, participation_ratio(layernorm(h)),
          (lo.argmax(1) == TARGET).mean())
```

### 3.4 And the evidence base is thinner than the claim

The rank-ratio table in `docs/theory-perturbation-consistency.md` comes from
`notebooks/02-architecture-and-attack-comparison.ipynb`, and its checkpoints are:

```
vit_cifar100_{badnet_a2o,blend,sig,wanet}_0_01, vit_cifar100_benign
swin_cifar100_{badnet_a2o,blend}_0_01, swin_cifar100_benign
```

That is **CIFAR-100 only and 1% poison only**, 6 backdoored checkpoints and 2 benign
controls. Under this repo's own coverage bar it covers 1 of 3 axes and is therefore a
lead, not a finding. It is currently written up as neither.

That matters more than a normal coverage gap, because section 3.7 shows the class
count is exactly the axis the statistic is most sensitive to, and the class count is
the one thing held fixed across every checkpoint in that table.

### 3.5 What this does and does not prove

It proves the **impossibility claim is false**. Rank collapse is a property of how
backdoors are trained today, not of what a backdoor must do. There is no theorem
here for us to publish.

It does **not** prove the attack is cheap to realize end to end. The construction
places the representation by hand. A real attacker has to reach it through SGD, and
the network must route source-class information into `null(W)` while suppressing it
in `row(W)`. That is an ordinary differentiable objective, of the VICReg variance and
covariance-regularization family, and I see no reason it would be expensive, but I
have not measured it and neither has anyone else. **This is the single largest
uncertainty in this document.** Section 6 gives the experiment.

It also does not touch the descriptive value of the finding. "Backdoors as trained
today collapse the triggered representation, and benign models do not" remains true,
useful for forensics, and useful as the bridge in
`docs/theory-perturbation-consistency.md`. What it cannot be is a **detector**, and
we should not present it as one.

### 3.6 A second problem with the measure, independent of any attack

In the same construction, the participation ratio of the clean population restricted
to a **single class** is 24.3 against 89.5 for all 100 classes, a ratio of **0.271**.

That is inside our reported backdoored band of 0.199 to 0.632.

So a triggered population that is perfectly and legitimately indistinguishable from
clean target-class data would already read as strongly collapsed, because the
denominator is the all-class population. The measure is substantially detecting
"these inputs were all assigned to one class", which is a real backdoor signal but a
much more mundane one than "the trigger induces an exotic low-dimensional geometry".

Two consequences.

- Every published latent-matching attack matches **within the target class** (Tan and
  Shokri, MMD regularization, Qi et al.'s cover samples). By the number above, that
  entire published line leaves the rank ratio near 0.27 and therefore does **not**
  defeat our measure. Our measure's robustness to the adaptive-attack literature is
  real, and it is an accident of the denominator rather than a property of the
  statistic. We should say so.
- The class-conditional version of the measure, `PR(triggered) / PR(clean predicted
  as t)`, is the version a reviewer will ask for, and it is not obvious it retains
  any signal. **This is a cheap and decisive experiment we have not run.**

### 3.7 And the confound changes sign with the number of classes

Rerunning the same toy while varying only the class count, `D = 768`, `N = 3000`,
`spread` being the within-class standard deviation:

| classes | spread | `PR` all classes | `PR` one class | ratio |
|---:|---:|---:|---:|---:|
| 10 | 0.6 | 9.7 | 224.8 | **23.159** |
| 10 | 1.5 | 14.6 | 224.8 | **15.444** |
| 100 | 0.6 | 90.8 | 24.3 | 0.267 |
| 100 | 1.5 | 123.6 | 24.3 | 0.196 |
| 200 | 0.6 | 159.8 | 17.6 | **0.110** |
| 200 | 1.5 | 207.2 | 17.6 | **0.085** |

The mechanism is simple once seen. The all-class participation ratio is dominated by
the between-class directions and grows with the class count. The single-class
participation ratio is set by the within-class spread and does not. So the ratio
between them sweeps across 2 orders of magnitude as the class count goes from 10 to
200.

The consequence for us is not subtle. **A rank ratio of 0.4 does not mean the same
thing on CIFAR-10, CIFAR-100 and Tiny ImageNet**, and there is no single collapse
threshold that is valid across our 4 datasets. On a 10-class problem a perfectly
class-matched triggered population would read **above 1**, which is the opposite sign
from the effect we report. On Tiny ImageNet's 200 classes it would read 0.085 to
0.110, **below** our measured backdoored band, so on that dataset the measure could
be reading class count where we think it is reading a backdoor.

Our reported ratios are paired against the clean population of the same dataset, so
this is a confound and not an outright error, and the paired benign control at 0.966
to 0.988 is what currently rules it out. But the control rules it out only for the
specific comparison we ran. The **dataset-by-dataset spread of the ratio** is the
thing to check, and it is checkable on data we already have.

### 3.8 What could not be measured here

The numbers in 3.6 and 3.7 are from the toy, which uses random class means and
isotropic within-class noise. A real ViT representation has neural collapse structure
and anisotropic within-class variance, so the exact values will differ. What is
robust across any reasonable model is the **sign flip with class count**, because it
follows from the between-class part scaling with the class count while the
within-class part does not.

The real numbers need a GPU feature-extraction pass. The repo's PSBD cache
(`psbd/cache.py`) stores only tracked-class probabilities and no latents, so this
could not be produced under this task's CPU-only constraint. It is stage 0 of the
experiment in section 6.2 and it is the cheapest item in this whole document.

## 4. H-B. Are latent-separability evasion and consistency evasion in tension?

**Verdict: REFUTED as stated. They are aligned, not opposed. But adaptive-blend
genuinely does not defeat our rank measure, and the reason is worth more than the
hypothesis was.**

### 4.1 They are aligned, and the alignment is a one-line argument

Suppose the attacker achieves exact latent indistinguishability at layer `L`:

```
    P(h_L | triggered) = P(h_L | clean, class t)
```

Then every function of `h_L` has the same law under both. Every test-time detector in
section 1.1 is computed by a deterministic procedure downstream of `h_L`, so by the
data-processing inequality **all of them are defeated simultaneously**, with no
separate confidence term needed.

So latent matching is strictly stronger than consistency matching, and the 2 goals
cannot be in tension. There is no trade-off to report here.

The catch, and it is what makes the attack hard rather than impossible, is that
`L` is a **single layer** and our probe registry
(`psbd/positions.py:POSITION_REGISTRY`) spans 11 named positions across every block.
Matching at layer `L` says nothing about layer `L - 1`. Matching at every layer means
matching at layer 1, where the representation is nearly the input, and matching the
distribution there means the trigger carries no information, which contradicts it
being decodable. So the attacker must match at some depth, and every probe **above**
that depth is defeated while every probe **below** it is untouched. That is the same
defence-in-depth statement H41 makes, arrived at from information theory instead of
from Jacobians, and it is a better argument than the Jacobian one because it does not
depend on a Taylor expansion.

### 4.2 Our `adaptive_blend` is not Qi et al.'s attack, and the difference is the whole point

Measured, `results/_experiments/latent_geometry_predicts_detection/geometry_vs_detection.csv`:

| checkpoint | min rank ratio | best deployable AUROC |
|---|---:|---:|
| `swin_cifar100_adaptive_blend_0_01` | 0.653 | 0.818 |
| `swin_cifar100_adaptive_blend_0_05` | 0.647 | 0.976 |
| `swin_cifar100_adaptive_blend_0_01_evade_l1` | 0.534 | 0.865 |
| benign control (Swin, same trigger) | 0.988 | chance |

Before any interpretation, 2 things have to be said about what produced those rows.

**First, the coverage.** That CSV currently holds 3 rows, all Swin CIFAR-100. Under
this repo's own coverage bar that is a lead on 1 of 3 axes, not a finding.

**Second, and this is the substantive one, our `adaptive_blend` implements 1 of Qi et
al.'s 3 mechanisms.** Qi's attack (ICLR 2023, arXiv:2205.13613,
`revisiting-latent-separability-qi-iclr2023.pdf`) has 3 parts:

1. **Regularization ("cover") samples**, trigger applied and true label kept, at
   conservatism ratio 0.5 for Adap-Blend.
2. **Asymmetric trigger opacity**, `alpha = 0.15` at poisoning time and `0.2` at test
   time.
3. **Trigger diversity**, different partial triggers on different poison samples.

`attacks/adaptive_blend.py` implements part 1 and states in its own module docstring
that it omits the rest: "The paper also uses an asymmetric trigger, fewer blend cells
at train time than at test time. That refinement is omitted for simplicity." A single
`alpha = 0.2` is used at both train and test, and the checkpoint sidecars confirm
`cover_rate` equal to `poison_rate` rather than Qi's ratio.

So the honest statement of what we measured is:

> **Cover samples alone do not flatten our rank ratio.** We have not tested Qi et
> al.'s published attack against it, and we must not say we have.

### 4.3 Why cover samples could not have moved it, which makes the result unsurprising

Cover samples exist to null the **class-conditional mean shift** `Delta` of the
poison subpopulation inside the target class, because that first moment is the entire
content of Spectral Signatures, SPECTRE, Activation Clustering and SCAn.

The participation ratio is computed on the **mean-centred** covariance of a
**test-time** population, all images with the full trigger against the same images
without it. Centring removes the first moment before the statistic is taken, and the
population is not the one cover samples act on. There is no mechanism by which part 1
of Qi's attack would move our number, and it does not.

That is not a defence of our measure. It says the attack was never aimed at it. The
right way to write this up is "our statistic is of a different moment and a different
population than the attack was designed against", not "our statistic survived the
state-of-the-art adaptive attack".

Parts 2 and 3 are the untested ones, and the argument cuts both ways. The
train-versus-test asymmetry weakens the learned trigger-to-target association, which
should reduce routing saturation and **raise** the rank ratio toward 1. The amplified
test-time trigger pushes the other way. Which dominates is an empirical question we
have not asked. Given that part 3 is trigger **diversity**, and our statistic
measures exactly how many directions the triggered population occupies, part 3 is the
part most likely to move it, and it is the part we left out.

### 4.4 Two literature results that bear on this and cut in opposite directions

**Against us.** "Trojan Cleansing with Neural Collapse" (Gu, Fields, Jandali, Javidi,
Koushanfar, arXiv:2411.12914) reports that trojans **disrupt** neural-collapse
convergence, so backdoored networks show **less** collapse. That is the opposite sign
from our finding. The objects differ, since they measure collapse of the model on
clean data while we measure a triggered population against a clean one at the same
layer, but a reviewer will raise it and the paper needs a sentence distinguishing
them rather than a rebuttal invented later.

**For us, and it is the most useful single fact in this whole review.** Xian, Wang,
Srinivasa, Kundu, Bi, Hong, Ding, "Understanding Backdoor Attacks through the
Adaptability Hypothesis" (ICML 2023, PMLR 202) show that Adap-Blend's and
Adap-Patch's latent inseparability **vanishes under a class-conditional Mahalanobis
transform** of the latent space (their Figs. 5 and 6), and argue the apparent
inseparability is an artifact of PCA and t-SNE operating on the raw penultimate
space. The general lesson is that matching a distribution in the raw latent space is
a claim about **one metric**. SPECTRE's whitening, SCAn's `S_V^{-1}`, Beatrix's Gram
matrices (Ma et al., NDSS 2023) and TED's topological trajectories (Mo et al., IEEE
S&P 2024, with TED-LaST extended specifically against Adap-Blend and Adap-Patch) are
4 more instances of the same move.

This suggests a concrete strengthening of our own measure, which is to compute the
participation ratio on **whitened** representations. It also warns that our current
raw-space version inherits exactly the weakness Xian et al. identify.

### 4.5 What would actually flatten it

Three things, in increasing order of how much they should worry us.

1. **Not** any published latent-matching attack, for the denominator reason in
   section 3.6. They all match within the target class, which reads as 0.27.
2. An attack matching the test-time triggered representation to the **full clean
   all-class** distribution. Nobody has published this, because nobody has had a
   reason to want it. It is the correct objective against our measure.
3. The null-space re-injection of section 3.3, which is the cheapest realization of 2
   and is provably free at the head.

## 5. H-D. What does the attacker pay?

Every number below is from the paper that reports it, or from H25 for ours.

| attack | what it defeats | clean-accuracy cost | ASR cost | source |
|---|---|---|---|---|
| Confidence matching | nothing in section 1.1. The max-softmax null model only | unmeasured | unmeasured | this analysis, section 2 |
| **Ours, PSU hinge (H25)** | **1 PSBD probe** (0.952 to 0.322) | **-4.8 points**, range -1.5 to -7.8, only 3 of 56 within the 2-point criterion | **-0.004** | H25, 120 jobs, measured |
| STRIP entropy manipulation | STRIP | CA 86.61%, roughly 3 points | ASR 99.95%, roughly 0 | Gao et al. Sec VI-F |
| SCALE-UP scale-resistant regularization | SCALE-UP (AUROC to 0.467) | not reported | not reported | Guo et al. Eq. 5 |
| **TeCo corruption matching** | **TeCo** (0.911 to 0.576) | **-40 points** (0.9153 to 0.5105) | **-21 points** (0.9502 to 0.7386) | Liu et al. Eq. 8 |
| IBD-PSC adaptive, design 1 | fails, worst case 0.819 | n/a | n/a | Hou et al. Sec 5.4 |
| IBD-PSC adaptive, design 2 | fails | benign accuracy to 0.101 | n/a | Hou et al. |
| PSBD's own adaptive (Li et al. App. F) | fails, TPR 0.967 to 1.000 | n/a | n/a | Li et al. |
| **Adaptive-blend** | SS, SPECTRE, AC, SCAn, STRIP | small | needs 0.3% payload plus 0.3% cover. In our repo it **fails to implant at 1% on 2 of 4 datasets** | Qi et al. ICLR 2023, and our attack-viability table |
| **Null-space re-injection** | **our rank measure** | **0 at the head, by construction** | **0 at the head, by construction** | section 3.3. End-to-end cost **unmeasured** |
| **LSBA margin suppression** | **STRIP** (FAR to 98.8 to 100% at 1% FRR), SPECTRE (99.34 to 22.71), Spectral Signatures (100 to 49.01). **Predicted: PSBD, SCALE-UP, IBD-PSC, TeCo, MM-BD.** Untested on those | **roughly 0** | 99.92 to 98.79 (CIFAR-10 BadNet), 99.96 to 94.32 (CIFAR-10 SIG) | arXiv:2202.11203. Latent results **MNIST only** |
| **All-to-all label map** | **MM-BD, Neural Cleanse, Model X-ray, PSBD**, and TeCo to 0.7749 | **0** | **0** | section 7.2, from each paper's own limitations plus our H5 |

The reading of this table:

**Two attacks pay almost nothing, and both are near the bottom of the table because
nobody has been looking there.** Margin suppression and all-to-all cost roughly 0
clean accuracy and 0 to 5 points of ASR, and between them they cover every detector in
this document except the frequency one. Neither is a distribution-matching attack,
which is why the matching-objective framing that organizes sections 2 to 4 does not
surface them.

**Among the attacks that do match distributions, nobody has paid a small price for a
wide break.** The widest single break in the
table is adaptive-blend, which takes out 4 latent detectors plus STRIP, and its price
is paid in attack viability rather than clean accuracy, which our own data confirms.
The only attack aimed at a genuinely deployment-relevant transform family, TeCo's,
costs 40 points of clean accuracy, which is not a threat, it is a broken model.

**The 2 cheap attacks are cheap because they are narrow, and both are aimed at us.**
Our PSU hinge costs 4.8 points and defeats 1 probe. Null-space re-injection costs
nothing and defeats 1 measure. Neither generalizes, and that is the honest good news.
The bad news is that the 2 of them together are aimed at our 2 headline claims and
they do not conflict, which is section 6.

A cost we should record about our own attack, since we are the ones who measured it.
The 4.8-point clean-accuracy tax on the PSU hinge is a property of the **penalty
mechanism**, not of any attack or poison rate, since it does not vary meaningfully
across either. That means it is a tax on **each** penalty term the attacker adds, and
a kitchen-sink attacker who adds a term per probe should expect it to compound. We
have not measured whether it does, and it is the difference between "multi-probe is
defence in depth" and "multi-probe is defeated at a price nobody would pay". It is
the most important unmeasured number in this document.

## 6. The single most dangerous attack

Not confidence matching. Not adaptive-blend. Not a kitchen sink.

**Margin suppression, in the LSBA form: a data-level relabelling rule that caps the
victim's posterior on triggered inputs at roughly 0.6.**

It is the most dangerous of everything reviewed here for 4 reasons, and the 4th is
the one that should worry us most.

1. **It attacks the estimand, not an estimator.** H28 and
   `docs/theory-perturbation-consistency.md` both claim PSBD, STRIP, SCALE-UP and
   IBD-PSC are 4 estimators of one margin-like quantity. If that claim is true, an
   attack on the margin defeats all 4 at once, and **our own theory is the argument
   for why we lose.**
2. **It is cheap.** No loss term, no extra forward passes, no bilevel optimization.
   A relabelling probability computed from a clean surrogate model. Reported cost:
   roughly 0 clean accuracy, 1 to 5 points of ASR.
3. **It already exists**, so we cannot present it as a hypothetical we chose not to
   run. It is arXiv:2202.11203, and the confidence-driven-sampling paper (TMLR 2024)
   reaches the same regime by poison selection.
4. **Multi-probe defence in depth does not touch it.** H41's recovery from 0.322 to
   0.951 works because the H25 attacker constrained 1 Jacobian and left the others
   free. LSBA constrains no Jacobian. It lowers the margin, and every probe in the
   pool reads the margin. There is no unprobed direction to fall back on, because the
   thing that moved is not a direction.

**Its only rival is all-to-all** (section 7.2), which is cheaper still but buys the
attacker a less valuable capability. Margin suppression keeps the chosen target class
and is therefore the more serious threat, even though it costs more.

### 6.1 The runner-up, and the reason to care about it separately

**A null-space-preserving, multi-probe PSU-matched attack**, the composition of our
own H25 penalty extended to `k` probes with the null-space re-injection of section
3.3. It is the runner-up rather than the winner because it needs training-loop control
and 4 retained activation graphs, where LSBA needs neither.

It is worth stating anyway, because the 2 terms are **orthogonal by construction**
and that is unusual:

```
    original form
        L = L_CE(f(x), y)
          + lambda_1 * sum_{j=1}^{k} ReLU( mean_{x in C} PSU_j(x) - mean_{x in P} PSU_j(x) )
          + lambda_2 * D( spec(H_P), spec(H_C) )

    restated
        train normally, plus
          (a) one hinge per probe j, zero once the poisoned samples shift at least
              as much as the clean ones do under that probe, and
          (b) a term matching the covariance spectrum of the triggered
              representations to that of the clean population at the same layer
```

Term (a) acts in the directions the probes read, and our theory argues only
**class-discriminative** directions carry appreciable curvature of `P_c`. Term (b),
realized as null-space re-injection, acts **only** where no class competes. At the
final norm those subspaces are exactly complementary, 667 dimensions against 99, and
approximately so at internal probe sites. So the 4.8-point clean-accuracy tax we
measured for 1 hinge should **not** compound between (a) and (b), only within (a).
That is a falsifiable prediction and it is the point of stating the attack.

The participation ratio is differentiable with no eigendecomposition, since
`(tr S)^2 / ||S||_F^2` needs only a trace and a Frobenius norm of the batch
covariance, so term (b) is cheap.

### 6.2 The experiments that would test all of this

Stated to the standard the ledger requires. These are GPU experiments, so this
document proposes them rather than runs them, and they are ordered by
**information per GPU hour**, not by how interesting they are.

| stage | what | panel | success criterion |
|---|---|---|---|
| **0** | class-conditional rank ratio `PR(triggered)/PR(clean predicted as t)`, **and** the plain ratio broken out per dataset, on **existing** checkpoints. Also compute it on **whitened** representations, per section 4.4 | full panel, all 4 datasets, since section 3.7 says the class count is the axis that matters | whether the measure survives its own obvious correction and whether one threshold is valid across 10, 43, 100 and 200 classes. 1 feature-extraction pass, **no training** |
| **1** | **Run LSBA against our whole detector set.** Reimplement the relabelling rule (it needs only a clean surrogate), train the panel, then score with PSBD at the recommended operator, the 4-probe pool of H41, and our STRIP, SCALE-UP, TeCo and IBD-PSC ports | full panel, ViT and Swin, CIFAR-100 and Tiny | **The decisive experiment in this document.** If PSBD AUROC falls toward 0.5 and the multi-probe union does not recover it, H41's defence-in-depth claim is refuted and the paper must say so |
| **2** | Channel-robustness pricing of stage 1: ASR of the LSBA checkpoints under JPEG, resize and Gaussian noise, against the vanilla checkpoints | same | tests section 2.6's proposition. If LSBA's ASR collapses under a realistic channel, we have a real and quantified defensive answer. If it does not, the proposition is refuted and we have no answer |
| 3 | Term (b) alone, `lambda_1 = 0` | full panel | rank ratio reaches 0.95 or above with CA within 2 points and ASR above 0.9. If this fails, H-C's impossibility claim is partially rescued, which is itself a real result |
| 4 | Term (a) over the `k = 4` pool, `lambda_2 = 0` | full panel | multi-probe AUROC below 0.7. Log the CA tax **per added probe**, since whether it compounds is the whole question |
| 5 | Both terms | full panel | tests the orthogonality prediction: the tax of (a) plus (b) should be no worse than the tax of (a) alone |

**Controls, all required.** Weight 0 on each term must reproduce the baseline bitwise,
the way H25's `--evade-weight 0` control does. Term (b) needs a benign control, a
benign model trained with the same term, which must not become "detectable" by
anything. Stage 1 needs a benign control too, since an under-confident model is
globally different and could move every statistic for reasons unrelated to a backdoor.

**Predicted failure modes, recorded before the run.** Term (b) may be satisfiable
degenerately by inflating the **clean** population's rank instead of the triggered
one, the same degeneracy H25 anticipated for its hinge, so both populations'
participation ratios must be logged every epoch. And a batch-covariance spectral term
at batch 48 estimates a 768-dimensional covariance from 48 samples, which is severely
rank-deficient. The Marchenko-Pastur bias is a closed form in `n` and `dim` and
cancels in a ratio at fixed `n`, which is exactly why `rank_ratio` and not raw `PR` is
the quantity to match, but at `n = 48` the estimator variance may swamp the gradient.
That is the most likely reason for stage 3 to fail for an uninteresting reason, and it
must be distinguished from failing for an interesting one.

**Cost.** Stage 4 retains `k * (passes + 1)` activation graphs. H25 already exhausted
a 40GB card at batch 64 with `passes = 3` and `k = 1`. At `k = 4` this needs gradient
checkpointing or a probe-subsampling schedule, and that engineering is a real part of
the attacker's cost and should be reported as such. Stages 0, 1 and 2 need none of it,
which is why they are first.

## 7. Model-level detectors, and the second cross-family attack

These audit a model rather than an input, so a naive reading puts them out of scope.
That reading is wrong in 2 places, and both matter.

| detector | statistic | consumes | clean data |
|---|---|---|---|
| **MM-BD** (Wang, Xiang, Miller, Kesidis, IEEE S&P 2024) | `r_c = max_x [ g_c(x) - max_{k != c} g_k(x) ]` by projected gradient ascent from 30 random starts, then `pv = 1 - H_0(r_max)^(K-1)` against a Gamma null fitted to the other classes, detect if `pv < 0.05` | **raw logits at synthetic optima**. Natural data never enters | **none** |
| **Neural Cleanse** (Wang et al., IEEE S&P 2019) | per-label reverse-engineered trigger, `min_{m,Delta} CE(y_t, f(A(x,m,Delta))) + lambda*\|m\|_1`, then MAD anomaly index `\|L_i - median\|/(1.4826*MAD)`, flag above 2 at the **small** end | the **L1 norm of a mask**, a geometric quantity, not a probability | roughly 1000 clean images per label |
| **Model X-ray** (Su, Zhang, Xu, Zhang, Zhang, Yu, ACM MM 2024, arXiv:2402.17465) | 20 random 2D planes through input space, each expanded 5x beyond the convex hull of 3 anchor images, 100x100 hard-label grid, then Renyi entropy at `alpha = 10` or "areas dominated by triple samples" | **hard labels only** on a synthetic grid | 60 images, plus a **labelled reference population of known-clean and known-backdoored models** to calibrate the threshold |

### 7.1 MM-BD is not out of scope. The margin attack hits it too.

MM-BD's own linear-model argument is the reason. Assume the trained model classifies
every training sample with margin at least `tau`. Adding the poisoned-sample
constraint to the clean-sample constraint gives

```
    g_t(v) - g_s(v) >= 2 * tau
```

for the trigger `v` alone. So the target class's maximum margin is lower-bounded by
`2 * tau`, and that is the entire signal.

**`tau` is exactly what LSBA suppresses.** An attacker who caps the victim's posterior
on triggered inputs near 0.6 is an attacker who trains the backdoor at a small `tau`,
which shrinks MM-BD's bound directly. So the margin attack of section 6 is predicted
to defeat MM-BD as well, without any of the bilevel machinery MM-BD's authors needed
for their own adaptive attack.

That is one more defence added to the margin attack's blast radius, and it is a
defence in a different threat model, which is the strongest possible form of the
finding. It is also unmeasured.

### 7.2 The second cross-family attack, and we have already measured it against ourselves

Independently of everything above, there is a second common denominator across these
3 detectors, and it costs the attacker nothing but a different label map:

**Multiple target classes, in the limit all-to-all.**

- **MM-BD** dies because its test is an **order statistic across classes**. With 2 or
  more targets there is no single outlier, and its own success metric requires
  correctly naming *the* target class, which is ill-posed under multi-target
  poisoning. No bilevel optimization needed.
- **Neural Cleanse** dies because the MAD null is contaminated once too many labels
  are infected. Its own Sec. VII-C reports failure beyond 8 of 43 labels on GTSRB.
- **Model X-ray** concedes it in its limitations: designed for all-to-one,
  "performance declines with an increasing number of attack target classes".
- **TeCo** concedes roughly 20 points, AUROC 0.7749 on all-to-all (its Table 20).
- **And PSBD.** Our own H5: "All-to-all is undetectable by PSBD", and the reason is
  mechanistic rather than statistical, since there is no single target class for the
  perturbed prediction to collapse onto.

So a plain all-to-all label map defeats MM-BD, Neural Cleanse, Model X-ray and PSBD,
and materially degrades TeCo. It requires no training-loop control, no surrogate
model, no extra loss term and no clean-accuracy sacrifice. **It is the cheapest
cross-family attack in this entire document, it is decades old, and we have already
published that it breaks us.**

Two honest qualifications, and they are the reason this is a limitation to report
rather than a crisis. All-to-all is a **less useful** attack: the adversary gets an
arbitrary misclassification rather than a chosen target, which is worth much less in
most threat models. And our H5 records that the signal is present with the **sign
reversed**, so something is measurable there even though choosing which tail to flag
needs the poison labels the detector exists to predict.

The recommendation is that the paper state the all-to-all gap **as a shared property
of the whole detector family**, with these citations, rather than as a quirk of PSBD.
Presented that way it is a field-level open problem we identified, which is a better
paper than a limitations bullet.

### 7.3 Two smaller notes worth carrying

**Model X-ray has a published ViT number and we do not.** ViT-B/16 on ImageNet-10,
AUROC 0.879 for the entropy variant and 0.944 for the area variant. It is the only one
of the 3 with a transformer result, it needs nothing but `argmax`, so it ports to ViT
and Swin unmodified. If we claim the transformer setting is under-served, this is the
counterexample and we should name it.

**These 3 produce 1 decision per model, not per sample.** MM-BD scores itself with
detect-and-identify accuracy at a fixed `theta = 0.05`, Model X-ray with model-level
AUROC, Neural Cleanse with a hard anomaly-index threshold of 2. None is comparable to
a per-sample AUROC over a poisoned test set. If any of them appears in a table beside
PSBD, STRIP, SCALE-UP or TeCo, the metric column is not the same quantity and needs an
explicit footnote.

## 8. Prior art: has anyone proposed a universal attack like this?

Deliverable item 5. The answer is that **each half exists, the conjunction does not,
and the 2 papers that come closest do so partly by accident.**

### 8.1 Latent-distribution matching is a mature 6-year lineage. We must cite it.

| work | objective | defeats | cost |
|---|---|---|---|
| **Tan and Shokri**, EuroS&P 2020, arXiv:1905.13409 | `min_theta L(f(x), y) - lambda * L_D( D(H(x)), B(x) )`, a GAN discriminator on the penultimate layer, `lambda` 10 to 50 | Spectral Signatures (poison retained 2.4% to 46.9%), Activation Clustering (ARI 0.998 to 6.31e-4) | not reported. **No test-time detector evaluated** |
| **Wasserstein Backdoor**, Doan, Lao, Li, NeurIPS 2021 | bilevel, with a discriminant sliced-Wasserstein distance between clean and poisoned penultimate distributions, sliced along the **rows of the normalized final-layer weight matrix** | Activation Clustering, Spectral Signatures, Neural Cleanse, **and STRIP** | CA/ASR 0.94/0.99 CIFAR-10, matching WaNet |
| **ML-MMDR**, Xia, Niu, Li, Li, IEEE TDSC 21(3) 2024, arXiv:2111.05077 | multi-level MMD between poisoned and target-class-clean latents at 3 depths, `lambda` up to 0.3 | AC, Spectral Signatures, Subspace Reconstruction (F1 90 to 100% down to 60 to 70%) | **BA 0.907 to 0.902, ASR 0.993 to 0.992.** Essentially free |
| **DEFEAT**, Zhao et al., CVPR 2022 | multi-layer latent constraint via auxiliary linear probes, bilevel with an L2-bounded universal additive trigger | **STRIP**, Neural Cleanse, Fine-Pruning, NAD | ASR 98.76% or above |
| **Adap-Blend / Adap-Patch**, Qi et al., ICLR 2023, arXiv:2205.13613 | no loss term. A poisoning recipe: cover samples, opacity asymmetry, trigger diversity | SS 13.3/10.0, AC 0.0/0.0, **SCAn 0.0/0.0, SPECTRE 6.9/0.0** elimination, against 96 to 100% for baselines | Adap-Blend ASR 76.5 against Blend's 89.0. A real ASR cost |

**Two admissions in this literature are worth quoting at ourselves.** ML-MMDR's own
limitations section says "**Constructing an attack that can bypass defense methods
with different principles requires further research**", which is the exact question
this document asks and confirms nobody has answered it. And Qi et al.'s Appendix D
Table 9 says their adaptive methods "**do not decrease the resistance of their vanilla
version to defenses other than latent-space ones**", with NAD driving Adap-Blend to
9.5% ASR.

### 8.2 Confidence and margin matching is nearly empty, and the one entry is the dangerous one

**LSBA** (arXiv:2202.11203) is covered in section 2.3. It is the closest existing work
to a joint objective, it does both halves, and it is an unpublished preprint with
roughly 4 citations that nobody has followed up. **If we do not cite it, a reviewer
who knows it will conclude we did not look.**

Beyond it:

- **No paper adds an entropy term to a backdoor training loss to defeat STRIP.**
- **No standalone attack paper targets SCALE-UP, TeCo, IBD-PSC or PSBD.** All 5
  confirmed objectives against those defences are **defence-authored**, in the
  papers' own adaptive-attack sections, which are catalogued in section 5.
- "Confidence matching backdoor" and "calibrated backdoor attack" do not exist as
  named methods.

### 8.3 Nobody has attacked a rank or dimensionality statistic. At all.

A dedicated search for "effective rank backdoor attack", "rank regularization
poisoning", "feature diversity backdoor" and "entropy of singular values poisoning"
returned **no such method**. This is the clearest open gap found.

That cuts both ways for us, and the honest reading is uncomfortable. It means our
rank measure is genuinely novel on the attack side as well as the defence side. It
also means its apparent robustness to the adaptive-attack literature is entirely
explained by nobody having tried, and section 3.3 shows what happens the moment
somebody does.

### 8.4 Multi-defence evasion that already works, by other means

The strongest multi-family results in the literature come from **trigger and threat
model design**, not distribution matching, and they should be in our related work
because they are the actual state of the art in evasion:

- **HCB**, Ma et al., ACM CCS 2024, arXiv:2310.00542. The trigger fires only when
  co-occurring with an innocuous class-independent feature. **11 defences, none
  robust**, including STRIP, SCAn, Beatrix and MM-BD. STRIP FAR 100%.
- **BELT**, Qiu et al., IEEE S&P 2024, arXiv:2312.04902. Cover samples with fuzzy
  triggers plus a momentum centre loss. Defeats STRIP, NC, ABS, MNTD, MOTH, SentiNet
  at roughly 0 cost.
- **WaveAttack**, Xia et al., NeurIPS 2024. Loss is only `L_c + ||g(HH)||_inf`, with
  **no matching term of any kind**, and it still evades STRIP, Spectral Signatures,
  ANP, NC and a frequency detector.
- **Model-outsourcing adaptive attacks**, Peng et al., IEEE TIFS 19, 2024. Evades
  Neural Cleanse, ABS and MNTD simultaneously, 2nd prize in the NeurIPS 2022 Trojan
  Detection Challenge evasive-trojans track.

WaveAttack is the uncomfortable one for the framing of this whole document. It
defeats both families with **no matching objective at all**, which means the design
space of "one attack, many defences" is not exhausted by the matching-objective
framing this analysis has used throughout.

### 8.5 The negative results, and what they do and do not license

There is **no paper arguing the 2 evasion goals are formally in tension**, so H-B has
no support in the literature either. What exists is 4 things, and 2 of them are useful
to us.

- **A quantitative trade curve, inside the flagship latent-evasion paper.** Qi et al.
  ICLR 2023 Table 10, Adap-Patch against STRIP at a fixed 10% sacrifice rate:

  | test-time trigger | ASR | STRIP elimination |
  |---|---:|---:|
  | 6c + 6e | 97.5 | 99.0 |
  | 6d + 6m | 86.5 | 70.0 |
  | 6c at opacity 0.5 + 6e at 0.7 | 59.7 | 35.0 |
  | 6d alone | 25.6 | 5.3 |

  Monotone. **Every point of STRIP evasion is bought with ASR**, and the authors say
  so. The same shape appears in BaDExpert's Table 12 and TeCo's Fig. 5. This is the
  best empirical support that exists for section 2.6's proposition.

- **The closest thing to a theorem, and it is on our side.** Xian et al., ICML 2023,
  Theorem 3.1: under kernel smoothing with Gaussian class-conditionals, the backdoor
  risk bound improves as the poisoned data is positioned **farther away**.
  Effectiveness requires distance. That is a published bound in the direction H-C
  wanted, though it is about distance rather than dimension and so does not rescue
  the rank claim.

- **Detection is impossible without assumptions.** Khaddaj et al., ICML 2023,
  arXiv:2307.10163: without structural information about the data distribution,
  backdoors are indistinguishable from naturally occurring features. Pichler et al.,
  AISTATS 2024, arXiv:2402.16926: a no-free-lunch theorem giving error 1/2 for any
  adversary-unaware detector on an infinite alphabet, with a finite-alphabet sample
  requirement that for CIFAR-10 exceeds 10^3697. **Detection must be
  adversary-aware**, which is an argument for this document existing, not against it.

- **Goldwasser, Kim, Vaikuntanathan, Zamir, FOCS 2022, arXiv:2204.06974**, and its
  scope has to be stated exactly or it will be used against us wrongly. They prove
  **model-level undetectability**: the returned model is computationally
  indistinguishable from an honestly trained one. The backdoor is **keyed**,
  activation requires a secret, and the backdoored inputs have **vanishing density**.
  The black-box result is general; the white-box result covers only Random Fourier
  Features and single-hidden-layer random ReLU networks, with the adversary tampering
  only with the random coins, and it is that branch that provably defeats Spectral
  Signatures and SPECTRE. **It says nothing about input-level detection of a poisoned
  sample at inference**, which is our problem, and their own section 7 proposes
  **randomized smoothing** as an evaluation-time defence, which is the same object our
  theory document builds its Cohen-style bridge on. Anyone citing this as "backdoors
  are undetectable, so your defence is pointless" is over-reading it, and we should
  pre-empt that in the paper rather than in the rebuttal.

### 8.6 The one-line answer to "has this been done"

A single loss carrying **both** a latent-matching term and an output-margin term has
never been published. WB gets STRIP for free while aiming at the latent family. LSBA
gets SPECTRE and Spectral Signatures for free while aiming at confidence, on MNIST.
Neither was designed for both, and no attack of any kind targets a dropout-uncertainty
statistic or a rank statistic.

## 9. Which of our claims survive

Blunt, as requested.

### 9.1 The internal inconsistency to resolve first

Two of our own results cannot both be load-bearing in their current form.

- **H28 and `docs/theory-perturbation-consistency.md`:** PSBD, STRIP, SCALE-UP and
  IBD-PSC are 4 estimators of **one quantity**, differing only in `Sigma` and where
  they evaluate `H`. Position variance 1.43x operator variance, Kendall tau 0.700.
- **H41:** probe diversity is defence in depth, because an attacker who constrains 1
  probe leaves the others free.

If the first is true, then an attacker who moves **the estimand** defeats every probe
at once, and the union of `k` probes is a union of `k` estimators of the same scalar.
Diversity buys protection against an attacker who games an **estimator**, which is
what H25's attacker did, and nothing at all against an attacker who moves the
**estimand**, which is what LSBA does.

Both results are correct as measured. The problem is the **scope of the H41 claim**,
which is currently written as though it covers adaptive attackers in general. It
covers exactly one kind, and the paper has to say which. Stage 1 of section 6.2
decides whether the other kind is a real threat or a theoretical one.

This is not a small edit. Defence in depth is currently one of the 5 things the theory
document lists as genuinely new, at number 3.

### 9.2 Survives unchanged

| claim | why it is safe |
|---|---|
| Position dominates operator, 1.43x variance ratio, Kendall tau 0.700 (H28) | A methodological claim about the design space. No adaptive attacker bears on it |
| Gaussian noise matches the best structured masks at 0.950 (H23) | Same, and it is a statement about what the mechanism is not |
| Structured, unit-aligned masking is a dead end (H18, H22, H26, H35) | 4 negative results plus a causal ablation |
| PSBD's published neuron-bias mechanism is refuted, 0.0000 against 0.0100 (H7) | A negative result about a published **explanation**, not a robustness claim |
| The rate rule overshoots on ViT, 12 of 12 (H11) | Tuning, not security |
| The +0.166 matched-shift-ratio gain at 1% on CIFAR-100 (H17) | Holds against the non-adaptive attacks it was measured on, and must be labelled that way. The ledger has withdrawn the +0.258 figure as unmatched, and **`.claude/CLAUDE.md` still quotes +0.258**, which should be fixed before anything is submitted |

### 9.3 Does not survive

| claim | verdict |
|---|---|
| **Effective-rank collapse discriminates backdoored from benign models** | **Retract the security framing.** Section 3.3 moves the ratio to 0.990 with the logits constant to 2.5e-14, and section 8.3 confirms nobody has attacked a rank statistic before, so its apparent robustness is explained entirely by nobody having tried. Keep it as a forensic observation and as the bridge in the theory document. Do not call it a detector |
| **"Adaptive-blend does not defeat our rank measure"** | **We never tested Qi et al.'s attack.** `attacks/adaptive_blend.py` implements cover samples only and says so in its own docstring, omitting the opacity asymmetry and the trigger diversity. Section 4.3 shows cover samples act on a first moment that our mean-centred statistic removes before measuring. The claim is true of what we ran and is not the claim a reader will hear |
| A single rank-ratio threshold is comparable across our 4 datasets | Section 3.7 shows the class-matched baseline sweeping from above 1 at 10 classes to 0.085 at 200. And section 3.4 shows the whole rank result is **CIFAR-100 at 1% only**, 6 backdoored checkpoints, which is 1 of the ledger's 3 required axes |
| Theory prediction 6, rank ratio predicts detection AUROC | The null-space attack severs the link by construction. Until stage 3 runs, state it as a correlation, not a mechanism |

### 9.4 At risk, and the risk is ours to close

| claim | exposure |
|---|---|
| **Operator diversity is defence in depth (H25, H41)** | **The biggest exposure in the project**, for 2 independent reasons. First, section 9.1: our own H28 implies diversity cannot help against a margin attack. Second, H25's attacker regularized against **1** probe, and the obvious extension to all `k` is untested, so the current threat model is an attacker who attacks one probe. Stages 1 and 4 of section 6.2 close both or do not |
| Median-rank reduction fixes the inverted-probe fragility | The theory document's own table shows median failing at 3 of 5 inverted (0.215), and a multi-probe attacker's optimal play is to invert a majority. Section 2.4 shows a margin attack inverts **every** probe at once, which is precisely the majority case the fix does not cover |
| **`A5-low-confidence-backdoor.md`'s central proof** | **Wrong as stated**, and it is a sibling page in this same directory. It shows the attacker gains nothing by moving to the uniform point, which is true, and concludes the low-confidence backdoor cannot beat PSU, which does not follow. The attacker's optimum is the peak between the 2 zeros, and LSBA's rule lands on 97.0% of it. Section 2.4 has the closed form. **One of the 2 pages has to change before either is cited** |
| PSBD and STRIP fuse into something better (H14) | STRIP's **own paper** publishes a free adaptive attack against it, CA 86.61% and ASR 99.95%, and section 8.1 adds 2 more attacks (WB, DEFEAT) that flatten STRIP's entropy as a side effect of aiming elsewhere. The fusion's STRIP half is the cheapest component in our pipeline to remove |
| All-to-all is a PSBD limitation | Section 7.2: it is a **family-wide** limitation covering MM-BD, Neural Cleanse, Model X-ray and TeCo as well. Reporting it as ours alone understates the finding and overstates the weakness |
| Our comparison set is a fair baseline set | Our IBD-PSC is a LayerNorm reinterpretation of a BatchNorm-only method (section 1.1). `papers/reference/README.md` already flags that our SCALE-UP and IBD-PSC were operator ports rather than the published methods. And `psbd/detectors/teco.py` records that both official TeCo implementations compound corruptions across severities, contradicting their own Algorithm 1, so a literal implementation will not reproduce the published numbers |

### 9.5 What would most change these verdicts

In order.

1. **Stage 1 of section 6.2.** If LSBA does not in fact defeat PSBD, then either H28's
   one-quantity claim is weaker than stated or the margin is harder to suppress than
   LSBA's numbers suggest. Either outcome is publishable and both are better than not
   knowing.
2. **Stage 0.** The class-conditional and per-dataset rank ratios, on checkpoints we
   already have. It is 1 feature-extraction pass and it decides whether section 3's
   retraction is total or partial.
3. **Implementing Qi et al.'s full attack**, with the asymmetry and the trigger
   diversity, so that section 4's claim becomes a claim about the published attack.

## 10. What this document is not

Recorded so nobody over-reads it.

- **No attack was implemented or run.** Every attack cost quoted is from the paper
  that reports it, or from H25 for ours.
- **The only original construction is the linear-algebra counterexample in section
  3.3**, and it is a statement about a linear head and a final LayerNorm, not about a
  trained network reached by SGD. Whether the objective is cheap end to end is
  **unmeasured and is the single largest uncertainty here**.
- **The toy calibrations in sections 3.6 and 3.7** use random class means and
  isotropic within-class noise. The sign flip with class count is robust to that
  modelling choice. The specific values are not.
- **The prediction that LSBA defeats PSBD is a prediction**, derived from our own
  framework plus LSBA's reported STRIP numbers. It is not a measurement, and LSBA is
  an unreviewed preprint whose latent-family results are MNIST only.
- Several literature items in section 8 were retrieved rather than read end to end.
  Where a number is load-bearing it should be confirmed against the source before it
  reaches a paper.
