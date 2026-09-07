# A3. Shared factor evasion: one objective that moves every probe at once

**Rank 3.** This is the attack against H41, and H41 is the paper's flagship
security contribution. It targets the assumption the multi probe defence rests
on, which is that $k$ probes impose $k$ constraints the attacker must satisfy
separately.

## The assumption being attacked

The theory document's prediction 4 says an attacker minimising the gap for a
probe with covariance $\Sigma_1$ can only constrain $H$'s projection onto
$\Sigma_1$'s eigenbasis, that $H$ has $d^2$ degrees of freedom, and that a second
probe reads a projection the attacker never constrained. H25 measured exactly
that: probed AUROC 0.322, transfer AUROC 0.887.

The master equation says the argument is incomplete. Write the statistic at probe
$\ell$ as

$$\phi_\ell(x) \;=\; \underbrace{-\tfrac{1}{2}\,p_c\Big[u^\top \Sigma_z^{(\ell)} u - \operatorname{tr}\big(A\,\Sigma_z^{(\ell)}\big)\Big]}_{T_1^{(\ell)}} \;+\; T_2^{(\ell)} .$$

$\Sigma_z^{(\ell)} = J_\ell \Sigma_\ell J_\ell^\top$ is probe specific, but
$p_c$, $u = e_c - p$ and $A = \operatorname{diag}(p) - pp^\top$ are **the same
objects in every probe's expression**. They are properties of the model's output,
not of the probe. So $T_1$ carries a common factor across all $k$ probes, and one
scalar objective on that factor moves all $k$ statistics in the same direction
simultaneously.

That is the majority attack the median rank rule was introduced to survive.
`multi_probe_score` with `reduction="median"` fails once the attacker controls
$\lceil k/2 \rceil$ probes, and its own table records 2 inverted of 5 at 0.751
and 3 of 5 at 0.215. The defence's threat model assumes the attacker has to buy
each probe separately. If a single term buys a shared component of all of them,
the majority is not $k/2$ training objectives, it is 1.

## Objective

    original form
        L = L_CE(f(x), y)
            + lambda_shared * ReLU( median_{x in C} phi_tilde(x)
                                    - mean_{x in P} phi_tilde(x) )
        with phi_tilde(x) = p_c(x) * ( p_c(x) - ||p(x)||_2^2 )

    descriptive form
        loss = cross_entropy
               + shared_weight * a hinge on the closed-form logit-layer
                 statistic, computed from the softmax vector of the ONE forward
                 pass cross-entropy already needs, with the clean side read at
                 its MEDIAN rather than its mean

| symbol | meaning |
|---|---|
| $\tilde\phi(x) = p_c(p_c - \|p\|_2^2)$ | the exact logit layer statistic with $\Sigma = I$, from the theory document's objection 3 |
| $P$, $C$ | poisoned and clean samples in the batch |
| $\lambda_{\text{shared}}$ | weight on the hinge |
| median over $C$ | the clean **median**, because the detector's threshold is a clean quantile and not a clean mean |

Two details are deliberate and each is a correction to what we published.

**The clean side is read at the median.** H25's `evasion_penalty` uses
`psu[~poisoned].mean()`. The detector's decision rule is
`psu(x) < quantile(validation_psu, 0.25)`. An attacker who matches the mean when
the decision reads a quantile is optimising the wrong functional, and the clean
PSU distribution has a long left tail that drags the mean below the median, so
the mean matching attacker systematically **undershoots**. Measured in
[A1](A1-operating-point-and-threshold.md): parking the poisoned mass at the clean
median alone takes TPR at $q = 0.25$ from 0.999 to 0.080 on
`vit_cifar100_badnet_a2o_0_01`.

**The hinge is kept, and it is kept because it inverts.** The docstring in
`psbd/evasion.py` argues the hinge avoids "overshooting into a separation the
defence could read from the other side". The measurement says the opposite
happened: the probed position goes to AUROC 0.023, which is not neutralisation,
it is a near perfect inversion. The mechanism is that the hinge is evaluated on a
noisy $k = 3$ estimate, so at true parity the estimated gap is positive about
half the time and the penalty is active about half the time, while it is never
active in the reverse direction. That is a one sided random walk and it drifts
upward without limit. **The hinge does not prevent overshoot, it manufactures
it.** An attacker who wants inversion should therefore keep the hinge and use a
small $k$, and an attacker who wants neutralisation should use a symmetric
penalty. Both are useful and the paper should say so, because it means the
inversion that broke the min rank union was an accident of our estimator and an
attacker can now produce it on purpose.

## Which detectors it defeats and which it does not

| detector | defeated | why |
|---|---|---|
| PSBD at `final_norm_out` or any post nonlinearity probe | **yes** | $\tilde\phi$ is the exact statistic there, up to the fixed $WW^\top$ shaping |
| PSBD at block output norms | **partly** | roughly the 0.19 explained share |
| PSBD at input side probes | **partly, and this is the open question** | $T_1$ moves, $T_2$ does not. Whether moving $T_1$ alone flips the ordering depends on the ratio $T_1/T_2$, which nobody has measured |
| multi probe min rank | **yes, if it inverts any single probe** | `ranks.min(dim=0)` takes the most extreme evidence, so 1 inversion is enough. Measured: 1 of 3 takes 0.969 to 0.621 |
| multi probe **median** rank | **yes, if the shared factor flips a majority at once** | this is the claim to test. The median needs $\lceil k/2 \rceil$ inversions and the shared factor is the only mechanism that could produce them from 1 term |
| the confidence null | **yes** | $\tilde\phi$ is a function of $p$ and the null is a function of $p$ |
| STRIP, SCALE-UP, TeCo | **no** | none of them reads $p$ at the deployed input |
| IBD-PSC | **no** | reads $p$ under a modified model |

## The measurement that decides it, and it costs nothing

Before training anything, the ratio the attack depends on can be estimated from
the caches on disk. For each probe $\ell$, regress $\phi_\ell$ on
$(p_c, \mathcal{H}(p), \|p\|_2^2)$ over the clean validation split. The explained
share is the fraction of $\phi_\ell$ the shared factor can move. Measured on ViT
CIFAR-100 at the sigma 0.6 matched rate:

| probe pool member (H41's pool) | explained share | population |
|---|---:|---|
| `token_mask` at `before_attention_norm` | 0.076 | mean over badnet_a2o, blend, lf, bpp at 1 percent |
| `dropout` at `before_attention_norm` | 0.100 | badnet_a2o at 1 percent only |
| `gaussian` at `before_mlp` | 0.201 | badnet_a2o at 1 percent only |
| `gain_scale` at `mlp_norm_out` | 0.188 | mean over the same 4 |

The 2 single checkpoint rows should be filled in over the same 4 attacks before
the table goes in a paper. The command is in
`docs/attack-design/measure_confidence_share.py` and it is CPU only.

**H41's pool is safer than it looks, and for a reason nobody chose.** All 4
members have a low explained share, so the shared factor governs at most about a
fifth of any of them. But this is luck, not design. Adding a `final_norm_out`
probe to the pool "for diversity" would add a member with an explained share of
0.98, and under the min rank rule a single attacker owned member is enough. The
right rule is:

> **Probe pool diversity must be measured in $R^2$ on $p$, not in operator
> name.** Two probes with a high explained share are not independent, however
> different their operators look, because they share the factor an attacker gets
> for the price of 1 forward pass.

That is a concrete change to the H41 protocol and it is defender legal, since
$R^2$ is measured on clean validation data alone.

## Predicted cost

- **Clean accuracy: 1 to 3 points**, less than H25's measured 4.8, because the
  term is 1 scalar per sample from an existing forward pass rather than a
  penalty computed through $k$ retained activation graphs.
- **ASR: within 2 points.** $\tilde\phi$ is maximised at an interior confidence
  and the arg max is unaffected across most of the range.
- **Wallclock: 1x**, against H25's 4x. This matters for the threat model, because
  an attack that costs 4x training is a real deterrent and one that costs nothing
  is not.

## Why it might fail, and the honest bound

The theory gives a bound on how far this can go. $T_2$ is untouched by any
objective on $p$, so at an input side probe the attacker can move at most the
$T_1$ share. If $T_1$ and $T_2$ have the same sign and comparable magnitude,
flipping the ordering requires driving $T_1$ far past 0, which means driving the
poisoned population's $p_c(p_c - \|p\|^2)$ far above the clean median, which by
the single peakedness of that function means parking poisoned samples exactly at
the curvature peak near $p_c \approx 0.79$ in the binary case. That is a narrow
target and it is unstable under any change to the data distribution. So the
predicted outcome is **neutralisation of the input side probes toward 0.6 to 0.8
and inversion only of the head adjacent ones**, which under a median rank rule
with H41's 4 member pool is not a majority.

If that is what is measured, it is a strong positive result for the paper: the
median rank union survives the cheapest possible simultaneous attack, and it
survives it for a reason the theory names in advance.

## Falsifiable prediction and the experiment

**Prediction.** Train `vit`, `cifar100`, `badnet_a2o` at 0.01 for 15 epochs with
$\lambda_{\text{shared}} = 1.0$. Sweep with H41's full 4 probe pool. Then:

| quantity | H25 evasive (measured) | prediction under A3 | refutes if |
|---|---:|---:|---|
| number of the 4 pool members with AUROC below 0.5 | 1 | **2 or more** | still 1 or 0, which would mean the shared factor does not carry across probes |
| multi probe AUROC, `reduction="min"` | 0.927 mean | **below 0.70** | above 0.85 |
| multi probe AUROC, `reduction="median"` | not yet measured on evasive | **0.75 to 0.90** | below 0.60, which would refute the median rank defence outright |
| `confidence` null AUROC | 0.687 | below 0.55 | above 0.65, meaning the term did not take |
| clean accuracy tax | 4.8 points | **below 3 points** | above 5 points, which would make it no cheaper than H25 |
| wallclock relative to plain training | 4x | **below 1.3x** | above 2x |

The decisive cell is the third row. **If the median rank union holds above 0.75
against an attacker who buys a shared component of every probe for the price of
1 forward pass, H41 is a real defence and the paper can say so with a named
adversary.** If it falls below 0.60, H41 has to be downgraded from "defeats the
adaptive attacker" to "defeats a probe specific adaptive attacker", and the
difference between those 2 sentences is the difference between acceptance and
rejection at a security venue.

**Concrete run.**

```bash
python cli/train_backdoor.py \
  --dataset cifar100 --attack badnet_a2o --poison-rate 0.01 \
  --architecture vit --epochs 15 \
  --evade-shared --evade-weight 1.0 --evade-reference median \
  --output checkpoints/vit_cifar100_badnet_a2o_0_01_shared

# H41's pool, 4 probes, 1 sweep each
python cli/sweep.py --checkpoint-folder vit_cifar100_badnet_a2o_0_01_shared \
  --position-config before_attention_norm --perturbation token_mask
python cli/sweep.py --checkpoint-folder vit_cifar100_badnet_a2o_0_01_shared \
  --position-config before_attention_norm --perturbation dropout
python cli/sweep.py --checkpoint-folder vit_cifar100_badnet_a2o_0_01_shared \
  --position-config before_mlp --perturbation gaussian
python cli/sweep.py --checkpoint-folder vit_cifar100_badnet_a2o_0_01_shared \
  --position-config mlp_norm_out --perturbation gain_scale

PYTHONPATH=. .venv/bin/python experiments/multi_probe/analyze.py \
  --checkpoint-folder vit_cifar100_badnet_a2o_0_01_shared
```

Budget: 1 training job at about 1x, 4 sweeps. Repeat on `blend` and on Swin
before claiming either outcome, because H41's per architecture table already
shows ViT at 0.888 and Swin at 0.965 and the 2 should not be assumed to behave
alike.

**A second experiment that needs no new checkpoint.** Recompute the H41 table on
the 126 existing `*_evade_l1` checkpoints with `reduction="median"`. H41 reports
the median rank numbers on **synthetic** probes with a known answer, not on the
real evasive checkpoints. Publishing a synthetic table as the evidence for the
defence that answers the adaptive attacker is the second easiest criticism to
make of the current draft, and closing it costs 1 CPU run of
`experiments/multi_probe/analyze.py`.
