> **CORRECTION, see audit finding A21.** The conclusion below that a low
> confidence backdoor provably cannot beat PSU is **WRONG**. It evaluates the
> uniform point, where the statistic is 0 and the sample does land on the flagged
> side. The attacker's optimum is the **peak between the 2 zeros**. At K = 100 the
> peak is at p_c = 0.668, an attacker capped at 0.60 reaches 97.0 percent of the
> attainable maximum, and a clean sample at 0.95 reaches only 30.6 percent. The
> backdoor therefore carries 3.17 times the PSU of a confident clean sample, and
> since PSBD flags low PSU as poisoned the detector **inverts** rather than merely
> failing. Do not cite the analysis below without reading A21 first.

# A5. The low confidence backdoor, and the proof that it cannot beat PSU

**Rank 5.** The prompt asks whether the single peakedness of the closed form
gives the attacker a second way to look benign, by placing poisoned samples at
the low confidence end rather than the saturated one. The answer is **no for
PSBD, proved below, and yes for every other detector in our comparison table**.
That asymmetry is a result and it should be in the paper.

## The shape of the statistic

At the logit layer with an isotropic perturbation,

$$\phi(x) \;=\; \sigma^2\,p_c\big(p_c - \|p\|_2^2\big) \;\ge\; 0 ,$$

non negative because $\|p\|_2^2 \le (\max_i p_i)(\sum_i p_i) = p_c$. In the 2
class case with $p_c = p \ge 1/2$ this is $\sigma^2\,p(2p-1)(1-p)$, which is 0 at
$p = 1/2$, 0 at $p = 1$, and single peaked with its maximum at
$p = (3 + \sqrt{3})/6 \approx 0.7887$. For $K$ classes the same shape holds with
the lower zero at the uniform point $p_c = 1/K$.

So there are 2 confidence regimes with near zero $\phi$:

- **the saturated regime**, $p_c$ near 1, which is where a normal backdoor sits
- **the boundary regime**, $p_c$ near the uniform point, which is where genuinely
  ambiguous clean samples sit

## The proof that the boundary regime does not evade PSBD

The detection rule is `psu(x) < threshold`, with the threshold a **low** quantile
of clean validation PSU. Low $\phi$ is the positive, poisoned evidence.

An attacker who moves poisoned samples from the saturated regime to the boundary
regime moves them from one zero of $\phi$ to the other zero of $\phi$. In both
regimes $\phi \to 0$, which is **below** any low quantile of a clean population
whose mass sits at the interior peak. So the samples are flagged in both regimes,
and the move buys the attacker nothing against this rule.

Formally: let $q_\tau$ be the clean $\tau$ quantile of $\phi$, and let clean mass
concentrate away from both zeros so that $q_\tau > 0$. Then for any poisoned
sample with $\phi < q_\tau$ the rule fires, and $\phi \to 0$ from either side
satisfies $\phi < q_\tau$. **Sign inversion of the statistic is not available,
because $\phi$ has a fixed sign to second order.** The 2 ways to look benign that
the prompt asks about are not 2 ways to look benign. They are 2 ways to look
identically guilty.

The only way to invert the detector is to move the **clean** population to a zero
of $\phi$, not the poisoned one. That is a different attack and it is
[A1](A1-operating-point-and-threshold.md).

## What the boundary regime does buy

Three things, and 2 of them hurt us.

### 1. It defeats the confidence family outright

| detector | statistic | effect of the boundary regime |
|---|---|---|
| `confidence` | $\max_c p_c$, flagged when high | **defeated.** A poisoned sample at $p_c = 0.3$ is less confident than the median clean sample |
| `ibd_psc` | probability an amplified model gives the unamplified prediction, Eq. (4), flagged when high | **defeated.** Amplifying $\gamma$ and $\beta$ breaks a thin margin as easily as it breaks a clean one, which is the mechanism the paper relies on |
| `scale_up` | fraction of amplified copies keeping the label, Eq. (2), flagged when high | **defeated.** A thin margin does not survive pixel amplification |
| `strip` | entropy under superimposition, flagged when low | **defeated.** A thin margin gives high entropy under an overlay, which is the clean signature |
| `teco` | spread of corruption hardness thresholds, flagged when high | **partly defeated.** A uniformly thin margin breaks at a low severity for every corruption type, so the spread is small, which is the clean signature |
| PSBD | $\phi$, flagged when low | **not defeated**, by the proof above |

**PSBD is the only detector in our own comparison table that survives it.** That
is a genuinely strong sentence about the method and it is available for free.
It should be stated as: the perturbation consistency family splits into methods
that read a margin's *size* and methods that read a margin's *curvature*, and
only the second kind is two sided in confidence.

### 2. It destroys TPR at a low FPR by camouflage

This is the half that hurts. The theory document's objection 3 already predicts
"inversions concentrated on low confidence clean samples". Those are the clean
samples that sit near the lower zero of $\phi$, and they are the clean false
positives that already dominate the left tail of the clean PSU distribution.

An attacker who parks poisoned samples in the boundary regime parks them
**exactly where the clean false positive mass already lives**. AUROC barely
moves, because the ranking is still roughly right. TPR at a 1 or 5 percent FPR
budget collapses, because any threshold low enough to sit under the poisoned mass
is also under a large block of clean mass.

This is the same failure mode as A1 reached by a different route, and the
combination of the 2 is the strongest cheap attack available. The measured
baseline gap it exploits is already visible with no attacker: on
`vit_cifar100_badnet_a2o_0_01` at `before_mlp` with gaussian, AUROC 0.892 and TPR
at 5 percent FPR **0.01**.

### 3. It is nearly free, and needs no penalty at all

The cheap version is not a loss term. It is a training schedule: stop the
poisoned samples' cross entropy early, or apply label smoothing to the poisoned
subset only, or train the trigger at a lower blend strength. Any of them lands
triggered inputs at a mid range target probability. ASR is an arg max question,
so on CIFAR-100 a triggered sample at $p_{\text{target}} = 0.3$ against a runner
up at 0.05 is a success.

## Objective, for the version that is optimised rather than scheduled

    original form
        L = L_CE(f(x), y)
            + lambda_low * mean_{x in P} ( p_c(x) - tau )^2

    descriptive form
        loss = cross_entropy
               + low_confidence_weight * squared distance between each poisoned
                 sample's predicted-class probability and a target confidence tau
                 chosen in the clean population's lower quartile

| symbol | meaning |
|---|---|
| $\tau$ | target confidence, set to the clean validation 20th percentile of $p_c$ |
| $P$ | poisoned samples in the batch |
| $\lambda_{\text{low}}$ | weight on the term |

## Predicted cost

- **Clean accuracy: under 1 point.** The term touches poisoned samples only, and
  at a 1 percent poison rate that is 1 percent of the gradient.
- **ASR: 3 to 12 points, and rising as the number of classes falls.** On
  CIFAR-100 a target probability of 0.3 is a safe arg max. On CIFAR-10 and GTSRB
  the runner up is much closer and the same $\tau$ loses real arg max mass. This
  predicts a **dataset dependent ASR cost**, which is a testable signature that
  distinguishes A5 from every other attack here.
- **Robustness: badly damaged.** A thin margin is a thin margin. The trigger will
  not survive JPEG, resize or blur. See the certified radius argument in
  [A4](A4-curvature-matching.md).

## The cheapest defence that catches it

**Report TPR at 0.01 and 0.05 FPR, and use a 2 sided rule only where the second
side is justified.** H15 retired 2 sided reporting for a good reason, that it can
flatter any result and needs oracle access to labels. A5 is the case where the
one sidedness is a genuine weakness rather than a reporting convenience, and the
honest response is not to reinstate 2 sided AUROC but to **fuse PSBD with a
confidence reading**, since the 2 flag opposite ends of the same curve.

Concretely, score with the pair $\big(\phi(x),\, p_c(x)\big)$ and flag when
$\phi$ is low **and** $p_c$ is not near the clean lower quartile. This is
defender legal, it is 1 extra number already in the cache, and it is exactly the
combination A5 cannot satisfy: the attacker cannot be simultaneously at the
saturated zero and the boundary zero.

## Falsifiable prediction and the experiment

**Prediction.** Train `vit` with $\lambda_{\text{low}} = 1.0$ and $\tau$ at the
clean 20th percentile, on `badnet_a2o` at 0.01, on **both** cifar100 and cifar10,
15 epochs.

| quantity | cifar100 prediction | cifar10 prediction | refutes if |
|---|---:|---:|---|
| ASR | drops 3 to 8 points | drops **more than 12 points** | the 2 datasets lose the same amount, refuting the class count mechanism |
| PSBD AUROC at `before_attention_norm` + token_mask | **above 0.85** | above 0.85 | falls below 0.65, which would refute the proof above and is the most valuable possible outcome because it would mean the second order account is wrong at these perturbation scales |
| PSBD TPR at $q$ = 0.05 | **below 0.25** | below 0.25 | stays above 0.5 |
| `confidence` null AUROC | **below 0.45** | below 0.45 | stays above 0.6, meaning the term did not take |
| `ibd_psc` AUROC | **below 0.55** | below 0.55 | stays above 0.7 |
| `scale_up` AUROC | **below 0.55** | below 0.55 | stays above 0.7 |
| `strip` AUROC | **below 0.55** | below 0.55 | stays above 0.7 |
| fused $(\phi, p_c)$ AUROC | **above 0.85** | above 0.85 | falls below 0.7, refuting the fusion defence |

The row that matters most is the second one. **A5 is the experiment that
falsifies the whole second order framing if it comes out wrong**, because the
framing makes a hard, signed prediction there and no fitted parameter can rescue
it. That is the strongest kind of experiment to have in a paper and it is worth
running for that reason alone, independently of whether the attack is dangerous.

**Concrete run.**

```bash
for ds in cifar100 cifar10; do
  python cli/train_backdoor.py \
    --dataset $ds --attack badnet_a2o --poison-rate 0.01 \
    --architecture vit --epochs 15 \
    --low-confidence-target 0.20 --low-confidence-weight 1.0 \
    --output checkpoints/vit_${ds}_badnet_a2o_0_01_lowconf
done

python cli/sweep.py --checkpoint-folder vit_cifar100_badnet_a2o_0_01_lowconf \
  vit_cifar10_badnet_a2o_0_01_lowconf \
  --position-config before_attention_norm --perturbation token_mask

python cli/compare_detectors.py --fpr 0.01 0.05
```

Budget: 2 training jobs at 1x, 1 sweep, 1 CPU comparison. The comparison table is
the deliverable, because A5's value to the paper is the row where PSBD survives
and the other 5 detectors do not.
