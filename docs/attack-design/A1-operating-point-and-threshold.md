# A1. Quantile shadowing: attack the threshold, not the statistic

**Rank 1.** Cheapest attack in this document, close to zero cost in ASR and clean
accuracy, and our headline metric cannot see it.

## The observation it is built on

The detection rule is `psu(x) < threshold` with

```
threshold = numpy.quantile(clean_validation_psu, q),  q = 0.25 headline
```

(`psbd.decision.threshold_at_quantile`). The quantile is read from the clean
validation PSU distribution, and that distribution is **a property of the
attacker's model**, not of the defender's data. The defender chooses the images.
The attacker chooses the function that maps them to PSU.

So the attacker does not have to move the poisoned samples at all. It is enough
to grow a left tail on the clean side. Every unit the threshold drops is a unit
of TPR the defender loses, and the achieved FPR stays exactly on target by
construction, so nothing in the defender's own diagnostics reports a problem.

The gap this exploits already exists with no attacker present. On
`vit_cifar100_badnet_a2o_0_01` at `before_mlp` with gaussian, sigma matched to
0.6: **AUROC 0.892, TPR at 5 percent FPR 0.01.** At `attention_norm_out` with
gain_scale: AUROC 0.804, TPR at 5 percent FPR 0.00. A statistic can have a good
ranking and a dead operating point, and the paper currently reports the ranking.

## Objective

The attacker needs a rule that selects part of the clean population by a property
of the **input**, so that it generalises from its own training data to the
defender's held out validation images, which it never sees. Any deterministic
input space predicate works. Write it as $\pi(x) \in \{0,1\}$.

    original form
        L = L_CE(f(x), y)
            + lambda_sharp * mean_{x : pi(x)=1, x clean} [ -log p_{y}(x) ]^{gamma}
            + lambda_mid   * | mean_{x in P} phi_hat(x) - median_{x in C} phi_hat(x) |

    descriptive form
        loss = cross_entropy
               + sharpen_weight * a focal-style term that drives the selected
                 clean slice to near-certainty, which sends its PSU toward 0
               + centring_weight * a penalty keeping poisoned PSU at the CLEAN
                 MEDIAN rather than at the clean mean

| symbol | meaning |
|---|---|
| $\pi(x)$ | a deterministic input space predicate, for example "mean luminance of the top left 8 by 8 patch exceeds the dataset median" |
| $\lambda_{\text{sharp}}$ | weight on the confidence sharpening term |
| $\gamma$ | focal style exponent, $\gamma > 1$ concentrates the term on already correct samples |
| $\hat\phi$ | any cheap PSU surrogate, the closed form $p_c(p_c - \|p\|^2)$ is enough |
| $P$, $C$ | poisoned and clean samples in the batch |
| $\lambda_{\text{mid}}$ | weight on the centring term |

The two halves do different jobs. The sharpening term manufactures a low PSU mode
in the clean population, which drags the $q$ quantile down. The centring term
parks the poisoned mass at the clean **median**, which is far above the sunk
threshold. Neither term needs a probe, a position, or an operator.

The predicate $\pi$ is the mechanism that makes this generalise. A random subset
of training indices would not, because the learned behaviour would not attach to
anything the defender's validation images share. A predicate computable from
pixels does, because the defender's validation images satisfy it at the same rate
the training images do.

## Which detectors it defeats and which it does not

| detector | defeated | why |
|---|---|---|
| PSBD at any position, at any quantile | **yes, at the operating point** | the threshold is a quantile of a distribution the attacker shaped. TPR at $q$ collapses while achieved FPR stays at $q$. |
| PSBD, AUROC | **no** | AUROC is threshold free. It falls only by whatever the centring term costs, which is small. This asymmetry is the whole point. |
| multi probe min rank (H41) | **yes** | `to_rank` ranks against the same corrupted clean validation reference, and `multi_probe_detection` takes the calibrated quantile of the combined validation score. Both inherit the corruption. |
| multi probe median rank | **yes** | same reason. The median fixes the inverted probe problem, not the corrupted reference problem. They are different failures with the same symptom. |
| STRIP | **yes, partly** | its rule is also a percentile of clean entropy at a target false rejection rate. Sharpening confidence lowers clean entropy on the selected slice and moves the same threshold. |
| SCALE-UP data free | **no** | Eq. (2) is a label consistency count with a fixed threshold $T$ and no clean quantile. |
| SCALE-UP data limited | **yes, partly** | Eq. (3) standardises against per class clean mean and standard deviation, so inflating the clean spread inflates $\sigma_i$ and compresses NSPC. |
| IBD-PSC | **yes, partly** | Algorithm 1 picks $k$ from the clean top-1 error rate crossing $\xi$. Sharpening clean confidence raises the amplification needed to break the prediction, which pushes $k$ up and weakens the ensemble. |
| TeCo | **no** | its statistic is a spread over corruption types and uses no clean reference at all. |
| the confidence null | **no, and it gets better** | the sharpened slice is clean data at high confidence, which the null already mis-ranks. This is a real tell, see below. |

## Predicted cost

- **ASR: unchanged.** No term touches the trigger.
- **Clean accuracy: within 1 point.** The sharpening term acts on samples the
  model already classifies correctly (that is what $\gamma > 1$ buys). It changes
  calibration, not decisions.
- **Detectable side effect:** expected calibration error on clean validation
  rises sharply, and the clean validation PSU histogram becomes visibly bimodal.
  This is the attack's only signature and it is what the defence below reads.

## The cheapest defence that catches it

Three lines of CPU code, in order of cost.

1. **Report TPR at 0.01 and 0.05 FPR beside every AUROC, everywhere.** This does
   not stop the attack, it stops the attack from being invisible, and it removes
   the easiest criticism of the current draft.
2. **Test the clean validation PSU distribution for bimodality before trusting
   the threshold.** A dip test, or simply the ratio of the 0.25 quantile to the
   median. On a healthy model that ratio sits in a narrow band. Under quantile
   shadowing it collapses. This is defender legal because it reads clean
   validation only.
3. **Threshold on a robust location and scale rather than a raw quantile.** Set
   `threshold = median - kappa * MAD` with $\kappa$ chosen once on benign models.
   A left tail then has to move the median, which is 100 times more expensive for
   the attacker than moving the 25th percentile.

The honest limit of defence 3 is that it decouples the threshold from the FPR
budget, which is the property `psbd.decision` was built around ("the quantile IS
the false positive budget"). The paper should report both and say why.

## It is already confirmed in simulation, on our own cached scores

`docs/attack-design/simulate_quantile_shadowing.py` applies both halves of the
attack to the cached PSU of a real checkpoint, with no training and no GPU. The
sharpened slice is selected by baseline confidence, standing in for a
deterministic input space predicate, and the same fraction is applied to
validation and to clean test, which is what an input side predicate guarantees
and a random training subset would not.

`vit_cifar100_badnet_a2o_0_01`, `before_attention_norm` with token_mask at rate
0.4:

| variant | AUROC | TPR at $q$=0.25 | FPR | TPR at $q$=0.05 | FPR |
|---|---:|---:|---:|---:|---:|
| baseline, no attack | 0.960 | 0.999 | 0.238 | 0.761 | 0.047 |
| centring only, which is H25's attack aimed at the median | 0.378 | 0.080 | 0.238 | 0.000 | 0.047 |
| **sharpen 2 percent of clean** | 0.945 | 0.999 | 0.240 | **0.618** | 0.050 |
| **sharpen 5 percent of clean** | **0.918** | 0.999 | 0.242 | **0.000** | 0.049 |
| sharpen 8 percent | 0.890 | 0.999 | 0.241 | 0.000 | 0.055 |
| sharpen 20 percent | 0.780 | 0.947 | 0.245 | 0.000 | 0.050 |
| sharpen 30 percent | 0.686 | 0.000 | 0.253 | 0.000 | 0.049 |

`vit_cifar100_blend_0_01`, same probe, reproduces the same shape: baseline 0.945
and TPR at 0.05 of 0.717, against 0.903 and **0.000** after sharpening 5 percent.

Read the 5 percent row carefully, because it is the finding.

- **AUROC falls 0.042.** A reviewer looking at our headline table sees nothing.
- **TPR at a 5 percent FPR budget goes from 0.761 to 0.000.** The detector is
  dead at the only operating point a deployment would use.
- **Achieved FPR stays on target at 0.049.** Every self check in
  `psbd.decision.detection_report` reports a healthy calibration.
- **Not one poisoned sample was moved and the trigger was not touched.**

The structure is exactly what the quantile rule implies: to sink the $q$
quantile the attacker needs to sharpen a fraction of order $q$. So the paper's
headline $q = 0.25$ needs 25 percent of clean data sharpened and costs the
attacker 0.27 AUROC, while the deployable $q = 0.05$ needs 5 percent and costs
0.04. **We report the one operating point this attack cannot reach cheaply and
omit the two it kills for free.**

The centring row is worth its own sentence. Aiming the H25 style penalty at the
clean **median** rather than the clean **mean** takes TPR at $q = 0.25$ to 0.080
on its own. H25's hinge targets the mean, which is dragged down by the clean
left tail, so the attacker undershoots. Changing 1 word in the penalty is a
strictly better attack than the one we published.

## Falsifiable prediction and the experiment

The simulation establishes that the score geometry is reachable. What it cannot
establish is that a trainable model reaches it at the stated ASR and clean
accuracy cost. That is the GPU experiment.

**Prediction.** Training with $\lambda_{\text{sharp}} = 1$, $\gamma = 2$ and
$\pi$ = "mean luminance of the top left 8 by 8 patch above the dataset median",
on `vit`, `cifar100`, `badnet_a2o`, poison rate 0.01, 15 epochs, with the
sharpening term applied to correctly classified clean samples only, gives at
`before_attention_norm` with `token_mask` at the sigma 0.6 matched rate:

| quantity | baseline (measured) | prediction under A1 | refutes if |
|---|---:|---:|---|
| AUROC | 0.960 | **0.88 to 0.94** | below 0.75, which would mean the attack works through the statistic rather than the threshold |
| TPR at $q = 0.25$ | 0.999 | above 0.90 | below 0.5, which would mean the sharpened fraction overshot |
| TPR at $q = 0.05$ | 0.761 | **below 0.10** | above 0.40 |
| achieved FPR at $q = 0.05$ | 0.047 | 0.05 plus or minus 0.01 | drifts off target, meaning the threshold is not the mechanism |
| ASR | baseline value | within 0.01 | drops more than 0.05 |
| clean accuracy | baseline value | within 1.0 point | worse than 2 points, which would make it no cheaper than H25 |
| clean validation expected calibration error | baseline value | **at least 3x baseline** | unchanged, meaning the sharpening term did not take |

The decisive cell is the pair (AUROC above 0.88, TPR at 0.05 below 0.10). No
other attack in this document produces that pair, so the experiment identifies
the mechanism rather than merely confirming an effect.

**Concrete run.**

```bash
# 1 GPU job, 15 epochs, the same budget as any H25 evasive run and about 4x
# cheaper, because there is no multi-pass probe retained in the graph.
python cli/train_backdoor.py \
  --dataset cifar100 --attack badnet_a2o --poison-rate 0.01 \
  --architecture vit --epochs 15 \
  --shadow-quantile --shadow-weight 1.0 --shadow-gamma 2.0 \
  --shadow-predicate topleft_luminance --shadow-share 0.08 \
  --output checkpoints/vit_cifar100_badnet_a2o_0_01_shadow

# 1 sweep, the deployment probe plus 2 transfer probes, so the claim that this
# attack is probe-independent is tested rather than assumed.
python cli/sweep.py \
  --checkpoint-folder vit_cifar100_badnet_a2o_0_01_shadow \
  --position-config before_attention_norm before_mlp mlp_norm_out \
  --perturbation token_mask

python cli/analyze.py --checkpoint-folder vit_cifar100_badnet_a2o_0_01_shadow
```

The `--shadow-*` flags do not exist yet. They are 1 loss term and 1 predicate
function, both belonging where `--evade-psbd` already lives in
`cli/train_backdoor.py`.

**Second prediction, free.** Because the attack never touches the poisoned
population, its effect must be **identical at every probe position and for every
operator**. That is the sharpest available test that the threshold and not the
statistic is the mechanism: the drop in TPR at $q = 0.05$ should be within
measurement noise across `before_attention_norm`, `before_mlp` and
`mlp_norm_out`, while H25's evasion is famously probe specific. If the drop is
probe dependent, A1 is not what happened.
