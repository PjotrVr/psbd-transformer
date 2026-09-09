# Do two PSBD probes catch different backdoors, or the same one twice?

## Question

Every result in this project reads ONE placement at a time. If two placements fail on
different attacks, combining them should cover both, and the deployment configuration would
be a set rather than a single config. That is the question here.

## Why it is nearly free

`results/<cell>/psbd/<placement>/rate_*.pt` stores `per_pass_probs` of shape `(k, N)`, the
baseline-predicted class's probability on each perturbed pass. That is exactly PSU's input,
so combining placements at the SCORE level needs no GPU: every number below comes from
tensors already on disk.

That cheapness is the trap. C(18,2) + C(18,3) is 969 combinations and reporting the best of
them manufactures a winner out of noise. So:

- **24 combinations are pre-registered** in `configs/psbd_basis.json`, each carrying a
  mechanism claim written before any fused AUROC was read.
- **4 of them are negative controls** that should gain little if the claimed mechanism is
  what carries the effect.
- **One combination is selected on CIFAR-10 and GTSRB and reported on CIFAR-100 and Tiny**,
  which never enter the selection.
- The comparison is against a **single fixed probe**, `before_attention_norm_token_mask`,
  not against each combination's own best member. A defender cannot know which member is
  best without labels, so "beats its best member" is an oracle question. "Beats the config I
  would have deployed anyway" is the deployable one.

Two combination rules are reported, both label-free:

| rule | what it asks | weakness |
|---|---|---|
| `min_rank` | is ANY probe suspicious of this sample | one inverted probe drags the union down |
| `mean_rank` | what does the average probe think | dilutes a single strong member |

Both are needed. `before_attention_norm_gaussian` reads AUROC **0.191** on
`vit_cifar100_badnet_a2o_0_01`, an inverted probe, and it takes `c3_ban_cm_gauss` down to
0.601 under `min_rank`. Reporting only the flattering rule is how a fusion result stops
meaning anything.

Every probe is read at its own rate matched to a clean-validation shift ratio, never at a
shared rate, and the achieved shift is recorded next to every number.

## Result, preliminary: fusion does not beat the single best placement

Run on the 56 cells that clear the 0.85 ASR bar, of which 39 carry the needed placements.

**Not one of the 20 computable combinations beats its own best member.** Mean gain over best
member ranges from -0.009 to -0.130, and the family ordering is in the predicted direction
(depth span -0.023, axis -0.023, family span -0.025, requested -0.026, operator -0.035,
negative controls -0.039), so the controls are worst, as designed. But every one loses.

Against the fixed probe, the pre-registered "requested" family leads:

| combination | family | n | d AUROC | d TPR@10%FPR | wins |
|---|---|---|---|---|---|
| `c5_ban_pre_two_bands` | C5 requested | 39 | **+0.018** | +0.047 | 24/39 |
| `c5_ban_pre_9_12` | C5 requested | 39 | +0.014 | +0.028 | 24/39 |
| `c5_ban_pre_5_8` | C5 requested | 39 | +0.009 | +0.045 | 18/39 |
| `n1_within_input_side` | NEG control | 47 | -0.013 | -0.039 | 4/47 |
| `n2_within_residual` | NEG control | 39 | -0.036 | -0.109 | 14/39 |
| `n3_same_operator` | NEG control | 38 | -0.054 | -0.219 | 10/38 |

**Under the held-out protocol it does not survive.** Selected on CIFAR-10 and GTSRB (n=20),
the winner is `c5_ban_pre_5_8`; on the held-out CIFAR-100 and Tiny cells it reads
**+0.0145, bootstrap 95% CI [-0.0032, +0.0349], winning 8 of 19**. The interval spans zero.

Worse for the claim, the held-out gain is carried entirely by the attack that should not be
generalised from:

| attack | n | delta |
|---|---|---|
| badnet_a2o | 6 | +0.056 |
| lc | 1 | +0.035 |
| lf | 6 | +0.000 |
| blend | 6 | **-0.016** |

## Verdict

**Not supported.** On present evidence a defender should deploy the single best placement
rather than a fused set. The negative controls behaving exactly as predicted says the design
is sound and the effect is simply absent or small.

**This is underpowered and provisional.** The selection split has 20 cells and its top six
combinations sit within 0.003 of each other, which at this n is a coin flip. Coverage is
still filling in: 25 basis jobs are extending the panel from 39 usable cells toward 56, and
the hard attacks (wanet, sig, adaptive_blend, bpp, tact) are almost absent from the present
sample, which is exactly where a fusion would most plausibly help. Re-run both scripts when
the batch lands before treating the verdict as final.

## Reproduce

```
PYTHONPATH=. python experiments/probe_fusion/measure.py --shift-target 0.6
PYTHONPATH=. python experiments/probe_fusion/summarise.py
PYTHONPATH=. python experiments/probe_fusion/summarise.py --rule mean_rank
```
