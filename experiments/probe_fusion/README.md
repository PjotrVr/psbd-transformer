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

## Result: on the COMPLETE panel, fusion does not beat the best single placement

Run on all 67 cells clearing the 0.85 ASR bar, every one carrying the complete 18-placement
basis.

Selected on CIFAR-10 and GTSRB, reported on the held-out CIFAR-100 and Tiny (n=33), against
the fixed single probe `before_attention_norm_token_mask`:

| quantity | value |
|---|---|
| mean delta AUROC | **+0.008** |
| bootstrap 95% CI | **[-0.007, +0.023]** |
| cells won | 17 of 33 |
| verdict | **not supported, the interval spans zero** |

Merging the two best single configurations specifically buys almost nothing, and the reason
is mechanical: `both_sublayer_inputs_token_mask` is token masking at `before_attention_norm`
AND `before_mlp_norm`, so it already CONTAINS the other. Fusing two probes that share a
position fuses two correlated views.

| configuration | AUROC, all (n=67) | AUROC, hard (n=31) | delta vs the best single |
|---|---|---|---|
| `before_attention_norm_token_mask` alone | 0.923 | 0.878 | reference |
| `both_sublayer_inputs_token_mask` alone | 0.908 | 0.864 | |
| merge of the two (shared position) | 0.927 | 0.885 | +0.005, CI [-0.001, +0.011] |
| merge across families, adding `pre_residual_blocks_9_12` | 0.937 | 0.891 | +0.015, CI [-0.006, +0.037] |
| all three | 0.940 | 0.897 | +0.018, CI [-0.002, +0.038] |

Spanning position families is still worth more than merging within one, which is the
direction the pre-registered C2 family predicted. It is simply not worth enough to clear a
confidence interval.

## This verdict replaces TWO earlier ones, and the reason is the same both times

| when | n usable cells | held-out delta | verdict |
|---|---|---|---|
| basis batch still running | 39 | +0.0145, CI [-0.003, +0.035] | not supported |
| basis batch landed, TaCT not yet corrected | 56 | +0.039, CI [+0.022, +0.056] | supported |
| **complete panel, TaCT corrected and swept** | **67** | **+0.008, CI [-0.007, +0.023]** | **not supported** |

The protocol and the pre-registered combination list never changed. The panel did. The middle
reading was taken while TaCT was absent entirely, and TaCT is 11 of the 31 hard cells. The
residual depth-band members that carry the combination score **0.624** on TaCT against
**0.869** for input-side token masking, so adding TaCT removes the combination's edge.

The lesson is about coverage, not about fusion: a fusion result read on a panel missing a
whole attack is a result about the attacks that happened to be present. This is the third
time in this project that unequal coverage produced a confident number that did not survive
equal coverage.

## Verdict

**Not supported.** Deploy the single best placement. `before_attention_norm_token_mask` has
the highest hard-attack AUROC of any configuration measured (0.878, n=31), reaches the
matched shift ratio on 100% of cells, and is rank #4/#1/#1 across poison rates on hard
attacks. Fusion adds at most +0.018 and no combination clears a confidence interval.

The structural caveat stands, and is worth keeping: band-restricted TOKEN masking cannot be
brought to the disturbance the adaptive rule requires (sigma 0.705 on blocks 5-8, 0.433 on
blocks 9-12, at masking probability 0.99, because the 8 unperturbed blocks still carry the
signal and a probability cannot exceed 1). Band-restricted residual DROPOUT has no such
ceiling, reaching 0.959 at p=0.99.

## Reproduce

```
PYTHONPATH=. python experiments/probe_fusion/measure.py --shift-target 0.6
PYTHONPATH=. python experiments/probe_fusion/summarise.py
PYTHONPATH=. python experiments/probe_fusion/summarise.py --rule mean_rank
```
