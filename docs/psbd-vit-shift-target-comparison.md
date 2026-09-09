# Which clean-validation shift ratio should the tables be read at?

Two different quantities were being used interchangeably in this repo, and they answer
different questions.

| constant | value | what it is |
|---|---|---|
| `ADAPTIVE_SHIFT_TARGET` | **0.8** | the **PSBD paper's own operating point**. Yang et al. select the dropout rate "where the $\sigma$ of clean validation data approach to a high value (0.8 in our experiments), while the difference between the $\sigma$ of the entire training data and that of the clean validation data reaches its maximum" (`papers/PSBD/sec/4_method.tex:185`) |
| `SHIFT_MATCH_TARGETS` | (0.2, 0.4, **0.6**, 0.8) | this project's ladder for **comparing** placements at equal effective strength, so that "this position is better" cannot be confounded with "this position was perturbed harder" |

The first tables were generated at **0.6**, the comparison midpoint, not at the paper's
operating point. That was wrong for a deployment table: a defender running PSBD runs the
paper's adaptive rule, so the number that describes deployment is the one at 0.8.

`docs/shift_06/` keeps the original documents. `docs/shift_08/` holds the same measurements
at the paper's value.

## 0.8 is better on every axis that matters

Averaged over all 18 basis placements, 67 cells whose attack implanted:

| target | AUROC | hard-attack AUROC | TPR@10%FPR | TPR@20%FPR | reach | inverted cells |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.739 | 0.687 | 0.320 | 0.510 | 97% | 132 |
| 0.4 | 0.791 | 0.731 | 0.451 | 0.620 | 88% | 93 |
| 0.6 | 0.838 | 0.776 | 0.574 | 0.702 | 82% | 78 |
| **0.8 (paper)** | **0.852** | **0.782** | **0.633** | **0.738** | **94%** | 83 |

**16 of 18 placements score their best AUROC at 0.8.** The two exceptions are
`both_sublayer_inputs_token_mask` (best at 0.6, 0.908 against 0.904) and
`pre_residual_blocks_9_12` (best at 0.4), both of which saturate early.

### The surprise: reach is HIGHER at 0.8, not lower

The obvious worry was that a higher target would be unreachable. The opposite happened:
reach rises from 82% at 0.6 to **94%** at 0.8. The reason is that a perturbation's shift
ratio does not climb smoothly. It stays low, then saturates quickly toward 1.0, so 0.6 often
falls in a GAP in a placement's ladder while 0.8 lands on the saturating part.

| placement | reach at 0.6 | reach at 0.8 |
|---|---:|---:|
| `pre_residual` | 63% | **100%** |
| `post_residual` | 63% | **100%** |
| `after_attention_residual_token_mask` | 40% | **94%** |
| `before_mlp_gaussian` | 57% | **85%** |
| `mlp_norm_out_gain_scale` | 48% | **73%** |

Only the band-restricted token masks get worse, and for the structural reason already
recorded: masking 4 of 12 blocks cannot drive the disturbance high enough, because the 8
unperturbed blocks still carry the signal and a masking probability cannot exceed 1.
`before_attention_norm_blocks_9_12_token_mask` falls from 75% to **60%**.

So measuring at 0.6 was worse in BOTH senses: it understated detection AND it compared more
placements outside their reachable range.

## What changes in the conclusions

**The winner does not change, and gets cleaner.** `before_attention_norm_token_mask` is
first overall and first on hard attacks at 0.4, 0.6 and 0.8. At the paper's value it is
ranked #2/#1/#1 across the three poison rates on hard attacks, the best worst-case rank of
any placement, at 100% reach.

| configuration | sigma 0.6 | sigma 0.8 |
|---|---|---|
| `before_attention_norm_token_mask` | AUROC 0.923, hard 0.878, TPR@10 0.747, reach 100% | **AUROC 0.935, hard 0.883, TPR@10 0.790, reach 100%** |
| `before_attention_residual_token_mask` | AUROC 0.905, hard 0.837, TPR@10 0.760, reach 97% | **AUROC 0.925, hard 0.864, TPR@10 0.801, reach 100%** |
| `both_sublayer_inputs_token_mask` | AUROC 0.908, hard 0.864, TPR@10 0.695, reach 82% | AUROC 0.904, hard 0.865, TPR@10 0.698, reach 97% |

**The second-best configuration changes.** At 0.6 it was `both_sublayer_inputs_token_mask`;
at 0.8 it is `before_attention_residual_token_mask`, which gains +0.020 AUROC and +0.027 on
hard attacks when read at the paper's strength.

**Fusion flips from unsupported to supported.** Under the identical pre-registered protocol
(selected on CIFAR-10 and GTSRB, reported on the held-out CIFAR-100 and Tiny):

| target | held-out delta | 95% CI | wins | verdict |
|---|---|---|---|---|
| 0.6 | +0.008 | [-0.007, +0.023] | 17/33 | not supported |
| **0.8** | **+0.009** | **[+0.003, +0.016]** | **23/33** | **holds** |

The effect size is the same; the variance is smaller because more members reach the target,
so the fusion is combining probes that are actually at matched strength.

Merging the top two at 0.8, which now spans the input-side and residual-adjacent families
rather than overlapping on one position:

| subset | n | merged AUROC vs best single | merged TPR@10%FPR vs best single |
|---|---:|---|---|
| all cells | 67 | **+0.017** [+0.007, +0.028] | **+0.060** [+0.027, +0.098] |
| **hard** attacks | 31 | **+0.028** [+0.010, +0.051] | **+0.110** [+0.045, +0.180] |
| held out (cifar100 + tiny) | 33 | **+0.011** [+0.001, +0.024] | +0.045 [-0.005, +0.104] |

Absolute numbers for that merge: AUROC **0.952**, TPR@10%FPR **0.850**, TPR@20%FPR **0.909**,
and on hard attacks AUROC **0.911** with TPR@10%FPR **0.696**.

## One thing 0.8 does not improve

Rate stability is marginally worse panel-wide: 14 of 18 placements swing more than 3 ranks
across poison rate at 0.8, against 12 of 18 at 0.6, and the 5%-vs-10% rank correlation falls
from +0.759 to +0.513. This does not affect the recommendation, because the winning
configuration becomes MORE stable (ranks 2/1/1 at 0.8 against 4/1/1 at 0.6). It does mean
the tail of the ranking should not be read as an ordering at either target.

## Verdict

Report at **0.8**, the paper's operating point. It is better on AUROC, on both TPR operating
points, and on reach, it leaves the winner unchanged, and it is the number that describes
what a deployed PSBD actually does. Keep the 0.2 to 0.8 ladder as the comparison instrument
it was built to be, and keep `docs/shift_06/` so the difference stays auditable.

## Reproduce

```
PYTHONPATH=. python experiments/probe_fusion/measure.py --shift-target 0.8 \
    --out results/coverage/probe_fusion_s0.8.json
PYTHONPATH=. python scripts/vit_shift_target_compare.py
PYTHONPATH=. python scripts/vit_config_tables.py --shift-target 0.8 \
    --placement before_attention_norm_token_mask \
    --out docs/shift_08/psbd-vit-config-1-before-attention-norm-token-mask.md
```
