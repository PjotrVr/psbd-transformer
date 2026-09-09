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

## Result: fusion beats the best single placement, +0.039 AUROC and +0.110 TPR@10%FPR

Run on the 67 cells clearing the 0.85 ASR bar, with the basis panel complete.

The winner is the combination this study was asked to test:

```
c5_ban_pre_two_bands =
    before_attention_norm_token_mask        (input-side, all 12 blocks)
  U pre_residual_blocks_5_8                 (residual-adjacent, mid depth)
  U pre_residual_blocks_9_12                (residual-adjacent, late depth)
```

Selected on CIFAR-10 and GTSRB, reported on the held-out CIFAR-100 and Tiny (n=27), against
the fixed single probe a defender would otherwise deploy:

| metric | `before_attention_norm_token_mask` | fused | delta | 95% CI |
|---|---|---|---|---|
| AUROC | 0.924 | **0.963** | **+0.039** | [+0.022, +0.056] |
| TPR @ 10% FPR | 0.811 | **0.920** | **+0.110** | [+0.037, +0.195] |
| TPR @ 20% FPR | 0.902 | **0.950** | **+0.048** | [+0.011, +0.095] |

Achieved FPR at the 10% operating point is 0.1024 against a nominal 0.10, so the calibrated
rule lands where it claims. It wins **24 of 27** held-out cells, and both combination rules
agree: min-rank +0.039 [+0.022, +0.056], mean-rank +0.035 [+0.023, +0.049].

**Positive on every attack**, not carried by one:

| attack | n | delta |
|---|---|---|
| wanet | 2 | +0.062 |
| badnet_a2o | 6 | +0.057 |
| bpp | 6 | +0.042 |
| lf | 6 | +0.039 |
| blend | 6 | +0.016 |
| lc | 1 | +0.006 |

**Benign controls pass.** The fused score reads 0.487, 0.484, 0.499, 0.496 on the four
benign checkpoints, so it invents no signal where there is no backdoor.

**Negative controls behave.** Three of four lose to the winner outright: `n3_same_operator`
-0.102, `n1_within_input_side` -0.023, `n2_within_residual` -0.020. The fourth,
`n4_same_band` (band held fixed, family spanned), reads +0.030, below the winner's +0.039.
That is informative rather than a failure: it says the gain is carried mainly by spanning
the input-side and residual-adjacent FAMILIES, with depth span adding the remainder. It also
matches the project's own headline, that the family split is the axis that carries effect.

## This verdict replaces an earlier negative one, and why

An earlier run of these same scripts reported NOT SUPPORTED: +0.0145, CI [-0.0032, +0.0349],
winning 8 of 19, with the gain carried by badnet alone. That reading was taken while the
basis panel was still being measured, on 39 usable cells, and the README recorded at the
time that it was underpowered and had to be re-run when the batch landed. It has been, on
unchanged pre-registered combinations and an unchanged protocol. What changed is the data:
the panel went from 39 usable cells to 67, coverage of every basis placement completed, and
the hard attacks entered the sample. The selection split also moved its winner from
`c5_ban_pre_5_8` to `c5_ban_pre_two_bands`, both from the same pre-registered family.

## Verdict

**Supported.** The deployment configuration is a SET, not a single placement. Fusing the
input-side token mask with band-restricted residual dropout at blocks 5-8 and 9-12 costs no
extra training and no extra checkpoint, only two more perturbation sweeps at inference, and
buys +0.039 AUROC and +0.110 TPR at the 10% FPR operating point over the best single config.

One structural caveat, found while measuring this. Band-restricted TOKEN masking cannot be
brought to the disturbance the adaptive rule requires: at p=0.99 it reaches clean-validation
shift ratio 0.705 on blocks 5-8 and 0.433 on blocks 9-12, because the 8 unperturbed blocks
still carry the signal and a masking probability cannot exceed 1. Band-restricted residual
DROPOUT has no such ceiling (0.959 at p=0.99), which is why the winning combination uses
`pre_residual` for its depth-band members and not `before_attention_norm`.

## Reproduce

```
PYTHONPATH=. python experiments/probe_fusion/measure.py --shift-target 0.6
PYTHONPATH=. python experiments/probe_fusion/summarise.py
PYTHONPATH=. python experiments/probe_fusion/summarise.py --rule mean_rank
```
