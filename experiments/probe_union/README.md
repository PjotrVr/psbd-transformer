# Probe union on ordinary, non-adaptive backdoored ViT-B/16 models

## Question

H41 (`docs/hypothesis/H41-multi-probe-defence.md`) built a min-rank union of
independent probes to defeat an attacker trained against 1 probed operator.
Nobody trains against a probe on the 65 clearing cells the paper's headline
reads (`paper/tables/headline.tex`). Does the same union rule help, hurt, or do
nothing on those ordinary models, and does it specifically rescue the 2
inverted cells the headline names, cifar10 wanet at 10% (0.459) and cifar10 sig
at 10% (0.418 here, 0.878 at the matched rule in the headline)?

## Method

The union rule is unchanged from H41: for probe j, rank_j(x) is the percentile
of x's fractional PSU (`psu_ratio`, the canon headline statistic) within probe
j's own clean-validation distribution, the combined score is min_j rank_j(x),
and the calibrated threshold is the target-FPR quantile of that combined score
on clean validation. `defences.decision.multi_probe_auroc` and
`multi_probe_detection` compute both, unmodified.

Models are the 65 clearing cells that carry both headline placements, selected
exactly as `scripts/paper/tab_headline.py` selects them
(`experiments/probe_union/measure.py:select_models`). Every per-sample PSU is
read from the stage-1 cache under `results/<folder>/psbd/<placement>/`, at the
rate `psbd_metrics.json`'s `adaptive_rate` chose for that placement on that
model, the same reader `cli.analyze` and `scripts/paper/fig_psu_histograms.py`
use.

6 probe sets:

1. `psbd_tm`: PSBD-TM alone (`before_attention_norm_token_mask`), the reference.
2. `psbd_tm_rd`: PSBD-TM plus PSBD-RD (`post_residual`).
3. `psbd_tm_attn_branch`: PSBD-TM plus token masking on the attention branch
   output (`before_attention_residual_token_mask`).
4. `adaptive_3probe`: H41's adaptive-attacker pool minus PSBD-RD and gaussian,
   PSBD-TM plus dropout at the attention input (`before_attention_norm`,
   operator dropout) plus gain scaling of the MLP LayerNorm output
   (`mlp_norm_out_gain_scale`). The dropout probe is cached on 37 of the 65
   models, so this set and the next are read on those 37.
5. `adaptive_4probe`: set 4 plus PSBD-RD.
6. `all_65_basis`: every basis placement whose `adaptive_rate` is set on all 65
   models, found by checking the cache rather than hardcoded
   (`basis_ids_present_on_all_models`), 15 placements.

Each model contributes 1 AUROC and 1 TPR at each of 2 target FPRs (0.10, 0.20)
under the calibrated rule. The paired gain over PSBD-TM alone is measured
within model (matched by folder name) over whichever models both sets cover,
with a 5000-resample bootstrap 95% interval, seed 0
(`scripts/paper/_common.bootstrap_ci`).

Run:

    PYTHONPATH=. .venv/bin/python experiments/probe_union/measure.py
    PYTHONPATH=. .venv/bin/python scripts/paper/tab_probe_union.py

Output: `results/_experiments/probe_union/probe_union.json` (per-model
numbers), `paper/tables/probe_union.tex` and `.macros.json`.

## Results

| Probes | n | AUROC | TPR@FPR 10% | TPR@FPR 20% | Gain over PSBD-TM [95% CI] |
|---|---:|---:|---:|---:|---|
| PSBD-TM alone | 65 | 0.935 | 0.800 | 0.859 | -- |
| PSBD-TM + PSBD-RD | 65 | 0.941 | 0.811 | 0.864 | +0.006 [-0.016, +0.032] |
| PSBD-TM + attention branch output token mask | 65 | 0.957 | 0.854 | 0.912 | +0.022 [+0.006, +0.043] |
| PSBD-TM + attention input dropout + MLP norm-out gain scale | 37 | 0.988 | 0.979 | 0.985 | +0.020 [+0.004, +0.048] |
| + PSBD-RD (4-probe) | 37 | 0.990 | 0.985 | 0.989 | +0.022 [+0.004, +0.052] |
| every basis placement present on all 65 models (15 probes) | 65 | 0.954 | 0.848 | 0.895 | +0.019 [-0.000, +0.044] |

### Per-attack mean AUROC, PSBD-TM alone vs PSBD-TM + PSBD-RD (both n = 65)

| Attack | PSBD-TM | PSBD-TM + PSBD-RD |
|---|---:|---:|
| badnet_a2o | 0.992 | 0.985 |
| blend | 0.978 | 0.992 |
| bpp | 0.948 | 0.957 |
| lf | 0.980 | 0.984 |
| sig | 0.418 | 0.908 |
| tact | 0.853 | 0.772 |
| wanet | 0.845 | 0.947 |

### The 2 named inverted cells, PSBD-TM alone vs PSBD-TM + PSBD-RD

| Cell | PSBD-TM | PSBD-TM + PSBD-RD |
|---|---:|---:|
| cifar10 wanet 10% | 0.459 | 0.910 |
| cifar10 sig 10% | 0.418 | 0.908 |

### Per-attack mean AUROC, the 4-probe union (n = 37, no bpp, tact or wanet cell holds all 4 probes)

| Attack | PSBD-TM (37-model subset) | 4-probe union |
|---|---:|---:|
| badnet_a2o | 0.987 | 0.996 |
| blend | 0.980 | 0.998 |
| lf | 0.983 | 0.982 |
| sig | 0.910 | 0.921 |

## Conclusion

The union still helps on ordinary models, but by a smaller and more uneven
margin than against an adaptive attacker, and it is not a free lunch. Adding
PSBD-RD to PSBD-TM rescues both named inverted cells outright (wanet 0.459 to
0.910, sig 0.418 to 0.908) but costs tact 0.081 AUROC on average, so its panel-wide
paired gain is a statistical wash (+0.006, CI crosses 0), the same failure mode
H41 already named for rank-averaging: a member that reads one attack badly
drags the minimum down on that attack even while it saves another. The
2-probe union with the attention branch output token mask instead of PSBD-RD
clears the panel more cleanly (+0.022, CI entirely positive) without the tact
cost, because its 2 members are less anti-correlated. The H41 adaptive pool
transfers well to its own 37-model subset (+0.020 to +0.022) but that subset
never contains a bpp, tact or wanet cell, so it cannot be read as evidence for
or against the rescue question; the 15-probe all-basis union gains the least
per probe added (+0.019, CI barely excludes 0, and both TPR columns fall below
the 3-probe set), confirming H41's caution that more probes only helps when
they disagree in the right way, not simply by number.
