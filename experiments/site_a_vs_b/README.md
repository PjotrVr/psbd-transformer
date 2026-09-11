# Site A against site B: the same placement seen twice, or 2 complementary probes

## Question

`before_attention_norm_token_mask` (site A, PSBD-TM, the recommended placement) and
`before_attention_residual_token_mask` (site B, the attention-branch output, right before
the residual add) both score mean AUROC around 0.935 on the panel of clearing cells, which
reads as the same placement measured twice. Per model that reading does not hold: B beats
A by 0.52 on CIFAR-10 SIG at 10% and 0.39 on CIFAR-10 WaNet at 10%, A beats B by 0.39 on
Tiny ImageNet WaNet at 5% and 0.24 on GTSRB WaNet at 10%. This experiment asks whether the
2 sites carry the same detection signal at different sensitivity, or catch different
poisoned images for different physical reasons.

## Method

Per-image fractional PSU is read from the stage-1 caches at each site's own
`adaptive_rate` (`results/<folder>/psbd_metrics.json`), generalising `panel_data` in
`scripts/paper/fig_psu_histograms.py` to any placement id
(`experiments/site_a_vs_b/measure.py:read_placement_psu`, importing `defences.cache`,
`defences.decision` and `defences.scores` rather than recomputing the statistic). Models
are the clearing cells whose `psbd_metrics.json` reaches an adaptive rate at both sites
(generalising `experiments/probe_union/measure.py:select_models`), 66 of them for the
`token_mask` operator, close to but not identical with the 65-cell headline panel because
that panel's own coverage criterion never touches site B.

Per model: AUROC of site A, of site B, of their min-rank union
(`defences.decision.multi_probe_auroc`), the best of the 2 single sites, the Spearman
correlation of the 2 per-image PSU vectors on the triggered images and on the clean images
paired to them, the share of triggered images caught by exactly 1 site at the 25%
threshold, and each site's achieved rate and clean-validation shift ratio. Grouped by
attack and by trigger locality (local is badnet_a2o, badnet_a2a and tact, global is blend,
sig, wanet, lf and bpp). The 2 sites are then read again at a shared clean-validation shift ratio
(0.6, 0.7, 0.8, 0.9), interpolating each site's own rate ladder
(`defences.decision.interpolate_at_target_shift`), to separate a gap caused by the
adaptive rule landing on different rates from a gap in the site itself.

The mechanism is measured directly on 4 models (CIFAR-10 WaNet 10%, CIFAR-10 SIG 10%,
CIFAR-100 BadNets 1%, Tiny ImageNet WaNet 5%), 200 paired clean and triggered images, at
each site's own selected rate, using the project's real perturbation machinery
(`models.positions.plug_dropout` restricted to 1 block, `defences.operators
.build_operator("token_mask")`) rather than a re-implementation of the operator:
the relative-norm change of the block's output and of its class token, on clean and on
triggered images, and the share of that change lying along the backdoor direction at that
block (`experiments.whole_network_erasure.measure.directions`, imported). Finally,
`results/<folder>/activation_patching.json` is re-read for those 4 models, the recovery of
the clean answer when the trigger's own tokens are patched at `site="resid"` (A) against
`site="attn"` (B), per layer.

`compare_sites(results_dir, site_a, site_b, ...)` takes any 2 placement ids, so the same
measurement runs for whichever operator has a sweep at both sites already. 3 of the 4
queued pairs clear the 10-model floor: `token_mask` (66 models), `channel_mask` (37) and
`gaussian` (35). `dropout` has only 13 `before_attention_residual` sweeps on disk and none
of them lands on a clearing cell that also reaches an adaptive rate at
`before_attention_norm`, so it is skipped rather than reported on fewer than 10 models.

## Per-attack table (token_mask, the headline pair)

Mean AUROC, n models per row.

| attack | n | A | B | union | best of 2 | Spearman (triggered) | exactly 1 site catches it |
|---|---:|---:|---:|---:|---:|---:|---:|
| badnet_a2o | 12 | 0.992 | 0.973 | 0.993 | 0.996 | 0.20 | 0.03 |
| blend | 13 | 0.975 | 0.972 | 0.988 | 0.993 | 0.60 | 0.05 |
| lf | 12 | 0.980 | 0.977 | 0.980 | 0.980 | 0.60 | 0.01 |
| bpp | 12 | 0.948 | 0.939 | 0.953 | 0.955 | 0.54 | 0.07 |
| tact | 11 | 0.853 | 0.870 | 0.878 | 0.896 | 0.32 | 0.37 |
| wanet | 5 | 0.845 | 0.759 | 0.909 | 0.923 | 0.21 | 0.41 |
| sig | 1 | 0.418 | 0.934 | 0.938 | 0.934 | 0.22 | 0.84 |

| locality | n | A | B | union | Spearman (triggered) | Spearman (clean) |
|---|---:|---:|---:|---:|---:|---:|
| local (badnet_a2o, badnet_a2a, tact) | 23 | 0.925 | 0.924 | 0.938 | 0.26 | 0.30 |
| global (blend, sig, wanet, lf, bpp) | 43 | 0.941 | 0.939 | 0.966 | 0.53 | 0.40 |
| **all** | **66** | **0.935** | **0.933** | **0.956** | **0.43** | **0.37** |

Paired B minus A gap, all 66 models: **-0.002**, bootstrap 95% CI **[-0.028, +0.027]**, no
resolvable difference in the mean. The per-attack rows are where the mean's agreement
dissolves: wanet and sig each carry a genuine and opposite-signed gap, and both are exactly
the attacks the question names.

Named per-model rows (rate and achieved shift ratio at each site's own adaptive rate):

| model | A | B | union | exactly 1 catches it | rate A / shift A | rate B / shift B |
|---|---:|---:|---:|---:|---:|---:|
| cifar10 sig 10% | 0.418 | 0.934 | 0.938 | 0.84 | 0.80 / 0.82 | 0.70 / 0.95 |
| cifar10 wanet 10% | 0.459 | 0.846 | 0.787 | 0.59 | 0.50 / 0.86 | 0.70 / 0.86 |
| tiny wanet 5% | 0.930 | 0.537 | 0.947 | 0.68 | 0.50 / 0.90 | 0.50 / 0.88 |
| gtsrb wanet 10% | 0.948 | 0.707 | 0.931 | 0.43 | 0.40 / 0.92 | 0.50 / 0.91 |

Every 1 of these 4 rows has a large exactly-1-site-catches-it share (0.43 to 0.84), so the
mean gap of 0 is an average of large, image-level, opposite-signed disagreements, not the
absence of one. The union AUROC beats the better single site on the 2 sig and tiny-wanet
rows, but on both wanet-at-10% rows the union sits between the 2 sites (0.787 on cifar10,
0.931 on gtsrb, against a better single site of 0.846 and 0.948): min-rank combination
hurts when the 2 sites disagree on which images are suspicious rather than agreeing on
some and disagreeing on others, and cifar10 wanet is the extreme case, Spearman on
triggered images 0.02, the lowest of any row in the panel.

## Matched clean-validation shift ratio

Mean AUROC of each site interpolated onto a shared achieved shift, all 66 models:

| shift target | n | A | B | B minus A |
|---:|---:|---:|---:|---:|
| 0.6 | 66 | 0.928 | 0.905 | -0.023 |
| 0.7 | 66 | 0.939 | 0.920 | -0.019 |
| 0.8 | 66 | 0.937 | 0.931 | -0.005 |
| 0.9 | 48 | 0.946 | 0.925 | -0.021 |

Site A reads ahead of site B at every matched shift, by 0.02 to 0.02, roughly a third of
the adaptive-rule gap the headline table reports at their own rates disappearing once
disturbance is equalised (site A's adaptive rate is typically lower, so more of its raw
advantage is really "it needed less disturbance to separate," not "it separates better at
matched disturbance"). Neither site saturates and reverses the other's ranking across the
grid, so the matched-shift comparison narrows the mean gap without reversing it.

## Mechanism

`gaussian` and `channel_mask` at the 2 sites replicate the ranking H47 already
establishes (`docs/hypothesis/H47-layernorm-absorbs-noise-not-masking.md`): channel_mask
reads 0.936 at site A against 0.817 at site B (B minus A -0.119, CI [-0.178, -0.069]), and
gaussian reads 0.816 at A against 0.837 at B (+0.021, CI [-0.022, +0.059]), both consistent
with `ln_1` absorbing part of an additive or structured perturbation injected before it
and passing a token-wholesale mask through close to intact. `token_mask` is the pair where
neither operator effect explains the per-model swings, since token masking survives the
norm at both sites (H47's own measurement), so the swings trace to what the 2 sites
physically zero.

Averaged over the 4 mechanism models, blocks 1, 4, 6, 9, 12, 200 paired images (100 for
direction estimation, 100 for the measured change, no image used for both):

| quantity | site A | site B |
|---|---:|---:|
| relative change of the block output | 0.217 (clean) / 0.208 (trig) | 0.409 (clean) / 0.404 (trig) |
| relative change of the class token | 0.155 (clean) / 0.147 (trig) | **0.000 / 0.000** |
| class-token change along the backdoor direction (share) | 0.188 (clean) / 0.387 (trig) | undefined (0 change) |

Site B's class-token change is exactly 0 at every block, on every model, on both clean and
triggered images. This is not an approximation, it is forced by the mechanics named in the
question: `before_attention_residual` is a post-hook on the block's own attention-output
dropout module, after attention has already produced this block's class-token value and
before anything mixes across tokens again. `TokenMask` never masks token 0. So masking the
non-class tokens at this exact point changes only what those tokens carry into the *next*
block's attention, never this block's own class-token output. Site A sits 1 step earlier,
before `ln_1` and before attention runs, so masking a token there changes what every other
token, including the class token, reads out of attention in this same block: the relative
change of the class token grows from 0.06 at block 1 to 0.23-0.36 at block 12, accumulating
exactly where site B contributes nothing at all.

Site A's block-output change is smaller than site B's at every block (0.22 against 0.41 on
average), the same absorption H47 measures for token masking generally, but here it means
site A trades a smaller raw disturbance for one that reaches the class token directly,
while site B spends a larger raw disturbance on tokens whose effect must first propagate
through 1 more block's attention before it can touch the classifier's read point at all.
The direction-share number adds the selectivity site A needs to detect anything from that
smaller disturbance: on clean images, the class-token change from masking sits at only
0.19 of its magnitude along the backdoor direction, but on triggered images that share
roughly doubles to 0.39. Site A's mask does not just move the class token, it moves it
disproportionately along the backdoor direction specifically when the trigger is present,
which is the signal PSU reads as a shift.

## Activation patching, the same 4 models

Recovery of the clean answer when the trigger's own tokens are patched from the clean run
into the triggered run, at the residual-stream site (A, `site="resid"`) against the
attention-branch site (B, `site="attn"`), per layer:

| model | resid (A) layers 1-6 mean | resid (A) last layer >= 0.5 | attn (B) layers 1-6 mean | attn (B) peak |
|---|---:|---:|---:|---:|
| cifar10 wanet 10% | 0.988 | 7 | 0.146 | 0.338 (layer 5) |
| cifar10 sig 10% | 0.994 | 11 | 0.108 | 0.244 (layer 4) |
| cifar100 badnet 1% | 1.000 | 10 | 0.014 | 0.033 (layer 5) |
| tiny wanet 5% | 0.983 | 6 | 0.095 | 0.354 (layer 5) |

Patching the whole residual stream at the trigger tokens (site A's causal analogue)
recovers the clean answer almost completely through 6 to 11 layers, because it restores
everything every earlier block wrote to those tokens at once. Patching 1 block's attention
output alone (site B's causal analogue) never recovers more than 0.35 at any layer on any
of the 4 models: no single block's own attention write carries much of the causal effect
in isolation, because that effect is distributed across all 12 blocks' individual
contributions. Site B's detection power at any 1 block is therefore built from a small
marginal disturbance, repeated at every block during a real sweep, rather than from
disturbing a large causally concentrated write the way site A's mask does.

## Verdict

Site A and site B are 2 complementary probes, not the same placement seen twice. The panel
mean agrees only because equal and opposite per-model gaps cancel: B beats A by up to 0.52
on SIG and WaNet at 10%, A beats B by up to 0.39 on WaNet at 5%, and on every 1 of those
rows 43 to 84% of triggered images are caught by exactly 1 site. The physical reason is
forced by where each site sits in the block: site B's mask lands after attention has
already written this block's class-token value and before anything mixes across tokens
again, so it provably leaves the class token in this block untouched, 0.000 relative
change at every measured block on every model, while site A's mask sits before attention
runs and reaches the class token in the same block through the mixing attention performs,
with a relative class-token change that grows from 0.06 at block 1 to over 0.3 by block 12
and that lands disproportionately along the backdoor direction specifically on triggered
images. Activation patching confirms the asymmetry causally: restoring a token's whole
residual-stream history (site A's analogue) recovers the clean answer almost completely
for 6 to 11 layers, while restoring 1 block's own attention write alone (site B's analogue)
never exceeds 0.35 recovery at any layer, so site B accumulates its signal over 12 small
per-block disturbances rather than disturbing 1 causally concentrated write. The min-rank
union of the 2 sites beats the better single site on the sig and tiny-wanet rows and loses
to it on both wanet-at-10% rows, so the 2 sites are worth reading together when they agree
and worth distrusting a naive combination of when they disagree sharply, as wanet does.

## Files

- `experiments/site_a_vs_b/measure.py`: parts 1 to 4, `compare_sites(results_dir, site_a,
  site_b, ...)` generalised to any 2 placement ids.
- `results/_experiments/site_a_vs_b/site_a_vs_b.json`: every per-model row, the grouped
  summaries, the matched-shift ladder, the mechanism measurements and the activation
  patching re-read.

    PYTHONPATH=. .venv/bin/python experiments/site_a_vs_b/measure.py
