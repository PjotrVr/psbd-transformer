# Three falsifiable predictions of the curvature framing, tested against cached data

`docs/theory-perturbation-consistency.md` reframes PSBD as curvature estimation and
records 3 predictions that follow from the closed form of the logit layer Hessian
trace. This directory tests all 3 on data already on disk. No GPU, no retraining,
no new forward pass.

| prediction | verdict | the number that decides it |
|---|---|---|
| **P1** a probe near the head cannot beat a confidence baseline | **CONFIRMED**, with the direction corrected | head probe AUROC = 1 minus the confidence baseline, Pearson **1.0000**, mean abs difference **0.0019**, 83 checkpoints |
| **P2** inversions concentrate on low confidence clean samples | **REFUTED** | dropping the least confident clean decile moves an inverted cell by **+0.0036** AUROC, 44 inverted cells |
| **P3** the certified radius reformulation changes thresholds, not ranking | **REFUTED as stated** (it changes neither); the AUROC identity holds | max threshold effect of `Phi^{-1}` is **1.3e-4** FPR on 1 of 332 cells |

**Data snapshot.** All numbers read `results/detection_summary.csv` as written at
2026-09-07 04:45 and the cached tensors under `results/<folder>/psbd/` as they
stood at 2026-09-07 05:35, against `HEAD` = `30c5936`. The working tree was being
edited by another process during the run (`docs/results-report.md`,
`docs/hypothesis/README.md` and `.claude/CLAUDE.md` all changed mid session), so
re-derive rather than assume if the summary has since been regenerated.

Everything below is reproducible from the repo root with

```bash
PYTHONPATH=. .venv/bin/python experiments/theory_predictions/<script>.py
```

Runtime is 1 to 20 minutes per script, all of it filesystem reads.

---

## 0. The closed form the whole thing rests on

`hessian_trace_identity.py`

The 3 predictions all descend from

```
    original form
        tr grad^2_z p_c = 2 p_c (||p||_2^2 - p_c),   p = softmax(z)

    descriptive form
        hessian_trace = 2 * prob_predicted * (sum_of_squared_probs - prob_predicted)
```

Checked by autograd before anything else was measured, because if it is wrong P1
and P2 have no basis.

| classes | logit scale | max abs gap, closed form against autograd |
|---:|---:|---:|
| 2 | 1.0 | 1.9e-16 |
| 10 | 1.0 | 1.4e-16 |
| 100 | 1.0 | 5.6e-17 |
| 100 | 3.0 | 3.3e-16 |
| 100 | 0.3 | 8.6e-18 |

40 random softmaxes each. The 2 class corollary `tr(p) = 2p(2p-1)(p-1)` is 0 at
both `p = 0.5` and `p = 1`, non positive throughout, has exactly 1 turning point,
and peaks at **p = 0.7887**, matching the 0.789 the theory document states.

**The formula is correct.** The tests proceed.

---

## P1. A probe near the classifier head cannot beat a confidence baseline

`head_probe_vs_confidence.py`, `head_probe_is_softmax_only.py`

### Question

At the logits the trace depends on nothing but the softmax vector `p`. So a probe
whose site approaches the classifier must lose whatever it knew that confidence
does not, and its detection power must decay toward a free max softmax baseline.

### Method

1. The free baseline: for each of **282 checkpoints** (166 backdoored ViT,
   111 backdoored Swin, 5 benign) load `baseline_clean.pt` and
   `baseline_backdoor.pt`, pair the clean split to the backdoor split through the
   manifest exactly as the detector does, and score every sample by its max
   softmax probability. Backdoor is the positive class and higher confidence is
   the poisoned direction, since a backdoored model is very sure about a
   triggered input. No perturbation is involved.
2. Mean detection AUROC per position from `results/detection_summary.csv`,
   `variant.isna()` only, positions ordered by forward depth, each position's gap
   measured against the baseline averaged over exactly the checkpoints that
   contributed to it.
3. The operator controlled version. `gain_scale` is the only operator that runs
   at 3 depths, so `attention_norm_out` (inside every block, early),
   `mlp_norm_out` (inside every block, late) and `final_norm_out` (once, after
   the whole stack, 1 linear layer from the logits) is a depth comparison with the
   operator held fixed, on the 83 checkpoints carrying all 3.
4. The sharp version, which needs no AUROC at all: reconstruct the probe's own
   output from the unperturbed softmax with **zero** forward passes. Scaling the
   final LayerNorm output by `omega` scales the logits, and since `p` determines
   `z` up to a constant, the perturbed confidence is `p_c^omega / sum_j p_j^omega`.
   `mlp_norm_out` is the control: same operator, mid stack, reconstruction must
   fail there.

**A depth caveat that has to travel with the table.** Every block scope position
attaches in EVERY block, so it spans the whole stack and has no single depth. Only
`input_pixels` and `after_embedding` (below the stack) and `final_norm_out` (above
it) sit at a fixed depth, and those are the 3 the prediction actually separates.

### Result

Confidence baseline, no perturbation, 282 checkpoints:

| architecture | backdoored | n | max softmax AUROC | closed form `p_c(p_c - ||p||^2)` AUROC |
|---|---|---:|---:|---:|
| swin | yes | 111 | 0.5481 | 0.5487 |
| vit | yes | 166 | 0.5323 | 0.5313 |
| swin | no | 1 | 0.4925 | 0.4928 |
| vit | no | 4 | 0.4942 | 0.4945 |

The closed form logit statistic and plain max softmax agree to 0.001, which is the
theory's own claim measured directly: **at the logit layer PSU is the confidence
detector**.

Detection AUROC by position, oracle rate rule (the upper bound each position could
reach with perfect rate selection):

| depth | position | cells | checkpoints | mean AUROC | gap to confidence |
|---:|---|---:|---:|---:|---:|
| 0 | `input_pixels` | 67 | 67 | 0.7843 | +0.2286 |
| 1 | `after_embedding` | 207 | 115 | 0.7745 | +0.2118 |
| 2 | `before_attention_norm` | 554 | 266 | 0.7690 | +0.2131 |
| 3 | `attention_norm_out` | 83 | 83 | 0.8239 | +0.2659 |
| 4 | `before_attention` | 191 | 99 | 0.8442 | +0.2820 |
| 5 | `attention_heads` | 67 | 67 | 0.8292 | +0.2735 |
| 6 | `before_attention_residual` | 242 | 83 | 0.8242 | +0.2646 |
| 7 | `after_attention_residual` | 175 | 83 | 0.8324 | +0.2713 |
| 8 | `before_mlp_norm` | 175 | 83 | 0.7915 | +0.2305 |
| 9 | `mlp_norm_out` | 203 | 203 | 0.8316 | +0.2770 |
| 10 | `before_mlp` | 227 | 135 | 0.8087 | +0.2511 |
| 11 | `mlp_neurons` | 159 | 67 | 0.8325 | +0.2714 |
| 12 | `before_mlp_residual` | 357 | 143 | 0.8301 | +0.2669 |
| 13 | `after_mlp_residual` | 175 | 83 | 0.8468 | +0.2857 |
| **14** | **`final_norm_out`** | **83** | **83** | **0.4436** | **-0.1144** |

Operator controlled, same 83 checkpoints, `gain_scale` only:

| rule | `attention_norm_out` | `mlp_norm_out` | `final_norm_out` |
|---|---:|---:|---:|
| oracle | 0.8239 | 0.8751 | **0.4436** |
| matched sigma 0.6 | 0.7178 | 0.8155 | **0.4420** |

`final_norm_out` is not merely close to the baseline, it is the baseline reflected:

| rule | n | head probe AUROC | 1 minus confidence AUROC | Pearson | mean abs difference | max |
|---|---:|---:|---:|---:|---:|---:|
| matched sigma 0.2 to 0.8 | 83 | 0.4420 | 0.4420 | **1.0000** | 0.0019 | 0.0112 |
| oracle | 83 | 0.4436 | 0.4420 | 0.9999 | 0.0023 | 0.0253 |

Zero pass reconstruction, 30 checkpoints, 240 (rate, split) cells each:

| placement | split | mean abs error of smoothed confidence | Spearman(measured PSU, reconstruction) | Spearman(measured PSU, raw confidence) |
|---|---|---:|---:|---:|
| `final_norm_out` | clean | **0.0013** | **0.9975** | **0.9892** |
| `final_norm_out` | backdoor | **0.0011** | **0.9956** | 0.9531 |
| `mlp_norm_out` (control) | clean | 0.7484 | 0.4917 | 0.4964 |
| `mlp_norm_out` (control) | backdoor | 0.6814 | 0.4266 | 0.4542 |

### Verdict: CONFIRMED, with the direction corrected

The head adjacent probe carries no information the unperturbed softmax does not.
Its per sample output is reproduced from the softmax alone to Spearman 0.9975 and
mean absolute error 0.0013, its rank correlation with raw max softmax confidence
is 0.989, and its detection AUROC is `1 - AUROC(confidence)` to 3 decimal places
across 83 checkpoints. Every position that is not at the head beats the baseline
by +0.21 to +0.29.

The one correction to the prediction as written: AUROC does not decay **toward**
the baseline, it decays **past** it. `gain_scale` amplifies the pre head
activation, which sharpens the softmax most for the LEAST confident samples, so
PSU there ranks by low confidence, and "low PSU means poisoned" flags the least
confident sample while a triggered input is the most confident one. The
statistic is not degraded, it is inverted, and 0.4436 is exactly the mirror of
0.5580.

**What would have falsified it.** `final_norm_out` scoring in the same band as the
mid stack positions, or its PSU being poorly predicted by any function of the
softmax. Neither happened, on either the AUROC or the per sample test.

**Confound that has to be stated.** The only operator that runs at
`final_norm_out` is `gain_scale`, and `gain_scale` is not mean preserving (see
flag 1 below), so the second order derivation's assumption 3 does not hold at the
site where P1 is tested. That is why the depth comparison is done with the
operator held fixed and why the reconstruction test, which uses no expansion at
all, is the load bearing evidence. The claim "a MEAN PRESERVING probe at the head
cannot beat confidence" remains untested, and cannot be tested from this cache,
because no zero mean operator was ever swept at `final_norm_out`.

---

## P2. Inversions concentrate on low confidence clean samples

`inversion_confidence_profile.py`

### Question

The trace is non monotone in confidence and vanishes at both extremes, so "low PSU
means large margin" fails at the low confidence end as well as being trivial at
the high end. That predicts inverted cells (AUROC below 0.5) are driven by their
low confidence clean samples.

### Method

**44 inverted cells** (AUROC below 0.5) and **43 working cells** (AUROC above 0.8),
plain variants only, benign checkpoints excluded, at most 1 cell per checkpoint per
group, **77 checkpoints** total. For each, load the baselines and the per pass
probabilities and compute an exact decomposition rather than a correlation:

```
    original form
        AUROC = (1 / (|C| |B|)) * sum_{i in C} sum_{j in B} 1[psu_j < psu_i]

    descriptive form
        auroc = mean over clean samples of
                (fraction of backdoor samples whose PSU is lower)
```

so every clean sample has an exact additive share of the cell's AUROC and the
question has a numeric answer. Recomputation reproduced the summary AUROC to
machine precision on **85 of 87** cells; the 2 that did not are both `gaussian`
and are flag 2 below.

**Falsification criteria, fixed before the numbers were read.** The prediction
fails if the per clean contribution does not rise with confidence, or if the
inversion is present in the high confidence bins as strongly as the low ones, or
if the clean PSU against confidence curve is not single peaked.

### Result

| check | prediction requires | measured | passes |
|---|---|---|---|
| 1. contribution rises with confidence | Spearman > 0 | mean **-0.0761**, positive in **14 of 44** cells | no |
| 2. deficit sits below `p_c = 0.5` | low half worse | local AUROC low half **0.4729**, high half **0.3895** | no |
| 3. clean PSU curve single peaked | most cells | **14 of 44** (9 monotone increasing, 21 multi turn) | no |
| 4. removing the least confident clean decile | restores AUROC | **+0.0036** mean, +0.0546 best case, 5 of 44 cross 0.5 | no |

Local AUROC by baseline confidence bin, mean over the 44 inverted cells:

| bin | confidence | clean share | local AUROC | mean contribution | mean clean PSU | mean backdoor PSU |
|---|---|---:|---:|---:|---:|---:|
| 0 | 0.0 to 0.1 | 0.0001 | 0.6296 | 0.1620 | -0.0211 | 0.0517 |
| 1 | 0.1 to 0.2 | 0.0030 | 0.5305 | 0.1418 | 0.0494 | 0.0728 |
| 2 | 0.2 to 0.3 | 0.0084 | 0.5172 | 0.1519 | 0.1008 | 0.1048 |
| 3 | 0.3 to 0.4 | 0.0148 | 0.4752 | 0.1872 | 0.1599 | 0.1497 |
| 4 | 0.4 to 0.5 | 0.0229 | 0.4241 | 0.2202 | 0.2207 | 0.2376 |
| 5 | 0.5 to 0.6 | 0.0334 | 0.4331 | 0.2351 | 0.2653 | 0.2935 |
| 6 | 0.6 to 0.7 | 0.0332 | 0.4426 | 0.2645 | 0.3314 | 0.3409 |
| 7 | 0.7 to 0.8 | 0.0375 | 0.4162 | 0.2967 | 0.3961 | 0.4478 |
| 8 | 0.8 to 0.9 | 0.0535 | 0.4109 | 0.3379 | 0.4718 | 0.5430 |
| **9** | **0.9 to 1.0** | **0.7932** | **0.2750** | **0.2971** | 0.3734 | 0.5474 |

The decisive row is the last one. **79.3 percent of clean samples sit above
`p_c = 0.9`, and restricted to them the cell is just as inverted as it is
overall**: mean contribution 0.2971 against the cell mean 0.2933, local AUROC
0.2750. An inversion that lived on low confidence clean samples would disappear
here. It does not move.

The working cells behave the way the prediction expects an uninverted cell to
behave, which shows the measurement is capable of seeing the effect: their local
AUROC rises monotonically from 0.5755 to 0.9026 across the same bins.

### Verdict: REFUTED

The honest partial credit: per sample, the low confidence clean bins do have lower
contributions (0.14 to 0.19 in bins 0 to 3 against 0.34 in bin 8), so among that
minority the misranking is worse. But bins 0 to 4 hold **4.9 percent** of the clean
mass, the trend reverses in the bin holding 79 percent of it, and the
within cell correlation between contribution and confidence is not positive in
2 cells out of 3. Whatever causes inversions, it is not the low confidence end of
the curvature curve.

This also weakens the framing more broadly: the clean PSU against confidence curve
is single peaked in only 14 of 44 inverted and 16 of 43 working cells, so the
shape the closed form predicts is not the shape the data has at an internal probe
site.

---

## P3. The certified radius reformulation changes thresholds, not ranking

`certified_radius_thresholds.py`

### Question

`R = sigma * Phi^{-1}(p_tilde)` where `p_tilde = g(h) - phi(x)` is the smoothed
confidence the sweep already computes. `Phi^{-1}` is strictly monotone, so AUROC in
radius space must equal AUROC in smoothed confidence space, while a deployable
threshold is a quantile and `Phi^{-1}` stretches the tails, so the 2 threshold
rules should differ.

Three questions hide in that, and they have different answers, so they are
measured separately:

  a. does `Phi^{-1}` preserve AUROC,
  b. does `Phi^{-1}` change a quantile threshold rule,
  c. is the radius ranking the same as the DEPLOYED PSU ranking. PSU is
     `g(h) - p_tilde` and `g(h)` varies per sample, so the map from PSU to radius
     is not one shared monotone function and there is no identity to expect.

### Method

**332 cells over 213 checkpoints, 16 positions, 7 operators**: every
`before_attention_norm_token_mask` adaptive cell (the deployment recommendation)
plus a random spread over everything else. `sigma` is a per cell constant and drops
out of every ranking, so it is set to 1. `Phi^{-1}` needs a domain clamp at 1e-6
because float32 smoothed confidences do reach 0 and 1; how often that binds is
reported rather than hidden.

Directions: PSU flags LOW, smoothed confidence and radius flag HIGH, so their
quantile threshold is the upper tail `1 - q` of clean validation. Both rules read
clean validation only.

### Result

**a. `Phi^{-1}` preserves AUROC.**

| quantity | value |
|---|---:|
| max abs difference AUROC(radius) minus AUROC(smoothed confidence) | **8.2e-7** |
| cells differing at all | 12 of 332 |
| share of probabilities hitting the clamp | 0.0002 mean, 0.0274 max |

**b. `Phi^{-1}` does not change the quantile threshold rule either.**

| nominal q | max abs dFPR | max abs dTPR | cells differing |
|---:|---:|---:|---:|
| 0.01 | 0.0 | 1.3e-4 | 1 |
| 0.05 | 0.0 | 0.0 | 0 |
| 0.10 | 0.0 | 0.0 | 0 |
| 0.25 | 1.3e-4 | 0.0 | 1 |

A quantile rule is rank based, so a strictly monotone reparameterization moves
nothing except where `numpy` interpolates between 2 adjacent order statistics. Even
the cell where 2.74 percent of probabilities saturate the clamp changed no
operating point. **The threshold half of the prediction is refuted, and refuted for
a reason that was available a priori.**

**c. The genuine difference is PSU against smoothed confidence.**

| statistic | mean AUROC | wins |
|---|---:|---:|
| PSU (deployed) | 0.7080 | 171 of 332 |
| radius / smoothed confidence | 0.6914 | 161 of 332 |

Mean gap -0.0165 in favour of PSU, identical on 0 cells, max abs gap 0.7034.

At matched nominal budget, mean over 332 cells:

| q | PSU FPR | radius FPR | PSU abs error | radius abs error | PSU TPR | radius TPR | mean dTPR | median dTPR | radius wins | Wilcoxon p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.0103 | 0.0103 | 0.0024 | 0.0023 | 0.2413 | **0.3286** | **+0.0873** | +0.0003 | 0.521 | 5.4e-9 |
| 0.05 | 0.0502 | 0.0498 | 0.0053 | 0.0056 | 0.3553 | **0.4360** | **+0.0807** | +0.0004 | 0.527 | 1.7e-5 |
| 0.10 | 0.0998 | 0.0998 | 0.0073 | 0.0078 | 0.4273 | **0.4915** | **+0.0642** | +0.0006 | 0.533 | 1.6e-3 |
| 0.25 | 0.2492 | 0.2505 | 0.0104 | 0.0108 | 0.5914 | 0.5906 | -0.0007 | -0.0011 | 0.383 | 3.1e-3 |

Both rules hit their FPR budget equally well, within 0.002 to 0.011 mean absolute
error, so neither calibrates better. The TPR difference is real and one sided at
the low FPR operating points a defender would actually pick, but the **median gap
is near 0 and the win rate is near 0.52**, so it is carried by a minority of cells
where the radius rule is much better rather than by a broad shift. That is the
honest reading and it should not be reported as "the radius rule gains 9 points of
TPR".

### Verdict: REFUTED as stated

The prediction says the reformulation changes thresholds and not ranking. It
changes **neither**: the AUROC identity holds to 8e-7 (that half is confirmed), and
the quantile threshold rule is invariant to 1e-4 on 1 of 332 cells. The framing
that survives is different and more useful: what changes anything is dropping
`g(h)` from the statistic, that is, scoring by the smoothed confidence instead of
by the drop. That trades -0.017 mean AUROC for +0.06 to +0.09 mean TPR at
1 to 10 percent FPR, at zero extra cost, and it is worth a proper paired study
that this panel is only a first look at.

---

## Files

| file | what it is |
|---|---|
| `hessian_trace_identity.py` | autograd verification of the closed form and its 2 class corollary |
| `head_probe_vs_confidence.py` | P1: confidence baseline per checkpoint, position depth table, operator controlled gain_scale comparison, mirror test |
| `head_probe_is_softmax_only.py` | P1 sharp form: zero pass reconstruction of the head probe, with a mid stack control |
| `inversion_confidence_profile.py` | P2: exact per clean sample AUROC decomposition, confidence binned local AUROC, curve shape |
| `certified_radius_thresholds.py` | P3: AUROC identity, quantile rule invariance, PSU against smoothed confidence operating points |
| `summary_matches_cache.py` | audit written after flag 2 surfaced: does `detection_summary.csv` still agree with the cached tensors |
| `p1_confidence_baseline.csv` | per checkpoint confidence and logit statistic AUROC, 282 rows |
| `p1_position_vs_confidence.csv` | position by rule table with the paired baseline |
| `p1_head_probe_reconstruction.csv` | per (checkpoint, rate, split) reconstruction error |
| `p2_inversion_cells.csv` | per cell P2 record, 87 rows |
| `p2_confidence_bins.csv` | per (cell, confidence bin) record, 870 rows |
| `p3_radius_vs_psu.csv` | per cell AUROC and operating points in all 3 score spaces, 332 rows |
| `summary_vs_cache.csv` | per cell summary against recomputation, 3215 rows |

---

## Flags found along the way

Reported, not fixed, as instructed.

### Flag 1. `docs/theory-perturbation-consistency.md:16` states an assumption 2 operators break

> "Every operator in this study is mean preserving by construction"

`GainScale` (`psbd/operators.py:386`, `amplified = x * (1.0 + self.rate)`) and
`ScaleUp` (`psbd/operators.py:429`) both have `delta = rate * x`, so
`E[delta] = rate * x`, not 0. Both are registered as deterministic
(`psbd/operators.py:536`), so their PSU has no expectation to take at all. The
derivation's line `docs/theory-perturbation-consistency.md:55`, "assumption 3 kills
the 1st order term", therefore does not apply to either, and what they measure is a
first order directional derivative `-rate * grad_g^T h`, not a curvature.

This is not a code bug, the operator classes say plainly what they do
(`psbd/operators.py:375` even calls `gain_scale` "the only operator in the study
that amplifies rather than removes"). It is the theory document overstating its
coverage. It matters because those 2 operators are the ONLY ones swept at
`attention_norm_out`, `mlp_norm_out`, `final_norm_out` and `input_pixels`, which is
4 of the 15 positions in the depth table above, including the head position P1 is
tested at.

### Flag 2. Every plain `gaussian` row in `results/detection_summary.csv` is stale

Two distinct failures, both confined to `gaussian` and to no other operator.

**2a. 735 of 1024 unique plain gaussian cells have no cache in `results/` at all,
and reproduce bit for bit from the superseded operator's archive.**

Taking `variant.isna()` rows with a rate, deduplicated on (folder, placement,
rate), and asking whether `results/<folder>/psbd/<placement>/rate_<tag>_clean.pt`
is present. Counted with one `listdir` per placement directory rather than a
per file `os.path.exists`, because on this Lustre mount a cold per file lookup
returned a handful of spurious misses that a directory listing then corrected:

| operator | cells | no placement directory | directory present, rate file missing | present |
|---|---:|---:|---:|---:|
| `gaussian` | 1024 | **732** | **3** | 289 |
| every other operator | 13161 | 0 | 0 | 13161 |

All **32** affected checkpoints have that placement under
`archive/gaussian_batchstd/<folder>/<placement>/`, and recomputing the AUROC from
there reproduces the CSV **exactly**, gap `0.00e+00`:

```
vit_cifar100_adaptive_blend_0_01  after_embedding_gaussian     rate 0.3  csv 0.800498  archive 0.800498
vit_cifar100_adaptive_blend_0_01  after_embedding_gaussian     rate 0.1  csv 0.548353  archive 0.548353
vit_cifar100_adaptive_blend_0_01  after_mlp_residual_gaussian  rate 0.5  csv 0.784424  archive 0.784424
```

So those rows ARE the batch coupled Gaussian that audit finding A2 superseded, and
they are presented as plain measurements. The tagging rule that should catch them,
`scripts/detection_summary.py:79`, matches only the literal suffix
`_gaussian_batchstd`; these 32 checkpoints kept the bare `_gaussian` key in their
`psbd_metrics.json`, so nothing tags them. 64 other checkpoints WERE renamed and
are correctly tagged, and the 2 sets are disjoint.

**2b. Of the cells that do have a current cache, 80.6 percent still disagree.**

`summary_matches_cache.py --cells 4000`, 3215 cells over 264 checkpoints, stratified
by operator:

| operator | cells | disagreeing (> 1e-6) | share | max gap |
|---|---:|---:|---:|---:|
| `gaussian` | 134 | **108** | **0.806** | **0.0492** |
| `channel_mask` | 500 | 0 | 0 | 0 |
| `dropout` | 500 | 0 | 0 | 0 |
| `droppath` | 500 | 0 | 0 | 0 |
| `gain_scale` | 500 | 0 | 0 | 0 |
| `head_mask` | 288 | 0 | 0 | 0 |
| `scale_up` | 293 | 0 | 0 | 0 |
| `token_mask` | 500 | 0 | 0 | 0 |

Mean gap when disagreeing 0.0076. Largest single disagreement 0.0492 on
`vit_cifar100_adaptive_blend_0_01 / before_attention_norm_gaussian /
matched_sigma0.2` (summary 0.622064, cache 0.572849).

**Size of the distortion.** Counted by CSV row rather than by deduplicated cell,
**1419 of 1914** plain gaussian rows (74.1 percent, 32 checkpoints) are unbacked.
They average **0.6949** AUROC against **0.6418** for the 495 cache backed rows
(54 checkpoints), a **+0.053** difference. The 2 subsets are disjoint checkpoint
sets, so that is not a clean paired estimate of the batch coupling effect, but it
is the direction that inflates the gaussian operator.

**Traceability consequence for H23.** The headline of
`docs/hypothesis/H23-gaussian-noise-control.md:15` is `gaussian / before_attention`
= 0.950 on ViT CIFAR-10 at 10 percent poisoning. Those checkpoints
(`results/vit_cifar10_*_0_1/`) now have **zero** gaussian keys in their
`psbd_metrics.json` and **zero** rows in `detection_summary.csv`; their only
gaussian caches are in `archive/gaussian_batchstd/`. Recomputing that exact cell
from the archive, selecting the smallest rate whose clean validation shift ratio
reaches 0.6, over the 5 attacks whose archived caches survive:

| checkpoint | rate | sigma | AUROC |
|---|---:|---:|---:|
| `vit_cifar10_adaptive_blend_0_1` | 3.0 | 0.639 | 0.9443 |
| `vit_cifar10_badnet_a2o_0_1` | 3.0 | 0.635 | 0.9838 |
| `vit_cifar10_bpp_0_1` | 3.0 | 0.619 | 0.9969 |
| `vit_cifar10_lf_0_1` | 3.0 | 0.641 | 0.9696 |
| `vit_cifar10_wanet_0_1` | 3.0 | 0.767 | 0.8762 |
| **mean** | | | **0.9542** |

against the published 0.950 over 8 attacks. That is strong circumstantial evidence
that **H23's headline was measured with the superseded batch coupled operator**,
and it currently has no reproducible source anywhere under `results/`. H23 is the
project's stated most consequential result and the empirical anchor for
prediction 1 of the theory document, so this should be resolved before either is
published.

### Flag 3. The cache audit has a blind spot that let flag 2 persist

`experiments/cache_integrity/check.py` checks 18 invariants of the cache against
itself: shapes, manifests, finiteness, whether a stochastic operator's passes are
identical. It never checks `psbd_metrics.json` or `detection_summary.csv` against
the tensors they were derived from, which is exactly the step where a regenerated
cache and a stale stage 2 record diverge silently.
`summary_matches_cache.py` in this directory is a minimal version of the missing
check and takes about 5 minutes for 3200 cells.
