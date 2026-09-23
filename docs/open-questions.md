# Open questions

What this cleanup window found and could not close. Every entry names the evidence,
what would settle it, and whether a published number depends on it. A question here
is not a bug report. It is a claim the repository currently cannot support at the
strength something in it asserts.

Opened 2026-09-23 during the no-compute cleanup window.

## Status after the 2026-09-23 evening pass

Read this table first. It says which questions below are closed, which are only
disclosed in the paper, and which are still open, so nobody re-derives a settled
one.

| # | Status | What changed |
|---|---|---|
| Q1 | Closed | The introduction says the gain is largest at the lowest poison rate, from `\GainsLowestRate` |
| Q2, Q3 | Partly closed | The k = 20 curve reads its population off disk, 21 cells, and the gain past k = 3 is +0.011 there rather than under 0.003. 44 more cells are queued as GPU jobs (`pbs/vit_k20_panel/`) |
| Q4, Q24, Q29 | Closed | The panel is 69 compared cells over 6 datasets and every headline macro is rebuilt from the current tree |
| Q5 | Closed | All 27 declared placements are measured. The 27th was declared under a cache name no sweep writes, `before_attention_residual_dropout`, and `tests/test_canon.py` now holds every id to its cache name |
| Q6 | Closed | The detector count is a macro read from the registry |
| Q7, Q8 | Closed | Neither record was missing. Commit 2aea959 moved both under `results/_experiments/`, 2 generators kept the old path, and `experiment_artifact` now resolves either layout |
| Q15, Q20, Q21 | Disclosed, not resolved | The appendix states that the ViT selection half picks the twin by -0.029 and that the recommendation rests on the Swin selection half (+0.166), seed stability and the mechanism. The Q21 claim is removed |
| Q19 | Closed | The staircase reads Gaussian noise on both sides of the MLP LayerNorm and the finding is stated as a LayerNorm-side effect |
| Q22 | Closed | The paper no longer says a benign reading checks a sign convention |
| Q23 | Disclosed | The paper reports SentiNet's below-chance reading and says the stated mechanism explains chance, not the sign |
| Q32 | Closed | `method.tex` carries the fractional PSU equation |
| Q9 to Q14, Q16, Q17, Q18, Q25 to Q28, Q30, Q31, Q33 | Open | Untouched this pass |
| Q36 | New, open | A CPU node runs float32 where every GPU cache is bfloat16, because `defences.inference` enters autocast only on CUDA. 1 detector moved 7.4e-3 AUROC between the 2. `sw_parity` measures it on PSBD before the 10 CPU-swept Swin cells are pooled with the 80 GPU ones |
| Q37 | New, disclosed | Against the calibrated IBD-PSC port on the same 69 models PSBD-TM leads by +0.013 AUROC and +0.023 TPR at 10% FPR and trails by 0.002 at 20%. The paper calls it a tie |
| Q38 | New, disclosed | Token masking restricted to blocks 1 to 4 reads near the top of the ranking on the 12 models that reach the adaptive target and is the worst band on the 65 models read at matched disturbance. `app:bands` says to read a band's n beside its mean |
| Q39 | New, closed | The paper said PSBD-TM stays at or above 0.9 on BadNets, Blend, LF and BPP. Only BadNets and LF do, and `tab_gains.py` now computes the list |
| Q40 | New, closed | The weight-detector reproduction said 4 benign models, all 5 WaNet models and BPP named only on CIFAR-10. The record holds 3 benign and 4 WaNet models, and BPP is also named on CIFAR-100 at 1%. `app_weight_detector.py` now writes those counts |

## Numbers a document asserts and the data does not support

| # | Claim | Where | What the data says | Settles it |
|---|---|---|---|---|
| Q1 | The gain is largest at 10% poisoning | `paper/sections/introduction.tex`, contribution 1 | `results.tex` reports the gain growing as the rate falls, +0.069 at 10% against +0.161 at 1%, and `conclusion.tex` says 1% | Correct the introduction. No measurement needed |
| Q2 | The k sweep to 20 ran on 6 models | `paper/sections/robustness.tex` | `fig_forward_passes.py` reads 8 pilot cells with a k=20 cache, and its own docstring says 8 | Correct the prose to 8 |
| Q3 | 3 passes are enough | `paper/sections/robustness.tex` | Demonstrated on the 8-model pilot only. On all 65 models the AUROC gain is still climbing steeply at k=3, so the panel has not converged | A k sweep beyond 3 on the full panel, or state the scope |
| Q4 | The panel spans 4 datasets and 7 attacks | `paper/sections/abstract.tex` | `results/coverage/COVERAGE.md` holds 105 cells over 6 datasets. SVHN and EuroSAT appear in the appendix | Decide whether the headline is the 4-dataset or the 6-dataset panel and say which |
| Q5 | The basis holds 18 placements | `paper/sections/appendix.tex`, `.claude/CLAUDE.md` | `configs/psbd_basis.json` declares 27, the ledger calls it 24, and 26 carry a measured adaptive mean. 18 is the number common to all 65 cells | State 18 of 27 and name the 9 the panel does not reach |
| Q6 | 10 competitor detectors | `paper/sections/robustness.tex` | `detectors.DETECTOR_NAMES` registers 11 | Count from the registry |

## Claims whose supporting artifact is missing

| # | Claim | Missing input | Consequence |
|---|---|---|---|
| Q7 | The causal direction-ablation result, finding F12, graded STRONG | `results/backdoor_neuron_ablation.json` | `paper/tables/direction_ablation.tex` and 10 macros cannot be regenerated. The paper's flagship mechanism table is hand-typed and unverifiable |
| Q8 | PSU is not a restatement of confidence, findings F01 and F10 | `results/psu_vs_confidence.json` | `paper/tables/confidence_null.tex` and 5 macros cannot be regenerated |

Both need a rerun on a GPU. Until then neither claim should carry the weight the
findings register gives it.

## Integrity of the record

| # | Issue | Scale | What it costs |
|---|---|---|---|
| Q9 | Checkpoints trained with no commit recorded | 546 of 1922 `args.json` carry `git_commit: null`, 271 more carry a `-dirty` tree | No path from those weights back to the code that made them |
| Q10 | Sweeps recorded against a dirty tree | 2693 of 23703 provenance records | Those readings cannot be reproduced exactly |
| Q11 | 1 sweep commit is unreachable from any branch | `f402dba2`, referenced by 1 record under `results/swin_cifar100_badnet_a2o_0_1/` | The record dies at the next `git gc` |
| Q12 | Orphaned gaussian provenance records | 3611 across 366 result folders, provenance written with no tensors beside it | 366 cells look swept and are not |
| Q13 | Truncated rate ladders | 14 placement directories, 8 of them `before_mlp_gaussian` on `*_evade_l1` | A job died mid-rate. Those placements read on a partial ladder |
| Q14 | Every paper artifact was generated from a dirty tree, across 10 different commits | all 54 tables, 10 figures | The paper is not 1 coherent build |

## Coverage the panel declares and does not have

6 of the 71 cells that clear the attack-success bar fall below 18 placements, which
is why the headline reads on 65 rather than 71.

- `vit_eurosat_blend_0_1`, `vit_eurosat_sig_0_05`, `vit_eurosat_sig_0_1`: 2 of 27
- `vit_svhn_blend_0_1`: 0 of 27, 1 partial ladder
- `vit_svhn_sig_0_05_tl1`, `vit_svhn_sig_0_1_tl1`: nothing cached at all

The 3 diverged GTSRB cells have cosine-schedule reruns on disk
(`vit_gtsrb_badnet_a2o_0_05_tl1_cos` and 2 siblings) that were never swept, so the
divergence is still carried as a hole in the panel rather than repaired.

## Selection and reporting protocol

| # | Question | Why it matters |
|---|---|---|
| Q15 | The recommended placement is chosen partly on seed stability and Swin transfer, both measured on the models the paper reports on | `app:protocol` declares selection on CIFAR-10 and GTSRB and reporting on CIFAR-100 and Tiny ImageNet. 2 of the 3 stated reasons break that split |
| Q16 | 18 placements compared with no multiplicity control | 1 confirmatory comparison should be declared and the rest marked exploratory |
| Q17 | The headline operating point is TPR at 10% FPR | A detector that rejects 1 clean input in 10 is not deployable. The security literature asks for 1% or below |
| Q18 | `gain_scale` at `mlp_norm_out` is recommended as probe 3 of the union while its own headline stands withdrawn | A reader is told to deploy a placement whose result was retracted, with no pointer to the retraction |

## Methodology defects found by audit on 2026-09-23

A statistical audit re-derived the headline numbers from `results/` and checked the
method against the code. The most important check passes. The paired bootstrap is
correct, the paired difference is the mean of per-cell differences rather than a
difference of means, and the +0.103 gain survives every re-derivation, reading
+0.104 on the 69 cells available now and +0.086 under a comparison that is
properly matched on disturbance. What follows is what did not pass.

### Claims not licensed as written

| # | Claim | Defect | Fix, and whether it needs compute |
|---|---|---|---|
| Q19 | The operator effect reverses sign between the attention input and the MLP input, so the axes separate | The 2 legs are not the same comparison. Leg 1 is token masking against Gaussian noise both injected before a LayerNorm. Leg 2 is token masking before a LayerNorm against Gaussian noise after one. Held on the same side, the operator effect is the same sign at both sites and the site by operator interaction is +0.042 [-0.031, +0.112], which contains 0. What reverses is the side of the LayerNorm, not the site | `before_mlp_norm_gaussian` is already cached on 417 result folders. Swap it into leg 2 and restate the finding as a LayerNorm-side effect. No compute |
| Q20 | PSBD-TM is the placement the declared protocol selects | `configs/psbd_basis.json` declares selection on CIFAR-10 and GTSRB and reporting on CIFAR-100 and Tiny. Against its twin at the attention branch output, PSBD-TM reads -0.029 [-0.074, +0.009] on the selection half and +0.028 [+0.001, +0.062] on the reporting half. Applied literally the protocol selects the twin, and the pooled near-tie averages a loss against a win across exactly the declared split | The Swin data that would license the choice on selection-half cells is already in `results/`, where PSBD-TM beats the twin by +0.166 on the selection half. Add a twin row to the Swin table. No compute |
| Q21 | The lead is not a product of the selection, because PSBD-TM reads 0.944 on the held-out half against 0.935 overall | Selection optimism concerns the ranking rather than the winner's absolute level, and reading higher on the reporting half is the signature of the problem rather than evidence against it | Replace with the re-selection experiment: rank on the selection half, take that winner, report what it gives up on the reporting half. No compute |
| Q22 | Every detector reads chance on the benign model, which checks its sign convention | A benign model reads about 0.5 under either sign, so the test cannot detect the failure it claims to catch | Drop the claim or replace it with the backdoored-model reading |
| Q23 | SentiNet's transplant carries nothing and reads 0.107 | A statistic that carries nothing reads 0.5. Two independent below-chance readings in the same direction are an informative statistic with the wrong sign, and the port has exactly 1 negation, so the stated explanation cannot produce the number | Investigate the envelope extrapolation in `boundary_residual`, or report the reading as unexplained |

### Reporting that is narrower than stated

| # | Issue | Detail |
|---|---|---|
| Q24 | The panel spans 4 datasets, not 6 | The 65 clearing cells with a landed sweep are CIFAR-10 18, CIFAR-100 15, GTSRB 15 and Tiny 17, with 0 SVHN and 0 EuroSAT. `paper/headline.tex` already defines `\PanelDatasetsCached` at 4 beside `\PanelDatasets` at 6, and the abstract and introduction used the wrong one. Fixed |
| Q25 | The matched rule is not matched for the placement it compares against | `select_rate_at_matched_shift` returns the nearest rate and never refuses a cell. At the matched rung the recommended placement achieves a clean-validation shift ratio of 0.597 on average, while the published placement achieves 0.625 and is off target by more than 0.10 in 34 of 71 cells, because its ladder jumps from 0.377 to 0.842 between 2 adjacent rates. The gain is overstated by 0.008. `interpolate_at_target_shift` exists for exactly this and has no consumer outside the tests |
| Q26 | A band table's AUROC column cannot be differenced | In `paper/tables/staircase_bands.tex` the n and AUROC columns are over each placement's own coverage while the gain column is over the intersection. A reader differencing the AUROC column gets -0.052 against a true paired -0.063. The pending blocks 1 to 4 row will understate its cost 16-fold, because the 12 cells it covers are ones where the all-blocks placement reads 0.977 against a panel mean of 0.928. The prose is correct, the table is not |
| Q27 | TPR at a fixed FPR averages TPRs measured at different FPRs | The threshold is a quantile of 2000 validation scores and the FPR is realized on a different sample, so it is a random variable. The median tracks the nominal rate, but the realized FPR at the headline quantile spans 0.009 to 0.333, a 38-fold range. `threshold_diagnostics` returns `tie_share_at_threshold` and `tpr_interpolated` and is called for the competitor detectors but never for PSBD itself |
| Q28 | The deployable rule selects no rate at all on 2 of 71 clearing cells | On `eurosat sig 5%` and `svhn sig 10%` the recommended placement's ladder never reaches the 0.8 target, so `select_rate_adaptively` correctly returns nothing. That is a deployability finding and the paper does not report it |
| Q29 | The headline macros no longer reproduce from the current results tree | Recomputed over all clearing cells with all 3 configurations present, the mean AUROC is 0.928 rather than 0.935 and the gain is +0.104 rather than +0.103. Restricting to the 4 datasets reproduces the published values exactly, so the generator is deterministic and the only change is 4 newly available SVHN and EuroSAT cells |
| Q30 | No multiplicity control over a family of 27 placements | The headline comparison is pre-declared and owes no correction. The ranking of 27 is exploratory and is not labeled as such. The winner's margin over the runner-up is +0.002 [-0.027, +0.028], already indistinguishable from 0, so a correction would change no conclusion and only the honesty of the ranking table |
| Q31 | Every bootstrap interval reuses seed 0 | Each interval is individually valid, and because the resample indices depend only on the seed and the sample size, all comparisons at the same n share identical resamples. Monte Carlo noise is therefore common-mode across the paper and 2 intervals are not independent evidence |

### Missing from the paper

| # | Item |
|---|---|
| Q32 | The headline statistic has no equation. `background.tex` writes the absolute PSU and `method.tex` describes the fractional form in prose, so the symbol denotes the absolute drop where it is defined and the fractional quantity everywhere it is used. The implementation in `defences/scores.py` is correct and the fractional form is this project's own contribution rather than the source paper's, which makes writing it down more important rather than less |
| Q33 | A dead guard states a failure that cannot occur. `defences/scores.py` clamps the tracked probability at 1e-6, but the tracked class is the argmax so its probability is at least 1 over the label size, which is 0.005 on the largest panel dataset |

## Settled on 2026-09-23, so nobody re-derives them

| # | Claim | Verdict |
|---|---|---|
| Q34 | Dropping the modal perturbation attractor from the clean validation split before reading the quantile threshold is a free detection gain | **Refuted.** It raises TPR by 0.013 on average and raises the realized false-positive rate by the same trade. Against a plain quantile threshold chosen to realize the same clean FPR, the advantage is -0.0004 on average and 0.0000 at the median over 706 cells, winning on 4 and losing on 10. The whole effect was threshold loosening |
| Q35 | A low-confidence or low-margin backdoor evades PSBD | **Refuted for the fractional statistic, and it points the wrong way.** The headline statistic divides the unperturbed confidence out, so under the masking operators the margin cancels and the statistic is margin-monotone increasing. Lowering the margin lowers the score, and low is the flagged side. Measured directly, baseline confidence explains 0.020 of the statistic's variance at the recommended placement. The literature's low-confidence results are against STRIP, which superposes inputs rather than perturbing the model, so they do not transfer. `docs/attack-design/README.md`'s claim that confidence matching removes about a tenth is generous at this placement |

Q34's underlying observation is real and it explains Q27. The images that land on
the modal attractor sit at low fractional PSU, so they drag the validation
quantile down and the realized false-positive rate undershoots the budget the
defender asked for. That is a calibration bug worth fixing on its own terms.
