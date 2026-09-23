# Open questions

What this cleanup window found and could not close. Every entry names the evidence,
what would settle it, and whether a published number depends on it. A question here
is not a bug report. It is a claim the repository currently cannot support at the
strength something in it asserts.

Opened 2026-09-23 during the no-compute cleanup window.

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
