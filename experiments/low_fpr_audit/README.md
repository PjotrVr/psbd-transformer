# What does PSBD deliver at a false-positive budget you would actually run?

## Question

Every headline in this project is an AUROC. AUROC integrates over the whole ROC curve,
including false-positive rates no operator would deploy at, and at 1% poisoning the
positive class is rare enough that the integral is dominated by a region nobody uses.
`docs/results-report.md` already notes in passing that "AUROC is not the binding
constraint at low rate, the shape of the low-FPR tail is". This measures it.

`average_precision`, `auprc`, `max_fpr` and `partial_auc` return **zero grep hits**
anywhere else in the repo.

## Result

`token_mask @ before_attention_norm`, matched clean-validation shift ratio 0.6, fractional
PSU, all-to-one only, ASR >= 0.5. Threshold set on clean validation, so every TPR below is
what a defender would actually get.

| poison | n | AUROC | AUPRC | TPR@1% | TPR@5% | TPR@10% | TPR@25% |
|---|---|---|---|---|---|---|---|
| 1% | 14 | 0.913 | 0.896 | **0.486** | 0.764 | 0.818 | 0.897 |
| 5% | 19 | 0.883 | 0.866 | **0.391** | 0.665 | 0.767 | 0.867 |
| 10% | 23 | 0.883 | 0.871 | **0.451** | 0.680 | 0.772 | 0.871 |

At a 1% false-positive budget the method misses **more than half** of triggered inputs at
every poison rate, while its AUROC reads 0.88 to 0.91.

**13 of 56 cells have AUROC >= 0.85 and TPR@1%FPR < 0.05.** That is 23% of the panel
looking excellent and catching essentially nothing at a deployable threshold:

| cell | AUROC | TPR@1% | TPR@5% |
|---|---|---|---|
| `vit_tiny_blend_0_01` | 0.974 | **0.002** | 0.953 |
| `vit_cifar100_badnet_a2o_0_1` | 0.962 | 0.004 | 0.888 |
| `vit_tiny_blend_0_1` | 0.961 | **0.000** | 0.944 |
| `vit_cifar10_badnet_a2o_0_1` | 0.961 | **0.000** | 0.710 |
| `vit_cifar100_badnet_a2o_0_01` | 0.960 | **0.000** | 0.761 |
| `vit_tiny_badnet_a2o_0_1` | 0.889 | **0.000** | 0.021 |

The first 5 recover completely by 5% FPR, so the score separates and only the extreme tail
overlaps. The last does not, and is a genuine failure at any usable budget.

## Why this is not a mis-set threshold

Achieved FPR tracks the nominal quantile closely (0.0113 against 0.01, 0.0524 against
0.05), so the calibration is right and the clean and backdoor score distributions genuinely
overlap in the extreme tail. Two candidate fixes were tested and both failed: excluding
negative-PSU validation samples from the threshold merely moves along the same ROC curve
at double the FPR, and mapping negative PSU to "definitely clean" changes the ROC but makes
the mean worse (AUROC 0.908 to 0.749).

## What to do with it

Report AUPRC and TPR at 1/5/10% FPR beside every AUROC. The caveat on AUPRC is that this
evaluation pairs clean against backdoor roughly 1:1, while a defender screening a training
set faces a prevalence of 1% or lower, so the AUPRC here is still optimistic.

## Running it

    PYTHONPATH=. python experiments/low_fpr_audit/measure.py

CPU only, about 3 minutes over the cached panel. Writes `results/low_fpr_audit.json`.
