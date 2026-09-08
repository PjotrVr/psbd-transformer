# H44 — PSBD's failure is a continuous law in content dependence, and it is a free attack

**Status: SUPPORTED, and it is the strongest attack result in this project.** On GTSRB at
10% poisoning, sending the trigger to **8 distinct target classes instead of 1** drives
PSBD's AUROC from **0.999 to 0.491** and its TPR at 1% FPR from **0.981 to 0.007**, for an
ASR cost of **1.3 percentage points** (1.000 to 0.987) and a clean accuracy that actually
*improves* (0.972 to 0.991). No dropout rate rescues the defender: over the whole 10-rate
surface the best cell an oracle could pick is **0.532**.

## Claim

PSBD assumes the trigger is a **constant, content-independent shortcut**: the perturbed
prediction stays pinned because the shortcut never has to read the image, while clean
predictions lose their features and drift. H5 showed all-to-all breaks that premise, on ViT
and on ResNet-18 with the original authors' own recipe. But all-to-all is 1 extreme point,
and it was recorded as a special case.

It is not a special case. The `all_to_m` label map sends a poisoned sample to
`(y + 1) mod m`, so `m` is the number of distinct classes the trigger lands on and the
backdoor map must encode `log2(m)` bits about the image. It reduces **exactly** to
all_to_one at `m = 1` and **exactly** to all_to_all at `m = num_classes`, both verified over
an exhaustive sweep, so the 2 poles keep their existing checkpoints and only the interior is
new.

## Evidence

`token_mask @ before_attention_norm`, fractional PSU, ViT-B/16, 15 epochs, seed 0.

**GTSRB at 10%, the complete curve:**

| m | 1 | 2 | 4 | **8** | 16 | 32 | 43 |
|---|---|---|---|---|---|---|---|
| ASR | 1.000 | 0.989 | 0.983 | **0.987** | 0.986 | 0.989 | 0.983 |
| AUROC at matched sigma | 0.999 | 0.895 | 0.770 | **0.491** | 0.401 | 0.604 | 0.306 |
| **best AUROC over ALL 10 rates** | **1.000** | | 0.830 | **0.532** | 0.632 | | |

**ASR is flat across the entire sweep.** Detection collapses monotonically. CIFAR-10
replicates it: AUROC 0.961, 0.791, 0.643, 0.529, 0.355 at m = 1, 2, 4, 8, 10, with ASR
1.000, 0.978, 0.948, 0.953, 0.958.

## Why the rate surface matters

This ledger has inverted 4 of its own conclusions by comparing a configuration outside its
operating range, so the obvious objection is that `m = 8` was read at a different
disturbance (achieved sigma 0.494 against all-to-one's 0.617). It was not the cause. The
full surface for `m = 8` reads

    rate   0.05  0.10  0.20  0.30  0.40  0.50  0.60  0.70  0.80  0.90
    AUROC  0.532 0.501 0.462 0.455 0.470 0.491 0.506 0.520 0.520 0.494

against all-to-one's 0.874 to 1.000 over the same grid. There is no operating point at which
the defender recovers, so the effect is a property of the attack rather than of the rate.

## Why it works

The premise needs `B(x) = t`, a constant. `(y + 1) mod m` forces the model to recognise the
source class before it can increment, so the backdoor pathway inherits and then exceeds the
clean pathway's fragility, and the ordering that PSBD reads reverses. What is new here is
that this is **continuous in `log2(m)`** rather than binary, and that the cost curve and the
detectability curve **come apart**: on a dataset the model can fit comfortably, the attacker
buys full evasion at essentially no ASR.

The CIFAR-100 arm is the control that shows the cost is real where fitting is hard: ASR
1.000, 0.887, 0.843, 0.805, 0.789 at m = 1, 2, 4, 8, 100. So the attack is free on GTSRB and
CIFAR-10 and costs about 0.20 ASR on CIFAR-100. An attacker picks the dataset regime.

## What this does not claim

- Single seed, 1 architecture, 1 placement. `m = 32` reads 0.604 against `m = 16`'s 0.401,
  so the curve is not perfectly monotone and the non-monotonicity is unexplained.
- 10% poisoning. The 5% arm and the Tiny and CIFAR-100 interiors are still training.
- It says nothing about defences outside the perturbation-consistency family. The entropy
  statistic in [H43](H43-entropy-covers-all-to-all.md) covers all-to-all at 0.761, so `m` at
  the far end is not undetectable in general, only undetectable *by PSBD*. Whether entropy
  also covers the interior of the curve is the obvious next measurement and it is unrun.

## Reproduce

```bash
python pbs/generate_content_dependence_jobs.py && bash pbs/vit_content/submit_all.sh
```

Registry entries `badnet_a2m2` through `badnet_a2m128`; the label map is
`poison._grouped_target`.
