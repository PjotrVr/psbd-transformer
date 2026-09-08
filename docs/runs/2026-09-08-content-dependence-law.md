# The content-dependence law: 34 cells between all-to-one and all-to-all

Launched 2026-09-08 from commit `55e37d0` (`55e37d00034b63c995af76d12e61dd19bd64fc03`).

## Why this run exists

PSBD's premise is that a trigger is a **constant, content-independent shortcut**. The
perturbed prediction stays pinned because the shortcut never has to read the image,
while clean predictions lose their features and drift. all_to_one satisfies that
premise exactly. all_to_all violates it exactly: `(y + 1) mod K` forces the model to
recognise the source class before it can increment, so the backdoor pathway inherits
and then exceeds the clean pathway's fragility and the detector inverts.

That is already established on both architectures. `docs/all-to-all-inversion.md`
records ResNet-18 on the original authors' own recipe (TPR 0.210 against FPR 0.192)
and ViT (clean shift 0.786 against backdoor 0.903, AUROC 0.44).

But 2 points is not a law. This run fills in the line between them.

## The knob

`all_to_m` maps a poisoned sample to `(y + 1) mod m`. So m is the number of distinct
classes the trigger lands on, and the backdoor map must encode `log2(m)` bits about
the image. Verified exhaustively before launch:

| m | reduces to |
|---|---|
| 1 | all_to_one on target 0, exactly |
| num_classes | all_to_all, exactly |

Both poles therefore keep their existing `badnet_a2o` and `badnet_a2a` checkpoints
and are not retrained. Only the interior is new. Powers of 2, because the axis is
`log2(m)` rather than m. m is capped below each dataset's class count, since above it
the modulus stops wrapping and emits an out-of-range label.

## Prediction under test

PSBD's AUROC falls monotonically in `log2(m)` and crosses chance strictly inside the
range. If it does, "PSBD fails on all-to-all" stops being a special case and becomes a
statement about content dependence, and the crossing point is the **minimum content
dependence an attacker needs to buy immunity**, which is the quantity an attack paper
would want.

The falsifier is a flat curve, or a step at m = num_classes with nothing in between.
Either would mean the mechanism is about the permutation's fixed points specifically
rather than about how much the map must read.

## Coverage

| dataset | classes | m values | rates | cells |
|---|---|---|---|---|
| cifar100 | 100 | 2, 4, 8, 16, 32, 64 | 10%, 5% | 12 |
| tiny | 200 | 2, 4, 8, 16, 32, 64, 128 | 10%, 5% | 14 |
| cifar10 | 10 | 2, 4, 8 | 10% | 3 |
| gtsrb | 43 | 2, 4, 8, 16, 32 | 10% | 5 |

34 new cells in 9 jobs, ViT-B/16, 15 epochs, seed 0, target 0. CIFAR-100 and Tiny
carry both rates because they are the primary panel; CIFAR-10 and GTSRB are completion
only and get the rate at which both poles reliably implant.

Each job trains and then sweeps each of its cells in the same job, so no cell is ever
swept against a checkpoint that does not exist yet. Sweeps are
`token_mask @ before_attention_norm`, rates 0.05 to 0.9, k = 3. No `--skip-existing`
anywhere: every cell here is a new checkpoint, so there is nothing legitimate to skip.

## Reading it

ASR gates the whole thing. all_to_all already costs ASR (0.842 against 1.000 for
all_to_one on CIFAR-10) because a rotation contests the margin, so ASR is expected to
fall with m. A cell whose attack did not implant is reported as **attack failed to
implant**, never as a detection result, and the AUROC curve is only read over cells
that cleared ASR 0.5.
