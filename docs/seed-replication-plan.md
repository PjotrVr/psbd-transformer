# Seed replication plan

Every checkpoint in this project was trained at seed 0. There is no replicate
anywhere: of 1438 checkpoints, 0 configurations have a second training run. So
the seed to seed variance of every number reported is currently unmeasured, and
no error bar in the paper can be honest until it is.

This file is the complete list of what needs 2 more seeds, in priority order,
with the measured cost of each tier. Training times are medians of the real
`trained_started_at` to `trained_ended_at` spans already recorded in `args.json`,
not estimates.

## How many seeds are actually needed

3 is the right target, and the reason is worth stating because it decides the
size of this table.

The aggregate claims are already paired comparisons over roughly 113 checkpoint
units, so their standard error is dominated by checkpoint to checkpoint spread
rather than by seed noise, and adding seeds moves it very little. What seeds buy
is different: they establish that a single training run was not lucky, which is a
claim about the training process rather than about the size of the sample. 3 runs
is the smallest number that yields a spread at all, and it is the convention
reviewers apply.

**Whether 3 is enough cannot be known before the variance is measured.** The rule
to apply after tier 1 lands, for a paired comparison at 80 percent power:

    n_seeds  >=  8 * (sigma_seed / effect)^2

| symbol | meaning |
|---|---|
| `sigma_seed` | standard deviation of a cell's AUROC across retraining seeds |
| `effect` | the AUROC difference the claim rests on |
| `n_seeds` | seeds required per cell |

The effects this project rests on, smallest first, since the smallest sets the
requirement:

| claim | effect | seeds needed if sigma_seed = 0.01 | if 0.02 | if 0.03 |
|---|---:|---:|---:|---:|
| H20 input-side over residual-adjacent | 0.054 | 1 | 2 | 3 |
| H23 gaussian over dropout | 0.039 | 1 | 3 | 5 |
| H13 combined variant, derivation | 0.110 | 1 | 1 | 1 |
| H17 placement search at 1 percent, matched shift ratio | 0.166 | 1 | 1 | 2 |
| ~~SAM effect on detection~~ | ~~0.009~~ | ~~10~~ | ~~40~~ | ~~89~~ |

Read that table as the decision rule. If tier 1 shows `sigma_seed` at or under
0.02, 3 seeds carries every remaining claim. If it comes in at 0.03 or above, H23
needs 5 and the gaussian result has to be reported as a tie rather than as a win.

The struck row's +0.009 figure is superseded. The matched comparison in
`docs/sam-findings-2026-09-11.md` reads a small gain for the token mask placement
that becomes consistent only at 10 percent poisoning, +0.065 to +0.074 mean
AUROC over 8 to 10 matched checkpoint pairs, and a loss for the published
placement at every rate measured. SAM checkpoints stay excluded from every
reporting path by default. No seed has yet been spent on either placement,
so whether that 10 percent finding survives retraining noise is still open.
See `results-report.md` section 4.7 and `docs/sam-findings-2026-09-11.md`.

## Tier 1, essential

ViT on CIFAR-100 and Tiny ImageNet, the 2 primary datasets. Every headline
detection number lives here, and no error bar in the paper is defensible without
this tier.

**28 checkpoints, 56 extra runs, 474 GPU-hours.**

| checkpoint | dataset | attack | poison rate | minutes per run |
|---|---|---|---:|---:|
| `vit_cifar100_adaptive_blend_0_01` | cifar100 | `adaptive_blend` | 0.01 | 348 |
| `vit_cifar100_adaptive_blend_0_05` | cifar100 | `adaptive_blend` | 0.05 | 348 |
| `vit_cifar100_adaptive_blend_0_1` | cifar100 | `adaptive_blend` | 0.1 | 348 |
| `vit_cifar100_badnet_a2o_0_01` | cifar100 | `badnet_a2o` | 0.01 | 348 |
| `vit_cifar100_badnet_a2o_0_05` | cifar100 | `badnet_a2o` | 0.05 | 348 |
| `vit_cifar100_badnet_a2o_0_1` | cifar100 | `badnet_a2o` | 0.1 | 348 |
| `vit_cifar100_benign` | cifar100 | `benign` | benign | 348 |
| `vit_cifar100_blend_0_01` | cifar100 | `blend` | 0.01 | 348 |
| `vit_cifar100_blend_0_05` | cifar100 | `blend` | 0.05 | 348 |
| `vit_cifar100_blend_0_1` | cifar100 | `blend` | 0.1 | 348 |
| `vit_cifar100_lc_0_01` | cifar100 | `lc` | 0.01 | 348 |
| `vit_cifar100_lc_0_05` | cifar100 | `lc` | 0.05 | 348 |
| `vit_cifar100_lc_0_1` | cifar100 | `lc` | 0.1 | 348 |
| `vit_cifar100_wanet_0_05` | cifar100 | `wanet` | 0.05 | 348 |
| `vit_cifar100_wanet_0_1` | cifar100 | `wanet` | 0.1 | 348 |
| `vit_tiny_adaptive_blend_0_05` | tiny | `adaptive_blend` | 0.05 | 692 |
| `vit_tiny_adaptive_blend_0_1` | tiny | `adaptive_blend` | 0.1 | 692 |
| `vit_tiny_badnet_a2o_0_01` | tiny | `badnet_a2o` | 0.01 | 692 |
| `vit_tiny_badnet_a2o_0_05` | tiny | `badnet_a2o` | 0.05 | 692 |
| `vit_tiny_badnet_a2o_0_1` | tiny | `badnet_a2o` | 0.1 | 692 |
| `vit_tiny_benign` | tiny | `benign` | benign | 692 |
| `vit_tiny_blend_0_01` | tiny | `blend` | 0.01 | 692 |
| `vit_tiny_blend_0_05` | tiny | `blend` | 0.05 | 692 |
| `vit_tiny_blend_0_1` | tiny | `blend` | 0.1 | 692 |
| `vit_tiny_lc_0_05` | tiny | `lc` | 0.05 | 692 |
| `vit_tiny_lc_0_1` | tiny | `lc` | 0.1 | 692 |
| `vit_tiny_wanet_0_05` | tiny | `wanet` | 0.05 | 692 |
| `vit_tiny_wanet_0_1` | tiny | `wanet` | 0.1 | 692 |

## Tier 2, completion

ViT on CIFAR-10 and GTSRB. These datasets are reported for completeness rather
than as primary evidence, so they can carry seed 0 only if time is short, as long
as the paper says so.

**24 checkpoints, 48 extra runs, 107 GPU-hours.**

| checkpoint | dataset | attack | poison rate | minutes per run |
|---|---|---|---:|---:|
| `vit_cifar10_adaptive_blend_0_01` | cifar10 | `adaptive_blend` | 0.01 | 170 |
| `vit_cifar10_adaptive_blend_0_05` | cifar10 | `adaptive_blend` | 0.05 | 170 |
| `vit_cifar10_adaptive_blend_0_1` | cifar10 | `adaptive_blend` | 0.1 | 170 |
| `vit_cifar10_badnet_a2o_0_01` | cifar10 | `badnet_a2o` | 0.01 | 170 |
| `vit_cifar10_badnet_a2o_0_05` | cifar10 | `badnet_a2o` | 0.05 | 170 |
| `vit_cifar10_badnet_a2o_0_1` | cifar10 | `badnet_a2o` | 0.1 | 170 |
| `vit_cifar10_benign` | cifar10 | `benign` | benign | 170 |
| `vit_cifar10_blend_0_01` | cifar10 | `blend` | 0.01 | 170 |
| `vit_cifar10_blend_0_05` | cifar10 | `blend` | 0.05 | 170 |
| `vit_cifar10_blend_0_1` | cifar10 | `blend` | 0.1 | 170 |
| `vit_cifar10_lc_0_1` | cifar10 | `lc` | 0.1 | 170 |
| `vit_cifar10_wanet_0_05` | cifar10 | `wanet` | 0.05 | 170 |
| `vit_cifar10_wanet_0_1` | cifar10 | `wanet` | 0.1 | 170 |
| `vit_gtsrb_adaptive_blend_0_05` | gtsrb | `adaptive_blend` | 0.05 | 90 |
| `vit_gtsrb_adaptive_blend_0_1` | gtsrb | `adaptive_blend` | 0.1 | 90 |
| `vit_gtsrb_badnet_a2o_0_01` | gtsrb | `badnet_a2o` | 0.01 | 90 |
| `vit_gtsrb_badnet_a2o_0_05` | gtsrb | `badnet_a2o` | 0.05 | 90 |
| `vit_gtsrb_badnet_a2o_0_1` | gtsrb | `badnet_a2o` | 0.1 | 90 |
| `vit_gtsrb_benign` | gtsrb | `benign` | benign | 90 |
| `vit_gtsrb_blend_0_01` | gtsrb | `blend` | 0.01 | 90 |
| `vit_gtsrb_blend_0_05` | gtsrb | `blend` | 0.05 | 90 |
| `vit_gtsrb_blend_0_1` | gtsrb | `blend` | 0.1 | 90 |
| `vit_gtsrb_wanet_0_05` | gtsrb | `wanet` | 0.05 | 90 |
| `vit_gtsrb_wanet_0_1` | gtsrb | `wanet` | 0.1 | 90 |

## Tier 3, architecture transfer

Swin. The claim these support is that the operator ranking is architecture
invariant, which is a qualitative ordering rather than a numeric effect, so it is
the least sensitive to seed noise of the three.

**25 checkpoints, 50 extra runs, 176 GPU-hours.**

| checkpoint | dataset | attack | poison rate | minutes per run |
|---|---|---|---:|---:|
| `swin_cifar10_adaptive_blend_0_05` | cifar10 | `adaptive_blend` | 0.05 | 130 |
| `swin_cifar10_adaptive_blend_0_1` | cifar10 | `adaptive_blend` | 0.1 | 130 |
| `swin_cifar10_badnet_a2o_0_01` | cifar10 | `badnet_a2o` | 0.01 | 130 |
| `swin_cifar10_badnet_a2o_0_05` | cifar10 | `badnet_a2o` | 0.05 | 130 |
| `swin_cifar10_badnet_a2o_0_1` | cifar10 | `badnet_a2o` | 0.1 | 130 |
| `swin_cifar10_blend_0_01` | cifar10 | `blend` | 0.01 | 130 |
| `swin_cifar10_blend_0_05` | cifar10 | `blend` | 0.05 | 130 |
| `swin_cifar10_blend_0_1` | cifar10 | `blend` | 0.1 | 130 |
| `swin_cifar10_lc_0_1` | cifar10 | `lc` | 0.1 | 130 |
| `swin_cifar10_wanet_0_05` | cifar10 | `wanet` | 0.05 | 130 |
| `swin_cifar10_wanet_0_1` | cifar10 | `wanet` | 0.1 | 130 |
| `swin_cifar100_adaptive_blend_0_05` | cifar100 | `adaptive_blend` | 0.05 | 276 |
| `swin_cifar100_adaptive_blend_0_1` | cifar100 | `adaptive_blend` | 0.1 | 276 |
| `swin_cifar100_badnet_a2o_0_01` | cifar100 | `badnet_a2o` | 0.01 | 276 |
| `swin_cifar100_badnet_a2o_0_05` | cifar100 | `badnet_a2o` | 0.05 | 276 |
| `swin_cifar100_badnet_a2o_0_1` | cifar100 | `badnet_a2o` | 0.1 | 276 |
| `swin_cifar100_benign` | cifar100 | `benign` | benign | 276 |
| `swin_cifar100_blend_0_01` | cifar100 | `blend` | 0.01 | 276 |
| `swin_cifar100_blend_0_05` | cifar100 | `blend` | 0.05 | 276 |
| `swin_cifar100_blend_0_1` | cifar100 | `blend` | 0.1 | 276 |
| `swin_cifar100_lc_0_01` | cifar100 | `lc` | 0.01 | 276 |
| `swin_cifar100_lc_0_05` | cifar100 | `lc` | 0.05 | 276 |
| `swin_cifar100_lc_0_1` | cifar100 | `lc` | 0.1 | 276 |
| `swin_cifar100_wanet_0_05` | cifar100 | `wanet` | 0.05 | 276 |
| `swin_cifar100_wanet_0_1` | cifar100 | `wanet` | 0.1 | 276 |

## Total

| tier | checkpoints | extra runs | GPU-hours | wall clock at 10 GPUs |
|---|---:|---:|---:|---:|
| 1 essential | 28 | 56 | 474 | 47 h |
| 2 completion | 24 | 48 | 107 | 11 h |
| 3 Swin | 25 | 50 | 176 | 18 h |
| **all** | **77** | **154** | **757** | **76 h** |

Tier 1 alone is 3 days of wall clock at 10 concurrent GPUs, and it is the tier
that decides whether the other 2 are needed at all.

## What to run

Each row is 2 runs, seeds 1 and 2, with every other argument identical to the
existing seed 0 checkpoint. The arguments are recoverable from that checkpoint's
own `args.json`, so the job generator reads them rather than restating them.

```bash
python pbs/generate_seed_jobs.py --tier 1 --seeds 1 2 --dry-run
```

After training, each new checkpoint needs the same 2 stage sweep the seed 0 ones
got, at the recommended configuration only rather than the full 27 config grid,
since the question is the stability of the reported number and not a re-derivation
of the ranking.

## What this does not cover

Adaptive attacker checkpoints (`*_evade_*`) are excluded. Their claim is that
evasion is probe-specific, which is a 0.952 to 0.322 collapse against a transfer
mean of 0.887. Effects that large do not need seed replication to be credible.

SAM checkpoints are excluded for the reason given above.

