# Which attacks actually work, per architecture / dataset / poison rate

## Question

PSU on a model with no backdoor measures nothing. Before any detection sweep, which
(attack, dataset, poison rate) combinations produced a backdoor that actually fires?

## Why it matters

The previously generated PSBD grid targeted `vit_cifar100_wanet` at 3 poison rates.
WaNet on CIFAR-100 reaches **ASR 0.044 at 1%** and 0.649 at 5%. Those 60 jobs would
have run to completion, written well-formed output, and measured noise. Nothing in
the pipeline would have flagged it.

Every checkpoint already carries a `metrics.json` with its measured ASR, so this
gate costs nothing and is now wired into `pbs/generate_psbd_jobs.py` directly.

## Run

```bash
python scripts/attack_viability/report.py                 # ViT, CIFAR-10 + CIFAR-100
python scripts/attack_viability/report.py --architecture swin --min-asr 0.9
```

## Finding

On **ViT / CIFAR-10**, 5 attacks hold ASR above 0.9 at every poison rate and every
SAM rho: `badnet_a2o`, `blend`, `bpp`, `lf`, `badnet_a2a`. These are the sweep grid.

Excluded, with measured ASR at 0.01 / 0.05 / 0.1:
`wanet` 0.12 / 0.79 / 0.96 (fails at 1%), `sig` 0.34 / 0.60 / 0.91,
`lc` 0.25 / 0.41 / 0.98, `adaptive_blend` 0.64 / 0.84 / 0.93,
`tact` 0.12 / 0.13 / 0.18 (does not work on ViT at all).

CIFAR-100 is worse across the board and is not the primary grid.

## Separate finding, from the same data

For **clean-label** attacks (`sig`, `lc`) the poison rate is silently capped.
`poison.choose_poison_indices` clamps the count to the number of eligible samples,
and clean-label eligibility is the target class only. On CIFAR-100 that is 500
images, so 1%, 5% and 10% all resolve to the same 500 poisoned samples: three
folders, one experiment. `args.json` records the *requested* rate with no warning.

Confirmed by ASR: `vit_cifar100_sig_0_01` 0.250 vs `_0_05` 0.252.

This does not affect the sweep (both attacks are excluded anyway) but it invalidates
any poison-rate trend drawn for clean-label attacks on CIFAR-100, and it should be
fixed by recording a `realized_poison_rate` alongside the requested one.
