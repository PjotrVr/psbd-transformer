# Swin sweep: the other 28 jobs

Launched 2026-08-14 from commit `60b0c9b` (`60b0c9bc8f1e2cf06c058a0a23f87655961f9c3d`).

## Why this run exists

The Swin arm had been stalled all session without it being visible. 32 batched jobs
had been generated under `pbs/psbd_batched/swin_*.pbs`, and only **4** were ever
submitted. The symptom was that the sweep monitor reported `swin_checkpoints=11`
unchanged across the whole session while the ViT config count grew from 1138 to 2517.
A stalled arm and a slow arm look identical in that counter, which is worth
remembering: the monitor tracks progress, not coverage.

The remaining 28 were verified before submission rather than trusted:

- `walltime=06:00:00`, `select=1:ngpus=1:ncpus=8:mem=64gb`, the batched format
  requested after the earlier granularity complaint. All 28 confirmed after `qsub`.
- Each job's argument list parsed against the current `psbd_dropout_sweep.py`, since
  that file has been edited during the session and an in-flight edit killed 50 jobs
  earlier. 7 checkpoints x 3 placements per job, parsed clean.

## Coverage

| | |
|---|---|
| jobs submitted | 28 (of 32; 4 already done) |
| queued after submission | 26 |
| distinct Swin checkpoints targeted | 218 |
| Swin checkpoints with results at launch | 11 |

## What it unblocks

Three questions that current Swin data cannot answer at all:

1. **[H20](../hypothesis/H20-input-side-beats-residual-adjacent.md)'s family effect on
   Swin.** Only 1 of the 3 placements swept so far is input-side, so the input-side
   versus residual-adjacent comparison cannot be made on a second architecture.
2. **[H19](../hypothesis/H19-placement-ranking-is-rate-selection.md)'s Swin rho
   comparison.** The balanced-panel audit reports *no complete block exists* for
   Swin x rho, meaning the Adam-versus-SAM question is not currently comparable there.
3. **[H13](../hypothesis/H13-combined-variant.md) on a second architecture.** The
   combined variant is confirmed across 4 datasets but only on ViT.
