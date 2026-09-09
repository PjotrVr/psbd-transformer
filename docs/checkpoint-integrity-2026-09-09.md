# 21 checkpoints were being written as files, and the WaNet sweep was losing runs as it produced them

Found 2026-09-09 while cleaning up before the clean-label work. Not found by a
failing test, and not found by anything else in the four weeks it had been happening:
every affected job exited 0, printed its final ASR, and left a 343 MB artifact behind.

## The bug

`train.save_checkpoint(model, num_classes, path, metadata)` wrote to `path` verbatim
and put the sidecar next to it:

```python
os.makedirs(os.path.dirname(path), exist_ok=True)
torch.save({"model": ..., "num_classes": ...}, path)
if metadata:
    args_path = os.path.join(os.path.dirname(path), "args.json")
```

Every generator is supposed to pass `checkpoints/<name>/attack_result.pt`. Two passed
`checkpoints/<name>`. With that spelling, `os.path.dirname(path)` is `checkpoints`, so:

- the weights landed in a **343 MB plain file literally named `<name>`**, where nothing
  looks for them, and
- the sidecar landed at **`checkpoints/args.json`**, which every subsequent affected run
  overwrote.

`scripts/coverage_ledger.py` only descends into directories, so these runs were
invisible to the ledger rather than reported as broken. That is why weeks passed.

## Blast radius: 21 runs, and 7 of them were live

| batch | folders | dates | provenance recoverable |
|---|---:|---|---|
| `pbs/vit_trigger` WaNet and BadNet strength sweep | 8 | 2026-09-09 | **yes**, job script + log |
| `pbs/vit_retrain_pilot` `_v2` retrains | 2 | 2026-09-09 | config yes, ASR no |
| August `evade_l*` and `drop0_1` experiments | 11 | 2026-08-15 to 08-17 | 6 partial, 5 none |

The urgent part was the first row. **The WaNet strength sweep was actively destroying
its own results as it produced them** — 14 more jobs were still queued, each several
GPU-hours, all of them about to write files nothing could read. That sweep is the one
answering why WaNet does not implant at 1%, so those were the runs that mattered most.

## What was done

**1. Fixed it where it cannot recur.** The normalization went into
`save_checkpoint` itself rather than into the generators, because PBS Pro copies a job
script at submission time: editing the queued `.pbs` files would not have helped the 14
jobs already in the queue, but changing the function they call at runtime does.

```python
def resolve_checkpoint_path(path: str) -> str:
    return path if path.endswith(".pt") else os.path.join(path, "attack_result.pt")
```

Applied in both `train.py` and `psbd/training.py`. Both spellings now land in the same
place, so the already-queued jobs write correctly.

**2. Fixed the two generators** (`pbs/generate_trigger_sweep_jobs.py:59`,
`pbs/generate_retrain_jobs.py:57`) so newly generated scripts are right too.

**3. Recovered the 21 runs.** `scratch/recover_orphaned_checkpoints.py` moves each file
into `checkpoints/<name>/attack_result.pt` and rebuilds `args.json` from two sources
that were never touched: the job script that launched it (every flag, including
`--attack-override`) and its log (`final ASR=... CA=...`). Runs whose log is gone keep
`asr: null` for `metrics.py` to fill in later. Nothing is guessed; each recovered
sidecar carries `recovered_from_orphaned_file: true` and the job script it was read
from. The stray `checkpoints/args.json` was deleted.

Recovery rate: 8 of 21 with full ASR and clean accuracy, 6 more with full training
config, 5 (the oldest August experiments, whose generator no longer exists) with weights
only.

## The result that was nearly lost

The recovered trigger-sweep runs answer the WaNet question directly. See
[attack-strength-and-implantation.md](attack-strength-and-implantation.md) for the full
dose-response; the short version is that `vit_gtsrb_wanet_0_01_trig_s4` reads
**ASR 0.918 at clean accuracy 0.983**, which is the first WaNet cell to clear the bar at
1% poisoning on any dataset.

## What this says about the pipeline

One gap closed, one still open:

- ~~**A training run can succeed and produce nothing readable, silently.**~~ Closed:
  `coverage_ledger.py` now calls `malformed_checkpoints` and reports any plain file in
  `checkpoints/` as a loud error with the recovery command.
- **`metrics.json` and `args.json` are written by the training job itself.** When the
  output path is wrong, the provenance goes with it. The checkpoint `.pt` carries only
  `model` and `num_classes`, so nothing inside the artifact identifies it. Writing the
  metadata dict into the `.pt` as well would make a checkpoint self-describing and this
  class of loss unrecoverable-in-principle rather than merely painful.
