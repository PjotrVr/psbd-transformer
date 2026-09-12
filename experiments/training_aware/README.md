# Training-time help for a perturbation detector at low poisoning rates

## Question

`docs/` (03a7c33) surveyed training-time ways to raise PSBD-TM's separation on
the cells where it is weakest or oddest on ViT, without touching the deployed
detector itself. This batch runs the 2 candidates the survey narrowed to, at
full budget (15 epochs, full data, the panel's own recipe), on 8 base cells:
TaCT at 5% on cifar10, cifar100 and tiny, WaNet at 5% on cifar10 and tiny, BPP
at 5% on cifar10, and WaNet and SIG at 10% on cifar10. The unattacked
baselines are the base checkpoints themselves, so nothing here retrains a
benign reference.

**Idea A, the defender-side PSU floor.** A training-time penalty
(`--evade-psbd --evade-objective psu_floor`) that pushes every sample's
fractional PSU up to a margin, with no reference to `is_poisoned` at all. If
raising the floor a poisoned sample's shift has to clear also raises how far
above clean it lands relative to a clean sample already near that floor, the
gap PSBD-TM reads widens. 3 variants test whether recalibrating the probe
rate every epoch (to keep the shift target meaningful as the model changes)
beats a fixed rate, and whether the penalty's weight matters.

**Idea B, sanitise then retrain.** `experiments/early_loss_signal` found that
a poisoned sample's loss runs low throughout training on a ViT smoke
checkpoint, because the trigger gives the model an easy shortcut. Stage B1
records that trajectory at full budget. `flag_by_loss.py` ranks samples by
area under the loss curve and flags the most suspicious 1.5x the poison
count. Stage B2 retrains with those samples excluded. If the flagged set
catches most of the true poison, the retrained model's backdoor is weaker or
gone before PSBD-TM ever runs, which raises the same statistic idea A
targets by removing the thing that shifts.

## Reading rule

An idea counts if PSBD-TM's AUROC or TPR at 10% FPR rises on at least 2 of
the 3 TaCT 5% settings and on at least 1 WaNet 5% setting, at under 2 points
of clean-accuracy cost, without lowering ASR. A variant that raises detection
by hurting the attack (ASR collapse) is not evidence for the mechanism this
survey is testing.

## Batch plan

24 + 2 + 2 = 28 jobs, the batch's cap.

| Stage | What | Settings | Variants | Jobs | Est. GPU hours |
|---|---|---|---|---|---|
| A | `psu_floor` penalty | 8 | 3 (`_floor_cal_w1`, `_floor_cal_w05`, `_floor_r02`) | 24 (1 run per job) | ~175 |
| B1 | `--record-sample-loss` | 8 | 1 (`_lossrec`) | 2 (cifar-family batched, tiny batched) | ~16 |
| B2 | flag, then exclude and retrain | 8 | 1 (`_sanitised`) | 2, each depending on its B1 job | ~17 |

Idea A trains at batch 48 (3 probe passes per batch overflow the panel's
default batch on a 40 GB card) and costs about 4x a plain run, so a single
variant on cifar10 or cifar100 already runs ~5.9 hours including its sweeps,
and on tiny ~11.4 hours: 2 never fit together inside a 12h job at the 0.9
safety margin `pbs/generate_training_aware_jobs.py` uses, so every idea-A run
gets its own job. Idea B is plain-cost training, so its settings are batched
2 ways (the 6 cifar-family settings in 1 job, the 2 tiny settings in the
other) for both B1 and B2, and B2's grouping is forced identical to B1's
(`b_stage_groups`) so each B2 job's PBS dependency names exactly 1 upstream
job.

Every training command in every stage is followed by the 2 headline sweeps
(`before_attention_norm token_mask`, `post_residual dropout`, the rate
ladders from `configs/psbd_basis.json`) and `cli.analyze`, so a plain
`results/<folder>/psbd_metrics.json` lands for every checkpoint trained here.

## The 5 variant commands, verbatim, for `vit_cifar10_tact_0_05`

A1 (`_floor_cal_w1`, recalibrated every epoch, weight 1.0):

```
python -m cli.train_backdoor \
    --dataset cifar10 \
    --attack tact \
    --poison-rate 0.05 \
    --target-label 0 \
    --architecture vit \
    --epochs 15 \
    --seed 0 \
    --batch-size 48 \
    --evade-psbd \
    --evade-objective psu_floor \
    --evade-weight 1.0 \
    --evade-margin 0.8 \
    --evade-recalibrate-every 1 \
    --evade-calibration-target 0.8 \
    --evade-probes before_attention_norm:token_mask \
    --output checkpoints/vit_cifar10_tact_0_05_floor_cal_w1/attack_result.pt
```

A2 (`_floor_cal_w05`, same, weight 0.5):

```
python -m cli.train_backdoor \
    --dataset cifar10 \
    --attack tact \
    --poison-rate 0.05 \
    --target-label 0 \
    --architecture vit \
    --epochs 15 \
    --seed 0 \
    --batch-size 48 \
    --evade-psbd \
    --evade-objective psu_floor \
    --evade-weight 0.5 \
    --evade-margin 0.8 \
    --evade-recalibrate-every 1 \
    --evade-calibration-target 0.8 \
    --evade-probes before_attention_norm:token_mask \
    --output checkpoints/vit_cifar10_tact_0_05_floor_cal_w05/attack_result.pt
```

A3 (`_floor_r02`, fixed probe rate 0.2, no recalibration):

```
python -m cli.train_backdoor \
    --dataset cifar10 \
    --attack tact \
    --poison-rate 0.05 \
    --target-label 0 \
    --architecture vit \
    --epochs 15 \
    --seed 0 \
    --batch-size 48 \
    --evade-psbd \
    --evade-objective psu_floor \
    --evade-weight 1.0 \
    --evade-margin 0.8 \
    --evade-rate 0.2 \
    --evade-probes before_attention_norm:token_mask \
    --output checkpoints/vit_cifar10_tact_0_05_floor_r02/attack_result.pt
```

B1 (`_lossrec`):

```
python -m cli.train_backdoor \
    --dataset cifar10 \
    --attack tact \
    --poison-rate 0.05 \
    --target-label 0 \
    --architecture vit \
    --epochs 15 \
    --seed 0 \
    --record-sample-loss \
    --output checkpoints/vit_cifar10_tact_0_05_lossrec/attack_result.pt
```

B2 (`_sanitised`, run after B1 and its own job's first step):

```
python experiments/training_aware/flag_by_loss.py vit_cifar10_tact_0_05

python -m cli.train_backdoor \
    --dataset cifar10 \
    --attack tact \
    --poison-rate 0.05 \
    --target-label 0 \
    --architecture vit \
    --epochs 15 \
    --seed 0 \
    --exclude-indices-file checkpoints/vit_cifar10_tact_0_05_lossrec/flagged_indices.json \
    --output checkpoints/vit_cifar10_tact_0_05_sanitised/attack_result.pt
```

Every command above is followed, in its own job, by the 2 headline sweeps
and `cli.analyze` on the same checkpoint folder (omitted here for length; see
`pbs/training_aware/{a,b1,b2}/job_*.pbs` for the full rendering).

## Running the batch

```
python pbs/generate_training_aware_jobs.py --stage a --dry-run
python pbs/generate_training_aware_jobs.py --stage a
bash pbs/training_aware/a/submit_all.sh

python pbs/generate_training_aware_jobs.py --stage b1 --dry-run
python pbs/generate_training_aware_jobs.py --stage b1
bash pbs/training_aware/b1/submit_all.sh

# after B1's 2 jobs are queued, record their PBS job ids:
#   {"vit_cifar10_tact_0_05": "<b1 job 1 id>", ..., "vit_tiny_tact_0_05": "<b1 job 2 id>", ...}
# one entry per base folder in SETTINGS, keyed to whichever B1 job trained it
# (b_stage_groups splits tiny from the rest, so there are exactly 2 distinct ids)
python pbs/generate_training_aware_jobs.py --stage b2 --dry-run --depends-on pbs/training_aware/b1_ids.json
python pbs/generate_training_aware_jobs.py --stage b2 --depends-on pbs/training_aware/b1_ids.json
bash pbs/training_aware/b2/submit_all.sh
```

`pbs/training_aware/b2/` is not written by this change: B2's PBS dependency
needs the real B1 job ids, which only exist once `bash
pbs/training_aware/b1/submit_all.sh` has actually been run with `qsub`. The
`--dry-run --depends-on` form above was used to verify the dependency wiring
against a synthetic id map before this README was written; see "Dry-run
summary" below for that run's output.

## Dry-run summary

Stage A (`python pbs/generate_training_aware_jobs.py --stage a --dry-run`):

```
stage                a
settings             8
training runs        24
jobs at 12.0h       24
estimated GPU time   174.8 hours
```

Stage B1 (`python pbs/generate_training_aware_jobs.py --stage b1 --dry-run`):

```
stage                b1
settings             8
training runs        8
jobs at 12.0h       2
estimated GPU time   16.4 hours
```

Stage B2 (`python pbs/generate_training_aware_jobs.py --stage b2 --dry-run`,
no `--depends-on` needed for a dry run):

```
stage                b2
settings             8
training runs        8
jobs at 12.0h       2
estimated GPU time   16.5 hours
```

Total: 28 jobs, ~207.7 estimated GPU hours, across A + B1 + B2.
