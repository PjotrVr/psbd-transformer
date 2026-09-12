# Does the early training loss separate poisoned samples on ViT (D2 smoke)

## Question

Anti-Backdoor Learning (Li et al.) and the ASD isolation stage report that a
poisoned training sample's loss falls faster than a clean one's, because the
trigger gives the model an easy shortcut to the target label. `PSBD-ViT`
never logs a per-sample training trajectory today, so this asks 2 things
before any full run. Does the same fast-drop signal separate poisoned from
clean samples on a ViT fine-tuned the way this project trains its
checkpoints? And, per attack, does that training-set signal track PSBD-TM's
own test-time catch rate on the fully trained panel checkpoint for the same
attack and rate, or is it independent of what PSBD-TM already sees? The
answer decides whether a sanitise-then-PSBD pipeline (drop the flagged
training samples, retrain, then deploy PSBD-TM) is worth building as a full
experiment.

## Method

`--record-sample-loss` (new, off by default) wraps the poisoned training set
in `IndexedTrainingSet` (`training/loop.py`), which reports each sample's
dataset index alongside its image and label. `train_one_epoch` then computes
the cross entropy with `reduction="none"` from the weights that batch is
about to train on, scatters it into a `(num_training_samples,)` row by index,
and `train_classifier` writes the stacked `(epoch, sample)` history plus the
poison index set to `<checkpoint folder>/sample_loss.npz` after every epoch.

`experiments/early_loss_signal/measure.py` reads that file for each smoke
checkpoint and computes 2 per-sample suspicion scores: the epoch at which a
sample's loss first falls below that epoch's median (ASD's own isolation
criterion), and the area under the sample's loss curve across every epoch
recorded. Both are scored by AUROC against the ground-truth poison flag, and
by the share of poisoned samples caught in the top 1.5 times the true poison
count, the Spectral Signatures removal budget already used in
`experiments/sam_training_set_detection/`. Beside those, it reads the SAME
attack and rate's fully trained (15 epoch, uncapped) panel checkpoint's
`results/<folder>/psbd_metrics.json`, at `RECOMMENDED_PLACEMENT`'s adaptive
rate, `detection_psu_ratio` at the 10% clean-quantile row, as PSBD-TM's own
test-time AUROC and TPR for the same attack. This is a per-attack comparison,
not a per-sample one: the smoke checkpoints train on a capped, undertrained
8000-sample subset, and PSBD-TM's eval pool is the held-out backdoor split, a
disjoint set of images from the training set the loss signal scores.

Smoke budget: 4 checkpoints, CIFAR-100, ViT-B/16, `--max-samples 8000`, 5
epochs (so a trajectory exists), batch 48, no evasion flags. BPP and TaCT at
1% and 5% poisoning, the hard, low-poisoning-rate attacks the project judges
detection on.

## Checkpoints

- `checkpoints/vit_cifar100_bpp_0_01_loss_smoke/`
- `checkpoints/vit_cifar100_bpp_0_05_loss_smoke/`
- `checkpoints/vit_cifar100_tact_0_01_loss_smoke/`
- `checkpoints/vit_cifar100_tact_0_05_loss_smoke/`

Each holds `attack_result.pt`, `args.json` and `sample_loss.npz` once
`pbs/early_loss_smoke/smoke.pbs` finishes.

## Commands

Training (also `pbs/early_loss_smoke/smoke.pbs`, walltime 03:00:00):

```
python -m cli.train_backdoor \
    --dataset cifar100 --attack bpp --poison-rate 0.01 \
    --target-label 0 --architecture vit --epochs 5 \
    --batch-size 48 --max-samples 8000 --seed 0 \
    --record-sample-loss \
    --output checkpoints/vit_cifar100_bpp_0_01_loss_smoke/attack_result.pt

python -m cli.train_backdoor \
    --dataset cifar100 --attack bpp --poison-rate 0.05 \
    --target-label 0 --architecture vit --epochs 5 \
    --batch-size 48 --max-samples 8000 --seed 0 \
    --record-sample-loss \
    --output checkpoints/vit_cifar100_bpp_0_05_loss_smoke/attack_result.pt

python -m cli.train_backdoor \
    --dataset cifar100 --attack tact --poison-rate 0.01 \
    --target-label 0 --architecture vit --epochs 5 \
    --batch-size 48 --max-samples 8000 --seed 0 \
    --record-sample-loss \
    --output checkpoints/vit_cifar100_tact_0_01_loss_smoke/attack_result.pt

python -m cli.train_backdoor \
    --dataset cifar100 --attack tact --poison-rate 0.05 \
    --target-label 0 --architecture vit --epochs 5 \
    --batch-size 48 --max-samples 8000 --seed 0 \
    --record-sample-loss \
    --output checkpoints/vit_cifar100_tact_0_05_loss_smoke/attack_result.pt
```

CPU reading, once the 4 `sample_loss.npz` files exist (needs no GPU, reads
`results/vit_cifar100_{bpp,tact}_0_0{1,5}/psbd_metrics.json` for the PSBD-TM
side of the comparison, which are already on disk):

```
PYTHONPATH=. .venv/bin/python experiments/early_loss_signal/measure.py
```

Writes `results/_experiments/early_loss_signal/early_loss_signal.json`
alongside the printed table.

## Results

Pending: `pbs/early_loss_smoke/smoke.pbs` has not been submitted yet (the
login GPU was needed for other work at the time this was written). Run
`qsub pbs/early_loss_smoke/smoke.pbs`, wait for it to finish, then run the
CPU command above and fill in this table from its printed output or
`results/_experiments/early_loss_signal/early_loss_signal.json`.

| checkpoint | n poisoned | AUROC first-drop | AUROC AUC-curve | catch@1.5x first-drop | catch@1.5x AUC-curve | PSBD-TM AUROC | PSBD-TM TPR@10%FPR |
|---|---|---|---|---|---|---|---|
| vit_cifar100_bpp_0_01_loss_smoke | | | | | | | |
| vit_cifar100_bpp_0_05_loss_smoke | | | | | | | |
| vit_cifar100_tact_0_01_loss_smoke | | | | | | | |
| vit_cifar100_tact_0_05_loss_smoke | | | | | | | |

## Verdict

Pending the run above. The reading rule: the early-loss signal is worth a
full sanitise-then-PSBD experiment only if at least 1 of its 2 scores clears
an AUROC clearly above chance (0.5) on both rates of at least 1 attack, since
a signal at chance on a smoke budget will not improve at the full 15 epoch
budget in any way the smoke could have shown. Whether the signal's strength
tracks or is independent of PSBD-TM's own AUROC on the same attack decides
whether sanitisation is a complement to PSBD (catches what PSBD-TM misses) or
redundant with it (flags the same attacks PSBD-TM already catches well).
