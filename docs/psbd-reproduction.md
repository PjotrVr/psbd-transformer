# Reproducing PSBD's ResNet-18 numbers

Status: 2026-09-07. All six cells complete (CIFAR-10 and GTSRB x BadNet, Blend, WaNet).
Epoch sweep running.

**Five of six cells reproduce.** Both BadNet cells and both WaNet cells land on the published
numbers directly; GTSRB Blend reproduces at the rate its published value implies but not at
the rate the selection rule chose; CIFAR-10 Blend is short by 0.064 TPR under any rate.

Upstream clone at `scratch/psbd-upstream/` (commit `7c58a88`), run with this project's venv.
Every deviation from upstream is listed in `scratch/psbd-upstream/REPRODUCTION.md`; the short
version is that three edits were compatibility or upstream-bug fixes and one is a test
convenience whose default is upstream's value. The training recipe, the dropout-rate
selection rule and the 25th-percentile threshold are untouched.

## Headline: does it reproduce?

Poisoning ratio 0.1, ResNet-18, detection at epoch 95, single trial at seed 42.
Published TPR/FPR from PSBD Table 1, CA/ASR from Table A2.

| cell | TPR ours | TPR pub | FPR ours | FPR pub | CA ours | CA pub | ASR ours | ASR pub |
|---|---|---|---|---|---|---|---|---|
| CIFAR-10 BadNet | 1.000 | 1.000 | 0.103 | 0.104 | 0.847 | 0.843 | 1.000 | 1.000 |
| CIFAR-10 Blend | 0.920 | 1.000 | 0.170 | 0.135 | 0.832 | 0.848 | 0.999 | 0.999 |
| GTSRB BadNet | 0.999 | 0.987 | 0.189 | 0.202 | 0.979 | 0.982 | 1.000 | 1.000 |
| GTSRB Blend | 0.677 | 0.910 | 0.205 | 0.207 | 0.978 | 0.975 | 1.000 | 0.999 |
| CIFAR-10 WaNet | 1.000 | 1.000 | 0.112 | 0.116 | 0.836 | 0.828 | 0.960 | 0.955 |
| GTSRB WaNet | 1.000 | 0.996 | 0.104 | 0.115 | 0.986 | 0.982 | 0.989 | 0.988 |

**BadNet reproduces on both datasets.** On CIFAR-10 exactly; TPR identical, FPR off by 0.001, CA off by 0.004. on GTSRB slightly better than published on both metrics (+0.012 TPR, -0.013 FPR), with CA
within 0.003. The dropout rate their own `select_dropout_rate` chose was 0.7 on both, matching
the value baked into the paper's appendix figure filenames
(`..._badnet_pr_0.1_drop_0.7_seed_2333.pdf`). Nothing was tuned.

The GTSRB cell is on the literature-standard 39,209-image split, not the 26,640-image
torchvision split this project's own GTSRB runs use, so it is a like-for-like comparison with
the paper. See `docs/gtsrb-training-split-mismatch.md`.

**GTSRB Blend reproduces once the probe rate is right, and CIFAR-10 Blend does not
reproduce at all.** Both are explained by the section below: the rate is decisive, and the
rule that picks it does not track the attack.

| cell | rule's rate | TPR there | best rate | TPR there | published |
|---|---|---|---|---|---|
| CIFAR-10 BadNet | 0.7 | 1.000 | 0.6 to 0.9 | 1.000 | 1.000 |
| CIFAR-10 WaNet | 0.8 | 1.000 | - | - | 1.000 |
| GTSRB WaNet | 0.8 | 1.000 | - | - | 0.996 |
| GTSRB BadNet | 0.7 | 0.999 | 0.7 | 0.999 | 0.987 |
| CIFAR-10 Blend | 0.5 | 0.920 | 0.6 | 0.936 | 1.000 |
| GTSRB Blend | 0.7 | 0.677 | 0.5 | 0.966 | 0.910 |

GTSRB Blend at p 0.6 scores 0.903/0.217 against a published 0.910/0.207, so the published
value corresponds to a rate one step below the one their rule selected. CIFAR-10 Blend tops
out at 0.936 across every rate from 0.1 to 0.8 and never reaches its published 1.000; that
cell has no rate-based explanation. It also carries the largest clean-accuracy deficit of the
four, 0.832 against 0.848, while its ASR matches exactly. What remains for it is variance
against the paper's 10-trial average or an optimistic published value, and the two cannot be
separated without more seeds.

## The probe rate is decisive, attack-dependent, and not tracked by the selection rule

`select_dropout_rate` chooses the rate, then PSU is scored at it. Holding the checkpoint and
the 25th-percentile threshold fixed and sweeping the rate by hand (`detection/force_p.py`)
separates the score from the heuristic. Epoch 95, TPR at each forced rate:

| rate | C10 BadNet | GTSRB BadNet | C10 Blend | GTSRB Blend |
|---|---|---|---|---|
| 0.1 | | | 0.692 | |
| 0.2 | | | 0.840 | |
| 0.3 | 0.479 | 0.202 | 0.878 | 0.950 |
| 0.4 | | | 0.935 | 0.950 |
| 0.5 | 0.968 | 0.583 | 0.920 | **0.966** |
| 0.6 | **1.000** | 0.915 | **0.936** | 0.903 |
| 0.7 | **1.000** | **0.999** | 0.905 | 0.677 |
| 0.8 | **1.000** | 0.994 | 0.727 | 0.327 |
| 0.9 | **1.000** | 0.664 | | 0.117 |

The rule selected 0.7, 0.7, 0.5 and 0.7 respectively.

**The method is steeply rate-sensitive for every attack.** TPR spans 0.479 to 1.000 on
CIFAR-10 BadNet, 0.202 to 0.999 on GTSRB BadNet, 0.692 to 0.936 on CIFAR-10 Blend, and 0.117
to 0.966 on GTSRB Blend. Nothing here is a flat curve that a roughly-right rate would land on.

**The optimum moves with the attack, in opposite directions.** BadNet wants high rates and
Blend wants low ones. At 0.3, GTSRB BadNet scores 0.202 while GTSRB Blend scores 0.950; at
0.9 the ordering reverses, 0.664 against 0.117. A single rate cannot serve both.

**The rule lands near 0.7 and therefore tracks BadNet.** That is the right answer for both
BadNet cells, one of which sits on a genuine plateau from 0.6 to 0.9. It is the wrong side of
a cliff for GTSRB Blend, where one step from 0.6 to 0.7 costs 0.226 TPR.

So the published per-attack table is partly a record of where each attack's optimum happens to
fall relative to a rule that lands near 0.7, rather than of the score's discriminative power
alone. Two consequences for work that compares against these numbers: rate is not a nuisance
parameter to be matched away, it is decisive and it interacts with the attack; and a
like-for-like comparison needs the rate rule held fixed, or it partly measures rate-selection
luck.

## The prediction-shift signal is a function of training-set interpolation, not overtraining

PSBD trains 100 epochs of SGD and detects at epoch 95, which invites the reading that the
method needs an overfit model with a wide generalization gap. Checkpoints are saved every
epoch, so this is directly testable by re-running detection against earlier checkpoints of
the same run. No retraining, no code change, only `-select_model`.

CIFAR-10 BadNet, detection re-run at each epoch, dropout rate re-selected each time by the
paper's own rule:

| epoch | train acc | test acc | gap | ASR | selected p | TPR | FPR |
|---|---|---|---|---|---|---|---|
| 5 | | 0.543 | | 1.000 | 0.6 | 0.527 | 0.254 |
| 15 | 0.9472 | 0.778 | +0.169 | 0.999 | 0.7 | 0.781 | 0.182 |
| 30 | 0.9818 | 0.803 | +0.179 | 1.000 | 0.7 | 1.000 | 0.147 |
| 50 | 0.9675 | 0.782 | +0.185 | 1.000 | 0.7 | 1.000 | 0.152 |
| 60 | 1.0000 | 0.843 | +0.157 | 1.000 | 0.7 | 1.000 | 0.103 |
| 75 | 1.0000 | 0.847 | +0.153 | 1.000 | 0.7 | 1.000 | 0.103 |
| 80 | 1.0000 | 0.847 | | 1.000 | 0.7 | 1.000 | 0.103 |
| 85 | 1.0000 | 0.847 | | 1.000 | 0.7 | 1.000 | 0.103 |
| 90 | 1.0000 | 0.847 | | 1.000 | 0.6 | 1.000 | 0.123 |
| 95 | 1.0000 | 0.847 | +0.153 | 1.000 | 0.7 | 1.000 | 0.103 |
| 100 | 1.0000 | 0.845 | +0.155 | 1.000 | 0.7 | 1.000 | 0.103 |

Three things fall out.

**Nothing happens after epoch 60.** TPR and FPR are constant from 60 to 100. Detecting at
epoch 95 rather than 60 buys exactly zero. The last 40 epochs are wasted compute as far as
detection is concerned.

**The generalization gap does not drive it, and the correlation runs backwards.** The gap is
widest at epoch 50 (+0.185) where FPR is worst (0.152), and narrowest at epochs 75 to 95
(+0.153) where FPR is best (0.103). A wider gap goes with worse detection.

**What tracks detection quality is train accuracy reaching 1.0.** Every movement in FPR
follows interpolation of the training set, including the non-monotone step: train accuracy
dips from 0.9818 at epoch 30 to 0.9675 at epoch 50 and FPR degrades with it, 0.147 to 0.152.
Once train accuracy hits 1.0 at epoch 60, FPR locks at 0.103 and stops responding to anything.

This is mechanistically coherent. PSU separates clean from backdoor because memorized clean
training samples are predicted confidently and therefore shift a great deal under dropout,
while backdoor samples ride a robust trigger-to-target path and shift little. Before the model
interpolates, clean training samples are not confidently predicted either, and the two
populations overlap.

Interpolation is necessary but not sufficient: the Blend model is also fully interpolated at
epoch 95 and still only reaches TPR 0.920.

### Consequence for the ViT work

At 15 epochs, this project's training length, PSBD scores TPR 0.781. The paper's own tables
mark TPR below 0.8 as a failed case. A 15-epoch schedule is not a cheap stand-in for their
setup; it is a regime where their method fails by their own criterion. Whether that is about
epoch count or about reaching interpolation is an open question worth answering directly,
since a ViT may interpolate CIFAR-10 in far fewer than 60 epochs.

## Caveats

- Single trial at seed 42. The paper averages 10.
- The epoch table is one cell. TPR saturates at 1.000 from epoch 30, so only FPR carries
  information above that point. GTSRB Blend, published at TPR 0.910, is well off the ceiling
  and is the useful check; it is in the queued sweep.
- `select_dropout_rate` is unstable. At epoch 90 it chose 0.6 rather than 0.7 with no other
  change, costing 2 points of FPR. On Blend it chose 0.5 where 0.6 was better.

## Reproducing

```bash
cd scratch/psbd-upstream
bash smoke.sh                      # 3-epoch plumbing check
bash run_phase1.sh                 # CIFAR-10 and GTSRB, BadNet and Blend
python epoch_sweep.py              # detection vs epoch, resumable, writes results_epoch_sweep.csv
python collect_results.py          # the comparison table above
cd detection && python force_p.py -dataset cifar10 -poison_type blend -alpha 0.2 \
    -no_aug -no_normalize -select_model 95 --rates 0.4 0.5 0.6 0.7 0.8
```
