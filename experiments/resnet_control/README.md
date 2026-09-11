# ResNet-18 control: does the hinge attacker beat PSBD, or only ViT

## Question

Our hinge attacker (`--evade-psbd --evade-objective psu_gap_hinge --evade-weight
1.0`, this project's own penalty, formerly named `hinge`) beats a single probe
on ViT. PSBD's own paper (Li, Chen, Liu, Wang, arXiv 2406.05826) reports that
PSBD survives its own adaptive attacker on ResNet-18. This control asks whether
our attacker also beats PSBD on the paper's own architecture, dataset and
placement, which tells us whether the vulnerability is about ViT or about the
attacker.

## The paper's recipe, as read

Sources: `papers/PSBD/sec/4_method.tex` ("Settings" under "Prediction Shift"),
`papers/PSBD/sec/5_experiments.tex` ("Experiment Settings"),
`papers/PSBD/sec/7_appendix.tex` (the training-hyperparameter table and the
dataset table).

| field | value | source |
|---|---|---|
| architecture | ResNet-18 (He et al. 2016) | sec/5_experiments.tex |
| datasets | CIFAR-10, GTSRB, Tiny ImageNet | sec/5_experiments.tex |
| input | 3x32x32 (CIFAR-10, GTSRB), no resize | sec/7_appendix.tex dataset table |
| classes | CIFAR-10 10, GTSRB 43 | sec/7_appendix.tex dataset table |
| optimizer | SGD, momentum 0.9, weight decay 1e-4 | sec/7_appendix.tex training table |
| learning rate | 0.1 initial, MultiStep decay at epochs 50, 75 | sec/7_appendix.tex training table |
| epochs | 100 | sec/7_appendix.tex training table |
| batch size | not stated anywhere in the paper text | (absent) |
| augmentation | none, for BadNets and Blend (the 2 attacks here) | sec/5_experiments.tex: augmentation is used only for Adaptive-Blend and Tiny ImageNet |
| normalization | none | sec/4_method.tex "PS setting": "excludes the use of dropout, data augmentation, and data normalization" |
| poison rate | 10% | sec/5_experiments.tex |
| target class | 0 | sec/5_experiments.tex |
| dropout placement | after each residual add, before the block's ReLU | sec/4_method.tex "PS setting" |
| k (forward passes) | 3 | sec/4_method.tex |
| adaptive shift target | 0.8 | sec/4_method.tex "Prediction Shift Backdoor Detection" |
| threshold quantile | 0.25 (25th percentile of clean validation PSU) | sec/4_method.tex |
| pretrained weights | not mentioned | (absent) |

## Deviations, and why

1. **Optimizer stays this project's Adam, not the paper's SGD.**
   `training/loop.py` owns `build_optimizer` and only offers Adam (or SAM-wrapped
   Adam) at learning rate 1e-4, weight decay 1e-4, no LR schedule by default.
   That file is out of this control's file set (the reorganisation runs several
   agents over disjoint file sets in parallel), so it cannot gain an SGD/MultiStep
   branch here. Every dataset and architecture in this project trains this way for
   comparability, so this control keeps that convention rather than making
   ResNet-18 the 1 architecture with a different optimizer, at the cost of no
   longer matching the paper bit-for-bit on this axis.
2. **`build_model` (`training/loop.py`) does not know "resnet18".** Its body is
   `if architecture == "vit": ... elif architecture == "swin": ... else: raise`.
   patches the `build_model` name already imported into `cli.train_backdoor` and
   `cli.train_benign` at runtime, adding the identical 1-line delegation to
   `models.backbones.build_resnet18` that vit and swin already get. vit and swin
   still resolve through the original function, unchanged.
   `cli.train_backdoor.main()`; every flag after that point is the unmodified CLI.
   The correct long-term fix is a 3-line addition to `training/loop.py`, owned by
   whoever holds that file.
3. **CIFAR-scale stem, not torchvision's ImageNet stem.** torchvision's
   `resnet18` starts with a 7x7 stride-2 conv and a 3x3 stride-2 maxpool, sized for
   a 224x224 input; at 32x32 that collapses the feature map to 1x1 by the last
   residual stage. The paper names only "ResNet-18 [He et al.]" with no stem
   detail, and every CIFAR-scale ResNet-18 in the backdoor-learning literature
   substitutes the standard CIFAR stem instead (3x3 stride-1 first conv, no
   maxpool). `models.backbones.build_resnet18` makes that substitution.
4. **Random init, not ImageNet-pretrained.** Nothing in the paper's recipe table
   names a pretrained source, unlike this project's ViT and Swin backbones, which
   deliberately keep their ImageNet init. `build_resnet18` calls
   `resnet18(weights=None, ...)`.
5. **Batch size** is this project's default (128), since the paper states none.
6. **The smoke run trains on `--max-samples 20000` for at most 15 epochs**, per
   this control's own instructions, not the paper's full dataset and 100 epochs.
   The full-recipe jobs (`pbs/generate_resnet_control_jobs.py`) use the paper's
   100 epochs and the full training set; see the job generator's own docstring
   for the walltime estimate that assumption drives.
7. **The evasion-objective names were renamed mid-task**: `psbd_paper` is now
   `psu_mean` and `hinge` is now `psu_gap_hinge` (`attacks/evasion.py`'s
   `EVASION_OBJECTIVE_ALIASES`), because the name should say what the loss
   computes, not whose loss it is. The old strings are still accepted as
   deprecated aliases, so an already-queued job script naming either keeps
   running.

## Architecture and placement (Task 1)

- `models/backbones.py`: `build_resnet18`, registered in `ARCHITECTURE_BUILDERS["resnet18"]`.
  `detect_architecture` gained a 3rd marker set (`layer1.` .. `layer4.`, unique to
  torchvision's ResNet) so `cli.sweep`'s architecture cross-check works unchanged.
  `network_core` needs no change: a bare (non-`Sequential`) model already falls
  through to `return model`.
- `models/positions.py`: `RESNET_POSITIONS` with 1 real site, `"post_residual"`
  (`PositionSpec("", "residual")`), attached with the same removable
  forward-wrapper technique `after_attention_residual` uses on ViT, since the
  add and the ReLU sit inside `BasicBlock.forward` with no module boundary
  between them. `_resnet_post_residual_forward` mirrors torchvision's
  `BasicBlock.forward` exactly except for probing the stream right after
  `out += identity`, before the final `relu`, matching the paper's own
  description ("dropout layers are applied after each residual connection ...,
  before the activation function").

  `cli.sweep` and `attacks.evasion` resolve a `--position` string through
  `DROPOUT_CONFIGS.get(position, (position,))` before ever touching
  `POSITION_REGISTRY`, and that lookup is architecture-agnostic: for the key
  `"post_residual"` it always returns the ViT/Swin pair
  `("after_attention_residual", "after_mlp_residual")`, regardless of
  architecture. `RESNET_POSITIONS` therefore carries all 3 names
  (`"post_residual"`, `"after_attention_residual"`, `"after_mlp_residual"`)
  pointing at the identical spec, so both the direct name and the
  `DROPOUT_CONFIGS`-mediated path resolve to the same site with no
  resnet18-specific branch anywhere outside `models/positions.py`. The indirect
  path attaches the wrapper twice per block (once per alias name); the second
  attach overwrites the first block's `forward`, so exactly 1 probe is ever
  invoked, at the requested rate. `tests/test_resnet.py` exercises both paths.
- `cli/train_backdoor.py`, `cli/train_benign.py`: `--architecture` gained
  `"resnet18"` as a 3rd choice.
- `cli/train_backdoor.py` (interface requirement): the 3 training variants
  differ by exactly the evasion flags, nothing else, because
  `DEFAULT_EVADE_PROBE_BY_ARCHITECTURE` resolves the probe from
  `--architecture` when neither `--evade-position` nor `--evade-operator` is
  given (`before_attention_norm:dropout` for vit/swin, unchanged;
  `post_residual:dropout` for resnet18). The 3 commands, verbatim:

  ```
  python -m cli.train_backdoor --dataset cifar10 --attack badnet_a2o \
      --poison-rate 0.1 --target-label 0 --architecture resnet18 \
      --epochs 100 --output checkpoints/resnet18_cifar10_badnet_a2o_0_1/attack_result.pt

  python -m cli.train_backdoor <the same flags> \
      --evade-psbd --evade-objective psu_mean --evade-weight 0.5

  python -m cli.train_backdoor <the same flags> \
      --evade-psbd --evade-objective psu_gap_hinge --evade-weight 1.0
  ```

- `tests/test_resnet.py` (new): the model builds and its logits have the right
  shape; `load_checkpoint`/`detect_architecture` round-trip a resnet18
  checkpoint; `network_core` returns the bare network; `plug_dropout` attaches
  exactly 1 probe per basic block (8, ResNet-18's 4 stages of 2 blocks) and
  `unplug_dropout` restores the model bit-for-bit; the perturbed output differs
  from the plain one only when the attached probe's rate is nonzero; the 3
  `RESNET_POSITIONS` keys alias the same spec; the `DROPOUT_CONFIGS`-mediated
  path still perturbs every block despite the double attach.

## Smoke test (Task 2)

`resnet18_cifar10_badnet_a2o_0_1_smoke` (unattacked) and
`resnet18_cifar10_badnet_a2o_0_1_evade_smoke` (hinge attacker,
`post_residual:dropout`), both BadNets at 10% poisoning, target class 0,
`--max-samples 20000`, 15 epochs, on the login A100. Both ran through
2 above). No pipeline part assumed a 224-pixel input or a class token: neither
`data/splits.py`, `defences/inference.py` nor `evaluation/` name either, and
`cli.sweep`'s only architecture-specific step (`resolve_architecture`) already
worked once `models/backbones.py`'s marker set was extended.

| | ASR | clean accuracy | wall clock |
|---|---|---|---|
| unattacked | 0.998 | 0.540 | 1.0 min |
| hinge attacker | 0.999 | 0.599 | 2.9 min |

The attacker calibrated its probe rate before training against the
sigma=0.6 matched-shift target on a throwaway (untrained) model:
`calibrated probe rate: 0.5 (sigma=0.242, target=0.6)` (the calibration curve
never reaches 0.6 on an untrained model at this sample size, so
`calibrate_probe_rate` returns its highest tried rate, 0.5).

### Full `post_residual` dropout ladder (`cli.analyze`'s `"rates"` list, `detection_psu_ratio` at q0.10)

Every rate cli.sweep tried, not only the one `select_rate_adaptively` (the
0.8 shift-target rule) picks. `adaptive_rate` and `oracle_rate` are analyze's
own selections; `calibrated` marks the rate the attacker trained against.

**`resnet18_cifar10_badnet_a2o_0_1_smoke` (unattacked)** — adaptive_rate=0.3, oracle_rate=0.3

| rate | clean shift ratio (validation) | AUROC | TPR @ q0.10 |
|---|---|---|---|
| 0.005 | 0.137 | 0.699 | 0.003 |
| 0.01 | 0.182 | 0.730 | 0.003 |
| 0.02 | 0.235 | 0.790 | 0.006 |
| 0.03 | 0.292 | 0.835 | 0.008 |
| 0.05 | 0.382 | 0.870 | 0.019 |
| 0.07 | 0.455 | 0.884 | 0.054 |
| 0.09 | 0.534 | 0.897 | 0.655 |
| 0.10 | 0.555 | 0.900 | 0.785 |
| 0.20 | 0.787 | 0.956 | 0.959 |
| **0.30** ← adaptive_rate, oracle_rate | **0.869** | **0.959** | **0.965** |
| 0.40 | 0.893 | 0.953 | 0.956 |
| 0.50 | 0.896 | 0.948 | 0.900 |
| 0.60 | 0.895 | 0.948 | 0.861 |
| 0.70 | 0.898 | 0.948 | 0.772 |
| 0.80 | 0.898 | 0.943 | 0.631 |
| 0.90 | 0.901 | 0.898 | 0.699 |

**`resnet18_cifar10_badnet_a2o_0_1_evade_smoke` (hinge attacker)** — adaptive_rate=0.4, oracle_rate=0.09, calibrated=0.5

| rate | clean shift ratio (validation) | AUROC | TPR @ q0.10 |
|---|---|---|---|
| 0.005 | 0.061 | 0.622 | 0.000 |
| 0.01 | 0.082 | 0.686 | 0.000 |
| 0.02 | 0.123 | 0.710 | 0.000 |
| 0.03 | 0.143 | 0.740 | 0.000 |
| 0.05 | 0.192 | 0.753 | 0.001 |
| 0.07 | 0.243 | 0.769 | 0.001 |
| **0.09** ← oracle_rate | **0.297** | **0.783** | **0.001** |
| 0.10 | 0.323 | 0.772 | 0.001 |
| 0.20 | 0.584 | 0.743 | 0.001 |
| 0.30 | 0.770 | 0.535 | 0.055 |
| **0.40** ← adaptive_rate | **0.851** | **0.327** | **0.001** |
| **0.50** ← calibrated | **0.891** | **0.284** | **0.000** |
| 0.60 | 0.913 | 0.373 | 0.000 |
| 0.70 | 0.914 | 0.410 | 0.000 |
| 0.80 | 0.924 | 0.425 | 0.000 |
| 0.90 | 0.920 | 0.441 | 0.000 |

No rate on the ladder restores detection to the unattacked model's level
(peak AUROC 0.783 at the oracle rate 0.09, against the unattacked peak of
0.959). The best rate is far from where the attacker calibrated (0.5) and far
from where the paper's own 0.8 adaptive rule lands (0.4, AUROC 0.327, worse
than the rate right below it): AUROC does not increase monotonically with
rate once the attacker has trained against it, it peaks early on the ladder
and then falls, which is the opposite of the unattacked model's own curve.
The deployable rule cannot find the 1 rate that helps, and that rate would
not have helped much even if found.

## Job generator dry run (Task 3)

`pbs/generate_resnet_control_jobs.py --dry-run`, 2 datasets (cifar10, gtsrb) x
2 attacks (BadNets, Blend) x 3 variants (plain, the paper's attacker at
`psu_mean`/weight 0.5, our attacker at `psu_gap_hinge`/weight 1.0) = 12 full
100-epoch training runs, each followed by the same `post_residual` dropout
ladder and `cli.analyze` used for the smoke pair. Every command validated
against `cli.train_backdoor`, `cli.sweep` and `cli.analyze`'s real argparse
parsers.

```
datasets          ('cifar10', 'gtsrb')
attacks           ('badnet_a2o', 'blend')
variants          ['(plain)', '_evade_paper', '_evade_hinge']
training runs     12
probe             post_residual dropout
rate ladder       (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
jobs at 11.0h      2
estimated GPU time   9.2 hours
  job_001: cifar10, 6 cells, 5.0h
  job_002: gtsrb, 6 cells, 4.2h
```

1 job per dataset (6 cells each), both comfortably under the 11-hour budget
(1 hour of margin under the 12-hour queue cap). The walltime estimate is
extrapolated from the smoke run's per-sample, per-epoch cost
(`pbs/generate_resnet_control_jobs.py`'s `run_minutes`), not measured at full
scale, since no full-recipe run has been launched; treat the 9.2-hour total as
approximate. Every training command's own log is piped through `tee` into
`checkpoints/<folder>/train.log`, and the job script greps that file for the
`"calibrated probe rate"` line right after training finishes, so the rate an
evasion run trained against is visible without reading the full job log.

## Reading

The hinge attacker beats PSBD on the paper's own architecture, dataset and
placement, not only on ViT: the smoke pair's adaptive rule falls from AUROC
0.959 unattacked to 0.327 under the attack. Detection is not merely weakened
but inverted in shape, since the unattacked ladder rises to its peak near
rate 0.3 while the attacked ladder peaks early, at rate 0.09, and then falls,
so no fixed rule tuned on the unattacked curve's assumptions can find the 1
rate (AUROC 0.783) that partially recovers detection. This single
BadNets/CIFAR-10 pair used `--max-samples 20000` and 15 epochs rather than the
paper's full dataset and 100 epochs, and only our attacker has been run here,
not the paper's own (`psu_mean`). The 12-run full-recipe batch this generator
packs would settle whether the paper's own adaptive attacker still fails where
ours succeeds, and whether the effect holds at the paper's full training
budget across both datasets and both attacks rather than only in miniature.
