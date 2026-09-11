# Paper adaptive attacker smoke test

## Question

The PSBD paper's own adaptive attacker (Appendix, "Resistance to Potential
Adaptive Attacks") trains against a convex combination of cross entropy and
the mean absolute PSU over the whole batch, `(1 - alpha) L_bd + alpha L_ada`,
with no reference to which samples are poisoned. Our own adaptive attacker
(`--evade-objective hinge`, the `_evade_l1` checkpoints) is reported as the
stronger threat, since it targets the poisoned/clean PSU gap directly rather
than pushing every sample's PSU down together. Is the paper's own attacker
in fact weaker against PSBD-TM (`before_attention_norm` `token_mask`, our
recommended placement) on a ViT?

This is a 3 epoch, 8000 sample smoke test, not a full 15 epoch run. Every
number below is undertrained relative to the panel and is read only against
an equally undertrained control, never against the 15 epoch checkpoints
directly.

## Checkpoints

- `checkpoints/vit_cifar100_badnet_a2o_0_05_smoke_control/`: same attack, no
  evasion, at the smoke budget. The accuracy and ASR baseline the 2 evasion
  runs are read against.
- `checkpoints/vit_cifar100_badnet_a2o_0_05_evade_paper_smoke_a0_5/`:
  `--evade-objective psbd_paper`, alpha 0.5.
- `checkpoints/vit_cifar100_badnet_a2o_0_05_evade_paper_smoke_a0_9/`:
  `--evade-objective psbd_paper`, alpha 0.9.

Every other training argument (dataset, attack, poison rate, target label,
architecture) is read from `checkpoints/vit_cifar100_badnet_a2o_0_05/args.json`,
the same source `pbs/generate_adaptive_attacker_jobs.py` reads for its own
runs. The probe the attacker trains against is `before_attention_norm`
`token_mask`, the deployed placement, calibrated (`--evade-rate 0`) rather
than fixed.

## Commands

Training, 3 epochs, 8000 samples, batch 48 (the single-probe hinge
checkpoints' own batch size):

```
python -m cli.train_backdoor \
    --dataset cifar100 --attack badnet_a2o --poison-rate 0.05 \
    --target-label 0 --architecture vit --epochs 3 \
    --batch-size 48 --max-samples 8000 --seed 0 \
    --output checkpoints/vit_cifar100_badnet_a2o_0_05_smoke_control/attack_result.pt

python -m cli.train_backdoor \
    --dataset cifar100 --attack badnet_a2o --poison-rate 0.05 \
    --target-label 0 --architecture vit --epochs 3 \
    --batch-size 48 --max-samples 8000 --seed 0 \
    --evade-psbd --evade-weight 0.5 \
    --evade-position before_attention_norm --evade-operator token_mask \
    --evade-objective psbd_paper --evade-rate 0 \
    --output checkpoints/vit_cifar100_badnet_a2o_0_05_evade_paper_smoke_a0_5/attack_result.pt

python -m cli.train_backdoor \
    --dataset cifar100 --attack badnet_a2o --poison-rate 0.05 \
    --target-label 0 --architecture vit --epochs 3 \
    --batch-size 48 --max-samples 8000 --seed 0 \
    --evade-psbd --evade-weight 0.9 \
    --evade-position before_attention_norm --evade-operator token_mask \
    --evade-objective psbd_paper --evade-rate 0 \
    --output checkpoints/vit_cifar100_badnet_a2o_0_05_evade_paper_smoke_a0_9/attack_result.pt
```

Sweep and analyze, per checkpoint, at both headline placements from
`configs/psbd_basis.json` (`before_attention_norm_token_mask`, PSBD-TM, the
deployed recommendation, and `post_residual`, PSBD-RD, the published ConvNet
placement):

```
python -m cli.sweep --checkpoint-folder <folder> \
    --position before_attention_norm --operator token_mask \
    --rates 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --forward-passes 3 --max-samples 1500 --skip-existing

python -m cli.sweep --checkpoint-folder <folder> \
    --position post_residual --operator dropout \
    --rates 0.005 0.01 0.02 0.03 0.05 0.07 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --forward-passes 3 --max-samples 1500 --skip-existing

python -m cli.analyze --checkpoint-folder <folder>
```

`<folder>` is each of the 3 checkpoint folders above.

## Results

AUROC and TPR read from `detection_psu_ratio` at the rate the adaptive rule
(`select_rate_adaptively`, target shift 0.8) selects for the placement and
checkpoint in question, from `results/<folder>/psbd_metrics.json`, not the
`adaptive` block, and the 0.10 clean-quantile threshold. ASR and clean
accuracy from `cli.train_backdoor`'s own training log.

Smoke checkpoints (3 epochs, 8000 samples, batch 48). PSBD-TM is
`before_attention_norm_token_mask`, PSBD-RD is `post_residual`. Rate is the
1 `select_rate_adaptively` (target 0.8) picks for that placement and
checkpoint. AUROC and TPR are read at the 0.10 clean-quantile threshold.

| row | ASR | clean acc | PSBD-TM rate | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-RD rate | PSBD-RD AUROC | PSBD-RD TPR@10 |
|---|---|---|---|---|---|---|---|---|
| control (no evasion) | 1.000 | 0.735 | 0.4 | 0.988 | 0.970 | 0.07 | 0.633 | 0.128 |
| alpha 0.5 (`psbd_paper`) | 1.000 | 0.735 | 0.5 | 0.997 | 1.000 | 0.07 | 0.240 | 0.006 |
| alpha 0.9 (`psbd_paper`) | 1.000 | 0.045 | 0.05 | 0.967 | 1.000 | 0.03 | 0.081 | 0.015 |

For reference, 2 full 15 epoch checkpoints already on disk:

| row | ASR | clean acc | PSBD-TM rate | PSBD-TM AUROC | PSBD-TM TPR@10 | PSBD-RD rate | PSBD-RD AUROC | PSBD-RD TPR@10 |
|---|---|---|---|---|---|---|---|---|
| 15 epoch, unattacked | 1.000 | 0.825 | 0.5 | 0.994 | 0.999 | 0.07 | 0.826 | 0.335 |
| 15 epoch, hinge-attacked (`evade_l1`, alpha=1.0) | 1.000 | 0.799 | 0.6 | 0.002 | 0.000 | n/a | n/a | n/a |

`evade_l1`'s own `psbd_metrics.json` was never swept at `post_residual`, so
its PSBD-RD cell is not available.

## Reading

This is a 3 epoch, 8000 sample smoke test, not a full 15 epoch run, so every
number here is a directional signal rather than a panel result. At this
budget the paper's own adaptive attacker does not move PSBD-TM's AUROC below
the undertrained control at either alpha, 0.997 and 0.967 against 0.988,
because its L_ada term pushes every sample's PSU down together rather than
closing the clean versus poisoned gap the defence actually reads, and alpha
0.9 instead destroys the main task, clean accuracy falling from 0.735 to
0.045 while ASR stays at 1.0, for essentially no evasion benefit. PSBD-RD
sinks under both alphas (0.240 and 0.081 against a 0.633 control), but it
already sits well below PSBD-TM on the unattacked model, so this restates
the placement gap rather than a differential attack effect. Against the 1
comparison point available, this is markedly weaker than the hinge attacker,
whose 15 epoch checkpoint drives PSBD-TM's AUROC to 0.002 at a milder 0.799
clean accuracy, consistent with the task's premise that the paper's own
recipe is the weaker adaptive attacker.
