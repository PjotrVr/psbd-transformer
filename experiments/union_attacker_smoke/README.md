# Union attacker smoke test

## Question

`pbs/generate_union_attacker_jobs.py` retrains the single-probe attacker's
CIFAR-100 panel against H41's 3-probe union
(`docs/hypothesis/H41-multi-probe-defence.md`: token masking and dropout at
`before_attention_norm`, gain scaling at `mlp_norm_out`) using the new
`--evade-probes` multi-probe hinge (`attacks/evasion.py`,
`cli/train_backdoor.py`). Before committing 14 full 15-epoch jobs to the
queue, this smoke test checks, on the login GPU, that the whole pipeline
(`cli.train_backdoor --evade-probes`, `cli.sweep`, `cli.analyze`,
`defences.decision.multi_probe_auroc`) runs end to end on 1 cheap checkpoint
and produces numbers `multi_probe_auroc` can consume.

It is not a measurement of whether the union defence survives the union
attacker. The deviations below (1 training epoch instead of 4, 8000 of
CIFAR-100's 50000 training images, a reduced sweep sample count) make every
number here too noisy and too far from the panel's own training recipe to
support that claim. What it establishes is narrower: the pipeline runs, and
what broke along the way.

## What actually happened, and why the plan changed mid-run

The task specification was `--epochs 4 --batch-size 64`. Both failed before
completing 1 step:

1. **Batch 64: CUDA OOM.** `attacks.evasion`'s multi-probe hinge keeps every
   probe's stochastic passes in the graph for 1 combined backward pass. 3
   probes at `passes=3` retain 1 shared base forward plus 9 stochastic
   forwards (1 + 3*3 = 10 retained ViT-B/16 graphs), against 1 + 3 = 4 for the
   single-probe hinge that trained the existing `_evade_l1` checkpoints at
   batch 48. 10 retained graphs at batch 64 exceeded the A100's 40GB.
2. **Batch 32: CUDA OOM again**, partway through epoch 1. Half the batch of
   (1) was still not enough headroom for 10 simultaneous graphs.
3. **Batch 16, 4 epochs: ran, but too slow for the 60-minute budget.** 1 epoch
   completed (loss 2.639, `psu_clean` 0.0999, `psu_poisoned` 0.2489) in about
   30 minutes including calibration, and 3 more epochs would have pushed the
   run well past an hour, so it was killed after epoch 1.
4. **Batch 16, 1 epoch: completed in 19.6 minutes** (calibration and eval
   included), and is the checkpoint every number below reads.

So the checkpoint trained is `vit_cifar100_badnet_a2o_0_05` at 5% BadNet A2O,
1 epoch (not 4), batch 16 (not 64), `--max-samples 8000` (as specified),
3-probe hinge at `--evade-weight 1.0`. The sweep step also used
`--max-samples 1500` (not the full validation/backdoor split) to keep the
5-placement, 46-rate sweep inside the remaining budget.

This is also why `pbs/generate_union_attacker_jobs.py`'s constants changed
after this smoke test: it now pins `--batch-size 16` on every training command
(the smoke test is the evidence that 64 and 32 both OOM for this exact
3-probe combination), and its walltime model was rebuilt from the smoke run's
own measured per-step cost rather than the original 6x-a-plain-run guess,
which put a full run at about 8.4 hours. The corrected estimate is about 13.9
hours per cell (see the generator's own comments for the arithmetic), because
batch 16 pays in step count, not just per-step FLOPs. `--hours` defaults to
16.0 accordingly.

## Commands

```
python -m cli.train_backdoor \
    --dataset cifar100 --attack badnet_a2o --poison-rate 0.05 \
    --target-label 0 --architecture vit --epochs 1 --batch-size 16 \
    --max-samples 8000 \
    --evade-psbd --evade-probes before_attention_norm:token_mask \
        before_attention_norm:dropout mlp_norm_out:gain_scale \
    --evade-weight 1.0 --evade-objective hinge --seed 0 \
    --output checkpoints/vit_cifar100_badnet_a2o_0_05_evade_union_smoke/attack_result.pt

python -m cli.sweep --checkpoint-folder vit_cifar100_badnet_a2o_0_05_evade_union_smoke \
    --position before_attention_norm --operator token_mask \
    --rates 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --forward-passes 3 --max-samples 1500 --skip-existing
# repeated for before_attention_norm dropout, mlp_norm_out gain_scale,
# post_residual dropout, before_attention_residual token_mask

python -m cli.analyze --checkpoint-folder vit_cifar100_badnet_a2o_0_05_evade_union_smoke
```

The min-rank union AUROCs were computed directly from the 2 checkpoints'
stage-1 caches with `defences.decision.multi_probe_auroc`, reading fractional
PSU (`defences.scores.psu_ratio_from_cache`) at each placement's
`adaptive_rate` from `psbd_metrics.json`, the same reader
`experiments/probe_union/measure.py` uses.

## Results

| | union-attacker smoke (1 epoch, batch 16, 8000 images) | unattacked `vit_cifar100_badnet_a2o_0_05` (15 epochs, full data) |
|---|---:|---:|
| ASR | 0.9997 | 1.0000 |
| Clean accuracy | 0.6165 | 0.8254 |
| `before_attention_norm_token_mask` AUROC (attacked) | 0.006 | 0.965 |
| `before_attention_norm` dropout AUROC (attacked) | 0.004 | 0.677 |
| `mlp_norm_out_gain_scale` AUROC (attacked) | 0.008 | 0.965 |
| `post_residual` dropout AUROC (not attacked) | 0.013 | 0.570 |
| `before_attention_residual_token_mask` AUROC (not attacked) | 0.287 | 0.948 |
| Union AUROC, 3 attacked probes | 0.014 | 0.999 |
| Union AUROC, all 5 probes | 0.787 | 0.999 |

Clean accuracy at 0.6165 against the panel's usual 0.83 for this cell is an
artifact of 1 epoch over 8000 images, not a property of the union attacker.
The unattacked column is read as-is from
`results/vit_cifar100_badnet_a2o_0_05/psbd_metrics.json`, a 15-epoch,
full-dataset checkpoint, so it is not a controlled comparison against the
smoke checkpoint. It is included only to show what these same 5 numbers look
like on a healthy model.

## What this does and does not show

**Does show:** the multi-probe hinge collapses all 3 probes it was trained
against (AUROC 0.965/0.677/0.965 to 0.004-0.008) even from an undertrained,
1-epoch model, so the loss is doing what Task 1 built it to do. The pipeline
end to end (`--evade-probes` training, multi-placement sweep, `cli.analyze`,
`multi_probe_auroc`) runs without further errors once the memory ceiling was
found. The 5-probe union (0.787) recovers substantially over the 3-attacked
union (0.014) even though 2 of its own added members individually read badly
here (`post_residual` 0.013, itself inverted, and the attention-branch token
mask at 0.287), which is the min-rank mechanism combining anti-correlated
per-sample errors rather than each probe's own marginal AUROC.

**Does not show:** whether the union defence detects a properly-trained
(15-epoch, full-data) 3-probe union attacker. A 1-epoch, 8000-image, 0.62
clean-accuracy model does not resemble the panel this defence is evaluated
against, and the sweep's own 1500-sample cap adds sampling noise beyond that.
That question is what the 14 full jobs `pbs/generate_union_attacker_jobs.py`
emits are for. It also does not establish a safe memory margin at batch 16
for a full 15-epoch run: the smoke test only ran 1 epoch, so a slow memory
leak across many more steps (plug/unplug cycles accumulate over the full run)
is not ruled out by this test.
