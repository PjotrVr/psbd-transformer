# The 0.459 AUROC audit: PSBD-TM against WaNet on CIFAR-10 at 10%

## Question

On `vit_cifar10_wanet_0_1` (WaNet, CIFAR-10, 10% poisoning, ASR 0.890), PSBD-TM
(`before_attention_norm_token_mask`) reads AUROC 0.459 at the deployable rate, while
PSBD-RD (`post_residual` dropout, the published ConvNet placement) reads 0.947 on
the same checkpoint, and both placements read above 0.94 on GTSRB and Tiny
ImageNet WaNet at the same 10% rate. Is the 0.459 real, a sweep artefact, or a
property of this 1 checkpoint, and is the user's recollection of `pre_residual`
doing well on WaNet consistent with what is on disk.

## What was checked

1. Tabulated every swept placement's AUROC and TPR at the 0.10 and 0.20 quantiles,
   at every rate of the ladder, alongside the achieved clean-validation shift ratio
   and the rate the 0.8 adaptive rule selects, from
   `results/vit_cifar10_wanet_0_1/psbd_metrics.json`.
2. Recomputed AUROC directly from the stage-1 cache under
   `results/vit_cifar10_wanet_0_1/psbd/before_attention_norm_token_mask/`, checked
   the split manifest, the baseline clean accuracy and ASR, and the per-pass
   tensors, with `defences.scores` and `defences.decision`.
3. Read the same tables for the WaNet models on GTSRB and Tiny ImageNet at 10%,
   for `vit_cifar10_wanet_0_05`, and for the project's own 3 Monte Carlo mask
   seeds already cached for this cell (`_seed1`, `_seed2`).
4. Independently swept BackdoorBench's own reference WaNet checkpoint for CIFAR-10
   at 10% (`backdoor_bench_checkpoints/cifar10_wanet_0_1/`, a ViT-B/16, confirmed
   from its `model_name` field and state-dict key layout) through its own PNG test
   images rather than this project's WaNet re-implementation, for
   `before_attention_norm` token_mask, `post_residual` dropout and `pre_residual`
   dropout, at the basis rate ladders, `k=3` forward passes.

## Step 1 to 3: the cell itself

AUROC (fractional PSU, the headline statistic) and TPR at the 0.10 quantile,
clean-validation shift ratio, on `vit_cifar10_wanet_0_1` (clean accuracy 0.9457,
ASR 0.8898):

| placement | rate | shift ratio | AUROC | TPR@q0.10 |
|---|---|---|---|---|
| before_attention_norm token_mask | 0.05 | 0.025 | 0.375 | 0.146 |
| before_attention_norm token_mask | 0.4 | 0.665 | 0.432 | 0.092 |
| before_attention_norm token_mask | **0.5 (adaptive)** | 0.855 | **0.459** | 0.086 |
| before_attention_norm token_mask | 0.9 (top of ladder) | 0.906 | 0.767 | 0.049 |
| post_residual dropout | 0.07 | 0.735 | 0.926 | 0.754 |
| post_residual dropout | **0.09 (adaptive)** | 0.866 | **0.927** | 0.747 |
| pre_residual dropout | 0.5 | 0.879 | 0.925 | 0.653 |
| pre_residual dropout | **0.5 (adaptive)** | 0.879 | **0.925** | 0.653 |

Token masking never separates clean from backdoor on this cell, at any rate on the
0.05 to 0.9 ladder: AUROC climbs from 0.37 at rate 0.05 to a ceiling of 0.77 at
rate 0.9, the top of the sweep, and never approaches the dropout placements'
0.92 to 0.93. The adaptive rule's 0.459 sits inside that flat, low range, not below
it, so a different rate choice would not have rescued the placement. The user's
recollection of `pre_residual` doing well on WaNet is confirmed: it reads 0.925 at
its adaptive rate, essentially tied with `post_residual`'s 0.927, both far above
`before_attention_norm` token masking on this checkpoint.

Recomputing rate 0.5's cache directly with `defences.scores.psu_ratio_from_cache`
and `defences.decision.detection_report` gives 0.4587, matching the stored value
to 4 decimal places. The split manifest pairs clean and backdoor rows correctly,
the baseline clean accuracy (0.9457) and ASR (0.8894) recomputed from
`baseline_clean.pt` and `baseline_backdoor.pt` match `args.json`, and the 3 Monte
Carlo mask seeds already cached for this cell (seed 0, 1, 2) all read 0.482 to
0.484 at rate 0.5, so the failure is not a single unlucky mask draw.

The neighbouring cells rule out a checkpoint-wide problem and a WaNet-wide
problem:

| cell | before_attention_norm token_mask (adaptive) | post_residual (adaptive) | pre_residual (adaptive) |
|---|---|---|---|
| cifar10 wanet 10% (this cell) | 0.459 @ p=0.5 | 0.927 @ p=0.09 | 0.925 @ p=0.5 |
| cifar10 wanet 5% | 0.936 @ p=0.6 | 0.959 @ p=0.07 | 0.950 @ p=0.4 |
| gtsrb wanet 10% | 0.965 @ p=0.4 | 0.966 @ p=0.07 | 0.963 @ p=0.3 |
| tiny wanet 10% | 0.887 @ p=0.5 | 0.860 @ p=0.05 | 0.567 @ p=0.4 |

Token masking succeeds strongly on the very same attack and dataset at 5%
poisoning, and on WaNet at 10% poisoning on GTSRB and Tiny ImageNet. The failure
is specific to the (cifar10, wanet, 10%, token_mask) cell, not to token masking in
general, not to WaNet in general and not to this checkpoint in general, since
`post_residual` and `pre_residual` both read strongly on the identical model.
`vit_cifar10_wanet_0_1` also carries the lowest ASR (0.890) of every WaNet cell
that clears the panel's 0.85 bar, and its clean shift-to-target fraction under
token masking stays under 0.08 at every rate, below the 1-in-10 chance rate for
CIFAR-10's class count, unlike GTSRB WaNet's 0.93. The shift-to-target mechanism
that explains detection on GTSRB is absent here, consistent with the panel
finding that it is a local-trigger, few-class phenomenon.

## Step 4: the BackdoorBench reference checkpoint

`backdoor_bench_checkpoints/cifar10_wanet_0_1/` is a `vit_b_16` (confirmed from
`attack_result.pt`'s `model_name` field and a strict state-dict load through
`models.backbones.load_checkpoint`), so it was swept. `experiments/wanet_cifar10_audit/measure.py`
builds the 3 PSBD splits from the folder's own `bd_test_dataset` PNGs through
`data.backdoorbench.load_backdoor_splits` and `split_validation_and_eval` (2000
held out, matching `PSBD_HELDOUT_SIZE`), then reuses the project's own cache
format, operators and forward-pass code unchanged.

This checkpoint's measured clean accuracy (0.864) and ASR (0.648) through our
loading path are both markedly lower than this project's own equivalent checkpoint
(0.946 / 0.890) and lower than BackdoorBench's usually-reported numbers for WaNet.
3 normalization schemes were tried (CIFAR-10 stats, ImageNet stats, a flat 0.5/0.5)
and none closed the gap, so an unresolved evaluation-protocol difference, likely in
the exact preprocessing BackdoorBench used at attack-generation time for a warp
trigger this sensitive to interpolation, cannot be ruled out. The sweep proceeded
anyway on CIFAR-10 stats, this project's own convention, because the question step
4 asks is a relative one (does token masking underperform dropout on this
independently-trained model), not an absolute reproduction of BackdoorBench's own
numbers.

AUROC (fractional PSU) at the adaptive rate, `bb_cifar10_wanet_0_1` (measured
clean accuracy 0.864, ASR 0.648):

| placement | adaptive rate | AUROC |
|---|---|---|
| before_attention_norm token_mask | 0.6 | 0.544 |
| post_residual dropout | 0.05 | 0.812 |
| pre_residual dropout | 0.4 | 0.814 |

Token masking peaks at 0.556 (rate 0.5) across its whole ladder and never reaches
either dropout placement's range, while both dropout placements reach 0.81 to 0.86
across a wide band of rates. The ordering, token masking well below both dropout
placements, reproduces on a second, independently trained ViT-B/16 WaNet CIFAR-10
model at the same poison rate, evaluated through an entirely separate data path.

## Conclusion

The 0.459 is real: it recomputes exactly from the cache, is stable across 3 Monte
Carlo mask seeds, and reflects token masking failing at every rate on the ladder
for this cell, not a bad adaptive-rate pick. It is not a WaNet-wide or
checkpoint-wide failure, since `pre_residual` and `post_residual` both score above
0.92 on the identical checkpoint and token masking itself scores 0.94 on the same
attack and dataset at 5% poisoning and above 0.96 on GTSRB and Tiny ImageNet
WaNet at 10%. `vit_cifar10_wanet_0_1` sits at the lowest ASR (0.890) among WaNet
cells clearing the panel's bar, and its clean shift-to-target fraction under token
masking stays near or below chance, unlike GTSRB's 0.93, so the shift-to-target
mechanism that supports detection elsewhere is absent here. An independently
trained BackdoorBench reference checkpoint for the same attack, dataset and rate,
evaluated through its own PNG test images, reproduces the same ordering, token
masking well below both dropout placements, despite reading a weaker backdoor
overall (ASR 0.648 against 0.890) and an unresolved gap to BackdoorBench's usual
published numbers for this model. The finding therefore stands as a genuine,
checkpoint-independent property of the (WaNet, CIFAR-10, 10% poisoning) cell:
input-side token masking specifically fails to separate clean from poisoned there,
while dropout-based operators at 2 other positions succeed strongly on the same
models.

## Files

- `measure.py`: the BackdoorBench independent check (step 4). Run with
  `PYTHONPATH=. .venv/bin/python -m experiments.wanet_cifar10_audit.measure`.
- `results/bb_cifar10_wanet_0_1/psbd/`: the stage-1 cache the script wrote, in the
  same layout `cli.sweep` would have used.
- `results/bb_cifar10_wanet_0_1/bb_reference_metrics.json`: the AUROC and shift
  ratio table above, in full, at every rate.
