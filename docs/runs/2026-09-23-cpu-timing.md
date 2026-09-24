# CPU timings for PSBD sweeps and the competitor detectors

2026-09-23. Measured because the GPU queue carries 1463 waiting jobs and the
login-node GPU is at 97% utilization from another job, so the question was
whether the remaining work runs on the CPU queue instead. It does. These are the
numbers `pbs/generate_swin_gap_jobs.py` and `pbs/generate_detector_jobs.py` use
to size a walltime, and they are measurements rather than estimates.

Both jobs ran on the `cpu` queue, which started them within a minute.

## Swin-S sweep, 32 threads

`pbs/swin_gap/timing_smoke.pbs`, job 1069376, node x8000c0s7b1n0. 1 checkpoint,
1 position, 1 rate, 3 forward passes, batch 32, 8 loader workers, into a
throwaway results directory so no real cache was touched.

| max samples | rows | forwards | seconds |
|---|---|---|---|
| 200 | 600 | 2400 | 92 |
| 1000 | 3000 | 12000 | 311 |

Rows are 3 splits at the cap. Forwards are 1 unperturbed baseline plus 3
perturbed passes per row. The 2 points give a marginal rate and a fixed cost
without either contaminating the other.

    marginal rate = (12000 - 2400) / (311 - 92) = 43.8 forwards per second
    fixed cost    = 92 - 2400 / 43.8 = 37 seconds

`FORWARDS_PER_SECOND_CPU` is set to 40, below the measurement, because a job that
ends at its wall leaves a half-written rate ladder the next run has to redo.

## Competitor detector, 16 threads

`pbs/detector_cpu_smoke/smoke.pbs`, job 1069357. `confidence` on
`vit_gtsrb_badnet_a2o_0_05` at the full split, 23208 rows at 1 forward per row,
took 1487 seconds. That is 15.6 rows per second against the 1000 per second the
login-node A100 reached in the smoke of 2026-09-10, so the CPU slowdown is 64.

`CPU_SLOWDOWN_DEFAULT` is 64. The remaining 10 detectors were still running when
this was written, so only `confidence` carries a per-detector measurement and the
rest inherit the default.

## What this licenses

The 20 Swin sweeps that complete the paired comparison of the recommended
placement, at 102 estimated CPU-hours, submitted as jobs 1069385 to 1069404.

It does not license the k=20 forward-pass sweep, which is GPU work: 1 cell hit
the 900 second wall unfinished on the contended login-node GPU.

It also does not license the competitor detectors on Swin at panel scale. At a
slowdown of 64, CD-L alone is 46 CPU-hours per checkpoint, so 90 Swin cells over
11 detectors is roughly 10000 CPU-hours. The recommended placement is already
cached on all 90 Swin cells, which is what the Swin section needs.

## Float32 on CPU against bfloat16 on GPU

`defenses.inference` enters bfloat16 autocast only on CUDA, so a CPU sweep runs in
float32 where every GPU cache is bfloat16. Job 1069557 (`sw_parity`) swept
`before_attention_residual_token_mask` on `swin_gtsrb_badnet_a2o_0_05` on a CPU node
into `scratch/swin_dtype_parity/results/` and hit its 8 hour wall with 9 of the 10
rates written, every rate but 0.9. Both caches were analyzed on the same 9 rates with
`cli.analyze`, each on a copy of its own cache that lists only those rates.

| rate | clean shift ratio, bf16 | fp32 | AUROC at q0.25, bf16 | fp32 | fp32 minus bf16 |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0230 | 0.0260 | 0.8426 | 0.8407 | -0.0019 |
| 0.1 | 0.0723 | 0.0690 | 0.9254 | 0.9240 | -0.0013 |
| 0.2 | 0.2417 | 0.2450 | 0.9781 | 0.9773 | -0.0008 |
| 0.3 | 0.5745 | 0.5657 | 0.9868 | 0.9865 | -0.0004 |
| 0.4 | 0.9105 | 0.9090 | 0.9960 | 0.9960 | -0.0000 |
| 0.5 | 0.9540 | 0.9540 | 0.9933 | 0.9928 | -0.0005 |
| 0.6 | 0.9757 | 0.9733 | 0.9990 | 0.9993 | +0.0003 |
| 0.7 | 0.9960 | 0.9960 | 1.0000 | 1.0000 | +0.0000 |
| 0.8 | 0.9960 | 0.9960 | 1.0000 | 1.0000 | +0.0000 |

The adaptive rule picks rate 0.4 on both and reads 0.996 on both. The largest gap at
any rate is 0.0019, at the smallest rate, where no rule reads the placement.

The per-image readings differ far more than the AUROC does, a mean absolute gap of
0.01 to 0.09 in the per-pass probability and argmax agreement as low as 0.79 at rate
0.6. That is not precision alone. The token mask is drawn from the device's own
random stream, and the 2 runs used batch sizes 64 and 32, so the 2 caches perturb
different tokens. The comparison therefore bounds dtype and mask draw together, and
together they move the reading less than the 3rd decimal at the chosen rate.

The 10 CPU-swept Swin cells are pooled with the GPU-swept ones. The check is 1 model
and 1 placement, so a cell whose reading sits near a decision boundary should be
read with this 0.002 margin in mind.
