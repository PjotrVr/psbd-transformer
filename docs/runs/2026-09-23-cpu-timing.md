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
