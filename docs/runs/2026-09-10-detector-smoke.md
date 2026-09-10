# Detector smoke on the login node, 2026-09-10

Every competitor detector in `detectors/` was run once on 3 GTSRB checkpoints on
the login-node A100 before any panel job was written: the benign reference, a
BadNet cell and a Blend cell, each at 5% poisoning. The purpose was acceptance,
not measurement. A port that reads chance on the benign model, separates the
2 backdoored models in the expected direction and finishes in a predictable time
is accepted for the panel. The records live under `scratch/detector_smoke/` and
were produced at commit `3a634c35de6916a7d9b78569de507a53a5be1705` by `scratch/detector_smoke/run.sh` and
`followup_fullsize.sh`, with 500 validation images per split (200 for the 2
gradient methods) unless the row says otherwise.

## Acceptance table

AUROC and TPR are read at the headline quantile q0.25 of the clean validation
scores, TPR at q0.01 shows the 1% false-positive budget, tie share is the share
of validation scores equal to the threshold. Fit seconds cover everything the
detector does on the validation split before scoring, and seconds per input
cover scoring over the validation, clean and backdoor rows together.

| model | detector | n val | AUROC q0.25 | TPR q0.25 | TPR q0.01 | tie share | fit s | s per input | precision | batch | record |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | confidence | 500 | 0.503 | 0.256 | 0.012 | 0.008 | 0.0 | 0.002 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_benign/detectors/confidence_metrics.json` |
| benign | strip | 500 | 0.500 | 0.195 | 0.020 | 0.000 | 0.2 | 0.005 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_benign/detectors/strip_metrics.json` |
| benign | scale_up | 500 | 0.499 | 0.209 | 0.000 | 0.098 | 0.0 | 0.005 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_benign/detectors/scale_up_metrics.json` |
| benign | scale_up_data_limited | 500 | 0.499 | 0.296 | 0.012 | 0.000 | 2.1 | 0.002 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_benign/detectors/scale_up_data_limited_metrics.json` |
| benign | ibd_psc | 500 | 0.500 | 0.203 | 0.012 | 0.502 | 10.7 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_benign/detectors/ibd_psc_metrics.json` |
| benign | teco | 500 | 0.495 | 0.211 | 0.008 | 0.010 | 0.0 | 0.059 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_benign/detectors/teco_metrics.json` |
| benign | cd_l | 200 | 0.493 | 0.197 | 0.005 | 0.000 | 0.0 | 0.112 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_benign/detectors/cd_l_metrics.json` |
| benign | beatrix | 500 | 0.510 | 0.262 | 0.006 | 0.000 | 7.2 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_benign/detectors/beatrix_metrics.json` |
| benign | ted | 500 | 0.509 | 0.258 | 0.026 | 0.000 | 2.1 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_benign/detectors/ted_metrics.json` |
| benign | sentinet | 200 | 0.499 | 0.197 | 0.005 | 0.000 | 17.6 | 0.057 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_benign/detectors/sentinet_metrics.json` |
| BadNet 5% | confidence | 500 | 0.738 | 0.431 | 0.000 | 0.002 | 0.0 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_badnet_a2o_0_05/detectors/confidence_metrics.json` |
| BadNet 5% | strip | 500 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1 | 0.005 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_badnet_a2o_0_05/detectors/strip_metrics.json` |
| BadNet 5% | scale_up | 500 | 0.914 | 1.000 | 0.000 | 0.068 | 0.0 | 0.004 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_badnet_a2o_0_05/detectors/scale_up_metrics.json` |
| BadNet 5% | scale_up_data_limited | 500 | 0.909 | 1.000 | 0.000 | 0.020 | 2.1 | 0.002 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_badnet_a2o_0_05/detectors/scale_up_data_limited_metrics.json` |
| BadNet 5% | ibd_psc | 500 | 0.481 | 0.000 | 0.000 | 0.320 | 9.8 | 0.002 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_badnet_a2o_0_05/detectors/ibd_psc_metrics.json` |
| BadNet 5% | teco | 500 | 0.994 | 1.000 | 0.757 | 0.008 | 0.0 | 0.057 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_badnet_a2o_0_05/detectors/teco_metrics.json` |
| BadNet 5% | cd_l | 200 | 1.000 | 1.000 | 1.000 | 0.000 | 0.0 | 0.112 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_badnet_a2o_0_05/detectors/cd_l_metrics.json` |
| BadNet 5% | beatrix | 500 | 0.453 | 0.000 | 0.000 | 0.000 | 7.1 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_badnet_a2o_0_05/detectors/beatrix_metrics.json` |
| BadNet 5% | ted | 500 | 0.999 | 1.000 | 0.990 | 0.000 | 2.1 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_badnet_a2o_0_05/detectors/ted_metrics.json` |
| BadNet 5% | sentinet | 200 | 0.107 | 0.066 | 0.056 | 0.000 | 17.6 | 0.057 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_badnet_a2o_0_05/detectors/sentinet_metrics.json` |
| Blend 5% | confidence | 500 | 0.876 | 0.877 | 0.034 | 0.004 | 0.0 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_blend_0_05/detectors/confidence_metrics.json` |
| Blend 5% | strip | 500 | 0.976 | 0.960 | 0.905 | 0.000 | 0.1 | 0.004 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_blend_0_05/detectors/strip_metrics.json` |
| Blend 5% | scale_up | 500 | 0.689 | 0.374 | 0.000 | 0.082 | 0.0 | 0.004 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_blend_0_05/detectors/scale_up_metrics.json` |
| Blend 5% | scale_up_data_limited | 500 | 0.678 | 0.507 | 0.000 | 0.018 | 2.1 | 0.002 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_blend_0_05/detectors/scale_up_data_limited_metrics.json` |
| Blend 5% | ibd_psc | 500 | 0.970 | 0.978 | 0.724 | 0.016 | 10.7 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_blend_0_05/detectors/ibd_psc_metrics.json` |
| Blend 5% | teco | 500 | 0.922 | 0.893 | 0.258 | 0.000 | 0.0 | 0.053 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_blend_0_05/detectors/teco_metrics.json` |
| Blend 5% | cd_l | 200 | 0.962 | 0.990 | 0.596 | 0.000 | 0.0 | 0.113 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_blend_0_05/detectors/cd_l_metrics.json` |
| Blend 5% | beatrix | 500 | 0.819 | 0.809 | 0.000 | 0.000 | 6.9 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_blend_0_05/detectors/beatrix_metrics.json` |
| Blend 5% | ted | 500 | 0.990 | 1.000 | 0.811 | 0.000 | 1.8 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_blend_0_05/detectors/ted_metrics.json` |
| Blend 5% | sentinet | 200 | 0.379 | 0.066 | 0.051 | 0.000 | 17.6 | 0.057 | bfloat16 | 64 | `scratch/detector_smoke/results/vit_gtsrb_blend_0_05/detectors/sentinet_metrics.json` |

The 4 follow-up runs, each changing 1 thing against the BadNet row above:

| model | detector | n val | AUROC q0.25 | TPR q0.25 | TPR q0.01 | tie share | fit s | s per input | precision | batch | record |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BadNet 5%, full split | ibd_psc | 2000 | 0.506 | 0.000 | 0.000 | 0.315 | 26.6 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results_full/vit_gtsrb_badnet_a2o_0_05/detectors/ibd_psc_metrics.json` |
| BadNet 5%, full split | beatrix | 2000 | 1.000 | 1.000 | 1.000 | 0.000 | 36.5 | 0.001 | bfloat16 | 64 | `scratch/detector_smoke/results_full/vit_gtsrb_badnet_a2o_0_05/detectors/beatrix_metrics.json` |
| BadNet 5%, batch 256 | teco | 500 | 0.994 | 1.000 | 0.738 | 0.006 | 0.0 | 0.035 | bfloat16 | 256 | `scratch/detector_smoke/results_bs256/vit_gtsrb_badnet_a2o_0_05/detectors/teco_metrics.json` |
| BadNet 5%, fp32 | cd_l | 200 | 1.000 | 1.000 | 1.000 | 0.000 | 0.0 | 0.619 | float32 | 64 | `scratch/detector_smoke/results_fp32/vit_gtsrb_badnet_a2o_0_05/detectors/cd_l_metrics.json` |

## Decisions taken from the table

TeCo runs at batch 256. The AUROC is identical to batch 64 at 3 decimals and the
scoring time drops from 0.057 to 0.035 seconds per input, because its corruption
loop is launch-bound rather than compute-bound. The job generator emits
`--batch-size 256` for the teco group.

CD-L stays in bfloat16. The fp32 run gives the same AUROC of 1.000 at 5.5 times
the cost, so the autocast policy in `detectors.PRECISION_POLICY` holds for the
1 gradient method where precision could plausibly have mattered.

Beatrix and IBD-PSC are rerun on the full 2000-image validation split before
being judged, since both fit per-class statistics. Beatrix goes from 0.453 at 500
images to 1.000 at 2000, so its chance reading was the truncation (43 classes
share 500 images, under 12 per class for a Gram-matrix band) and not the port.
IBD-PSC does not move, which makes it a failure and not a truncation artefact.

## The 2 failures

IBD-PSC saturates on this ViT. Its score is the negated retained probability of
the original label under amplified LayerNorms, and on the full split every
backdoor score is exactly -1.000 while the clean scores also sit at -1.000 for
most images, so the tie share at the threshold is 0.315 and the AUROC is 0.506.

| split | n | mean score | min | max |
|---|---|---|---|---|
| validation | 2000 | -0.981 | -1.000 | -0.000 |
| clean | 10630 | -0.982 | -1.000 | -0.000 |
| backdoor | 10578 | -1.000 | -1.000 | -1.000 |

The paper's factor omega of 1.5 on BatchNorm collapses a ConvNet's clean accuracy
within a few layers, which is what Algorithm 1 relies on to pick its depth. The
same factor on a ViT's LayerNorm scales 1 branch input of a residual block rather
than the stream, and a 99%-accurate GTSRB model keeps its predictions through
every amplified layer, so the amplified ensemble agrees with the deployed model
on clean and poisoned inputs alike. The faithful port is kept as `ibd_psc`. A
calibrated variant `ibd_psc_calibrated` searches omega upward until Algorithm 1
finds a depth that crosses the paper's clean-error threshold, then scores there,
and records the omega it chose. Both run on the panel, and the deviation is
stated in `docs/detectors/ibd_psc.md`.

SentiNet carries nothing on this ViT. Its AUROC of 0.107 on BadNet is inverted,
and the reason is visible in the diagnostic `scratch/detector_smoke/
sentinet_cam_layers.py`, which hooks the class-activation map at every block
of the BadNet checkpoint over 95 triggered images (8.9 trigger pixels each, the
model predicting the target on every one of them) and measures how much of the
trigger the thresholded mask covers.

| hooked layer | trigger pixels covered by the mask | mask size, share of image | fooled share |
|---|---|---|---|
| 3 | 0.00 (median 0.00) | 0.03 | 0.01 |
| 5 | 0.00 (median 0.00) | 0.04 | 0.00 |
| 6 | 0.01 (median 0.00) | 0.03 | 0.02 |
| 7 | 0.01 (median 0.00) | 0.04 | 0.01 |
| 8 | 0.00 (median 0.00) | 0.08 | 0.00 |
| 9 | 0.00 (median 0.00) | 0.07 | 0.01 |
| 10 | 0.00 (median 0.00) | 0.15 | 0.01 |
| 11 | 0.02 (median 0.00) | 0.03 | 0.02 |
| 12 | 0.00 (median 0.00) | 0.00 | 0.00 |

Class saliency on this model never points at the 3 by 3 patch, so the transplant
carries no trigger and the fooled share stays at the benign level. The port
stays faithful to the paper (Grad-CAM at the input of the last block, mask at
0.85 of the normalized map, 100 overlays and 100 inert-noise controls, a fitted
decision boundary on the clean split), the record keeps this diagnostic, and an
attention-rollout mask is listed as an optional variant in
`docs/detectors/sentinet.md` rather than built now.

## Cost per input for the job generator

The measured scoring seconds per input at batch 64 in bfloat16, over the
validation, clean and backdoor rows together, and the fit seconds per validation
image, go into `pbs/generate_detector_jobs.py` as the cost model in place of the
forward-count estimate it had before the smoke.

| group | detector | s per input | fit s per validation image | source rows |
|---|---|---|---|---|
| cheap | confidence | 0.001 | 0 | BadNet, Blend |
| cheap | strip | 0.005 | 0 | BadNet, Blend |
| cheap | scale_up | 0.004 | 0 | BadNet, Blend |
| cheap | scale_up_data_limited | 0.002 | 0.004 | BadNet, Blend |
| cheap | ibd_psc | 0.002 | 0.021 | BadNet, follow-up full split 0.013 |
| cheap | beatrix | 0.001 | 0.018 | follow-up full split |
| cheap | ted | 0.001 | 0.004 | BadNet, Blend |
| teco | teco | 0.035 | 0 | follow-up batch 256 |
| cd_l | cd_l | 0.112 | 0 | BadNet, Blend |
| sentinet | sentinet | 0.057 | 0.088 | BadNet, Blend at 200 validation images |

## What the smoke does not establish

3 checkpoints on 1 dataset at 1 poison rate say nothing about the panel ranking,
and the 2 gradient methods were run on 200 images. The panel runs from the main
checkout after the merge, and `cli.compare_detectors` produces the comparison
over the 65 clearing cells and the 4 benign references at that point.
