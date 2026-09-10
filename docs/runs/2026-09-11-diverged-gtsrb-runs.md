# Diverged GTSRB training runs

Read-only analysis of the GTSRB ViT training runs that diverged in their last epochs and were saved anyway. Paths below are relative to the worktree unless they start with `/lustre`, and `checkpoints/` and `logs/` in the worktree are symlinks into the main checkout.

| item | value | source |
|---|---|---|
| written | 2026-09-11 | this file |
| worktree | /lustre/home/pstika/projects/PSBD-ViT-refactor | `ls -la` of the worktree root |
| main checkout the jobs ran from | /lustre/home/pstika/projects/PSBD-ViT | `cd` line of every pbs script cited below |
| commit the September runs trained at | 2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty | checkpoints/vit_gtsrb_blend_0_05_tl1/args.json |

## What happened

The inventory reported a set of GTSRB training runs that finished with a wrecked checkpoint and a clean exit. I verified it in both directions: every `args.json` under `checkpoints/` was scanned for a clean accuracy below half the benign GTSRB reference, and every training log under `logs/` was parsed into per-run epoch series and checked for a final validation accuracy below half the run's own best. The second scan matters because a sidecar with no clean accuracy recorded is invisible to the first, and that is how the last Adam folder in the tables below was found.

Both scans land on the same population and it is larger than the inventory's count. Every hit is a ViT run on GTSRB, none is a Swin run and none is on CIFAR-10, CIFAR-100 or Tiny ImageNet. The inventory named a subset of this population, so any of the folders it did not name could be its unnamed entry, and the honest count is the one in the table rather than the inventory's.

| quantity | value | source |
|---|---|---|
| sidecars scanned | 1702 | checkpoints/*/args.json |
| sidecars with clean_accuracy below half the benign reference | 12 | checkpoints/*/args.json |
| sidecars with no clean_accuracy recorded at all | 165 | checkpoints/*/args.json |
| sidecars with no clean_accuracy recorded, vit on gtsrb | 10 | checkpoints/vit_gtsrb_*/args.json |
| log files scanned | 2060 | logs/*/*.log, logs/*/*.OU, logs/*/*.out, logs/*.log |
| training runs parsed from those logs | 1772 | same files |
| runs whose final val_acc fell below half their own best | 13 | same files |
| of those, ViT on GTSRB | 13 | same files |
| of those, on any other dataset or on Swin | 0 | same files |
| collapsed folders on disk (the 12 sidecar hits plus 1 unscored folder) | 13 | checkpoints/*/args.json and logs/vit_no_sam/train_lc.out |
| benign GTSRB reference clean accuracy | 0.9913 | checkpoints/vit_gtsrb_benign/args.json |
| half the benign reference | 0.4956 | computed from checkpoints/vit_gtsrb_benign/args.json |
| runs the inventory counted | 9 | the question |
| runs the inventory named | 8 | the question |
| collapsed folders the inventory did not name | 5 | the tables below |

| folder | seed | optimizer | turn epoch | val_acc before turn | val_acc at turn | val_acc final epoch | final ASR | final CA | log (epoch columns) | source of final columns |
|---|---|---|---|---|---|---|---|---|---|---|
| vit_gtsrb_lc_0_05_tl1_adv8_pilot | 0 | adam | 14 | 0.9910 | 0.0570 | 0.0928 | 0.4371 | 0.0928 | logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log | checkpoints/vit_gtsrb_lc_0_05_tl1_adv8_pilot/args.json |
| vit_gtsrb_lc_0_05_tl1_adv32_pilot | 0 | adam | 14 | 0.9905 | 0.0515 | 0.0807 | 0.3210 | 0.0807 | logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log | checkpoints/vit_gtsrb_lc_0_05_tl1_adv32_pilot/args.json |
| vit_gtsrb_blend_0_05_tl1 | 0 | adam | 13 | 0.9927 | 0.0570 | 0.0797 | 0.9508 | 0.0797 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_5.log | checkpoints/vit_gtsrb_blend_0_05_tl1/args.json |
| vit_gtsrb_badnet_a2o_0_05_tl1 | 0 | adam | 12 | 0.9907 | 0.0570 | 0.0856 | 0.5733 | 0.0856 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_4.log | checkpoints/vit_gtsrb_badnet_a2o_0_05_tl1/args.json |
| vit_gtsrb_sig_0_005_tl1_seed_3 | 3 | adam | 13 | 0.9918 | 0.0744 | 0.0749 | 0.1537 | 0.0749 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log | checkpoints/vit_gtsrb_sig_0_005_tl1_seed_3/args.json |
| vit_gtsrb_sig_0_005_tl1_seed_4 | 4 | adam | 13 | 0.9923 | 0.0644 | 0.0678 | 0.2292 | 0.0678 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log | checkpoints/vit_gtsrb_sig_0_005_tl1_seed_4/args.json |
| vit_gtsrb_sig_0_01_tl1_seed_1 | 1 | adam | 15 | 0.9907 | 0.0682 | 0.0682 | 0.4846 | 0.0682 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log | checkpoints/vit_gtsrb_sig_0_01_tl1_seed_1/args.json |
| vit_gtsrb_badnet_a2o_0_05_trig_p12 | 0 | adam | 15 | 0.9893 | 0.0546 | 0.0546 | 0.0000 | 0.0546 | logs/vit_trigger/trigger_16.log | checkpoints/vit_gtsrb_badnet_a2o_0_05_trig_p12/args.json |
| vit_gtsrb_badnet_a2a_0_01 | not recorded | adam | 14 | 0.9918 | 0.0570 | 0.0690 | 0.0480 | 0.0694 | logs/vit_no_sam/train_badnet_a2a.out | checkpoints/vit_gtsrb_badnet_a2a_0_01/args.json (psbd_baseline_cache values) |
| vit_gtsrb_lc_0_005 | not recorded | adam | 13 | 0.9925 | 0.0546 | 0.0756 | 0.0000 | 0.0756 | logs/vit_no_sam/train_lc.out | logs/vit_no_sam/train_lc.out (the sidecar carries no clean_accuracy or asr) |
| vit_gtsrb_tact_0_01 | not recorded | adam | 12 | 0.9914 | 0.0724 | 0.1232 | 0.0000 | 0.1210 | logs/vit_no_sam/train_tact.out | checkpoints/vit_gtsrb_tact_0_01/args.json (psbd_baseline_cache values) |
| vit_gtsrb_adaptive_blend_0_05_sam_rho_0_05 | 0 | sam, rho 0.05 | 15 | 0.9876 | 0.4633 | 0.4633 | 0.9994 | 0.4633 | logs/vit_sam/1001027.x3000c0s25b0n0.hsn.hpc.srce.hr.OU | checkpoints/vit_gtsrb_adaptive_blend_0_05_sam_rho_0_05/args.json |
| vit_gtsrb_tact_0_05_sam_rho_0_05 | 0 | sam, rho 0.05 | 15 | 0.9906 | 0.0530 | 0.0530 | 0.4636 | 0.0530 | logs/vit_sam/1001030.x3000c0s25b0n0.hsn.hpc.srce.hr.OU | checkpoints/vit_gtsrb_tact_0_05_sam_rho_0_05/metrics.json for ASR, its args.json for CA (args.json carries no asr) |

Every Adam run in the table turned in the same way. The mean epoch loss had sat at the interpolation floor for consecutive epochs, then within a single epoch the mean loss jumped by orders of magnitude, validation accuracy fell to the size of a single test class and the loss settled just below the value a uniform prediction over all classes gives. It stayed there for every remaining epoch, so the wrecked weights are exactly what the job saved, and the constant validation accuracies say the network ended as a constant predictor rather than a noisy one.

| observation | value | source |
|---|---|---|
| GTSRB test images | 12630 | raw_data/gtsrb/gtsrb/GT-final_test.csv |
| val_acc at the turn in 4 Adam runs | 0.0570 | the log column of the table above |
| test images that fraction corresponds to | 720 | computed from raw_data/gtsrb/gtsrb/GT-final_test.csv |
| test classes holding exactly that many images | 1 and 13 | raw_data/gtsrb/gtsrb/GT-final_test.csv |
| target label of the 3 `_tl1` runs among those 4 | 1 | their args.json |
| val_acc at the turn in 2 further Adam runs | 0.0546 | the log column of the table above |
| test images that fraction corresponds to | 690 | computed from raw_data/gtsrb/gtsrb/GT-final_test.csv |
| test classes holding exactly that many images | 12 and 38 | raw_data/gtsrb/gtsrb/GT-final_test.csv |
| number of classes | 43 | data/registry.py lines 51 to 57 |
| loss of a uniform prediction, ln 43 | 3.7612 | computed from data/registry.py |
| mean epoch loss after the turn, across the 9 Adam runs that had epochs left after it | 3.1693 to 3.4786 | the logs in the table above |

| folder | e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 | e9 | e10 | e11 | e12 | e13 | e14 | e15 | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vit_gtsrb_lc_0_05_tl1_adv8_pilot | 0.9585 | 0.9707 | 0.9669 | 0.9689 | 0.9853 | 0.9473 | 0.9789 | 0.9872 | 0.9775 | 0.9926 | 0.9921 | 0.9914 | 0.9910 | 0.0570 | 0.0928 | logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log |
| vit_gtsrb_lc_0_05_tl1_adv32_pilot | 0.9842 | 0.9760 | 0.9719 | 0.9839 | 0.9590 | 0.9553 | 0.9857 | 0.9802 | 0.9881 | 0.9903 | 0.9922 | 0.9917 | 0.9905 | 0.0515 | 0.0807 | logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log |
| vit_gtsrb_blend_0_05_tl1 | 0.9680 | 0.9856 | 0.9789 | 0.9820 | 0.9808 | 0.9877 | 0.9836 | 0.9923 | 0.9918 | 0.9911 | 0.9910 | 0.9927 | 0.0570 | 0.0741 | 0.0797 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_5.log |
| vit_gtsrb_badnet_a2o_0_05_tl1 | 0.9770 | 0.9793 | 0.9568 | 0.9781 | 0.9831 | 0.9682 | 0.9884 | 0.9918 | 0.9919 | 0.9919 | 0.9907 | 0.0570 | 0.0687 | 0.0689 | 0.0856 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_4.log |
| vit_gtsrb_sig_0_005_tl1_seed_3 | 0.9713 | 0.9721 | 0.9709 | 0.9511 | 0.9880 | 0.9949 | 0.9848 | 0.9821 | 0.9911 | 0.9913 | 0.9924 | 0.9918 | 0.0744 | 0.0784 | 0.0749 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log |
| vit_gtsrb_sig_0_005_tl1_seed_4 | 0.9578 | 0.9760 | 0.9679 | 0.9770 | 0.9834 | 0.9740 | 0.9900 | 0.9823 | 0.9907 | 0.9929 | 0.9923 | 0.9923 | 0.0644 | 0.0606 | 0.0678 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log |
| vit_gtsrb_sig_0_01_tl1_seed_1 | 0.9793 | 0.9806 | 0.9690 | 0.9756 | 0.9867 | 0.9733 | 0.9916 | 0.9722 | 0.9823 | 0.9841 | 0.9902 | 0.9913 | 0.9907 | 0.9907 | 0.0682 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log |
| vit_gtsrb_badnet_a2o_0_05_trig_p12 | 0.9837 | 0.9681 | 0.9725 | 0.9517 | 0.9522 | 0.9910 | 0.9727 | 0.9787 | 0.9823 | 0.9925 | 0.9936 | 0.9922 | 0.9906 | 0.9893 | 0.0546 | logs/vit_trigger/trigger_16.log |
| vit_gtsrb_badnet_a2a_0_01 | 0.9690 | 0.9644 | 0.9819 | 0.9862 | 0.9842 | 0.9803 | 0.9842 | 0.9846 | 0.9784 | 0.9854 | 0.9918 | 0.9916 | 0.9918 | 0.0570 | 0.0690 | logs/vit_no_sam/train_badnet_a2a.out |
| vit_gtsrb_lc_0_005 | 0.9747 | 0.9829 | 0.9917 | 0.9919 | 0.9638 | 0.9851 | 0.9890 | 0.9689 | 0.9930 | 0.9928 | 0.9908 | 0.9925 | 0.0546 | 0.0807 | 0.0756 | logs/vit_no_sam/train_lc.out |
| vit_gtsrb_tact_0_01 | 0.9732 | 0.9846 | 0.9676 | 0.9823 | 0.9785 | 0.9698 | 0.9936 | 0.9940 | 0.9936 | 0.9924 | 0.9914 | 0.0724 | 0.0754 | 0.0637 | 0.1232 | logs/vit_no_sam/train_tact.out |
| vit_gtsrb_adaptive_blend_0_05_sam_rho_0_05 | 0.9877 | 0.9739 | 0.9908 | 0.9553 | 0.9896 | 0.9922 | 0.9895 | 0.9911 | 0.9808 | 0.9863 | 0.9885 | 0.9891 | 0.9905 | 0.9876 | 0.4633 | logs/vit_sam/1001027.x3000c0s25b0n0.hsn.hpc.srce.hr.OU |
| vit_gtsrb_tact_0_05_sam_rho_0_05 | 0.9799 | 0.9845 | 0.9574 | 0.9836 | 0.9932 | 0.9936 | 0.9897 | 0.9850 | 0.9916 | 0.9915 | 0.9866 | 0.9921 | 0.9876 | 0.9906 | 0.0530 | logs/vit_sam/1001030.x3000c0s25b0n0.hsn.hpc.srce.hr.OU |

| folder | consecutive epochs with mean loss at or below 0.001 right before the turn | those losses | loss at the turn epoch | loss in the epochs after the turn | log |
|---|---|---|---|---|---|
| vit_gtsrb_lc_0_05_tl1_adv8_pilot | 4 (e10 to e13) | 0.0007, 0.0004, 0.0005, 0.0005 | 0.9869 | 3.4161 | logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log |
| vit_gtsrb_lc_0_05_tl1_adv32_pilot | 3 (e11 to e13) | 0.0010, 0.0004, 0.0005 | 0.3395 | 3.4632 | logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log |
| vit_gtsrb_blend_0_05_tl1 | 4 (e9 to e12) | 0.0005, 0.0005, 0.0006, 0.0009 | 0.5884 | 3.3984, 3.3008 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_5.log |
| vit_gtsrb_badnet_a2o_0_05_tl1 | 3 (e9 to e11) | 0.0005, 0.0005, 0.0007 | 0.7635 | 3.4120, 3.3563, 3.3376 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_4.log |
| vit_gtsrb_sig_0_005_tl1_seed_3 | 3 (e10 to e12) | 0.0004, 0.0005, 0.0005 | 1.4733 | 3.4059, 3.3849 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log |
| vit_gtsrb_sig_0_005_tl1_seed_4 | 3 (e10 to e12) | 0.0006, 0.0006, 0.0005 | 1.3461 | 3.4363, 3.3876 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log |
| vit_gtsrb_sig_0_01_tl1_seed_1 | 3 (e12 to e14) | 0.0005, 0.0005, 0.0005 | 2.3562 | none, the turn was the last epoch | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log |
| vit_gtsrb_badnet_a2o_0_05_trig_p12 | 4 (e11 to e14) | 0.0006, 0.0004, 0.0005, 0.0005 | 0.2166 | none, the turn was the last epoch | logs/vit_trigger/trigger_16.log |
| vit_gtsrb_badnet_a2a_0_01 | 3 (e11 to e13) | 0.0009, 0.0004, 0.0005 | 1.1316 | 3.4327 | logs/vit_no_sam/train_badnet_a2a.out |
| vit_gtsrb_lc_0_005 | 3 (e10 to e12) | 0.0004, 0.0005, 0.0007 | 0.1679 | 3.4786, 3.3933 | logs/vit_no_sam/train_lc.out |
| vit_gtsrb_tact_0_01 | 4 (e8 to e11) | 0.0005, 0.0005, 0.0005, 0.0005 | 2.4043 | 3.4226, 3.3972, 3.1693 | logs/vit_no_sam/train_tact.out |
| vit_gtsrb_adaptive_blend_0_05_sam_rho_0_05 | 0 (lowest epoch loss 0.0296) | 0.0357 at e14 | 0.0316 | none, the turn was the last epoch | logs/vit_sam/1001027.x3000c0s25b0n0.hsn.hpc.srce.hr.OU |
| vit_gtsrb_tact_0_05_sam_rho_0_05 | 0 (lowest epoch loss 0.0014) | 0.0014, 0.0019 at e13 and e14 | 3.2507 | none, the turn was the last epoch | logs/vit_sam/1001030.x3000c0s25b0n0.hsn.hpc.srce.hr.OU |

The pair of SAM runs turned at the final epoch and share the timing but not the loss floor. The TaCT run's epoch losses never reached the floor the Adam runs did, and the Adaptive-Blend run's mean loss at the turn epoch is indistinguishable from the epochs before it while its validation accuracy halved and its ASR survived, so that run collapsed only partially and in the last steps of its final epoch. Both are listed for retraining, but the mechanism argument in the next section rests on the Adam runs.

The job scripts and the trainer both report success on a wrecked run. The trainer's epoch loop prints validation accuracy and never reads it back, the entrypoint evaluates and saves unconditionally, and every generated job ends with an unconditional zero exit so that a failing run inside a batched job cannot stop its siblings. The trigger-sweep job also passed an output path without the filename, which wrote the weights as a plain file that was later recovered into the folder, a separate defect the ledger already reports through `malformed_checkpoints`.

| where success is asserted | line | source |
|---|---|---|
| epoch loop prints val_acc and never inspects it | 282 to 315 | training/loop.py |
| same loop in the tree the jobs ran | 257 to 287 | /lustre/home/pstika/projects/PSBD-ViT/train.py |
| evaluation and save run unconditionally after training | 531 and 584 | cli/train_backdoor.py |
| same in the tree the jobs ran | 503 and 556 | /lustre/home/pstika/projects/PSBD-ViT/train_backdoor.py |
| unconditional `exit 0` at the end of every generated job | 71 | pbs/generate_cleanlabel_jobs.py |
| the same `exit 0` in the submitted scripts | last line | /lustre/home/pstika/projects/PSBD-ViT/pbs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.pbs and every sibling cited in this file |
| `--output checkpoints/vit_gtsrb_badnet_a2o_0_05_trig_p12` with no filename | the first train call | /lustre/home/pstika/projects/PSBD-ViT/pbs/vit_trigger/trigger_16.pbs |
| the resulting `saved checkpoints/vit_gtsrb_badnet_a2o_0_05_trig_p12` line | 26 | logs/vit_trigger/trigger_16.log |

| batch | GTSRB runs trained | collapsed | dates | source |
|---|---|---|---|---|
| vit_cleanlabel_sig_gtsrb, jobs 1 to 5 | 17 | 5 | 2026-09-10 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log to _5.log |
| vit_cleanlabel_pilot, job 4 (the GTSRB job) | 3 | 2 | 2026-09-10 | logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log |
| vit_trigger, job 16 | 2 | 1 | 2026-09-09 | logs/vit_trigger/trigger_16.log |
| September total | 22 | 8 | 2026-09-09 to 2026-09-10 | the 3 rows above |
| vit_no_sam, GTSRB sections of train_lc.out, train_badnet_a2a.out, train_tact.out | 12 | 3 | 2026-07-06 | those 3 files |
| vit_sam rho sweep jobs 1001027 and 1001030 | 8 | 2 | 2026-07-22 | logs/vit_sam/1001027.x3000c0s25b0n0.hsn.hpc.srce.hr.OU and 1001030 |

| folder | git_commit | trained_started_at | trained_ended_at | source |
|---|---|---|---|---|
| vit_gtsrb_lc_0_05_tl1_adv8_pilot | 2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty | 2026-09-10T05:22:19 | 2026-09-10T06:07:49 | checkpoints/vit_gtsrb_lc_0_05_tl1_adv8_pilot/args.json |
| vit_gtsrb_lc_0_05_tl1_adv32_pilot | 2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty | 2026-09-10T06:54:04 | 2026-09-10T07:39:35 | checkpoints/vit_gtsrb_lc_0_05_tl1_adv32_pilot/args.json |
| vit_gtsrb_blend_0_05_tl1 | 2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty | 2026-09-10T03:55:13 | 2026-09-10T04:40:47 | checkpoints/vit_gtsrb_blend_0_05_tl1/args.json |
| vit_gtsrb_badnet_a2o_0_05_tl1 | 2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty | 2026-09-10T06:08:10 | 2026-09-10T06:53:41 | checkpoints/vit_gtsrb_badnet_a2o_0_05_tl1/args.json |
| vit_gtsrb_sig_0_005_tl1_seed_3 | 2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty | 2026-09-10T06:01:18 | 2026-09-10T06:47:00 | checkpoints/vit_gtsrb_sig_0_005_tl1_seed_3/args.json |
| vit_gtsrb_sig_0_005_tl1_seed_4 | 2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty | 2026-09-10T03:45:29 | 2026-09-10T04:30:57 | checkpoints/vit_gtsrb_sig_0_005_tl1_seed_4/args.json |
| vit_gtsrb_sig_0_01_tl1_seed_1 | 2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty | 2026-09-10T05:17:10 | 2026-09-10T06:02:39 | checkpoints/vit_gtsrb_sig_0_01_tl1_seed_1/args.json |
| vit_gtsrb_badnet_a2o_0_05_trig_p12 | 697b72b6da47f1247c474f83fbd13e38cd0340cd | 2026-09-09T18:12:43 | 2026-09-09T18:58:17 | checkpoints/vit_gtsrb_badnet_a2o_0_05_trig_p12/args.json |
| vit_gtsrb_badnet_a2a_0_01 | not recorded | not recorded, log header says 2026-07-06 06:03 | attack_result.pt mtime 2026-07-06 06:49 | checkpoints/vit_gtsrb_badnet_a2a_0_01/args.json, logs/vit_no_sam/train_badnet_a2a.out |
| vit_gtsrb_lc_0_005 | not recorded | not recorded, log header says 2026-07-06 05:17 | attack_result.pt mtime 2026-07-06 06:03 | checkpoints/vit_gtsrb_lc_0_005/args.json, logs/vit_no_sam/train_lc.out |
| vit_gtsrb_tact_0_01 | not recorded | not recorded, log header says 2026-07-06 06:03 | attack_result.pt mtime 2026-07-06 06:48 | checkpoints/vit_gtsrb_tact_0_01/args.json, logs/vit_no_sam/train_tact.out |
| vit_gtsrb_adaptive_blend_0_05_sam_rho_0_05 | 72ec8d59ede8e4b581dd077f67e34b3eef1007fa | 2026-07-22T12:23:28 | 2026-07-22T13:52:55 | checkpoints/vit_gtsrb_adaptive_blend_0_05_sam_rho_0_05/args.json |
| vit_gtsrb_tact_0_05_sam_rho_0_05 | 72ec8d59ede8e4b581dd077f67e34b3eef1007fa | 2026-07-22T13:09:57 | 2026-07-22T14:39:30 | checkpoints/vit_gtsrb_tact_0_05_sam_rho_0_05/args.json |

## The schedule they share

Every run in this document was launched by a PBS script that calls the root `train_backdoor.py` of the main checkout, which delegates to `train_classifier` in the root `train.py`. The worktree carries the same loop as `training/loop.py` behind `cli/train_backdoor.py` with identical optimizer construction, so the fix locations later in this file are given for both trees. Nothing in either tree builds a learning-rate scheduler, a warmup, gradient clipping or mixed precision for the training step, and a grep for those terms across the training package, the cli package and the root entrypoints returns only the constructor arguments and their defaults.

| setting | value | where |
|---|---|---|
| entrypoint the jobs ran | `python train_backdoor.py` at the main checkout root | /lustre/home/pstika/projects/PSBD-ViT/pbs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.pbs, _2.pbs, _4.pbs, _5.pbs, /lustre/home/pstika/projects/PSBD-ViT/pbs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.pbs, /lustre/home/pstika/projects/PSBD-ViT/pbs/vit_trigger/trigger_16.pbs |
| optimizer | `torch.optim.Adam` | /lustre/home/pstika/projects/PSBD-ViT/train.py line 73, training/loop.py line 75 |
| optimizer of the 2 SAM runs | `SAM` wrapping `torch.optim.Adam`, rho 0.05 | /lustre/home/pstika/projects/PSBD-ViT/train.py lines 66 to 72, training/loop.py lines 66 to 73, their args.json |
| peak learning rate | 1e-4 | /lustre/home/pstika/projects/PSBD-ViT/train.py line 233, training/loop.py line 264 |
| learning-rate schedule | none, constant for every step | grep of scheduler, lr_scheduler, warmup, OneCycle, cosine over training/*.py, cli/*.py, /lustre/home/pstika/projects/PSBD-ViT/train.py and train_backdoor.py returns 0 hits |
| warmup | none | same grep |
| weight decay | 1e-4, passed to Adam as coupled L2, not AdamW | /lustre/home/pstika/projects/PSBD-ViT/train.py lines 73 to 75, training/loop.py lines 75 to 77 |
| Adam betas and eps | (0.9, 0.999) and 1e-8, torch defaults the repo never sets | .venv/lib/python3.11/site-packages/torch/optim/adam.py lines 38 to 41 |
| gradient clipping | none between backward and step | /lustre/home/pstika/projects/PSBD-ViT/train.py lines 78 to 83, training/loop.py lines 81 to 93 |
| training precision | fp32, no autocast around the update, bfloat16 only inside the per-epoch validation pass | training/loop.py line 305, /lustre/home/pstika/projects/PSBD-ViT/train.py line 279 |
| epochs | 15 | `--epochs 15` in every pbs above, `epochs` in every args.json above |
| batch size | 128 | /lustre/home/pstika/projects/PSBD-ViT/train_backdoor.py line 217, cli/train_backdoor.py line 271 |
| GTSRB training images | 26640 | pbs/generate_cleanlabel_jobs.py line 47, and n_poisoned 1332 at poison_rate 0.05 in checkpoints/vit_gtsrb_badnet_a2o_0_05_trig_p12/args.json |
| optimizer steps per epoch | 209 (26640 divided by 128 is 208.125, last batch kept) | computed from the 2 rows above and the DataLoader at cli/train_backdoor.py lines 216 to 221 |
| optimizer steps per run | 3135 | computed from the row above |
| Adam second-moment averaging window, 1 over (1 minus beta2) | 1000 steps, about 4.8 epochs | computed from .venv/lib/python3.11/site-packages/torch/optim/adam.py line 39 |
| backbone | `vit_b_16` with `ViT_B_16_Weights.IMAGENET1K_V1` | models/backbones.py lines 45 to 46, /lustre/home/pstika/projects/PSBD-ViT/models.py lines 44 to 45 |
| GTSRB normalization | mean (0.0, 0.0, 0.0), std (1.0, 1.0, 1.0), inputs stay in [0, 1] | data/registry.py lines 51 to 57 |
| CIFAR-10 normalization, for contrast | mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010) | data/registry.py lines 37 to 43 |
| seeding | `seed_everything(args.seed)` before data and again before the model | cli/train_backdoor.py lines 482 and 503 |
| seeds of the collapsed runs | 0, 1, 3, 4 and 3 runs with none recorded | the main table above |

The project notes describe the optimizer as SAM on top of AdamW. Both trees construct `torch.optim.Adam` with a `weight_decay` argument, which is L2 regularization folded into the gradient rather than decoupled decay, so this document records what the code does and leaves the wording to a separate decision.

A pair of properties single GTSRB out among the datasets. It is the only registry entry whose inputs are not standardized, and it is the dataset on which the mean epoch loss reaches the interpolation floor within the epoch budget on most runs, which the healthy table below also shows. Under a constant learning rate with no clipping, an Adam step taken after many near-zero gradients divides a fresh gradient by the square root of a second-moment estimate that has been decaying toward those tiny values, so a single unremarkable batch can move the weights far. That account fits the timing in the tables, and the control reruns in the fix section are the test of it rather than this paragraph.

## Healthy runs for comparison

The same job scripts trained the runs below on the same days with the same schedule, and they finished at the benign level. Several of them sat at the same loss floor for as many epochs as the collapsed runs did and survived, so interpolation is a precondition of the collapse rather than the collapse itself. A single run in the batch shows a loss spike at the final epoch with only a small accuracy dip, which reads as the same event caught before the weights left the basin.

| folder | seed | e12 val_acc | e13 val_acc | e14 val_acc | e15 val_acc | final ASR | final CA | epochs among e12 to e15 with mean loss at or below 0.001 | log |
|---|---|---|---|---|---|---|---|---|---|
| vit_gtsrb_sig_0_005_tl1 | 0 | 0.9918 | 0.9914 | 0.9904 | 0.9918 | 0.3474 | 0.9918 | 4 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log |
| vit_gtsrb_sig_0_005_tl1_seed_1 | 1 | 0.9840 | 0.9889 | 0.9891 | 0.9899 | 0.5206 | 0.9899 | 2 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log |
| vit_gtsrb_sig_0_005_tl1_seed_2 | 2 | 0.9894 | 0.9892 | 0.9893 | 0.9885 | 0.5073 | 0.9885 | 3 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log |
| vit_gtsrb_sig_0_01_tl1 | 0 | 0.9895 | 0.9852 | 0.9812 | 0.9812 | 0.3564 | 0.9812 | 0 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log |
| vit_gtsrb_sig_0_01_tl1_seed_2 | 2 | 0.9845 | 0.9924 | 0.9929 | 0.9662 | 0.6327 | 0.9662 | 1 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log |
| vit_gtsrb_sig_0_01_tl1_seed_3 | 3 | 0.9919 | 0.9702 | 0.9801 | 0.9686 | 0.3863 | 0.9686 | 0 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_3.log |
| vit_gtsrb_sig_0_01_tl1_seed_4 | 4 | 0.9728 | 0.9713 | 0.9775 | 0.9888 | 0.5571 | 0.9888 | 0 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_3.log |
| vit_gtsrb_sig_0_05_tl1 | 0 | 0.9879 | 0.9895 | 0.9889 | 0.9903 | 0.6730 | 0.9903 | 0 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_3.log |
| vit_gtsrb_sig_0_05_tl1_seed_1 (loss 0.1351 at e15 after 0.0004, 0.0005, 0.0005) | 1 | 0.9911 | 0.9914 | 0.9910 | 0.9718 | 0.8794 | 0.9718 | 3 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_3.log |
| vit_gtsrb_sig_0_05_tl1_seed_2 | 2 | 0.9808 | 0.9769 | 0.9902 | 0.9931 | 0.6003 | 0.9931 | 0 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_4.log |
| vit_gtsrb_sig_0_05_tl1_seed_3 | 3 | 0.9844 | 0.9911 | 0.9930 | 0.9929 | 0.8225 | 0.9929 | 2 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_4.log |
| vit_gtsrb_sig_0_05_tl1_seed_4 | 4 | 0.9880 | 0.9898 | 0.9860 | 0.9827 | 0.6369 | 0.9827 | 0 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_4.log |
| vit_gtsrb_lc_0_05_tl1_adv16_pilot | 0 | 0.9896 | 0.9857 | 0.9873 | 0.9895 | 0.9374 | 0.9895 | 0 | logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log |
| vit_gtsrb_badnet_a2o_0_05_trig_p16 | 0 | 0.9899 | 0.9903 | 0.9899 | 0.9881 | 1.0000 | 0.9881 | 4 | logs/vit_trigger/trigger_16.log |

The siblings of the SAM pair and the July batch tell the same story at other rho values and dates. The BadNet all-to-all run at the highest rate sat at the floor for the last third of training and finished at the benign level, and the SAM siblings at larger rho never reached the floor at all.

| folder | e12 val_acc | e13 val_acc | e14 val_acc | e15 val_acc | final ASR | final CA | epochs among e12 to e15 with mean loss at or below 0.001 | log |
|---|---|---|---|---|---|---|---|---|
| vit_gtsrb_badnet_a2a_0_1 | 0.9931 | 0.9922 | 0.9925 | 0.9916 | 0.9826 | 0.9916 | 4 (and e11 as well) | logs/vit_no_sam/train_badnet_a2a.out |
| vit_gtsrb_lc_0_1 | 0.9812 | 0.9808 | 0.9886 | 0.9883 | 1.0000 | 0.9883 | 2 | logs/vit_no_sam/train_lc.out |
| vit_gtsrb_tact_0_005 | 0.9689 | 0.9861 | 0.9899 | 0.9912 | 0.0610 | 0.9912 | 1 | logs/vit_no_sam/train_tact.out |
| vit_gtsrb_tact_0_05_sam_rho_0_1 | 0.9945 | 0.9952 | 0.9317 | 0.9908 | 0.0704 | 0.9908 | 0 | logs/vit_sam/1001030.x3000c0s25b0n0.hsn.hpc.srce.hr.OU |
| vit_gtsrb_adaptive_blend_0_05_sam_rho_0_1 | 0.9929 | 0.9901 | 0.9854 | 0.9917 | 0.9998 | 0.9917 | 0 | logs/vit_sam/1001027.x3000c0s25b0n0.hsn.hpc.srce.hr.OU |
| vit_gtsrb_benign, the reference | not logged | not logged | not logged | not logged | 0.0001 | 0.9913 | not logged | checkpoints/vit_gtsrb_benign/args.json |

## Proposed fix

The lever I recommend is cosine decay of the learning rate to zero by the final epoch, stepped once per epoch, behind a flag that defaults to the current constant schedule so the provenance of every existing checkpoint stays truthful. The peak stays where it is, so the early trajectory of a rerun matches its siblings and only the tail changes, which is the only place a collapse has ever happened in this project. Gradient-norm clipping goes in behind a second flag as a bound on any single step, because the cosine tail alone still leaves a nonzero rate during the epochs where the turns occurred.

Lowering the peak instead would change every epoch of every rerun and force the whole GTSRB panel to be retrained for a like-for-like comparison, so it is the fallback if the cosine reruns still collapse. Rerunning under the same schedule with a new seed is not a fix, because the batch tally above puts the collapse rate high enough that a rerun would plausibly fail again and the checkpoint would carry no record of why it differs from its neighbours.

| change | file and line | what goes there |
|---|---|---|
| flag `--lr-schedule` with choices `constant` and `cosine`, default `constant` | cli/train_backdoor.py after line 302 (the `--rho` argument), /lustre/home/pstika/projects/PSBD-ViT/train_backdoor.py after line 248 | argparse entry, passed to `train_classifier` at cli/train_backdoor.py lines 507 to 528 |
| flag `--clip-grad-norm`, default 0.0 meaning off | same 2 places | argparse entry, passed the same way |
| scheduler construction | training/loop.py after line 280 (`optimizer = build_optimizer(...)`), /lustre/home/pstika/projects/PSBD-ViT/train.py after line 255 | `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)` when cosine, else None. Under SAM the scheduler wraps the SAM object, whose `param_groups` the base Adam shares, training/sam.py line 60 |
| scheduler step | training/loop.py after line 313 (end of the epoch body, after `on_epoch_end`), /lustre/home/pstika/projects/PSBD-ViT/train.py after line 285 | `scheduler.step()` once per epoch |
| clipping in the plain update | training/loop.py between lines 91 and 92 (after `loss.backward()`, before `optimizer.step()`), /lustre/home/pstika/projects/PSBD-ViT/train.py between lines 81 and 82 | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` when max_norm is above 0 |
| clipping in the SAM update | training/loop.py before line 115 (`first_step`) and before line 118 (`second_step`) | same call before each SAM step |
| provenance | training/loop.py `checkpoint_metadata` lines 168 to 225, and its twin at /lustre/home/pstika/projects/PSBD-ViT/train.py line 131 | new keys `lr_schedule` and `clip_grad_norm` in args.json, so the ledger can separate reruns from constant-schedule cells |
| job generator | pbs/generate_cleanlabel_jobs.py `TRAIN_CALL` lines 74 to 84 | append `--lr-schedule cosine --clip-grad-norm 1.0` for GTSRB runs |
| trigger sweep output path | pbs/generate_trigger_sweep_jobs.py line 59 already ends in `attack_result.pt`, the submitted /lustre/home/pstika/projects/PSBD-ViT/pbs/vit_trigger/trigger_16.pbs does not | regenerate the trigger job rather than resubmitting the old file |

| epoch | learning rate under cosine with T_max 15, stepped per epoch | source |
|---|---|---|
| 1 | 1.00e-4 | computed from the peak in the schedule table and the CosineAnnealingLR formula, lr times (1 plus cos(pi k over 15)) over 2 with k the completed epochs |
| 5 | 8.35e-5 | same |
| 10 | 3.45e-5 | same |
| 12 | 1.65e-5 | same |
| 13 | 9.55e-6 | same |
| 14 | 4.32e-6 | same |
| 15 | 1.09e-6 | same |
| earliest turn epoch in the tables above | 12 | the main table |
| latest turn epoch in the tables above | 15 | the main table |

The rerun recipe is the original job line plus both new flags, at the same seed, epochs, target label and overrides as the original pbs, written to the same folder so every downstream path keeps working. Controls go first: rerun a pair of healthy GTSRB cells under the new flags into folders carrying the `_pilot` token, which the basis declaration already excludes from the panel, and compare their ASR and clean accuracy to the constant-schedule originals before any panel cell is retrained.

```bash
cd /lustre/home/pstika/projects/PSBD-ViT
source .venv/bin/activate

python -m cli.train_backdoor \
    --dataset gtsrb \
    --attack badnet_a2o \
    --poison-rate 0.05 \
    --architecture vit \
    --epochs 15 \
    --seed 0 \
    --lr-schedule cosine \
    --clip-grad-norm 1.0 \
    --output checkpoints/vit_gtsrb_badnet_a2o_0_05_cos_pilot/attack_result.pt

python -m cli.train_backdoor \
    --dataset gtsrb \
    --attack blend \
    --poison-rate 0.05 \
    --target-label 1 \
    --architecture vit \
    --epochs 15 \
    --seed 0 \
    --lr-schedule cosine \
    --clip-grad-norm 1.0 \
    --output checkpoints/vit_gtsrb_blend_0_05_tl1/attack_result.pt
```

| control cell to rerun as `<folder>_cos_pilot` | current ASR | current CA | source |
|---|---|---|---|
| vit_gtsrb_badnet_a2o_0_05 | 1.000 | 0.986 | results/coverage/COVERAGE.md line 103 |
| vit_gtsrb_sig_0_05_tl1_seed_3 | 0.8225 | 0.9929 | logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_4.log |
| `_pilot` is an excluded panel token | yes | configs/psbd_basis.json lines 31 to 41 |

| folder to retrain | role | priority | note | source |
|---|---|---|---|---|
| vit_gtsrb_blend_0_05_tl1 | dirty-label control for the clean-label target switch, panel cell | 1 | its ASR clears the bar, which makes the GTSRB blend slot at that rate ambiguous in the ledger today | scripts/coverage_ledger.py `resolve_one_per_attack`, dry run in the guard section |
| vit_gtsrb_badnet_a2o_0_05_tl1 | dirty-label control, panel cell | 1 | | pbs/generate_cleanlabel_jobs.py lines 204 to 221 |
| vit_gtsrb_lc_0_05_tl1_adv8_pilot | epsilon pilot that picks the GTSRB epsilon | 1 | the full clean-label stage waits on this choice | pbs/generate_cleanlabel_jobs.py lines 9 to 19 |
| vit_gtsrb_lc_0_05_tl1_adv32_pilot | epsilon pilot | 1 | same | same |
| vit_gtsrb_tact_0_01 | panel cell | 1 | its ASR of 0.000 in the ledger is the collapse, not the attack | results/coverage/COVERAGE.md line 162 |
| vit_gtsrb_sig_0_005_tl1_seed_3 | seed replicate | 2 | | the main table |
| vit_gtsrb_sig_0_005_tl1_seed_4 | seed replicate | 2 | | the main table |
| vit_gtsrb_sig_0_01_tl1_seed_1 | seed replicate | 2 | | the main table |
| vit_gtsrb_badnet_a2o_0_05_trig_p12 | trigger dose-response cell | 2 | regenerate its job so `--output` carries the filename | /lustre/home/pstika/projects/PSBD-ViT/pbs/vit_trigger/trigger_16.pbs |
| vit_gtsrb_badnet_a2a_0_01 | all-to-all cell outside the ViT panel | 2 | excluded from the panel by the `a2a` token but read by the all-to-all analysis | configs/psbd_basis.json line 36 |
| vit_gtsrb_lc_0_005 | patch-only Label-Consistent, superseded by the `_adv` canonical variant | 3 | retrain only if the patch-only variant is still reported anywhere | configs/psbd_basis.json lines 43 to 45 |
| vit_gtsrb_adaptive_blend_0_05_sam_rho_0_05 | SAM rho sweep cell | 3 | | the main table |
| vit_gtsrb_tact_0_05_sam_rho_0_05 | SAM rho sweep cell | 3 | | the main table |

Some of the folders carry PSBD caches in the main checkout's results tree that were computed from the wrecked weights, and the cache loader reuses any baseline whose row count matches, so those directories must be removed before the resweep or the new model is scored against the old model's confidence. The remaining folders have no cache and need nothing beyond the retrain.

| cache to remove before resweeping | entries | source |
|---|---|---|
| /lustre/home/pstika/projects/PSBD-ViT/results/vit_gtsrb_badnet_a2a_0_01/psbd | 6 | `ls` of that directory |
| /lustre/home/pstika/projects/PSBD-ViT/results/vit_gtsrb_tact_0_01/psbd | 6 | `ls` of that directory |
| /lustre/home/pstika/projects/PSBD-ViT/results/vit_gtsrb_adaptive_blend_0_05_sam_rho_0_05/psbd | 62 | `ls` of that directory |
| collapsed folders with no psbd cache in either results tree | 10 | `ls` of both results trees |
| the reuse hazard the ledger already documents | `stale_baseline` | scripts/coverage_ledger.py lines 168 to 182 |

## Proposed guard

The guard has a trainer layer and a ledger layer. The trainer layer stops a wrecked model from ever becoming a checkpoint, and the ledger layer catches the ones already on disk and any future run that slips past.

In `train_classifier`, keep the best validation accuracy seen across epochs and, after the loop and before the return, call a new `check_not_diverged` function that raises when the final validation accuracy is below half the best. The bar is the run's own best rather than the benign reference because the loop has no reference at hand, and for every run in this document both bars agree by a wide margin. The exception fires before the entrypoint reaches its evaluation and save calls, so no `attack_result.pt` is written, the traceback lands in the job log and a generator that skips existing checkpoints requeues the cell. The metadata should also carry the best and final validation accuracy so a later audit does not need the log.

| trainer guard piece | file and line | source |
|---|---|---|
| track `best_validation_accuracy` inside the epoch loop | training/loop.py lines 282 to 313, twin at /lustre/home/pstika/projects/PSBD-ViT/train.py lines 257 to 285 | this file |
| `check_not_diverged(best, final)` raising `RuntimeError` | training/loop.py, a new function beside `train_one_epoch`, called between line 313 and the `return model` at line 315 | this file |
| threshold | final below 0.5 times best | this file |
| `best_validation_accuracy` and `final_validation_accuracy` keys | training/loop.py `checkpoint_metadata` lines 168 to 225 | this file |
| calls the exception prevents | cli/train_backdoor.py lines 531 (`evaluate_attack`) and 584 (`save_checkpoint`) | cli/train_backdoor.py |
| generator behaviour that requeues a missing checkpoint | pbs/generate_trigger_sweep_jobs.py line 176 reports skipped existing runs | pbs/generate_trigger_sweep_jobs.py |

In the ledger, add `classify_divergence` beside `classify_by_asr` and call it from `build_ledger` right after the clean-accuracy drop is computed, storing the flag on the cell and overriding `asr_class` so a diverged run can never clear the bar. `resolve_one_per_attack` must drop diverged candidates before it looks for a single clearing cell, because today the diverged blend control clears the ASR bar and makes the GTSRB slot at its rate ambiguous, so the ledger cannot be regenerated at all. `render_markdown` prints `DIVERGED` as the verdict and a count in the summary bullets, and `main` prints the list beside the malformed-checkpoints error so it is the last thing on screen.

| ledger guard piece | function | file and line | what changes |
|---|---|---|---|
| new predicate | `classify_divergence(cell, reference)` | scripts/coverage_ledger.py, new function beside `classify_by_asr` at line 265 | true when `clean_accuracy` and `reference` are both present and `clean_accuracy` is below 0.5 times `reference` |
| call site | `build_ledger` | scripts/coverage_ledger.py lines 319 to 325, right after `clean_accuracy_drop` | set `cell["diverged"]`, and set `cell["asr_class"]` to `"diverged"` when true |
| slot resolution | `resolve_one_per_attack` | scripts/coverage_ledger.py line 342, before the single-candidate check at line 357 | filter out candidates with `diverged` set |
| verdict column | `render_markdown` | scripts/coverage_ledger.py lines 488 to 491 | print `DIVERGED` regardless of ASR |
| summary bullet | `render_markdown` | scripts/coverage_ledger.py line 422 onward | count of diverged cells beside the ASR bar counts |
| loud print | `main` | scripts/coverage_ledger.py line 533 onward, beside the `malformed_checkpoints` block | list the diverged folders |
| reference it reads | `benign_reference_accuracy` | scripts/coverage_ledger.py lines 131 to 139 | unchanged, already the per-dataset denominator |

| what the rule flags today | value | source |
|---|---|---|
| benign GTSRB reference | 0.9913 | checkpoints/vit_gtsrb_benign/args.json |
| bar, half the reference | 0.4956 | computed |
| sidecars below the bar | 12, all listed in the main table | checkpoints/*/args.json |
| sidecars below the bar on any dataset other than GTSRB | 0 | checkpoints/*/args.json |
| highest clean accuracy among the flagged sidecars | 0.4633 | checkpoints/vit_gtsrb_adaptive_blend_0_05_sam_rho_0_05/args.json |
| panel cells among them | 3 (vit_gtsrb_badnet_a2o_0_05_tl1, vit_gtsrb_blend_0_05_tl1, vit_gtsrb_tact_0_01) | read-only call of `panel_cells` from scripts/coverage_ledger.py against checkpoints/ |
| flagged panel cell whose ASR clears the bar | vit_gtsrb_blend_0_05_tl1 at 0.9508 | same call and checkpoints/vit_gtsrb_blend_0_05_tl1/args.json |
| `resolve_one_per_attack` on GTSRB at poison rate 0.05 today | ValueError, 2 cells clear (vit_gtsrb_blend_0_05 and vit_gtsrb_blend_0_05_tl1) | read-only call of `resolve_one_per_attack` from scripts/coverage_ledger.py |
| `resolve_one_per_attack` on GTSRB at poison rate 0.01 today | ValueError, unrelated duplicate (vit_gtsrb_adaptive_blend_0_01 and vit_gtsrb_adaptive_blend_0_01_v2) | same |
| `resolve_one_per_attack` on GTSRB at poison rate 0.1 today | resolves | same |
| ledger on disk generated at | 2026-09-09T19:38:19+00:00, before the September runs landed | results/coverage/COVERAGE.md line 6 |
| vit_gtsrb_tact_0_01 in that ledger | ASR 0.000, CA 0.121, dCA -0.870, verdict below_bar | results/coverage/COVERAGE.md line 162 |

The ledger rule cannot see a folder whose sidecar carries no clean accuracy, and the sidecar scan at the top of this file shows how many such folders exist. Those are covered by the trainer guard for new runs and by the log sweep in this document for old ones, so the ledger should at least count unmeasured cells in its summary line rather than leaving them silent.

## Files read

Worktree paths come first, relative to /lustre/home/pstika/projects/PSBD-ViT-refactor. A glob marks files that were scanned programmatically rather than opened by hand.

- checkpoints/*/args.json, every sidecar, scanned programmatically
- checkpoints/vit_gtsrb_benign/args.json
- checkpoints/vit_gtsrb_lc_0_05_tl1_adv8_pilot/args.json
- checkpoints/vit_gtsrb_lc_0_05_tl1_adv32_pilot/args.json
- checkpoints/vit_gtsrb_blend_0_05_tl1/args.json
- checkpoints/vit_gtsrb_badnet_a2o_0_05_tl1/args.json
- checkpoints/vit_gtsrb_sig_0_005_tl1_seed_3/args.json
- checkpoints/vit_gtsrb_sig_0_005_tl1_seed_4/args.json
- checkpoints/vit_gtsrb_sig_0_01_tl1_seed_1/args.json
- checkpoints/vit_gtsrb_badnet_a2o_0_05_trig_p12/args.json
- checkpoints/vit_gtsrb_adaptive_blend_0_05_sam_rho_0_05/args.json and metrics.json
- checkpoints/vit_gtsrb_badnet_a2a_0_01/args.json and metrics.json
- checkpoints/vit_gtsrb_tact_0_01/args.json and metrics.json
- checkpoints/vit_gtsrb_tact_0_05_sam_rho_0_05/args.json and metrics.json
- checkpoints/vit_gtsrb_lc_0_005/args.json, vit_gtsrb_lc_0_01, vit_gtsrb_lc_0_05, vit_gtsrb_lc_0_1, vit_gtsrb_badnet_a2a_0_005, vit_gtsrb_badnet_a2a_0_05, vit_gtsrb_badnet_a2a_0_1, vit_gtsrb_tact_0_05, vit_gtsrb_tact_0_1, vit_gtsrb_blend_0_05, vit_gtsrb_badnet_a2o_0_05 (args.json of each)
- logs/*/*.log, logs/*/*.OU, logs/*/*.out and logs/*.log, every file, parsed programmatically
- logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_1.log through vit_cleanlabel_pilot_7.log
- logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log through vit_cleanlabel_sig_gtsrb_5.log
- logs/vit_trigger/trigger_16.log
- logs/vit_no_sam/train_lc.out, train_badnet_a2a.out, train_tact.out
- logs/vit_sam/1001027.x3000c0s25b0n0.hsn.hpc.srce.hr.OU and 1001030.x3000c0s25b0n0.hsn.hpc.srce.hr.OU
- training/loop.py
- training/sam.py
- cli/train_backdoor.py
- pbs/generate_cleanlabel_jobs.py
- pbs/generate_trigger_sweep_jobs.py
- scripts/coverage_ledger.py, read in full and its `panel_cells`, `benign_reference_accuracy`, `classify_by_asr` and `resolve_one_per_attack` called read-only
- configs/psbd_basis.json
- data/registry.py
- data/loading.py
- evaluation/loaders.py
- evaluation/metrics.py
- models/backbones.py
- results/coverage/COVERAGE.md
- results/ directory listing for the collapsed folders
- raw_data/gtsrb/gtsrb/GT-final_test.csv
- docs/runs/ listing and docs/runs/2026-09-08-session-findings.md
- .claude/styles/writing-style.md
- .venv/lib/python3.11/site-packages/torch/optim/adam.py

Main checkout paths follow, under /lustre/home/pstika/projects/PSBD-ViT. These are the files the jobs actually ran from or the scripts that submitted them.

- train.py
- train_backdoor.py
- models.py
- pbs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.pbs
- pbs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.pbs, _2.pbs, _4.pbs, _5.pbs
- pbs/vit_trigger/trigger_16.pbs
- pbs/ directory listings for vit_cleanlabel_pilot, vit_cleanlabel_sig_gtsrb, vit_trigger and vit_sam
- results/ directory listings for the collapsed folders and results/coverage/COVERAGE.md
- git log entries for commits 2da1b6c, 697b72b, 72ec8d5 and b551d5c, read-only
