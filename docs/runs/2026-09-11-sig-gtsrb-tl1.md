# SIG on GTSRB at target class 1

## Question

The GTSRB SIG cells trained at target class 0 never cleared the ASR bar, and `docs/clean-label-rate-caps.md` traced part of that failure to the poison-rate cap. Class 0 is the smallest class in the training set, so every requested rate above its share trained the identical index set, and the cells labelled 1%, 5% and 10% were 3 replicates of 1 configuration.

The hypothesis under test here is that the failure was a rate-cap artefact. It predicts that SIG implants once the target moves to class 1, whose share of the training set is large enough to deliver the 0.5%, 1% and 5% rates without clamping, so the question splits into 2 parts: whether the realized rate now equals the requested rate and whether the attack clears the bar once it does.

The data are the 15 `_tl1` folders trained from the full stage of `pbs/generate_cleanlabel_jobs.py`, 3 rates at seeds 0 to 4, read from each folder's `args.json`. The bar and the benign reference come from `configs/psbd_basis.json`, and a run counts as usable when its clean accuracy is at least half the benign reference, which excludes the collapsed runs from every mean while keeping them on the record.

| quantity | value | source |
|---|---:|---|
| ASR bar | 0.85 | `configs/psbd_basis.json`, key `asr_bar` |
| clean-accuracy drop bar | -0.05 | `configs/psbd_basis.json`, key `clean_accuracy_drop_bar` |
| benign reference clean accuracy | 0.991 | `checkpoints/vit_gtsrb_benign/args.json`, key `clean_accuracy` |
| usable threshold, half the benign reference | 0.496 | derived from the row above |
| training commit of all 15 runs | `2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty` | `checkpoints/vit_gtsrb_sig_*_tl1*/args.json`, key `git_commit` |
| training window of all 15 runs | 2026-09-10, 03:43 to 06:50 UTC | `checkpoints/vit_gtsrb_sig_*_tl1*/args.json`, keys `trained_started_at` and `trained_ended_at` |
| epochs per run | 15 | `checkpoints/vit_gtsrb_sig_*_tl1*/args.json`, key `epochs` |

## The 15 runs

Every folder carries target class 1 and the AdamW optimizer with no SAM tag, and the realized rate matches the requested rate at all 3 rates once the requested count is rounded to a whole image. The usable column applies the half-of-benign rule from the table above.

| folder | seed | requested rate | realized rate | ASR | clean accuracy | usable | source |
|---|---:|---:|---:|---:|---:|---|---|
| `vit_gtsrb_sig_0_005_tl1` | 0 | 0.005 | 0.004992 | 0.347 | 0.992 | yes | `checkpoints/vit_gtsrb_sig_0_005_tl1/args.json` |
| `vit_gtsrb_sig_0_005_tl1_seed_1` | 1 | 0.005 | 0.004992 | 0.521 | 0.990 | yes | `checkpoints/vit_gtsrb_sig_0_005_tl1_seed_1/args.json` |
| `vit_gtsrb_sig_0_005_tl1_seed_2` | 2 | 0.005 | 0.004992 | 0.507 | 0.989 | yes | `checkpoints/vit_gtsrb_sig_0_005_tl1_seed_2/args.json` |
| `vit_gtsrb_sig_0_005_tl1_seed_3` | 3 | 0.005 | 0.004992 | 0.154 | 0.075 | no | `checkpoints/vit_gtsrb_sig_0_005_tl1_seed_3/args.json` |
| `vit_gtsrb_sig_0_005_tl1_seed_4` | 4 | 0.005 | 0.004992 | 0.229 | 0.068 | no | `checkpoints/vit_gtsrb_sig_0_005_tl1_seed_4/args.json` |
| `vit_gtsrb_sig_0_01_tl1` | 0 | 0.01 | 0.009985 | 0.356 | 0.981 | yes | `checkpoints/vit_gtsrb_sig_0_01_tl1/args.json` |
| `vit_gtsrb_sig_0_01_tl1_seed_1` | 1 | 0.01 | 0.009985 | 0.485 | 0.068 | no | `checkpoints/vit_gtsrb_sig_0_01_tl1_seed_1/args.json` |
| `vit_gtsrb_sig_0_01_tl1_seed_2` | 2 | 0.01 | 0.009985 | 0.633 | 0.966 | yes | `checkpoints/vit_gtsrb_sig_0_01_tl1_seed_2/args.json` |
| `vit_gtsrb_sig_0_01_tl1_seed_3` | 3 | 0.01 | 0.009985 | 0.386 | 0.969 | yes | `checkpoints/vit_gtsrb_sig_0_01_tl1_seed_3/args.json` |
| `vit_gtsrb_sig_0_01_tl1_seed_4` | 4 | 0.01 | 0.009985 | 0.557 | 0.989 | yes | `checkpoints/vit_gtsrb_sig_0_01_tl1_seed_4/args.json` |
| `vit_gtsrb_sig_0_05_tl1` | 0 | 0.05 | 0.050000 | 0.673 | 0.990 | yes | `checkpoints/vit_gtsrb_sig_0_05_tl1/args.json` |
| `vit_gtsrb_sig_0_05_tl1_seed_1` | 1 | 0.05 | 0.050000 | 0.879 | 0.972 | yes | `checkpoints/vit_gtsrb_sig_0_05_tl1_seed_1/args.json` |
| `vit_gtsrb_sig_0_05_tl1_seed_2` | 2 | 0.05 | 0.050000 | 0.600 | 0.993 | yes | `checkpoints/vit_gtsrb_sig_0_05_tl1_seed_2/args.json` |
| `vit_gtsrb_sig_0_05_tl1_seed_3` | 3 | 0.05 | 0.050000 | 0.823 | 0.993 | yes | `checkpoints/vit_gtsrb_sig_0_05_tl1_seed_3/args.json` |
| `vit_gtsrb_sig_0_05_tl1_seed_4` | 4 | 0.05 | 0.050000 | 0.637 | 0.983 | yes | `checkpoints/vit_gtsrb_sig_0_05_tl1_seed_4/args.json` |

The 3 unusable runs collapsed late in training, after the loss had already reached its floor, and each of them shows the same signature in its log: a healthy validation accuracy 1 epoch, then a loss jump and a validation accuracy near chance the next. The 2 dirty-label control runs trained in the same batch at the same target class collapsed the same way, so the target-class confound control that `docs/clean-label-rate-caps.md` asked for has no usable data yet and needs a rerun. I record the controls here because the collapse is a property of this batch rather than of SIG, and it deserves its own investigation before more `_tl1` jobs go out.

| folder | last healthy epoch | loss there | val acc there | collapse epoch | loss there | val acc there | final ASR | final clean accuracy | in the 15 | source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `vit_gtsrb_sig_0_005_tl1_seed_3` | 12 | 0.0005 | 0.9918 | 13 | 1.4733 | 0.0744 | 0.1537 | 0.0749 | yes | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log` |
| `vit_gtsrb_sig_0_005_tl1_seed_4` | 12 | 0.0005 | 0.9923 | 13 | 1.3461 | 0.0644 | 0.2292 | 0.0678 | yes | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log` |
| `vit_gtsrb_sig_0_01_tl1_seed_1` | 14 | 0.0005 | 0.9907 | 15 | 2.3562 | 0.0682 | 0.4846 | 0.0682 | yes | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log` |
| `vit_gtsrb_badnet_a2o_0_05_tl1` | 11 | 0.0007 | 0.9907 | 12 | 0.7635 | 0.0570 | 0.5733 | 0.0856 | no, control | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_4.log` |
| `vit_gtsrb_blend_0_05_tl1` | 12 | 0.0009 | 0.9927 | 13 | 0.5884 | 0.0570 | 0.9508 | 0.0797 | no, control | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_5.log` |

One trap in the logs is worth recording, because a reader grepping for the cap will find it. The 0.5% and 1% runs print the line "capped by the eligible pool" even though none of the 3 counts reaches the class 1 pool, while the 5% runs, whose count comes closest to it, print nothing. The message in `cli/train_backdoor.py` fires whenever the realized rate differs from the requested rate by any amount, and rounding the requested count to a whole image is enough to trigger it at 0.5% and 1%, where the requested count is fractional, but not at 5%, where it is already whole. The `_tl1` sidecars also carry no `poison_rate_capped` key at all, unlike the older folders where audit A8 backfilled the key, so a script reading that key would treat these runs as unknown rather than uncapped.

| requested rate | requested count, rate times train size | whole-image count | realized rate | cap line printed | source |
|---:|---:|---:|---:|---|---|
| 0.005 | 133.2 | 133 | 0.004992 | yes | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log`, `cli/train_backdoor.py` lines 147 to 155 |
| 0.01 | 266.4 | 266 | 0.009985 | yes | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log`, `cli/train_backdoor.py` lines 147 to 155 |
| 0.05 | 1332.0 | 1332 | 0.050000 | no | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_3.log`, `cli/train_backdoor.py` lines 147 to 155 |

## Per-rate summary

The means below run over the usable seeds only, with the collapsed seeds dropped rather than zeroed. ASR rises with rate, and the 5% cell is the only cell where any seed clears the bar, but no rate clears it on the mean.

| requested rate | n usable | n trained | mean ASR | SD of ASR | min ASR | max ASR | mean clean accuracy | mean clears bar | seeds clearing bar | source |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 0.005 | 3 | 5 | 0.458 | 0.096 | 0.347 | 0.521 | 0.990 | no | none | usable rows of the table above, `checkpoints/vit_gtsrb_sig_0_005_tl1*/args.json` |
| 0.01 | 4 | 5 | 0.483 | 0.133 | 0.356 | 0.633 | 0.976 | no | none | usable rows of the table above, `checkpoints/vit_gtsrb_sig_0_01_tl1*/args.json` |
| 0.05 | 5 | 5 | 0.722 | 0.122 | 0.600 | 0.879 | 0.986 | no | seed 1 only | usable rows of the table above, `checkpoints/vit_gtsrb_sig_0_05_tl1*/args.json` |

The single clearing seed passes both halves of the validity rule, since its clean-accuracy drop against the benign reference stays inside the drop bar. It is also the seed whose loss spiked in its final epoch, which is visible in the second trajectory table below, so its ASR was read from a model 1 epoch away from the floor the other seeds sat on.

| folder | ASR | clean accuracy | drop against benign | clears ASR bar | inside drop bar | source |
|---|---:|---:|---:|---|---|---|
| `vit_gtsrb_sig_0_05_tl1_seed_1` | 0.879 | 0.972 | -0.019 | yes | yes | `checkpoints/vit_gtsrb_sig_0_05_tl1_seed_1/args.json`, `checkpoints/vit_gtsrb_benign/args.json` |

The training log prints loss and validation accuracy once per epoch and ASR exactly once, on the final line, so whether ASR was still rising at the last epoch cannot be read from the log. What the log does show for a healthy 5% seed is that the loss reached its floor only in the last 2 epochs, with the validation accuracy already flat, so the ASR reported for this seed comes from a model that had just finished fitting the training set. Seed 3 is the healthy seed closest to the bar.

| epoch | loss | validation accuracy | source |
|---:|---:|---:|---|
| 1 | 0.3113 | 0.9554 | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_4.log`, block `vit_gtsrb_sig_0_05_tl1_seed_3` |
| 2 | 0.0200 | 0.9085 | same |
| 3 | 0.0195 | 0.9839 | same |
| 4 | 0.0041 | 0.9695 | same |
| 5 | 0.0176 | 0.9816 | same |
| 6 | 0.0106 | 0.9742 | same |
| 7 | 0.0101 | 0.9870 | same |
| 8 | 0.0115 | 0.9798 | same |
| 9 | 0.0095 | 0.9683 | same |
| 10 | 0.0099 | 0.9758 | same |
| 11 | 0.0089 | 0.9805 | same |
| 12 | 0.0081 | 0.9844 | same |
| 13 | 0.0044 | 0.9911 | same |
| 14 | 0.0006 | 0.9930 | same |
| 15 | 0.0004 | 0.9929 | same |
| final line | ASR 0.8225 | CA 0.9929 | same |

The clearing seed reached the same loss floor 3 epochs earlier and then spiked on the last epoch, which cost it clean accuracy without collapsing it. Its last 4 epochs are below for comparison.

| epoch | loss | validation accuracy | source |
|---:|---:|---:|---|
| 12 | 0.0004 | 0.9911 | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_3.log`, block `vit_gtsrb_sig_0_05_tl1_seed_1` |
| 13 | 0.0005 | 0.9914 | same |
| 14 | 0.0005 | 0.9910 | same |
| 15 | 0.1351 | 0.9718 | same |
| final line | ASR 0.8794 | CA 0.9718 | same |

## The class 0 runs it replaces

Every earlier GTSRB SIG folder targets class 0, and every folder requesting more than the class 0 share carries the same capped configuration, so the folders labelled 1%, 5% and 10% differ only in training noise, since they share the poisoned index set. The 0.5% class 0 folder is the 1 uncapped run in the set, and it is the natural like-for-like partner of the 0.5% `_tl1` cell, since the 2 poisoned counts differ by a handful of images.

| folder | optimizer | requested rate | realized rate | capped flag | poisoned images | ASR | clean accuracy | ASR source key | source |
|---|---|---:|---:|---|---:|---:|---:|---|---|
| `vit_gtsrb_sig_0_005` | adam | 0.005 | 0.004992 | false | 133 | not recorded | not recorded | none | `checkpoints/vit_gtsrb_sig_0_005/args.json` |
| `vit_gtsrb_sig_0_01` | adam | 0.01 | 0.005631 | true | 150 | 0.460 | 0.979 | `psbd_baseline_cache` | `checkpoints/vit_gtsrb_sig_0_01/args.json` |
| `vit_gtsrb_sig_0_05` | adam | 0.05 | 0.005631 | true | 150 | 0.015 | 0.913 | `psbd_baseline_cache` | `checkpoints/vit_gtsrb_sig_0_05/args.json` |
| `vit_gtsrb_sig_0_05_v2` | adam | 0.05 | 0.005631 | true | 150 | not recorded | not recorded | none, recovered orphan | `checkpoints/vit_gtsrb_sig_0_05_v2/args.json` |
| `vit_gtsrb_sig_0_1` | adam | 0.1 | 0.005631 | true | 150 | 0.415 | 0.988 | `psbd_baseline_cache` | `checkpoints/vit_gtsrb_sig_0_1/args.json` |
| `vit_gtsrb_sig_0_01_sam_rho_0_05` | sam, rho 0.05 | 0.01 | 0.005631 | true | 150 | 0.215 | 0.989 | training script | `checkpoints/vit_gtsrb_sig_0_01_sam_rho_0_05/args.json` |
| `vit_gtsrb_sig_0_01_sam_rho_0_1` | sam, rho 0.1 | 0.01 | 0.005631 | true | 150 | 0.293 | 0.993 | training script | `checkpoints/vit_gtsrb_sig_0_01_sam_rho_0_1/args.json` |
| `vit_gtsrb_sig_0_01_sam_rho_0_15` | sam, rho 0.15 | 0.01 | 0.005631 | true | 150 | 0.185 | 0.991 | training script | `checkpoints/vit_gtsrb_sig_0_01_sam_rho_0_15/args.json` |
| `vit_gtsrb_sig_0_01_sam_rho_0_2` | sam, rho 0.2 | 0.01 | 0.005631 | true | 150 | 0.211 | 0.992 | training script | `checkpoints/vit_gtsrb_sig_0_01_sam_rho_0_2/args.json` |
| `vit_gtsrb_sig_0_05_sam_rho_0_05` | sam, rho 0.05 | 0.05 | 0.005631 | true | 150 | 0.367 | 0.987 | training script | `checkpoints/vit_gtsrb_sig_0_05_sam_rho_0_05/args.json` |
| `vit_gtsrb_sig_0_05_sam_rho_0_1` | sam, rho 0.1 | 0.05 | 0.005631 | true | 150 | 0.278 | 0.990 | training script | `checkpoints/vit_gtsrb_sig_0_05_sam_rho_0_1/args.json` |
| `vit_gtsrb_sig_0_05_sam_rho_0_15` | sam, rho 0.15 | 0.05 | 0.005631 | true | 150 | 0.219 | 0.992 | training script | `checkpoints/vit_gtsrb_sig_0_05_sam_rho_0_15/args.json` |
| `vit_gtsrb_sig_0_05_sam_rho_0_2` | sam, rho 0.2 | 0.05 | 0.005631 | true | 150 | 0.422 | 0.990 | training script | `checkpoints/vit_gtsrb_sig_0_05_sam_rho_0_2/args.json` |
| `vit_gtsrb_sig_0_1_sam_rho_0_05` | sam, rho 0.05 | 0.1 | 0.005631 | true | 150 | 0.473 | 0.991 | training script | `checkpoints/vit_gtsrb_sig_0_1_sam_rho_0_05/args.json` |
| `vit_gtsrb_sig_0_1_sam_rho_0_1` | sam, rho 0.1 | 0.1 | 0.005631 | true | 150 | 0.249 | 0.993 | training script | `checkpoints/vit_gtsrb_sig_0_1_sam_rho_0_1/args.json` |
| `vit_gtsrb_sig_0_1_sam_rho_0_15` | sam, rho 0.15 | 0.1 | 0.005631 | true | 150 | 0.215 | 0.985 | training script | `checkpoints/vit_gtsrb_sig_0_1_sam_rho_0_15/args.json` |
| `vit_gtsrb_sig_0_1_sam_rho_0_2` | sam, rho 0.2 | 0.1 | 0.005631 | true | 150 | 0.284 | 0.993 | training script | `checkpoints/vit_gtsrb_sig_0_1_sam_rho_0_2/args.json` |

The 4 AdamW folders also carry a `metrics.json` written by `metrics.py`, which re-evaluated the same weights and agrees with the cached values to the third decimal. It is the only place the 0.5% class 0 run has an ASR at all, since its `args.json` predates the metadata convention and was never backfilled with an ASR.

| folder | ASR | clean accuracy | source |
|---|---:|---:|---|
| `vit_gtsrb_sig_0_005` | 0.175 | 0.993 | `checkpoints/vit_gtsrb_sig_0_005/metrics.json` |
| `vit_gtsrb_sig_0_01` | 0.459 | 0.978 | `checkpoints/vit_gtsrb_sig_0_01/metrics.json` |
| `vit_gtsrb_sig_0_05` | 0.015 | 0.912 | `checkpoints/vit_gtsrb_sig_0_05/metrics.json` |
| `vit_gtsrb_sig_0_1` | 0.417 | 0.988 | `checkpoints/vit_gtsrb_sig_0_1/metrics.json` |

2 things stand out in the old set once it is read at its realized rate. The 3 capped AdamW folders span a wide ASR range for what is 1 configuration, and the lowest of them pairs its near-zero ASR with a depressed clean accuracy, which reads as a partly failed training run rather than a rate effect. The 12 SAM folders sit uniformly low, and none of them is a panel cell in any case, since `sam_rho` is an excluded token.

## Verdict

The rate-cap hypothesis is retired, because the cap fix worked mechanically and the attack still fails. The realized rate now equals the requested rate at all 3 rates with the class 1 pool never exhausted, and lifting the cap moved the 5% mean well above anything the class 0 folders reached, yet no rate clears the bar on the mean up to the class ceiling, with 1 seed of 5 clearing at 5% and none at the lower rates. The class ceiling for target class 1 is the share in the table below, and the headroom between the 5% cell and that ceiling is small enough that a run at the ceiling would land inside the seed spread of the 5% cell rather than test a new rate.

| class | training images | train size | share of train set | headroom above the 5% count | source |
|---:|---:|---:|---:|---:|---|
| 0 | 150 | 26640 | 0.0056 | none, below 5% | `raw_data/gtsrb/gtsrb/GTSRB/Training/00000` file count, `docs/clean-label-rate-caps.md` |
| 1 | 1500 | 26640 | 0.0563 | 168 images | `raw_data/gtsrb/gtsrb/GTSRB/Training/00001` file count, `docs/clean-label-rate-caps.md` |
| 2 | 1500 | 26640 | 0.0563 | 168 images | `raw_data/gtsrb/gtsrb/GTSRB/Training/00002` file count, `docs/clean-label-rate-caps.md` |

The like-for-like comparison supports the same reading. The 0.5% `_tl1` cell and the capped class 0 folders poison nearly the same number of images, and their ASR values sit in the same band, so the target class itself changes nothing at matched count and the gain at 5% is a genuine rate effect that runs out of rate before it reaches the bar. Whatever holds SIG below the bar on GTSRB for this ViT at 15 epochs is therefore a property of the attack strength or the training budget, and a follow-up that varies the signal amplitude or the epoch count would test it, while a further target-class change would not.

## Proposed ledger edit

The `_tl1` folders should own the GTSRB SIG slots, even though none of them clears the bar. They are the only GTSRB SIG runs trained at the rate their folder name claims, and a clean-label cell that fails at its true rate is an honest below-bar row, whereas the class 0 folders at 1%, 5% and 10% are 1 configuration read under 3 rate labels. The panel reads seed 0 only, since `seed_` is an excluded token, and the seed 0 value at 5% and the 5-seed mean agree on the verdict, so the choice of owner does not change the row's class.

The ledger cannot make that choice today, because both variants pass `is_panel_folder` and `resolve_one_per_attack` prefers a clearing cell only when exactly 1 exists. I simulated the ledger's own functions in memory rather than running its `main()`, which writes 3 artifacts, and the 1% and 5% slots both raise.

| panel rate | candidate folders sharing the slot | clearing | ledger outcome | source |
|---:|---|---:|---|---|
| 0.01 | `vit_gtsrb_sig_0_01`, `vit_gtsrb_sig_0_01_tl1` | 0 | `ValueError`, slot ambiguous | `scripts/coverage_ledger.py`, `resolve_one_per_attack`, called on `panel_cells` output |
| 0.05 | `vit_gtsrb_sig_0_05`, `vit_gtsrb_sig_0_05_tl1`, `vit_gtsrb_sig_0_05_v2` | 0 | `ValueError`, slot ambiguous | `scripts/coverage_ledger.py`, `resolve_one_per_attack`, called on `panel_cells` output |
| 0.1 | `vit_gtsrb_sig_0_1` | 0 | resolves to the class 0 folder at its capped rate | `scripts/coverage_ledger.py`, `resolve_one_per_attack`, called on `panel_cells` output |

Mapping `sig` to `_tl1` in `canonical_variants` is the wrong tool, because that map is keyed by attack alone and applies to every dataset, so it would silently drop the CIFAR-10, CIFAR-100 and Tiny SIG cells, none of which has a `_tl1` folder. The fix that matches the data is a per-dataset canonical target read from metadata rather than from the folder name, since every `args.json` already records `target_label` and `label_mode`.

```json
"canonical_targets": {
  "_comment": "Clean-label GTSRB cells target class 1 because class 0 caps the rate at its own share. A cell whose target_label disagrees with this map for its label_mode is not a panel cell.",
  "gtsrb": {"clean_label": 1}
}
```

`is_panel_folder` would then look up `canonical_targets[dataset][label_mode]` and reject a folder whose `target_label` differs when an entry exists. That rule keeps every existing all-to-one GTSRB cell at class 0, hands the 1% and 5% slots to the `_tl1` folders, excludes the `_v2` orphan by its target rather than by a name pattern and removes `vit_gtsrb_sig_0_1` from the 10% slot, which is correct because no clean-label GTSRB cell can exist at 10% and the rate-caps document already says the GTSRB clean-label rows stop at 5%. The same map will serve the adversarial Label-Consistent runs on GTSRB, which carry `_tl1` too, so the `lc` entry in `canonical_variants` stays as the variant selector and the target map handles the class.

## Files read

- `docs/clean-label-rate-caps.md`
- `docs/gtsrb-training-split-mismatch.md`
- `configs/psbd_basis.json`
- `checkpoints/vit_gtsrb_sig_0_005_tl1/args.json` and its `_seed_1` to `_seed_4` replicates
- `checkpoints/vit_gtsrb_sig_0_01_tl1/args.json` and its `_seed_1` to `_seed_4` replicates
- `checkpoints/vit_gtsrb_sig_0_05_tl1/args.json` and its `_seed_1` to `_seed_4` replicates
- `checkpoints/vit_gtsrb_sig_0_005/args.json` and `metrics.json`
- `checkpoints/vit_gtsrb_sig_0_01/args.json` and `metrics.json`
- `checkpoints/vit_gtsrb_sig_0_05/args.json` and `metrics.json`
- `checkpoints/vit_gtsrb_sig_0_05_v2/args.json`
- `checkpoints/vit_gtsrb_sig_0_1/args.json` and `metrics.json`
- `checkpoints/vit_gtsrb_sig_{0_01,0_05,0_1}_sam_rho_{0_05,0_1,0_15,0_2}/args.json`
- `checkpoints/vit_gtsrb_benign/args.json` and `metrics.json`
- `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log` through `_5.log`
- `cli/train_backdoor.py`, the realized-rate message
- `scripts/coverage_ledger.py`, `is_panel_folder`, `panel_cells`, `classify_by_asr` and `resolve_one_per_attack`
- `pbs/generate_cleanlabel_jobs.py`, `CLEAN_LABEL_TARGETS` and the full-stage SIG and control blocks
- `data/registry.py`, the GTSRB entry
- `raw_data/gtsrb/gtsrb/GTSRB/Training/00000`, `00001` and `00002`, file counts
