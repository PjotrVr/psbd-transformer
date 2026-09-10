# Swin seed replicates and benign references

## Question

The Swin panel gained 3 benign references and 44 training-seed replicates on the night of September 10, and this note asks what those runs say about seed-to-seed spread and which cells can no longer be reported from seed 0 alone. Every number below is read from an `args.json`, a training log or a PBS script, and the path sits beside it in the same table row.

The bars come from the basis config, and the flag threshold on ASR range was set by the question that commissioned this note. Both are listed here so the verdict columns further down can be checked against their source.

| quantity | value | source |
|---|---|---|
| ASR bar an attack must clear | 0.85 | `configs/psbd_basis.json` key `asr_bar` |
| clean accuracy drop bar | -0.05 | `configs/psbd_basis.json` key `clean_accuracy_drop_bar` |
| ASR range that flags a cell | 0.05 | set by the question |

The replicates were trained by the `pbs/swin_trainseed` batch, which is ignored by git and has no generator on disk, so its PBS bodies are the only record of what was run. Each job calls `train_backdoor.py` with `--architecture swin --epochs 15 --seed k` and the seed-tagged output folder, and the `pbs/swin_seedvar` batch that sits next to it is a mask-seed detection sweep over the seed 0 checkpoints rather than a training batch.

| batch fact | value | source |
|---|---|---|
| jobs | 19 | `pbs/swin_trainseed/submit_all.sh` |
| replicate folders trained | 44 | `pbs/swin_trainseed/swinseed_*.pbs` |
| cells | 22 | `checkpoints/swin_*_seed_1/` |
| seeds per cell | 3 | `checkpoints/swin_*[_seed_k]/args.json` |
| walltimes requested | 10:00:00 on 1 job, 16:00:00 on 12, 17:00:00 on 6 | `pbs/swin_trainseed/swinseed_*.pbs` |
| git tracking | ignored by `.gitignore` line 68 pattern `pbs/swin_*/` | `.gitignore` |
| generator docstring | none on disk | `pbs/*.py` grep for `swin_trainseed` |
| detection caches on the 44 replicates | 0 | `results/swin_*_seed_*/psbd/` |
| `swin_seedvar` batch content | `psbd_dropout_sweep.py` at `--mask-seed 1` on seed 0 folders | `pbs/swin_seedvar/swin_1.pbs` |

## Benign references

The 3 new Swin references landed in 1 job on September 10, after a first attempt on September 9 died at argparse in every run because the script passed `--output`, which `train_benign.py` does not take (`logs/swin_benign/benign_failed_2026-09-09.log`), and the rerun in `logs/swin_benign/benign.log` trained all 3 in sequence. The ViT references have no seed, commit or date in their sidecars because their `args.json` was backfilled from the PSBD baseline cache, so the ViT column of the difference table carries less provenance than the Swin column.

| architecture | dataset | clean accuracy | asr field | seed | git commit | trained | source |
|---|---|---|---|---|---|---|---|
| swin | cifar10 | 0.9695 | null | 0 | `2da1b6c` dirty | 2026-09-10 | `checkpoints/swin_cifar10_benign/args.json` |
| swin | cifar100 | 0.8668 | 0.0009 | 0 | `c14f565` | 2026-08-16 | `checkpoints/swin_cifar100_benign/args.json` |
| swin | gtsrb | 0.9852 | null | 0 | `2da1b6c` dirty | 2026-09-10 | `checkpoints/swin_gtsrb_benign/args.json` |
| swin | tiny | 0.8189 | null | 0 | `2da1b6c` dirty | 2026-09-10 | `checkpoints/swin_tiny_benign/args.json` |
| vit | cifar10 | 0.9526 | 0.0111 | null | null | null | `checkpoints/vit_cifar10_benign/args.json` |
| vit | cifar100 | 0.8105 | 0.0001 | null | null | null | `checkpoints/vit_cifar100_benign/args.json` |
| vit | gtsrb | 0.9913 | 0.0001 | null | null | null | `checkpoints/vit_gtsrb_benign/args.json` |
| vit | tiny | 0.7549 | 0.0003 | null | null | null | `checkpoints/vit_tiny_benign/args.json` |

Swin is the stronger clean model on 3 of the 4 datasets and the gap widens with class count, while GTSRB is the 1 dataset where ViT is ahead. The difference column is Swin minus ViT, so a positive value favours Swin.

| dataset | vit clean accuracy | swin clean accuracy | swin minus vit | source |
|---|---|---|---|---|
| cifar10 | 0.9526 | 0.9695 | +0.0169 | `checkpoints/vit_cifar10_benign/args.json`, `checkpoints/swin_cifar10_benign/args.json` |
| cifar100 | 0.8105 | 0.8668 | +0.0563 | `checkpoints/vit_cifar100_benign/args.json`, `checkpoints/swin_cifar100_benign/args.json` |
| gtsrb | 0.9913 | 0.9852 | -0.0061 | `checkpoints/vit_gtsrb_benign/args.json`, `checkpoints/swin_gtsrb_benign/args.json` |
| tiny | 0.7549 | 0.8189 | +0.0640 | `checkpoints/vit_tiny_benign/args.json`, `checkpoints/swin_tiny_benign/args.json` |

The premise that `swin_cifar100_benign` does not exist is false, and the folder was verified on disk with both `attack_result.pt` and `args.json`. It was trained in August at a different commit from the 3 September references, and its clean accuracy is recorded 3 ways that disagree with each other, so a Swin coverage ledger does have a CIFAR-100 clean-accuracy reference but has to pick which of the 3 values to cite.

| fact about `swin_cifar100_benign` | value | source |
|---|---|---|
| folder exists | yes, `args.json` and `attack_result.pt` | `checkpoints/swin_cifar100_benign/` |
| git commit | `c14f565` | `checkpoints/swin_cifar100_benign/args.json` |
| trained | 2026-08-16 | `checkpoints/swin_cifar100_benign/args.json` |
| days before the other 3 references | 25 | `checkpoints/swin_cifar100_benign/args.json`, `checkpoints/swin_cifar10_benign/args.json` |
| clean accuracy, `clean_accuracy` field | 0.8668 | `checkpoints/swin_cifar100_benign/args.json` |
| clean accuracy, `clean_accuracy_from_cache` field | 0.8662 | `checkpoints/swin_cifar100_benign/args.json` |
| clean accuracy, training log at epoch 15 | 0.8645 | `logs/psbd_swin_benign.log` line 25 |
| training duration in minutes | 65.4 | `logs/psbd_swin_benign.log` |

The September job ran under a generous walltime and the 3 runs together used well under half of it. The August CIFAR-100 run took about as long as the CIFAR-10 run, which sets the cost of any rerun discussed in the queue section.

| run | job id | duration in minutes | walltime requested | source |
|---|---|---|---|---|
| `swin_cifar10_benign` | 1044604 | 65.5 | 22:00:00 | `logs/swin_benign/benign.log`, `pbs/swin_benign/benign.pbs` |
| `swin_gtsrb_benign` | 1044604 | 36.2 | 22:00:00 | `logs/swin_benign/benign.log`, `pbs/swin_benign/benign.pbs` |
| `swin_tiny_benign` | 1044604 | 129.8 | 22:00:00 | `logs/swin_benign/benign.log`, `pbs/swin_benign/benign.pbs` |
| failed first attempt, all 3 | 1043955 | 0.3 | not recorded in the log | `logs/swin_benign/benign_failed_2026-09-09.log` |

## Seed spread on Swin

The table below covers every cell that has a `_seed_1` folder, with seed 0 read from the untagged folder and the path pattern `checkpoints/<cell>[_seed_k]/args.json` standing for the 3 sidecars of a cell. ASR is given to 3 decimals and clean accuracy to 4, the range is the maximum minus the minimum over the 3 seeds, and the 3 verdict columns apply the ASR bar to every seed, to the mean of the 3 seeds and to no seed.

| cell | label mode | seeds | ASR seed 0 | ASR seed 1 | ASR seed 2 | ASR range | ASR mean | clean accuracy min | clean accuracy max | clean accuracy range | realized rate | bar on every seed | bar on mean | bar on no seed | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `swin_cifar100_adaptive_blend_0_1` | all_to_one | 0 1 2 | 0.971 | 0.522 | 0.535 | 0.449 | 0.676 | 0.8508 | 0.8639 | 0.0131 | 0.1 | no | no | no | `checkpoints/swin_cifar100_adaptive_blend_0_1[_seed_k]/args.json` |
| `swin_cifar100_badnet_a2o_0_1` | all_to_one | 0 1 2 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.8591 | 0.8635 | 0.0044 | 0.1 | yes | yes | no | `checkpoints/swin_cifar100_badnet_a2o_0_1[_seed_k]/args.json` |
| `swin_cifar100_blend_0_1` | all_to_one | 0 1 2 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.8539 | 0.8645 | 0.0106 | 0.1 | yes | yes | no | `checkpoints/swin_cifar100_blend_0_1[_seed_k]/args.json` |
| `swin_cifar100_bpp_0_1` | all_to_one | 0 1 2 | 0.998 | 0.992 | 0.999 | 0.007 | 0.996 | 0.8495 | 0.8618 | 0.0123 | 0.1 | yes | yes | no | `checkpoints/swin_cifar100_bpp_0_1[_seed_k]/args.json` |
| `swin_cifar100_lc_0_01` | clean_label | 0 1 2 | 0.996 | 0.915 | 0.891 | 0.105 | 0.934 | 0.8526 | 0.8625 | 0.0099 | 0.01 equals the cap | yes | yes | no | `checkpoints/swin_cifar100_lc_0_01[_seed_k]/args.json` |
| `swin_cifar100_lf_0_1` | all_to_one | 0 1 2 | 0.998 | 0.998 | 0.994 | 0.004 | 0.997 | 0.8592 | 0.8600 | 0.0008 | 0.1 | yes | yes | no | `checkpoints/swin_cifar100_lf_0_1[_seed_k]/args.json` |
| `swin_cifar100_wanet_0_1` | all_to_one | 0 1 2 | 0.925 | 0.986 | 0.980 | 0.061 | 0.963 | 0.8494 | 0.8621 | 0.0127 | 0.1 | yes | yes | no | `checkpoints/swin_cifar100_wanet_0_1[_seed_k]/args.json` |
| `swin_cifar10_adaptive_blend_0_1` | all_to_one | 0 1 2 | 0.992 | 0.556 | 0.408 | 0.583 | 0.652 | 0.9543 | 0.9642 | 0.0099 | 0.1 | no | no | no | `checkpoints/swin_cifar10_adaptive_blend_0_1[_seed_k]/args.json` |
| `swin_cifar10_badnet_a2o_0_1` | all_to_one | 0 1 2 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.9593 | 0.9677 | 0.0084 | 0.1 | yes | yes | no | `checkpoints/swin_cifar10_badnet_a2o_0_1[_seed_k]/args.json` |
| `swin_cifar10_blend_0_1` | all_to_one | 0 1 2 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.9585 | 0.9697 | 0.0112 | 0.1 | yes | yes | no | `checkpoints/swin_cifar10_blend_0_1[_seed_k]/args.json` |
| `swin_cifar10_bpp_0_1` | all_to_one | 0 1 2 | 1.000 | 0.999 | 0.999 | 0.001 | 0.999 | 0.9632 | 0.9675 | 0.0043 | 0.1 | yes | yes | no | `checkpoints/swin_cifar10_bpp_0_1[_seed_k]/args.json` |
| `swin_cifar10_lc_0_1` | clean_label | 0 1 2 | 0.953 | 0.977 | 0.987 | 0.034 | 0.972 | 0.8694 | 0.8725 | 0.0031 | 0.1 equals the cap | yes | yes | no | `checkpoints/swin_cifar10_lc_0_1[_seed_k]/args.json` |
| `swin_cifar10_lf_0_1` | all_to_one | 0 1 2 | 0.998 | 1.000 | 1.000 | 0.001 | 0.999 | 0.9572 | 0.9673 | 0.0101 | 0.1 | yes | yes | no | `checkpoints/swin_cifar10_lf_0_1[_seed_k]/args.json` |
| `swin_cifar10_sig_0_1` | clean_label | 0 1 2 | 0.945 | 0.888 | 0.986 | 0.098 | 0.940 | 0.8690 | 0.8721 | 0.0031 | 0.1 equals the cap | yes | yes | no | `checkpoints/swin_cifar10_sig_0_1[_seed_k]/args.json` |
| `swin_cifar10_wanet_0_1` | all_to_one | 0 1 2 | 0.977 | 0.966 | 0.943 | 0.034 | 0.962 | 0.9594 | 0.9692 | 0.0098 | 0.1 | yes | yes | no | `checkpoints/swin_cifar10_wanet_0_1[_seed_k]/args.json` |
| `swin_gtsrb_adaptive_blend_0_1` | all_to_one | 0 1 2 | 1.000 | 0.411 | 0.477 | 0.588 | 0.629 | 0.9771 | 0.9836 | 0.0065 | 0.1 | no | no | no | `checkpoints/swin_gtsrb_adaptive_blend_0_1[_seed_k]/args.json` |
| `swin_gtsrb_badnet_a2o_0_1` | all_to_one | 0 1 2 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.9827 | 0.9893 | 0.0067 | 0.1 | yes | yes | no | `checkpoints/swin_gtsrb_badnet_a2o_0_1[_seed_k]/args.json` |
| `swin_gtsrb_blend_0_1` | all_to_one | 0 1 2 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.9832 | 0.9869 | 0.0037 | 0.1 | yes | yes | no | `checkpoints/swin_gtsrb_blend_0_1[_seed_k]/args.json` |
| `swin_gtsrb_bpp_0_1` | all_to_one | 0 1 2 | 1.000 | 0.998 | 0.996 | 0.004 | 0.998 | 0.9805 | 0.9867 | 0.0062 | 0.1 | yes | yes | no | `checkpoints/swin_gtsrb_bpp_0_1[_seed_k]/args.json` |
| `swin_gtsrb_lc_0_1` | clean_label | 0 1 2 | 0.991 | 0.477 | 0.636 | 0.514 | 0.701 | 0.9781 | 0.9850 | 0.0070 | 0.0056 capped from 0.1, 150 images | no | no | no | `checkpoints/swin_gtsrb_lc_0_1[_seed_k]/args.json` |
| `swin_gtsrb_lf_0_1` | all_to_one | 0 1 2 | 0.995 | 1.000 | 0.998 | 0.005 | 0.998 | 0.9842 | 0.9893 | 0.0051 | 0.1 | yes | yes | no | `checkpoints/swin_gtsrb_lf_0_1[_seed_k]/args.json` |
| `swin_gtsrb_wanet_0_1` | all_to_one | 0 1 2 | 0.955 | 0.984 | 0.975 | 0.028 | 0.971 | 0.9694 | 0.9824 | 0.0130 | 0.1 | yes | yes | no | `checkpoints/swin_gtsrb_wanet_0_1[_seed_k]/args.json` |

The dirty-label patch and blend families are saturated on every seed and their spread is at the third decimal, while the whole of the spread lives in adaptive_blend and the clean-label cells. No cell fails the bar on every seed, so the panel has no attack that reliably does not implant, and the 4 cells that fail on the mean are exactly the cells where the verdict depends on which seed is read.

| tally over the 22 cells | count | source |
|---|---|---|
| cells clearing the bar on every seed | 18 | table above |
| cells clearing the bar on the mean | 18 | table above |
| cells clearing the bar on no seed | 0 | table above |
| cells whose verdict flips between seeds | 4 | table above |
| cells with ASR range above the flag threshold | 7 | table above |
| cells with ASR range under the flag threshold | 15 | table above |
| largest clean accuracy range over seeds | 0.0131 | `checkpoints/swin_cifar100_adaptive_blend_0_1[_seed_k]/args.json` |

The seed 0 folders are older than their replicates by the gap in the table below and were trained by different code, so the seed 0 against seed 1 and 2 comparison is a code-version comparison before it is a seed comparison. The July runs carry no seed at all, because the seeding fix landed after the last of them had finished, and for adaptive_blend the September runs plant the asymmetric trigger and build cover samples that the July runs never had, so the drop from seed 0 to the replicates on that attack cannot be read as seed noise.

| property | seed 0 originals | seeds 1 and 2 | source |
|---|---|---|---|
| trained | 2026-07-09 to 2026-07-10 | 2026-09-10 | `checkpoints/<cell>/attack_result.pt` mtime, `checkpoints/<cell>_seed_k/args.json` |
| days from the last July run to the seeding fix | 9 | | `checkpoints/<cell>/attack_result.pt` mtime, git history of `train_backdoor.py` |
| `seed` field | null | 1 or 2 | `checkpoints/<cell>[_seed_k]/args.json` |
| `git_commit` field | null | `2da1b6c` dirty on 42 runs, `b551d5c` on 2 | `checkpoints/<cell>_seed_k/args.json` |
| `Seed set to` lines in the training log | 0 | present on every run | `logs/swin_no_sam_individual/train_swin_cifar10_adaptive_blend.out`, `logs/swin_trainseed/swinseed_1.log` |
| seeding fix `9f6acfb` of 2026-07-19 | predates it | includes it | git history of `train_backdoor.py` |
| adaptive_blend trigger fix `33696d3` of 2026-09-09 | predates it, symmetric trigger | includes it, subset at train and full pattern at eval | git history of `attacks/adaptive_blend.py` |
| adaptive_blend cover samples | 0 cover lines in the log | cover rate 0.1, 5000 samples on CIFAR and 2664 on GTSRB | `logs/swin_no_sam_individual/train_swin_*_adaptive_blend.out`, `logs/swin_trainseed/swinseed_{1,2,3}.log` |
| `cover_rate` field | 0.0 | 0.0 although the log shows 0.1 | `checkpoints/swin_*_adaptive_blend_0_1[_seed_k]/args.json` |
| ASR source | PSBD baseline cache re-evaluation | training-time evaluation | `checkpoints/<cell>[_seed_k]/args.json` key `asr_source` |
| the 2 runs at `b551d5c` | | `swin_gtsrb_adaptive_blend_0_1_seed_1`, `swin_gtsrb_blend_0_1_seed_1` | `checkpoints/swin_gtsrb_*_0_1_seed_1/args.json` |

The new benign references also make the clean accuracy drop bar computable on Swin for the first time, and the breaching cells are listed first in the table below. Both are the CIFAR-10 clean-label cells at the rate that poisons the whole target class, and their drop is about the share of a single class in that dataset, which is what a model that has stopped recognising clean images of the target class would show.

| cell | swin benign clean accuracy | worst clean accuracy over seeds | worst drop | breaches the drop bar | source |
|---|---|---|---|---|---|
| `swin_cifar10_lc_0_1` | 0.9695 | 0.8694 | -0.1001 | yes on every seed | `checkpoints/swin_cifar10_benign/args.json`, `checkpoints/swin_cifar10_lc_0_1[_seed_k]/args.json` |
| `swin_cifar10_sig_0_1` | 0.9695 | 0.8690 | -0.1005 | yes on every seed | `checkpoints/swin_cifar10_benign/args.json`, `checkpoints/swin_cifar10_sig_0_1[_seed_k]/args.json` |
| `swin_cifar100_wanet_0_1` | 0.8668 | 0.8494 | -0.0174 | no | `checkpoints/swin_cifar100_benign/args.json`, `checkpoints/swin_cifar100_wanet_0_1[_seed_k]/args.json` |
| `swin_cifar100_bpp_0_1` | 0.8668 | 0.8495 | -0.0173 | no | `checkpoints/swin_cifar100_benign/args.json`, `checkpoints/swin_cifar100_bpp_0_1[_seed_k]/args.json` |
| `swin_cifar100_adaptive_blend_0_1` | 0.8668 | 0.8508 | -0.0160 | no | `checkpoints/swin_cifar100_benign/args.json`, `checkpoints/swin_cifar100_adaptive_blend_0_1[_seed_k]/args.json` |

## Cells that must not be reported from 1 seed

The rule applied here is that a cell whose ASR range over 3 seeds exceeds the flag threshold has no single-seed number worth citing, and a cell whose bar verdict flips between seeds has no single-seed verdict either. The July training-time ASR is listed next to the cached value so the reader can see that the seed 0 numbers for the clean-label cells were re-evaluated after the clean-label eligibility fix and are not the pre-fix log values.

| cell | family | ASR range | verdict flips | seed 0 is the highest seed | July training-time ASR of seed 0 | cached ASR of seed 0 | source |
|---|---|---|---|---|---|---|---|
| `swin_gtsrb_adaptive_blend_0_1` | adaptive_blend | 0.588 | yes | yes | 1.000 | 1.000 | `logs/swin_no_sam_individual/train_swin_gtsrb_adaptive_blend.out`, `checkpoints/swin_gtsrb_adaptive_blend_0_1[_seed_k]/args.json` |
| `swin_cifar10_adaptive_blend_0_1` | adaptive_blend | 0.583 | yes | yes | 0.992 | 0.992 | `logs/swin_no_sam_individual/train_swin_cifar10_adaptive_blend.out`, `checkpoints/swin_cifar10_adaptive_blend_0_1[_seed_k]/args.json` |
| `swin_gtsrb_lc_0_1` | clean_label | 0.514 | yes | yes | 1.000 | 0.991 | `logs/swin_no_sam_individual/train_swin_gtsrb_lc.out`, `checkpoints/swin_gtsrb_lc_0_1[_seed_k]/args.json` |
| `swin_cifar100_adaptive_blend_0_1` | adaptive_blend | 0.449 | yes | yes | 0.972 | 0.971 | `logs/swin_no_sam_individual/train_swin_cifar100_adaptive_blend.out`, `checkpoints/swin_cifar100_adaptive_blend_0_1[_seed_k]/args.json` |
| `swin_cifar100_lc_0_01` | clean_label | 0.105 | no | yes | 1.000 | 0.996 | `logs/swin_no_sam_individual/train_swin_cifar100_lc.out`, `checkpoints/swin_cifar100_lc_0_01[_seed_k]/args.json` |
| `swin_cifar10_sig_0_1` | clean_label | 0.098 | no | no | 1.000 | 0.945 | `logs/swin_no_sam_individual/train_swin_cifar10_sig.out`, `checkpoints/swin_cifar10_sig_0_1[_seed_k]/args.json` |
| `swin_cifar100_wanet_0_1` | wanet | 0.061 | no | no | 0.924 | 0.925 | `logs/swin_no_sam_individual/train_swin_cifar100_wanet.out`, `checkpoints/swin_cifar100_wanet_0_1[_seed_k]/args.json` |

The claim that variance concentrates in adaptive_blend and the clean-label attacks is verified, with wanet as a marginal third family that never flips a verdict. Every adaptive_blend cell flips and 3 of the 4 clean-label cells are flagged, while the 12 dirty-label patch, blend, bpp and lf cells sit under the threshold by more than an order of magnitude.

| family | cells with replicates | cells flagged | cells whose verdict flips | source |
|---|---|---|---|---|
| adaptive_blend | 3 | 3 | 3 | spread table above |
| clean_label, lc and sig | 4 | 3 | 1 | spread table above |
| wanet | 3 | 1 | 0 | spread table above |
| badnet_a2o, blend, bpp, lf | 12 | 0 | 0 | spread table above |

The `swin_gtsrb_lc_0_1` cell is the clearest case of a single-seed number that must not be cited. Its folder name promises a 10 percent poison rate, but class 0 of the GTSRB training split holds too few images for that and the selection clamps to the whole class, so all 3 seeds trained on the same 150 poisoned images and differ only in initialisation and batch order.

| seed | ASR | clean accuracy | requested rate | realized rate | poisoned images | training pool | source |
|---|---|---|---|---|---|---|---|
| 0 | 0.991 | 0.9850 | 0.1 | 0.0056 | 150 | 26640 | `checkpoints/swin_gtsrb_lc_0_1/args.json` |
| 1 | 0.477 | 0.9812 | 0.1 | 0.0056 | 150 | 26640 | `checkpoints/swin_gtsrb_lc_0_1_seed_1/args.json` |
| 2 | 0.636 | 0.9781 | 0.1 | 0.0056 | 150 | 26640 | `checkpoints/swin_gtsrb_lc_0_1_seed_2/args.json` |
| mean | 0.701 | | | | | | computed from the 3 rows |

Any claim that Label-Consistent implants on Swin GTSRB rests, at seed 0, on an unseeded July run that happened to land near the ceiling, and the 2 seeded runs at the current commit both fall below the bar with the mean below it too. The honest reading of this cell is a coin flip at a poison budget of 150 images, and the panel already has the fix for the underlying cap in `docs/clean-label-rate-caps.md`, which moves GTSRB clean-label cells to target class 1 under a `_tl1` tag so the requested rate is actually realized.

## Seed replicates on ViT

The ViT panel does carry seed replicates, from 4 training batches, and the table lists every cell that has at least 1 `_seed_k` folder with the batch that trained it. The GTSRB SIG `_tl1` cells and the SVHN and EuroSAT probes are listed for completeness only, since other notes analyse them, and the ASR per seed is given to 3 decimals with the range to 4 so a value near the flag threshold can be placed on the correct side.

| batch | cells | replicate folders | seeds added | source |
|---|---|---|---|---|
| `pbs/psbd_seed` | 26 | 52 | 1 2 | `pbs/psbd_seed/seed_*.pbs`, `pbs/generate_seed_jobs.py` |
| `pbs/vit_cleanlabel_sig_gtsrb` | 3 | 12 | 1 2 3 4 | `pbs/vit_cleanlabel_sig_gtsrb/*.pbs` |
| `pbs/vit_newdata_probe_seeds` | 8 | 16 | 1 2 | `pbs/vit_newdata_probe_seeds/*.pbs` |
| `pbs/vit_multitarget_seeds` | 4 | 8 | 1 2 | `pbs/vit_multitarget_seeds/*.pbs` |
| all batches | 41 | 88 | | `checkpoints/vit_*_seed_*/` |
| ViT cells with ASR range above the flag threshold | 23 | | | table below |
| ViT benign seed replicates on disk | 0 | | | `checkpoints/vit_*benign*seed*` absent |

| cell | batch | seed 0 field | seeds | ASR per seed in seed order | ASR range | source |
|---|---|---|---|---|---|---|
| `vit_cifar100_adaptive_blend_0_01` | psbd_seed | 0 | 0 1 2 | 0.536 0.616 0.571 | 0.0800 | `checkpoints/vit_cifar100_adaptive_blend_0_01[_seed_k]/args.json` |
| `vit_cifar100_adaptive_blend_0_05` | psbd_seed | 0 | 0 1 2 | 0.579 0.774 0.914 | 0.3353 | `checkpoints/vit_cifar100_adaptive_blend_0_05[_seed_k]/args.json` |
| `vit_cifar100_adaptive_blend_0_1` | psbd_seed | 0 | 0 1 2 | 0.604 0.949 0.974 | 0.3706 | `checkpoints/vit_cifar100_adaptive_blend_0_1[_seed_k]/args.json` |
| `vit_cifar100_badnet_a2o_0_01` | psbd_seed | null | 0 1 2 | 1.000 0.999 1.000 | 0.0010 | `checkpoints/vit_cifar100_badnet_a2o_0_01[_seed_k]/args.json` |
| `vit_cifar100_badnet_a2o_0_05` | psbd_seed | null | 0 1 2 | 1.000 1.000 1.000 | 0.0002 | `checkpoints/vit_cifar100_badnet_a2o_0_05[_seed_k]/args.json` |
| `vit_cifar100_badnet_a2o_0_1` | psbd_seed | null | 0 1 2 | 1.000 1.000 1.000 | 0.0000 | `checkpoints/vit_cifar100_badnet_a2o_0_1[_seed_k]/args.json` |
| `vit_cifar100_blend_0_01` | psbd_seed | null | 0 1 2 | 0.989 0.997 0.988 | 0.0089 | `checkpoints/vit_cifar100_blend_0_01[_seed_k]/args.json` |
| `vit_cifar100_blend_0_05` | psbd_seed | null | 0 1 2 | 1.000 1.000 0.999 | 0.0011 | `checkpoints/vit_cifar100_blend_0_05[_seed_k]/args.json` |
| `vit_cifar100_blend_0_1` | psbd_seed | null | 0 1 2 | 1.000 0.999 1.000 | 0.0008 | `checkpoints/vit_cifar100_blend_0_1[_seed_k]/args.json` |
| `vit_cifar100_lc_0_01` | psbd_seed | null | 0 1 2 | 0.873 0.744 0.848 | 0.1291 | `checkpoints/vit_cifar100_lc_0_01[_seed_k]/args.json` |
| `vit_cifar100_lc_0_05` | psbd_seed | null | 0 1 2 | 0.545 0.912 0.844 | 0.3675 | `checkpoints/vit_cifar100_lc_0_05[_seed_k]/args.json` |
| `vit_cifar100_lc_0_1` | psbd_seed | null | 0 1 2 | 0.785 0.887 0.882 | 0.1015 | `checkpoints/vit_cifar100_lc_0_1[_seed_k]/args.json` |
| `vit_cifar100_sig_0_01_m2` | vit_multitarget_seeds | 0 | 0 1 2 | 0.185 0.391 0.053 | 0.3380 | `checkpoints/vit_cifar100_sig_0_01_m2[_seed_k]/args.json` |
| `vit_cifar100_sig_0_01_m3` | vit_multitarget_seeds | 0 | 0 1 2 | 0.500 0.632 0.429 | 0.2034 | `checkpoints/vit_cifar100_sig_0_01_m3[_seed_k]/args.json` |
| `vit_cifar100_wanet_0_05` | psbd_seed | 0 | 0 1 2 | 0.643 0.605 0.713 | 0.1079 | `checkpoints/vit_cifar100_wanet_0_05[_seed_k]/args.json` |
| `vit_cifar100_wanet_0_1` | psbd_seed | 0 | 0 1 2 | 0.793 0.896 0.790 | 0.1061 | `checkpoints/vit_cifar100_wanet_0_1[_seed_k]/args.json` |
| `vit_eurosat_blend_0_1` | vit_newdata_probe_seeds | 0 | 0 1 2 | 1.000 1.000 1.000 | 0.0000 | `checkpoints/vit_eurosat_blend_0_1[_seed_k]/args.json` |
| `vit_eurosat_sig_0_01` | vit_newdata_probe_seeds | 0 | 0 1 2 | 0.813 0.762 0.695 | 0.1179 | `checkpoints/vit_eurosat_sig_0_01[_seed_k]/args.json` |
| `vit_eurosat_sig_0_05` | vit_newdata_probe_seeds | 0 | 0 1 2 | 0.873 0.887 0.922 | 0.0494 | `checkpoints/vit_eurosat_sig_0_05[_seed_k]/args.json` |
| `vit_eurosat_sig_0_1` | vit_newdata_probe_seeds | 0 | 0 1 2 | 0.922 0.844 0.871 | 0.0778 | `checkpoints/vit_eurosat_sig_0_1[_seed_k]/args.json` |
| `vit_gtsrb_sig_0_005_tl1` | vit_cleanlabel_sig_gtsrb | 0 | 0 1 2 3 4 | 0.347 0.521 0.507 0.154 0.229 | 0.3669 | `checkpoints/vit_gtsrb_sig_0_005_tl1[_seed_k]/args.json` |
| `vit_gtsrb_sig_0_01_tl1` | vit_cleanlabel_sig_gtsrb | 0 | 0 1 2 3 4 | 0.356 0.485 0.633 0.386 0.557 | 0.2762 | `checkpoints/vit_gtsrb_sig_0_01_tl1[_seed_k]/args.json` |
| `vit_gtsrb_sig_0_05_tl1` | vit_cleanlabel_sig_gtsrb | 0 | 0 1 2 3 4 | 0.673 0.879 0.600 0.823 0.637 | 0.2791 | `checkpoints/vit_gtsrb_sig_0_05_tl1[_seed_k]/args.json` |
| `vit_svhn_blend_0_1` | vit_newdata_probe_seeds | 0 | 0 1 2 | 1.000 1.000 1.000 | 0.0003 | `checkpoints/vit_svhn_blend_0_1[_seed_k]/args.json` |
| `vit_svhn_sig_0_01_tl1` | vit_newdata_probe_seeds | 0 | 0 1 2 | 0.700 0.696 0.784 | 0.0880 | `checkpoints/vit_svhn_sig_0_01_tl1[_seed_k]/args.json` |
| `vit_svhn_sig_0_05_tl1` | vit_newdata_probe_seeds | 0 | 0 1 2 | 0.880 0.930 0.918 | 0.0495 | `checkpoints/vit_svhn_sig_0_05_tl1[_seed_k]/args.json` |
| `vit_svhn_sig_0_1_tl1` | vit_newdata_probe_seeds | 0 | 0 1 2 | 0.971 0.994 0.991 | 0.0224 | `checkpoints/vit_svhn_sig_0_1_tl1[_seed_k]/args.json` |
| `vit_tiny_adaptive_blend_0_05` | psbd_seed | 0 | 0 1 2 | 0.783 0.961 0.951 | 0.1781 | `checkpoints/vit_tiny_adaptive_blend_0_05[_seed_k]/args.json` |
| `vit_tiny_adaptive_blend_0_1` | psbd_seed | 0 | 0 1 2 | 0.505 0.989 0.920 | 0.4840 | `checkpoints/vit_tiny_adaptive_blend_0_1[_seed_k]/args.json` |
| `vit_tiny_badnet_a2o_0_01` | psbd_seed | null | 0 1 2 | 0.999 1.000 1.000 | 0.0006 | `checkpoints/vit_tiny_badnet_a2o_0_01[_seed_k]/args.json` |
| `vit_tiny_badnet_a2o_0_05` | psbd_seed | null | 0 1 2 | 1.000 1.000 1.000 | 0.0003 | `checkpoints/vit_tiny_badnet_a2o_0_05[_seed_k]/args.json` |
| `vit_tiny_badnet_a2o_0_1` | psbd_seed | null | 0 1 2 | 1.000 1.000 1.000 | 0.0000 | `checkpoints/vit_tiny_badnet_a2o_0_1[_seed_k]/args.json` |
| `vit_tiny_blend_0_01` | psbd_seed | null | 0 1 2 | 0.999 1.000 1.000 | 0.0012 | `checkpoints/vit_tiny_blend_0_01[_seed_k]/args.json` |
| `vit_tiny_blend_0_05` | psbd_seed | null | 0 1 2 | 0.999 1.000 0.997 | 0.0031 | `checkpoints/vit_tiny_blend_0_05[_seed_k]/args.json` |
| `vit_tiny_blend_0_1` | psbd_seed | null | 0 1 2 | 1.000 1.000 1.000 | 0.0004 | `checkpoints/vit_tiny_blend_0_1[_seed_k]/args.json` |
| `vit_tiny_lc_0_05` | psbd_seed | null | 0 1 2 | 0.706 0.460 0.331 | 0.3746 | `checkpoints/vit_tiny_lc_0_05[_seed_k]/args.json` |
| `vit_tiny_lc_0_1` | psbd_seed | null | 0 1 2 | 0.623 0.320 0.269 | 0.3535 | `checkpoints/vit_tiny_lc_0_1[_seed_k]/args.json` |
| `vit_tiny_sig_0_01_m2` | vit_multitarget_seeds | 0 | 0 1 2 | 0.642 0.567 0.663 | 0.0964 | `checkpoints/vit_tiny_sig_0_01_m2[_seed_k]/args.json` |
| `vit_tiny_sig_0_01_m3` | vit_multitarget_seeds | 0 | 0 1 2 | 0.508 0.496 0.576 | 0.0806 | `checkpoints/vit_tiny_sig_0_01_m3[_seed_k]/args.json` |
| `vit_tiny_wanet_0_05` | psbd_seed | 0 | 0 1 2 | 0.922 0.945 0.828 | 0.1170 | `checkpoints/vit_tiny_wanet_0_05[_seed_k]/args.json` |
| `vit_tiny_wanet_0_1` | psbd_seed | 0 | 0 1 2 | 0.968 0.967 0.931 | 0.0371 | `checkpoints/vit_tiny_wanet_0_1[_seed_k]/args.json` |

The same families carry the spread on ViT as on Swin, with adaptive_blend, lc and sig dominating the flagged list and the patch and blend cells saturated. The `psbd_seed` batch also scheduled benign replicates for CIFAR-100 and Tiny, and those jobs died at the same `--output` argparse error that killed the first Swin benign attempt, so no ViT benign replicate exists on disk.

| planned ViT benign replicate | job date | outcome | source |
|---|---|---|---|
| `vit_tiny_benign_seed_1` | 2026-09-07 | argparse error on `--output`, no folder | `logs/psbd_seed/seed_021.log` |
| `vit_tiny_benign_seed_2` | 2026-09-07 | argparse error on `--output`, no folder | `logs/psbd_seed/seed_022.log` |
| `vit_cifar100_benign_seed_1`, `vit_cifar100_benign_seed_2` | 2026-09-07 | scheduled in the same batch, no folder | `pbs/psbd_seed/seed_005.pbs` |

## Proposed queue

Every Swin cell in the spread table already holds 3 seeds, so the gap is not a missing third draw but the fact that seed 0 is an unseeded July run under older attack code. The proposal is therefore to add seed 3 at the current commit for the 7 flagged cells, so that each has 3 seeded same-code runs (seeds 1 to 3) and the July run can be retired from any spread claim, while the 15 saturated cells keep their July seed 0 because no bar verdict there depends on it.

| item | cells | runs | minutes per run | GPU hours | priority | source of the per-run cost |
|---|---|---|---|---|---|---|
| seed 3 at the current commit | `swin_cifar100_adaptive_blend_0_1`, `swin_cifar100_lc_0_01`, `swin_cifar100_wanet_0_1`, `swin_cifar10_adaptive_blend_0_1`, `swin_cifar10_sig_0_1` | 5 | 66 | 5.5 | 1 | `logs/swin_trainseed/swinseed_*.log` |
| seed 3 at the current commit | `swin_gtsrb_adaptive_blend_0_1`, `swin_gtsrb_lc_0_1` | 2 | 36 | 1.2 | 1 | `logs/swin_trainseed/swinseed_*.log` |
| GTSRB LC rebuilt at target class 1 with adversarial bases, seeds 1 to 3 | replacement for `swin_gtsrb_lc_0_1` | 3 | 36 | 1.8 | 2 | `logs/swin_trainseed/swinseed_13.log`, `docs/clean-label-rate-caps.md` |
| metrics re-evaluation of `swin_cifar100_benign` to settle its clean accuracy | 1 | 1 evaluation | under 10 | under 0.2 | 3 | `logs/psbd_swin_benign.log` |
| `swin_cifar100_benign` training rerun at the current commit | 1 | 0 recommended, 1 optional | 65.4 | 1.1 | 4 | `logs/psbd_swin_benign.log`, `pbs/swin_benign/benign.pbs` walltime 22:00:00 |
| detection sweeps on the 44 Swin replicates | 22 | 44 | not measured here | not estimated | after item 1 | `results/swin_*_seed_*/psbd/` empty |
| ViT benign replicates for CIFAR-100 and Tiny | 2 | 4 | 348 and 692 | 34.7 | blocked | `pbs/generate_seed_jobs.py` `MEDIAN_MINUTES` |

A `swin_cifar100_benign` training run should not be queued for the reason given in the question, because the reference exists and a Swin ledger can cite it today. What the reference lacks is commit parity with the other 3 and 1 agreed clean accuracy value, and the second problem is solved by a re-evaluation rather than a retrain, so the retrain stays optional and last.

The GTSRB LC replacement follows the ViT panel's own fix rather than adding seeds to a capped cell, since a seed 3 of `swin_gtsrb_lc_0_1` would be a fourth draw over the same 150 images. The ViT benign replicates remain blocked until `train_benign.py` can write to a seed-tagged folder, because it derives `{architecture}_{dataset}_benign` itself and refuses `--output`, which is exactly how both the first Swin attempt and the earlier ViT jobs died.

## Files read

| file | what was read |
|---|---|
| `checkpoints/swin_{cifar10,cifar100,gtsrb,tiny}_benign/args.json` | Swin benign clean accuracy, asr field, seed, commit, dates |
| `checkpoints/vit_{cifar10,cifar100,gtsrb,tiny}_benign/args.json` | ViT benign clean accuracy and provenance |
| `checkpoints/swin_*_seed_{1,2}/args.json` | 44 replicate ASR, clean accuracy, seed, commit, cover_rate, realized rate |
| `checkpoints/swin_<cell>/args.json` for the 22 seed 0 originals | seed 0 ASR, clean accuracy, asr_source, poison_rate_capped, n_poisoned |
| `checkpoints/swin_<cell>/attack_result.pt` | mtime dating the seed 0 originals to July |
| `checkpoints/vit_*_seed_*/args.json` and their seed 0 originals | ViT replicate inventory and ASR |
| `configs/psbd_basis.json` | asr_bar, clean_accuracy_drop_bar, benign_reference map |
| `pbs/swin_trainseed/swinseed_*.pbs`, `pbs/swin_trainseed/submit_all.sh` | the training batch, walltimes, commands |
| `pbs/swin_seedvar/swin_1.pbs` | confirmation that this batch is a mask-seed detection sweep |
| `pbs/swin_benign/benign.pbs` | walltime and the corrected `train_benign.py` invocation |
| `pbs/generate_seed_jobs.py`, `pbs/generate_seed_replicate_jobs.py`, `pbs/generate_seed_variance_jobs.py` | docstrings, tiering, per-run cost medians |
| `pbs/psbd_seed/*.pbs`, `pbs/vit_cleanlabel_sig_gtsrb/*.pbs`, `pbs/vit_newdata_probe_seeds/*.pbs`, `pbs/vit_multitarget_seeds/*.pbs` | which batch trained each ViT replicate |
| `logs/swin_benign/benign.log`, `logs/swin_benign/benign_failed_2026-09-09.log` | the failed and the successful Swin benign runs, durations, job ids |
| `logs/psbd_swin_benign.log` | the August `swin_cifar100_benign` training run and its logged clean accuracy |
| `logs/swin_trainseed/swinseed_*.log` | replicate training-time ASR, durations, cover rate lines, seeding lines |
| `logs/swin_no_sam_individual/train_swin_*.out` | July training-time ASR of the seed 0 originals, absence of seeding and cover lines |
| `logs/psbd_seed/seed_021.log`, `logs/psbd_seed/seed_022.log` | the failed ViT benign replicate jobs |
| `results/swin_*_seed_*/psbd/` | absence of detection caches on the replicates |
| `.gitignore` | the `pbs/swin_*/` ignore pattern |
| git history of `attacks/adaptive_blend.py`, `attacks/lc.py`, `train_backdoor.py` | commits `9f6acfb`, `33696d3`, `207dbf8`, `6d980f2` and their messages |
| `train_backdoor.py` | `resolve_cover_rate` and the cover rate multiples |
| `docs/clean-label-rate-caps.md` | the GTSRB class 0 cap and the `_tl1` fix |
| `docs/seed-replication-plan.md` | the stated seed target and tiering |
| `.claude/styles/writing-style.md` | the prose rules this note follows |
