# The Label-Consistent epsilon per dataset

The full stage of the clean-label fix trains Label-Consistent with its adversarial bases at every reachable rate and at every seed, with a single epsilon per dataset. This entry picks that epsilon from the pilot batch, records where the rule gives no answer and lays out the choices that remain on the datasets where it does not.

| provenance | value | source |
|---|---|---|
| pilot batch | `logs/vit_cleanlabel_pilot/` | `/lustre/home/pstika/projects/PSBD-ViT/logs/vit_cleanlabel_pilot/` |
| PBS jobs | 1044629 to 1044635 | `/lustre/home/pstika/projects/PSBD-ViT/logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_1.log` to `_7.log` |
| pilot started | 2026-09-10 | `/lustre/home/pstika/projects/PSBD-ViT/logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_1.log` |
| training commits | `2da1b6c` dirty and `a7a0003` dirty | `/lustre/home/pstika/projects/PSBD-ViT/checkpoints/vit_*_lc_*_adv*_pilot/args.json` |
| bases commit | `2da1b6c` dirty | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/*/manifest.json` |
| pilot walltime, 2 CIFAR runs per job | 13:00:00 | `/lustre/home/pstika/projects/PSBD-ViT/pbs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_1.pbs` |
| pilot walltime, 3 GTSRB runs per job | 11:00:00 | `/lustre/home/pstika/projects/PSBD-ViT/pbs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.pbs` |

## Question and rule

The rule comes from the generator's docstring: the smallest epsilon clearing the ASR bar without costing more than the clean-accuracy drop bar, read on a single seed per dataset and epsilon at each dataset's highest reachable clean-label rate. The rate is the highest reachable one on purpose, because an epsilon that cannot implant at all fails most visibly there. Both bars are declared once in the basis file and the drop is measured against the dataset's benign reference at the same architecture, optimizer and epoch count, which is what makes it like-for-like.

| bar | value | applied as | source |
|---|---|---|---|
| `asr_bar` | 0.85 | ASR at or above the bar | `configs/psbd_basis.json` |
| `clean_accuracy_drop_bar` | -0.05 | CA minus benign CA at or above the bar | `configs/psbd_basis.json` |

| dataset | benign clean accuracy | source |
|---|---|---|
| cifar10 | 0.9526 | `checkpoints/vit_cifar10_benign/args.json` |
| cifar100 | 0.8105 | `checkpoints/vit_cifar100_benign/args.json` |
| gtsrb | 0.9913 | `checkpoints/vit_gtsrb_benign/args.json` |
| tiny | 0.7549 | `checkpoints/vit_tiny_benign/args.json` |

A clean-label attack may only poison images that already carry the target label, so the highest reachable rate is the target class size over the training set size, and every rate above it silently trains the identical index set. GTSRB is the only dataset where the target class choice moves that cap, which is why its clean-label runs sit at the larger target class and carry the `_tl1` tag, while the class-uniform datasets stay at the default class.

| dataset | target class | class size | training set | cap | pilot rate | source |
|---|---|---|---|---|---|---|
| cifar10 | 0 | 5000 | 50000 | 10.00% | 0.1 | `docs/clean-label-rate-caps.md` |
| cifar100 | 0 | 500 | 50000 | 1.00% | 0.01 | `docs/clean-label-rate-caps.md` |
| gtsrb | 1 | 1500 | 26640 | 5.63% | 0.05 | `docs/clean-label-rate-caps.md` |
| tiny | 0 | 500 | 100000 | 0.50% | 0.005 | `docs/clean-label-rate-caps.md` |

## The adversarial bases

Each base set is a single untargeted PGD pass over the target class of the ViT benign surrogate, at the dataset's native resolution, and its manifest records the surrogate's accuracy on that class before and after the perturbation. An attacked accuracy that collapses means the natural features stopped supporting the label, which is the whole point of Turner's step, and every set reaches it except GTSRB at the smallest budget. There the smallest epsilon leaves about a third of the class still recognisable, so that base set is weaker by construction and its pilot is not the same attack at a smaller epsilon.

| dataset | target | epsilon over 255 | epsilon | steps | step size | images | surrogate accuracy on target class | attacked accuracy on target class | manifest |
|---|---|---|---|---|---|---|---|---|---|
| cifar10 | 0 | 8 | 0.031373 | 100 | 0.00078432 | 5000 | 0.9984 | 0.0002 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/cifar10_tl0_eps8/manifest.json` |
| cifar10 | 0 | 16 | 0.062745 | 100 | 0.00156863 | 5000 | 0.9984 | 0.0000 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/cifar10_tl0_eps16/manifest.json` |
| cifar10 | 0 | 32 | 0.125490 | 100 | 0.00313725 | 5000 | 0.9984 | 0.0000 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/cifar10_tl0_eps32/manifest.json` |
| cifar100 | 0 | 8 | 0.031373 | 100 | 0.00078432 | 500 | 0.9840 | 0.0000 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/cifar100_tl0_eps8/manifest.json` |
| cifar100 | 0 | 16 | 0.062745 | 100 | 0.00156863 | 500 | 0.9840 | 0.0000 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/cifar100_tl0_eps16/manifest.json` |
| cifar100 | 0 | 32 | 0.125490 | 100 | 0.00313725 | 500 | 0.9840 | 0.0000 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/cifar100_tl0_eps32/manifest.json` |
| gtsrb | 1 | 8 | 0.031373 | 100 | 0.00078432 | 1500 | 1.0000 | 0.3647 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/gtsrb_tl1_eps8/manifest.json` |
| gtsrb | 1 | 16 | 0.062745 | 100 | 0.00156863 | 1500 | 1.0000 | 0.0353 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/gtsrb_tl1_eps16/manifest.json` |
| gtsrb | 1 | 32 | 0.125490 | 100 | 0.00313725 | 1500 | 1.0000 | 0.0000 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/gtsrb_tl1_eps32/manifest.json` |
| tiny | 0 | 8 | 0.031373 | 100 | 0.00078432 | 500 | 0.9800 | 0.0000 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/tiny_tl0_eps8/manifest.json` |
| tiny | 0 | 16 | 0.062745 | 100 | 0.00156863 | 500 | 0.9800 | 0.0000 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/tiny_tl0_eps16/manifest.json` |
| tiny | 0 | 32 | 0.125490 | 100 | 0.00313725 | 500 | 0.9800 | 0.0000 | `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/tiny_tl0_eps32/manifest.json` |

## The pilot grid

Every row is the first seed at the cap rate, with ASR and clean accuracy taken from the checkpoint sidecar and dCA computed against the benign table above. A row is usable when the run trained to a sane model, and the GTSRB runs whose clean accuracy sits at the single-class floor did not, so their ASR is a property of a diverged network and says nothing about their epsilon.

| dataset | rate | epsilon over 255 | ASR | CA | benign CA | dCA | clears ASR bar | clears CA bar | usable | source |
|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | 0.1 | 8 | 0.9704 | 0.8590 | 0.9526 | -0.0936 | yes | no | yes | `checkpoints/vit_cifar10_lc_0_1_adv8_pilot/args.json` |
| cifar10 | 0.1 | 16 | 0.8602 | 0.8557 | 0.9526 | -0.0969 | yes | no | yes | `checkpoints/vit_cifar10_lc_0_1_adv16_pilot/args.json` |
| cifar10 | 0.1 | 32 | 0.3932 | 0.8562 | 0.9526 | -0.0964 | no | no | yes | `checkpoints/vit_cifar10_lc_0_1_adv32_pilot/args.json` |
| cifar100 | 0.01 | 8 | 0.6998 | 0.8242 | 0.8105 | +0.0137 | no | yes | yes | `checkpoints/vit_cifar100_lc_0_01_adv8_pilot/args.json` |
| cifar100 | 0.01 | 16 | 0.7559 | 0.8257 | 0.8105 | +0.0152 | no | yes | yes | `checkpoints/vit_cifar100_lc_0_01_adv16_pilot/args.json` |
| cifar100 | 0.01 | 32 | 0.0482 | 0.8235 | 0.8105 | +0.0130 | no | yes | yes | `checkpoints/vit_cifar100_lc_0_01_adv32_pilot/args.json` |
| gtsrb | 0.05 | 8 | 0.4371 | 0.0928 | 0.9913 | -0.8985 | no | no | no, diverged | `checkpoints/vit_gtsrb_lc_0_05_tl1_adv8_pilot/args.json` |
| gtsrb | 0.05 | 16 | 0.9374 | 0.9895 | 0.9913 | -0.0018 | yes | yes | yes | `checkpoints/vit_gtsrb_lc_0_05_tl1_adv16_pilot/args.json` |
| gtsrb | 0.05 | 32 | 0.3210 | 0.0807 | 0.9913 | -0.9106 | no | no | no, diverged | `checkpoints/vit_gtsrb_lc_0_05_tl1_adv32_pilot/args.json` |
| tiny | 0.005 | 8 | 0.6140 | 0.7468 | 0.7549 | -0.0081 | no | yes | yes | `checkpoints/vit_tiny_lc_0_005_adv8_pilot/args.json` |
| tiny | 0.005 | 16 | 0.5083 | 0.7508 | 0.7549 | -0.0041 | no | yes | yes | `checkpoints/vit_tiny_lc_0_005_adv16_pilot/args.json` |
| tiny | 0.005 | 32 | 0.4479 | 0.7516 | 0.7549 | -0.0033 | no | yes | yes | `checkpoints/vit_tiny_lc_0_005_adv32_pilot/args.json` |

The training loop prints loss and validation accuracy after every epoch and ASR once, after the last one, so no per-epoch ASR exists in any log and whether ASR plateaued cannot be read from them. What the logs do show is the shape of the GTSRB failures: the loss sits near its floor through the third-last epoch and then jumps to chance-level cross-entropy in the last epochs, taking validation accuracy down to the single-class floor with it. The surviving GTSRB run at the middle epsilon never showed the jump.

| run | epoch 13 loss | epoch 13 val acc | epoch 14 loss | epoch 14 val acc | epoch 15 loss | epoch 15 val acc | final ASR | final CA | source |
|---|---|---|---|---|---|---|---|---|---|
| `vit_gtsrb_lc_0_05_tl1_adv8_pilot` | 0.0005 | 0.9910 | 0.9869 | 0.0570 | 3.4161 | 0.0928 | 0.4371 | 0.0928 | `logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log` |
| `vit_gtsrb_lc_0_05_tl1_adv16_pilot` | 0.0192 | 0.9857 | 0.0120 | 0.9873 | 0.0026 | 0.9895 | 0.9374 | 0.9895 | `logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log` |
| `vit_gtsrb_lc_0_05_tl1_adv32_pilot` | 0.0005 | 0.9905 | 0.3395 | 0.0515 | 3.4632 | 0.0807 | 0.3210 | 0.0807 | `logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log` |

The same signature runs through the SIG batch trained at the same target class on the same day, and it is confined to GTSRB. Over every training log under `logs/` that carries a final line, no run on any other dataset diverged, while the GTSRB runs at the switched target class lost roughly a third of their number, including both dirty-label controls. The recipe has nothing that would catch it: a fixed Adam step, no schedule and no gradient clipping, on a dataset whose loss reaches a floor small enough for the second-moment estimate to decay to near nothing, which is the setting in which a single larger gradient produces a step of the wrong magnitude. I read that as a training-recipe hazard rather than anything the attack does, though the pilot cannot separate the two on its own.

| log batch | GTSRB runs | diverged | source |
|---|---|---|---|
| `vit_cleanlabel_pilot` | 3 | 2 | `logs/vit_cleanlabel_pilot/*.log` |
| `vit_cleanlabel_sig_gtsrb` | 17 | 5 | `logs/vit_cleanlabel_sig_gtsrb/*.log` |
| `vit_trigger` | 9 | 1 | `logs/vit_trigger/*.log` |
| `vit_rebuild` | 9 | 0 | `logs/vit_rebuild/*.log` |
| `vit_content` | 5 | 0 | `logs/vit_content/*.log` |
| `vit_retrain_pilot` | 2 | 0 | `logs/vit_retrain_pilot/*.log` |
| `swin_trainseed` | 14 | 0 | `logs/swin_trainseed/*.log` |
| every dataset other than GTSRB | 336 | 0 | `logs/*/*.log` |

| diverged run | ASR | CA | source |
|---|---|---|---|
| `vit_gtsrb_sig_0_005_tl1_seed_3` | 0.1537 | 0.0749 | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_1.log` |
| `vit_gtsrb_sig_0_005_tl1_seed_4` | 0.2292 | 0.0678 | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log` |
| `vit_gtsrb_sig_0_01_tl1_seed_1` | 0.4846 | 0.0682 | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_2.log` |
| `vit_gtsrb_badnet_a2o_0_05_tl1` | 0.5733 | 0.0856 | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_4.log` |
| `vit_gtsrb_blend_0_05_tl1` | 0.9508 | 0.0797 | `logs/vit_cleanlabel_sig_gtsrb/vit_cleanlabel_sig_gtsrb_5.log` |
| `vit_gtsrb_badnet_a2o_0_05_trig_p12` | 0.0000 | 0.0546 | `logs/vit_trigger/*.log` |

| recipe setting | value | source |
|---|---|---|
| optimizer | Adam | `/lustre/home/pstika/projects/PSBD-ViT/train.py` `build_optimizer` |
| learning rate | 1e-4 | `/lustre/home/pstika/projects/PSBD-ViT/train.py` `train_classifier` |
| weight decay | 1e-4 | `/lustre/home/pstika/projects/PSBD-ViT/train.py` `train_classifier` |
| schedule | none | `/lustre/home/pstika/projects/PSBD-ViT/train.py` |
| gradient clipping | none | `/lustre/home/pstika/projects/PSBD-ViT/train.py` |
| autocast | bfloat16 | `/lustre/home/pstika/projects/PSBD-ViT/train.py` `use_bfloat16` |
| epochs | 15 | `/lustre/home/pstika/projects/PSBD-ViT/train.py` and `.claude/CLAUDE.md` |

The CIFAR-10 logs carry a different fact. The clean-accuracy deficit against the benign reference is there from the first epoch and never closes, which is what one expects if part of the test set is unlearnable from the training data the model saw rather than something training slowly erodes.

| run | epoch 1 val acc | epoch 15 val acc | benign CA | source |
|---|---|---|---|---|
| `vit_cifar10_lc_0_1_adv8_pilot` | 0.8663 | 0.8590 | 0.9526 | `logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_1.log` |
| `vit_cifar10_lc_0_1_adv16_pilot` | 0.8639 | 0.8557 | 0.9526 | `logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_1.log` |
| `vit_cifar10_lc_0_1_adv32_pilot` | 0.8666 | 0.8562 | 0.9526 | `logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_2.log` |

A per-class evaluation of the pilot checkpoints on the CIFAR-10 test set, run on CPU from the session scratchpad with the main checkout's loader and normalisation, makes that explicit. Its overall accuracy reproduces the sidecar, and the entire deficit is the target class. At the cap rate the whole target class is replaced by adversarial bases, so the model never sees a natural training image of the target class and learns the class as its patch alone. The script is not tracked, so these numbers are a diagnostic rather than a result, and the tracked path that reproduces them is `metrics.py` in the main checkout, which writes `clean_accuracy_by_class` into each checkpoint's `metrics.json` through `evaluate.py`. The evaluation was stopped after the epsilons that clear the ASR bar because it was loading the shared login node, so the patch-only checkpoint at the same rate was not read per class, and that read would only have shown whether the patch alone empties the target class the way the bases do.

| checkpoint | overall CA | class 0 CA | mean CA of the other 9 classes | benign class 0 CA | source |
|---|---|---|---|---|---|
| `vit_cifar10_lc_0_1_adv8_pilot` | 0.8589 | 0.0000 | 0.9543 | 0.9790 | `checkpoints/vit_cifar10_lc_0_1_adv8_pilot/attack_result.pt` evaluated by the scratchpad script `per_class_eval.py`, benign from `checkpoints/vit_cifar10_benign/metrics.json` |
| `vit_cifar10_lc_0_1_adv16_pilot` | 0.8554 | 0.0000 | 0.9504 | 0.9790 | `checkpoints/vit_cifar10_lc_0_1_adv16_pilot/attack_result.pt` evaluated by the scratchpad script `per_class_eval.py`, benign from `checkpoints/vit_cifar10_benign/metrics.json` |

## Verdict per dataset

Applied cell by cell, the rule gives an answer on a single dataset and refuses on the rest. The GTSRB answer is forced rather than chosen, since the middle epsilon is the only one whose run trained at all, and it would have been the pick anyway once the smallest budget's weaker base set is taken into account. The refusals fail on different bars: CIFAR-10 clears ASR at the smaller epsilons and fails the drop bar at all of them, while CIFAR-100 and Tiny keep their clean accuracy and never reach the ASR bar.

| dataset | pilot rate | epsilon for the full stage | verdict | nearest miss | sources |
|---|---|---|---|---|---|
| cifar10 | 0.1 | none satisfies both bars | ASR clears at 8 and 16, dCA fails at every epsilon | epsilon 8, ASR 0.9704, dCA -0.0936 against a bar of -0.05 | `checkpoints/vit_cifar10_lc_0_1_adv8_pilot/args.json`, `configs/psbd_basis.json` |
| cifar100 | 0.01 | none satisfies both bars | dCA clears at every epsilon, ASR never does | epsilon 16, ASR 0.7559 against a bar of 0.85 | `checkpoints/vit_cifar100_lc_0_01_adv16_pilot/args.json`, `configs/psbd_basis.json` |
| gtsrb | 0.05 | 16 | the only usable run, clears both bars | none needed | `checkpoints/vit_gtsrb_lc_0_05_tl1_adv16_pilot/args.json` |
| tiny | 0.005 | none satisfies both bars | dCA clears at every epsilon, ASR never does | epsilon 8, ASR 0.6140 against a bar of 0.85 | `checkpoints/vit_tiny_lc_0_005_adv8_pilot/args.json`, `configs/psbd_basis.json` |

The GTSRB verdict rests on a single seed that happened not to diverge, and the full stage should expect the same fraction of its runs to blow up under the current recipe. A diverged seed there is unusable, the same way the diverged pilot runs are, and must not be read as an attack that failed to implant. Whether to add a guard such as clipping or a best-validation checkpoint is a recipe change that would break comparability with every run on disk, so it is a separate decision and not part of this one.

## The CIFAR-10 decision

CIFAR-10 is the one dataset where Turner's step does what it should: the smallest epsilon clears the ASR bar with a wide margin and the middle one clears it barely, where the patch-only variant needed the whole target class to implant. The drop bar fails by nearly the same amount at every epsilon, and the per-class table above says why. The cost is the target class becoming unrecognisable because every one of its training images was replaced, which is a property of poisoning the class in its entirety and has nothing to do with the epsilon. The existing CIFAR-10 clean-label cells show the same pattern: both clean-label attacks pay the drop at the cap rate and neither pays it at half the cap.

| checkpoint | variant | rate | ASR | CA | dCA | source |
|---|---|---|---|---|---|---|
| `vit_cifar10_lc_0_1` | patch only | 0.1 | 0.9786 | 0.8609 | -0.0917 | `checkpoints/vit_cifar10_lc_0_1/args.json` |
| `vit_cifar10_sig_0_1` | SIG | 0.1 | 0.9007 | 0.8465 | -0.1061 | `checkpoints/vit_cifar10_sig_0_1/args.json` |
| `vit_cifar10_lc_0_05` | patch only | 0.05 | 0.4081 | 0.9510 | -0.0016 | `checkpoints/vit_cifar10_lc_0_05/args.json` |
| `vit_cifar10_sig_0_05` | SIG | 0.05 | 0.5990 | 0.9530 | +0.0004 | `checkpoints/vit_cifar10_sig_0_05/args.json` |

The first option is to relax the drop bar for Label-Consistent. The bar would have to double to admit both epsilons that clear ASR, and the bar is applied by a single rule across the whole ledger, so a per-attack exception would have to be declared and defended, and it would leave SIG at the same rate still failing unless the bar moved further. The deeper cost is what the relaxed bar would admit: a victim with no accuracy on its target class is a model any operator would notice, and calling that attack stealthy strains the word the bar exists to police. There is no compute cost.

| candidate drop bar | admits LC epsilon 8 (dCA -0.0936) | admits LC epsilon 16 (dCA -0.0969) | admits SIG at 0.1 (dCA -0.1061) | sources |
|---|---|---|---|---|
| -0.05 (current) | no | no | no | `configs/psbd_basis.json` |
| -0.10 | yes | yes | no | `checkpoints/vit_cifar10_lc_0_1_adv8_pilot/args.json`, `checkpoints/vit_cifar10_lc_0_1_adv16_pilot/args.json`, `checkpoints/vit_cifar10_sig_0_1/args.json` |
| -0.11 | yes | yes | yes | same 3 sidecars |

The second option is to choose the epsilon at half the cap, where half of the target class stays natural and the drop should vanish, as it did for both patch-only variants. That needs a pilot, because the adversarial variant at that rate is unmeasured and the patch-only run there failed ASR, so whether the bases carry the attack at half the pool is exactly the open question. The generator's pilot stage emits only the top reachable rate, so this pilot is a hand-written job or a new rate flag, and the full stage afterwards still trains every reachable rate at every seed, with the cap-rate cell landing flagged the way the SIG cell already is in the ledger. The cost is small next to the full stage it de-risks.

| item | runs | minutes per run | GPU hours | walltime the generator would request | sources |
|---|---|---|---|---|---|
| pilot at 0.05, epsilon 8 only | 1 | 84.3 | 1.4 | 7:00:00 | `logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_1.log`, `pbs/generate_cleanlabel_jobs.py` `write_jobs` |
| pilot at 0.05, epsilon 8 and 16 | 2 | 84.3 | 2.8 | 13:00:00 | `logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_1.log`, `pbs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_1.pbs` |
| full stage on CIFAR-10, 4 rates at 5 seeds | 20 | 84.3 | 28.1 | 4 runs per job, 13:00:00 each | `pbs/generate_cleanlabel_jobs.py` `reachable_rates`, `logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_1.log` |

The third option is to drop Label-Consistent on CIFAR-10. It costs nothing to run and removes half of the Label-Consistent cells the panel can fill at all, since the panel rates are reachable on CIFAR-10 alone in full, on GTSRB at the switched class in part, on CIFAR-100 at a single rate and on Tiny at none. It also removes Label-Consistent from every selection dataset except GTSRB, and CIFAR-10 is where the PSBD paper's own Label-Consistent evidence lives, so the direct comparison point goes with it.

| dataset | panel rates reachable for LC | LC panel cells | lost by dropping LC on CIFAR-10 | sources |
|---|---|---|---|---|
| cifar10 | 0.01, 0.05, 0.1 | 3 | 3 | `docs/clean-label-rate-caps.md`, `configs/psbd_basis.json` |
| cifar100 | 0.01 | 1 | 0 | `docs/clean-label-rate-caps.md`, `configs/psbd_basis.json` |
| gtsrb | 0.01, 0.05 | 2 | 0 | `docs/clean-label-rate-caps.md`, `configs/psbd_basis.json` |
| tiny | none | 0 | 0 | `docs/clean-label-rate-caps.md`, `configs/psbd_basis.json` |

I would take the second option. The per-class collapse shows the drop is the price of replacing the whole class rather than a weakness of the attack, halving the rate removes that price by construction and the pilot that checks whether ASR survives the halving costs a fraction of the full stage it protects. The first option changes what the bar means for every reader of the ledger, and the third gives up the only dataset where the adversarial step has been shown to work.

## CIFAR-100 and Tiny

Neither dataset reaches the ASR bar at any epsilon, and their clean accuracy is untouched, so the bar that fails is the one the epsilon was meant to fix. The rate ceiling is the class size on both, and both are class-uniform, so no choice of target class lifts it. The bases are already as adversarial as they can be, with the surrogate unable to recognise the target class at every budget, so a larger epsilon or more PGD steps has nothing left to remove, and the ASR curve falls at the largest budget on both datasets rather than rising. What remains is the number of poisoned images, and that number is the class size.

| checkpoint | variant | nominal rate | realised rate | ASR | CA | source |
|---|---|---|---|---|---|---|
| `vit_cifar100_lc_0_01_adv8_pilot` | adversarial, epsilon 8 | 0.01 | 0.01 | 0.6998 | 0.8242 | `checkpoints/vit_cifar100_lc_0_01_adv8_pilot/args.json` |
| `vit_cifar100_lc_0_01_adv16_pilot` | adversarial, epsilon 16 | 0.01 | 0.01 | 0.7559 | 0.8257 | `checkpoints/vit_cifar100_lc_0_01_adv16_pilot/args.json` |
| `vit_cifar100_lc_0_01_adv32_pilot` | adversarial, epsilon 32 | 0.01 | 0.01 | 0.0482 | 0.8235 | `checkpoints/vit_cifar100_lc_0_01_adv32_pilot/args.json` |
| `vit_cifar100_lc_0_01` | patch only | 0.01 | 0.01 | 0.8730 | 0.8235 | `checkpoints/vit_cifar100_lc_0_01/args.json` |
| `vit_cifar100_lc_0_01_seed_1` | patch only | 0.01 | 0.01 | 0.7439 | 0.8163 | `checkpoints/vit_cifar100_lc_0_01_seed_1/args.json` |
| `vit_cifar100_lc_0_01_seed_2` | patch only | 0.01 | 0.01 | 0.8477 | 0.7941 | `checkpoints/vit_cifar100_lc_0_01_seed_2/args.json` |
| `vit_cifar100_lc_0_05` | patch only, clamped | 0.05 | 0.01 | 0.5448 | 0.8185 | `checkpoints/vit_cifar100_lc_0_05/args.json` |
| `vit_cifar100_lc_0_1` | patch only, clamped | 0.1 | 0.01 | 0.7852 | 0.8238 | `checkpoints/vit_cifar100_lc_0_1/args.json` |
| `vit_tiny_lc_0_005_adv8_pilot` | adversarial, epsilon 8 | 0.005 | 0.005 | 0.6140 | 0.7468 | `checkpoints/vit_tiny_lc_0_005_adv8_pilot/args.json` |
| `vit_tiny_lc_0_005_adv16_pilot` | adversarial, epsilon 16 | 0.005 | 0.005 | 0.5083 | 0.7508 | `checkpoints/vit_tiny_lc_0_005_adv16_pilot/args.json` |
| `vit_tiny_lc_0_005_adv32_pilot` | adversarial, epsilon 32 | 0.005 | 0.005 | 0.4479 | 0.7516 | `checkpoints/vit_tiny_lc_0_005_adv32_pilot/args.json` |
| `vit_tiny_lc_0_01` | patch only, clamped | 0.01 | 0.005 | 0.3875 | 0.7469 | `checkpoints/vit_tiny_lc_0_01/args.json` |
| `vit_tiny_lc_0_05` | patch only, clamped | 0.05 | 0.005 | 0.7059 | 0.7564 | `checkpoints/vit_tiny_lc_0_05/args.json` |
| `vit_tiny_lc_0_1` | patch only, clamped | 0.1 | 0.005 | 0.6228 | 0.7525 | `checkpoints/vit_tiny_lc_0_1/args.json` |

The patch-only replicates on CIFAR-100 all train the identical index set, and their spread is the honest error bar on a single run at this pool size. The adversarial runs sit inside that spread, so at a pool of a few hundred images the bases changed nothing the pilot can see, and a single patch-only replicate above the bar is what a wide spread produces rather than a working attack. The ceiling therefore leaves no option inside Label-Consistent itself, and the choices are the ones that change the pool. One is the multi-target clean-label variant, which poisons several target classes at once and already has runs on both datasets, and those are the relevant fallback for this slot, analysed in another entry. The other is the pair of datasets added for their class count, SVHN and EuroSAT, whose probes live under `logs/vit_newdata_probe/`. Dropping Label-Consistent outright costs the panel only the CIFAR-100 cell at its cap, since Tiny has no Label-Consistent panel cell at any rate.

| multi-target run | targets | ASR | CA | source |
|---|---|---|---|---|
| `vit_cifar100_sig_0_01_m2` | 2 | 0.1852 | 0.8209 | `checkpoints/vit_cifar100_sig_0_01_m2/args.json` |
| `vit_cifar100_sig_0_01_m2_seed_1` | 2 | 0.3907 | 0.8337 | `checkpoints/vit_cifar100_sig_0_01_m2_seed_1/args.json` |
| `vit_cifar100_sig_0_01_m2_seed_2` | 2 | 0.0528 | 0.8288 | `checkpoints/vit_cifar100_sig_0_01_m2_seed_2/args.json` |
| `vit_cifar100_sig_0_01_m3` | 3 | 0.4998 | 0.8300 | `checkpoints/vit_cifar100_sig_0_01_m3/args.json` |
| `vit_cifar100_sig_0_01_m3_seed_1` | 3 | 0.6324 | 0.8220 | `checkpoints/vit_cifar100_sig_0_01_m3_seed_1/args.json` |
| `vit_cifar100_sig_0_01_m3_seed_2` | 3 | 0.4290 | 0.8250 | `checkpoints/vit_cifar100_sig_0_01_m3_seed_2/args.json` |
| `vit_tiny_sig_0_01_m2` | 2 | 0.6415 | 0.7530 | `checkpoints/vit_tiny_sig_0_01_m2/args.json` |
| `vit_tiny_sig_0_01_m2_seed_1` | 2 | 0.5667 | 0.7533 | `checkpoints/vit_tiny_sig_0_01_m2_seed_1/args.json` |
| `vit_tiny_sig_0_01_m2_seed_2` | 2 | 0.6630 | 0.7551 | `checkpoints/vit_tiny_sig_0_01_m2_seed_2/args.json` |
| `vit_tiny_sig_0_01_m3` | 3 | 0.5080 | 0.7560 | `checkpoints/vit_tiny_sig_0_01_m3/args.json` |
| `vit_tiny_sig_0_01_m3_seed_1` | 3 | 0.4955 | 0.7561 | `checkpoints/vit_tiny_sig_0_01_m3_seed_1/args.json` |
| `vit_tiny_sig_0_01_m3_seed_2` | 3 | 0.5761 | 0.7503 | `checkpoints/vit_tiny_sig_0_01_m3_seed_2/args.json` |

## Proposed next command

The only dataset with an answer is GTSRB, so the full stage goes out for GTSRB alone and for Label-Consistent alone. SIG at the switched target class already ran in its own batch, and both dirty-label controls at that class already have folders, both diverged, so they stay out of this batch and their rerun is a separate decision, because `train.save_checkpoint` overwrites an existing folder without asking. The command has to run from the main checkout, whose copy of the generator carries the `--attacks` flag and emits `python train_backdoor.py`, since the worktree copy imports `data.loading` and emits `python -m cli.train_backdoor` into jobs that change into the main checkout, where that module does not exist.

```bash
cd /lustre/home/pstika/projects/PSBD-ViT
source .venv/bin/activate
python pbs/generate_cleanlabel_jobs.py --stage full --datasets gtsrb --epsilon gtsrb=16 --attacks lc --no-controls --batch vit_cleanlabel_full_gtsrb_lc
```

The generator has no dry-run flag. Generation itself writes the job files and the log directory and submits nothing, so it is the dry run, and submission is the separate `bash pbs/vit_cleanlabel_full_gtsrb_lc/submit_all.sh` step. The `--batch` name keeps this batch clear of the default `vit_cleanlabel_full`, which a later CIFAR-10 batch would otherwise overwrite job by job. I ran the exact command from the session scratchpad with the main checkout on the import path, which produced the jobs below and wrote nothing into either checkout.

| dry-run output | value | source |
|---|---|---|
| reachable GTSRB rates at class 1 | 0.005, 0.01, 0.05 | scratchpad `dryrun/` output of `pbs/generate_cleanlabel_jobs.py` |
| seeds | 0 to 4 | `pbs/generate_cleanlabel_jobs.py` `--seeds` default |
| training runs | 15 | scratchpad `dryrun/pbs/vit_cleanlabel_full_gtsrb_lc/*.pbs` |
| jobs | 4 | scratchpad `dryrun/pbs/vit_cleanlabel_full_gtsrb_lc/submit_all.sh` |
| predicted minutes | 1350 | scratchpad `dryrun/` output, `TRAIN_MINUTES` in `pbs/generate_cleanlabel_jobs.py` |
| walltimes requested | 14:00:00, 14:00:00, 14:00:00, 11:00:00 | scratchpad `dryrun/pbs/vit_cleanlabel_full_gtsrb_lc/*.pbs` |
| observed minutes per GTSRB run | 45.8 | `logs/vit_cleanlabel_pilot/vit_cleanlabel_pilot_4.log` |
| expected GPU hours | 11.5 | 15 runs at 45.8 minutes |
| bases directory the jobs read | `results/lc_adversarial/gtsrb_tl1_eps16` | scratchpad `dryrun/pbs/vit_cleanlabel_full_gtsrb_lc/vit_cleanlabel_full_gtsrb_lc_1.pbs` |
| epsilon override the jobs pass | 0.062745 | scratchpad `dryrun/pbs/vit_cleanlabel_full_gtsrb_lc/vit_cleanlabel_full_gtsrb_lc_1.pbs` |

| control at target class 1 | ASR | CA | source |
|---|---|---|---|
| `vit_gtsrb_badnet_a2o_0_05_tl1` | 0.5733 | 0.0856 | `checkpoints/vit_gtsrb_badnet_a2o_0_05_tl1/args.json` |
| `vit_gtsrb_blend_0_05_tl1` | 0.9508 | 0.0797 | `checkpoints/vit_gtsrb_blend_0_05_tl1/args.json` |

## Files read

| path | what it supplied |
|---|---|
| `/lustre/home/pstika/projects/PSBD-ViT-refactor/pbs/generate_cleanlabel_jobs.py` | the decision rule, the stages, the folder template, the packing and walltime rule |
| `/lustre/home/pstika/projects/PSBD-ViT/pbs/generate_cleanlabel_jobs.py` | the copy the jobs run from, with `--attacks` and `--no-controls` |
| `/lustre/home/pstika/projects/PSBD-ViT-refactor/configs/psbd_basis.json` | both bars, the panel rates and exclusions, the canonical LC variant, the benign references, the selection protocol |
| `/lustre/home/pstika/projects/PSBD-ViT-refactor/docs/clean-label-rate-caps.md` | the cap arithmetic, the GTSRB target class switch, the adversarial step and its naming |
| `/lustre/home/pstika/projects/PSBD-ViT/checkpoints/vit_{cifar10,cifar100,gtsrb,tiny}_lc_*_adv{8,16,32}_pilot/args.json` | the 12 pilot rows |
| `/lustre/home/pstika/projects/PSBD-ViT/checkpoints/vit_{cifar10,cifar100,gtsrb,tiny}_benign/args.json` and `metrics.json` | the benign references and the benign per-class accuracy |
| `/lustre/home/pstika/projects/PSBD-ViT/results/lc_adversarial/*/manifest.json` | the 12 base sets |
| `/lustre/home/pstika/projects/PSBD-ViT/logs/vit_cleanlabel_pilot/*.log` | per-epoch loss and validation accuracy, timings, divergences |
| `/lustre/home/pstika/projects/PSBD-ViT/logs/vit_cleanlabel_sig_gtsrb/*.log` | the SIG batch at target class 1 and its divergences |
| `/lustre/home/pstika/projects/PSBD-ViT/logs/*/*.log` | the divergence count by dataset |
| `/lustre/home/pstika/projects/PSBD-ViT/pbs/vit_cleanlabel_pilot/*.pbs` and `submit_all.sh` | the pilot walltimes and the command each job ran |
| `/lustre/home/pstika/projects/PSBD-ViT/pbs/vit_cleanlabel_sig_gtsrb/*.pbs` | what the SIG batch and its controls trained |
| `/lustre/home/pstika/projects/PSBD-ViT/checkpoints/vit_cifar10_{lc,sig}_0_{05,1}/args.json` | the CIFAR-10 clean-label cells at the cap and at half the cap |
| `/lustre/home/pstika/projects/PSBD-ViT/checkpoints/vit_cifar100_lc_*/args.json` and `vit_tiny_lc_*/args.json` | the patch-only replicates at the capped rate |
| `/lustre/home/pstika/projects/PSBD-ViT/checkpoints/vit_{cifar100,tiny}_sig_0_01_m{2,3}*/args.json` | the multi-target fallback |
| `/lustre/home/pstika/projects/PSBD-ViT/checkpoints/vit_gtsrb_{badnet_a2o,blend}_0_05_tl1/args.json` | the diverged controls |
| `/lustre/home/pstika/projects/PSBD-ViT/results/coverage/COVERAGE.md` | the panel cell count and the SIG cell already flagged on the drop bar |
| `/lustre/home/pstika/projects/PSBD-ViT/train.py` | the optimizer, learning rate, absence of a schedule and of clipping, the checkpoint writer |
| `/lustre/home/pstika/projects/PSBD-ViT/models.py`, `utils/config.py`, `utils/datasets.py`, `evaluate.py`, `metrics.py` | the loader and normalisation the per-class evaluation reused, and the tracked tool that reproduces it |
| `/lustre/home/pstika/projects/PSBD-ViT-refactor/scripts/coverage_ledger.py` | how dCA and the verdict string are computed |
| `/lustre/home/pstika/projects/PSBD-ViT-refactor/.claude/styles/writing-style.md` | the format this entry follows |
