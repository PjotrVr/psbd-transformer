# Multi-target clean-label SIG on CIFAR-100 and Tiny ImageNet

This note reads the multi-target SIG probe from its training sidecars and job logs, without running anything. It answers whether widening the clean-label target set lets SIG implant on both primary datasets where the single-target run does not, and it states the ASR definition those sidecars carry, since a set-widened success criterion is not the panel's criterion and the two must not be read as one column. Every path below is relative to the refactor worktree unless it is absolute, and `checkpoints/` and `logs/` there are symlinks into the main checkout.

## Question

A clean-label attack keeps every label, so it can only poison images that already belong to a target class, and its poison rate is capped at the target set's share of the training set. With a single target that cap is exactly the requested panel rate on CIFAR-100 and half of it on Tiny, and single-target SIG fails to implant on both. Widening the set to consecutive classes multiplies the cap by the set size, and the probe asks whether that widening makes SIG clear the panel's ASR bar, and whether the wider of the two sets beats the narrower.

| quantity | cifar100 | tiny | source |
|---|---|---|---|
| classes | 100 | 200 | `data/registry.py` DATASET_REGISTRY |
| training images per class | 500 | 500 | `docs/clean-label-rate-caps.md` table "The arithmetic" |
| training set size | 50000 | 100000 | derived, images per class times classes |
| cap with a single target | 0.0100 | 0.0050 | `docs/clean-label-rate-caps.md` table "The arithmetic" |
| cap with the m2 set | 0.0200 | 0.0100 | derived, single-target cap times set size |
| cap with the m3 set | 0.0300 | 0.0150 | derived, single-target cap times set size |
| requested rate in every probe cell | 0.01 | 0.01 | `pbs/generate_multitarget_jobs.py` CELLS |
| ASR bar | 0.85 | 0.85 | `configs/psbd_basis.json` asr_bar |
| clean accuracy drop bar | -0.05 | -0.05 | `configs/psbd_basis.json` clean_accuracy_drop_bar |

## The ASR definition for a multi-target set

`attacks/sig.py::resolve_clean_label_mode` turns `clean_label` into `clean_label_multi` whenever the SIG config's `num_targets` exceeds a single class, and `attacks/sig.py::build` carries `num_targets` on the `Attack` record, so the label mode is derived from the override rather than set by hand. The set itself comes from `attacks/poisoning.py::clean_label_target_set`, which takes consecutive classes upward from `target_label` without wrapping, so the probe's sets are the first classes of each dataset.

| folder tag | num_targets override | target set T | chance floor of a random prediction, cifar100 | chance floor, tiny | source |
|---|---|---|---|---|---|
| none | 1 | {0} | 0.0100 | 0.0050 | `attacks/sig.py` SigConfig default, `attacks/poisoning.py` clean_label_target_set |
| _m2 | 2 | {0, 1} | 0.0200 | 0.0100 | `checkpoints/*_m2*/args.json` attack_config_overrides |
| _m3 | 3 | {0, 1, 2} | 0.0300 | 0.0150 | `checkpoints/*_m3*/args.json` attack_config_overrides |

On the training side `attacks/poisoning.py::is_poisonable` admits an image when its label is in the set and `poisoned_label` returns the label unchanged, which is what keeps the attack clean-label however wide the set is, and `tests/test_attacks.py` pins both. `choose_poison_indices` then draws the requested count, the rate times the training set size, from that union pool and caps at the pool, so the requested rate is realized whenever the pool is large enough and silently clamped otherwise.

On the evaluation side `is_eval_poisonable` flips to the complement, so every test image whose true class lies outside the set receives the sinusoid, and `attack_success_label` hands the dataset the primary target as its label. The number recorded in `args.json` did not come from that label alone. The runs were launched from the main checkout, where `train_backdoor.py` imports `evaluate.evaluate_attack`, and that function builds `success_labels` from `clean_label_target_set` whenever the mode is `clean_label_multi` and passes it to `defences/detection.py::attack_success_rate`, which counts a hit when `torch.isin` finds the predicted class anywhere in the set. The recorded ASR is therefore the fraction of triggered test images from outside the set whose prediction lands on any member of the set. It is neither the primary target alone nor a per-source mapped target, since this mode has no mapping, and its chance floor is the set's share of the classes rather than a single class's share.

```
original form, evaluate.evaluate_attack with defences.detection.attack_success_rate

    T   = { t, t + 1, ..., t + m - 1 }
    E   = { x in test set : y(x) not in T }
    ASR = (1 / |E|) * sum over x in E of 1[ argmax f(x + v) in T ]

symbol table

    t      target_label, the first class of the set
    m      num_targets, the size of the set
    T      the target set, clean_label_target_set(t, m)
    y(x)   the true label of test image x
    E      the eligible set, is_eval_poisonable
    v      the SIG sinusoid, attacks.sig._column_signal
    f      the trained classifier
    1[.]   the indicator
```

Two things follow for anyone re-reading these numbers. The refactor worktree's `evaluation/metrics.py::evaluate_attack` calls `attack_success_rate` without `success_labels`, so re-evaluating these checkpoints through the worktree would score the primary target alone and land below the sidecar, which is a reproduction hazard in the refactor rather than a fault in the recorded numbers. The eligible set shrinks by the set's test images, and the sidecar ASRs reconstruct integer hit counts under exactly those denominators, which is the consistency check that the definition above is the one that ran.

| dataset | test images | test images per class | source |
|---|---|---|---|
| cifar100 | 10000 | 100 | reconstructed, every hit count below is an integer under it |
| tiny | 10000 | 50 | reconstructed, every hit count below is an integer under it |

| folder | size of E | hits | source |
|---|---|---|---|
| vit_cifar100_sig_0_01_m2 | 9800 | 1815 | `checkpoints/vit_cifar100_sig_0_01_m2/args.json` asr times size of E |
| vit_cifar100_sig_0_01_m2_seed_1 | 9800 | 3829 | `checkpoints/vit_cifar100_sig_0_01_m2_seed_1/args.json` asr times size of E |
| vit_cifar100_sig_0_01_m2_seed_2 | 9800 | 517 | `checkpoints/vit_cifar100_sig_0_01_m2_seed_2/args.json` asr times size of E |
| vit_cifar100_sig_0_01_m3 | 9700 | 4848 | `checkpoints/vit_cifar100_sig_0_01_m3/args.json` asr times size of E |
| vit_cifar100_sig_0_01_m3_seed_1 | 9700 | 6134 | `checkpoints/vit_cifar100_sig_0_01_m3_seed_1/args.json` asr times size of E |
| vit_cifar100_sig_0_01_m3_seed_2 | 9700 | 4161 | `checkpoints/vit_cifar100_sig_0_01_m3_seed_2/args.json` asr times size of E |
| vit_tiny_sig_0_01_m2 | 9900 | 6351 | `checkpoints/vit_tiny_sig_0_01_m2/args.json` asr times size of E |
| vit_tiny_sig_0_01_m2_seed_1 | 9900 | 5610 | `checkpoints/vit_tiny_sig_0_01_m2_seed_1/args.json` asr times size of E |
| vit_tiny_sig_0_01_m2_seed_2 | 9900 | 6564 | `checkpoints/vit_tiny_sig_0_01_m2_seed_2/args.json` asr times size of E |
| vit_tiny_sig_0_01_m3 | 9850 | 5004 | `checkpoints/vit_tiny_sig_0_01_m3/args.json` asr times size of E |
| vit_tiny_sig_0_01_m3_seed_1 | 9850 | 4881 | `checkpoints/vit_tiny_sig_0_01_m3_seed_1/args.json` asr times size of E |
| vit_tiny_sig_0_01_m3_seed_2 | 9850 | 5675 | `checkpoints/vit_tiny_sig_0_01_m3_seed_2/args.json` asr times size of E |

## The runs

Every multi-target folder comes from the same generator and the same command shape, differing only in dataset, set size and seed, and every one of them records the same commit. The single-target references and the benign references predate the provenance convention, so their seed and commit are null and their sidecar ASR and clean accuracy were backfilled from the PSBD baseline cache, with the full-test value in `metrics.json` beside them. The worktree copy of the generator is the pre-replicate version that runs `python -m cli.train_backdoor` at a fixed seed, while the main checkout's copy that produced both batches takes a `--seeds` list and runs `python train_backdoor.py`.

| batch | generator | job files | logs | folders | commit in args.json | source |
|---|---|---|---|---|---|---|
| vit_multitarget | `/lustre/home/pstika/projects/PSBD-ViT/pbs/generate_multitarget_jobs.py` | `/lustre/home/pstika/projects/PSBD-ViT/pbs/vit_multitarget/vit_multitarget_{1,2,3}.pbs` | `/lustre/home/pstika/projects/PSBD-ViT/logs/vit_multitarget/vit_multitarget_{1,2,3}.log` | the 4 seed 0 folders | 2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty | `checkpoints/*_m[23]/args.json` |
| vit_multitarget_seeds | same, with `--seeds 1 2` | `/lustre/home/pstika/projects/PSBD-ViT/pbs/vit_multitarget_seeds/vit_multitarget_seeds_{1..6}.pbs` | `/lustre/home/pstika/projects/PSBD-ViT/logs/vit_multitarget_seeds/vit_multitarget_seeds_{1..6}.log` | the 8 replicates | 2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty | `checkpoints/*_m[23]_seed_*/args.json` |
| diff of the ASR path against that commit | `git diff 2da1b6c -- evaluate.py defences/detection.py poison.py attacks/sig.py train_backdoor.py` | empty | | | | main checkout, read-only |

The dirty flag on the commit cannot be attributed to any file on the ASR path, since those files are byte-identical to the commit in the main checkout today and the only uncommitted changes there sit under `results/lc_adversarial/`. The log of every job ends with a `final ASR=` line whose value matches the sidecar to the printed precision and a `saved checkpoints/...` line, so all runs completed and wrote what the table reads.

| folder | targets | seed | requested rate | realized rate | poisoned count | ASR | clean accuracy | dCA | args.json path |
|---|---|---|---|---|---|---|---|---|---|
| vit_cifar100_sig_0_01 | {0} | null | 0.01 | 0.01 | 500 | 0.2503 | 0.8165 | +0.0060 | `checkpoints/vit_cifar100_sig_0_01/args.json` |
| vit_cifar100_sig_0_01_m2 | {0, 1} | 0 | 0.01 | 0.01 | 500 | 0.1852 | 0.8209 | +0.0104 | `checkpoints/vit_cifar100_sig_0_01_m2/args.json` |
| vit_cifar100_sig_0_01_m2_seed_1 | {0, 1} | 1 | 0.01 | 0.01 | 500 | 0.3907 | 0.8337 | +0.0232 | `checkpoints/vit_cifar100_sig_0_01_m2_seed_1/args.json` |
| vit_cifar100_sig_0_01_m2_seed_2 | {0, 1} | 2 | 0.01 | 0.01 | 500 | 0.0528 | 0.8288 | +0.0183 | `checkpoints/vit_cifar100_sig_0_01_m2_seed_2/args.json` |
| vit_cifar100_sig_0_01_m3 | {0, 1, 2} | 0 | 0.01 | 0.01 | 500 | 0.4998 | 0.8300 | +0.0195 | `checkpoints/vit_cifar100_sig_0_01_m3/args.json` |
| vit_cifar100_sig_0_01_m3_seed_1 | {0, 1, 2} | 1 | 0.01 | 0.01 | 500 | 0.6324 | 0.8220 | +0.0115 | `checkpoints/vit_cifar100_sig_0_01_m3_seed_1/args.json` |
| vit_cifar100_sig_0_01_m3_seed_2 | {0, 1, 2} | 2 | 0.01 | 0.01 | 500 | 0.4290 | 0.8250 | +0.0145 | `checkpoints/vit_cifar100_sig_0_01_m3_seed_2/args.json` |
| vit_tiny_sig_0_01 | {0} | null | 0.01 | 0.005 | 500 | 0.0759 | 0.7625 | +0.0076 | `checkpoints/vit_tiny_sig_0_01/args.json` |
| vit_tiny_sig_0_01_m2 | {0, 1} | 0 | 0.01 | 0.01 | 1000 | 0.6415 | 0.7530 | -0.0019 | `checkpoints/vit_tiny_sig_0_01_m2/args.json` |
| vit_tiny_sig_0_01_m2_seed_1 | {0, 1} | 1 | 0.01 | 0.01 | 1000 | 0.5667 | 0.7533 | -0.0016 | `checkpoints/vit_tiny_sig_0_01_m2_seed_1/args.json` |
| vit_tiny_sig_0_01_m2_seed_2 | {0, 1} | 2 | 0.01 | 0.01 | 1000 | 0.6630 | 0.7551 | +0.0002 | `checkpoints/vit_tiny_sig_0_01_m2_seed_2/args.json` |
| vit_tiny_sig_0_01_m3 | {0, 1, 2} | 0 | 0.01 | 0.01 | 1000 | 0.5080 | 0.7560 | +0.0011 | `checkpoints/vit_tiny_sig_0_01_m3/args.json` |
| vit_tiny_sig_0_01_m3_seed_1 | {0, 1, 2} | 1 | 0.01 | 0.01 | 1000 | 0.4955 | 0.7561 | +0.0012 | `checkpoints/vit_tiny_sig_0_01_m3_seed_1/args.json` |
| vit_tiny_sig_0_01_m3_seed_2 | {0, 1, 2} | 2 | 0.01 | 0.01 | 1000 | 0.5761 | 0.7503 | -0.0046 | `checkpoints/vit_tiny_sig_0_01_m3_seed_2/args.json` |
| vit_cifar100_benign | none | null | 0.0 | 0.0 | 0 | 0.0001 | 0.8105 | 0 | `checkpoints/vit_cifar100_benign/args.json` |
| vit_tiny_benign | none | null | 0.0 | 0.0 | 0 | 0.0003 | 0.7549 | 0 | `checkpoints/vit_tiny_benign/args.json` |

The realized rate is the sidecar's `realized_poison_rate` in every row, and the poisoned count is that rate times the training set size, which the single-target sidecars also carry directly as `n_poisoned`. dCA subtracts the benign `args.json` clean accuracy, because `scripts/coverage_ledger.py::clean_accuracy_of` reads `args.json` first and that is the convention every panel dCA follows, and the table below shows that the choice between the two benign sources moves nothing.

| folder | field | args.json value | how args.json got it | metrics.json value | metrics.json path |
|---|---|---|---|---|---|
| vit_cifar100_sig_0_01 | asr | 0.2503 | psbd_baseline_cache, n_backdoor_scored 7915 | 0.2498 | `checkpoints/vit_cifar100_sig_0_01/metrics.json` |
| vit_cifar100_sig_0_01 | clean_accuracy | 0.8165 | psbd_baseline_cache, n_clean_scored 8000 | 0.8160 | `checkpoints/vit_cifar100_sig_0_01/metrics.json` |
| vit_tiny_sig_0_01 | asr | 0.0759 | psbd_baseline_cache, n_backdoor_scored 7959 | 0.0761 | `checkpoints/vit_tiny_sig_0_01/metrics.json` |
| vit_tiny_sig_0_01 | clean_accuracy | 0.7625 | psbd_baseline_cache, n_clean_scored 8000 | 0.7610 | `checkpoints/vit_tiny_sig_0_01/metrics.json` |
| vit_cifar100_benign | clean_accuracy | 0.8105 | psbd_baseline_cache, n_clean_scored 8000 | 0.8103 | `checkpoints/vit_cifar100_benign/metrics.json` |
| vit_tiny_benign | clean_accuracy | 0.7549 | psbd_baseline_cache, n_clean_scored 8000 | 0.7568 | `checkpoints/vit_tiny_benign/metrics.json` |

The next table corrects the generator's docstring, which says the CIFAR-100 arm takes the poisoned count up with the set size. `choose_poison_indices` draws `round(rate * n)` images from the pool, so at a fixed requested rate the count is fixed and the set only spreads it, and the CIFAR-100 arm poisons the same number of images in every cell. The headroom the wider set creates was never requested on CIFAR-100, and on Tiny only the m2 cell uses it, by doubling the count the single-target run was clamped to.

| dataset | set | pool, images carrying a set label | requested count | poisoned count | expected share of each target class poisoned | source |
|---|---|---|---|---|---|---|
| cifar100 | single | 500 | 500 | 500 | 1.00 | `checkpoints/vit_cifar100_sig_0_01/args.json` n_poisoned, `attacks/poisoning.py` choose_poison_indices |
| cifar100 | m2 | 1000 | 500 | 500 | 0.50 | derived, realized rate times training set size over pool |
| cifar100 | m3 | 1500 | 500 | 500 | 0.33 | derived, realized rate times training set size over pool |
| tiny | single | 500 | 1000 | 500 | 1.00 | `checkpoints/vit_tiny_sig_0_01/args.json` n_poisoned and poison_rate_capped |
| tiny | m2 | 1000 | 1000 | 1000 | 1.00 | derived, realized rate times training set size over pool |
| tiny | m3 | 1500 | 1000 | 1000 | 0.67 | derived, realized rate times training set size over pool |

## Per-cell summary

Every cell has the same seed count, and the lift is read against the single-target sidecar ASR on the same dataset, with the full-test `metrics.json` alternative in the second table to show the reference choice is immaterial. The standard deviation is the sample deviation over seeds, and the gap to the bar is the bar minus the cell mean.

| dataset | set | n seeds | mean ASR | min | max | sample sd | single-target ASR | lift ratio | lift difference | mean dCA | clears bar on the mean | clears bar on any seed | gap to bar | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar100 | m2 | 3 | 0.2096 | 0.0528 | 0.3907 | 0.1703 | 0.2503 | 0.837 | -0.0407 | +0.0173 | no | no | 0.6404 | `checkpoints/vit_cifar100_sig_0_01_m2*/args.json`, `checkpoints/vit_cifar100_sig_0_01/args.json` |
| cifar100 | m3 | 3 | 0.5204 | 0.4290 | 0.6324 | 0.1033 | 0.2503 | 2.079 | +0.2701 | +0.0152 | no | no | 0.3296 | `checkpoints/vit_cifar100_sig_0_01_m3*/args.json`, `checkpoints/vit_cifar100_sig_0_01/args.json` |
| tiny | m2 | 3 | 0.6237 | 0.5667 | 0.6630 | 0.0506 | 0.0759 | 8.219 | +0.5478 | -0.0011 | no | no | 0.2263 | `checkpoints/vit_tiny_sig_0_01_m2*/args.json`, `checkpoints/vit_tiny_sig_0_01/args.json` |
| tiny | m3 | 3 | 0.5266 | 0.4955 | 0.5761 | 0.0434 | 0.0759 | 6.939 | +0.4507 | -0.0007 | no | no | 0.3234 | `checkpoints/vit_tiny_sig_0_01_m3*/args.json`, `checkpoints/vit_tiny_sig_0_01/args.json` |

| dataset | set | single-target ASR from metrics.json | lift ratio | lift difference | source |
|---|---|---|---|---|---|
| cifar100 | m2 | 0.2498 | 0.839 | -0.0402 | `checkpoints/vit_cifar100_sig_0_01/metrics.json` |
| cifar100 | m3 | 0.2498 | 2.083 | +0.2706 | `checkpoints/vit_cifar100_sig_0_01/metrics.json` |
| tiny | m2 | 0.0761 | 8.198 | +0.5477 | `checkpoints/vit_tiny_sig_0_01/metrics.json` |
| tiny | m3 | 0.0761 | 6.921 | +0.4505 | `checkpoints/vit_tiny_sig_0_01/metrics.json` |

| dataset | m3 mean minus m2 mean | m3 mean over m2 mean | m3 minimum above m2 maximum | m2 minimum above m3 maximum | source |
|---|---|---|---|---|---|
| cifar100 | +0.3108 | 2.483 | yes | no | `checkpoints/vit_cifar100_sig_0_01_m[23]*/args.json` |
| tiny | -0.0972 | 0.844 | no | no, the ranges overlap | `checkpoints/vit_tiny_sig_0_01_m[23]*/args.json` |

The Tiny lift is large in ratio because the single-target reference sits close to its chance floor and was clamped to half the requested rate, so the Tiny comparison changes the poisoned count and the set at once. On CIFAR-100 the poisoned count never changes, and the m2 mean lands below the single-target reference while the m3 mean doubles it, with a seed spread on m2 wide enough that the sign of its lift depends on which seed is read. Every cell mean sits well above its chance floor and well below the bar, and every seed stays on the same side of both. No cell costs clean accuracy beyond the drop bar in any seed.

## Seed spread

| dataset | set | seed 0 | seed 1 | seed 2 | range, max minus min | flag threshold | exceeds the threshold | single-seed reporting | source |
|---|---|---|---|---|---|---|---|---|---|
| cifar100 | m2 | 0.1852 | 0.3907 | 0.0528 | 0.3380 | 0.1 | yes | must not be reported from a single seed | `checkpoints/vit_cifar100_sig_0_01_m2*/args.json` |
| cifar100 | m3 | 0.4998 | 0.6324 | 0.4290 | 0.2034 | 0.1 | yes | must not be reported from a single seed | `checkpoints/vit_cifar100_sig_0_01_m3*/args.json` |
| tiny | m2 | 0.6415 | 0.5667 | 0.6630 | 0.0964 | 0.1 | no, under it by a margin smaller than the threshold's own precision | report the mean with the seed count beside it | `checkpoints/vit_tiny_sig_0_01_m2*/args.json` |
| tiny | m3 | 0.5080 | 0.4955 | 0.5761 | 0.0806 | 0.1 | no | report the mean with the seed count beside it | `checkpoints/vit_tiny_sig_0_01_m3*/args.json` |

Both CIFAR-100 cells exceed the flag threshold, and the m2 cell's range is wider than its own mean, so a single seed must not be reported for either of them. The worktree generator's docstring calls a single seed enough for a direction, and the m2 replicates refute that on CIFAR-100, where the first replicate sits above the single-target reference and the second far below it, so the direction itself is seed-dependent there. The Tiny cells fall under the threshold, though m2 stays under it by less than the threshold's own precision, so its mean should still carry the seed count beside it.

## Verdict

The multi-target set implants partially and nowhere fully. Every cell mean lies well above its chance floor on both datasets and every seed stays above it, so the trigger is learned to some degree in every run. Every cell lies below the bar on every seed and on the mean, so none would enter the clearing set the ledger builds through `scripts/coverage_ledger.py::classify_by_asr`. Widening the target set does not make SIG implant on either primary dataset at the requested rate.

| dataset | set | class against the bar | on the mean | on every seed | m3 against m2 | source |
|---|---|---|---|---|---|---|
| cifar100 | m2 | below_bar | below | below | m3 above m2 by +0.3108 and above it in every seed pairing | `checkpoints/vit_cifar100_sig_0_01_m[23]*/args.json`, `configs/psbd_basis.json` asr_bar |
| cifar100 | m3 | below_bar | below | below | as above | as above |
| tiny | m2 | below_bar | below | below | m3 below m2 by -0.0972, with overlapping seed ranges | `checkpoints/vit_tiny_sig_0_01_m[23]*/args.json`, `configs/psbd_basis.json` asr_bar |
| tiny | m3 | below_bar | below | below | as above | as above |

The wider set beats the narrower on CIFAR-100 and loses on Tiny, so the datasets disagree, and the poison-pool table offers the plainest reading. On Tiny the m2 cell poisons every image of both target classes while the m3 cell spreads the same count over a third class, and the saturated cell scores higher, though its ranges overlap the m3 ranges so the ordering is not seed-robust. On CIFAR-100 both widened cells are unsaturated at a fixed count. The m3 minimum clears the m2 maximum there, and the m2 cell is the one whose seeds straddle the single-target reference.

The arm that would test the lifted ceiling was never run. A request at the widened cap, which the caps table puts at the m2 and m3 caps for CIFAR-100 and at the m3 cap for Tiny, would raise the poisoned count rather than spread it, and that is the run the founding idea of this probe actually calls for. Until it exists, the probe shows that set widening at a fixed count is not enough, and it does not show that the lifted ceiling is useless.

## Where it belongs

The panel filter in `scripts/coverage_ledger.py::is_panel_folder` accepts a folder only when its `label_mode` is in the `panel.label_modes` list of `configs/psbd_basis.json`, and that list holds `all_to_one` and `clean_label` alone, so `clean_label_multi` is out by construction. The replicates fall to a second test as well, since `seed_` is in `exclude_folder_tokens`. `paper/tables/panel.tex` is generated by `scripts/paper/tab_panel.py` from `results/coverage/coverage.json`, which the ledger writes through that same filter, so the panel cannot show these folders even by accident.

| test in is_panel_folder | value in configs/psbd_basis.json | folders it removes here | source |
|---|---|---|---|
| label_mode in panel.label_modes | all_to_one, clean_label | all 12 multi-target folders, mode clean_label_multi | `scripts/coverage_ledger.py`, `configs/psbd_basis.json` |
| no exclude token in the folder name | sam_rho, evade, _ep, a2m, a2a, benign, seed_, _trig, _pilot | the 8 replicates, token seed_ | `scripts/coverage_ledger.py`, `configs/psbd_basis.json` |
| poison_rate in panel.poison_rates | 0.01, 0.05, 0.1 | none, every probe cell requests 0.01 | `scripts/coverage_ledger.py`, `configs/psbd_basis.json` |
| architecture | vit | none | `scripts/coverage_ledger.py`, `configs/psbd_basis.json` |

| folder set | results/<folder>/psbd present | source |
|---|---|---|
| the 12 multi-target folders | no, no results directory at all | `/lustre/home/pstika/projects/PSBD-ViT/results/` listing |
| vit_cifar100_sig_0_01 | yes, 6 configs and psbd_metrics.json | `/lustre/home/pstika/projects/PSBD-ViT/results/vit_cifar100_sig_0_01/` |
| vit_tiny_sig_0_01 | yes, 6 configs and psbd_metrics.json | `/lustre/home/pstika/projects/PSBD-ViT/results/vit_tiny_sig_0_01/` |

A separate multi-target table is the right home, and the panel is the wrong one for a reason beyond the filter. The panel's ASR column is the single-class definition and this probe's is set membership with a higher chance floor, so placing the two in one column would compare quantities that are not the same, and the caption of the separate table has to state the definition and the chance floor for that reason. None of these folders has a detection cache, so no detection number exists to place anywhere and the table would be an attack-outcome table only. By the criticality scale in `paper/README.md` it rates at most SUPPORTING, given its cell count, its datasets, its seed replication and the absence of an interval. As a result that refuses the founding idea at the requested rate, it reads as a note on the clean-label ceiling for the failures or limitations material rather than as a detection finding. It should be generated by its own script under `scripts/paper/` from the sidecars in this note with every number a macro, and `paper/findings.md` currently carries no SIG or clean-label entry it could attach to, so the entry would be new.

## Files read

| path | what it supplied |
|---|---|
| `pbs/generate_multitarget_jobs.py` | the cells, the fixed requested rate, the `--attack-override num_targets` command shape, the folder tags and the docstring the poison-pool table corrects |
| `/lustre/home/pstika/projects/PSBD-ViT/pbs/generate_multitarget_jobs.py` | the `--seeds` version that produced both batches |
| `/lustre/home/pstika/projects/PSBD-ViT/pbs/vit_multitarget/*.pbs` and `/lustre/home/pstika/projects/PSBD-ViT/pbs/vit_multitarget_seeds/*.pbs` | the exact training commands per folder |
| `/lustre/home/pstika/projects/PSBD-ViT/logs/vit_multitarget/*.log` and `/lustre/home/pstika/projects/PSBD-ViT/logs/vit_multitarget_seeds/*.log` | the `final ASR=` and `saved` lines confirming completion and the sidecar values |
| `attacks/sig.py` | `SigConfig.num_targets`, `resolve_clean_label_mode`, `build` |
| `attacks/poisoning.py` | `LABEL_MODES`, `clean_label_target_set`, `is_poisonable`, `poisoned_label`, `is_eval_poisonable`, `attack_success_label`, `choose_poison_indices`, `AttackSuccessSet` |
| `evaluation/metrics.py` and `evaluation/loaders.py` | the worktree `evaluate_attack`, `attack_success_rate` with `success_labels`, `build_poisoned_loader` |
| `cli/train_backdoor.py` | the realized rate measurement and the sidecar fields written |
| `/lustre/home/pstika/projects/PSBD-ViT/evaluate.py`, `/lustre/home/pstika/projects/PSBD-ViT/defences/detection.py`, `/lustre/home/pstika/projects/PSBD-ViT/poison.py`, `/lustre/home/pstika/projects/PSBD-ViT/attacks/sig.py`, `/lustre/home/pstika/projects/PSBD-ViT/train_backdoor.py` at commit 2da1b6c | the ASR path that produced the recorded numbers, read with `git show` and `git diff` |
| `tests/test_attacks.py` | the pins on multi-target eligibility and label preservation |
| `checkpoints/vit_cifar100_sig_0_01*/args.json` and `checkpoints/vit_tiny_sig_0_01*/args.json` | the 14 sidecars |
| `checkpoints/vit_cifar100_sig_0_01/metrics.json` and `checkpoints/vit_tiny_sig_0_01/metrics.json` | the full-test single-target reference values |
| `checkpoints/vit_cifar100_benign/args.json`, `checkpoints/vit_cifar100_benign/metrics.json`, `checkpoints/vit_tiny_benign/args.json`, `checkpoints/vit_tiny_benign/metrics.json` | the benign references for dCA |
| `configs/psbd_basis.json` | the ASR bar, the drop bar, the panel label modes, the exclude tokens and the benign references |
| `scripts/coverage_ledger.py` | `is_panel_folder`, `clean_accuracy_of`, `classify_by_asr` |
| `scripts/paper/tab_panel.py` and `paper/README.md` | where the panel table reads from and the criticality scale |
| `data/registry.py` | the class counts |
| `docs/clean-label-rate-caps.md` | the per-class caps and class sizes |
| `docs/status-2026-09-09.md` | the entry that scheduled this probe |
| `/lustre/home/pstika/projects/PSBD-ViT/results/` | the absence of detection caches for the probe folders |
| `/tmp/claude-5225/-lustre-home-pstika-projects-PSBD-ViT/e54a8c64-6982-4399-8268-78a75b86bf49/scratchpad/summarize.py` | the arithmetic over the sidecars, a temporary script outside the repo |
