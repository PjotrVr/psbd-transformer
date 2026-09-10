# SVHN and EuroSAT as clean-label SIG carriers

## Question

The clean-label SIG attack cannot clear the panel's ASR bar on GTSRB, CIFAR-100 or Tiny, because a clean-label attack may only poison images that already carry the target label and on those datasets the target class is too small a share of the training set to reach the panel's rates (`docs/clean-label-rate-caps.md`). SVHN and EuroSAT were trained as datasets whose target class is large enough for every panel rate, each with a dirty-label blend control beside it and a seed sweep on every cell. This note asks whether the attack implants on them at each rate, how far the seeds spread, what it costs in clean accuracy and which edit to `configs/psbd_basis.json` admits the cells to the coverage panel.

Every number below is read from the `args.json` sidecar of the folder named in the same row, and I trained and swept nothing. The sidecars record the training commit with a dirty suffix, so the exact tree the runs came from cannot be recovered from the hash alone, and the bars, the commit and the checkout the ledger code was read from are in the table that follows.

| item | value | source |
|---|---|---|
| ASR bar | 0.85 | `configs/psbd_basis.json`, key `asr_bar` |
| clean accuracy drop bar | -0.05 | `configs/psbd_basis.json`, key `clean_accuracy_drop_bar` |
| epochs on every run | 15 | `checkpoints/vit_svhn_benign/args.json` and every sibling folder, key `epochs` |
| optimizer on every run | adam | the same sidecars, key `optimizer` |
| training commit on every run | `2da1b6c321db4f9c661d98dff81a3f8024913fa5-dirty` | the same sidecars, key `git_commit` |
| branch carrying that commit | `rewrite` | `git branch --contains` in `/lustre/home/pstika/projects/PSBD-ViT` |
| training window, UTC | 2026-09-10T01:38 to 2026-09-10T06:49 | `trained_started_at` and `trained_ended_at` across the same sidecars |
| ledger code read at commit | `848ee81` on `refactor/publication-layout` | `git log` in `/lustre/home/pstika/projects/PSBD-ViT-refactor` |
| ledger logic identical to the main checkout | yes, only docstrings and comments differ | `diff` of `scripts/coverage_ledger.py` across the 2 checkouts |
| basis JSON identical to the main checkout | yes | `diff` of `configs/psbd_basis.json` across the 2 checkouts |

## The runs

I read the benign reference first, because dCA is the poisoned run's clean accuracy minus the benign run's clean accuracy on the same dataset at the same architecture, optimizer and epoch count. Both benign folders exist at a single seed, so every dCA in this note is measured against a single reference model and inherits that model's own seed noise.

| folder | clean accuracy | seed | source |
|---|---:|---:|---|
| `vit_svhn_benign` | 0.9635 | 0 | `checkpoints/vit_svhn_benign/args.json` |
| `vit_eurosat_benign` | 0.9767 | 0 | `checkpoints/vit_eurosat_benign/args.json` |

On SVHN the clean-label runs target class `1`, the most frequent digit and the largest class, and carry the `_tl1` tag, while the blend control keeps the default target class. A row clears the bar when its ASR is at or above the ASR bar and its dCA is at or above the drop bar, and the panel cell column says whether the ledger's folder rule admits the row, since seed replicates are excluded by the `seed_` token and only the unsuffixed folders are panel cells.

| folder | attack | rate | realized rate | seed | target | ASR | clean acc | dCA | clears bar | panel cell | source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `vit_svhn_sig_0_01_tl1` | sig | 0.01 | 0.01001 | 0 | 1 | 0.6995 | 0.9597 | -0.0038 | no | yes | `checkpoints/vit_svhn_sig_0_01_tl1/args.json` |
| `vit_svhn_sig_0_01_tl1_seed_1` | sig | 0.01 | 0.01001 | 1 | 1 | 0.6957 | 0.9558 | -0.0078 | no | no, `seed_` excluded | `checkpoints/vit_svhn_sig_0_01_tl1_seed_1/args.json` |
| `vit_svhn_sig_0_01_tl1_seed_2` | sig | 0.01 | 0.01001 | 2 | 1 | 0.7838 | 0.9607 | -0.0028 | no | no, `seed_` excluded | `checkpoints/vit_svhn_sig_0_01_tl1_seed_2/args.json` |
| `vit_svhn_sig_0_05_tl1` | sig | 0.05 | 0.05000 | 0 | 1 | 0.8804 | 0.9576 | -0.0059 | yes | yes | `checkpoints/vit_svhn_sig_0_05_tl1/args.json` |
| `vit_svhn_sig_0_05_tl1_seed_1` | sig | 0.05 | 0.05000 | 1 | 1 | 0.9299 | 0.9613 | -0.0022 | yes | no, `seed_` excluded | `checkpoints/vit_svhn_sig_0_05_tl1_seed_1/args.json` |
| `vit_svhn_sig_0_05_tl1_seed_2` | sig | 0.05 | 0.05000 | 2 | 1 | 0.9177 | 0.9577 | -0.0058 | yes | no, `seed_` excluded | `checkpoints/vit_svhn_sig_0_05_tl1_seed_2/args.json` |
| `vit_svhn_sig_0_1_tl1` | sig | 0.1 | 0.10000 | 0 | 1 | 0.9713 | 0.9578 | -0.0058 | yes | yes | `checkpoints/vit_svhn_sig_0_1_tl1/args.json` |
| `vit_svhn_sig_0_1_tl1_seed_1` | sig | 0.1 | 0.10000 | 1 | 1 | 0.9937 | 0.9513 | -0.0123 | yes | no, `seed_` excluded | `checkpoints/vit_svhn_sig_0_1_tl1_seed_1/args.json` |
| `vit_svhn_sig_0_1_tl1_seed_2` | sig | 0.1 | 0.10000 | 2 | 1 | 0.9905 | 0.9609 | -0.0027 | yes | no, `seed_` excluded | `checkpoints/vit_svhn_sig_0_1_tl1_seed_2/args.json` |
| `vit_svhn_blend_0_1` | blend | 0.1 | 0.10000 | 0 | 0 | 0.9997 | 0.9513 | -0.0122 | yes | yes | `checkpoints/vit_svhn_blend_0_1/args.json` |
| `vit_svhn_blend_0_1_seed_1` | blend | 0.1 | 0.10000 | 1 | 0 | 1.0000 | 0.9540 | -0.0095 | yes | no, `seed_` excluded | `checkpoints/vit_svhn_blend_0_1_seed_1/args.json` |
| `vit_svhn_blend_0_1_seed_2` | blend | 0.1 | 0.10000 | 2 | 0 | 1.0000 | 0.9592 | -0.0043 | yes | no, `seed_` excluded | `checkpoints/vit_svhn_blend_0_1_seed_2/args.json` |

EuroSAT keeps the default target class for both attacks, since its class `0` (AnnualCrop) already exceeds the cap, so its folders carry no target tag. The single row whose dCA prints as a signed zero is the blend control at the last seed, whose clean accuracy matches the benign reference to the stored precision, and the residual sign comes from the reference being stored in single precision.

| folder | attack | rate | realized rate | seed | target | ASR | clean acc | dCA | clears bar | panel cell | source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `vit_eurosat_sig_0_01` | sig | 0.01 | 0.01000 | 0 | 0 | 0.8133 | 0.9691 | -0.0076 | no | yes | `checkpoints/vit_eurosat_sig_0_01/args.json` |
| `vit_eurosat_sig_0_01_seed_1` | sig | 0.01 | 0.01000 | 1 | 0 | 0.7616 | 0.9796 | +0.0030 | no | no, `seed_` excluded | `checkpoints/vit_eurosat_sig_0_01_seed_1/args.json` |
| `vit_eurosat_sig_0_01_seed_2` | sig | 0.01 | 0.01000 | 2 | 0 | 0.6954 | 0.9809 | +0.0043 | no | no, `seed_` excluded | `checkpoints/vit_eurosat_sig_0_01_seed_2/args.json` |
| `vit_eurosat_sig_0_05` | sig | 0.05 | 0.05000 | 0 | 0 | 0.8728 | 0.9789 | +0.0022 | yes | yes | `checkpoints/vit_eurosat_sig_0_05/args.json` |
| `vit_eurosat_sig_0_05_seed_1` | sig | 0.05 | 0.05000 | 1 | 0 | 0.8873 | 0.9809 | +0.0043 | yes | no, `seed_` excluded | `checkpoints/vit_eurosat_sig_0_05_seed_1/args.json` |
| `vit_eurosat_sig_0_05_seed_2` | sig | 0.05 | 0.05000 | 2 | 0 | 0.9222 | 0.9800 | +0.0033 | yes | no, `seed_` excluded | `checkpoints/vit_eurosat_sig_0_05_seed_2/args.json` |
| `vit_eurosat_sig_0_1` | sig | 0.1 | 0.10000 | 0 | 0 | 0.9220 | 0.9720 | -0.0046 | yes | yes | `checkpoints/vit_eurosat_sig_0_1/args.json` |
| `vit_eurosat_sig_0_1_seed_1` | sig | 0.1 | 0.10000 | 1 | 0 | 0.8443 | 0.9772 | +0.0006 | no | no, `seed_` excluded | `checkpoints/vit_eurosat_sig_0_1_seed_1/args.json` |
| `vit_eurosat_sig_0_1_seed_2` | sig | 0.1 | 0.10000 | 2 | 0 | 0.8710 | 0.9789 | +0.0022 | yes | no, `seed_` excluded | `checkpoints/vit_eurosat_sig_0_1_seed_2/args.json` |
| `vit_eurosat_blend_0_1` | blend | 0.1 | 0.10000 | 0 | 0 | 1.0000 | 0.9796 | +0.0030 | yes | yes | `checkpoints/vit_eurosat_blend_0_1/args.json` |
| `vit_eurosat_blend_0_1_seed_1` | blend | 0.1 | 0.10000 | 1 | 0 | 1.0000 | 0.9852 | +0.0085 | yes | no, `seed_` excluded | `checkpoints/vit_eurosat_blend_0_1_seed_1/args.json` |
| `vit_eurosat_blend_0_1_seed_2` | blend | 0.1 | 0.10000 | 2 | 0 | 1.0000 | 0.9767 | -0.0000 | yes | no, `seed_` excluded | `checkpoints/vit_eurosat_blend_0_1_seed_2/args.json` |

## Seed summary

Every (dataset, attack, rate) cell was trained at each seed in the sweep, and the summary below pools them. A cell clears on the mean when its mean ASR and its mean dCA both sit on the right side of their bars, and the seeds clearing column counts the individual replicates that clear on their own.

| dataset | attack | rate | n | mean ASR | min ASR | max ASR | ASR sd | mean clean acc | mean dCA | min dCA | clears on mean | seeds clearing | sources |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| svhn | sig | 0.01 | 3 | 0.7263 | 0.6957 | 0.7838 | 0.0498 | 0.9587 | -0.0048 | -0.0078 | no | 0/3 | `checkpoints/vit_svhn_sig_0_01_tl1/args.json` plus `_seed_1` and `_seed_2` |
| svhn | sig | 0.05 | 3 | 0.9094 | 0.8804 | 0.9299 | 0.0258 | 0.9589 | -0.0046 | -0.0059 | yes | 3/3 | `checkpoints/vit_svhn_sig_0_05_tl1/args.json` plus `_seed_1` and `_seed_2` |
| svhn | sig | 0.1 | 3 | 0.9852 | 0.9713 | 0.9937 | 0.0121 | 0.9566 | -0.0069 | -0.0123 | yes | 3/3 | `checkpoints/vit_svhn_sig_0_1_tl1/args.json` plus `_seed_1` and `_seed_2` |
| svhn | blend | 0.1 | 3 | 0.9999 | 0.9997 | 1.0000 | 0.0002 | 0.9549 | -0.0087 | -0.0122 | yes | 3/3 | `checkpoints/vit_svhn_blend_0_1/args.json` plus `_seed_1` and `_seed_2` |
| eurosat | sig | 0.01 | 3 | 0.7568 | 0.6954 | 0.8133 | 0.0591 | 0.9765 | -0.0001 | -0.0076 | no | 0/3 | `checkpoints/vit_eurosat_sig_0_01/args.json` plus `_seed_1` and `_seed_2` |
| eurosat | sig | 0.05 | 3 | 0.8941 | 0.8728 | 0.9222 | 0.0254 | 0.9799 | +0.0033 | +0.0022 | yes | 3/3 | `checkpoints/vit_eurosat_sig_0_05/args.json` plus `_seed_1` and `_seed_2` |
| eurosat | sig | 0.1 | 3 | 0.8791 | 0.8443 | 0.9220 | 0.0395 | 0.9760 | -0.0006 | -0.0046 | yes | 2/3 | `checkpoints/vit_eurosat_sig_0_1/args.json` plus `_seed_1` and `_seed_2` |
| eurosat | blend | 0.1 | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.9805 | +0.0038 | -0.0000 | yes | 3/3 | `checkpoints/vit_eurosat_blend_0_1/args.json` plus `_seed_1` and `_seed_2` |

On SVHN the ASR ladder is monotone, both on the means and on every seed taken individually, and no middle rate seed exceeds any top rate seed. On EuroSAT the ladder is not monotone, since the top rate mean sits below the middle rate mean and the top rate first replicate falls below every middle rate seed and below the ASR bar itself. The inversion sits inside the seed spread, because the drop from the middle rate minimum down to that replicate is smaller than the top rate's own seed range and the gap between the two means is smaller than the top rate's seed standard deviation, so on this evidence the top rate on EuroSAT saturates rather than declines. One reading is that the top rate poisons most of the target class on EuroSAT and leaves few unpoisoned class images against which the sinusoid can stand out as the easier feature, whereas on SVHN the top rate covers about half of the target class, and the share column in the next section carries both figures.

| quantity | value | source |
|---|---:|---|
| EuroSAT sig top rate, seed 1 ASR | 0.8443 | `checkpoints/vit_eurosat_sig_0_1_seed_1/args.json` |
| EuroSAT sig middle rate, seed 2 ASR | 0.9222 | `checkpoints/vit_eurosat_sig_0_05_seed_2/args.json` |
| EuroSAT sig middle rate, lowest seed ASR (seed 0) | 0.8728 | `checkpoints/vit_eurosat_sig_0_05/args.json` |
| drop from the middle rate minimum to the top rate seed 1 | 0.0285 | the 2 rows above |
| EuroSAT sig top rate seed range (max minus min) | 0.0778 | `checkpoints/vit_eurosat_sig_0_1/args.json` plus `_seed_1` and `_seed_2` |
| EuroSAT sig top rate seed standard deviation | 0.0395 | the same 3 sidecars |
| EuroSAT sig middle rate seed range | 0.0494 | `checkpoints/vit_eurosat_sig_0_05/args.json` plus `_seed_1` and `_seed_2` |
| EuroSAT sig top rate mean minus middle rate mean | -0.0150 | the 6 sidecars above |
| EuroSAT sig lowest rate maximum against middle rate minimum | 0.8133 against 0.8728 | `checkpoints/vit_eurosat_sig_0_01/args.json` and `checkpoints/vit_eurosat_sig_0_05/args.json` |
| SVHN sig lowest rate maximum against middle rate minimum | 0.7838 against 0.8804 | `checkpoints/vit_svhn_sig_0_01_tl1_seed_2/args.json` and `checkpoints/vit_svhn_sig_0_05_tl1/args.json` |
| SVHN sig middle rate seeds above any top rate seed | 0 | the 6 SVHN sig sidecars at the middle and top rates |

## Why these datasets reach 10%

A clean-label attack keeps the label, so it can only poison images already in the target class, and its highest reachable rate is the target class size over the train set size. `poison.choose_poison_indices` clamps silently above that cap, so every requested rate above it trains the identical index set (`docs/clean-label-rate-caps.md`). The table gives the class count, train size, target class size and share for the new datasets beside the existing ones, together with the image count each panel rate demands, and the counts for SVHN and EuroSAT were measured by loading the raw data through the repo's own loader.

| dataset | classes | train size | target class | target class size | share of train | images at 10% | images at 5% | images at 1% | share of target class poisoned at 10% | source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| svhn | 10 | 73257 | 1 | 13861 | 0.1892 | 7326 | 3663 | 733 | 0.529 | `raw_data/svhn/train_32x32.mat` through `data/loading.py`, class count from `data/registry.py` |
| eurosat | 10 | 21600 of 27000 after the split | 0 | 2436 | 0.1128 | 2160 | 1080 | 216 | 0.887 | `raw_data/eurosat/eurosat` through `split_eurosat` in `data/loading.py` (`EUROSAT_SPLIT_SEED` 0, `EUROSAT_TEST_FRACTION` 0.2), class count from `data/registry.py` |
| cifar10 | 10 | 50000 | 0 | 5000 | 0.1000 | 5000 | 2500 | 500 | 1.000 | `docs/clean-label-rate-caps.md` |
| gtsrb, class 1 | 43 | 26640 | 1 | 1500 | 0.0563 | 2664 | 1332 | 266 | unreachable | `docs/clean-label-rate-caps.md` |
| gtsrb, class 0 | 43 | 26640 | 0 | 150 | 0.0056 | 2664 | 1332 | 266 | unreachable | `docs/clean-label-rate-caps.md` |
| cifar100 | 100 | 50000 | 0 | 500 | 0.0100 | 5000 | 2500 | 500 | unreachable | `docs/clean-label-rate-caps.md` |
| tiny | 200 | 100000 | 0 | 500 | 0.0050 | 10000 | 5000 | 1000 | unreachable | `docs/clean-label-rate-caps.md` |

SVHN's target class is the most frequent street view digit and holds close to a fifth of the train set, and the `_tl1` tag records the departure from the default target class in the same way the GTSRB clean-label runs did before it. EuroSAT is near class balanced, so its default class already clears the cap after the fixed permutation split, and both datasets share their class count with CIFAR-10, the only existing dataset that reaches the top rate. GTSRB's largest class stops between the middle and top rates, and CIFAR-100 and Tiny stop at or below the lowest rate, which is why the clean-label column of the panel was empty above CIFAR-10 before these runs.

## Proposed basis edit

`is_panel_folder` in `scripts/coverage_ledger.py` applies no dataset filter at all. It rejects a folder when any `exclude_folder_tokens` entry appears in the name, when the attack has a `canonical_variants` entry the name lacks, or when the architecture, label mode or poison rate fall outside the panel, and nothing else. `_tl1` matches none of the excluded tokens and `sig` has no canonical variant, so I ran the function over every SVHN and EuroSAT folder, and every unsuffixed folder passes while every seed replicate is dropped by `seed_`, as the table records.

| folder | `is_panel_folder` | excluded token hit | slot resolution | source |
|---|---|---|---|---|
| `vit_svhn_sig_0_01_tl1` | true | none | single candidate for (svhn, sig, 0.01), class below_bar | dry run of `is_panel_folder` (lines 36 to 51) and `resolve_one_per_attack` (lines 342 to 372) in `scripts/coverage_ledger.py` |
| `vit_svhn_sig_0_05_tl1` | true | none | single candidate for (svhn, sig, 0.05), class clears | the same dry run |
| `vit_svhn_sig_0_1_tl1` | true | none | single candidate for (svhn, sig, 0.1), class clears | the same dry run |
| `vit_svhn_blend_0_1` | true | none | single candidate for (svhn, blend, 0.1), class clears | the same dry run |
| `vit_eurosat_sig_0_01` | true | none | single candidate for (eurosat, sig, 0.01), class below_bar | the same dry run |
| `vit_eurosat_sig_0_05` | true | none | single candidate for (eurosat, sig, 0.05), class clears | the same dry run |
| `vit_eurosat_sig_0_1` | true | none | single candidate for (eurosat, sig, 0.1), class clears | the same dry run |
| `vit_eurosat_blend_0_1` | true | none | single candidate for (eurosat, blend, 0.1), class clears | the same dry run |
| every `*_seed_1` and `*_seed_2` sibling | false | `seed_` | not a panel cell | the same dry run |
| `vit_svhn_benign` and `vit_eurosat_benign` | false | `benign` | not a panel cell | the same dry run |

What the ledger lacks is a benign reference for the new datasets. `benign_reference_accuracy` builds its dictionary from the `benign_reference` block alone, so for a cell on a dataset absent from that block `benign.get(cell["dataset"])` returns `None`, `clean_accuracy_drop` is `None`, the dCA column prints as missing and the `clears ASR, FAILS dCA` verdict can never fire. The edit that admits both datasets is therefore the addition of their benign folders to that block, and nothing else in the file has to change.

The current fragment of `configs/psbd_basis.json` reads as follows. It is quoted exactly as the file holds it, comment included.

```json
  "benign_reference": {
    "_comment": "Same architecture, optimizer and 15 epochs as every panel cell, so dCA is like-for-like.",
    "cifar10": "vit_cifar10_benign",
    "cifar100": "vit_cifar100_benign",
    "gtsrb": "vit_gtsrb_benign",
    "tiny": "vit_tiny_benign"
  },
```

The proposed replacement follows. It adds both references at the end of the block and touches nothing else.

```json
  "benign_reference": {
    "_comment": "Same architecture, optimizer and 15 epochs as every panel cell, so dCA is like-for-like.",
    "cifar10": "vit_cifar10_benign",
    "cifar100": "vit_cifar100_benign",
    "gtsrb": "vit_gtsrb_benign",
    "tiny": "vit_tiny_benign",
    "svhn": "vit_svhn_benign",
    "eurosat": "vit_eurosat_benign"
  },
```

The `canonical_variants` block stays as it is. Its key is the attack name alone, since the ledger reads `panel.get("canonical_variants", {}).get(metadata.get("attack"))`, so mapping `sig` to `_tl1` would apply on every dataset and drop each `sig` folder without the tag, including `vit_cifar10_sig_0_1`, which clears the ASR bar in the current ledger, and every CIFAR-100 and Tiny `sig` cell. On SVHN the tag needs no rule, because every SVHN `sig` folder carries it, so each (svhn, sig, rate) slot holds a single candidate and `resolve_one_per_attack` returns it without consulting the ASR class.

```json
    "canonical_variants": {
      "lc": "_adv"
    },
```

The other readers of `benign_reference` iterate the block rather than a fixed list, so both new entries flow into them without further edits. I list each with what the new entries do there, so nothing is admitted by accident.

| file | line | effect of the 2 new entries |
|---|---:|---|
| `scripts/coverage_ledger.py` | 131 to 139 and 296 | supplies the dCA denominator for every SVHN and EuroSAT cell |
| `scripts/coverage_ledger.py` | 544 to 548 | the Swin override rewrites `vit_` to `swin_`, which names folders that do not exist, and `clean_accuracy_of` returns `None` for them, so a Swin ledger is unaffected |
| `pbs/generate_detector_jobs.py` | 145 | appends both benign folders to the detector job list as `benign` cells, which the per-dataset detector baselines need |
| `cli/compare_detectors.py` | 128 | adds both as benign cells with no poison rate |
| `scripts/paper/mech_shift_target.py` | 496 | loads `psbd_metrics` for both and filters them out while they are unswept |

The panel itself needs no dataset list, but the report generators iterate a hardcoded `DATASET_ORDER` tuple, so the new cells reach `coverage.json`, `gaps.json` and the attack strength table of `COVERAGE.md` on the next ledger run and stay out of the paper tables until those tuples grow. `selection_protocol` names neither dataset in `select_on_datasets` or `report_on_datasets`, and since that protocol was frozen before these runs existed I leave the placement as an open choice, where the natural side is `report_on_datasets` and adding to a pre-registered list is a protocol decision the user should make explicitly.

| file | line | current value | role |
|---|---:|---|---|
| `scripts/vit_detection_tables.py` | 28 | `("cifar10", "cifar100", "gtsrb", "tiny")` | iterated as the row set at line 150 |
| `scripts/vit_config_tables.py` | 45 | the same tuple | iterated at line 196 |
| `scripts/paper/tab_panel.py` | 28 | the same tuple | assigned to `datasets` at line 97 |
| `scripts/paper/tab_headline.py` | 50 | the same tuple | iterated at line 113 |
| `scripts/paper/tab_family_split.py` | 36 | the same tuple | iterated at line 104 |
| `configs/psbd_basis.json` | 680 and 684 | `select_on_datasets` holds cifar10 and gtsrb, `report_on_datasets` holds cifar100 and tiny | read by `experiments/probe_fusion/summarise.py` at lines 85 to 86 |

One thing will stop the next ledger run before it rewrites `COVERAGE.md`, and it comes from the same `_tl1` token on GTSRB rather than on SVHN. The GTSRB clean-label `sig` folders at target class `1` finished after the ledger last ran, so the current `COVERAGE.md` predates them, and on the next run `resolve_one_per_attack` sees the class `0` run and the `_tl1` run in the same (gtsrb, sig, rate) slot with neither clearing the bar and raises. `write_artifacts` writes `coverage.json` and `gaps.json` before `render_markdown` is called, so the JSON artifacts and the job generator still work and `COVERAGE.md` is the file left stale. An `exclude_folder_tokens` entry cannot single out the class `0` GTSRB runs because they carry no tag of their own, and `canonical_variants` cannot express a per-dataset choice, so the fix is either a rename of the superseded runs or a per-dataset key in the ledger, and I leave that choice open since it lies outside this note's edit.

| slot | candidates | candidate ASRs | candidates clearing | outcome | source |
|---|---|---|---:|---|---|
| (gtsrb, sig, 0.01) | `vit_gtsrb_sig_0_01`, `vit_gtsrb_sig_0_01_tl1` | 0.460, 0.356 | 0 | `ValueError`, slot ambiguous | dry run of `resolve_one_per_attack`, ASRs from `checkpoints/vit_gtsrb_sig_0_01/args.json` and `checkpoints/vit_gtsrb_sig_0_01_tl1/args.json` |
| (gtsrb, sig, 0.05) | `vit_gtsrb_sig_0_05`, `vit_gtsrb_sig_0_05_tl1`, `vit_gtsrb_sig_0_05_v2` | 0.015, 0.673, unmeasured | 0 | `ValueError`, slot ambiguous | the same dry run, ASRs from the 3 matching `args.json` sidecars |
| (gtsrb, sig, 0.1) | `vit_gtsrb_sig_0_1` | 0.415 | 0 | resolves, single candidate | the same dry run, `checkpoints/vit_gtsrb_sig_0_1/args.json` |
| ledger last generated | | 2026-09-09T19:38:19+00:00 | | | `results/coverage/COVERAGE.md` line 6 |
| `vit_gtsrb_sig_0_01_tl1` finished | | 2026-09-10T05:16:48+00:00 | | | `checkpoints/vit_gtsrb_sig_0_01_tl1/args.json`, key `trained_ended_at` |
| `vit_gtsrb_sig_0_05_tl1` finished | | 2026-09-10T06:04:49+00:00 | | | `checkpoints/vit_gtsrb_sig_0_05_tl1/args.json`, key `trained_ended_at` |

## Cells that would enter the panel

The ledger classifies a cell by ASR alone in `classify_by_asr`, and the dCA bar is a separate annotation on the verdict column, so a cell's membership in the clearing set is settled by its `args.json` on the next ledger run and does not wait for a sweep. Of the unsuffixed folders, the middle and top rate `sig` cells and the blend control clear on both datasets, and both lowest rate `sig` cells fall below the ASR bar with dCA well inside the drop bar. Once swept, each clearing cell owes the full basis of placements, so the gap list grows by the product of clearing cells and basis size at `--asr-class clears`, and the count table gives the totals.

| folder | attack | rate | ASR | dCA | verdict | source |
|---|---|---:|---:|---:|---|---|
| `vit_svhn_sig_0_05_tl1` | sig | 0.05 | 0.8804 | -0.0059 | clears | `checkpoints/vit_svhn_sig_0_05_tl1/args.json` |
| `vit_svhn_sig_0_1_tl1` | sig | 0.1 | 0.9713 | -0.0058 | clears | `checkpoints/vit_svhn_sig_0_1_tl1/args.json` |
| `vit_svhn_blend_0_1` | blend | 0.1 | 0.9997 | -0.0122 | clears | `checkpoints/vit_svhn_blend_0_1/args.json` |
| `vit_eurosat_sig_0_05` | sig | 0.05 | 0.8728 | +0.0022 | clears | `checkpoints/vit_eurosat_sig_0_05/args.json` |
| `vit_eurosat_sig_0_1` | sig | 0.1 | 0.9220 | -0.0046 | clears | `checkpoints/vit_eurosat_sig_0_1/args.json` |
| `vit_eurosat_blend_0_1` | blend | 0.1 | 1.0000 | +0.0030 | clears | `checkpoints/vit_eurosat_blend_0_1/args.json` |
| `vit_svhn_sig_0_01_tl1` | sig | 0.01 | 0.6995 | -0.0038 | below_bar | `checkpoints/vit_svhn_sig_0_01_tl1/args.json` |
| `vit_eurosat_sig_0_01` | sig | 0.01 | 0.8133 | -0.0076 | below_bar | `checkpoints/vit_eurosat_sig_0_01/args.json` |

| count | value | source |
|---|---:|---|
| clearing cells in the current ledger | 65 | `results/coverage/COVERAGE.md` line 11 |
| new clearing cells | 6 | the table above |
| clearing cells after the next ledger run | 71 | the 2 rows above |
| panel cells in the current ledger | 96 | `results/coverage/COVERAGE.md` line 7 |
| new panel cells | 8 | the dry run table in the previous section |
| panel cells after the next ledger run | 104 | the 2 rows above |
| below bar cells in the current ledger | 31 | `results/coverage/COVERAGE.md` line 11 |
| below bar cells after the next ledger run | 33 | the row above plus the 2 below_bar rows |
| basis size | 18 | `results/coverage/COVERAGE.md` line 8 |
| rate units per cell across the basis | 197 | the `rates` arrays under `basis` in `configs/psbd_basis.json` |
| new missing (cell, placement) slots at `--asr-class clears` | 108 | 6 cells by 18 placements |
| new missing rate units at `--asr-class clears` | 1182 | 6 cells by 197 rate units |

## Files read

| path | what it supplied |
|---|---|
| `checkpoints/vit_svhn_benign/args.json` | SVHN benign reference clean accuracy, commit, epochs, optimizer |
| `checkpoints/vit_eurosat_benign/args.json` | EuroSAT benign reference clean accuracy, commit, epochs, optimizer |
| `checkpoints/vit_svhn_sig_0_01_tl1/args.json`, `_seed_1`, `_seed_2` | SVHN SIG lowest rate, all seeds |
| `checkpoints/vit_svhn_sig_0_05_tl1/args.json`, `_seed_1`, `_seed_2` | SVHN SIG middle rate, all seeds |
| `checkpoints/vit_svhn_sig_0_1_tl1/args.json`, `_seed_1`, `_seed_2` | SVHN SIG top rate, all seeds |
| `checkpoints/vit_svhn_blend_0_1/args.json`, `_seed_1`, `_seed_2` | SVHN dirty-label control, all seeds |
| `checkpoints/vit_eurosat_sig_0_01/args.json`, `_seed_1`, `_seed_2` | EuroSAT SIG lowest rate, all seeds |
| `checkpoints/vit_eurosat_sig_0_05/args.json`, `_seed_1`, `_seed_2` | EuroSAT SIG middle rate, all seeds |
| `checkpoints/vit_eurosat_sig_0_1/args.json`, `_seed_1`, `_seed_2` | EuroSAT SIG top rate, all seeds |
| `checkpoints/vit_eurosat_blend_0_1/args.json`, `_seed_1`, `_seed_2` | EuroSAT dirty-label control, all seeds |
| `checkpoints/vit_gtsrb_sig_0_01_tl1/args.json`, `checkpoints/vit_gtsrb_sig_0_05_tl1/args.json`, `checkpoints/vit_gtsrb_sig_0_005_tl1/args.json` | GTSRB `_tl1` SIG finish times and ASRs, for the slot ambiguity |
| `checkpoints/vit_gtsrb_sig_0_01/args.json`, `checkpoints/vit_gtsrb_sig_0_05/args.json`, `checkpoints/vit_gtsrb_sig_0_05_v2/args.json`, `checkpoints/vit_gtsrb_sig_0_1/args.json` | GTSRB class `0` SIG ASRs, for the slot ambiguity |
| every other `checkpoints/vit_*_sig_*/args.json` | target label, realized rate and ASR, to confirm no SIG folder outside GTSRB and SVHN carries `_tl1` |
| `configs/psbd_basis.json` | bars, panel rule, excluded tokens, canonical variants, benign references, basis rate ladders, selection protocol |
| `scripts/coverage_ledger.py` | `is_panel_folder`, `resolve_one_per_attack`, `classify_by_asr`, `benign_reference_accuracy`, `write_artifacts`, the Swin override |
| `results/coverage/COVERAGE.md` | current panel, clearing and below bar counts, basis size, generation time, current SIG rows |
| `data/registry.py` | `DATASET_REGISTRY` class counts and the comment stating why SVHN and EuroSAT are in the panel |
| `data/loading.py` | `split_eurosat`, `EUROSAT_SPLIT_SEED`, `EUROSAT_TEST_FRACTION`, the SVHN loader |
| `raw_data/svhn/train_32x32.mat` | SVHN train labels, read through the repo loader for the class counts |
| `raw_data/eurosat/eurosat` | EuroSAT image folder, read through `split_eurosat` for the train split class counts |
| `docs/clean-label-rate-caps.md` | the cap arithmetic and the class sizes for CIFAR-10, CIFAR-100, GTSRB and Tiny |
| `pbs/generate_basis_jobs.py` | confirmation that the job generator reads `gaps.json` and `coverage.json` and carries no dataset list |
| `pbs/generate_detector_jobs.py`, `cli/compare_detectors.py`, `scripts/paper/mech_shift_target.py` | the other readers of `benign_reference` |
| `scripts/vit_detection_tables.py`, `scripts/vit_config_tables.py`, `scripts/paper/tab_panel.py`, `scripts/paper/tab_headline.py`, `scripts/paper/tab_family_split.py` | the hardcoded `DATASET_ORDER` tuples |
| `experiments/probe_fusion/summarise.py` | the reader of `select_on_datasets` and `report_on_datasets` |
| `/lustre/home/pstika/projects/PSBD-ViT/scripts/coverage_ledger.py`, `/lustre/home/pstika/projects/PSBD-ViT/configs/psbd_basis.json`, `/lustre/home/pstika/projects/PSBD-ViT/results/coverage/COVERAGE.md` | the main checkout copies, diffed against the worktree |
| `.claude/styles/writing-style.md` and `.claude/CLAUDE.md` in the main checkout | the format rules this note follows |
