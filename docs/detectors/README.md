# Detector registry

`detectors/` holds the published input-level backdoor detectors PSBD is compared against, every 1 of them solving the same problem PSBD solves, given a suspicious input and a deployed model, decide whether the input carries a trigger, at the same test-time threat model. Methods that score a whole poisoned training set by clustering its representations, Spectral Signatures, Activation Clustering, SCAn, are out of scope entirely, since they need the training pool and produce a partition of it rather than a per-input decision.

This page is the entry point into the registry, `detectors/__init__.py`'s own docstring, and the shared rules that make a comparison across them meaningful. `docs/detector-ports.md` used to hold the full deviation record for STRIP, SCALE-UP, IBD-PSC and TeCo in 1 file, and that record now lives 1 paper per file under this directory, `strip.md`, `scale_up.md`, `ibd_psc.md`, `teco.md`, alongside the already-finished `cd_l.md` and `beatrix.md`.

## Shared rules

3 rules apply to every detector in `DETECTOR_NAMES`, stated once in `detectors/__init__.py` rather than repeated per module, so a comparison table's columns differ only in the score being thresholded.

**Direction.** Every detector returns per-sample scores where low means poisoned, PSBD's own convention, so `defences.decision.detection_report` applies to all of them unchanged. STRIP, CD-L and Beatrix already read low for poisoned in their own terms and are returned unnegated. SCALE-UP, IBD-PSC and TeCo define statistics that are high for poisoned, and each is negated exactly once at its own scoring boundary. Getting this wrong is silent, since it produces a well-formed, exactly inverted result rather than a crash, which is why every per-detector doc closes with a `Direction` section naming where the 1 negation happens, and why `experiments/preflight/check_signs.py` exists to catch the class of mistake directly on a checkpoint with a known answer.

**Data budget.** Every method that needs clean data gets the same clean validation split PSBD uses, the 2000-sample held-out slice `data.splits.build_psbd_loaders_from_checkpoint` produces. 2 papers, STRIP and SCALE-UP's data-limited variant, budget clean data differently in their own text, and those departures are recorded in their own docs rather than absorbed silently.

**Cost.** Each module states its forward-pass count per input in its own docstring, and `FORWARD_PASSES_PER_INPUT` repeats it in machine-readable form. The counts are far from equal, 1 for confidence and beatrix against 251 for CD-L, and that is a real deployment constraint a comparison table has to carry alongside the AUROC column rather than hide.

**Out-of-fit validation scores.** A 4th rule applies to the 2 methods that fit per-sample statistics on the clean validation split before scoring it, SCALE-UP's data-limited variant and Beatrix, named in `CROSS_FITTED`. A sample compared against class statistics it helped fit sits closer to that class's mean than a fresh sample would, so an in-sample threshold would be set too tight and the achieved false positive rate on the paired clean split would exceed the quantile budget. Both methods instead split the validation set into folds and score each fold against statistics fitted on the other folds only, `cross_fitted_validation_scores` at 2 folds for SCALE-UP and `jackknife_deviations` at 5 for Beatrix, so the validation column of a comparison table is never read off a fit the same rows produced.

## Registry

| Detector | Paper | Statistic | Clean data | Forwards per input | Precision | Job group |
|---|---|---|---|---|---|---|
| `confidence` | none, the null model | max softmax | none | 1 | autocast | cheap |
| `strip` | Gao et al., ACSAC 2019 | entropy under superimposition | 8 images, unlabelled, from the shared split | 8 | autocast | cheap |
| `scale_up` | Guo et al., ICLR 2023 | label consistency under pixel amplification | none | 6 | autocast | cheap |
| `scale_up_data_limited` | Guo et al., ICLR 2023 | the same, standardized per predicted class | the shared split, labelled | 6 | autocast | cheap |
| `ibd_psc` | Hou et al., ICML 2024 | retained confidence under parameter amplification | the shared split, labelled | 6 | autocast | cheap |
| `ibd_psc_calibrated` | Hou et al., ICML 2024, omega searched upward | the same at the smallest omega whose Algorithm 1 crosses xi | the shared split, labelled | 6 | autocast | cheap |
| `teco` | Liu et al., CVPR 2023 | spread of corruption hardness thresholds | none | 71 | autocast | teco |
| `cd_l` | Huang et al., ICLR 2023 | L1 norm of the distilled input mask | none | 251 | autocast, bfloat16 forward and backward with the mask, Adam state and objective in float32, contingent on a smoke pair against float32 | cd_l |
| `beatrix` | Ma et al., NDSS 2023 | Gram-matrix deviation from class bands | the shared split, unlabelled | 1 | autocast | cheap |
| `ted` | Mo et al., IEEE S&P 2024 | outlier rank trajectory over depth | the shared split, labelled | 1 | autocast | cheap |
| `sentinet` | Chou et al., S&P Workshops 2020 | residual above the clean (avgConf, fooled) envelope | 100 images plus the split | 202 | autocast | sentinet |

This is `DETECTOR_NAMES` as `detectors/__init__.py` declares it, 11 entries. `ibd_psc_calibrated` is the 1 registered variant of a faithful port, added after the smoke of 2026-09-10 showed the paper's amplification factor leaves a ViT's predictions intact (`ibd_psc.md`). `bad_expert` is registered under `EXPERIMENTAL_DETECTOR_NAMES` once its smoke run has fixed its settings and never joins a default run or a job until then.

## Clean-data budgets

Every method that reads clean data reads the same 2000-sample split, but how far that split goes per class varies enormously by dataset, and how far it falls short of a paper's own stated budget varies by paper.

| Dataset | Classes | Samples from the shared split | Per class |
|---|---|---|---|
| CIFAR-10 | 10 | 2000 | about 200 |
| CIFAR-100 | 100 | 2000 | about 20 |
| GTSRB | 43 | 2000 | about 46 |
| Tiny ImageNet | 200 | 2000 | about 10 |

| Detector | The paper's own stated budget |
|---|---|
| STRIP | 100 images |
| SCALE-UP, data-limited | 100 per class |
| IBD-PSC | 100 total |
| TeCo | none, the score reads no clean data at all |
| CD-L | 1% of the training set |
| Beatrix | 30 per class |
| TED | 1000 correctly predicted images |
| SentiNet | 100 images plus 400 for the boundary fit |

CIFAR-100 and Tiny ImageNet are the primary datasets this project reports on, and they are also the 2 where the shared split falls shortest of a paper's own budget, about 20 per class against SCALE-UP's 100 on CIFAR-100 and about 10 per class against Beatrix's 30 on Tiny. Both of those docs record the fallback rule their detector takes below a minimum per-class count, and a reader comparing an AUROC on 1 of those 2 datasets against the paper's own table should read that fallback section first.

## Where records land and how to regenerate them

`python -m cli.baselines --checkpoint-folder <folder>` scores every detector in `DETECTOR_NAMES` against 1 checkpoint's PSBD splits and writes 1 record per detector, `results/<folder>/detectors/<name>_metrics.json`, beside that detector's per-split score tensors, `<name>_scores_{validation,clean,backdoor}.pt`. `--all` runs every folder that already has a PSBD cache under `--results-dir`, `--detectors` narrows to a subset, `--skip-existing` skips a detector whose record already reached the `scored` status on a folder, and `--max-samples` truncates every split for a smoke run, which requires an explicit `--results-dir` away from the real results tree so a truncated record can never be mistaken for a full one.

`python -m cli.compare_detectors --markdown docs/detectors/comparison.md --per-detector-dir docs/detectors` reads every folder's PSBD metrics and detector records off disk and writes 2 things: a combined markdown table at `--markdown`'s path, and, for every column `--per-detector-dir` can address, the `<!-- results:begin --> ... <!-- results:end -->` block inside `<dir>/<name>.md`. The 2 SCALE-UP variants share `scale_up.md`, and their 2 columns land in its 1 block. A doc without the markers makes the command fail rather than write elsewhere.

`pbs/generate_detector_jobs.py --dry-run` prints the panel's unscored `(checkpoint, detector)` pairs, grouped by cost, `cheap`, `teco`, `cd_l`, `sentinet`, packed into jobs sized by `--hours`, without writing anything. Dropping `--dry-run` writes `pbs/psbd_detectors/<group>_<index>.pbs`. The smoke protocol this registry's numbers were produced under is recorded separately at `docs/runs/2026-09-10-detector-smoke.md`.

## Per-detector docs

| Doc | Paper | Status |
|---|---|---|
| `confidence.md` | none, the null model | in `DETECTOR_NAMES` |
| `strip.md` | Gao et al., ACSAC 2019 | in `DETECTOR_NAMES` |
| `scale_up.md` | Guo et al., ICLR 2023 | in `DETECTOR_NAMES`, both variants in 1 file |
| `ibd_psc.md` | Hou et al., ICML 2024 | in `DETECTOR_NAMES` |
| `teco.md` | Liu et al., CVPR 2023 | in `DETECTOR_NAMES` |
| `cd_l.md` | Huang et al., ICLR 2023 | in `DETECTOR_NAMES` |
| `beatrix.md` | Ma et al., NDSS 2023 | in `DETECTOR_NAMES` |
| `ted.md` | Mo et al., IEEE S&P 2024 | ported, `detectors/ted.py` exists, pending `DETECTOR_NAMES` registration |
| `sentinet.md` | Chou et al., IEEE S&P Workshops (DLS) 2020 | ported, `detectors/sentinet.py` exists, pending `DETECTOR_NAMES` registration |
| `bad_expert.md` | Xie et al., ICLR 2024, arXiv:2308.12439 | pending, no module and no doc yet |

BaDExpert is the 1 entry here with nothing behind it yet. `docs/paper-proposals.md` names it as the strongest baseline this project does not currently have, the 1 method in the test-time input-detection literature already shown to cover ViT-B/16, and porting it would follow the same pattern as the 8 detectors above, a module under `detectors/`, a registration in `DETECTOR_NAMES`, and a doc under this directory matching the skeleton `cd_l.md` and `beatrix.md` set.
