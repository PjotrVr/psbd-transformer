# Architecture

A walkthrough of the tree as it stands after the 2026-09-10 reorganisation, in
dependency order, leaf packages first and entry points last. For the raw
directory listing see `docs/repository-layout.md`. For the constants every
table reads see `.claude/CLAUDE.md`.

## The 10 packages

### attacks

Owns the 10 trigger implementations and everything that decides which images
a poisoning run touches. `attacks/__init__.py` is the only package that
re-exports its submodules, since it is a registry: `ATTACK_NAMES` lists every
attack, `build_attack` constructs one from a name and a config, and
`default_config` gives that attack's published hyperparameters. The eligibility
and labelling logic that every other package trusts lives in `poisoning.py`:
`AttackSuccessSet` builds the evaluation set a checkpoint's success rate is
measured on, `is_eval_poisonable` and `attack_success_label` decide which
images belong in it, and `choose_poison_indices` selects the training-time
poisoned set under the clean-label cap. `patterns.py` holds the shared trigger
masks, `bases.py` the Label-Consistent adversarial base images, `adversarial.py`
the PGD step those bases are built from, and `evasion.py` the adaptive attacker
that trains against a named probe.

### data

Owns dataset identity and the 2 ways a checkpoint's evaluation data gets
rebuilt. `registry.py` is the single source of a dataset's class count,
normalisation statistics and folder-name parsing (`DATASET_REGISTRY`,
`label_mode_from_folder`). `loading.py` turns that registry entry into
loadable tensors (`load_clean_datasets`). `splits.py` holds the 2000-image
PSBD validation split (`psbd_split_permutation`) and `read_checkpoint_metadata`,
which rebuilds an attack's evaluation set from a checkpoint's own `args.json`
with no PNG folder needed. `backdoorbench.py` is the parallel path for
checkpoints under `backdoor_bench_checkpoints/`, which carry no `args.json`
and read their triggers from PNG files instead (`PngPathDataset`,
`load_backdoor_splits`).

### models

Owns the 2 architectures and the mechanism that injects a probe into either
of them without touching a weight. `backbones.py` builds and loads ViT-B/16
and Swin-S (`build_vit`, `build_swin`, `load_checkpoint`, `network_core`).
`positions.py` holds `POSITION_REGISTRY`, 1 `PositionSpec` table per
architecture naming every submodule boundary a probe can attach to, and
`plug_dropout` and `unplug_dropout`, which attach and remove a fresh
perturbation module through forward hooks. `activate_model_dropout` is the
single deliberate exception, switching a model's own trained dropout back on
to study what that conflates with the probe.

### training

Owns the training loop and its 2 optimisers. `loop.py` holds `train_classifier`,
the full 15-epoch run, `save_checkpoint`, which writes the `args.json`
provenance sidecar next to `attack_result.pt`, and `check_not_diverged`, which
refuses to save a run whose final validation accuracy collapsed against its
own best epoch. `sam.py` holds `SAM`, the 2-pass sharpness-aware optimiser
that wraps the same Adam every run uses.

### defences

Owns PSBD itself, split by what each stage of the pipeline needs.
`operators.py` defines what gets injected at a probed position: `TokenMask`,
`GaussianNoise`, `DropPath` and the rest, built from a name by
`build_operator`. `inference.py` runs the forward passes a stage-1 sweep
needs (`forward_logits`, `compute_dropout_pass_probs`, `frozen_parameters`).
`scores.py` turns cached probabilities into the prediction shift unit and its
fractional form (`psu_from_cache`, `psu_ratio_from_cache`, `shift_ratio`).
`decision.py` holds the canonical constants (`ADAPTIVE_SHIFT_TARGET`,
`PLACEMENT_MATCH_TARGET`, `RECOMMENDED_PLACEMENT`, `PUBLISHED_PLACEMENT`) and
the functions that turn a score distribution into a verdict
(`select_rate_adaptively`, `select_rate_at_matched_shift`, `detection_report`).
`cache.py` owns the on-disk layout of `results/<folder>/psbd/`, reading and
writing the raw tensors the other 2 modules consume.

### detectors

Owns the 11 competitor input detectors behind a single registry.
`detectors/__init__.py` holds `DETECTOR_NAMES`, `build_detector` and
`DetectorContext`, the bundle of loaders and a checkpoint a detector needs to
score. `CROSS_FITTED` names the detectors that fit per-sample statistics on
the validation split and so must return out-of-fit scores for it. Each method
gets its own module (`strip.py`, `ibd_psc.py`, `beatrix.py`, `ted.py` and the
rest), and `records.py` owns the on-disk layout under
`results/<folder>/detectors/` (`save_report`, `save_scores`, `scored_detectors`).

### analysis

Owns the latent-space statistics behind the paper's mechanistic claims, with
no matplotlib import anywhere in the package. `features.py` extracts per-layer
CLS features from a model (`captured_layers`, `transformer_blocks`). `cka.py`
computes debiased linear CKA between 2 feature sets (`linear_cka`,
`unbiased_hsic`). `direction.py` estimates and projects onto the backdoor
direction (`backdoor_direction`, `project_onto_direction`). `embedding.py`
gives PCA and UMAP projections (`pca_project`, `umap_project`), UMAP preferred
over t-SNE throughout the project. `lipschitz.py`, `distribution.py`,
`stealth.py`, `cases.py`, `latent.py`, `neurons.py`, `attribution.py`,
`samples.py`, `curvature.py`, `landscape.py` and `synthesis.py` round out the
package, the last 6 ported from BackdoorBench.

### visualization

Owns every figure built over the statistics in `analysis/`, 1 module per
figure family (`bars.py`, `heatmaps.py`, `matrices.py`, `scatter.py`,
`spectra.py`, `surfaces.py`, `distribution_plots.py`) plus `style.py` for the
shared look. Every tool a `cli.visualize` subcommand can call is registered in
`cheap_tools.py` or `expensive_tools.py`, split by whether it costs seconds or
minutes of GPU time per checkpoint, and every figure gets a JSON sidecar of
the numbers behind it, written by `sidecar.py`'s `write_sidecar`. See
`docs/visualization/expensive-tools.md` for which BackdoorBench script each
of the expensive tools ports and every deviation from it.

### evaluation

Owns attack success and clean accuracy, the metrics that decide whether a
trained checkpoint is worth sweeping at all. `metrics.py` holds
`attack_success_rate`, `clean_accuracy`, `evaluate_checkpoint` and `auroc`, the
last also used outside this package as the shared detection metric.
`loaders.py` builds the clean and poisoned evaluation loaders
(`build_clean_loader`, `build_poisoned_loader`, `build_balanced_eval_loaders`).
`summary.py` reads `results/detection_summary.csv` back into a data frame
(`load_detection_summary`, `summary_coverage`).

### utils

Owns numerics only, with no subject of its own. `numerics.py` holds
`safe_ratio` and `safe_ratio_positive`, the floor-guarded division every rate
computation in `defences/` and `evaluation/` goes through, and `is_defined`.
`provenance.py` holds `current_git_commit` and `utc_timestamp`, the 2 values
every checkpoint's `args.json` and every PSBD run's provenance record carries.

## cli

The only place a `main()` lives. Every command is a thin argument parser over
a package call, so deleting a `cli/` module costs nothing but the parsing.
Training: `train_backdoor.py`, `train_benign.py`. The PSBD pipeline:
`sweep.py` (stage 1, GPU) and `analyze.py` (stage 2, CPU). Competitors:
`baselines.py`. Reporting: `compare.py`, which folds what used to be 7
separate modules (`report`, `summary`, `tables`, `variants`,
`operating_points`, `compare_detectors`, `fuse_detectors`) into 4 subcommands
under 1 entry point, described below. The 7 old module names still work, each
now a 2-line shim forwarding its arguments into `compare.py`, so a PBS script
or a doc example spelling `python -m cli.report` keeps running unchanged.
Others: `evaluate.py`, `backfill.py`, `lc_bases.py`, `head_profile.py`,
`analyze_latent.py`, `visualize.py`.

`compare.py` has 4 subcommands. `placements` covers what `report`, `summary`,
`tables` and `variants` used to do separately, with a further action picking
which of the 4 (`python -m cli.compare placements report`). `operating-points`
reads detection at low false-positive rates, 1% and 5%. `detectors` builds 1
table per metric with PSBD against every competitor detector. `fused` combines
PSBD with 1 named competitor by rank. Running `python -m cli.compare --help`
and `python -m cli.compare <subcommand> --help` lists every flag.

## Data flow, checkpoint to paper

1. `python -m cli.train_backdoor` trains a backdoored checkpoint and writes
   `checkpoints/<folder>/attack_result.pt` beside its `args.json` provenance
   sidecar. `python -m cli.train_benign` writes the benign negative controls
   the same way.
2. `python -m cli.evaluate` reads every checkpoint under `checkpoints/` and
   writes each one's attack-success and clean-accuracy `metrics.json` next to
   `attack_result.pt`. A checkpoint whose backdoor never fires is dropped from
   later sweeps by the job generators in `pbs/`, which read this file.
3. `python -m cli.sweep` is PSBD stage 1, on GPU. Given a checkpoint folder,
   a position and an operator, it runs the stochastic forward passes and
   writes the raw per-pass probabilities to
   `results/<folder>/psbd/<placement>/`, with a `run_<placement>.json`
   provenance record naming the position, the operator and the git commit
   that produced it.
4. `python -m cli.analyze` is PSBD stage 2, on CPU, seconds per checkpoint. It
   reads the stage-1 cache and writes `results/<folder>/psbd_metrics.json`,
   the detection numbers for every dropout rate that was swept.
5. `python -m cli.baselines` scores the 11 competitor detectors on the exact
   same splits and quantiles, writing 1 report and 1 score tensor per
   detector under `results/<folder>/detectors/`.
6. `python -m cli.compare` folds every checkpoint's `psbd_metrics.json` and
   every detector's report into the comparison tables described above.
7. `scripts/coverage_ledger.py` reads `configs/psbd_basis.json`, the
   declared basis of placements every dataset and attack combination should
   carry, against what actually exists under `results/`, and writes
   `results/coverage/coverage.json` and `results/coverage/gaps.json`. This is
   the file the job generators in `pbs/` read to decide what still needs
   running.
8. `scripts/paper/build_all.py` runs every generator under `scripts/paper/`
   (the `tab_*` table generators, the `fig_*` figure generators, the `mech_*`
   mechanism generators and the `app_*` appendix generators), each reading
   `results/coverage/coverage.json` and the cached tensors it points at,
   folds their macro sidecars into `paper/headline.tex`, and fails the build
   if a rendered chapter carries a bare number that is not backed by a macro
   a generator produced.

## On-disk layout

`checkpoints/<folder>/` holds `attack_result.pt`, `args.json` and, once
evaluated, `metrics.json`. The folder name follows the canonical template in
`.claude/CLAUDE.md`. `backdoor_bench_checkpoints/` is the parallel,
untracked, read-only tree of BackdoorBench's own downloaded checkpoints, which
carry no `args.json` and are read through `data.backdoorbench` instead.

`results/<folder>/` is not written at training time. It exists only once
something has run against that checkpoint, with 2 subtrees.
`results/<folder>/psbd/` is created by `cli.sweep`. Under it sits 1 directory
per placement (`<position>_<operator>`, for example
`before_attention_norm_token_mask/`), holding the raw per-pass probability
tensors and a `run_<placement>.json` provenance file, plus a shared
`baseline_clean.pt` and `split_manifest.json` for the checkpoint. Stage 2
writes `results/<folder>/psbd_metrics.json` as a sibling of `psbd/`, never
inside it. `results/<folder>/detectors/` is created by `cli.baselines`, 1
`<name>_metrics.json` report and 1 `<name>_scores_<split>.pt` tensor per
competitor detector. An orphaned `results/<folder>/` with no matching
`checkpoints/<folder>/` is inert and is never pruned automatically.

`results/_experiments/<name>/` is where an `experiments/` directory writes
its own output once that output is more than a single stray file, keyed by
the experiment's own name rather than by checkpoint. Some older
`experiments/` scripts still write directly under `results/` as a single
named file (`results/low_fpr_audit.json`, `results/sam_backdoor_effect.json`)
rather than under `results/_experiments/`, a naming split that predates the
`_experiments/` convention and has not been backfilled.

`results/coverage/` holds the ledger: `coverage.json` (what the basis
declares against what is on disk), `gaps.json` (what is still missing),
`split_integrity.json` and `COVERAGE.md`, the human-readable render of the
same numbers.

`paper/` is the built paper. `paper/sections/*.tex` are the chapters,
`paper/tables/*.tex` and `*.macros.json` are 1 pair per table generator,
`paper/figures/*.pdf` and their `*.json` sidecars are 1 pair per figure
generator, and `paper/headline.tex` is the folded macro file every chapter
reads its numbers from. `paper/main.tex` assembles the chapters and compiles
to `paper/main.pdf`.

## pbs

Job generators, not job scripts. Each `pbs/generate_*.py` is a Python module
that derives its checkpoint or configuration list from something already on
disk (a folder's `args.json`, the coverage ledger, an earlier run's
`metrics.json`), rather than a hand-typed list, and writes 1 `.pbs` file per
job into a same-named subdirectory (`pbs/vit_basis/`, `pbs/psbd_seed_2026_09_11/`
and the rest). `grid.py` is the reusable core behind several generators: it
flattens a parameter grid into many small job files, 1 GPU each, given a
command template and which parameters stay in-job. Generation only writes
files. Submitting a job with `qsub` is a separate, manual step and is never
run by a generator itself. A generated job script activates the project
virtual environment, exports the outbound proxy variables the compute nodes
need, and calls a `cli.*` module directly, for example
`python -m cli.train_backdoor --dataset gtsrb --attack badnet_a2o
--poison-rate 0.05 --architecture vit --epochs 15 --output
checkpoints/<folder>/attack_result.pt`.
