# PSBD-ViT

Research project adapting Prediction Shift Backdoor Detection (PSBD) from ConvNets to Vision Transformers (ViT-B/16 and Swin-S). The project's founding claim, that dropout placed **before** the residual add beats placing it after, is **refuted** ([H1](docs/hypothesis/H1-pre-beats-post.md)): measured at matched shift ratio, the pre-versus-post gap is **+0.002**, indistinguishable from noise ([H20](docs/hypothesis/H20-input-side-beats-residual-adjacent.md)). What the evidence supports instead is a search result. Where the perturbation is injected dominates what is injected (position variance 1.43x operator variance), and the split that carries the effect is input-side against residual-adjacent, **+0.054** mean AUROC with bootstrap CI [+0.031, +0.080], with both pre-residual and post-residual sitting in the losing family. The recommended deployment configuration is `token_mask` at `before_attention_norm`, which gains **+0.089** mean AUROC over the published ConvNet placement at matched clean-validation shift ratio 0.6 across the full 48-cell panel, stable at +0.081 to +0.103 across shift ratios 0.2 to 0.8, and on CIFAR-100 at 1% poisoning specifically **+0.166** (n=4). Its absolute numbers are mean AUROC 0.911, worst-case floor 0.632, 0 inversions over 48/48 cells. An earlier headline of +0.258 for `gain_scale` at `mlp_norm_out` is **withdrawn**: it read the winner at shift ratio 0.95 to 0.98 and the baseline at 0.65 to 0.76, a disturbance gap the same size as the reported effect, and at matched shift ratio over the full panel that arm beats the published placement by **-0.007** ([the audit](docs/audit-2026-09-07.md)). The mechanism underneath is decision-margin estimation. The direct evidence is that no shifted clean prediction lands on the attacker's target class, against a uniform expectation of 1%, so the neuron-bias account in the original PSBD paper is not what carries the method on ViT. H23's supporting claim is restated after the batch-coupled Gaussian was corrected and its cells re-swept: gaussian at `before_attention` scores **0.897**, not the withdrawn 0.950, which places it 6th rather than 1st but still within 0.002 of `channel_mask` and above `dropout` at `pre_residual` ([audit A19](docs/audit-2026-09-07.md)). An operator removing no capacity remains competitive with ones that do, which is the claim that refutes the capacity-removal account. Validated on CIFAR-10, CIFAR-100, GTSRB and Tiny ImageNet. Full numbers in `docs/results-report.md`, verdicts in `docs/hypothesis/README.md`.

## Style guides

3 personal guides under `.claude/styles/` override any default: `coding-style.md`
(descriptive style, shape annotations on every tensor op, comments directly above the block
they explain, a `return` never carries logic), `math-style.md` (source notation first with a
symbol table, descriptive form on request) and `writing-style.md` (no semicolons, em dashes,
arrows or hyphens as punctuation, digits, noun-phrase headers, paragraphs of more than 1
sentence). Read them before writing code, formulas or prose. Library packages and `cli/` are
production code under the coding guide. `experiments/`, `notebooks/` and `scratch/` are
research code under its research rules.

## Running the code

- Environment is managed with `uv`, Python 3.11 pinned, `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`.
- Inside PBS jobs, scripts run with `python` after `source .venv/bin/activate`. Do not use `uv run` inside jobs. Every command is a module: `python -m cli.<name>`.
- Cluster is Supek at SRCE, PBS Pro scheduler, GPU queue. Compute and login nodes need proxy exports for outbound network: `export http_proxy=http://10.150.1.1:3128` and `export https_proxy=http://10.150.1.1:3128`.
- Claude Code itself belongs on the login node or a local machine, never inside a submitted batch job (a batch job is not an interactive terminal).

## Package layout

9 packages plus `cli/`, each named for its subject. Relative imports inside a package, absolute imports across packages, and no package re-exports its submodules except `attacks/__init__.py`, which is a registry (`ATTACK_NAMES`, `build_attack`, `default_config`).

- `attacks/`: the 10 attacks (`badnet`, `blend`, `sig`, `wanet`, `lf`, `lc`, `bpp`, `adaptive_blend`, `tact`, `generated`), shared trigger `patterns`, `poisoning` (training and eval eligibility rules, index selection, `AttackSuccessSet`), `adversarial` and `bases` (Label-Consistent's adversarial bases) and `evasion`.
- `data/`: `registry` (`DATASET_REGISTRY`, normalisation statistics, `label_mode_from_folder`), `loading`, `splits` (the PSBD split and `read_checkpoint_metadata`) and `backdoorbench` (the PNG path for `backdoor_bench_checkpoints/`).
- `models/`: `backbones` (the 2 architectures, `load_checkpoint`, `network_core`) and `positions` (`POSITION_REGISTRY`, `plug_dropout`, `unplug_dropout`, `activate_model_dropout`).
- `training/`: `loop` (the training loop, `save_checkpoint` and provenance) and `sam`.
- `defences/`: PSBD itself. `operators` (what is injected), `inference` (forward passes, `forward_logits`, `frozen_parameters`), `scores` (PSU and the shift ratio), `decision` (rate rules, quantile thresholds, `detection_report`, the canonical placements) and `cache` (the stage-1 tensors under `results/<folder>/psbd/`).
- `detectors/`: the competitor input detectors behind 1 registry (`DETECTOR_NAMES`, `build_detector`, `DetectorContext`), 1 module per method, `records` for their on-disk layout under `results/<folder>/detectors/`.
- `analysis/`: latent-space tools (`features` with `captured_layers`, `cka`, `direction`, `embedding`, `lipschitz`, `distribution`, `stealth`, `cases`, `latent`).
- `evaluation/`: `metrics` (attack success and clean accuracy), `loaders` and `summary`.
- `utils/`: `numerics` only, helpers with no subject of their own.
- `cli/`: 1 module per command, the only place a `main()` lives. Training: `train_backdoor`, `train_benign`. PSBD: `sweep` (GPU, writes raw per-pass probabilities under `results/<folder>/psbd/`), `analyze` (CPU, writes `psbd_metrics.json`), `summary`, `tables`, `report`, `operating_points`, `variants`. Competitors: `baselines`, `compare_detectors`, `fuse_detectors`. Others: `evaluate`, `backfill`, `lc_bases`, `head_profile`, `analyze_latent`.

`experiments/` holds hypothesis-driven experiments, each in its own directory with a README stating the question, the method and the result. They are tracked because a number in `docs/` that cannot be traced to a commit is not a result. `scripts/` holds repo-level tooling: the coverage ledger, the table generators, the prose audit and the prose-only guard. `pbs/` holds job generators and the job scripts they emit. `notebooks/` is the guided tour. `scratch/` is untracked and never the only source of a published number. `third_party/` holds untracked clones of reference implementations at pinned commits.

- Use PyTorch Lightning. Seed with Lightning's `seed_everything`, not a custom `seeding.py`.
- Checkpoints from local training save to `checkpoints/`, never `backdoor_bench_checkpoints/` (BackdoorBench's own downloaded reference data, read-only, evaluated but never written to). `checkpoints/` folder names follow 1 canonical template, and every folder there gets an `args.json` training-provenance sidecar written alongside `attack_result.pt`. See "Checkpoint naming and metadata" below.

## Code style

- Strict functional decomposition. Prefer simple single-purpose functions over object-oriented abstractions or deep class hierarchies.
- `main()` and top-level orchestration must read linearly, like high-level pseudocode.
- Isolate side effects. I/O, networking and state mutation go in their own dedicated functions.
- Self-documenting names. The structure explains the what and the how.
- Comments explain only the why: domain context, architectural decisions, memory or network constraints, edge cases. Never explain syntax or obvious logic.
- No decorative comment banners, no ASCII dividers, no `----- Model` style filler comments.
- In prose (comments, docstrings, notes) do not use `;`, em dashes, arrows or the Oxford comma. Use `;` only where a programming language requires it.
- Numbers as digits, never spelled out.
- Formulas: the source paper's original form first, with a symbol table naming every symbol, as `.claude/styles/math-style.md` asks. A descriptive form with named quantities only where a reader asked for it. Existing docstrings that carry both forms stay as they are. Use LaTeX or real pseudocode blocks, never simplified prose.
- `scripts/prose_audit.py` counts violations of these rules and must report 0 hits on every file a commit touches. `scripts/check_prose_only.py HEAD` proves a docs commit changed no code.

## Modularity

Modular but not over-modular. Split when a function does more than 1 thing or when a unit needs isolated testing. Do not fragment logic into so many tiny pieces that following a single flow requires jumping across many files. When unsure, keep it in 1 place and split later once the seams are obvious.

## Canon

The constants every table reads live in `defences/decision.py` and are held to `configs/psbd_basis.json` by `tests/test_canon.py`. The deployable rate rule is `select_rate_adaptively` at `ADAPTIVE_SHIFT_TARGET` 0.8, the smallest rate whose clean-validation shift ratio reaches the target. The cross-placement comparison device is `select_rate_at_matched_shift` at `PLACEMENT_MATCH_TARGET` 0.6. `RECOMMENDED_PLACEMENT` is `before_attention_norm_token_mask`, `PUBLISHED_PLACEMENT` is `post_residual` (dropout after the residual add, the ConvNet placement). The decision rule is one-sided, low score means poisoned, and `auroc_two_sided` is a diagnostic field with no live consumer. Fractional PSU (`detection_psu_ratio`) is the headline statistic, absolute PSU is reported beside it. Any other shift target, placement default or decision rule in code or docs is drift to be fixed, not a second convention.

## Correctness rules that must not regress

- ASR set construction at eval time uses `attacks.poisoning.is_eval_poisonable` and `attack_success_label`, not the training-time `is_poisonable` and `poisoned_label`: a clean-label (SIG, LC) sample is eligible when its source class is *not* already the target (the opposite of training eligibility), because eval asks whether the trigger fools a non-target image, not which images were poisoned. All-to-one and all-to-all ask the same question at both training and eval time. `attacks.poisoning.AttackSuccessSet` is the single class both `cli.train_backdoor` and `data.splits` use for this. `data.backdoorbench.PngPathDataset` applies the same 2 functions to BackdoorBench's PNG triggers when something reads from `backdoor_bench_checkpoints/` directly. The `"a2a"` substring in a folder name selects all-to-all. A `sig` or `lc` attack token selects clean_label (`data.registry.label_mode_from_folder`, for folder-name-only metadata with no `args.json`). TaCT's `source_classes` travel on the `Attack` record, so `AttackSuccessSet` restricts the eval set to the source classes itself.
- PSBD threshold: 2000-sample clean validation set (`data.splits.PSBD_HELDOUT_SIZE`, seed `PSBD_SPLIT_SEED`), threshold at a quantile of clean validation scores (`defences.decision.PSBD_QUANTILES`, headline 0.25). FPR is mechanically set by the quantile choice. Clean and backdoor eval sets are paired from the same images (`pair_clean_to_backdoor`). The 2-stage pipeline is `cli.sweep` (GPU, writes raw per-pass probabilities under `results/<folder>/psbd/`) and `cli.analyze` (CPU, reads the cache, writes `psbd_metrics.json`). Detection metrics live in `defences.scores` and `defences.decision`.
- Competitor detectors run through `cli.baselines` on exactly the PSBD splits and quantiles, return low for poisoned with at most 1 negation at their own scoring boundary, and write 1 record per detector under `results/<folder>/detectors/`. A detector that fits per-sample statistics on the validation split returns out-of-fit scores for it (`detectors.CROSS_FITTED`). Every port carries a numerical cross-check against the reference implementation in `third_party/`.
- SAM optimizer: 2-pass update, rho = 0.1 (Zhang et al.). No feature scaling for PSBD.
- Uniform 15 epochs across all training runs, for methodological comparability.
- Clean-label attacks (SIG, LC) can only poison the target class, so their maximum poison rate is `|target class| / |train set|`: 10% on CIFAR-10, 1% on CIFAR-100, 0.5% on Tiny and 0.56% on GTSRB at class 0 against 5.63% at class 1 or 2. `attacks.poisoning.choose_poison_indices` clamps silently, so every rate above the cap trains the identical index set. Clean-label GTSRB runs therefore use target class 1 with a `_tl1` folder tag, and Label-Consistent carries Turner's adversarial bases (generated by `cli.lc_bases`, folder tag `_adv` and declared canonical in `configs/psbd_basis.json`). Full record in `docs/clean-label-rate-caps.md`.
- `models.positions` never toggles a dropout the model already contains. Perturbation is injected through a position registry (`POSITION_REGISTRY`, 1 `PositionSpec` table per architecture) that `plug_dropout` reads to attach a FRESH perturbation module at each named submodule boundary, via forward pre-hooks and forward hooks, and `unplug_dropout` removes by handle. A trained dropout's inverted-scaling factor was calibrated against the next layer's weights, so switching it back on at inference would conflate the model's own regularization with PSBD's probe. 2 positions cannot be expressed as a hook, because the tensor they perturb is a local variable that never crosses a module boundary: `after_attention_residual` (the stream between the attention add and its 2 consumers) and `attention_heads` (per-head outputs inside `F.multi_head_attention_forward`). Both use a removable per-instance forward wrapper instead, which mutates no weights. The single deliberate exception is `activate_model_dropout`, which exists to study that conflation directly. It switches the model's own dropouts on and skips `*.encoder.dropout` (ViT's embedding dropout, applied once before the block stack, so including it would change the depth profile of the disturbance rather than its magnitude).

## Checkpoint naming and metadata

- `checkpoints/` folder names follow 1 canonical template: `{architecture}_{dataset}_{attack_or_benign}[_{poison_rate_tag}][_sam_rho_{rho_tag}][_seed_{seed}]`. Architecture is always explicit (`vit` or `swin`). SAM is always SAM-on-top-of-AdamW, so adam is the unmarked default and gets no optimizer tag at all. Only a SAM run adds `_sam_rho_{rho_tag}`, always with an underscore before the digits (`sam_rho_0_15`, never `sam_rho0_15`). Seed 0 is unmarked, a replicate carries `_seed_{N}`. Examples: `vit_cifar100_badnet_a2o_0_01`, `vit_cifar100_badnet_a2o_0_01_seed_1`, `swin_cifar100_benign_sam_rho_0_1`.
- Every `checkpoints/` folder has an `args.json` sidecar next to `attack_result.pt`, written by `training.loop.save_checkpoint`'s `metadata` argument: `dataset`, `attack`, `label_mode`, `target_label`, `poison_rate`, `cover_rate`, `architecture`, `optimizer`, `rho`, `epochs`, `seed`, `max_samples`, `git_commit`, `trained_started_at` and `trained_ended_at`. `data.splits.read_checkpoint_metadata` reads this to rebuild an attack's eval set in memory, without needing a `bd_test_dataset` PNG folder. `backdoor_bench_checkpoints/` folders get no `args.json`. They are evaluated through the PNG path instead (`data.backdoorbench.load_backdoor_splits`).
- `cli.evaluate` writes each checkpoint's attack-success and clean-accuracy `metrics.json` into `checkpoints/<folder_name>/`, beside `attack_result.pt` and `args.json`. It creates nothing under `results/` and prunes nothing. `results/<folder_name>/` is written on demand: `cli.sweep` creates `results/<folder_name>/psbd/`, `cli.baselines` creates `results/<folder_name>/detectors/`. Each future defence writes its own `<defence>_metrics.json`, never a file named `metrics.json`.

## Analysis

- Latent tools in scope: TAC, backdoor direction, CKA with debiased HSIC estimator, PCA and UMAP. Prefer UMAP over t-SNE for latent visualization.
- Attacks in scope: BadNet A2O/A2A, Blend, SIG, WaNet, LF, LC, BPP, Adaptive-Blend, TaCT.

## External libraries

BackdoorBench, BackdoorBox, backdoor-toolbox and each method's official repository are reference implementations, cloned untracked under `third_party/` at pinned commits. When porting an attack or a detector, read their source directly, match semantics, keep the ported code in project style, record every deviation in the module's docstring and in `docs/detectors/<name>.md`, and add a numerical cross-check test against the reference. Do not add them as runtime dependencies without asking.

## Checks before finishing

Run the relevant eval or a quick sanity check rather than assuming correctness. Run the test suite, `ruff format` and `ruff check` on the files you changed (never a repo-wide import sort), and `scripts/prose_audit.py` on them.
