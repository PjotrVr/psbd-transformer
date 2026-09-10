# PSBD-ViT

Research project adapting Prediction Shift Backdoor Detection (PSBD) from ConvNets to Vision Transformers (ViT-B/16 and Swin-S). The project's founding claim, that dropout placed **before** the residual add beats placing it after, is **REFUTED** ([H1](docs/hypothesis/H1-pre-beats-post.md)): measured at matched shift ratio, the pre-versus-post gap is **+0.002**, indistinguishable from noise ([H20](docs/hypothesis/H20-input-side-beats-residual-adjacent.md)). What the evidence supports instead is a search result. Where the perturbation is injected dominates what is injected (position variance 1.43x operator variance), and the split that carries the effect is input-side against residual-adjacent, **+0.054** mean AUROC with bootstrap CI [+0.031, +0.080], with both pre-residual and post-residual sitting in the losing family. The recommended deployment configuration is `token_mask` at `before_attention_norm`, which gains **+0.089** mean AUROC over the published ConvNet placement at matched clean-validation shift ratio 0.6 across the full 48-cell panel, stable at +0.081 to +0.103 across shift ratios 0.2 to 0.8, and on CIFAR-100 at 1% poisoning specifically **+0.166** (n=4). Its absolute numbers are mean AUROC 0.911, worst-case floor 0.632, zero inversions over 48/48 cells. An earlier headline of +0.258 for `gain_scale` at `mlp_norm_out` is **withdrawn**: it read the winner at shift ratio 0.95 to 0.98 and the baseline at 0.65 to 0.76, a disturbance gap the same size as the reported effect, and at matched shift ratio over the full panel that arm beats the published placement by **-0.007** ([the audit](docs/audit-2026-09-07.md)). The mechanism underneath is decision-margin estimation. The direct evidence is that not one shifted clean prediction lands on the attacker's target class, against a uniform expectation of 1%, so the neuron-bias account in the original PSBD paper is not what carries the method on ViT. H23's supporting claim is restated after the batch-coupled Gaussian was corrected and its cells re-swept: gaussian at `before_attention` scores **0.897**, not the withdrawn 0.950, which places it 6th rather than 1st but still within 0.002 of `channel_mask` and above `dropout` at `pre_residual` ([audit A19](docs/audit-2026-09-07.md)). An operator removing no capacity remains competitive with ones that do, which is the claim that refutes the capacity-removal account. Validated on CIFAR-10, CIFAR-100, GTSRB, Tiny ImageNet. Full numbers in `docs/results-report.md`, verdicts in `docs/hypothesis/README.md`.

## Style guides

3 personal guides under `.claude/styles/` override any default: `coding-style.md`
(descriptive style, shape annotations on every tensor op, comments directly above the block
they explain, a `return` never carries logic), `math-style.md` (source notation first with a
symbol table, descriptive form on request) and `writing-style.md` (no semicolons, em dashes,
arrows or hyphens as punctuation, digits, noun-phrase headers, paragraphs of more than 1
sentence). Read them before writing code, formulas or prose.

## How to run

- Environment is managed with `uv`, Python 3.11 pinned, `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`.
- Inside PBS jobs, scripts run with `python` after `source .venv/bin/activate`. Do not use `uv run` inside jobs.
- Cluster is Supek at SRCE, PBS Pro scheduler, GPU queue. Compute and login nodes need proxy exports for outbound network: `export http_proxy=http://10.150.1.1:3128` and `export https_proxy=http://10.150.1.1:3128`.
- Claude Code itself belongs on the login node or a local machine, never inside a submitted batch job (a batch job is not an interactive terminal).

## Package structure

Five packages, each with its own `__init__.py`, plus a flat root for entrypoints and the shared attack/eval pipeline core that no single package owns:

- `attacks/` — the 10 attack implementations (`badnet.py`, `blend.py`, `sig.py`, `wanet.py`, `lf.py`, `lc.py`, `bpp.py`, `adaptive_blend.py`, `tact.py`, `generated.py`) plus the registry in `__init__.py` (`ATTACK_NAMES`, `build_attack`, `default_config`).
- `analysis/` — latent-space tools: `cka.py`, `features.py`, `direction.py`, `lipschitz.py`, `embedding.py`, and the worked-example entrypoint `analyze_latent.py`.
- `defences/` — the PSBD detection mechanism: `dropout.py`, `inference.py`, `detection.py`, `checkpoint_eval.py`, `perturbations.py`, `psbd_metrics.py`, `psbd_cache.py`.
- `utils/` — `config.py`, `datasets.py`. `models.py` deliberately stays at repo root, not here (see below).
- Repo root — `train_backdoor.py`, `train_benign.py`, `train.py`, `sam.py`, `metrics.py` (entrypoints and training/eval-pipeline code whose only real callers are the two training scripts, not reusable utilities or a distinct domain package) and `poison.py`, `backdoor_data.py`, `models.py` (shared by both attacks and defences-side eval code, or loaded from nearly everywhere the same way the entrypoints are, so not owned by any one package).

`experiments/` contains hypothesis-driven experiments, each in its own subdirectory with a README explaining the question, methodology, and result. These are CPU-only analysis or GPU measurement scripts, not throwaway scratch work, and they are tracked because a number in `docs/` that cannot be traced to a commit is not a result. `scripts/` keeps only the repo-level tools that are not experiments: `detection_summary.py`, `verify_results.py`, `backfill_metadata.py`. `scratch/` is gitignored and holds smoke tests, job generators, already-run migrations, and regenerable caches, never the only source of a published number.

Import convention: within a package, sibling modules import each other with **relative** imports (`from .cka import ...`). Across a package boundary, always **absolute** (`from attacks import build_attack`, `from utils.config import DATASET_REGISTRY`, `from defences.dropout import configure_dropout`), never a relative import reaching outside its own package. `attacks/__init__.py` re-exports its registry's public names, so external call sites (`from attacks import build_attack`) don't need to know which submodule an attack lives in — a genuine registry/dispatcher pattern. `analysis/__init__.py`, `defences/__init__.py`, `utils/__init__.py` stay empty (docstring only); every caller imports the specific submodule explicitly instead.

- Use PyTorch Lightning. Seed with Lightning's `seed_everything`, not a custom `seeding.py`.
- Checkpoints from local training save to `checkpoints/`, never `backdoor_bench_checkpoints/` (that directory is BackdoorBench's own downloaded reference data, read-only, evaluated but never written to by this repo). `checkpoints/` folder names follow one canonical template, and every folder there gets an `args.json` training-provenance sidecar written alongside `attack_result.pt`; see "Checkpoint naming and metadata" below.

## Code style

- Strict functional decomposition. Prefer simple single-purpose functions over object-oriented abstractions or deep class hierarchies.
- `main()` and top-level orchestration must read linearly, like high-level pseudocode.
- Isolate side effects. I/O, networking, and state mutation go in their own dedicated functions.
- Self-documenting names. The structure explains the what and the how.
- Comments explain only the why: domain context, architectural decisions, memory or network constraints, edge cases. Never explain syntax or obvious logic.
- No decorative comment banners, no ASCII dividers, no `----- Model` style filler comments.
- In prose (comments, docstrings, notes) do not use `;`, `-` em dashes, or arrows. Use `;` only where a programming language requires it.
- Write numbers as numeric literals, not spelled out.
- Formulas: give the original form as in the source paper first, then a simplified form with descriptive names instead of Greek letters. Use LaTeX and real pseudocode blocks, not simplified prose.

## Modularity

Modular but not over-modular. Split when a function does more than one thing or when a unit needs isolated testing. Do not fragment logic into so many tiny pieces that following a single flow requires jumping across many files. When unsure, keep it in one place and split later once the seams are obvious.

## Correctness rules that must not regress

- ASR set construction at eval time uses `poison.is_eval_poisonable`/`attack_success_label`, not the training-time `is_poisonable`/`poisoned_label`: a clean-label (SIG, LC) sample is eligible when its source class is *not* already the target (the opposite of training eligibility), because eval asks whether the trigger fools a non-target image, not which images were poisoned. All-to-one and all-to-all ask the same question at both training and eval time. `poison.AttackSuccessSet` is the one class both `train_backdoor.py` and `defences.checkpoint_eval` use for this; `backdoor_data.PngPathDataset` applies the same two functions to BackdoorBench's PNG triggers when something reads from `backdoor_bench_checkpoints/` directly (`metrics.py` itself is scoped to `checkpoints/` only, see below). The `"a2a"` substring in a folder name selects all-to-all; a `sig`/`lc` attack token selects clean_label (`utils.config.label_mode_from_folder`, for folder-name-only metadata with no `args.json`).
- PSBD threshold: 2000-sample clean validation set, threshold at the 25th percentile quantile. FPR is mechanically set by the quantile choice. Clean and backdoor eval sets are paired from the same images. The two-stage pipeline is `psbd_dropout_sweep.py` (GPU, writes raw per-pass probabilities) and `psbd_analyze.py` (CPU, reads cached data, writes `psbd_metrics.json`). Detection metrics live in `defences/psbd_metrics.py`.
- SAM optimizer: two-pass update, rho = 0.1 (Zhang et al.). No feature scaling for PSBD.
- Uniform 15 epochs across all training runs, for methodological comparability.
- Clean-label attacks (SIG, LC) can only poison the target class, so their maximum poison rate is `|target class| / |train set|`: 10% on CIFAR-10, 1% on CIFAR-100, 0.5% on Tiny, and 0.56% on GTSRB at class 0 against 5.63% at class 1 or 2. `poison.choose_poison_indices` clamps silently, so every rate above the cap trains the identical index set. Clean-label GTSRB runs therefore use target class 1 with a `_tl1` folder tag, and Label-Consistent carries Turner's adversarial bases (generated by `cli/lc_bases.py`, folder tag `_adv`, declared canonical in `configs/psbd_basis.json`). Full record in `docs/clean-label-rate-caps.md`.
- `defences.dropout` never toggles a dropout the model already contains. Perturbation is injected through a position registry (`POSITION_REGISTRY`, one `PositionSpec` table per architecture) that `plug_dropout` reads to attach a FRESH perturbation module at each named submodule boundary, via forward pre-hooks and forward hooks, and `unplug_dropout` removes by handle. A trained dropout's inverted-scaling factor was calibrated against the next layer's weights, so switching it back on at inference would conflate the model's own regularization with PSBD's probe. Two positions cannot be expressed as a hook, because the tensor they perturb is a local variable that never crosses a module boundary: `after_attention_residual` (the stream between the attention add and its two consumers) and `attention_heads` (per-head outputs inside `F.multi_head_attention_forward`). Both use a removable per-instance forward wrapper instead, which mutates no weights. The single deliberate exception is `activate_model_dropout`, which exists to study that conflation directly. It switches the model's own dropouts on and skips `*.encoder.dropout` (ViT's embedding dropout, applied once before the block stack, so including it would change the depth profile of the disturbance rather than its magnitude).

## Checkpoint naming and metadata

- `checkpoints/` folder names follow one canonical template: `{architecture}_{dataset}_{attack_or_benign}[_{poison_rate_tag}][_sam_rho_{rho_tag}]`. Architecture is always explicit (`vit` or `swin`). SAM is always SAM-on-top-of-AdamW, so adam is the unmarked default and gets no optimizer tag at all; only a SAM run adds `_sam_rho_{rho_tag}`, always with an underscore before the digits (`sam_rho_0_15`, never `sam_rho0_15`). Examples: `vit_cifar100_badnet_a2o_0_01`, `swin_cifar100_benign_sam_rho_0_1`.
- Every `checkpoints/` folder has an `args.json` sidecar next to `attack_result.pt`, written by `train.save_checkpoint`'s `metadata` argument: `dataset`, `attack`, `label_mode`, `target_label`, `poison_rate`, `cover_rate`, `architecture`, `optimizer`, `rho`, `epochs`, `seed`, `max_samples`, `git_commit`, `trained_started_at`, `trained_ended_at`. `defences.checkpoint_eval.read_checkpoint_metadata` reads this to rebuild an attack's eval set in memory, without needing a `bd_test_dataset` PNG folder. `backdoor_bench_checkpoints/` folders get no `args.json`; they're evaluated through the PNG path instead (`backdoor_data.load_backdoor_splits`).
- `metrics.py` writes each checkpoint's baseline attack-success/clean-accuracy `metrics.json` into `checkpoints/<folder_name>/`, beside `attack_result.pt` and `args.json`; it creates nothing under `results/` and prunes nothing (it used to mirror and prune `results/`, and `docs/architecture.md` still describes that removed behaviour). `backdoor_bench_checkpoints/` is out of `metrics.py`'s scope entirely. `results/<folder_name>/` is written on demand by the sweep instead, which creates `results/<folder_name>/psbd/`, so orphaned result directories are inert rather than deleted. A future PSBD sweep rewrite should write `psbd_metrics.json` into the same directory, and any future defense its own `<defense>_metrics.json` (for example `strip_metrics.json`), rather than reusing the `metrics.json` name. Not named `analysis/`, which is the source package for latent-analysis tooling (TAC, CKA, PCA, UMAP, Lipschitz).

## Analysis

- Latent tools in scope: TAC, backdoor direction, CKA with debiased HSIC estimator, PCA, UMAP. Prefer UMAP over t-SNE for latent visualization.
- Attacks in scope: BadNet A2O/A2A, Blend, SIG, WaNet, LF, LC, BPP, Adaptive-Blend, TaCT.

## External libraries

BackdoorBench and Backdoor-Toolbox are reference implementations. When porting an attack or metric, read their source directly and match semantics, but keep the ported code in project style. Do not add them as hard runtime dependencies without asking.

## Before finishing a task

Run the relevant eval or a quick sanity check rather than assuming correctness.
If all checks satisfy, run `ruff format .` to format all code.