# PSBD-ViT

Research project adapting Prediction Shift Backdoor Detection (PSBD) from ConvNets to Vision Transformers (ViT-B/16 and Swin). Core contribution: dropout placed **before** the residual add (pre-residual), motivated by the CKA homogeneity result showing ViT residual streams are persistent (Raghu et al.). Validated on CIFAR-10, CIFAR-100, GTSRB, Tiny ImageNet.

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

`scripts/` contains hypothesis-driven experiments, each in its own subdirectory with a README explaining the question, methodology, and result. These are CPU-only analysis or GPU measurement scripts, not throwaway scratch work.

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
- `defences.dropout.configure_pre_residual_dropout` skips any module named `*.encoder.dropout` (ViT's embedding dropout, applied once before the block stack, not a pre-residual placement), so it touches the 36 true per-block dropouts, not all 37.

## Checkpoint naming and metadata

- `checkpoints/` folder names follow one canonical template: `{architecture}_{dataset}_{attack_or_benign}[_{poison_rate_tag}][_sam_rho_{rho_tag}]`. Architecture is always explicit (`vit` or `swin`). SAM is always SAM-on-top-of-AdamW, so adam is the unmarked default and gets no optimizer tag at all; only a SAM run adds `_sam_rho_{rho_tag}`, always with an underscore before the digits (`sam_rho_0_15`, never `sam_rho0_15`). Examples: `vit_cifar100_badnet_a2o_0_01`, `swin_cifar100_benign_sam_rho_0_1`.
- Every `checkpoints/` folder has an `args.json` sidecar next to `attack_result.pt`, written by `train.save_checkpoint`'s `metadata` argument: `dataset`, `attack`, `label_mode`, `target_label`, `poison_rate`, `cover_rate`, `architecture`, `optimizer`, `rho`, `epochs`, `seed`, `max_samples`, `git_commit`, `trained_started_at`, `trained_ended_at`. `defences.checkpoint_eval.read_checkpoint_metadata` reads this to rebuild an attack's eval set in memory, without needing a `bd_test_dataset` PNG folder. `backdoor_bench_checkpoints/` folders get no `args.json`; they're evaluated through the PNG path instead (`backdoor_data.load_backdoor_splits`).
- `results/<folder_name>/` mirrors `checkpoints/` 1:1 (`metrics.py` creates and prunes it to match exactly; `backdoor_bench_checkpoints/` is out of `metrics.py`'s scope entirely). `results/<folder_name>/metrics.json` is baseline attack-success/clean-accuracy output (`metrics.py`), a different file from the old, now-archived per-quantile PSBD sweep output. A future PSBD sweep rewrite should write `psbd_metrics.json` into the same directory, and any future defense its own `<defense>_metrics.json` (for example `strip_metrics.json`), rather than reusing the `metrics.json` name. Not named `analysis/`, which is the source package for latent-analysis tooling (TAC, CKA, PCA, UMAP, Lipschitz).

## Analysis

- Latent tools in scope: TAC, backdoor direction, CKA with debiased HSIC estimator, PCA, UMAP. Prefer UMAP over t-SNE for latent visualization.
- Attacks in scope: BadNet A2O/A2A, Blend, SIG, WaNet, LF, LC, BPP, Adaptive-Blend, TaCT.

## External libraries

BackdoorBench and Backdoor-Toolbox are reference implementations. When porting an attack or metric, read their source directly and match semantics, but keep the ported code in project style. Do not add them as hard runtime dependencies without asking.

## Before finishing a task

Run the relevant eval or a quick sanity check rather than assuming correctness.
If all checks satisfy, run `ruff format .` to format all code.