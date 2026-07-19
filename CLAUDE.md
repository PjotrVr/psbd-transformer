# PSBD-ViT

Research project adapting Prediction Shift Backdoor Detection (PSBD) from ConvNets to Vision Transformers (ViT-B/16 and Swin). Core contribution: dropout placed **before** the residual add (pre-residual), motivated by the CKA homogeneity result showing ViT residual streams are persistent (Raghu et al.). Validated on CIFAR-10, CIFAR-100, GTSRB, Tiny ImageNet.

## How to run

- Environment is managed with `uv`, Python 3.11 pinned, `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`.
- Inside PBS jobs, scripts run with `python` after `source .venv/bin/activate`. Do not use `uv run` inside jobs.
- Cluster is Supek at SRCE, PBS Pro scheduler, GPU queue. Compute and login nodes need proxy exports for outbound network: `export http_proxy=http://10.150.1.1:3128` and `export https_proxy=http://10.150.1.1:3128`.
- Claude Code itself belongs on the login node or a local machine, never inside a submitted batch job (a batch job is not an interactive terminal).

## Package structure

- Flat package `psbd/` with no nested subpackages.
- Flat imports only, for example `from config import ...`, never relative imports.
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

## Scratch workflow

- Throwaway experiments live in `scratch/` (git-ignored). Move fast there, no style or structure requirements.
- Promote a script into `.` only after it is validated and worth keeping. Promotion means: rewrite to project conventions, add the isolated side-effect boundaries, and confirm it runs end to end.
- Do not integrate unvalidated scratch code into the package.

## Correctness rules that must not regress

- ASR set construction at eval time uses `poison.is_eval_poisonable`/`attack_success_label`, not the training-time `is_poisonable`/`poisoned_label`: a clean-label (SIG, LC) sample is eligible when its source class is *not* already the target (the opposite of training eligibility), because eval asks whether the trigger fools a non-target image, not which images were poisoned. All-to-one and all-to-all ask the same question at both training and eval time. `poison.AttackSuccessSet` is the one class both `train_backdoor.py` and `checkpoint_eval.py` use for this; `backdoor_data.PngPathDataset` applies the same two functions to BackdoorBench's PNG triggers when something reads from `backdoor_bench_checkpoints/` directly (`metrics.py` itself is scoped to `checkpoints/` only, see below). The `"a2a"` substring in a folder name selects all-to-all; a `sig`/`lc` attack token selects clean_label (`config.label_mode_from_folder`, for folder-name-only metadata with no `args.json`).
- PSBD threshold: 2000-sample clean validation set, threshold at the 25th percentile quantile. FPR is mechanically set by the quantile choice. Clean and backdoor eval sets are paired from the same images. This mechanism (the dropout-rate sweep, PSU shift, threshold, TPR/FPR/AUROC) is currently archived in `_archive/sweep.py` and `_archive/run_sweep.py` pending a rewrite, so it is not live code right now; the rule stands for whatever replaces it.
- SAM optimizer: two-pass update, rho = 0.1 (Zhang et al.). No feature scaling for PSBD.
- Uniform 15 epochs across all training runs, for methodological comparability.
- `configure_pre_residual_dropout` skips any module named `*.encoder.dropout` (ViT's embedding dropout, applied once before the block stack, not a pre-residual placement), so it touches the 36 true per-block dropouts, not all 37.

## Checkpoint naming and metadata

- `checkpoints/` folder names follow one canonical template: `{architecture}_{dataset}_{attack_or_benign}[_{poison_rate_tag}][_sam_rho_{rho_tag}]`. Architecture is always explicit (`vit` or `swin`). SAM is always SAM-on-top-of-AdamW, so adam is the unmarked default and gets no optimizer tag at all; only a SAM run adds `_sam_rho_{rho_tag}`, always with an underscore before the digits (`sam_rho_0_15`, never `sam_rho0_15`). Examples: `vit_cifar100_badnet_a2o_0_01`, `swin_cifar100_benign_sam_rho_0_1`. `normalize_checkpoints.py` is the one-time migration that produced this; it infers architecture from the checkpoint's own state_dict keys (`models.detect_architecture`) when a folder name gives no hint.
- Every `checkpoints/` folder has an `args.json` sidecar next to `attack_result.pt`, written by `train.save_checkpoint`'s `metadata` argument: `dataset`, `attack`, `label_mode`, `target_label`, `poison_rate`, `cover_rate`, `architecture`, `optimizer`, `rho`, `epochs`, `seed`, `git_commit`, `trained_started_at`, `trained_ended_at`. `checkpoint_eval.read_checkpoint_metadata` reads this to rebuild an attack's eval set in memory, without needing a `bd_test_dataset` PNG folder. `backdoor_bench_checkpoints/` folders get no `args.json`; they're evaluated through the PNG path instead (`backdoor_data.load_backdoor_splits`).
- `results/<folder_name>/` mirrors `checkpoints/` 1:1 (`metrics.py` creates and prunes it to match exactly; `backdoor_bench_checkpoints/` is out of `metrics.py`'s scope entirely). `results/<folder_name>/metrics.json` is baseline attack-success/clean-accuracy output (`metrics.py`), a different file from the old, now-archived per-quantile PSBD sweep output. A future PSBD sweep rewrite should write `psbd_metrics.json` into the same directory, and any future defense its own `<defense>_metrics.json` (for example `strip_metrics.json`), rather than reusing the `metrics.json` name. Not named `analysis/`, which is the source package for latent-analysis tooling (TAC, CKA, PCA, UMAP, Lipschitz).

## Analysis

- Latent tools in scope: TAC, backdoor direction, CKA with debiased HSIC estimator, PCA, UMAP. Prefer UMAP over t-SNE for latent visualization.
- Attacks in scope: BadNet A2O/A2A, Blend, SIG, WaNet, LF, LC, BPP, Adaptive-Blend, TaCT.

## External libraries

BackdoorBench and Backdoor-Toolbox are reference implementations. When porting an attack or metric, read their source directly and match semantics, but keep the ported code in project style. Do not add them as hard runtime dependencies without asking.

## Before finishing a task

Run the relevant eval or a quick sanity check rather than assuming correctness.