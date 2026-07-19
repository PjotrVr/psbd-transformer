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
- Checkpoints save to `checkpoints/`, not `backdoor_bench_checkpoints/`. Each checkpoint gets a `metrics.json` written alongside it.

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

- ASR set construction: exclude samples whose source class already equals the intended target. Under all-to-one this drops the whole target class (default target 0). Under all-to-all the intended label is `(source + 1) % num_classes`, which never equals source, so nothing is dropped. The `"a2a"` substring in a folder name selects all-to-all.
- Do not use `poison.FullyPoisonedTestSet` with `is_poisonable` for clean-label attacks (SIG, LC) at test time. `eval_backdoor` builds its own `AttackSuccessSet`.
- PSBD threshold: 2000-sample clean validation set, threshold at the 25th percentile quantile. FPR is mechanically set by the quantile choice. Clean and backdoor eval sets are paired from the same images.
- SAM optimizer: two-pass update, rho = 0.1 (Zhang et al.). No feature scaling for PSBD.
- Uniform 15 epochs across all training runs, for methodological comparability.
- `configure_pre_residual_dropout` skips any module named `*.encoder.dropout` (ViT's embedding dropout, applied once before the block stack, not a pre-residual placement), so it touches the 36 true per-block dropouts, not all 37.

## Analysis

- Latent tools in scope: TAC, backdoor direction, CKA with debiased HSIC estimator, PCA, UMAP. Prefer UMAP over t-SNE for latent visualization.
- Attacks in scope: BadNet A2O/A2A, Blend, SIG, WaNet, LF, LC, BPP, Adaptive-Blend, TaCT.

## External libraries

BackdoorBench and Backdoor-Toolbox are reference implementations. When porting an attack or metric, read their source directly and match semantics, but keep the ported code in project style. Do not add them as hard runtime dependencies without asking.

## Before finishing a task

Run the relevant eval or a quick sanity check rather than assuming correctness.