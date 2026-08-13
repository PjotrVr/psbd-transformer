# PSBD-ViT: repo reorganization (TASKS.md)

## Context

The first cleanup pass (`docs/plans/audit-fixes.md`) fixed correctness bugs and normalized checkpoint metadata, but left the codebase flat: 39 `.py` files at repo root spanning attacks, latent-space analysis, the PSBD detection mechanism, and generic utilities, with no folder structure to signal which is which. `TASKS.md` asks for that structure — `attacks/`, `analysis/`, `defences/`, `utils/`, `plotting/`, `notebooks/`, reorganized `logs/`/`pbs/`, dead code moved out of the way, tests confirming nothing broke, and a comprehensive `docs/architecture.md` at the end. This plan is scoped to produce that reorganization, ending in a working, fully-tested tree with each subtask landing as its own commit. It does **not** touch attack/defence logic itself, only where files live and how they import each other. Implementation is a separate, later step — this plan is written, not executed, in this pass.

This reorganization is a real convention change: `CLAUDE.md` currently states "Flat package with no nested subpackages... never relative imports." That rule is being replaced, and `CLAUDE.md` gets rewritten as part of this plan to say what replaces it (see §10).

## Final layout

```
attacks/       badnet.py, blend.py, sig.py, wanet.py, lf.py, lc.py, bpp.py,
               adaptive_blend.py, tact.py, generated.py, __init__.py (the
               registry: ATTACK_NAMES, build_attack, default_config — what
               attacks.py already is today, relocated and renamed)
analysis/      cka.py, features.py, direction.py, lipschitz.py, embedding.py,
               analyze_latent.py, __init__.py
defences/      dropout.py, inference.py, detection.py, checkpoint_eval.py,
               __init__.py
utils/         config.py, datasets.py, models.py, __init__.py
plotting/      __init__.py only for now — a placeholder home so future
               plotting code has somewhere to go that isn't analysis/ or root
scratch/       normalize_checkpoints.py, download.py (one-off scripts that
               already did their one job; gitignored, not shipped code)
notebooks/     psbd_analysis.ipynb, backdoor_train_dev.ipynb, sam_dev.ipynb,
               psbd_sweep.ipynb
_archive/      + analyze.py, plotting.py, experiment_io.py (added to what's
               already there from the prior pass)
repo root      train_backdoor.py, train_benign.py, train.py, sam.py,
               metrics.py, poison.py, backdoor_data.py — entrypoints and the
               shared attack/eval pipeline core, deliberately not folded into
               any package (see decisions below)
```

Note: `_archive/_attacks/`, `_archive/_defences/`, `_archive/_models/` already exist from the prior cleanup pass (unrelated legacy code, permanently archived) — distinct from the new `attacks/`, `defences/`, and `utils/` (which absorbs `models.py`) being created now. Don't confuse the two when working in this tree.

## Decisions already made (don't re-litigate these)

- `poison.py` and `backdoor_data.py` stay at repo root — shared by both attacks and defences-side eval code, not owned by either package.
- `sam.py`, `train.py`, `metrics.py` stay at repo root — training/eval-pipeline code whose only real callers are `train_backdoor.py`/`train_benign.py`, not reusable utilities or a distinct domain package.
- `analyze.py`, `plotting.py`, `experiment_io.py` move to `_archive/` together, not into `plotting/`. They form one working chain, but all three exist only to read/render the old `experiments/` sweep format that nothing produces anymore since `sweep.py`/`run_sweep.py` were archived in the prior pass. Archiving is consistent with that earlier decision.
- `lipschitz.py` moves into `analysis/` even though nothing currently calls it — it's correct, in-scope code (CLAUDE.md already lists Lipschitz analysis), just not yet wired into `analyze_latent.py`.
- `normalize_checkpoints.py` and `download.py` move to `scratch/`. Both are one-off scripts that already did their job (all 557 `checkpoints/` folders are canonical and carry `args.json`; `raw_data/` is already downloaded) — keeping them at repo root as if they were maintained, reusable code overstates what they are. `scratch/` keeps them available for reference (re-running `normalize_checkpoints.py` is still safe/idempotent if it's ever needed again) without implying they're part of the live pipeline.

## Import convention for the new packages

- `attacks/__init__.py` uses **relative** imports to gather its own 10 sibling modules (`from .badnet import ...`, etc.) and re-exports `ATTACK_NAMES`, `build_attack`, `default_config` — this is a genuine registry/dispatcher pattern, and re-exporting means external call sites (`from attacks import build_attack`) don't change at all, only `attacks.py`'s internals move.
- `analysis/__init__.py`, `defences/__init__.py`, `utils/__init__.py` stay empty (docstring only) — no re-exporting. External code imports the specific submodule explicitly: `from defences.dropout import configure_dropout`, `from utils.models import build_vit`, `from analysis.cka import cka_debiased`. This keeps call sites greppable and matches the project's existing preference for explicit, unambiguous imports; a registry re-export only makes sense for `attacks/`, where "give me the attack by name" is the actual interface external code wants.
- Within a package, sibling modules import each other with relative imports (e.g. `analysis/analyze_latent.py` importing `analysis/cka.py` as `from .cka import ...`). Across package boundaries, always absolute (`from attacks import build_attack`, `from utils.config import DATASET_REGISTRY`), never a relative import reaching outside its own package.

## Subtasks, in commit order

Each lands as its own commit once its own verification passes — don't batch multiple subtasks into one commit.

### 1. `utils/` — config.py, datasets.py, models.py
Move the three files in, add `utils/__init__.py`. Update every importer across the repo (attacks, defences, analysis, entrypoints, tests all currently do `from config import ...` / `from datasets import ...` / `from models import ...`) to `from utils.config import ...` / `from utils.datasets import ...` / `from utils.models import ...`. This is the widest-blast-radius single move (these three are imported by nearly every other file) — doing it first means every later subtask's import updates only ever reference the new `utils.*` paths, not a moving target.
**Verify:** `pytest tests/` passes; every root `.py` file still imports cleanly (`python -c "import <module>"` loop).

### 2. `attacks/` — the 10 attack files + the registry
`git mv attack_badnet.py attacks/badnet.py` (and so on for all 10), `git mv attacks.py attacks/__init__.py`. Inside the new `__init__.py`, change its 10 `import attack_X` lines to relative `from . import X` (or `from .X import ...`, whichever matches how it currently consumes each module — check the current `attacks.py` body rather than assuming). Each attack file's own `from poison import Attack` stays as-is (absolute, `poison.py` didn't move). Update the handful of external importers (`train_backdoor.py`, `checkpoint_eval.py`, `metrics.py`, `normalize_checkpoints.py` before it moves to scratch/, `analyze_latent.py`, tests) — likely no change needed for most of them, since `from attacks import build_attack, default_config, ATTACK_NAMES` is unchanged by the re-export; only files that did `from attack_badnet import BadNetConfig` directly (check whether any do) need updating.
**Verify:** `pytest tests/` passes (`tests/test_attacks.py` exercises every attack by name via `build_attack`, so this is a strong signal); `python -c "from attacks import ATTACK_NAMES, build_attack, default_config"` still works unchanged.

### 3. `defences/` — dropout.py, inference.py, detection.py, checkpoint_eval.py
Move all four, add `defences/__init__.py`. Update importers: `detection.py` itself imports `inference.forward_probs` (becomes relative `from .inference import forward_probs`); external callers (`train.py`, `train_backdoor.py`, `train_benign.py`, `metrics.py`, `analysis/features.py`) change `from detection import ...` → `from defences.detection import ...`, `from dropout import ...` → `from defences.dropout import ...`, `from checkpoint_eval import ...` → `from defences.checkpoint_eval import ...`.
**Verify:** `pytest tests/` passes (`tests/test_dropout.py` and `tests/test_checkpoint_metadata.py` directly exercise this package); one real `train_backdoor.py` invocation end to end (small config, 1 epoch) to confirm the training path's `defences.detection` calls work outside the test suite too.

### 4. `analysis/` — cka.py, features.py, direction.py, lipschitz.py, embedding.py, analyze_latent.py
Move all six, add `analysis/__init__.py`. `analyze_latent.py`'s imports of its five siblings become relative; its imports of `attacks`, `utils.config`, `utils.datasets`, `defences.dropout`, `utils.models`, `poison` become the updated absolute paths from subtasks 1–3. `features.py`'s `from inference import forward_probs, from models import vit_core` become `from defences.inference import forward_probs`, `from utils.models import vit_core`.
**Verify:** `python -c "import analysis.analyze_latent"` succeeds; if a GPU and a real checkpoint are available, one real `analyze_latent.py` run against an existing `checkpoints/` folder, otherwise a clean import is the bar (no other file depends on this package, so there's no test suite coverage for it beyond import success — say so rather than overstating verification).

### 5. `plotting/` scaffold + archive the legacy chain
Create `plotting/__init__.py` (docstring only, e.g. "Figure-rendering code lives here, not in analysis/ or at repo root — currently empty pending the PSBD sweep rewrite"). Separately, `git mv analyze.py plotting.py experiment_io.py _archive/` (these are being archived, not moved into the new `plotting/` — see the decision above for why).
**Verify:** `git status` shows the three files moved into `_archive/`; confirm nothing outside `_archive/` still imports any of them (`grep -rln "import experiment_io\|import plotting\b\|import analyze\b"` excluding `_archive/`).

### 6. `scratch/` — normalize_checkpoints.py, download.py
`mv normalize_checkpoints.py download.py scratch/` (plain `mv`, not `git mv` — `scratch/` is gitignored, so this is the same untracking pattern used for the legacy directories in the prior cleanup pass; the git history for `normalize_checkpoints.py` up to this point stays recoverable via `git log -- normalize_checkpoints.py` even after it leaves the tracked tree). Update `tests/test_checkpoint_metadata.py`'s `from normalize_checkpoints import parse_folder_name` — either move those specific test cases into `scratch/` alongside the script (informal, not run by CI) or keep them in `tests/` importing from the new `scratch.normalize_checkpoints` path; decide based on whether `scratch/` being gitignored makes a tracked test depending on it a problem (it does — a fresh clone wouldn't have `scratch/` populated by git at all, so a tracked test importing from it would fail for anyone who didn't manually run the migration once). Recommend: keep the parser-unit-test cases as plain assertions run manually from within `scratch/` (a `scratch/test_normalize_checkpoints.py`, not part of the tracked `tests/` suite), and drop that coverage from `tests/`.
**Verify:** `pytest tests/` still passes after removing the now-inapplicable test cases; `checkpoints/` and its `args.json` files are untouched (this subtask only moves the script, not its prior output).

### 7. `notebooks/`
`git mv psbd_analysis.ipynb backdoor_train_dev.ipynb sam_dev.ipynb psbd_sweep.ipynb notebooks/`. No import updates needed (notebooks aren't imported by anything), but `psbd_analysis.ipynb` and `psbd_sweep.ipynb` both reference the now-archived `plotting`/`experiment_io`/sweep mechanism — add a markdown note at the top of each flagging that they need updating once the PSBD sweep is rewritten, rather than silently leaving them looking runnable.
**Verify:** the four files exist under `notebooks/` and nowhere else; nothing else in the repo referenced them by path (grep to confirm, e.g. no PBS script or doc pointed at `./psbd_sweep.ipynb`).

### 8. `logs/` and `pbs/` reorganization
`logs/` is gitignored (pure filesystem cleanup, no git history concern) — move the 130 flat `train_*.err/out` files into the matching subdirectory that already exists for their category (`vit_no_sam/`, `vit_no_sam_tiny/`, `swin_no_sam/`, `swin_no_sam_individual/`), mirroring `pbs/`'s six-subdirectory structure exactly, so `logs/<category>/` and `pbs/<category>/` line up 1:1. The stray `sam_rho_benign_tiny.{err,out}` and `sam_rho_wanet_cifar10.{err,out}` files that match no existing category need a judgment call at implementation time (most likely `vit_sam/`, matching their content) rather than a blind move. `pbs/` itself doesn't need restructuring (already consistently organized, confirmed no stray files, no references to archived modules) — but for "add better logging mechanisms... so we can figure out what is bad" (TASKS.md), have each PBS script redirect output to a filename that includes `$PBS_JOBID` (or a timestamp) instead of the current fixed name that a rerun silently overwrites.
**Verify:** `find logs -maxdepth 1 -type f | wc -l` is 0 after the move (nothing left flat at `logs/` root); spot-check one rerun of a single PBS job produces a uniquely-named log instead of overwriting.

### 9. Update `tests/` for the new layout
Every test file's imports need the same absolute-path updates as production code (`tests/test_attacks.py`'s `from attacks import ...` stays the same by design, but `tests/test_dropout.py`'s `from dropout import ...` → `from defences.dropout import ...`, `from models import build_vit` → `from utils.models import build_vit`, and `tests/test_checkpoint_metadata.py`'s `from checkpoint_eval import read_checkpoint_metadata` → `from defences.checkpoint_eval import read_checkpoint_metadata`, `from train import save_checkpoint` stays unchanged since `train.py` didn't move). Add one new test, `tests/test_imports.py`, that imports every module in every new package plus every entrypoint left at root, so a future move that breaks an import surfaces immediately instead of silently (this is the concrete "add tests to see if everything works still after reorganization" TASKS.md asks for, beyond just not breaking the existing suite).
**Verify:** `pytest tests/` passes in full (the real bar for this whole reorganization — every subtask above should already keep this green incrementally, this subtask is where the suite itself gets updated to match).

### 10. Rewrite `CLAUDE.md`'s package-structure section
Replace "Flat package `psbd/` with no nested subpackages... never relative imports" with the actual structure from this plan: the six packages, what lives at root and why (§ decisions above), and the import convention (§ import convention above — relative within a package, absolute across packages, `attacks/` re-exports via its `__init__.py`, the others don't). Keep everything else in `CLAUDE.md` (correctness rules, checkpoint naming/`args.json`, code style) as-is except updating any file path it names that moved (e.g. "`configure_pre_residual_dropout`" references stay correct by name, but if the doc ever says `dropout.py` specifically, that becomes `defences/dropout.py`).
**Verify:** re-read the full file after editing for internal consistency — no leftover reference to a path that no longer exists.

### 11. `docs/architecture.md`
Only after 1–10 are done and verified (this documents the *result*, not a moving target). Two levels, per TASKS.md's own framing:
- **Macro**: one paragraph per package (`attacks/`, `analysis/`, `defences/`, `utils/`, plus root-level entrypoints and the shared `poison.py`/`backdoor_data.py`/`train.py`/`sam.py`/`metrics.py`) explaining what it's responsible for and the main call chains between packages (e.g. `train_backdoor.py` → `attacks.build_attack` → `poison.PoisonedTrainingSet` → `train.train_classifier` → `defences.detection.attack_success_rate`).
- **Micro**: per file, per function — what it does, its signature's meaning, and why it's needed (implementation-detail level, not just a docstring restate).
- **Final section**: worked `argparse` command examples for the main entrypoints, including the specific example requested (training a benign Swin model with particular hyperparameters), plus one example each for `train_backdoor.py`, `metrics.py`, and `analyze_latent.py`.
**Verify:** every file this document claims exists actually does (no drift from a plan that assumed a layout before implementation confirmed it); every argparse example given actually parses (`--help` or a dry construction of `parse_args()` with those flags, not just a plausible-looking string).

## Verification (whole plan)

- `pytest tests/` green after every subtask, not just at the end — this is what makes small, frequent commits safe to make.
- After subtask 9, one final full pass: `pytest tests/` (fast suite) and `pytest tests/ -m slow` (the GPU overfit check) both green, plus one real `python train_backdoor.py ...` and one real `python metrics.py --folder ...` invocation end to end, confirming the reorganization didn't just pass unit tests but the actual scripts people run still work.
- `git log --oneline` after all subtasks shows one commit per subtask (11 commits, roughly), each independently revertable if a later step turns out to depend on something the plan got wrong.
