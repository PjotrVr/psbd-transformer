# PSBD-ViT: fix pipeline bugs, normalize checkpoint metadata, add metrics.py, clean up dead code

## Context

The psbd-reviewer audit found training runs with no seeding, zero checkpoint metadata, a clean-label ASR set that measures nearly the opposite of attack success, stale argparse defaults, and several unimportable/unused files. Investigating the fix for checkpoint metadata surfaced a deeper issue: the PNG-based BackdoorBench evaluation path (`sweep.py`/`run_sweep.py`) points at a directory (`vit_b_16_weights/`) that doesn't exist on disk — the real downloaded data lives in `backdoor_bench_checkpoints/`, a name the code never adopted. Given that, and that the PSBD dropout-rate detection sweep itself is being redesigned in a later pass, this plan's scope is: fix every confirmed correctness bug, normalize `checkpoints/` into a self-describing naming convention with real metadata, add a lean `metrics.py` for baseline attack-success/clean-accuracy reporting, retarget the BackdoorBench directory name, move dead code out of the way, and get verification into `tests/`. The PSBD sweep mechanism itself (`sweep.py`, `run_sweep.py`, `experiment_io.py`'s per-quantile rows) is explicitly **archived, not rewritten** — that comes later, as its own task. This plan ends in one commit; the larger codebase rewrite the user wants after that is a separate, future planning session.

## A. Correctness fixes

### A1. Argparse defaults + dropout scope
- `train_backdoor.py`/`train_benign.py`: `--epochs` default `10→15`, `--rho` default `0.05→0.1`.
- `sam.py:SAM.__init__` and `train.py:train_classifier`: `rho` default `0.05→0.1`.
- `dropout.py:configure_pre_residual_dropout`: switch to `model.named_modules()`, skip names ending in `"encoder.dropout"` (excludes ViT's embedding dropout — applied once before the 12-block stack, not pre-residual — leaves the 36 true per-block dropouts; structurally a no-op for Swin, which has no such module). Update the CLAUDE.md "open note" about this from a question to settled behavior.

### A2. Clean-label ASR set — `is_eval_poisonable` / `attack_success_label` / `AttackSuccessSet`
`poison.py:is_poisonable`'s `clean_label` branch (`original_label == target_label`) is correct for training but backwards for eval, which must ask "does the trigger fool a *non*-target image into predicting target." `all_to_one`/`all_to_all` already ask the right question at eval time.

- Add to `poison.py`: `is_eval_poisonable(label_mode, original_label, target_label)` (same as `is_poisonable` except `clean_label` returns `original_label != target_label`) and `attack_success_label(label_mode, original_label, target_label, num_classes)` (same as `poisoned_label` except `clean_label` returns `target_label`).
- Rename `FullyPoisonedTestSet` → `AttackSuccessSet`, swap its internal calls to the two new functions. Byte-identical behavior for `all_to_one`/`all_to_all`; only `clean_label` changes.
- Update references: `train_backdoor.py`, `verify_attacks.py`, `test_attacks.py` (rename + add a clean-label test), `attack_tact.py` docstring.
- `checkpoint_eval.py:build_eval_loaders_from_attack` currently hand-rolls a force-poison-every-index `PoisonedTrainingSet` for `backdoor_eval` — same clean-label bug plus an extra one for all_to_one (includes already-target images). Replace with `AttackSuccessSet`.
- `backdoor_data.py:PngPathDataset`/`load_backdoor_splits` (kept — see §E, still needed for `backdoor_bench_checkpoints/`): change from a constant `trigger_label` to per-sample eligibility/labeling via `is_eval_poisonable`/`attack_success_label` — **not** the training-time `is_poisonable`/`poisoned_label` (that would silently reproduce the clean-label bug inside the PNG path). Add `label_mode_from_folder(folder_name)` to `config.py` (checks for `"a2a"` substring) since these folders have no metadata file to read `label_mode` from.

### A3. Seeding
- `train_backdoor.py:main()`: `seed_everything(args.seed)` right after `args = parse_args()`, before any loader/model construction.
- `train_benign.py`: add `--seed` (currently missing entirely), call `seed_everything(args.seed)` as the first statement inside `train_one_benign()` (per-dataset, not once before the loop, so each dataset's run is reproducible independent of loop order or an earlier dataset's failure).
- `verify_attacks.py:overfit_sanity_check` — leave unseeded with a one-line comment; it's a smoke test, not a result-producing run.

## B. Normalize `checkpoints/` — canonical names + `args.json`

Folder names today are inconsistent in two ways: the SAM rho tag is `_sam_rho_0_1` in backdoor sweeps but `_sam_rho0_1` in benign runs, and the architecture token appears as a prefix (`swin_cifar100_...`) for attacks but mid-name (`cifar100_swin_benign_...`) for benign. Standardize on:

```
{architecture}_{dataset}_{attack_or_benign}[_{poison_rate_tag}]_{optimizer_tag}
```
SAM is always SAM-on-top-of-AdamW here, so adam is the unmarked default: the optimizer tag is omitted entirely for adam runs, and only present as `sam_rho_{rho_tag}` for SAM runs. Examples: `vit_cifar100_badnet_a2o_0_01` (adam), `swin_cifar100_benign_sam_rho_0_1` (SAM), `vit_tiny_wanet_0_05_sam_rho_0_2` (SAM). The `args.json` `"optimizer"` field is still always explicit (`"adam"` or `"sam"`) regardless of what the folder name omits.

New tracked script `normalize_checkpoints.py` at repo root (one-time, re-runnable, `--dry-run` by default):
1. Parser handling both known rho-tag formats and both architecture-token positions, producing `(architecture, dataset, attack_or_benign, poison_rate, optimizer, rho)`. When a folder name has neither a `vit`/`swin` token at all, do not guess — open `attack_result.pt` and inspect the state_dict's key names directly (no need to fully construct the model): ViT and Swin checkpoints have structurally distinct key sets (e.g. ViT's wrapped `vit_b_16` has `conv_proj`/`class_token`/`encoder.layers.encoder_layer_*` keys, Swin's has `features.*` keys). Pick whichever architecture's key pattern matches, and treat that as ground truth for the renamed folder's `vit_`/`swin_` prefix and the `architecture` field in `args.json` — the folder name is cosmetic, the checkpoint's actual weights are the source of truth. Log every folder this heuristic had to resolve (i.e. every currently-ambiguous folder) so it's easy to spot-check a few by hand.
2. Canonical name builder from the template above; skip folders already canonical.
3. `git mv checkpoints/<old> checkpoints/<new>` for every folder that needs it. Dry-run prints the full rename table first.
4. Write `args.json` into every folder (renamed or not) with:
   ```json
   {"dataset", "attack", "label_mode", "target_label": 0, "poison_rate", "cover_rate",
    "architecture", "optimizer", "rho": null_if_adam, "epochs": 15, "seed": null,
    "git_commit": null, "trained_started_at": null, "trained_ended_at": null}
   ```
   `label_mode` and `cover_rate` come from `attacks.default_config(attack).label_mode` / `getattr(config, "cover_rate", 0.0)` (for benign: `attack="benign"`, `label_mode=null`, `poison_rate=0.0`, `cover_rate=0.0`). `seed`/`git_commit`/timestamps are `null` — genuinely unrecoverable, per your instruction to null anything we can't know (no run was seeded before this fix, so `null` for seed is also just accurate).
5. Report any folder name that doesn't parse cleanly instead of guessing — flag for manual naming.

`backdoor_bench_checkpoints/` is **not** renamed or given `args.json` — it's external downloaded data in BackdoorBench's own convention; only the code's directory-name reference changes (§F).

### Write side going forward
- `train.py:save_checkpoint`: unchanged `.pt` contents; when `metadata` is passed, write it to `args.json` next to the checkpoint instead of merging into the `.pt` dict.
- Add `train.checkpoint_metadata(...)` builder (same schema as above, minus the `null`s — fresh runs record real `seed`/`git_commit`/timestamps) so `train_backdoor.py` and `train_benign.py` build identical key sets.
- `train_backdoor.py`: widen `build_poisoned_loaders`'s return to also surface the `attack`/`config` objects it already builds internally (needed for `label_mode`/`cover_rate`), capture start/end timestamps around training, pass `metadata=` into the existing `save_checkpoint` call (currently called with none, despite its own docstring claiming otherwise).
- `train_benign.py`: switch its existing partial inline metadata dict to the shared builder.
- `checkpoint_eval.py:read_checkpoint_metadata`: read `args.json` next to the checkpoint path instead of pulling keys out of the `.pt` via `torch.load`.

## C. `metrics.py` — baseline attack/benign metrics

A lean, new root-level script, decoupled from the (archived) PSBD sweep. For every checkpoint folder in `checkpoints/` and `backdoor_bench_checkpoints/`:
- **Mode detection**: `checkpoints/` folders read `args.json`'s `"attack"` field (`"benign"` or an attack name). `backdoor_bench_checkpoints/` folders have no `args.json` and are confirmed 100% attack folders (0 benign among the 101) — mode is always `"attack"` there, attack/dataset inferred from the folder name.
- **Benign mode**: load the checkpoint, build the clean test loader, compute `clean_accuracy` (existing `detection.clean_accuracy`) and `clean_accuracy_by_class` (new function in `detection.py`, sibling to `_prediction_accuracy`, tracking correct/total per label).
- **Attack mode, `checkpoints/` folders**: reconstruct the poisoned eval set in memory using the attack rebuilt from `args.json` (`attacks.build_attack`/`default_config`, same as `checkpoint_eval.py:build_eval_loaders_from_attack`, now fixed to use `AttackSuccessSet` per §A2) — no PNGs needed, since we know exactly which attack produced these.
- **Attack mode, `backdoor_bench_checkpoints/` folders**: use `backdoor_data.load_backdoor_splits` (the PNG path, fixed per §A2) since there's no `args.json` and no local attack object to rebuild from — this is the one place PNG-based loading survives.
- **ASR definition** (standard, confirmed): fraction of eligible (non-already-target, or as `is_eval_poisonable` defines per label_mode) triggered samples the model now predicts as `target_label`. Reuses `detection.attack_success_rate` against the `AttackSuccessSet`/PNG-based eligible set.
- **Output**: `analysis/<folder_name>/metrics.json` — one dict for benign (`{folder_name, architecture, clean_accuracy, clean_accuracy_by_class}`), one for attack (`{folder_name, architecture, dataset, attack, label_mode, poison_rate, target_label, asr, clean_accuracy}`). This filename is reserved for baseline attack/benign metrics only — the (future, archived-for-now) PSBD sweep will write its own `psbd_metrics.json` into the same per-folder directory, and future defenses their own `<defense>_metrics.json` (e.g. `strip_metrics.json`), all living side by side under `analysis/<folder_name>/`.
- Runnable as `python metrics.py` (iterates every folder in both directories) with a `--folder` override for a single one, useful for the verification step below.
- Before computing anything, mirror the full `checkpoints/` (and `backdoor_bench_checkpoints/`) folder listing into `analysis/` 1:1 — create `analysis/<folder_name>/` for every folder in both source directories, even ones `metrics.py` hasn't gotten to yet or that fail mid-run, so `analysis/`'s directory listing always matches the checkpoint inventory exactly and nothing is silently missing from it.

## D. `analysis/` output directory
Replaces the old `experiments/<placement>/<folder>/` concept for this pass (no PSBD-sweep output exists yet to migrate, since that mechanism is archived). `analysis/<folder_name>/metrics.json` is the main thing written by this plan; the directory itself is created for every checkpoint up front (see §C) and is otherwise ready for the future PSBD sweep and per-checkpoint cached arrays (e.g. `pre_residual_outputs.npy`) the later rewrite will add.

## E. Dead code and PNG-sweep archival → `_archive/`

New tracked top-level `_archive/` directory (not gitignored — kept as reference, "copy just in case" per your instruction).

**Confirmed unimportable / unused, `git mv` (preserves history):** `psbd.py` (`ImportError` on `vgg13`), `process_dataset.py` and `utils.py` (`ImportError` on `ATTACK_REGISTRY`, which doesn't exist).

**PSBD-sweep-specific, archived per your instruction (not rewritten now):** `run_sweep.py`, `sweep.py`. Before archiving, grep every other file in the repo for imports of `experiment_io` and `inference` (both currently used only by `sweep.py` as far as this audit found, but `analyze_latent.py` is a plausible other consumer worth checking directly rather than assuming) — archive `experiment_io.py` too if genuinely orphaned once `sweep.py` is gone, otherwise leave it in place for the future rewrite to reuse and note why in the same commit.

**Explicitly kept, not archived:** `backdoor_data.py` (its `PngPathDataset`/`load_backdoor_splits` are now `metrics.py`'s only source of eval data for `backdoor_bench_checkpoints/`, per §C; its `balance_by_class`/`split_validation_and_eval`/`extract_labels` helpers are reused by the in-memory path too) — only `sweep.py`/`run_sweep.py` go, not the PNG-reading utility itself.

**Untracked/gitignored legacy directories, `mv` + `git add` + remove the corresponding `.gitignore` lines (currently invisible to git; this makes them tracked reference material instead):** `_attacks/`, `_defences/`, `_models/`, `copied_attacks/`, `copied_train/`. Confirmed via repo-wide grep: nothing outside these directories imports from any of them.

**Untracked/gitignored, same treatment:** `tmp.py` (not even valid standalone Python — references `nn`/`models` without importing them; confirmed nothing imports it).

## F. Rename `vit_b_16_weights` → `backdoor_bench_checkpoints` (read paths only)
`vit_b_16_weights` only ever meant one thing: the externally-downloaded BackdoorBench reference data (PNGs + their checkpoints), read-only, evaluated but never written to by this repo. That rename applies only to **read paths**: `config.py:RunConfig.weights_dir` default (feeds `sweep.py`'s PNG-based evaluation) and `run_sweep.py`'s `--weights-dir` default (moot once archived, but fix before archiving so the archived copy isn't misleading).

**Correction:** `train_benign.py`'s `--weights-dir` is a **write path** (where new locally-trained checkpoints land) and was mistakenly renamed to `backdoor_bench_checkpoints` in the first implementation pass — it must default to `checkpoints` instead, matching `train_backdoor.py`'s own output convention and CLAUDE.md's rule that local training output lives in `checkpoints/`. Same correction for the docstring examples in `train_backdoor.py`'s `--output` and `analyze_latent.py`'s `--checkpoint` — both are local-checkpoint examples and should read `checkpoints/vit_cifar10_badnet_a2a_0_1/attack_result.pt`-style paths (matching §B's canonical naming), not `backdoor_bench_checkpoints/...`. (Already fixed directly, ahead of the agent pass that introduced the mistake.)

`.gitignore` already has both `vit_b_16_weights/` and `backdoor_bench_checkpoints/` entries — no gitignore change needed.

## G. `tests/` — done
- `pytest` added as a dev dependency (`[dependency-groups] dev`), plus `[tool.pytest.ini_options]` in `pyproject.toml`: `pythonpath = ["."]` so flat-import modules at repo root resolve when pytest runs from `tests/`, a registered `slow` marker, and `addopts = "-m 'not slow'"` so a plain `pytest` run skips GPU/training tests by default.
- `git mv test_attacks.py tests/test_attacks.py` — already proper pytest style (18 `test_*` functions, including the §A2 clean-label addition), just relocated.
- `verify_attacks.py` was a CLI script (`check_*` functions, no `test_*` names, a GPU-training `overfit_sanity_check`), not a pytest suite. Rewritten as `tests/test_attack_triggers.py` (named for what it verifies, not the old script name): the cheap/fast checks become real `test_*` functions (`test_label_policy`, `test_badnet`, `test_blend`, `test_sig`, `test_wanet`, `test_low_frequency`); `dump_visuals` stays as a manual (non-test) utility runnable via `python tests/test_attack_triggers.py`; `overfit_sanity_check` is now `test_overfit_sanity_check`, marked `@pytest.mark.slow`, with real assertions in place of the old print-a-verdict pattern. Old `verify_attacks.py` removed (`git rm`, not a straight rename since the content changed substantially — history stays in `git log`).
- Add new tests, beyond what §A2 already added to `test_attacks.py`:
  - **Per-attack coverage**: confirm every attack in `attacks.ATTACK_NAMES` has its own dedicated test, not just a representative subset — trigger correctness (`apply_trigger` produces the expected perturbation) and `AttackSuccessSet` eligibility/labeling for that attack's actual `label_mode`, parametrized over all attack names rather than hand-picking a few.
  - **Checkpoint loading**: `models.load_checkpoint` round-trips for both architectures (save a small model via `train.save_checkpoint`, reload via `models.load_vit_checkpoint`/`load_swin_checkpoint`, confirm weights match); `checkpoint_eval.read_checkpoint_metadata` round-trips through a real `args.json` (already planned) plus the missing-file and missing-key error cases; `normalize_checkpoints.py`'s folder-name parser handles both known rho-tag formats and both architecture-token positions correctly (unit tests against literal example strings pulled from the real `checkpoints/` listing, not live filesystem access).
  - **Dropout placement precision**: `configure_pre_residual_dropout` touches exactly the 36 expected per-block modules and never `encoder.dropout` (regression test for the §A1 fix, not just a manual count check); `reset_dropout` correctly zeroes the rate back to 0 and returns the model to eval-mode dropout; and a placement-selectivity test confirming `configure_dropout(model, rate, placement)` only mutates modules belonging to the requested placement (`pre_residual` vs `post_residual`) and leaves modules outside that placement at their prior rate, so partial/selective dropout placement is verified structurally rather than assumed.

## H. `CLAUDE.md` updates (same commit)
- Checkpoint naming convention (§B's template) and the `args.json` schema, replacing the current "Each checkpoint gets a `metrics.json`" line (that name is now reserved for `analysis/<folder>/metrics.json`, a different file with a different purpose).
- `backdoor_bench_checkpoints/` as the correct external-data directory name (the doc currently already says this, code just didn't match — confirm no remaining drift).
- Dropout open-note resolved (§A1).
- Note that `sweep.py`/`run_sweep.py`/PSBD-sweep output are archived pending a future rewrite, so nobody goes looking for `experiments/` or tries to run `run_sweep.py` expecting it to work.

## Verification
- A1: `configure_pre_residual_dropout(build_vit(10), 0.1)` → `36`.
- A2/A3: `pytest tests/test_attacks.py` including the new clean-label `AttackSuccessSet` case; two identical `train_backdoor.py` runs on a 1-epoch config produce identical state_dicts.
- B: `normalize_checkpoints.py --dry-run` reviewed folder-by-folder for a sample across every naming pattern found (plain attack, SAM attack, plain benign, SAM benign, swin variants of each) before the real run; spot-check a handful of renamed folders' `args.json` afterward.
- C: `python metrics.py --folder <one benign checkpoints/ folder> <one attack checkpoints/ folder> <one backdoor_bench_checkpoints/ folder>`, confirm `analysis/<folder>/metrics.json` has the right shape for each of the three cases and that ASR/CA numbers are sane (CA near training-time reported accuracy, ASR high for a badnet checkpoint at a normal poison rate).
- E: confirm `import psbd`, `import process_dataset`, `import utils` still fail (unchanged) from repo root; confirm nothing else in the repo fails to import after `sweep.py`/`run_sweep.py`/etc. are gone (`python -c "import <every remaining top-level .py module>"` loop, or just run the new `tests/` suite).
- G: `pytest` (default) runs fast and skips `slow`-marked tests; `pytest -m slow` runs the GPU overfit check separately.
- Final step: review `git status`/`git diff` for the full change set, then one commit covering all of A–H.
