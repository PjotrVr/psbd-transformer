# PSBD-ViT: `--max-samples` for fast, reproducible smoke tests

## Context

The only smoke-testing mechanism today is passing `--epochs 1` to `train_backdoor.py`/`train_benign.py` with a scratch `--output`/`--weights-dir` (see `tmp/psbd_smoke/`, `tmp/psbd_sam_smoke/` — ad hoc, not tracked, not documented). Epoch count and dataset size are independent knobs, and only epoch count is exposed: a 1-epoch smoke run on Tiny ImageNet still does a full forward+backward pass over 100k training images, then a full forward pass over its 10k-image test set for the post-training ASR/clean-accuracy eval. It is dramatically slower than a smoke test needs to be, and it exercises no fewer code paths than a real run — it's just epoch=1.

This plan adds a `--max-samples` flag to both entrypoints that truncates the *dataset*, not the epoch count. Truncating the dataset makes "1 epoch" naturally mean "a couple of batches" for free, so there is no separate batch-count concept to add.

**Design principle: one code path, not two.** `--max-samples -1` (the default) means "use the whole dataset." This is implemented as `max_samples=None` flowing into a single helper, `limit_dataset`, that returns the input dataset **unchanged** (same object, not a copy or a trivial wrap) when `max_samples is None`. A full run and a `--max-samples N` run therefore execute the exact same functions in the exact same order — `build_training_loader`, `build_clean_loader`, `evaluate_attack`, `train_classifier`, all of it — differing only in how many samples the underlying `Dataset` objects report. There is no `if smoke_test: ... else: ...` branch anywhere. This is what makes "smoke test and whole-dataset run have no difference between them" true by construction rather than by discipline.

**Design principle: seeded-random selection, not first-N.** Tiny ImageNet is loaded via `torchvision.datasets.ImageFolder` (`utils/datasets.py:63`), which lists samples sorted by class-folder name, then filename. `dataset[0:64]` there is the first one or two classes alphabetically, not a sample of the dataset — a literal first-N smoke run on Tiny would train and evaluate on essentially one class, which is close to useless as a check that the Tiny pipeline works, and could zero out the eligible ASR pool entirely for `all_to_one` if the truncated slice happens to be the target class. CIFAR-10/100 and GTSRB don't have this problem (their test sets are already class-shuffled), but the fix has to be dataset-agnostic, not a Tiny-specific special case. `limit_dataset` therefore always draws a **seeded random subset**, via `numpy`'s new Generator API (`np.random.default_rng(seed)`), which is deliberately isolated from the legacy global `np.random` state that `lightning.seed_everything` seeds — the same idiom `poison.py:choose_poison_indices` already uses. Same seed in, same subset out, every time; not first-N, not a different random subset on every invocation.

**Scope**: `utils/datasets.py`, `loaders.py`, `evaluate.py`, `train.py`, `train_backdoor.py`, `train_benign.py`, `CLAUDE.md` (schema line), `tests/`. Everything else — `metrics.py`, `defences/checkpoint_eval.py`, `analysis/analyze_latent.py`, the future PSBD sweep — calls the functions this plan touches without ever passing `max_samples`, so their behavior is provably unchanged (default `None` everywhere).

## A. `utils/datasets.py` — `limit_dataset`

New function, alongside `extract_labels` (which it composes with — `extract_labels` already recurses through `Subset` at `utils/datasets.py:88`, so no change needed there):

```python
def limit_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    """A reproducible random subset of dataset, or dataset itself when max_samples is None.

    Uses numpy's Generator API, which is isolated from the legacy global
    np.random state seed_everything seeds, so calling this never perturbs the
    RNG stream that model init or DataLoader shuffling later draw from.
    """
    n = len(dataset)
    if max_samples is None or max_samples >= n:
        return dataset
    indices = np.random.default_rng(seed).choice(n, size=max_samples, replace=False)
    return Subset(dataset, indices)
```

Requires adding `import numpy as np` to `utils/datasets.py` (not currently imported there; `Subset` already is, at `utils/datasets.py:11`).

Every call site below composes `limit_dataset` with whatever it already does — subset first, then derive labels/eligibility/loaders from the subset, mirroring the existing precedent in `tests/test_attack_triggers.py:213` (`Subset(train_clean, ...)` before `extract_labels`/`choose_poison_indices`).

## B. CLI: `--max-samples` on both entrypoints

`train_backdoor.py:parse_args` and `train_benign.py:parse_args`, each gets:

```python
parser.add_argument(
    "--max-samples",
    type=int,
    default=-1,
    help="Truncate each dataset to this many samples, reproducibly, for a fast "
         "smoke run (combine with --epochs 1). -1 (default) uses the whole dataset.",
)
```

`-1` is a CLI-only sentinel. It must never reach `limit_dataset`/`Subset`/slicing code — Python slicing treats `-1` as "all but the last element," so leaking the raw sentinel into subsetting logic would silently drop one sample instead of meaning "no limit," a quiet correctness bug. Normalize once, immediately after parsing, before anything reads `args.max_samples`:

- `train_backdoor.py:main()`, first line after `args = parse_args()`:
  ```python
  args.max_samples = None if args.max_samples == -1 else args.max_samples
  ```
- `train_benign.py:main()`, same line, before the per-dataset loop (normalized once, not per dataset).

Every function below reads `args.max_samples` after this point, so they only ever see `None` or a non-negative int — consistent with every other function in this codebase that already reads fields straight off `args` (`build_training_loader(args, image_size)` already does this for `args.dataset`, `args.poison_rate`, etc.), so no new parameter-passing convention is introduced.

No extra validation for `--max-samples 0`: it degrades to an empty `Subset`, which `train_one_epoch`'s `running_loss / max(len(loader), 1)` and `_prediction_accuracy`'s `total > 0 else 0.0` both already guard against crashing on. Harmless, obviously useless if someone does it, not worth a special-cased rejection.

## C. Reproducible seeding — the loaders/regular-workflow bracket

Per your requirement, bracket loader construction so training and evaluation start from a known, identical RNG state regardless of whether `--max-samples` triggered any subsetting:

```python
seed_everything(args.seed)      # (1) existing: before any loader/model work
...
train_loader, ... = build_training_loader(args, image_size)   # "create loaders"
val_loader = build_clean_loader(..., max_samples=args.max_samples, seed=args.seed)

seed_everything(args.seed)      # (2) new: right before the regular workflow starts
model = train_classifier(...)   # model init + training loop, now identical between
...                              # a full run and a --max-samples run
metrics = evaluate_attack(..., max_samples=args.max_samples, seed=args.seed)
```

Worth being explicit about what this bracket does and doesn't do, so it isn't mistaken for the only thing protecting determinism: since `limit_dataset` draws from an **isolated** `np.random.default_rng(seed)` rather than the global RNG, subsetting is already reproducible and already never touches the state `seed_everything` sets up, with or without the bracket. `DataLoader(shuffle=True)` doesn't draw at construction time either — shuffling happens at iteration time, inside `train_classifier`'s epoch loop, which already runs after step (2). So the bracket is defense-in-depth against future code (a different subsetting implementation, a library call that happens to touch global state) rather than the only thing standing between a smoke run and a divergent full run today — but it costs nothing and it's exactly the guarantee you asked for, so it's included as specified.

No third `seed_everything` is needed around the post-training `evaluate_attack`/`evaluate_benign` call: nothing downstream of it depends on RNG state, and `limit_dataset` is independently reproducible on every call regardless of global state.

`train_backdoor.py:main()` gets exactly this shape. `train_benign.py:train_one_benign()` gets the same shape, but keep the existing per-dataset seeding intact (`docs/plans/audit-fixes.md` §A3: `seed_everything` is called fresh inside the per-dataset loop, not once before it, so one dataset's run stays reproducible independent of loop order or an earlier dataset's failure):

```python
def train_one_benign(dataset_name, args, device):
    seed_everything(args.seed)                     # (1) existing, per-dataset
    ...
    train_loader, num_classes = build_benign_train_loader(
        dataset_name, args.raw_data_dir, args.batch_size, args.num_workers,
        args.max_samples, args.seed,
    )
    val_loader = build_clean_loader(
        dataset_name, args.raw_data_dir, args.batch_size, args.num_workers,
        max_samples=args.max_samples, seed=args.seed,
    )

    seed_everything(args.seed)                     # (2) new
    model = train_classifier(...)
    ...
    accuracy = evaluate_benign(
        model, dataset_name, device, args.raw_data_dir, args.batch_size,
        max_samples=args.max_samples, seed=args.seed,
    )["clean_accuracy"]
```

## D. Training-side threading

**`train_backdoor.py:build_training_loader`** — subset the clean training set before poison-index selection sees it, so `poison_rate` is measured against the truncated pool (mirrors the existing test precedent exactly):

```python
train_clean, _ = load_clean_datasets(args.dataset, transform, args.raw_data_dir)
train_clean = limit_dataset(train_clean, args.max_samples, args.seed)
normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

config = resolve_config(args.attack, args.poisoned_dir)
attack = build_attack(args.attack, config, image_size, args.target_label)

poisoned_train = build_training_set(train_clean, attack, config, args.poison_rate, args.seed, normalize, spec.num_classes)
```

No change needed inside `build_training_set`/`choose_poison_indices`/`choose_indices_with_cover` (`poison.py`) — they already read `labels = extract_labels(train_clean)`, and `extract_labels` already recurses through `Subset`. Both already clamp `count = min(round(rate * n), len(eligible))` and return an empty set rather than erroring when a pool is smaller than requested (`poison.py:119`, `poison.py:220`, `poison.py:236`), so a tiny `--max-samples` value degrades gracefully for every attack, including the cover-sample ones (`adaptive_blend`, `tact`) with `source_classes` constraints — worst case an attack gets zero poison/cover samples for one smoke run, not a crash.

**`train_benign.py:build_benign_train_loader`** — add `max_samples`/`seed` parameters, subset before wrapping in `DataLoader`:

```python
def build_benign_train_loader(
    dataset_name, raw_data_dir, batch_size, num_workers=8, max_samples=None, seed=0,
) -> tuple[DataLoader, int]:
    spec = DATASET_REGISTRY[dataset_name]
    transform = transforms_v2.Compose([...])  # unchanged
    train_dataset, _ = load_clean_datasets(dataset_name, transform, raw_data_dir)
    train_dataset = limit_dataset(train_dataset, max_samples, seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, spec.num_classes
```

## E. Evaluation-side threading

**`loaders.py:build_clean_loader`/`build_poisoned_loader`** — add `max_samples: int | None = None, seed: int = 0`, subset `test_base` right after it comes out of the (cached) loader, before anything downstream reads it:

```python
def build_clean_loader(dataset_name, raw_data_dir="raw_data", batch_size=64, num_workers=2, max_samples=None, seed=0) -> DataLoader:
    test_base, spec = _load_test_base(dataset_name, raw_data_dir)
    test_base = limit_dataset(test_base, max_samples, seed)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    clean_set = PoisonedTrainingSet(test_base, None, set(), normalize, spec.num_classes)
    return DataLoader(clean_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def build_poisoned_loader(dataset_name, attack, raw_data_dir="raw_data", batch_size=64, num_workers=2, max_samples=None, seed=0) -> DataLoader:
    test_base, spec = _load_test_base(dataset_name, raw_data_dir)
    test_base = limit_dataset(test_base, max_samples, seed)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    true_labels = extract_labels(test_base)
    poisoned_set = AttackSuccessSet(test_base, true_labels, attack, normalize, spec.num_classes)
    return DataLoader(poisoned_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
```

Important: subsetting happens **after** `_load_test_base`'s `@functools.lru_cache` call returns, never by changing what's passed into the cached function. `_load_test_base` is cached per `(dataset_name, raw_data_dir)` and shared across every checkpoint a sweep evaluates in a loop; if `max_samples` became part of what gets cached, a truncated dataset could get cached under the same key a full-dataset caller expects, corrupting every other consumer. Subsetting outside the cached call keeps the cache exactly as it is today and makes each `build_clean_loader`/`build_poisoned_loader` call's truncation fully independent of every other call's.

No change needed inside `AttackSuccessSet`/`is_eval_poisonable`/`attack_success_label` (`poison.py`) — eligibility filtering runs over whatever `base_dataset`/`labels` it's handed, same as the training side. A too-small eligible pool degrades to `ASR=0.0` (already guarded in `defences/detection.py:64`, `_prediction_accuracy`'s `total > 0 else 0.0`), not a crash.

**`evaluate.py:evaluate_benign`/`evaluate_attack`** — add the same two parameters, forwarded straight down:

```python
def evaluate_benign(model, dataset_name, device, raw_data_dir="raw_data", batch_size=64, max_samples=None, seed=0) -> dict:
    loader = build_clean_loader(dataset_name, raw_data_dir, batch_size, max_samples=max_samples, seed=seed)
    ...

def evaluate_attack(model, dataset_name, attack_name, config, target_label, device, raw_data_dir="raw_data", batch_size=64, max_samples=None, seed=0) -> dict:
    ...
    clean_loader = build_clean_loader(dataset_name, raw_data_dir, batch_size, max_samples=max_samples, seed=seed)
    poisoned_loader = build_poisoned_loader(dataset_name, attack, raw_data_dir, batch_size, max_samples=max_samples, seed=seed)
    ...
```

**Explicitly not touched**: `evaluate_checkpoint`, `evaluate_all_checkpoints` (`evaluate.py`), `metrics.py`, `defences/checkpoint_eval.py`, `analysis/analyze_latent.py`, and the future PSBD sweep rewrite. None of them ever pass `max_samples`, so the new parameter's default (`None`) means every one of these keeps evaluating the full dataset, byte-for-byte the same as before this plan. This is the concrete guarantee that a smoke-test feature for the two training entrypoints can't leak into result-producing evaluation code elsewhere.

## F. Checkpoint metadata

`max_samples` changes what data a model actually saw, which makes it training provenance, not an implementation detail — it belongs in the `args.json` sidecar CLAUDE.md already documents exhaustively. Add it to `train.py:checkpoint_metadata`:

```python
def checkpoint_metadata(..., max_samples: int | None, ...) -> dict:
    return {
        ...,
        "max_samples": max_samples,
        ...
    }
```

Both call sites (`train_backdoor.py:main`, `train_benign.py:train_one_benign`) pass `max_samples=args.max_samples`. A real run's checkpoints get `"max_samples": null`; a smoke run's get the actual int — so anyone reading `checkpoints/<folder>/args.json` later (a human, `results-aggregator`, `provenance-auditor`) can immediately tell a smoke checkpoint apart from a real one, even though — per your instruction — it was saved to the exact same `checkpoints/` layout via the exact same `--output`/`--weights-dir` path as any other run, with no special redirect.

`CLAUDE.md`'s "Checkpoint naming and metadata" section lists the `args.json` key set explicitly (`dataset`, `attack`, `label_mode`, ..., `git_commit`, `trained_started_at`, `trained_ended_at`) — add `max_samples` to that list in the same commit so the doc doesn't drift from the schema.

## G. Correctness rules this must not regress

- **Uniform 15 epochs** (CLAUDE.md) governs real, result-producing runs. This plan doesn't auto-override `--epochs` for a `--max-samples` run — you control both flags independently (`--epochs 1 --max-samples 64` is your own stated usage), so nothing stops someone from accidentally combining `--max-samples` with `--epochs 15` and getting a fast-but-still-15-epoch run. That's intentional per your answer that we shouldn't add logic here; worth a one-line docstring note in both entrypoints so it's clear `--max-samples` alone doesn't imply smoke semantics.
- **ASR eligibility** (`is_eval_poisonable`/`attack_success_label`, a2o vs a2a vs clean_label) is untouched — see §E. Confirmed safe under truncation via the existing divide-by-zero guard.
- **SAM (rho=0.1, two-pass update)** is untouched — `_sam_update` just runs on whatever loader it's given, so `--use-sam` gets exercised by a smoke run exactly as it would by a real one.
- **Pre-residual dropout placement** (`defences/dropout.py`) is out of scope — it's applied at PSBD eval-time, not during `train_backdoor.py`/`train_benign.py` training.
- **Checkpoint naming template** is untouched — per your decision, no smoke-specific output redirect; §F adds a metadata field precisely so this doesn't need one.

## H. Tests

- New unit tests (`tests/test_datasets.py` or added to the closest existing file covering `utils/datasets.py`) for `limit_dataset`:
  - `max_samples=None` returns the identical object (`is`, not just `==`).
  - `max_samples >= len(dataset)` also returns the identical object.
  - `max_samples < len(dataset)` returns a `Subset` of length `max_samples`, indices drawn without replacement.
  - same `(n, max_samples, seed)` called twice returns the same indices; a different seed (usually) returns different indices.
- Determinism check, mirroring `docs/plans/audit-fixes.md`'s own verification for the original seeding fix: run `train_backdoor.py --max-samples 16 --epochs 1 ...` twice with the same `--seed` and diff the resulting state_dicts — must be identical. This is the concrete verification of §C's bracket actually working end to end, not just in isolated unit tests.
- Cross-dataset smoke sanity via `train_benign.py --max-samples 64 --epochs 1 --datasets cifar10 cifar100 gtsrb tiny`: confirm it finishes fast, and specifically confirm Tiny's per-class label distribution in the truncated subset spans more than one or two classes (the concrete regression check for the ImageFolder-ordering problem this plan exists to avoid).
- Attack-mode smoke sanity: one `train_backdoor.py --max-samples 64 --epochs 1` run per label mode (`badnet_a2o`, `badnet_a2a`, `sig` or `lc` for clean_label) plus one cover-sample attack (`tact` or `adaptive_blend`), confirming none crash and each produces a checkpoint with a non-crashing `asr`/`clean_accuracy` printout.

## Verification

- `pytest tests/` green, including the new `limit_dataset` unit tests.
- `python train_backdoor.py --dataset cifar10 --attack badnet_a2o --poison-rate 0.1 --max-samples 64 --epochs 1 --output tmp/smoke/attack_result.pt` completes in seconds, not minutes, and prints a sane (not NaN, not crashing) final ASR/CA line.
- Same command run twice with the same `--seed` produces identical state_dicts (§H determinism check).
- `python train_benign.py --datasets cifar10 cifar100 gtsrb tiny --max-samples 64 --epochs 1 --weights-dir tmp/smoke` completes for all four datasets, and Tiny's subset is confirmed multi-class.
- A full run with `--max-samples -1` (or the flag omitted entirely) is spot-checked to behave identically to the current `main` branch's behavior — same functions called, same loader sizes, confirming the "no separate code path" guarantee holds in practice, not just by code inspection.
- `checkpoints/<smoke-run-folder>/args.json` has `"max_samples"` set to the value used; a normal run's has `null`.
- `ruff format .` after all edits.
