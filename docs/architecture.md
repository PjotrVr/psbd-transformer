# PSBD-ViT Architecture Reference

Generated as a detailed, file:line-anchored implementation reference. Describes
what the code actually does, not what it should do. Sections are appended in
dependency order (leaf utilities first, entry points last), and each section
was written after reading the corresponding source file directly.

## Overview

Package-by-package summary:

- **`attacks/`** — the 10 trigger implementations (BadNet, Blend, SIG, WaNet,
  LF, LC, BPP, Adaptive-Blend, TaCT, and a generic PNG-based `generated`
  attack for BackdoorBench-style pregenerated triggers), plus a registry in
  `attacks/__init__.py` (`ATTACK_NAMES`, `build_attack`, `default_config`)
  that dispatches by attack name string. Every attack is a plain `Attack`
  dataclass (name, `apply_trigger` function, label mode, target label) built
  from `poison.py`'s shared interface, not a class hierarchy.
- **`defences/`** — the PSBD detection mechanism itself: pre-residual dropout
  injection (`dropout.py`), stochastic multi-pass inference (`inference.py`),
  detection statistics and eval-loop metrics (`detection.py`), and the
  higher-level checkpoint-to-eval-set pipeline (`checkpoint_eval.py`).
- **`analysis/`** — latent-space tooling used for the paper's mechanistic
  claims: debiased linear CKA (`cka.py`), backdoor direction / TAC
  (`direction.py`), PCA/UMAP embedding (`embedding.py`), layer-wise feature
  extraction (`features.py`), Lipschitz constant estimation (`lipschitz.py`),
  and a worked-example CLI entrypoint that ties them together
  (`analyze_latent.py`).
- **`utils/`** — `config.py` (dataset registry, `RunConfig`, BackdoorBench
  folder-name parsing) and `datasets.py` (clean-dataset loaders). `models.py`
  deliberately stays at the repo root, not here, since it is loaded from
  nearly every package and entrypoint the same way.
- **`plotting/`** — currently an empty scaffold (docstring-only
  `__init__.py`); the previous plotting code moved to `_archive/` pending a
  rewrite.
- **Repo root** — `train_backdoor.py`, `train_benign.py`, `train.py`, `sam.py`,
  `metrics.py` (entrypoints and the training/eval pipeline core whose only
  real callers are the two training scripts) and `poison.py`,
  `backdoor_data.py`, `models.py` (shared building blocks used from nearly
  everywhere: the attack interface, BackdoorBench PNG-trigger loading, and
  model construction/loading).

### Data flow: dataset to trained checkpoint to eval metric

1. **Dataset loading.** `utils/datasets.py:load_clean_datasets` returns clean
   (train, test) datasets in `[0, 1]` pixel range for one of the four
   registered datasets (`utils/config.py:DATASET_REGISTRY`), with no
   normalization baked in.
2. **Poisoning.** `train_backdoor.py` builds an `Attack` via
   `attacks/__init__.py:build_attack`, picks indices to poison with
   `poison.py:choose_poison_indices` (or `choose_indices_with_cover` for
   Adaptive-Blend/TaCT), and wraps the clean training set in
   `poison.py:PoisonedTrainingSet` (or `CoverPoisonedTrainingSet`), which
   applies the trigger and normalization lazily in `__getitem__`.
3. **Training.** `train.py:train_classifier` runs the actual PyTorch loop
   (SGD/Adam optionally wrapped in `sam.py:SAM`'s two-pass update) over a
   model built by `models.py:build_vit`/`build_swin`. `train_backdoor.py` and
   `train_benign.py` are the two entrypoints that assemble loaders and call
   it; both delegate checkpoint writing to `train.py:save_checkpoint`, which
   writes `checkpoints/{folder_name}/attack_result.pt` plus an `args.json`
   provenance sidecar (`train.py:checkpoint_metadata`).
4. **Eval metrics.** `metrics.py` (an entrypoint, not a package) walks
   `checkpoints/`, reads each `args.json` via
   `defences/checkpoint_eval.py:read_checkpoint_metadata`, rebuilds the same
   `AttackSuccessSet`/clean test set in memory, loads the model with
   `models.py:load_checkpoint`, and calls
   `defences/detection.py:clean_accuracy`/`attack_success_rate` to write
   `results/{folder_name}/metrics.json`. `metrics.py` is scoped to
   `checkpoints/` only and never touches `backdoor_bench_checkpoints/` (no
   `args.json`) at all. `backdoor_data.py:load_backdoor_splits` is the
   module meant to read that folder's pregenerated PNG triggers, but as of
   this reading it has **no live callers anywhere in the tracked tree**,
   only in the archived `_archive/sweep.py` — see the `backdoor_data.py`
   section's Sharp edges below.
5. **PSBD detection** (currently archived pending a rewrite, see
   `_archive/sweep.py`) sits downstream of step 4: it takes a trained
   checkpoint, applies `defences/dropout.py:configure_pre_residual_dropout`
   (or `configure_post_residual_dropout`), runs
   `defences/inference.py:stochastic_forward_passes` over a paired
   clean/backdoor eval set, and scores samples with
   `defences/detection.py`'s PSU-shift statistic against a quantile
   threshold.

---

## `utils/datasets.py`

Purpose: clean-dataset loading (train/test pairs for CIFAR-10, CIFAR-100,
GTSRB, Tiny ImageNet), a label-extraction helper that avoids decoding every
image just to read `y`, and two transform helpers. Docstring states side
effects (disk reads, downloads) are isolated here so detection/analysis code
stays pure — consistent with CLAUDE.md's "isolate side effects" rule.

### Functions

**`build_transform(dataset_name: str) -> transforms_v2.Compose`**
(`utils/datasets.py:17-26`)
- `Compose([Resize((224,224)), ToTensor(), Normalize(mean=spec.mean,
  std=spec.std)])` — resize, tensor conversion, and normalization all baked
  into one transform.
- **No callers found anywhere in the tracked tree** (grepped for
  `build_transform` across the repo, excluding `_archive/`). Every actual
  caller of `load_clean_datasets` (`train_backdoor.py`, `metrics.py`,
  `defences/checkpoint_eval.py`, `analysis/analyze_latent.py`) instead
  defines its own local `base_transform` that stops at `ToTensor()` — no
  `Normalize` — and applies normalization separately downstream (see Sharp
  edges).

**`denormalize(image: torch.Tensor, dataset_name: str) -> torch.Tensor`**
(`utils/datasets.py:29-34`)
- `image * std + mean`, with `std`/`mean` reshaped to `(-1, 1, 1)` for
  channel-wise broadcast against a CHW tensor.
- **No callers found** anywhere in the tracked tree either. A pure utility
  presumably intended for visualization/trigger-inspection notebooks, not
  currently wired into any tracked script.

**`load_clean_datasets(dataset_name, transform, raw_data_dir) -> tuple[Dataset, Dataset]`**
(`utils/datasets.py:37-65`)
- Looks up `spec = DATASET_REGISTRY[dataset_name]`, `root =
  raw_data_dir/dataset_name`.
- `spec.loader_kind == "gtsrb"` → `tv_datasets.GTSRB(root, split="train"/
  "test", download=True, transform=transform)`.
- `spec.loader_kind == "image_folder"` (Tiny ImageNet) → `ImageFolder(root/
  "train")` / `ImageFolder(root/"val")` — relies on Tiny ImageNet's
  validation images already being reorganized into per-class subfolders
  (the standard layout BackdoorBench's own download/prep produces; the raw
  Tiny ImageNet val split ships as a flat folder with a separate annotation
  file, which `ImageFolder` cannot read directly).
- Otherwise (`"cifar10"`/`"cifar100"`) → dispatches through a local dict to
  `tv_datasets.CIFAR10`/`CIFAR100(root, train=True/False, download=True,
  transform=transform)`.
- I/O: `download=True` on every branch means this can trigger a network
  fetch on first use (needs the proxy exports from CLAUDE.md's cluster
  setup on Supek). Side effect: writes into `raw_data_dir` if not already
  present.
- Called by: `train_backdoor.py:83`, `train_benign.py:50`, `metrics.py:104`,
  `defences/checkpoint_eval.py:60`, `analysis/analyze_latent.py:61`,
  `tests/test_attack_triggers.py:177`.

**`extract_labels(dataset: Dataset) -> list[int]`** (`utils/datasets.py:68-86`)
- Recursion: if `dataset` is a `Subset`, recursively calls
  `extract_labels(dataset.dataset)` on the parent then indexes by
  `dataset.indices` — correctly handles nested/wrapped subsets without
  decoding any images.
- Otherwise tries, in order: `.targets` attribute (CIFAR-10/100) → `.samples`
  attribute, taking the label half of each `(path, label)` tuple
  (`ImageFolder`) → `._samples` (torchvision's internal GTSRB attribute) →
  falls back to `[int(dataset[i][1]) for i in range(len(dataset))]`, which
  **does** decode every image (the slow path the docstring calls out,
  previously hit on Tiny ImageNet before this function existed).
- Called by: `train_backdoor.py:62,103`, `metrics.py:108`,
  `backdoor_data.py:91,117,144`, `defences/checkpoint_eval.py:80`,
  `tests/test_attack_triggers.py:182`.

### Interactions

Imports only `.config` (`DATASET_REGISTRY`, `DatasetSpec`) via a relative
import, per the project's within-package convention. Every downstream caller
of `load_clean_datasets` passes its own locally-defined transform rather than
`build_transform` — the actual normalization boundary (dataset yields 0-1
pixel-space images; a separate `transforms_v2.Normalize` is applied later, at
the point where `PoisonedTrainingSet`/`AttackSuccessSet` also apply it) is
implemented redundantly in `train_backdoor.py`, `metrics.py`,
`defences/checkpoint_eval.py`, and `analysis/analyze_latent.py`, not
centralized in this module despite `build_transform` existing here for
exactly that purpose.

### Sharp edges

- `build_transform` bakes `Normalize` directly into the dataset transform,
  which is incompatible with the poisoning pipeline: `PoisonedTrainingSet`
  and `AttackSuccessSet` expect their `base_dataset` to yield 0-1,
  *unnormalized* images so `attack.apply_trigger` (pixel-space) runs before
  normalization (per `poison.py`'s module docstring). If any future caller
  passes `build_transform`'s output as the `transform` into
  `load_clean_datasets` and then wraps the result in `PoisonedTrainingSet`,
  triggers would be stamped onto already-normalized tensors — silently
  wrong pixel-space math, not an exception. This is presumably *why* every
  real call site rolls its own non-normalizing transform instead of using
  this function — but nothing enforces or documents that constraint at the
  call boundary.
- `extract_labels`'s fallback branch (`[int(dataset[i][1]) for i in
  range(len(dataset))]`) silently decodes every image if none of `.targets`
  / `.samples` / `._samples` exist — no warning is printed, unlike
  `models.py`'s missing/unexpected key prints. A new dataset loader kind
  added without one of those three attributes would quietly regress to the
  slow path.
- `._samples` is a private torchvision `GTSRB` attribute (leading
  underscore) — a torchvision version bump that renames or removes it would
  break this branch silently by falling through to the slow per-item
  decode, not raising an `AttributeError` (since `hasattr` catches it).

---

## `poison.py`

Purpose: defines the shared `Attack` interface every attack file builds an
instance of, the label-mode policy functions that decide which samples get
poisoned and what label they receive (at training time vs eval time), and the
four `Dataset` wrappers that apply triggers lazily. This is the one module
that all ten attacks and both training entrypoints route through, so its
label-mode semantics are the single source of truth for what "poisoned" means
per attack family.

### Module-level state

- `ApplyTrigger` (`poison.py:20`): type alias `Callable[[torch.Tensor, int],
  torch.Tensor]`. Every attack's `apply_trigger` takes a CHW image tensor in
  `[0, 1]` (pre-normalization) and the sample's dataset index, and returns a
  perturbed CHW tensor. The index argument exists so sample-specific attacks
  (WaNet, BPP) can look up a pregenerated per-sample warp/noise; static
  attacks (BadNet, Blend) ignore it.
- `LABEL_MODES` (`poison.py:22`): `("all_to_one", "all_to_all",
  "clean_label")`, the three label-mode strings every policy function
  switches on.
- `Attack` (`poison.py:26-30`): frozen dataclass — `name: str`,
  `apply_trigger: ApplyTrigger`, `label_mode: str`, `target_label: int`. No
  methods; purely a data record built by each `attacks/*.py` module's
  `build(config)` and consumed by `poison.py`'s dataset wrappers and
  `defences/detection.py`'s eval loops.

### Functions

**`is_poisonable(label_mode, original_label, target_label) -> bool`**
(`poison.py:33-46`)
- Logic: `all_to_one` → `original_label != target_label`; `all_to_all` →
  always `True`; `clean_label` → `original_label == target_label`; anything
  else raises `ValueError`.
- Pure function, no side effects, no randomness.
- Called by: `choose_poison_indices` (`poison.py:112`),
  `choose_indices_with_cover`'s inner `is_poison_eligible`
  (`poison.py:205`). Determines training-time eligibility only — see
  `is_eval_poisonable` for the eval-time counterpart, which differs for
  `clean_label`.

**`poisoned_label(label_mode, original_label, target_label, num_classes) -> int`**
(`poison.py:49-63`)
- Logic: `all_to_one` → `target_label`; `all_to_all` → `(original_label + 1) %
  num_classes`; `clean_label` → `original_label` (no-op, since a clean-label
  attack must not change the label); else raises.
- Called by: `PoisonedTrainingSet.__getitem__` (`poison.py:140`),
  `CoverPoisonedTrainingSet.__getitem__` (`poison.py:256`), and
  `attack_success_label` (`poison.py:98`, for the non-clean-label branches).

**`is_eval_poisonable(label_mode, original_label, target_label) -> bool`**
(`poison.py:66-82`)
- Logic: identical to `is_poisonable` for `all_to_one`/`all_to_all`. For
  `clean_label` it flips to `original_label != target_label` — the opposite
  of the training-time condition.
- Why: training only poisons target-class clean-label images (the trigger
  must co-occur with the class it already implies, or the label would need to
  change). Measuring attack success asks whether the trigger fools a
  *non-target* image into being classified as the target, so eligibility for
  the eval set is the complementary condition.
- Called by: `AttackSuccessSet.__init__` (`poison.py:164`),
  `backdoor_data.py:49` (BackdoorBench PNG path, same semantics applied to a
  different data source).

**`attack_success_label(label_mode, original_label, target_label, num_classes) -> int`**
(`poison.py:85-98`)
- Logic: `clean_label` → always `target_label`; every other mode delegates to
  `poisoned_label`.
- Why not just call `poisoned_label` for `clean_label` too:
  `poisoned_label`'s `clean_label` branch returns `original_label`, which is
  only correct at training time because `is_poisonable` already restricts
  clean_label training samples to `original_label == target_label` (making
  the no-op correct by construction). At eval time `is_eval_poisonable`
  flips clean_label eligibility to `original_label != target_label`, so
  reusing `poisoned_label` there would compare against the sample's own
  original label instead of the target — the intended eval label is always
  `target_label`.
- Called by: `AttackSuccessSet.__getitem__` (`poison.py:174`),
  `backdoor_data.py:60`.

**`choose_poison_indices(labels, attack, poison_rate, seed) -> set[int]`**
(`poison.py:101-116`)
- Steps: (1) build `eligible` list of indices via `is_poisonable`; (2)
  `count = min(round(poison_rate * len(labels)), len(eligible))` — the rate is
  measured against the *whole* dataset, not the eligible pool, so for
  clean_label the realized poison fraction of the eligible pool can exceed
  `poison_rate` (capped at `len(eligible)`); (3) `np.random.default_rng(seed)`
  and `rng.choice(eligible, size=count, replace=False)`.
- Randomness: seeded via `seed` param, deterministic given the same
  `(labels, attack, poison_rate, seed)` — verified directly by
  `tests/test_attacks.py:185`.
- Called by: `train_backdoor.py:74`.

**`PoisonedTrainingSet(Dataset)`** (`poison.py:119-143`)
- `__init__(self, base_dataset, attack, poison_indices, normalize, num_classes)`
  (`poison.py:126-131`): stores references only, no copying or precompute.
- `__len__` (`poison.py:133-134`): `len(base_dataset)`.
- `__getitem__(self, index)` (`poison.py:136-143`): fetches `(image, label)`
  from `base_dataset`; if `index in poison_indices`, applies
  `attack.apply_trigger(image, index)` and relabels via `poisoned_label`;
  always returns `(self.normalize(image), label)` — normalization runs on
  every sample, poisoned or not, so poisoned images are still fed to the
  model normalized like clean ones.
- An empty `poison_indices` set makes this a plain normalized-clean-set
  wrapper, which is how `metrics.py:107` and `analysis/analyze_latent.py:70`
  reuse it to build a clean split without a separate code path.
- Called by: `train_backdoor.py:75` (poisoned train set) and `:100` (clean
  test set with empty indices), `metrics.py:107`,
  `defences/checkpoint_eval.py:72,75`, `analysis/analyze_latent.py:70-71`.

**`AttackSuccessSet(Dataset)`** (`poison.py:146-177`)
- `__init__(self, base_dataset, labels, attack, normalize, num_classes)`
  (`poison.py:156-165`): precomputes `self.indices` — every index where
  `is_eval_poisonable(attack.label_mode, labels[i], attack.target_label)`
  holds. This is the *eval-time* eligibility, not the training-time one.
- `__len__` (`poison.py:167-168`): `len(self.indices)`, i.e. only the
  eligible subset, not the full base dataset.
- `__getitem__(self, position)` (`poison.py:170-177`): maps `position` through
  `self.indices` to the real dataset `index`; fetches the image, applies the
  trigger unconditionally (every sample here gets poisoned — that's the
  point of an attack-success set), computes `target` via
  `attack_success_label`, returns `(normalize(poisoned), target)`.
- Accuracy over this dataset is exactly the attack success rate, since every
  returned label is the attack's *intended* label for that sample.
- Called by: `train_backdoor.py:104` (`backdoor_test`), `metrics.py:109`,
  `defences/checkpoint_eval.py:78`.

**`choose_indices_with_cover(labels, attack, poison_rate, cover_rate, source_classes, seed) -> tuple[set[int], set[int]]`**
(`poison.py:180-231`)
- Used only by adaptive attacks (Adaptive-Blend, TaCT) that need cover
  samples: images stamped with the trigger but *not* relabeled, which teach
  the model the trigger alone doesn't imply the target class — this is what
  flattens the latent separation that many defenses (including PSBD) rely on
  to detect backdoors.
- Steps: (1) `rng = np.random.default_rng(seed)`; (2) inner
  `is_poison_eligible(index)` — if `source_classes` is given, restrict to
  those classes (TaCT's source-specific setting), then apply `is_poisonable`;
  (3) build `poison_pool`, sample `poison_count = min(round(poison_rate *
  dataset_size), len(poison_pool))` indices into `poison_indices` (empty set
  if `poison_count == 0`, since `rng.choice` with `size=0` on some inputs
  still needs a non-empty pool otherwise — guarded explicitly); (4) inner
  `is_cover_eligible(index)` — excludes anything already in
  `poison_indices`, anything already `target_label`, and (if
  `source_classes` given) anything *in* `source_classes` (cover samples must
  come from classes the attack does not source from); (5) build `cover_pool`,
  sample `cover_count = min(round(cover_rate * dataset_size),
  len(cover_pool))` into `cover_indices`.
- Both counts are rounded and capped independently; `poison_indices` and
  `cover_indices` are guaranteed disjoint by construction (cover eligibility
  explicitly excludes poison indices).
- Called by: `train_backdoor.py:67-69`, only when the attack config carries
  `cover_rate > 0` (Adaptive-Blend, TaCT).

**`CoverPoisonedTrainingSet(Dataset)`** (`poison.py:234-261`)
- `__init__` (`poison.py:241-247`): same shape as `PoisonedTrainingSet` plus
  `cover_indices`.
- `__getitem__(self, index)` (`poison.py:252-261`): if `index in
  poison_indices` → trigger + relabel (same as `PoisonedTrainingSet`); elif
  `index in cover_indices` → trigger applied but label untouched; else
  untouched. Always normalizes before returning.
- Called by: `train_backdoor.py:70-73`, the training-set branch used
  whenever `cover_rate > 0`.

### Interactions

`poison.py` has zero internal-package imports (only `dataclasses`, `typing`,
`numpy`, `torch`); every attack module and both training entrypoints import
`Attack` from here and route dataset construction through the two (or four,
with cover) wrapper classes. `defences/checkpoint_eval.py` and `metrics.py`
rebuild the same eval sets from a saved checkpoint's `args.json` metadata
using these same classes, which is how eval-time reconstruction stays
consistent with training-time construction without re-running training.

### Sharp edges

- Changing `is_eval_poisonable`'s `clean_label` branch to match
  `is_poisonable` (i.e. `original_label == target_label`) would silently
  break ASR measurement for SIG and LC: the eval set would then only contain
  already-target-class images, where "does the trigger flip the label" is
  meaningless (it's already the target), collapsing ASR to a near-useless
  number instead of measuring whether the trigger fools non-target images.
- `attack_success_label`'s `clean_label` branch must stay hardcoded to
  `target_label`, not delegate to `poisoned_label` — delegating would return
  the sample's own original (non-target) label as the expected label,
  making every clean-label eval example count as "attack failed" even when
  the model was fooled.
- `choose_poison_indices` measures `poison_rate` against `len(labels)` (the
  whole dataset), not `len(eligible)`. For `clean_label` attacks where the
  eligible pool is only the target class, this means the *realized* fraction
  of poisoned samples within the target class can be far higher than
  `poison_rate` — silently, since the count is capped rather than raising.
- `choose_indices_with_cover`'s cover pool excludes `source_classes` samples
  entirely (not just the poisoned ones) — a cover sample is guaranteed to
  come from a class the attack never sources from, otherwise TaCT's
  source-specific claim (only `source_classes` carry the trigger→target
  association) would be violated by cover samples from a source class.
- `Attack` is `frozen=True` — attempting to mutate `target_label` etc. after
  construction raises `dataclasses.FrozenInstanceError` rather than silently
  no-op-ing.

---

## `utils/config.py`

Purpose: the dataset registry (per-dataset normalization stats and loader
routing) and `RunConfig`, the parameter bag for a PSBD detection sweep
(currently only consumed by the archived sweep code, but the dataclass itself
is live). Also owns the two BackdoorBench folder-name parsing helpers used
when a checkpoint has no `args.json` sidecar.

### Module-level state

- `DatasetSpec` (`utils/config.py:5-12`): frozen dataclass — `num_classes:
  int`, `mean: tuple[float,float,float]`, `std: tuple[float,float,float]`,
  `loader_kind: str` (one of `"cifar10"`, `"cifar100"`, `"gtsrb"`,
  `"image_folder"`).
- `DATASET_REGISTRY` (`utils/config.py:17-28`): `dict[str, DatasetSpec]` for
  `"cifar10"`, `"cifar100"`, `"gtsrb"`, `"tiny"`. GTSRB uses identity
  normalization (`mean=(0,0,0)`, `std=(1,1,1)`) deliberately, to match
  BackdoorBench's own unnormalized GTSRB training and avoid a train/eval
  mismatch when evaluating their checkpoints.
- `DROPOUT_PLACEMENTS` (`utils/config.py:35`): `("pre_residual",
  "post_residual")`.
- `ARCHITECTURES` (`utils/config.py:37`): `("vit", "swin")`.
- `RunConfig` (`utils/config.py:40-73`): mutable dataclass (not frozen) with
  defaults for a PSBD sweep: `seed=0`, `trigger_label=0`, `batch_size=16`,
  `clean_val_size=2000` (matches the paper's 5% of CIFAR-10's 50000-image
  train set), `examples_per_class=150`, `forward_passes=3` (k in the PSBD
  paper), `dropout_rates` (0.1 through 0.9 step 0.1), `psbd_quantiles=(0.1,
  0.15, 0.2, 0.25)`, `architecture="vit"`, `dropout_placement="pre_residual"`,
  `weights_dir="backdoor_bench_checkpoints"`, `raw_data_dir="raw_data"`,
  `results_root="experiments"`, `use_bfloat16=True`,
  `attack_folders=()` (via `field(default_factory=tuple)`).
  - `use_bfloat16=True` runs the forward pass under bfloat16 autocast but
    keeps score arithmetic in float32 — a deliberate fix noted in the
    comment (`utils/config.py:64-68`) replacing a prior global
    `torch.set_default_dtype(bfloat16)` that had silently downcast quantile
    thresholds and numpy round-trips.
  - `results_dir(self) -> str` method (`utils/config.py:72-73`): returns
    `os.path.join(results_root, dropout_placement)`.

### Functions

**`dataset_name_from_folder(folder_name: str) -> str`** (`utils/config.py:76-81`)
- `folder_name.split("_")[0]`. Example: `"cifar10_wanet_0_1"` → `"cifar10"`.
- No callers found outside this module in the current tree (grep found no
  external call sites); likely used by the archived sweep code.

**`CLEAN_LABEL_ATTACK_TOKENS`** (`utils/config.py:86`): `("sig", "lc")`.

**`label_mode_from_folder(folder_name: str) -> str`** (`utils/config.py:89-100`)
- Splits `folder_name` on `_`; if any token is in
  `CLEAN_LABEL_ATTACK_TOKENS` → `"clean_label"`; elif `"a2a"` substring is
  present in the full folder name → `"all_to_all"`; else →
  `"all_to_one"` (BackdoorBench's dirty-label default).
- Called by `defences/checkpoint_eval.py` for `backdoor_bench_checkpoints/`
  folders that carry no `args.json`, where folder-name parsing is the only
  available source of label-mode metadata.

### Interactions

Zero internal imports. `RunConfig` and `DATASET_REGISTRY` are imported
directly (`from utils.config import DATASET_REGISTRY`, per project
convention — `utils/__init__.py` re-exports nothing) by `utils/datasets.py`,
`backdoor_data.py`, `metrics.py`, `train_backdoor.py`, `train_benign.py`,
`defences/checkpoint_eval.py`, `analysis/analyze_latent.py`.

### Sharp edges

- `label_mode_from_folder` checks the `"a2a"` substring against the *whole*
  folder name, not tokenized, so any folder name containing `"a2a"` anywhere
  (not just as a delimited token) is classified all_to_all. In practice
  BackdoorBench folder names are clean enough for this not to matter, but
  it's a substring check, not a token check.
- The clean-label check *does* tokenize (`token in
  CLEAN_LABEL_ATTACK_TOKENS`), so a folder name would need `sig` or `lc` as
  an exact underscore-delimited token, not merely a substring — asymmetric
  with the `a2a` check right above it.
- GTSRB's identity normalization is a correctness-critical special case, not
  an oversight: swapping in real per-channel stats for GTSRB would silently
  mismatch every BackdoorBench GTSRB checkpoint's expected input
  distribution, since those checkpoints were trained on unnormalized inputs.

---

## `models.py`

Purpose: build fresh ViT-B/16 / Swin-S classifiers with ImageNet-pretrained
backbones and a re-headed classifier layer, and load BackdoorBench-format
checkpoints (`{"model": state_dict, "num_classes": int}`) into them. Also
resolves which architecture a checkpoint belongs to purely from its
state_dict keys, for folders whose name gives no hint.

### Functions

**`_wrap_with_resize(network: nn.Module) -> nn.Module`** (`models.py:23-25`)
- `nn.Sequential(transforms_v2.Resize((224, 224)), network)`. Private helper;
  lets the wrapped model accept any input spatial size (e.g. CIFAR's 32×32)
  by resizing to ViT/Swin's expected 224×224 as the first module in the
  Sequential, so it runs as part of the forward pass, not as a separate
  preprocessing step.
- Called by `build_vit` (`models.py:31`), `build_swin` (`models.py:37`).

**`build_vit(num_classes: int) -> nn.Module`** (`models.py:28-31`)
- `vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)`; replaces
  `network.heads.head` with `nn.Linear(in_features, num_classes)`; wraps with
  `_wrap_with_resize`.
- Downloads/loads pretrained ImageNet1K weights via torchvision (network
  side effect on first call, cached by torchvision afterward).
- Returns an `nn.Sequential(Resize, ViT)`, not a bare `VisionTransformer` —
  significant for `defences/dropout.py` and `vit_core`, which must unwrap it.
- Called by `_load_checkpoint_into` (`models.py:47`, as the `builder` arg
  from `load_vit_checkpoint`), `train.py` (fresh-model construction for
  training).

**`build_swin(num_classes: int) -> nn.Module`** (`models.py:34-37`)
- Same pattern for `swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)`, re-heading
  `network.head`.
- Called by `_load_checkpoint_into` (via `load_swin_checkpoint`), `train.py`.

**`_strip_dataparallel_prefix(state_dict: dict) -> dict`** (`models.py:40-42`)
- `{key.replace("module.", "", 1): value for key, value in
  state_dict.items()}` — strips at most one leading `"module."` per key
  (DataParallel's checkpoint prefix).
- Called by `_load_checkpoint_into` (`models.py:52`).

**`_load_checkpoint_into(builder, checkpoint_path, device) -> nn.Module`**
(`models.py:45-60`)
- Steps: (1) `torch.load(checkpoint_path, map_location="cpu",
  weights_only=False)` — full unpickling, not the safer weights-only mode;
  (2) `model = builder(checkpoint["num_classes"])`; (3)
  `state_dict = checkpoint.get("model", checkpoint)` — falls back to treating
  the whole loaded object as the state dict if there's no `"model"` key; (4)
  if that value is itself an `nn.Module` (not a plain dict), calls
  `.state_dict()` on it; (5) strips DataParallel prefixes; (6)
  `model.load_state_dict(state_dict, strict=False)` — mismatches are
  tolerated, not raised; (7) prints counts of missing/unexpected keys to
  stdout if any (side effect: console I/O, not a raised error or logged
  warning); (8) `model.to(device).eval()`.
- I/O: reads `checkpoint_path` from disk. Side effect: `print()` calls on
  key mismatches — silent otherwise.
- Called by `load_vit_checkpoint` (`models.py:64`), `load_swin_checkpoint`
  (`models.py:68`).

**`load_vit_checkpoint(checkpoint_path, device) -> nn.Module`** (`models.py:63-64`)
/ **`load_swin_checkpoint(checkpoint_path, device) -> nn.Module`** (`models.py:67-68`)
- Thin wrappers passing `build_vit`/`build_swin` as `builder`.

**`load_checkpoint(architecture, checkpoint_path, device) -> nn.Module`**
(`models.py:71-76`)
- Dispatches on `architecture` string (`"vit"` / `"swin"`), else raises
  `ValueError`.
- Called by `defences/checkpoint_eval.py`, `metrics.py`,
  `analysis/analyze_latent.py`, `analysis/lipschitz.py` (indirectly, via
  whatever loads the model before calling `vit_core`).

**`VIT_STATE_DICT_MARKERS`** (`models.py:82`): `("conv_proj", "class_token",
"encoder.layers.encoder_layer_")`. **`SWIN_STATE_DICT_MARKERS`**
(`models.py:83`): `("features.",)`.

**`detect_architecture(checkpoint_path: str) -> str`** (`models.py:86-97`)
- Loads the checkpoint (again a full `torch.load`, independent load from
  whatever else touches this path), inspects `checkpoint["model"].keys()`
  for the marker substrings above. Requires exactly one of `is_vit`/`is_swin`
  to be `True`; raises `ValueError` with both booleans in the message
  otherwise (covers both "neither matched" and "both matched" failure
  modes).
- Used by `scratch/normalize_checkpoints.py` (already-run migration, now
  archived in `scratch/`) to infer architecture for folders whose name gave
  no hint — not on the hot path of any currently-tracked entrypoint.

**`vit_core(model: nn.Module) -> nn.Module`** (`models.py:100-102`)
- `model[1] if isinstance(model, nn.Sequential) else model` — unwraps the
  `Sequential(Resize, network)` wrapper `build_vit`/`load_vit_checkpoint`
  produce, falling back to returning `model` unchanged if it's already a bare
  network (defensive against being called on an unwrapped model).
- Called by `analysis/features.py:extract_layer_features`,
  `analysis/lipschitz.py` — both need direct access to the ViT's internal
  blocks/encoder, which the Resize wrapper hides at index 0.

### Interactions

No internal-package imports; only `torch`, `torch.nn`,
`torchvision.transforms.v2`, `torchvision.models`. This is the shared model
construction/loading surface every training script, eval script, and latent
analysis tool imports directly (`from models import load_checkpoint`, etc.),
per the CLAUDE.md convention that `models.py` stays at the repo root because
it's used the same way from everywhere, not owned by one package.

### Sharp edges

- `load_state_dict(..., strict=False)` in `_load_checkpoint_into` means a
  checkpoint saved for a *different* architecture (e.g. loading a Swin
  checkpoint through `load_vit_checkpoint`) would load silently (0 or near-0
  matching keys) rather than raising — only the printed missing/unexpected
  key counts would reveal it, and nothing enforces the caller reads stdout.
  Always pair with `detect_architecture` when the architecture is not known
  from `args.json`.
- `vit_core` and any Swin equivalent are not symmetric: there is no
  `swin_core` function. Code that needs to descend into a Swin model's
  internals (e.g. `defences/dropout.py`) must know to skip this helper and
  index the Sequential itself, or operate on `model.named_modules()`
  directly.
- `torch.load(..., weights_only=False)` is used for both loading and
  architecture detection — this executes arbitrary pickled Python objects
  from the checkpoint file; safe only because these checkpoints are trusted
  local BackdoorBench/local-training outputs, not untrusted downloads.

---

## `sam.py`

Purpose: `SAM`, a `torch.optim.Optimizer` wrapper implementing Sharpness-Aware
Minimization (Foret et al. 2020) over any base optimizer (used with AdamW per
CLAUDE.md's "SAM is always SAM-on-top-of-AdamW" convention). Two-pass update:
ascend to a local worst-case point, then let the base optimizer step using the
gradient computed there.

### Functions

**`SAM.__init__(self, params, base_optimizer_cls, rho=0.1, adaptive=False, **base_kwargs)`**
(`sam.py:26-33`)
- Validates `rho >= 0.0` (raises `ValueError` otherwise). Builds `defaults =
  dict(rho=rho, adaptive=adaptive, **base_kwargs)`, calls
  `Optimizer.__init__(params, defaults)`. Constructs
  `self.base_optimizer = base_optimizer_cls(self.param_groups,
  **base_kwargs)` — the base optimizer is built over SAM's own param groups.
  Then **replaces** `self.param_groups` with
  `self.base_optimizer.param_groups` (so subsequent mutation goes through
  the base optimizer's groups) and merges
  `self.base_optimizer.defaults` into `self.defaults`.
- `rho` default 0.1 matches CLAUDE.md's stated project-wide SAM rho
  convention (though actual training runs pass rho explicitly per
  checkpoint's `sam_rho_*` folder tag).

**`SAM.first_step(self, zero_grad=False) -> None`** (`sam.py:36-53`, decorated
`@torch.no_grad()`)
- original form: `eps_hat = rho * grad(L(w)) / ||grad(L(w))||` (standard SAM)
  or the ASAM-adaptive per-parameter-scaled variant.
- Steps: (1) `gradient_norm = self._gradient_norm()`; (2) for each param
  group, `scale = rho / (gradient_norm + 1e-12)`; (3) for each parameter with
  a non-`None` `.grad`: save `self.state[parameter]["original"] =
  parameter.data.clone()` (state mutation — this is how `second_step`
  restores weights later); compute `per_parameter = parameter**2 if adaptive
  else 1.0`; `parameter.add_(per_parameter * parameter.grad * scale)` — the
  in-place ascent step.
- If `zero_grad=True`, calls `self.zero_grad()` after ascending.
- Side effects: mutates `parameter.data` in place, writes into
  `self.state[parameter]`.
- Called by: `train.py`'s training step, whenever `optimizer` is a `SAM`
  instance (guarded by an `isinstance` or similar check — see `train.py`
  section below).

**`SAM.second_step(self, zero_grad=False) -> None`** (`sam.py:56-69`,
`@torch.no_grad()`)
- Steps: (1) for each param with non-`None` `.grad`, restore
  `parameter.data = self.state[parameter]["original"]` (undoes the
  `first_step` ascent); (2) `self.base_optimizer.step()` — applies the real
  update using the gradient computed at the ascended point (the caller must
  have re-run forward+backward between `first_step` and `second_step` for
  this gradient to be the worst-case one, not the original); (3) optional
  `zero_grad()`.
- This is the two-pass update CLAUDE.md's "SAM two-pass" correctness rule
  refers to: `first_step` then a fresh forward/backward at `w + eps_hat`
  (done by the caller, not by `sam.py`) then `second_step`.

**`SAM._gradient_norm(self) -> torch.Tensor`** (`sam.py:71-81`,
`@torch.no_grad()`, private)
- `reference_device = self.param_groups[0]["params"][0].device`. For every
  parameter with a grad, `weighting = |parameter| if adaptive else 1.0`,
  appends `(weighting * parameter.grad).norm(p=2).to(reference_device)`.
  Returns the global L2 norm of the stacked per-parameter norms:
  `torch.norm(torch.stack(per_parameter_norms), p=2)`.
- Called by `first_step` (`sam.py:43`).

**`SAM.load_state_dict(self, state_dict) -> None`** (`sam.py:83-85`)
- Calls `super().load_state_dict(state_dict)` then re-points
  `self.param_groups = self.base_optimizer.param_groups` — necessary because
  the base `Optimizer.load_state_dict` would otherwise leave
  `self.param_groups` desynced from the base optimizer's own groups after a
  checkpoint restore.

### Interactions

No internal-package imports, only `torch`. Consumed exclusively by
`train.py`'s training loop, which is itself called by `train_backdoor.py` and
`train_benign.py`. `sam.py` never touches `rho` values from `args.json`
directly — the rho tag encoded in a checkpoint folder name
(`sam_rho_0_15` etc.) is a `train.py`/naming-convention concern, not
something this file reads.

### Sharp edges

- `first_step` and `second_step` **must** be called in strict alternation
  with a fresh forward+backward pass in between (recompute loss and call
  `.backward()` again at the ascended weights before `second_step`).
  Calling `second_step` without an intervening backward pass at `w + eps_hat`
  silently applies the base optimizer update using the *original* gradient,
  defeating the entire point of SAM without raising any error — this is a
  caller discipline requirement, not something `sam.py` enforces.
  `train.py`'s two-pass step is the only place this is done correctly; any
  future training loop copy-pasting the optimizer must replicate the
  double-backward pattern.
- `first_step`'s ascent skips any parameter whose `.grad is None` — frozen
  or not-yet-backward'd parameters are silently left untouched rather than
  raising, which is correct for partial fine-tuning but would silently mask
  a bug where gradients were never populated at all (e.g. forgetting
  `.backward()` before `first_step`).
- `second_step` restores `parameter.data` only for parameters with a
  (non-`None`) `.grad` at that point — if new parameters gained a grad
  between `first_step` and `second_step` that didn't have one during
  `first_step` (unusual, but possible with dynamic freezing), their
  `self.state[parameter]["original"]` key would be missing and this raises
  `KeyError`, not a graceful message.

---

## `attacks/` package

Purpose: ten trigger implementations sharing one `build(config, image_size,
target_label) -> Attack` interface (`poison.py:Attack`), plus a registry
(`attacks/__init__.py`) that dispatches by name string. Every `apply_trigger`
closure operates on a single CHW tensor in `[0, 1]`, pre-normalization, per
`poison.py`'s module contract. Each attack module also defines a frozen
config dataclass holding its own hyperparameters (patch size, blend alpha,
etc.) plus `label_mode`, which is read out into the resulting `Attack`.

### `attacks/__init__.py` — the registry

- `_ATTACKS` (`attacks/__init__.py:36-49`): `dict[str, tuple[builder_fn,
  config_factory_or_None]]`, one entry per registered attack name:
  `"badnet"`/`"badnet_a2o"` (alias) and `"badnet_a2a"` both map to
  `badnet.build` but with different config factories
  (`_badnet_all_to_one`/`_badnet_all_to_all`, `attacks/__init__.py:26-31`,
  each just constructing `badnet.BadNetConfig(label_mode=...)`); `"blend"`,
  `"sig"`, `"wanet"`, `"lf"`, `"lc"`, `"bpp"`, `"adaptive_blend"`, `"tact"`
  map name directly to their module's `build`/`<Name>Config` class (used
  as a zero-arg factory since every config field has a default); `"generated"`
  maps to `generated.build` with factory `None`, since `GeneratedConfig`
  requires `poisoned_dir` with no sensible default.
- `ATTACK_NAMES = tuple(_ATTACKS)` (`attacks/__init__.py:51`): the 12
  registered name strings (`badnet`, `badnet_a2o`, `badnet_a2a`, `blend`,
  `sig`, `wanet`, `lf`, `lc`, `bpp`, `adaptive_blend`, `tact`, `generated`).
- **`default_config(attack_name: str)`** (`attacks/__init__.py:54-58`): looks
  up the factory, raises `ValueError` if it's `None` (the `"generated"`
  case), else calls it with no args.
- **`build_attack(attack_name, config, image_size, target_label) -> Attack`**
  (`attacks/__init__.py:61-62`): `_ATTACKS[attack_name][0](config,
  image_size, target_label)` — dispatches to the module-level `build`
  function of whichever attack `attack_name` names, ignoring the config
  factory entirely (the caller supplies `config` explicitly, which may or
  may not be what `default_config` would have produced).
- Called by: `train_backdoor.py` (`build_attack` at line 91, `default_config`
  via `resolve_config` at line 54), `defences/checkpoint_eval.py`,
  `analysis/analyze_latent.py`, `metrics.py`.
- Attacks intentionally **not** registered here (per module docstring,
  `attacks/__init__.py:1-9`): generator-coupled attacks (Input-aware, LIRA)
  that need a co-trained generator and a bespoke training loop, out of scope
  for this registry's `build(config, image_size, target_label)` signature.

### `attacks/badnet.py` — BadNet (Gu et al. 2017)

- `BadNetConfig` (`attacks/badnet.py:15-17`): `patch_size: int = 3`,
  `label_mode: str = "all_to_one"` (use `"all_to_all"` for the BadNet-A2A
  variant, wired through the registry's `badnet_a2a` alias rather than
  changed here).
- `_checkerboard(patch_size) -> torch.Tensor` (`attacks/badnet.py:20-25`):
  builds a `(3, patch_size, patch_size)` tensor, `1.0` where `(row + column)
  % 2 == 0` else `0.0` — a checkerboard pattern, replicated identically
  across all 3 channels.
- `build(config, image_size, target_label) -> Attack`
  (`attacks/badnet.py:28-37`): precomputes the patch once; `apply_trigger`
  clones the image and overwrites the bottom-right `patch_size × patch_size`
  block (`stamped[:, image_size-size:, image_size-size:] = patch`) — static,
  ignores the `_index` argument.

### `attacks/blend.py` — Blend (Chen et al. 2017)

- `BlendConfig` (`attacks/blend.py:16-19`): `alpha: float = 0.2` (blend
  ratio; literature range 0.1–0.2), `pattern_seed: int = 0`, `label_mode:
  str = "all_to_one"`.
- `_random_pattern(image_size, seed)` (`attacks/blend.py:22-24`): a
  `torch.Generator().manual_seed(seed)`-seeded uniform `(3, image_size,
  image_size)` tensor — a seeded random pattern replaces the original
  paper's fixed Hello Kitty image, to keep the repo free of bundled image
  assets.
- `build(...)` (`attacks/blend.py:27-35`): `apply_trigger` computes `(1 -
  alpha) * image + alpha * pattern` — original form given directly in the
  inline comment, matching CLAUDE.md's "formulas: original form then
  simplified" rule (here they coincide since the formula is already
  simple). Static, ignores `_index`.

### `attacks/sig.py` — SIG (Barni et al. 2019)

- `SigConfig` (`attacks/sig.py:16-19`): `amplitude: float = 0.1` (0-to-1
  pixel units), `frequency: float = 6.0`, `label_mode: str = "clean_label"`
  — the default is clean_label by design (SIG's defining property).
- `_column_signal(image_size, amplitude, frequency)`
  (`attacks/sig.py:22-27`): original form `v(i,j) = amplitude * sin(2*pi*
  frequency*j/width)` (docstring comment gives this explicitly); computed as
  one value per column, `signal.view(1, 1, image_size)` broadcasts over rows
  and all 3 channels identically — the trigger is a horizontal sinusoid
  constant along each column, varying across columns.
- `build(...)` (`attacks/sig.py:30-36`): `apply_trigger` returns `(image +
  signal).clamp(0, 1)`. Static.

### `attacks/lf.py` — LF, low-frequency additive trigger (Zeng et al. 2021)

- `LowFrequencyConfig` (`attacks/lf.py:17-21`): `strength: float = 0.1`,
  `cutoff: int = 4` (frequency-domain keep-radius around spectrum center),
  `pattern_seed: int = 0`, `label_mode: str = "all_to_one"`.
- `_low_frequency_pattern(image_size, cutoff, seed)`
  (`attacks/lf.py:24-35`): (1) seeded uniform noise in `[-1, 1]`; (2)
  `torch.fft.fft2` then `fftshift` to center the spectrum; (3) build a
  square binary mask of half-width `cutoff` around the spectrum center,
  zeroing everything outside; (4) `ifftshift` + `ifft2`, take `.real`; (5)
  normalize by the max absolute value (`clamp_min(1e-8)` guards
  division-by-zero) so the pattern sits in `[-1, 1]`.
- `build(...)` (`attacks/lf.py:38-45`): `apply_trigger` returns `(image +
  strength * pattern).clamp(0, 1)`. Static. Docstring notes LF tends to
  reach low ASR on ViT ("treat it as a stress case") and that the exact
  BackdoorBench trigger can instead be served via `attacks/generated.py`.

### `attacks/lc.py` — Label-Consistent (Turner et al. 2019)

- `LabelConsistentConfig` (`attacks/lc.py:22-24`): `patch_size: int = 3`,
  `label_mode: str = "clean_label"`.
- `_corner_pattern(patch_size)` (`attacks/lc.py:27-32`): identical
  checkerboard construction to `badnet.py:_checkerboard` (independently
  duplicated, not shared).
- `build(...)` (`attacks/lc.py:35-48`): `apply_trigger` stamps the pattern
  into **all four corners** (top-left, top-right, bottom-left,
  bottom-right), unlike BadNet's single bottom-right corner. Static.
- Docstring notes the full attack requires an offline adversarial/GAN
  perturbation of the base target images before applying this patch (not
  implemented here); using the patch alone on unperturbed images is
  explicitly called "the weaker self-contained variant." Perturbed bases can
  be supplied via `attacks/generated.py` instead to reproduce the full
  attack.

### `attacks/bpp.py` — BppAttack (Wang et al. 2022)

- `BppConfig` (`attacks/bpp.py:22-25`): `bit_depth: int = 3` (gives `2**3 =
  8` levels per channel), `dither: bool = False`, `label_mode: str =
  "all_to_one"`.
- `_quantize(image, levels)` (`attacks/bpp.py:28-31`): `round(image *
  (levels-1)) / (levels-1)` — rounds each pixel to the nearest of `levels`
  evenly spaced values.
- `_floyd_steinberg(channel, levels)` (`attacks/bpp.py:34-52`):
  error-diffusion dithering over a single `(H, W)` channel, sequential
  per-pixel Python loop (`for row ... for column ...`), diffusing the
  quantization error to the right (7/16), bottom-left (3/16), bottom (5/16),
  bottom-right (1/16) neighbors — standard Floyd–Steinberg weights.
  `.item()` calls per pixel mean this runs on CPU scalars, not vectorized.
- `build(...)` (`attacks/bpp.py:55-65`): if `not dither`, applies
  `_quantize` to the whole tensor at once (fast, vectorized); if `dither`,
  runs `_floyd_steinberg` independently per channel and `torch.stack`s the
  results (slow, sequential — module docstring explicitly says dithering is
  **off by default** to keep dataset construction fast, "especially on Tiny
  ImageNet where it runs every epoch").

### `attacks/adaptive_blend.py` — Adaptive-Blend (Qi et al. 2023)

- `AdaptiveBlendConfig` (`attacks/adaptive_blend.py:22-26`): `alpha: float =
  0.2`, `cover_rate: float = 0.01`, `pattern_seed: int = 0`, `label_mode:
  str = "all_to_one"`.
- Trigger construction (`_random_pattern`,
  `attacks/adaptive_blend.py:29-31`) and `apply_trigger`
  (`attacks/adaptive_blend.py:38-40`) are **identical logic to
  `attacks/blend.py`** — same blend formula, same seeded-random-pattern
  generation, independently duplicated rather than imported/shared.
- The distinguishing field is `cover_rate`, read by
  `train_backdoor.py:build_training_set` via `getattr(config, "cover_rate",
  0.0)` to decide whether to route through
  `poison.py:choose_indices_with_cover`/`CoverPoisonedTrainingSet` instead of
  the plain poisoning path — this file itself has no cover-sample logic; it
  only carries the config field that downstream code reads.
- Docstring notes the paper's asymmetric trigger (fewer blend cells at train
  time than test time) is omitted here for simplicity.

### `attacks/tact.py` — TaCT (Tang et al. 2021)

- `TactConfig` (`attacks/tact.py:21-25`): `patch_size: int = 3`,
  `source_classes: tuple[int, ...] = (1,)` (which classes the trigger is
  allowed to flip to the target), `cover_rate: float = 0.01`, `label_mode:
  str = "all_to_one"`.
- `_patch`/`apply_trigger` (`attacks/tact.py:28-43`): identical
  checkerboard-in-bottom-right-corner logic to `attacks/badnet.py`,
  independently duplicated again.
- Like `adaptive_blend.py`, the source-specific behavior lives entirely in
  `train_backdoor.py`/`poison.py`, which read `config.source_classes` and
  `config.cover_rate` via `getattr` — this module only supplies the trigger
  function and the two config fields.
- Docstring explicitly flags an eval-time caveat: TaCT's ASR is
  conventionally measured only on source-class images, but the general
  `AttackSuccessSet` measures over all non-target-eligible images — callers
  must filter to `source_classes` themselves if they want the conventional
  TaCT ASR number; nothing in the codebase currently does this filtering
  automatically.

### `attacks/wanet.py` — WaNet (Nguyen and Tran 2021)

- `WaNetConfig` (`attacks/wanet.py:21-25`): `control_grid_size: int = 4`,
  `strength: float = 0.5`, `field_seed: int = 0`, `label_mode: str =
  "all_to_one"`.
- `_identity_grid(image_size)` (`attacks/wanet.py:28-31`): builds a
  `(1, H, W, 2)` sampling grid via `torch.meshgrid` over `linspace(-1, 1,
  image_size)` on both axes, stacked as `(columns, rows)` — the identity
  `grid_sample` grid (no warping).
- `_warping_grid(image_size, control_grid_size, strength, seed)`
  (`attacks/wanet.py:34-41`): (1) seeded random control offsets, shape `(1,
  2, control_grid_size, control_grid_size)`, in `[-1, 1]`; (2) normalize by
  `control.abs().mean()`; (3) `F.interpolate(..., size=image_size,
  mode="bicubic", align_corners=True)` upsamples the coarse control grid to
  full resolution; (4) `.permute(0, 2, 3, 1)` to `(1, H, W, 2)`; (5)
  `grid = identity_grid + strength * field / image_size`, then `clamp(-1,
  1)`.
- `build(...)` (`attacks/wanet.py:44-51`): precomputes `grid` once from the
  config's seed (so the warp field is fixed across all poisoned samples,
  not per-sample). `apply_trigger` calls `F.grid_sample(image.unsqueeze(0),
  grid, align_corners=True, padding_mode="border")` then squeezes the batch
  dim — a backward warp resampling pixel *positions*, not values, which is
  what makes the trigger visually subtle. Static (same grid for every
  image, `_index` ignored) despite WaNet being a "sample-specific" attack
  family in the broader literature (some WaNet variants use per-sample
  noise-mode augmentation; this implementation is the single-fixed-field
  variant).
- Docstring caveat: control-offset normalization follows the paper "at a
  level of fidelity sufficient to produce a working attack," not
  guaranteed to exactly match any specific benchmark's saved warp field.

### `attacks/generated.py` — adapter for pregenerated per-sample triggers

- `GeneratedConfig` (`attacks/generated.py:25-28`): `poisoned_dir: str`
  (required, no default), `name: str = "generated"`, `label_mode: str =
  "all_to_one"`.
- `_index_to_path(poisoned_dir) -> dict[int, str]`
  (`attacks/generated.py:31-33`): globs `f"{poisoned_dir}/**/*.png"`
  recursively, builds `{int(Path(path).stem): path}` — expects filenames to
  be bare integers (e.g. `42.png`), matching dataset index `42`'s poisoned
  counterpart.
- `build(...)` (`attacks/generated.py:36-48`): builds the index→path map
  once; `to_tensor = Compose([Resize((image_size,image_size)),
  ToTensor()])`. `apply_trigger(_image, index)` **ignores its `_image`
  argument entirely** — opens `index_to_path[index]` from disk, converts to
  RGB, resizes, and returns the tensor. This is the one attack whose trigger
  function performs file I/O and can raise `KeyError` if `index` has no
  corresponding PNG (no fallback or error message beyond the raw
  `KeyError`).
- Purpose per docstring: adapts sample-specific, generator-produced attacks
  (SSBA) and per-model-optimized attacks (TrojanNN) that are impractical to
  reproduce from scratch, by serving BackdoorBench's already-generated
  poisoned images through the same `Attack` interface.

### Interactions

All ten attack modules import only `poison.Attack` (absolute import, per
CLAUDE.md's cross-package-boundary convention) plus `torch`/`dataclasses`;
none import each other or any other project module. `attacks/__init__.py` is
the only file that imports all of them, via a single relative `from . import
(...)` (`attacks/__init__.py:11-22`), and re-exports `ATTACK_NAMES`,
`default_config`, `build_attack` as the package's public surface — external
callers (`train_backdoor.py`, `metrics.py`, `defences/checkpoint_eval.py`,
`analysis/analyze_latent.py`) only ever import from `attacks`, never reach
into `attacks.badnet` etc. directly, except `train_backdoor.py`, which
imports `attacks.generated.GeneratedConfig` directly (`train_backdoor.py:23`)
since it needs to construct that one config type explicitly (its factory is
`None` in the registry).

### Sharp edges

- **Checkerboard/corner-patch construction is duplicated three times**
  (`badnet.py:_checkerboard`, `lc.py:_corner_pattern`, `tact.py:_patch`) —
  identical code, not shared. A change to the patch generation logic (e.g.
  fixing an off-by-one) made in one file silently does not propagate to the
  other two.
- **Blend pattern generation is duplicated twice** (`blend.py`,
  `adaptive_blend.py`) — same non-sharing risk.
- All static attacks (BadNet, Blend, SIG, LF, LC, Adaptive-Blend, TaCT,
  WaNet) precompute their pattern/grid **once inside `build()`**, closed
  over by `apply_trigger`. This means the pattern is fixed for the lifetime
  of one `Attack` instance/one training run, not resampled per epoch or per
  call — changing `apply_trigger` to recompute the pattern per call would
  silently turn a static attack into a much weaker, inconsistent one (the
  model would never see a stable trigger to learn).
- `attacks/generated.py:apply_trigger` **discards its `image` argument**.
  Any caller assuming `apply_trigger(image, index)` always derives its
  output from `image` (true for all 9 other attacks) will get surprising
  behavior here — the clean image passed in is irrelevant; only `index`
  matters. Passing an out-of-range `index` (not present as a PNG in
  `poisoned_dir`) raises a bare `KeyError`, not a descriptive error.
- `tact.py`'s ASR caveat (docstring, `attacks/tact.py:8-10`) is not enforced
  anywhere in code — `AttackSuccessSet` and every eval path in this repo
  measure ASR over all `is_eval_poisonable` samples (all non-target classes
  for `all_to_one`), not filtered to `source_classes`. A TaCT ASR number
  read from `metrics.json` is therefore **not** the conventional
  source-specific TaCT ASR from the paper unless something downstream
  filters it — nothing currently does.
- `attacks/__init__.py:build_attack` takes `config` as a caller-supplied
  argument and never validates it against the attack's expected config
  type — passing e.g. a `BlendConfig` to `badnet.build` would fail with an
  `AttributeError` on the first config field access, not a clear type
  error, since `build_attack` performs no isinstance check.
- BadNet's `"badnet"` and `"badnet_a2o"` registry entries are true aliases
  (both map to the exact same `(badnet.build, _badnet_all_to_one)` tuple) —
  they are interchangeable in every respect, not just documentation.

---

## `defences/dropout.py`

Purpose: this file implements the paper's central experimental variable —
where dropout sits relative to the residual add in a transformer block. Two
placements are supported: `pre_residual` (toggle the Dropout modules ViT/Swin
already contain on their branch, before the residual add — architecture
agnostic) and `post_residual` (wrap each block so a *new* Dropout runs after
the residual add, mirroring the original ConvNet-era PSBD paper's placement —
architecture-specific, needs a bespoke wrapper per block type). Module
docstring states the empirical headline result motivating the whole repo:
pre-residual dropout separates clean/backdoor PSU on ViT-B/16, post-residual
mostly does not, plausibly explained by the CKA homogeneity of ViT residual
streams (Raghu et al. 2021).

### Module-level state

- `_SWIN_BLOCK_TYPES` (`defences/dropout.py:21-28`): built via a
  `try`/`except ImportError` — if the installed torchvision exposes
  `SwinTransformerBlockV2`, `_SWIN_BLOCK_TYPES = (SwinTransformerBlock,
  SwinTransformerBlockV2)`; on older torchvision without it, falls back to
  `(SwinTransformerBlock,)` only. Determines what
  `_wrap_as_post_residual`/isinstance checks treat as "a Swin block."

### Functions

**`configure_pre_residual_dropout(model: nn.Module, rate: float) -> int`**
(`defences/dropout.py:31-50`)
- Iterates `model.named_modules()`; for every `nn.Dropout` whose name does
  **not** end in `"encoder.dropout"`, sets `module.p = float(rate)` and
  calls `module.train()` if `rate > 0.0` else `module.eval()` (a Dropout
  only samples a mask in train mode, so eval-mode-with-rate-0 is the
  no-dropout baseline — an explicit design note in the docstring). Returns
  `count`, the number of modules touched.
- The `"encoder.dropout"` suffix exclusion specifically skips ViT's
  `Encoder.dropout` — the single embedding dropout applied once before the
  12-block stack, which runs *before any residual connection exists* and so
  is structurally not a pre-residual placement. This is CLAUDE.md's stated
  correctness rule: 36 true per-block dropouts get swept (attention branch +
  2 MLP dropouts × 12 blocks), not all 37. Swin has no module named that
  suffix, so the filter is a no-op there (verified as a structural fact, not
  Swin-specific logic).
- Side effect: mutates `module.p` and train/eval mode on every matching
  submodule in place; no return value carries the model, callers rely on the
  in-place mutation.
- Called by: `configure_dropout` (`defences/dropout.py:149`), `reset_dropout`
  (`defences/dropout.py:160`, with `rate=0.0`), directly by
  `tests/test_dropout.py`.

**`PostResidualEncoderBlock(nn.Module)`** (`defences/dropout.py:53-74`)
- `__init__(self, base_block, rate)`: wraps a ViT `EncoderBlock`, creates two
  fresh `nn.Dropout(p=rate)` instances (`attention_dropout`, `mlp_dropout`).
- `forward(self, input_tensor)` (`defences/dropout.py:62-74`): manually
  re-implements `EncoderBlock`'s forward pass instead of delegating to
  `base_block.forward`, inserting a Dropout after each residual add:
  `x = ln_1(input) → self_attention → block.dropout(x)` (the block's
  original pre-residual dropout, still applied) `→ x = input + x` (first
  residual add) `→ x = self.attention_dropout(x)` (**new** post-residual
  dropout) `→ y = ln_2(x) → mlp(y) → x = x + y` (second residual add) `→ x =
  self.mlp_dropout(x)` (second new post-residual dropout). Returns the final
  `x`.
- Shapes: `input_tensor` is the transformer's token sequence, `(batch,
  seq_len, hidden_dim)` for ViT-B/16 hidden_dim=768; output is the same
  shape.

**`PostResidualSwinBlock(nn.Module)`** (`defences/dropout.py:77-97`)
- Same wrapping pattern for a Swin block. `forward` re-implements:
  `x = input + stochastic_depth(attn(norm1(input)))` (first residual, using
  Swin's own stochastic-depth regularization on the branch, unchanged) `→ x
  = attention_dropout(x)` (new) `→ x = x + stochastic_depth(mlp(norm2(x)))`
  (second residual) `→ x = mlp_dropout(x)` (new).
- Docstring notes Swin normally regularizes via stochastic depth, not
  per-activation dropout, so the post-residual Dropout added here is a
  perturbation the model *never saw during training* — a caveat for
  interpreting Swin post-residual PSBD results.

**`_wrap_as_post_residual(module, rate)`** (`defences/dropout.py:100-114`,
private)
- Dispatch: if `module` is already a `PostResidualEncoderBlock` or
  `PostResidualSwinBlock`, re-wraps `module.base_block` at the new `rate`
  (so calling this repeatedly, e.g. across a rate sweep, doesn't nest
  wrappers — idempotent re-wrap, not accumulation); elif a bare
  `EncoderBlock` → wraps fresh; elif in `_SWIN_BLOCK_TYPES` → wraps fresh;
  else returns `None` (not a recognized block type).
- Called by `configure_post_residual_dropout` (`defences/dropout.py:139`, as
  the `make_replacement` closure passed to `_replace_blocks`).

**`_unwrap_post_residual(module)`** (`defences/dropout.py:117-120`, private)
- If `module` is either post-residual wrapper type, returns
  `module.base_block` (undoing the wrap); else `None`.
- Called by `remove_post_residual_dropout` (`defences/dropout.py:143`).

**`_replace_blocks(parent, make_replacement) -> None`** (`defences/dropout.py:123-135`,
private)
- Walks `parent.named_children()` (one level, but recurses); for each child,
  calls `make_replacement(child)`. If it returns non-`None`,
  `setattr(parent, name, replacement)` swaps the child in place; if `None`,
  recurses into that child (`_replace_blocks(child, make_replacement)`)
  instead. Recursion explicitly **stops** at a replaced block — its new
  wrapped contents are never revisited in the same call, avoiding
  double-wrapping within one pass.
- Relies on ViT/Swin's block stacks being `nn.Sequential`-like containers
  whose children are named `"0"`, `"1"`, ... — `setattr(parent, "0",
  replacement)` works because `nn.Sequential.__setattr__` supports this for
  numeric-string names.
- Called by `configure_post_residual_dropout`, `remove_post_residual_dropout`.

**`configure_post_residual_dropout(model, rate) -> None`**
(`defences/dropout.py:138-139`)
- `_replace_blocks(model, lambda module: _wrap_as_post_residual(module,
  rate))`. Mutates the model tree in place, replacing every recognized
  transformer block with its post-residual-wrapped version.
- Called by `configure_dropout` (`defences/dropout.py:151`).

**`remove_post_residual_dropout(model) -> None`** (`defences/dropout.py:142-143`)
- `_replace_blocks(model, _unwrap_post_residual)`. Restores plain
  `EncoderBlock`/Swin blocks from any wrapped ones present.
- Called by `reset_dropout` (`defences/dropout.py:159`, only when
  `placement == "post_residual"`).

**`configure_dropout(model, rate, placement) -> None`** (`defences/dropout.py:146-153`)
- Dispatches to `configure_pre_residual_dropout` or
  `configure_post_residual_dropout` based on `placement` string; raises
  `ValueError` for anything else. The package-level public entrypoint named
  in `defences/__init__.py`'s docstring example.
- No callers found in currently-tracked non-test code (the archived sweep
  presumably called it; live callers currently call the two placement
  functions directly instead — see `analysis/analyze_latent.py`, which calls
  `reset_dropout` but not `configure_dropout`).

**`reset_dropout(model, placement) -> None`** (`defences/dropout.py:156-160`)
- If `placement == "post_residual"`, first calls
  `remove_post_residual_dropout(model)` to unwrap blocks back to plain
  `EncoderBlock`/Swin-block form. Then **always** calls
  `configure_pre_residual_dropout(model, rate=0.0)` regardless of
  `placement` — this both handles the pre_residual case (setting rate to 0,
  eval mode) and, for the post_residual case, ensures any lingering
  `nn.Dropout` instances (including the block's own original pre-residual
  ones, now exposed again after unwrapping) are zeroed and set to eval,
  fully returning the model to a deterministic no-dropout inference state.
- Called by `analysis/analyze_latent.py:175`, directly by
  `tests/test_dropout.py`.

### Interactions

Imports only `torch.nn` and torchvision's `EncoderBlock`/Swin block classes
— no internal project imports. `analysis/analyze_latent.py` is the only
currently-tracked non-test caller, using `reset_dropout` to return a loaded
checkpoint to a clean deterministic state before extracting features (dropout
must be off for feature extraction to be reproducible run-to-run).
`defences/inference.py`'s `enable_dropout_modules` is a separate, simpler
function that puts *every* `nn.Dropout` into train mode without touching
`.p` — used together with whatever set the rate (`configure_pre_residual_dropout`)
in the (currently archived) PSBD sweep flow.

### Sharp edges

- The `"encoder.dropout"` name-suffix exclusion in
  `configure_pre_residual_dropout` is the exact mechanism behind CLAUDE.md's
  stated correctness rule ("skips any module named `*.encoder.dropout`... so
  it touches the 36 true per-block dropouts, not all 37"). Changing or
  removing this check would silently fold ViT's embedding dropout into the
  pre-residual sweep, contaminating the "residual-stream-untouched" premise
  the whole placement comparison rests on — this dropout runs before any
  residual stream exists.
- `_wrap_as_post_residual`'s re-wrap branch (lines 106-109) means calling
  `configure_post_residual_dropout` twice at different rates is safe and
  idempotent (rate updates in place via a fresh wrapper), but it also means
  the **original** `attention_dropout`/`mlp_dropout` submodules from the
  first wrap are discarded and replaced, not mutated — any external
  reference held to those specific submodule objects (e.g. a hook
  registered on them) would silently stop firing after a second
  `configure_post_residual_dropout` call.
- `reset_dropout` for `placement="post_residual"` unwraps first, *then*
  zeroes pre-residual dropouts — reversing the order (zeroing while still
  wrapped) would leave the wrapper's `attention_dropout`/`mlp_dropout`
  instances at their old rate, since `configure_pre_residual_dropout` only
  ever touches modules that are already `nn.Dropout` instances reachable via
  `named_modules()`, and the wrapper's own two Dropouts *are* reachable that
  way — but the base block's now-nested-inside-the-wrapper original dropout
  would also get zeroed either way, so the actual bug this order avoids is
  leaving `EncoderBlock`/Swin-block objects wrapped rather than plain after
  a reset, which several tests (`test_dropout.py:81`) explicitly assert
  against.
- `_replace_blocks`'s reliance on numeric-string child names (`"0"`, `"1"`,
  ...) is specific to how `nn.Sequential`-based containers name their
  children; a future architecture whose block container uses a plain
  `nn.ModuleList` accessed by integer indexing rather than attribute name
  would still work (ModuleList also names children by stringified index),
  but a `nn.ModuleDict` with non-numeric keys would still work too since
  `setattr` uses whatever name `named_children()` reports — the real
  constraint is just that `setattr(parent, name, replacement)` must be a
  valid way to replace that child, which holds for all of `Sequential`,
  `ModuleList`, and `ModuleDict`.

---

## `defences/inference.py`

Purpose: the forward-pass primitives PSBD's detection statistic is built
from — softmax probabilities under optional bfloat16 autocast, a no-dropout
baseline cache (computed once per eval split to avoid recomputing the
deterministic baseline for every dropout rate in a sweep), and the core PSU
(Prediction Shift Uncertainty) score plus per-split shift ratio. Implements
Equation 2 of the PSBD paper, given in both original and simplified form in
the module docstring (`defences/inference.py:3-14`), per CLAUDE.md's formula
convention:

```
original form
    phi_PSU(x) = P_c(x; theta) - (1/k) * sum_{i=1..k} P_c(x; p, theta_i')
    with c = argmax_c P(x; theta)

descriptive form
    psu(x) = prob_no_dropout(argmax_class) - mean_over_k_passes(
                 prob_with_dropout(argmax_class))
```

A low PSU means the no-dropout-predicted class's confidence barely moves
under stochastic dropout — the signal PSBD reads as "likely poisoned."

### Functions

**`_autocast_context(device, use_bfloat16)`** (`defences/inference.py:26-30`,
private)
- Returns `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` only if
  `use_bfloat16` is `True` **and** `device.type == "cuda"`; otherwise returns
  `contextlib.nullcontext()` (a no-op context manager). CPU runs never
  autocast even if `use_bfloat16=True` is requested.
- Called by `forward_probs` (`defences/inference.py:40`).

**`forward_probs(model, images, device, use_bfloat16) -> torch.Tensor`**
(`defences/inference.py:33-42`)
- Runs `model(images.to(device))` inside the autocast context, then
  `F.softmax(logits.float(), dim=1)` — the `.float()` cast happens *after*
  the forward pass, so the returned probabilities are always float32
  regardless of what dtype the forward pass itself ran under. This is the
  concrete mechanism behind `RunConfig.use_bfloat16`'s stated design intent
  (`utils/config.py:64-68`): forward-pass speed from bfloat16, score
  arithmetic precision from float32.
- Shapes: `images` is `(batch, 3, H, W)`; returns `(batch, num_classes)`
  float32 probabilities summing to 1 along dim 1.
- Called by: `defences/detection.py:57,103` (`clean_accuracy`,
  `attack_success_rate`), `analysis/features.py:77`,
  `build_baseline_cache` (`defences/inference.py:68`),
  `compute_psu_and_shift` (`defences/inference.py:104`).

**`enable_dropout_modules(model) -> None`** (`defences/inference.py:45-49`)
- Iterates `model.modules()`, calls `.train()` on every `nn.Dropout`
  instance found — puts dropout into sampling mode without touching `.p`
  (rate is assumed already set by `defences/dropout.py`'s configure
  functions). Leaves all other modules (BatchNorm, etc.) in whatever mode
  they were already in — this is not a blanket `model.train()`.
- Called by `compute_psu_and_shift` (`defences/inference.py:90`).

**`build_baseline_cache(model, loader, device, use_bfloat16) -> list[dict]`**
(`defences/inference.py:52-70`, `@torch.inference_mode()`)
- Precondition (assumed, not checked): dropout is already off (model
  in a no-dropout state) when called — the docstring states this
  explicitly.
- Steps: `model.eval()`; for each `(images, _)` batch from `loader`, compute
  `probs = forward_probs(...)`, append `{"probs": probs.cpu(), "labels":
  probs.argmax(dim=1).cpu()}` to `cache`.
- Purpose: the no-dropout baseline is deterministic and identical across
  every dropout-rate iteration in a sweep, so computing it once per split
  and reusing the cache is "the dominant cost saving across the run" (from
  the docstring) — avoids re-running a full forward pass over the whole
  split for every rate in `RunConfig.dropout_rates`.
- Shape of each cache entry: `probs` is `(batch, num_classes)` CPU float32,
  `labels` is `(batch,)` CPU int64 argmax indices.
- `@torch.inference_mode()`: disables autograd and view-tracking for the
  whole function, standard for eval-only forward passes.
- No live (non-archived) callers currently — consumed by
  `_archive/sweep.py`.

**`compute_psu_and_shift(model, loader, baseline_cache, device,
forward_passes, use_bfloat16, seed) -> tuple[torch.Tensor, float]`**
(`defences/inference.py:73-116`, `@torch.inference_mode()`)
- Steps: (1) `enable_dropout_modules(model)` then `seed_everything(seed)` —
  reseeding **after** enabling dropout, right before the stochastic passes,
  so the sampled dropout masks are reproducible; the docstring states this
  reseeding is what keeps the clean/backdoor/validation calls' dropout
  masks *identical* across calls with the same seed, which is what makes
  their comparison paired (per CLAUDE.md's "clean and backdoor eval sets
  are paired from the same images" correctness rule — this is the paired
  mechanism at the score-computation level, not just the dataset level).
  (2) For each `(images, _)` batch zipped against the corresponding
  `baseline_cache` row (`zip(loader, baseline_cache)` — **relies on
  `loader` iterating in the exact same order as when `baseline_cache` was
  built**, no shuffling allowed between the two passes): move `images` and
  the cached `baseline_probs`/`baseline_labels` to `device`; run
  `forward_passes` stochastic forward passes, each computing `probs =
  forward_probs(...)`, incrementing `shift_count` by how many predictions
  in this pass disagree with `baseline_labels`
  (`(probs.argmax(dim=1) != baseline_labels).sum().item()`), and collecting
  all `forward_passes` probability tensors into `dropout_probs`.
  (3) `mean_dropout_probs = stack(dropout_probs, dim=0).mean(dim=0)` — the
  `(1/k) * sum` term of the PSU formula. (4) `confidence_drop =
  baseline_probs - mean_dropout_probs` (full `(batch, num_classes)`
  difference, all classes). (5) `scores =
  confidence_drop.gather(1, baseline_labels.view(-1,1)).squeeze(1)` —
  selects only the entry at the no-dropout-argmax class `c`, giving the
  actual scalar-per-sample PSU: `P_c(no-dropout) - mean_k(P_c(dropout))`.
  Appends `scores.cpu()` to `per_sample_scores`.
- After the loop: `scores = torch.cat(per_sample_scores)` (or
  `torch.empty(0)` if the loader was empty); `shift_ratio = shift_count /
  max(total_pass_samples, 1)` — the fraction of (sample, dropout-pass) pairs
  whose argmax disagreed with the no-dropout baseline, averaged over the
  whole split and all `forward_passes` passes together (not per-pass).
- Returns `(scores.float(), shift_ratio)` — `scores` shape `(N,)` where `N`
  is the split size, one PSU value per sample; `shift_ratio` a Python float.
- Randomness: reseeded via `seed_everything(seed)` at the top, so repeated
  calls with the same `seed`/`model`/`loader` produce identical dropout
  masks and thus identical `scores`.
- No live (non-archived) callers currently — consumed by
  `_archive/sweep.py:129`.

### Interactions

No internal-package imports beyond `lightning.seed_everything`. Two of its
four functions (`forward_probs`, `enable_dropout_modules`) are live,
consumed by `defences/detection.py` and `analysis/features.py` today; the
other two (`build_baseline_cache`, `compute_psu_and_shift`) implement the
actual PSU statistic but are currently only exercised by the archived sweep
(`_archive/sweep.py`), pending the rewrite CLAUDE.md references. Any rewrite
of the PSBD sweep mechanism will almost certainly re-wire these two back in.

### Sharp edges

- `compute_psu_and_shift`'s `zip(loader, baseline_cache)` has **no
  order-consistency check**. If `loader`'s `DataLoader` has `shuffle=True`,
  or if the loader passed here iterates a different subset/order than the
  loader that produced `baseline_cache`, the function silently pairs the
  wrong baseline probabilities with the wrong dropout-pass images —
  `confidence_drop` and every downstream PSU score would be computed
  against a mismatched sample, with no error raised (shapes would still
  match as long as batch sizes align, only the row *content* correspondence
  is wrong). This is exactly the kind of silent-break case CLAUDE.md's
  "paired eval sets" correctness rule and this task's brief flag explicitly.
- `enable_dropout_modules` only calls `.train()` on `nn.Dropout` submodules,
  not `model.train()` globally — BatchNorm layers (if any existed in these
  architectures) would stay in eval mode, which is intentional (BatchNorm
  running-stats updates would otherwise corrupt inference-time statistics),
  but it means this function's name is narrower than "set eval mode for
  stochastic inference" might suggest — it's dropout-specific, not a general
  train-mode toggle.
- `_autocast_context` silently no-ops on CPU even when `use_bfloat16=True`
  is explicitly requested — there is no warning that the requested
  precision mode was ignored; `use_bfloat16` is a CUDA-only knob in
  practice.
- `build_baseline_cache`'s "assumes dropout is already off when called"
  precondition is enforced by convention (caller discipline), not by any
  assertion in this function — calling it while dropout is still enabled
  would silently produce a *stochastic* "baseline," defeating the entire
  cache-reuse premise, since every downstream `compute_psu_and_shift` call
  would then be comparing against a noisy rather than deterministic
  reference.

---

## `defences/detection.py`

Purpose: two distinct concerns bundled in one file per its docstring —
(1) PSBD threshold selection and detection-quality metrics (TPR/FPR/AUROC
from PSU scores), currently only reachable from the archived sweep, and
(2) general model-behavior metrics (clean accuracy, ASR, per-class accuracy)
that are live and used throughout the training/eval entrypoints. Both
concerns share `forward_probs` from `defences/inference.py` as their forward
pass primitive.

### Functions — PSBD detection metrics (archived-sweep consumers only)

**`threshold_from_validation(validation_scores, quantile) -> float`**
(`defences/detection.py:17-23`)
- `float(torch.quantile(validation_scores.float(), quantile).item())` — the
  detection threshold is a low quantile (e.g. 0.25 per CLAUDE.md's stated
  25th-percentile convention) of the clean validation set's PSU scores.
  Docstring: this needs no backdoor knowledge and directly reads as the
  tolerable false-positive rate on clean data — the quantile choice
  *mechanically sets* FPR (CLAUDE.md's stated correctness rule), since by
  construction that fraction of clean validation scores falls below the
  threshold.
- No live (non-archived) callers.

**`detection_rates(clean_scores, backdoor_scores, threshold) -> tuple[float, float]`**
(`defences/detection.py:26-34`)
- `tpr = (backdoor_scores < threshold).float().mean().item()`; `fpr =
  (clean_scores < threshold).float().mean().item()` — PSU below threshold is
  flagged as poisoned (low PSU = confidence barely shifts under dropout =
  suspicious), so TPR is the fraction of backdoor-eval scores correctly
  flagged, FPR the fraction of clean-eval scores incorrectly flagged.
- No live (non-archived) callers.

**`auroc(clean_scores, backdoor_scores) -> float`** (`defences/detection.py:37-41`)
- Builds `scores = concat([-clean_scores, -backdoor_scores])` (negated,
  since lower PSU means more likely poisoned, and `roc_auc_score` expects
  higher score = more likely positive) and `labels = concat([zeros(len
  clean)], [ones(len backdoor)])`; returns `roc_auc_score(labels, scores)` —
  threshold-free separability between clean and backdoor PSU distributions.
- No live (non-archived) callers.

### Functions — model behavior metrics (live)

**`_prediction_accuracy(model, loader, device, use_bfloat16) -> float`**
(`defences/detection.py:44-60`, private, `@torch.inference_mode()`)
- `model.eval()`; iterates `loader`, computing `predictions =
  forward_probs(...).argmax(dim=1)` per batch, accumulating `correct`/
  `total` counts; returns `correct/total` (or `0.0` if `total == 0`, guards
  divide-by-zero on an empty loader rather than raising).
- Called by `attack_success_rate` (`defences/detection.py:70`),
  `clean_accuracy` (`defences/detection.py:80`) — both are thin
  docstring-only wrappers over this shared implementation, distinguished
  purely by which loader the caller passes in.

**`attack_success_rate(model, backdoor_loader, device, use_bfloat16) -> float`**
(`defences/detection.py:63-70`)
- `_prediction_accuracy(model, backdoor_loader, ...)`. Docstring: "backdoor
  loader carries the trigger label, so accuracy on it is the ASR" — this is
  correct precisely because the loader is built from an `AttackSuccessSet`
  (or `backdoor_data.py`'s PNG equivalent), whose labels are already the
  attack's *intended* label per `poison.py:attack_success_label`, not the
  ground truth.
- Called by `metrics.py:140`, `train_backdoor.py:171`,
  `tests/test_attack_triggers.py:194,200`.

**`clean_accuracy(model, clean_loader, device, use_bfloat16) -> float`**
(`defences/detection.py:73-80`)
- `_prediction_accuracy(model, clean_loader, ...)`.
- Called by `metrics.py:141`, `train_benign.py:85`, `train.py:188`
  (validation accuracy during training), `train_backdoor.py:172`,
  `tests/test_attack_triggers.py:201` (there used to measure training-set
  memorization, not held-out accuracy — same function, different loader
  semantics supplied by the caller).

**`class_correct_and_total(model, loader, device, num_classes, use_bfloat16) -> tuple[torch.Tensor, torch.Tensor]`**
(`defences/detection.py:83-108`, `@torch.inference_mode()`)
- One forward pass over `loader`; for each batch, computes `predictions`,
  then for every `label in range(num_classes)` builds a boolean `mask =
  labels == label` and accumulates `total[label] += mask.sum()`,
  `correct[label] += (predictions[mask] == label).sum()`. Returns two
  `(num_classes,)` float tensors.
- Cost note: this is an `O(num_classes)` inner loop per batch (a mask
  computed per class), not a scatter/bincount — fine for the dataset sizes
  here (max 200 classes, Tiny ImageNet) but not the asymptotically fastest
  approach.
- Exists specifically so a caller wanting both pooled and per-class accuracy
  can compute the underlying counts once instead of running two full passes
  — explicit in the docstring, and exactly how `metrics.py:evaluate_benign`
  uses it (calls this once, then both `pooled_accuracy_from_counts` and
  `accuracy_by_class_from_counts` on the same `(correct, total)`, rather
  than calling the composed `clean_accuracy_by_class` and a separate
  `clean_accuracy` — see Sharp edges).
- Called by `metrics.py:118`, `clean_accuracy_by_class`
  (`defences/detection.py:137`).

**`accuracy_by_class_from_counts(correct, total) -> dict[int, float]`**
(`defences/detection.py:111-115`)
- `{label: (correct[label]/total[label]).item() if total[label] > 0 else
  0.0 for label in range(len(total))}` — per-class accuracy dict, keyed by
  integer class index; classes with zero samples in `total` get `0.0` rather
  than `NaN` or a `KeyError`.
- Called by `metrics.py:122`, `clean_accuracy_by_class`
  (`defences/detection.py:138`).

**`pooled_accuracy_from_counts(correct, total) -> float`**
(`defences/detection.py:118-121`)
- `(correct.sum() / total.sum()).item()` if `total.sum() > 0` else `0.0` —
  explicitly a **count-weighted (micro) average**, not a mean of per-class
  accuracies (a macro average). Comment states this matters for an
  imbalanced test set (GTSRB, whose 43 classes are far from balanced),
  where a macro average would over-weight rare classes.
- Called by `metrics.py:121`.

**`clean_accuracy_by_class(model, clean_loader, device, num_classes, use_bfloat16) -> dict[int, float]`**
(`defences/detection.py:124-138`)
- Composes `class_correct_and_total` then `accuracy_by_class_from_counts` —
  the convenience wrapper the individual pieces exist to support. Docstring:
  reported *alongside* pooled `clean_accuracy`, not instead of it, since
  overall accuracy can hide a class the model never learned at all.
- **No live callers** — `metrics.py:evaluate_benign` inlines the same two
  calls itself (using the shared `class_correct_and_total` result for both
  pooled and per-class figures in one pass) rather than calling this
  composed function, which would require a second full pass to also get the
  pooled figure via a separate `clean_accuracy` call. This function is
  effectively superseded by `metrics.py`'s own composition, though it
  remains a valid public entrypoint (exercised by `tests/test_detection.py`).

### Interactions

Imports `.inference.forward_probs` (relative, within-package) and
`sklearn.metrics.roc_auc_score`. `train_backdoor.py`, `train_benign.py`,
`train.py`, and `metrics.py` all import from this module directly
(`from defences.detection import ...`), never reaching into
`defences.inference` themselves for accuracy computation — this module is
the accuracy/ASR-facing surface, `defences/inference.py` is the
probability/PSU-facing surface underneath it.

### Sharp edges

- `threshold_from_validation`/`detection_rates`/`auroc` are dead in current
  live code (archived-sweep-only) — CLAUDE.md's PSBD threshold correctness
  rule ("2000-sample clean validation set, threshold at the 25th percentile
  quantile... FPR is mechanically set by the quantile choice") describes
  what these three functions implement together, and that rule "stands for
  whatever replaces it" per CLAUDE.md, meaning a rewrite is expected to
  either reuse these three functions as-is or reimplement equivalent logic
  here.
- `pooled_accuracy_from_counts` vs. a naive `mean(accuracy_by_class_from_counts(...).values())`
  give **different numbers** on any imbalanced dataset (GTSRB, Tiny
  ImageNet) — this is deliberate (the comment at
  `defences/detection.py:119-120` states it explicitly), but a future
  caller reaching for "the" accuracy from a per-class dict via a naive mean
  would silently compute the macro average instead of the pooled/micro one
  this codebase treats as canonical.
- `_prediction_accuracy` and `class_correct_and_total` independently
  iterate the same kind of loader and independently call `forward_probs` —
  there is no shared "run predictions once, feed rates in" abstraction
  layer above `forward_probs` itself; `metrics.py:evaluate_benign`'s
  choice to call `class_correct_and_total` directly and derive both
  pooled and per-class from it (rather than calling `clean_accuracy` +
  `clean_accuracy_by_class`, which would each do a separate pass) is the
  one place in the codebase where this double-pass cost is deliberately
  avoided; every other call site pays for a full forward pass per metric.

---

## `backdoor_data.py`

Purpose: reads BackdoorBench's `bd_test_dataset` folder of poisoned test PNGs
(each filename is the original dataset index, letting the clean counterpart
be recovered for a paired comparison), and provides three splitting/balancing
utilities: carve a stratified clean validation set out of a clean/backdoor
pair, and subsample to a fixed count per class. **Two of its five
definitions (`PngPathDataset`, `load_backdoor_splits`) are the actual
PNG-reading path and currently have no live callers anywhere in the tracked
tree** — see Sharp edges. The other three (`split_validation_and_eval`,
`balance_by_class`) are live, reused by `defences/checkpoint_eval.py` for
the in-code-trigger (not PNG) path.

### `PngPathDataset(Dataset)` (`backdoor_data.py:22-63`)

- `__init__(self, paths, transform, true_labels, label_mode, target_label,
  num_classes)` (`backdoor_data.py:32-50`): stores references; precomputes
  `self.eligible_positions` — every position where
  `is_eval_poisonable(label_mode, true_labels[position], target_label)`
  holds, mirroring `poison.py:AttackSuccessSet.__init__`'s eligibility
  computation exactly but over a list of file paths instead of an in-memory
  dataset.
- `__len__` (`backdoor_data.py:52-53`): `len(self.eligible_positions)`.
- `__getitem__(self, idx)` (`backdoor_data.py:55-63`): maps `idx` through
  `eligible_positions` to `position`; `Image.open(paths[position]).convert
  ("RGB")`, applies `transform` if given; computes `label` via
  `attack_success_label(label_mode, true_labels[position], target_label,
  num_classes)` — same function `poison.py:AttackSuccessSet` uses. Returns
  `(image, label)`.
- Docstring: eligibility and labeling deliberately mirror
  `AttackSuccessSet`'s in-memory logic exactly, so ineligible PNGs (e.g.
  already-target-class images under `all_to_one`) are filtered out at
  construction rather than served with an incorrect label.

### `load_backdoor_splits(folder_name, clean_test_dataset, transform,
weights_dir, label_mode, target_label, num_classes) -> tuple[Dataset, Dataset]`
(`backdoor_data.py:66-102`)

- Steps: (1) `backdoor_dir = weights_dir/folder_name/bd_test_dataset`; (2)
  glob all `**/*.png` under it, sorted; raises `FileNotFoundError` if empty;
  (3) `original_indices = [int(Path(p).stem) for p in backdoor_paths]` —
  parses each filename as the integer dataset index; (4)
  `clean_counterparts = Subset(clean_test_dataset, original_indices)`; (5)
  `true_labels = extract_labels(clean_counterparts)`; (6) builds
  `backdoor_test = PngPathDataset(backdoor_paths, transform, true_labels,
  label_mode, target_label, num_classes)`; (7)
  `eligible_counterparts = Subset(clean_counterparts,
  backdoor_test.eligible_positions)` — applies the *same* eligibility filter
  to the clean side, so `backdoor_test` and `eligible_counterparts` stay
  index-aligned position-for-position.
- Returns `(backdoor_test, eligible_counterparts)` — docstring emphasizes
  `backdoor_test` may be shorter than the full PNG folder (ineligible
  samples dropped), not merely shorter than `clean_test_dataset`.
- **No live callers** — see Sharp edges.

### `split_validation_and_eval(clean_counterparts, backdoor_test,
clean_val_size, seed) -> tuple[Subset, Subset, Subset]`
(`backdoor_data.py:105-130`)

- Steps: (1) `labels = extract_labels(clean_counterparts)` as a numpy array,
  `indices = arange(len(clean_counterparts))`; (2)
  `train_test_split(indices, test_size=len(indices)-clean_val_size,
  stratify=labels, random_state=seed)` → `(val_idx, eval_idx)` — stratified
  by label so every class is represented in the validation quantile
  threshold's support; (3) `clean_val = Subset(clean_counterparts,
  val_idx.tolist())`, `clean_eval = Subset(clean_counterparts,
  eval_idx.tolist())`, `backdoor_eval = Subset(backdoor_test,
  eval_idx.tolist())` — **the same `eval_idx` indices** index into both
  `clean_counterparts` and `backdoor_test`, which is what keeps the clean
  and backdoor eval sets paired (CLAUDE.md's "clean and backdoor eval sets
  are paired from the same images" correctness rule, implemented here at
  the split level).
- Randomness: `random_state=seed` makes the split reproducible for a given
  `seed`.
- Called by `defences/checkpoint_eval.py:build_eval_loaders_from_attack`
  (`defences/checkpoint_eval.py:65-67`) — notably called there with
  `clean_counterparts` and `backdoor_test` both bound to the **same**
  object (`test_base`, the plain clean 0-to-1 test set) since that path
  doesn't have a pre-poisoned PNG dataset — trigger application happens
  later via `AttackSuccessSet` wrapping, not via this split. This function
  only needs the two arguments to be the *same length* to slice them in
  parallel; it never actually inspects whether `backdoor_test`'s samples
  carry a trigger.

### `balance_by_class(clean_eval, backdoor_eval, examples_per_class, seed)
-> tuple[Subset, Subset]` (`backdoor_data.py:133-158`)

- Steps: (1) `clean_labels = extract_labels(clean_eval)`; (2) build
  `class_to_indices: dict[int, list[int]]` mapping each class label to the
  list of `clean_eval` positions holding it; (3) `seed_everything(seed)`
  (Lightning's global seeding — a side effect touching Python's, NumPy's,
  and PyTorch's global RNG state, not scoped to this function) **then**
  `rng = np.random.default_rng(seed)` (a second, separate seeded generator);
  (4) for each class id in sorted order, `rng.shuffle(candidates)` in place,
  take the first `examples_per_class` — `selected.extend(candidates[:examples_per_class])`;
  classes with fewer than `examples_per_class` examples contribute all of
  them (no padding, no error).
- Returns `(Subset(clean_eval, selected), Subset(backdoor_eval, selected))`
  — **the same `selected` index list** used for both, preserving pairing
  through the balancing step, same principle as `split_validation_and_eval`.
- Randomness: both `seed_everything(seed)` and a freshly constructed
  `np.random.default_rng(seed)` are invoked; only the latter's `rng` object
  is actually used for the shuffle, making the `seed_everything(seed)` call
  redundant for this function's own random selection (see Sharp edges).
- Called by `defences/checkpoint_eval.py:build_eval_loaders_from_attack`
  (`defences/checkpoint_eval.py:68-70`).

### Interactions

Imports `utils.datasets.extract_labels` and `poison.{attack_success_label,
is_eval_poisonable}` (both absolute, cross-package). `defences/checkpoint_eval.py`
is the only current consumer, and only of the two splitting/balancing
functions — never of the PNG-reading path this module's docstring and
module-level purpose foreground. Both live functions preserve index
alignment between a "clean" and "backdoor" dataset argument purely through
shared integer index lists (`Subset(..., same_indices)`), a pattern that
works whether or not the "backdoor" argument actually contains triggered
images yet.

### Sharp edges

- **`PngPathDataset` and `load_backdoor_splits` have zero live callers.**
  Grepping the entire tracked tree (excluding `_archive/`) turns up no
  import of either symbol outside `backdoor_data.py` itself. The only
  consumer is `_archive/sweep.py:14,44`. `defences/checkpoint_eval.py`'s
  own `read_checkpoint_metadata` error message
  (`defences/checkpoint_eval.py:33-36`) even suggests "provide a
  BackdoorBench `bd_test_dataset` folder instead" as a fallback when
  `args.json` is missing, but **no code path in `defences/checkpoint_eval.py`
  actually implements that fallback** — the message describes an option
  that isn't wired up. Any future PSBD sweep rewrite that wants to evaluate
  `backdoor_bench_checkpoints/` folders (which have no `args.json`) will
  need to actually call `load_backdoor_splits`, not just reference it in an
  error string.
- `balance_by_class`'s `seed_everything(seed)` call
  (`backdoor_data.py:150`) mutates Lightning's *global* RNG state as a side
  effect, immediately followed by constructing a separate
  `np.random.default_rng(seed)` that is what's actually used for the
  shuffle. The global reseed is not load-bearing for this function's own
  output (the local `rng` is fully self-contained and deterministic from
  `seed` alone), but it **does** affect any code that runs afterward and
  relies on unseeded global RNG state (e.g. a subsequent `torch.rand()`
  call elsewhere in the same process) — a caller unaware of this would see
  their own "random" downstream values silently become deterministic
  functions of `seed` after calling `balance_by_class`.
- `split_validation_and_eval`'s `train_test_split(..., stratify=labels)`
  requires **every class to have at least 2 members** (scikit-learn's
  stratified split constraint) — a class with only 1 example in
  `clean_counterparts` raises a `ValueError` from scikit-learn, not a
  project-level error message; this would surface for any dataset/split
  combination where `clean_val_size` and the natural class distribution
  interact badly (unlikely for the four registered datasets at their
  current sizes, but not guarded against explicitly here).
- `load_backdoor_splits` parses `Path(p).stem` as `int(...)` with no
  try/except — a `bd_test_dataset` folder containing any non-integer-named
  PNG (e.g. a stray `.DS_Store`-adjacent file, or BackdoorBench emitting a
  differently-named sample) would raise a raw `ValueError` from `int()`,
  not a descriptive error naming the offending file.

---

## `analysis/features.py`

Purpose: the single extraction primitive every other latent-analysis tool in
this package consumes — per-layer residual-stream features for ViT-B/16,
captured via forward hooks so the rest of `analysis/` stays pure (no model
internals knowledge required elsewhere). Documents its own layer-indexing
convention up front (Karayalcin et al.): index 0 is the token embedding fed
into block 1, indices 1–12 are the outputs of the 12 encoder blocks.

### Functions

**`_reduce_tokens(activation, reduction) -> torch.Tensor`**
(`analysis/features.py:23-37`, private)
- `activation` shape `(batch, tokens, dim)`. `reduction="cls"` → `activation[:,
  0, :]` (the classification token, shape `(batch, dim)`); `"mean"` →
  `activation.mean(dim=1)` (shape `(batch, dim)`, used for Swin which has no
  CLS token); `"flatten"` → `activation.flatten(1)` (shape `(batch,
  tokens*dim)`, "memory heavy," docstring says use only for small sample
  counts); else raises `ValueError`.
- Called by `_make_block_hook`, `_make_embedding_hook`.

**`_make_block_hook(storage, layer_index, reduction)`**
(`analysis/features.py:40-45`, private, closure factory)
- Returns a `hook(_module, _inputs, output)` forward-hook function that
  reduces `output` via `_reduce_tokens` and appends
  `.detach().float().cpu()` to `storage.setdefault(layer_index, [])`. Side
  effect: mutates the closed-over `storage` dict on every forward call the
  hook fires for.

**`_make_embedding_hook(storage, reduction)`** (`analysis/features.py:48-53`,
private, closure factory)
- Returns a forward-**pre**-hook `pre_hook(_module, inputs)` (note: pre-hook,
  fires before the module runs, reading `inputs[0]` — the encoder's input
  before any block processes it) that reduces and appends to
  `storage.setdefault(0, [])` — this is what populates layer index 0, the
  embedding, distinct from `_make_block_hook`'s post-hooks on each block.

**`extract_layer_features(model, loader, device, use_bfloat16, reduction="cls") -> dict[int, torch.Tensor]`**
(`analysis/features.py:56-82`, `@torch.inference_mode()`)
- Steps: (1) `encoder_blocks = vit_core(model).encoder.layers` — unwraps the
  `Sequential(Resize, ViT)` via `models.py:vit_core` then reaches the 12
  `EncoderBlock`s; (2) registers one forward-**pre**-hook on
  `encoder_blocks` itself (fires once per forward pass, before any block
  runs, capturing the embedding) via `_make_embedding_hook`, plus one
  forward-hook per individual block (`enumerate(encoder_blocks, start=1)`,
  so hook indices run 1 through 12) via `_make_block_hook`; (3) inside a
  `try`/`finally`, iterates `loader`, calling `forward_probs(model, images,
  device, use_bfloat16)` purely to *trigger* the forward pass — the returned
  probabilities are discarded, only the hook side effects matter; (4) in
  `finally`, removes every registered handle unconditionally, even if the
  loader iteration raised — guarantees hooks never leak onto the model
  object past this function call; (5) returns `{layer: torch.cat(chunks,
  dim=0) for layer, chunks in sorted(storage.items())}` — concatenates all
  batches per layer into one tensor.
- Shapes: return value is `dict[int, Tensor]`, keys `0..12`, each tensor
  `(num_samples, dim)` float32 CPU (`dim=768` for ViT-B/16's CLS/mean
  reduction, `768*197` for `"flatten"` at 224×224/16×16 patches +1 CLS
  token).
- Docstring note: because hooks read whatever a block *returns*, they
  automatically capture post-residual dropout's effect when that placement
  is active (the hook fires after the wrapper's `forward`, which is after
  its internal Dropout calls) — this is exactly what the placement
  comparison experiment needs, with no special-casing required in this
  file.
- Called by `analysis/analyze_latent.py` (twice — once for clean features,
  once for triggered/backdoor features).

### Interactions

Imports `defences.inference.forward_probs` (to drive the forward pass) and
`models.vit_core` (to reach block internals) — the only `analysis/` module
that reaches into `defences/` or `models.py` directly; every other
`analysis/` module operates on already-extracted feature tensors.

### Sharp edges

- `extract_layer_features` is **ViT-specific** — it calls
  `vit_core(model).encoder.layers` directly, which assumes the ViT
  `EncoderBlock` structure (`.encoder.layers`, an iterable of blocks each
  supporting a forward hook that returns the residual-stream tensor
  directly). There is no Swin equivalent in this file (`models.py`'s
  asymmetry — no `swin_core` — propagates here); calling this on a Swin
  model would raise `AttributeError` on `.encoder.layers` not existing in
  that form.
- The embedding hook is a forward-**pre**-hook on `encoder_blocks` (the
  whole `nn.Sequential` of blocks), not on the embedding module itself —
  it works because `encoder_blocks`'s pre-hook input *is* the embedding
  output, but a reader expecting "layer 0 hook is on the embedding layer"
  would be looking in the wrong place structurally.
- If `loader` raises partway through iteration, `finally` still removes all
  hooks, but `storage` will contain a **partial** and *ragged* set of
  per-layer sample counts (some batches processed before the failure,
  others not) — the exception propagates after cleanup, so the caller
  never receives a return value in that case, avoiding silently returning
  incomplete features; correct behavior, just worth noting as the reason
  this function doesn't need to validate row-count consistency.

---

## `analysis/cka.py`

Purpose: Centered Kernel Alignment, biased and debiased, linear and RBF
kernel variants, for comparing two sets of features on the same inputs.
Module docstring frames the central usage: comparing a layer to itself with
dropout off vs. on, separately for clean and backdoor inputs, to see which
dropout placement perturbs the backdoor signal more — the paper's core
mechanistic evidence. Both compared feature tensors must be `[num_samples,
dim]`, same samples, same order (index-aligned, same convention as
`analysis/direction.py`'s and `analysis/features.py`'s outputs).

### Functions

**`_center_columns(features) -> torch.Tensor`** (`analysis/cka.py:27-28`,
private)
- `features - features.mean(dim=0, keepdim=True)` — subtracts the
  per-feature-dimension mean across samples.

**`linear_cka(features_x, features_y) -> float`** (`analysis/cka.py:31-43`)
- original form: `CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)`
  (given explicitly in the docstring). Centers both inputs (`.float()`
  cast), computes `cross = (y.t() @ x).norm()**2`, `normalizer = (x.t() @
  x).norm() * (y.t() @ y).norm()`, returns `(cross /
  normalizer.clamp_min(1e-12)).item()`.
- This is the **biased** estimator — docstring (module-level, lines 6-8)
  warns it assigns even unrelated representations a positive baseline
  similarity when sample count is not much larger than feature dimension,
  shrinking only as sample count grows. **No callers found** anywhere in
  the tracked tree — superseded by `debiased_linear_cka` at every live call
  site.

**`_linear_gram(features) -> torch.Tensor`** (`analysis/cka.py:46-50`,
private)
- `x = features.double(); return x @ x.t()` — the `(n, n)` linear Gram
  matrix, computed in **float64**. Comment: double precision is needed
  because the unbiased HSIC estimator sums over `n²` gram entries, which
  loses accuracy in float32 for larger sample counts.
- Called by `debiased_linear_cka`, `layerwise_cka_matrix`.

**`_rbf_gram(features, sigma) -> torch.Tensor`** (`analysis/cka.py:53-61`,
private)
- `x = features.double()`; `squared_distances = cdist(x, x)**2`; if
  `sigma is None`, `bandwidth = squared_distances.median().sqrt().clamp_min(1e-8)`
  (the median heuristic, removing the bandwidth as a free hyperparameter);
  else `bandwidth = tensor(float(sigma))`. Returns `exp(-squared_distances /
  (2 * bandwidth**2))` — the RBF/Gaussian kernel Gram matrix.
- Called by `rbf_cka`, `debiased_rbf_cka`.

**`biased_hsic(gram_k, gram_l) -> torch.Tensor`** (`analysis/cka.py:64-77`)
- original form: `HSIC(K,L) = 1/(n-1)^2 * trace(K H L H)` with `H = I -
  (1/n) 1 1^T` (given explicitly). Builds `centering = I - ones(n,n)/n`,
  computes `centered_k = centering @ gram_k @ centering`, returns
  `(centered_k * gram_l).sum() / (n-1)**2` — centers `K` then sums its
  elementwise product with `L` (the simplified form given in the docstring;
  algebraically equal to the trace formula since `trace(AB) = sum(A ⊙
  B^T)` and both centered/gram matrices here are symmetric).
- Called by `_cka_from_grams` when `debiased=False`; `rbf_cka`.

**`unbiased_hsic(gram_k, gram_l) -> torch.Tensor`** (`analysis/cka.py:80-108`)
- original form given explicitly (Song et al. 2012's unbiased HSIC
  estimator, `analysis/cka.py:86-88`), with `K̃`/`L̃` the grams with
  diagonals zeroed. Requires `n >= 4`, raising `ValueError` otherwise (the
  formula divides by `n(n-3)`, undefined/degenerate below 4). Steps: zero
  the diagonals of both grams (`.clone().fill_diagonal_(0.0)`); `dot_term =
  (k*l).sum()`; `total_k = k.sum()`, `total_l = l.sum()`; `row_coupling =
  (k.sum(dim=0) * l.sum(dim=0)).sum()`; returns `(dot_term + total_k *
  total_l/((n-1)(n-2)) - 2*row_coupling/(n-2)) / (n*(n-3))`.
- Called by `_cka_from_grams` when `debiased=True` (the default path),
  `debiased_linear_cka`, `debiased_rbf_cka`.

**`_cka_from_grams(gram_x, gram_y, hsic) -> float`** (`analysis/cka.py:111-114`,
private)
- `numerator = hsic(gram_x, gram_y)`; `denominator =
  sqrt(clamp_min(hsic(gram_x,gram_x) * hsic(gram_y,gram_y), 1e-12))`;
  returns `(numerator/denominator).item()` — the CKA normalization common
  to every gram-based variant (biased or unbiased HSIC, linear or RBF gram),
  factored out so it's implemented once.
- Called by `debiased_linear_cka`, `rbf_cka`, `debiased_rbf_cka`,
  `layerwise_cka_matrix`.

**`debiased_linear_cka(features_x, features_y) -> float`**
(`analysis/cka.py:117-121`)
- `_cka_from_grams(_linear_gram(features_x), _linear_gram(features_y),
  unbiased_hsic)` — the estimator the module docstring recommends whenever
  sample count is close to or below feature dimension (768 for ViT-B/16 CLS
  features), which the docstring says is "a few hundred samples" in
  practice for this project.
- Called by `analysis/analyze_latent.py`.

**`rbf_cka(features_x, features_y, sigma=None) -> float`**
(`analysis/cka.py:124-130`) / **`debiased_rbf_cka(...)`**
(`analysis/cka.py:133-139`)
- Same `_cka_from_grams` pattern with `_rbf_gram` instead of
  `_linear_gram`, biased vs. unbiased HSIC respectively. **No callers
  found** in the tracked tree.

**`layerwise_cka_matrix(features_by_layer_a, features_by_layer_b,
debiased=True) -> torch.Tensor`** (`analysis/cka.py:142-163`)
- Sorts both layer-index dicts' keys; picks `unbiased_hsic` or
  `biased_hsic` per `debiased`; **precomputes** every layer's linear Gram
  matrix once (`grams_a`, `grams_b` dicts, `analysis/cka.py:156-157`)
  rather than recomputing inside the double loop; fills an `(len(layers_a),
  len(layers_b))` matrix by calling `_cka_from_grams(grams_a[layer_a],
  grams_b[layer_b], hsic)` for every `(i, j)` pair.
- Returns a full pairwise CKA matrix across every layer combination — e.g.
  every-clean-layer-vs-every-backdoor-layer, not just the diagonal
  (same-layer) comparisons the module docstring's main use case describes.
- **No callers found** in the tracked tree — a heavier tool than what
  `analyze_latent.py`'s worked example currently exercises (which only
  calls `debiased_linear_cka` on matched same-layer pairs directly).

### Interactions

Only `torch` imported; no internal-package dependencies. Consumed
exclusively by `analysis/analyze_latent.py` today (just
`debiased_linear_cka`) — the RBF variants, the biased linear variant, and
`layerwise_cka_matrix` are implemented but not currently exercised outside
this file.

### Sharp edges

- `_linear_gram`/`_rbf_gram` upcast to **float64** specifically to keep the
  unbiased HSIC estimator numerically accurate at realistic sample counts;
  calling `unbiased_hsic` directly on a float32 Gram matrix built some other
  way (bypassing these helpers) would silently reintroduce the precision
  loss this file's comment explicitly warns about, with no runtime check
  enforcing the dtype.
- `unbiased_hsic`'s `n >= 4` guard is the only shape validation anywhere in
  this file — nothing checks that `features_x`/`features_y` have the same
  number of samples (`n`) before computing Grams; mismatched sample counts
  produce grams of different sizes and `_cka_from_grams`'s elementwise
  `centered_k * gram_l` (inside `biased_hsic`) or the analogous
  `unbiased_hsic` sums would raise a shape-mismatch error deep inside HSIC,
  not a clear message at the CKA entrypoint about "features_x and
  features_y must have the same sample count."
- CKA values are only meaningful when `features_x` and `features_y` are
  computed on the exact same samples in the exact same order (module
  docstring states this explicitly) — nothing in `cka.py` itself verifies
  this; the pairing discipline is entirely the caller's responsibility
  (same class of risk as `defences/inference.py`'s paired-eval-set
  requirement, but unenforced here too).
- Debiased CKA "can fall slightly outside 0 to 1, which is expected for a
  finite-sample unbiased estimate" (module docstring) — a caller asserting
  `0 <= cka_value <= 1` would be asserting something this estimator does
  not guarantee.

---

## `analysis/direction.py`

Purpose: the paired-difference toolkit — everything downstream of the single
idea "the difference between a backdoor feature and its clean counterpart at
one layer, on the same samples." The backdoor direction is that difference's
mean (a vector); TAC is its per-dimension magnitude (also a vector, but of
absolute-value magnitudes, not a mean-then-magnitude of the mean). Also
provides projection (how much backdoor signal a representation carries along
that direction), an outlier-dimension rule reused for TAC/Lipschitz values,
a steering forward-hook factory, and weight orthogonalization. Inputs
throughout are the index-aligned `[num_samples, dim]` tensors
`analysis/features.py:extract_layer_features` returns per layer.

### Functions

**`_unit(direction) -> torch.Tensor`** (`analysis/direction.py:16-17`,
private)
- `direction / direction.norm().clamp_min(1e-8)` — L2-normalizes, guarding
  divide-by-zero for a near-zero direction vector.
- Called by `project_onto_direction`, `orthogonalize_weight`.

**`backdoor_direction(clean_features, backdoor_features) -> torch.Tensor`**
(`analysis/direction.py:20-30`)
- original form: `r_l = (1/|pairs|) * sum(backdoor_activation -
  clean_activation)` (given explicitly). `(backdoor_features -
  clean_features).mean(dim=0)` — a single `(dim,)` vector, the mean paired
  difference across all samples: "the trigger's representation at this
  layer."
- Called by `analysis/analyze_latent.py`.

**`trigger_activated_change(clean_features, backdoor_features) -> torch.Tensor`**
(`analysis/direction.py:33-46`)
- original form given explicitly: per-channel TAC as the mean L2 norm of
  the per-sample feature difference; simplified/implemented form here (for
  a single CLS-or-mean-token representation, one scalar per residual
  dimension) is `(backdoor_features - clean_features).abs().mean(dim=0)` —
  note this differs from `backdoor_direction`: TAC takes the **absolute
  value before averaging** (so cancellation across samples with opposite-
  sign shifts on the same dimension does not hide a real per-sample effect),
  while `backdoor_direction` averages the signed difference first (so
  opposite-sign per-sample shifts on the same dimension *do* cancel,
  producing a smaller net direction). A high TAC value on a dimension means
  that dimension moves a lot when the trigger is applied — a candidate
  backdoor dimension.
- Called by `analysis/analyze_latent.py`.

**`project_onto_direction(features, direction) -> torch.Tensor`**
(`analysis/direction.py:49-57`)
- `features @ _unit(direction).to(features.dtype)` — signed scalar
  projection length per sample onto the unit direction; shape `(dim,) →`
  input `(num_samples, dim) @ (dim,) → (num_samples,)`. Docstring: intended
  use is measuring how much backdoor signal a representation still carries,
  e.g. before/after dropout at each placement. **No callers found** in the
  tracked tree outside this file.

**`outlier_dimensions(values, sensitivity=3.0) -> torch.Tensor`**
(`analysis/direction.py:60-67`)
- `threshold = values.mean() + sensitivity * values.std()`; returns
  `torch.nonzero(values > threshold, as_tuple=False).squeeze(1)` — indices
  of dimensions exceeding mean + `sensitivity` standard deviations. Named
  after the CLP (Channel Lipschitz Pruning) outlier rule, explicitly
  reusable for TAC values or Lipschitz values (`analysis/lipschitz.py`'s
  outputs are the same `(dim,)`-shaped tensors this function expects). A
  larger `sensitivity` flags fewer, more extreme dimensions. **No callers
  found** in the tracked tree outside this file.

**`make_steering_hook(direction, scale)`** (`analysis/direction.py:70-82`)
- `shift = scale * direction`; returns a `hook(_module, _inputs, output)`
  forward-hook closure that returns `output + shift.to(output.dtype).to(
  output.device)`. Docstring: register on the encoder block whose output
  should be steered — positive `scale` on clean inputs is meant to activate
  the backdoor behavior; negative `scale` on backdoor inputs is meant to
  recover the original (non-backdoored) prediction — a causal-intervention
  probe for whether the identified direction is actually behaviorally
  responsible for the backdoor, not just correlated with it. **No callers
  found** in the tracked tree outside this file — an analysis primitive
  meant for interactive/notebook use, not currently wired into any script.

**`orthogonalize_weight(weight, direction) -> torch.Tensor`**
(`analysis/direction.py:85-97`)
- original form given explicitly: `W_new = W - r̂ r̂^T W` with `r̂` the unit
  direction. `unit_direction = _unit(direction).to(weight.dtype)`; returns
  `weight - torch.outer(unit_direction, unit_direction) @ weight` — removes
  the direction's component from every column of `weight`. Docstring: `weight`
  has shape `[residual_dim, input_dim]`, since the direction lives in the
  residual stream that the matrix writes into (i.e. this projects out the
  backdoor direction from a layer's output-writing weight matrix, a
  weight-editing ablation). **No callers found** in the tracked tree outside
  this file.

### Interactions

Only `torch` imported; no internal dependencies. Only `backdoor_direction`
and `trigger_activated_change` are currently exercised, both by
`analysis/analyze_latent.py`; `project_onto_direction`,
`outlier_dimensions`, `make_steering_hook`, `orthogonalize_weight` are
implemented analysis primitives with no current call sites — a toolkit for
follow-up mechanistic work (causal steering/ablation experiments) beyond
the worked example.

### Sharp edges

- `backdoor_direction` (signed mean) and `trigger_activated_change`
  (mean of absolute values) are **not** the same quantity and are easy to
  conflate given their similar inputs/outputs (both `(dim,)` vectors from
  the same two input tensors) — a dimension with a strong but
  sample-inconsistent-sign effect would show up prominently in TAC but be
  suppressed or near-zero in the backdoor direction. Choosing the wrong one
  for a given question (e.g. using TAC where a directional steering vector
  is needed) would silently produce a vector that looks plausible but
  encodes a different statistic.
- `make_steering_hook`'s docstring describes the *intended* causal
  direction of `scale`'s sign (positive on clean → activate; negative on
  backdoor → recover), but nothing in the hook itself enforces or checks
  which kind of input it's being applied to — the caller must know which
  regime they're steering and choose the sign accordingly; applying the
  wrong sign silently produces a hook that shifts activations in the wrong
  direction without any error.
- `orthogonalize_weight` mutates nothing in place — it returns a **new**
  tensor. A caller intending to actually ablate the direction from a live
  model must explicitly reassign the returned tensor back into the model's
  parameter (e.g. `module.weight.data = orthogonalize_weight(...)`);
  calling this function alone has no effect on the model.
- `outlier_dimensions`'s `sensitivity * std()` rule assumes an
  approximately unimodal, roughly-symmetric distribution of per-dimension
  values — on a heavily skewed TAC or Lipschitz distribution (plausible,
  since both are non-negative quantities with a long right tail by
  construction), a fixed-sensitivity mean+k·std threshold can flag far more
  or far fewer dimensions than intended, with no diagnostic output warning
  that the assumption may not hold.

---

## `analysis/embedding.py`

Purpose: 2D projections of feature tensors for visualization — PCA (linear)
and UMAP (nonlinear). Module docstring is explicit about when to reach for
each: PCA is "the honest first choice for the backdoor question because the
backdoor is hypothesized to be a linear direction, and a linear projection
cannot invent structure that is not there"; UMAP reveals nonlinear cluster
structure PCA misses but "warps distances and can produce clusters that are
artifacts of its hyperparameters," so treat it as qualitative and always
report `n_neighbors`/`min_dist`. t-SNE is deliberately excluded (UMAP
preserves more global structure at similar-or-lower cost, per CLAUDE.md's
stated "prefer UMAP over t-SNE" analysis convention).

### Functions

**`pca_project(features, num_components=2) -> np.ndarray`**
(`analysis/embedding.py:23-24`)
- `PCA(n_components=num_components).fit_transform(features.float().numpy())`
  — scikit-learn PCA, fit and transform on the same data (not fit on a
  reference set and applied elsewhere). Input `(num_samples, dim)` tensor,
  output `(num_samples, num_components)` numpy array.
- Called by `analysis/analyze_latent.py`.

**`umap_project(features, num_neighbors=15, min_distance=0.1, seed=0) ->
np.ndarray`** (`analysis/embedding.py:27-46`)
- Lazily imports `umap` inside the function body (`analysis/embedding.py:39`),
  not at module top level — raises a descriptive `ImportError` ("umap_project
  needs umap-learn, install it with pip install umap-learn") if the optional
  `umap-learn` package isn't installed, rather than failing at import time
  for callers who never call this function. `projector =
  umap.UMAP(n_neighbors=num_neighbors, min_dist=min_distance,
  random_state=seed)`; returns `projector.fit_transform(features.float()
  .numpy())`.
- Randomness: `random_state=seed` makes the projection reproducible for
  fixed inputs and seed.
- **No callers found** in the tracked tree — implemented but not currently
  exercised by `analyze_latent.py`'s worked example (which only calls
  `pca_project`), consistent with the module docstring's framing of PCA as
  the primary tool and UMAP as a qualitative supplement.

### Interactions

Imports only `numpy`, `torch`, `sklearn.decomposition.PCA` at module level;
`umap` is an optional, lazily-imported dependency (CLAUDE.md's "do not add
[external libraries] as hard runtime dependencies without asking" applies
here — `umap-learn` is opt-in, not required to import this module or even
most of this package).

### Sharp edges

- `umap_project`'s deferred import means a missing `umap-learn` install is
  invisible until this specific function is actually called — `import
  analysis.embedding` succeeds regardless (confirmed by
  `tests/test_imports.py`'s exhaustive import-only test, which does not
  exercise `umap_project` itself), so a broken UMAP dependency would only
  surface at the point of use, not at any earlier import-time check.
- `pca_project`'s `n_components` is a free parameter but the module
  docstring's contract ("returns a `[num_samples, 2]` numpy array") assumes
  the default of 2 — calling with a different `num_components` still works
  functionally but breaks the documented shape contract other code might
  assume when consuming the result (e.g. code that unpacks `x, y =
  result.T` would fail or silently misbehave for `num_components != 2`).

---

## `analysis/lipschitz.py`

Purpose: data-free (weight-only, no forward pass needed) ViT weight analysis
inspired by Channel Lipschitzness (Zheng et al. 2022, originally a ConvNet
backdoor-channel detector). Module docstring is explicit about where the
ConvNet-to-ViT transfer breaks down: standard dot-product self-attention is
*not* Lipschitz on an unbounded domain (Kim et al. 2021), so there is no
clean weight-only Lipschitz bound for attention as a whole. What transfers:
the MLP block's final linear layer (linear map + near-1-Lipschitz GELU, so
per-output-channel Lipschitz constants are meaningful), and treating
attention's QKV/output-projection spectral norms as sensitivity proxies
(not true Lipschitz bounds, since the mixing between them isn't Lipschitz).
Also implements `head_weight_alignment`, described as an adaptation of the
Karayalcin et al. (Section 8) fully data-free ViT-native detector, which the
docstring calls "the better fit than raw Lipschitz numbers" for a fully
data-free detector.

### Functions

**`spectral_norm(weight) -> float`** (`analysis/lipschitz.py:32-34`)
- `torch.linalg.svdvals(weight.float())[0].item()` — the largest singular
  value (operator 2-norm) of a linear map's weight matrix. **No callers
  found** in the tracked tree outside this file.

**`linear_channel_lipschitz(weight) -> torch.Tensor`**
(`analysis/lipschitz.py:37-43`)
- `weight.float().norm(dim=1)` — for a linear map `out_k = row_k · input`,
  the Lipschitz constant of output channel `k` is exactly that row's L2
  norm (stated directly in the docstring, a simple/exact fact for a linear
  map, not an approximation). Returns one value per output channel, shape
  `(out_features,)`.
- Called by `mlp_output_channel_lipschitz`.

**`mlp_output_channel_lipschitz(model) -> dict[int, torch.Tensor]`**
(`analysis/lipschitz.py:46-58`)
- For each block in `vit_core(model).encoder.layers` (`enumerate(...,
  start=1)`, so keys run 1–12, matching `analysis/features.py`'s
  layer-indexing convention where block outputs are indices 1–12): finds
  `output_linear = _last_linear(block.mlp)` (the MLP's second/final linear
  layer, the one that writes into the residual stream), computes
  `linear_channel_lipschitz(output_linear.weight)`. Returns `{layer_index:
  (mlp_hidden_or_output_dim,) tensor}`.
- Docstring: high values here are candidate backdoor channels "in the same
  spirit as CLP," directly comparable against TAC computed from paired data
  (`analysis/direction.py:trigger_activated_change`) — both are
  `(dim,)`-shaped per-layer vectors over the same residual dimension space,
  one data-free (this), one data-driven (TAC).
- **No callers found** in the tracked tree outside this file — a
  data-free companion to the (also currently unexercised outside
  `analyze_latent.py`'s TAC call) mechanistic toolkit.

**`_last_linear(module) -> nn.Linear`** (`analysis/lipschitz.py:61-65`,
private)
- `linears = [layer for layer in module.modules() if isinstance(layer,
  nn.Linear)]`; returns `linears[-1]`; raises `ValueError("No linear layer
  found in the MLP block")` if `linears` is empty. Relies on
  `nn.Module.modules()`'s traversal order matching construction/definition
  order (true for standard `nn.Sequential`-based MLP blocks, which is what
  ViT-B/16's MLP is) to find the *last* linear layer — i.e. the
  output-projection layer, not the first (expansion) layer.
- Called by `mlp_output_channel_lipschitz`.

**`head_weight_alignment(model, num_layers, threshold) -> torch.Tensor`**
(`analysis/lipschitz.py:68-95`)
- original form given explicitly: `s_i = sum over layers of count(
  |c_i^T W|_l > threshold )` for each class row `c_i` of the classifier
  head. Steps: (1) `core = vit_core(model)`; `class_directions =
  core.heads.head.weight` — shape `(num_classes, dim)`, one row per class;
  (2) collects `projection_weights` — the `self_attention.out_proj.weight`
  (shape `(dim, dim)`) of only the **first `num_layers` blocks**
  (`list(core.encoder.layers)[:num_layers]`) — an early-layers-only scope,
  motivated by the docstring's claim that "a backdoor target class tends to
  align with an early-layer shortcut"; (3) for every class row and every
  collected projection weight matrix, computes `alignment = |class_direction
  @ weight|` (a `(dim,)` vector) and adds `(alignment > threshold).sum()`
  to that class's running score; (4) returns a `(num_classes,)` score
  tensor — the count of large-magnitude alignments between each class
  direction and the early output-projection weights, summed across the
  `num_layers` early blocks.
- Docstring frames the expected signature of an attack: a backdoor target
  class's score is an outlier among all classes' scores (high alignment
  with an early-layer shortcut the trigger exploits, bypassing most of the
  network's depth).
- **No callers found** in the tracked tree outside this file — described
  in its own docstring as "adapted here for reference," i.e. present as a
  documented reimplementation rather than a wired-in detector.

### Interactions

Imports only `torch`, `torch.nn`, and `models.vit_core` — the only other
`analysis/` module besides `features.py` that reaches outside the package
(both reach into `models.py`, neither into `defences/`).
`mlp_output_channel_lipschitz`'s output is explicitly designed to be
compared against `analysis/direction.py:trigger_activated_change`'s output
(same shape, same indexing convention), though no code currently performs
that comparison automatically — it's a manual/notebook-level analysis step.

### Sharp edges

- Every function in this file is **ViT-specific** (`vit_core`, `.encoder.layers`,
  `.heads.head`, `.self_attention.out_proj`) — none has a Swin
  counterpart, consistent with `models.py`'s asymmetric `vit_core`/no-`swin_core`
  split and `analysis/features.py`'s same limitation. Calling any function
  here on a Swin model raises an `AttributeError` on the first
  ViT-specific attribute access, not a clear "Swin not supported" message.
- `head_weight_alignment`'s `threshold` parameter has no principled default
  documented or provided (required positional argument) — the "outlier
  among classes" signature it's meant to reveal depends entirely on
  choosing a `threshold` that produces a meaningfully differentiated score
  distribution; too low saturates every class's score near
  `num_layers * dim`, too high collapses every score toward 0, and nothing
  in this file helps calibrate it (contrast with
  `analysis/direction.py:outlier_dimensions`, which derives its own
  threshold statistically from the data rather than requiring a fixed
  input).
- `_last_linear`'s reliance on `nn.Module.modules()` traversal order to
  find "the last" linear layer is a structural assumption about how the
  MLP block is built (sequential definition order), not a robust
  "output-projection layer" identification by name or role — an MLP block
  refactored to register its linear layers in a different order (or to use
  named submodules accessed non-sequentially) would silently change which
  layer `mlp_output_channel_lipschitz` reports on, with no error.

---

## `defences/checkpoint_eval.py`

Purpose: assembles the three-way (validation, clean-eval, backdoor-eval)
PSBD loader split for **this repo's own trained checkpoints**
(`checkpoints/`, which carry an `args.json` sidecar), as opposed to
BackdoorBench's downloaded checkpoints (`backdoor_bench_checkpoints/`, which
ship a `bd_test_dataset` PNG folder that `backdoor_data.py`'s PNG path
reads instead). Since this repo's own triggers are defined in code, not
PNGs, the poisoned test set is rebuilt in memory from the attack recorded
in the checkpoint's metadata, reusing the same split/balance functions
(`backdoor_data.split_validation_and_eval`, `backdoor_data.balance_by_class`)
the PNG path would use, so both paths produce the same three-way structure
downstream.

### Functions

**`working_resolution(dataset_name) -> int`** (`defences/checkpoint_eval.py:23-26`)
- `64 if dataset_name == "tiny" else 32` — the native/trigger-definition
  resolution for a dataset (triggers are defined at this size; the model's
  own `Resize` inside `build_vit`/`build_swin` upsamples to 224 afterward).
- **Independently duplicated** in `train_backdoor.py:39-41`,
  `train_benign.py:32-34`, `analysis/analyze_latent.py:43-45` — four
  separate identical one-line function bodies across the codebase, not a
  shared helper. `metrics.py:97,128` is the one caller that actually
  imports this specific copy (`from defences.checkpoint_eval import
  working_resolution`, `metrics.py:29`) rather than redefining it locally.

**`read_checkpoint_metadata(checkpoint_path) -> dict`**
(`defences/checkpoint_eval.py:29-43`)
- Steps: (1) `args_path = os.path.dirname(checkpoint_path)/args.json`; (2)
  if missing, raises `FileNotFoundError` with a message suggesting either
  retraining (which writes `args.json`) or "provide a BackdoorBench
  bd_test_dataset folder instead" (a suggestion this file does not itself
  implement — see `backdoor_data.py`'s Sharp edges for the corresponding
  gap); (3) loads the JSON; (4) checks `required = ("dataset", "attack",
  "target_label")` are all present, raising `KeyError` naming any missing
  keys; (5) returns the full metadata dict (which per `train.py`'s
  `checkpoint_metadata` also includes `label_mode`, `poison_rate`,
  `cover_rate`, `architecture`, `optimizer`, `rho`, `epochs`, `seed`,
  `git_commit`, timestamps — only the three required keys are validated
  here, the rest pass through unchecked).
- I/O: reads a JSON file from disk.
- Called by `build_eval_loaders_from_checkpoint`
  (`defences/checkpoint_eval.py:100`), directly by
  `tests/test_checkpoint_metadata.py`.

**`build_eval_loaders_from_attack(dataset_name, attack, config, image_size) ->
tuple[DataLoader, DataLoader, DataLoader]`** (`defences/checkpoint_eval.py:46-89`)
- Steps: (1) loads the 0-to-1 base test set via a local
  `base_transform = Compose([Resize((image_size,image_size)), ToTensor()])`
  (yet another independent redefinition of the "stop before normalizing"
  transform pattern — see `utils/datasets.py`'s Sharp edges); (2)
  `clean_val_base, clean_eval_base, backdoor_eval_base =
  split_validation_and_eval(test_base, test_base, config.clean_val_size,
  config.seed)` — passes the **same** `test_base` object as both the
  "clean_counterparts" and "backdoor_test" arguments, since no poisoned
  version exists yet at this point (poisoning happens via `AttackSuccessSet`
  wrapping two steps later); this only works because
  `split_validation_and_eval` merely needs two same-length datasets to slice
  in parallel, and doesn't inspect their contents (documented in
  `backdoor_data.py`'s own section above); (3) `balance_by_class` further
  subsamples `clean_eval_base`/`backdoor_eval_base` to
  `config.examples_per_class` per class; (4) wraps: `clean_val` and
  `clean_eval` are both `PoisonedTrainingSet(..., set(), normalize, ...)`
  (empty poison-index set, so both are effectively plain normalized clean
  wrappers — reusing `PoisonedTrainingSet` for its "apply normalize only"
  behavior, same pattern `metrics.py` and `analyze_latent.py` use);
  `backdoor_eval` is an `AttackSuccessSet(backdoor_eval_base,
  extract_labels(backdoor_eval_base), attack, normalize, num_classes)` — the
  trigger is applied here, for the first time in this pipeline; (5) returns
  three `DataLoader`s, all `batch_size=config.batch_size, shuffle=False`
  (unshuffled — required for any code downstream that assumes loader order
  is stable across repeated iteration, e.g.
  `defences/inference.py:compute_psu_and_shift`'s `zip(loader,
  baseline_cache)` pairing).
- Docstring explicitly states the split/balance run on the 0-to-1 base
  test set "where labels are cheap to read" (i.e. before any expensive
  per-sample trigger application), and that wrapping happens last so only
  the clean sets normalize while the backdoor set both triggers and
  normalizes.
- Called by `build_eval_loaders_from_checkpoint`
  (`defences/checkpoint_eval.py:109`). **No other callers** — see Sharp
  edges.

**`build_eval_loaders_from_checkpoint(checkpoint_path, config) ->
tuple[DataLoader, DataLoader, DataLoader]`** (`defences/checkpoint_eval.py:92-109`)
- Steps: (1) `metadata = read_checkpoint_metadata(checkpoint_path)`; (2)
  `image_size = working_resolution(metadata["dataset"])`; (3) `attack =
  build_attack(metadata["attack"], default_config(metadata["attack"]),
  image_size, metadata["target_label"])` — rebuilds the attack from its
  **default** config, not whatever config the checkpoint was actually
  trained with; (4) delegates to `build_eval_loaders_from_attack`.
- Docstring explicitly flags the default-config assumption: "matches how
  `train_backdoor.py` builds it. A custom attack config would need to be
  recorded in the checkpoint too" — i.e. this is only correct as long as no
  checkpoint was ever trained with a non-default attack config (patch size,
  alpha, cover_rate, etc. all at their dataclass defaults); `args.json`
  does not currently record attack-specific hyperparameters beyond
  `poison_rate`/`cover_rate`/`target_label`/`label_mode`, so there would be
  no way to detect or recover a non-default config even if one existed.
- **No callers found** anywhere in the tracked tree outside this file
  (not even from `metrics.py`, which instead calls `attacks.build_attack`
  directly with `default_config` itself, inlining essentially the same
  logic — see `metrics.py`'s section).

### Interactions

Imports `attacks.{build_attack, default_config}`,
`backdoor_data.{balance_by_class, split_validation_and_eval}`,
`utils.config.{DATASET_REGISTRY, RunConfig}`,
`utils.datasets.{extract_labels, load_clean_datasets}`,
`poison.{Attack, AttackSuccessSet, PoisonedTrainingSet}` — the single file
in the repo that ties together the attack registry, the dataset registry,
and the poisoning dataset wrappers specifically to reconstruct a
*checkpoint's* eval set from its saved metadata, rather than from
training-time arguments directly available in memory (which is what
`train_backdoor.py` does instead, since it has `args` from `argparse`
rather than a saved `args.json`).

### Sharp edges

- **`build_eval_loaders_from_attack` and `build_eval_loaders_from_checkpoint`
  have no live callers anywhere in the tracked tree.** They are not called
  by `metrics.py` (which inlines an equivalent but separate construction in
  `build_full_eval_loaders`, using the whole test set rather than a
  validation-held-out, class-balanced subset), and the archived
  `_archive/sweep.py` has its own independent `build_eval_loaders`
  function that does **not** import from this module — it reads
  BackdoorBench PNGs directly (predating this file, which is written for
  the checkpoints-with-`args.json` case the archived sweep never handled).
  This means this file's actual eval-loader assembly is currently staged
  infrastructure, presumably intended for the pending PSBD sweep rewrite
  CLAUDE.md references, not something exercised by any current script.
  Only `read_checkpoint_metadata` and `working_resolution` are live,
  consumed by `metrics.py`.
- `build_eval_loaders_from_checkpoint`'s use of `default_config(metadata["attack"])`
  instead of any recorded custom config is a **silent correctness gap** for
  any checkpoint trained with non-default attack hyperparameters (a
  different `patch_size`, `alpha`, `frequency`, etc. than the dataclass
  default) — the eval set rebuilt here would use the *wrong* trigger,
  producing a meaningless ASR number with no error or warning, since
  `args.json` doesn't record enough to reconstruct a non-default attack
  config. In practice `train_backdoor.py` itself always calls
  `resolve_config`/`default_config` too (see `train_backdoor.py`'s section),
  so this gap is currently latent rather than active — but nothing
  structurally prevents someone from later adding a CLI flag to
  `train_backdoor.py` that overrides e.g. `patch_size` without a
  corresponding `args.json`/eval-side update, at which point this becomes a
  real, silent bug.
- The four independent copies of `working_resolution` (this file,
  `train_backdoor.py`, `train_benign.py`, `analysis/analyze_latent.py`) are
  a duplication risk: changing the resolution rule (e.g. adding a fifth
  dataset) in one copy and not the other three would silently desync
  trigger-definition resolution between training and eval for that
  dataset, since nothing enforces the four bodies stay identical.

---

## `train.py`

Purpose: the shared training loop, optimizer construction, and checkpoint
I/O both `train_backdoor.py` and `train_benign.py` delegate to — deliberately
agnostic to which attack (if any) produced the training data. Docstring:
this module "takes any `(train_loader, val_loader)` whose training set
already carries the trigger and the correct labels" — poisoning is entirely
the caller's concern, this file only trains. Also the module that makes
Swin runs possible at all (BackdoorBench ships no Swin backdoored
checkpoints, so Swin needs models trained here) and where the SAM optimizer
is actually wired into a training loop (`sam.py` itself is optimizer-only,
no loop).

### Functions

**`build_model(architecture, num_classes) -> nn.Module`** (`train.py:28-33`)
- Dispatches to `models.build_vit`/`build_swin`; raises `ValueError` for
  anything else. A **fresh**, ImageNet-pretrained-backbone model each call
  (not a checkpoint load) — this is training-from-pretrained, not
  fine-tuning a saved checkpoint.
- Called by `train_classifier` (`train.py:180`).

**`build_optimizer(model, use_sam, learning_rate, weight_decay, rho) ->
torch.optim.Optimizer`** (`train.py:36-58`)
- If `use_sam`: `SAM(model.parameters(), torch.optim.Adam, rho=rho,
  lr=learning_rate, weight_decay=weight_decay)`. Else: plain
  `torch.optim.Adam(model.parameters(), lr=learning_rate,
  weight_decay=weight_decay)`. Docstring: Adam is the base optimizer in
  **both** branches, so the only difference between a vanilla and a SAM run
  is the sharpness-aware two-step — this is the concrete implementation of
  CLAUDE.md's "SAM is always SAM-on-top-of-AdamW" checkpoint-naming
  convention (the code says `torch.optim.Adam`, not `AdamW` — see Sharp
  edges).
- Called by `train_classifier` (`train.py:182`).

**`_plain_update(model, images, labels, criterion, optimizer) -> torch.Tensor`**
(`train.py:61-66`, private)
- Standard single-pass step: `zero_grad()` → `loss = criterion(model(images),
  labels)` → `loss.backward()` → `optimizer.step()`. Returns `loss` (still
  attached to the graph at return time, but the caller only reads
  `.item()`).
- Called by `train_one_epoch` (`train.py:87`, via the `update` variable)
  when `use_sam` is `False`.

**`_sam_update(model, images, labels, criterion, optimizer) -> torch.Tensor`**
(`train.py:69-79`, private)
- The two-pass SAM step: first forward+backward at current weights,
  `optimizer.first_step(zero_grad=True)` (ascends to the local worst-case
  point, per `sam.py:SAM.first_step`); **second**, fresh forward+backward
  (`criterion(model(images), labels).backward()`) computed at the now-
  ascended weights; `optimizer.second_step(zero_grad=True)` (restores
  original weights, applies the base Adam update using the gradient from
  the ascended point, per `sam.py:SAM.second_step`). Returns the **first**
  pass's `loss` (the loss at the pre-ascent weights, not the worst-case
  loss) for logging.
- Comment (`train.py:71-72`) explains why this is safe for ViT/Swin
  specifically: both use LayerNorm, not BatchNorm, so running two forward
  passes per step carries none of the BatchNorm running-statistics hazard
  SAM has on BatchNorm-based architectures (where a second forward pass at
  perturbed weights would corrupt the running mean/variance estimates).
- Called by `train_one_epoch` (`train.py:87`) when `use_sam` is `True`.

**`train_one_epoch(model, loader, criterion, optimizer, device, use_sam) ->
float`** (`train.py:82-89`)
- `model.train()`; picks `update = _sam_update if use_sam else
  _plain_update` once per epoch (not per batch — a single dispatch,
  reused for every batch in the loop); iterates `loader`, moving
  `images`/`labels` (`.long()`) to `device`, accumulating
  `update(...).item()` into `running_loss`. Returns
  `running_loss / max(len(loader), 1)` (guards divide-by-zero on an empty
  loader).
- No autocast/bfloat16 anywhere in this function or `_plain_update`/
  `_sam_update` — training always runs in the model's default dtype
  (float32), unlike the eval-side `forward_probs` path. `use_bfloat16` is a
  parameter of `train_classifier` but is **never threaded into
  `train_one_epoch`** — see Sharp edges.
- Called by `train_classifier` (`train.py:185`).

**`current_git_commit() -> str | None`** (`train.py:92-99`)
- `subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
  text=True, check=True)`, returns `stdout.strip()`; returns `None` on
  `CalledProcessError` or `FileNotFoundError` (git not installed, or not
  run inside a repo) rather than raising — a best-effort provenance field.
- Called by `checkpoint_metadata` (`train.py:136`).

**`checkpoint_metadata(dataset, attack, label_mode, target_label,
poison_rate, cover_rate, architecture, use_sam, rho, epochs, seed,
clean_accuracy, asr, started_at, ended_at) -> dict`** (`train.py:102-141`)
- Builds the exact `args.json` dict — the concrete implementation of
  CLAUDE.md's "Checkpoint naming and metadata" correctness rule listing
  `dataset, attack, label_mode, target_label, poison_rate, cover_rate,
  architecture, optimizer, rho, epochs, seed, git_commit,
  trained_started_at, trained_ended_at` (plus `clean_accuracy`/`asr` here,
  which the CLAUDE.md summary doesn't enumerate but the actual dict
  includes). `"optimizer": "sam" if use_sam else "adam"`; `"rho": rho if
  use_sam else None` (rho is only meaningful/recorded for SAM runs,
  explicitly nulled for plain Adam — matches the checkpoint-folder-naming
  convention where only SAM runs get a `_sam_rho_*` tag). `git_commit` is
  computed fresh via `current_git_commit()` inside this call, not passed
  in.
- Docstring: both `train_backdoor.py` and `train_benign.py` build this the
  same way, "so the key set never drifts between the two entrypoints" —
  the single source of truth for the sidecar's schema.
- Called by `train_backdoor.py:179`, `train_benign.py:98`.

**`utc_timestamp() -> str`** (`train.py:144-145`)
- `datetime.now(timezone.utc).isoformat()`. Called twice per training run
  by each entrypoint (`started_at`/`ended_at`), not by this module itself.
- Called by `train_backdoor.py:152,169`, `train_benign.py:69,84`.

**`save_checkpoint(model, num_classes, path, metadata=None) -> None`**
(`train.py:148-163`)
- `os.makedirs(dirname(path), exist_ok=True)`; `torch.save({"model":
  model.state_dict(), "num_classes": num_classes}, path)` — the exact
  BackdoorBench `attack_result.pt` format `models.py:_load_checkpoint_into`
  reads back (`checkpoint.get("model", checkpoint)` /
  `checkpoint["num_classes"]`). If `metadata` is truthy, writes it as
  `dirname(path)/args.json` (pretty-printed, `indent=2`) — a **separate**
  file from the `.pt`, specifically so metadata can be read
  (`read_checkpoint_metadata`) without loading model weights.
- I/O: creates directories, writes two files.
- Called by `train_backdoor.py:175`, `train_benign.py:94`, directly by
  `tests/test_checkpoint_metadata.py`.

**`train_classifier(architecture, num_classes, train_loader, val_loader,
device, epochs, use_sam, learning_rate=1e-4, weight_decay=1e-4, rho=0.1,
use_bfloat16=True) -> nn.Module`** (`train.py:166-193`)
- Steps: (1) `model = build_model(architecture, num_classes).to(device)`;
  (2) `criterion = nn.CrossEntropyLoss()`; (3) `optimizer =
  build_optimizer(...)`; (4) for `epoch in range(1, epochs+1)`:
  `train_one_epoch(...)` then `validation_accuracy =
  clean_accuracy(model, val_loader, device, use_bfloat16)`
  (`defences/detection.clean_accuracy`, which **does** run under bfloat16
  autocast via `forward_probs`, unlike training itself); prints
  `f"epoch {epoch}: loss={average_loss:.4f} val_acc={validation_accuracy:.4f}"`
  (side effect: stdout, the only per-epoch progress signal — no logging
  framework, no returned history). Returns the trained `model` after all
  epochs complete — no early stopping, no best-checkpoint selection, no
  checkpoint saved mid-training (saving is the caller's responsibility, via
  `save_checkpoint` after this returns).
- Does **not** seed anything internally — reproducibility depends entirely
  on the caller having called `lightning.seed_everything` beforehand
  (`train_backdoor.py`/`train_benign.py` both do, before building loaders).
- Called by `train_backdoor.py:159`, `train_benign.py:74`, directly by
  `tests/test_attack_triggers.py:197` (there, training briefly on a tiny
  subset to check a trigger is actually learnable end-to-end, not to
  produce a real checkpoint).

### Interactions

Imports `defences.detection.clean_accuracy`, `models.{build_swin,
build_vit}`, `sam.SAM` — no `poison.py`, no `attacks/` import at all,
consistent with the module's stated agnosticism to how the training data
was poisoned. This is the one shared core both `train_backdoor.py` and
`train_benign.py` route through for the actual model-fitting and
checkpoint-writing steps; everything upstream of `train_classifier` (data
loading, poisoning, loader construction) is each entrypoint's own
responsibility.

### Sharp edges

- **`use_bfloat16` on `train_classifier` only affects the per-epoch
  validation accuracy computation, not training itself.** `train_one_epoch`
  and both `_plain_update`/`_sam_update` never receive or read
  `use_bfloat16` — every training forward/backward pass runs at the
  optimizer's/model's default dtype regardless of this flag. A caller
  expecting `use_bfloat16=True` (the default) to speed up or change the
  precision of the actual training compute, not just the printed
  per-epoch validation number, would be mistaken — this is CLAUDE.md's
  documented bfloat16-for-forward-passes-only intent
  (`utils/config.py:64-68`), but the *scope* of "forward pass" here is
  narrower than "all forward passes in this file" — training's forward
  passes are excluded.
- `checkpoint_metadata`'s base optimizer is literally `torch.optim.Adam`
  (`train.py:51,56`), not `torch.optim.AdamW` — CLAUDE.md's checkpoint-
  naming convention describes "SAM is always SAM-on-top-of-AdamW," but the
  code implements SAM-on-top-of-Adam (no decoupled weight decay). This is
  either a documentation/code mismatch worth resolving, or CLAUDE.md's
  prose is using "AdamW" loosely to mean "Adam with a weight_decay
  argument" (which `torch.optim.Adam` also accepts, just with L2
  regularization folded into the gradient rather than decoupled from it,
  the actual mathematical difference between Adam-with-weight-decay and
  AdamW). Either way, `"optimizer": "sam" if use_sam else "adam"` in the
  saved metadata reports `"adam"`, never `"adamw"`.
- `_sam_update` returns the **first**-pass loss (pre-ascent), used for the
  epoch's running-loss average and printed log — this is a reasonable
  choice (the pre-ascent loss is the one comparable to a plain-Adam run's
  loss at the same weights) but means the printed/logged SAM training loss
  is *not* the worst-case loss SAM is actually minimizing against; a reader
  comparing SAM vs. non-SAM loss curves should know both are the
  "ordinary" loss at current weights, not SAM's internal objective.
- `save_checkpoint`'s `if metadata:` check treats an **empty dict**
  (`metadata={}`) as falsy, silently skipping the `args.json` write — a
  caller passing an intentionally-empty-but-present metadata dict (as
  opposed to `None`, meaning "no metadata") would get no sidecar file
  written and no error, indistinguishable at the call site from having
  passed `None`.
- `train_classifier` has no early stopping, checkpoint-per-epoch, or
  best-validation-accuracy tracking — CLAUDE.md's "uniform 15 epochs
  across all training runs" correctness rule is enforced entirely by every
  caller passing `epochs=15` consistently, not by anything in this
  function defaulting to or capping at that value (there is no default for
  `epochs`; it's a required parameter here).

---

## `train_backdoor.py`

Purpose: the entrypoint that poisons a dataset and trains a backdoored
ViT-B/16 or Swin-S, following BackdoorBench's flow (load clean data, poison
a fraction of the training set, train, evaluate ASR/clean accuracy, save in
`attack_result.pt` format). A CLI script (`if __name__ == "__main__":
main()`), run per CLAUDE.md as `python train_backdoor.py ...` inside an
activated venv within a PBS batch job.

### Functions

**`working_resolution(dataset_name) -> int`** (`train_backdoor.py:39-42`)
- Same body as `defences/checkpoint_eval.py`'s copy (see that section's
  Sharp edges for the duplication note): `64 if dataset_name == "tiny" else
  32`.

**`base_transform(image_size) -> transforms_v2.Compose`**
(`train_backdoor.py:45-49`)
- `Compose([Resize((image_size,image_size)), ToTensor()])` — stops at 0-to-1
  pixel values, comment states explicitly this is so the trigger can be
  applied before normalizing. Another independent instance of the pattern
  discussed in `utils/datasets.py`'s Sharp edges.
- Called by `build_poisoned_loaders` (`train_backdoor.py:82`).

**`resolve_config(attack_name, poisoned_dir)`** (`train_backdoor.py:52-55`)
- If `attack_name == "generated"`, returns `GeneratedConfig(poisoned_dir=
  poisoned_dir)` (the one attack whose config has no zero-arg default,
  per `attacks/__init__.py`'s registry); else `default_config(attack_name)`
  — every other attack's default dataclass instance.
- Called by `build_poisoned_loaders` (`train_backdoor.py:88`).

**`build_training_set(train_clean, attack, config, poison_rate, seed,
normalize, num_classes)`** (`train_backdoor.py:58-77`)
- Steps: (1) `labels = extract_labels(train_clean)`; (2)
  `cover_rate = getattr(config, "cover_rate", 0.0)`, `source_classes =
  getattr(config, "source_classes", None)` — duck-typed reads, since only
  `AdaptiveBlendConfig` and `TactConfig` define these fields, every other
  config dataclass doesn't have them at all (not even set to a default of
  `0.0`/`None` on the class); (3) if `cover_rate > 0.0` **or**
  `source_classes is not None`, routes through
  `choose_indices_with_cover`/`CoverPoisonedTrainingSet`; else the plain
  `choose_poison_indices`/`PoisonedTrainingSet` path.
- Docstring: "route to the cover-sample dataset when the attack config asks
  for it" — this function is the actual mechanism behind CLAUDE.md's stated
  behavior "attacks that use cover samples (adaptive_blend, tact) are
  detected from their config."
- Called by `build_poisoned_loaders` (`train_backdoor.py:91`).

**`build_poisoned_loaders(args, image_size)`** (`train_backdoor.py:80-123`)
- Steps: (1) `spec = DATASET_REGISTRY[args.dataset]`; `transform =
  base_transform(image_size)`; (2) `train_clean, test_clean =
  load_clean_datasets(args.dataset, transform, args.raw_data_dir)`; (3)
  `normalize = Normalize(mean=spec.mean, std=spec.std)`; (4) `config =
  resolve_config(args.attack, args.poisoned_dir)`; `attack =
  build_attack(args.attack, config, image_size, args.target_label)`; (5)
  `poisoned_train = build_training_set(train_clean, attack, config,
  args.poison_rate, args.seed, normalize, spec.num_classes)`; (6)
  `clean_test = PoisonedTrainingSet(test_clean, attack, set(), normalize,
  spec.num_classes)` — empty poison-index set, i.e. a plain normalized
  clean-test wrapper (same reuse pattern seen throughout the codebase); (7)
  `test_labels = extract_labels(test_clean)`; `backdoor_test =
  AttackSuccessSet(test_clean, test_labels, attack, normalize,
  spec.num_classes)`; (8) builds three `DataLoader`s via a local `loader`
  closure — `poisoned_train` **shuffled**, `clean_test`/`backdoor_test`
  **not** shuffled — with `batch_size=args.batch_size,
  num_workers=args.num_workers`.
- Returns a 6-tuple: `(train_loader, clean_loader, backdoor_loader,
  num_classes, attack, config)` — `attack` and `config` are returned
  alongside the loaders specifically so `main()` can read
  `attack.label_mode` and `config.cover_rate` later, when building the
  `args.json` metadata.
- Called by `main()` (`train_backdoor.py:155-157`).

**`parse_args() -> argparse.Namespace`** (`train_backdoor.py:126-144`)
- Required: `--dataset` (choices = `DATASET_REGISTRY` keys), `--attack`
  (choices = `ATTACK_NAMES`), `--poison-rate`, `--output`. Defaults:
  `--target-label 0`, `--architecture vit`, `--epochs 15` (CLAUDE.md's
  uniform-15-epochs convention, as a default only — a caller can still
  override it, nothing enforces 15), `--batch-size 128`, `--use-sam`
  (flag, `store_true`), `--rho 0.1`, `--poisoned-dir ""` (only meaningful
  for the `generated` attack), `--raw-data-dir "raw_data"`, `--seed 0`,
  `--num-workers 8`.

**`main() -> None`** (`train_backdoor.py:147-204`)
- Steps: (1) `args = parse_args()`; (2) `seed_everything(args.seed)` —
  Lightning's global seed, called **once**, before any data loading or
  model construction, so poisoning-index selection, model weight
  initialization order (irrelevant here since weights come from a
  pretrained checkpoint, but dropout/augmentation randomness if any would
  be seeded too), and training all derive from this single seed; (3)
  `device = cuda if available else cpu`; (4) timing/provenance:
  `start = time.time()`, `started_at = utc_timestamp()`; (5) `image_size =
  working_resolution(args.dataset)`; (6) `build_poisoned_loaders(args,
  image_size)` → the 6-tuple; (7) `model = train_classifier(args.architecture,
  num_classes, train_loader, clean_loader, device, epochs=args.epochs,
  use_sam=args.use_sam, rho=args.rho)` — note **`clean_loader` doubles as
  the validation loader** passed into `train_classifier`'s `val_loader`
  parameter, so the per-epoch "val_acc" `train_classifier` prints is clean
  accuracy on the poisoned model's held-out clean test set, not a
  backdoor-aware metric; (8) `ended_at = utc_timestamp()`; (9) final eval:
  `asr = attack_success_rate(model, backdoor_loader, device,
  use_bfloat16=True)`, `ca = clean_accuracy(model, clean_loader, device,
  use_bfloat16=True)`, printed; (10)
  `save_checkpoint(model, num_classes, args.output, metadata=
  checkpoint_metadata(...))` — passes `dataset=args.dataset,
  attack=args.attack, label_mode=attack.label_mode` (the **built** attack's
  resolved label mode, not necessarily `args.attack`'s naive default — for
  `badnet_a2a` this correctly resolves to `"all_to_all"` via the registry's
  alias mechanism), `target_label=args.target_label,
  poison_rate=args.poison_rate` (the **requested** rate, not the realized
  count — see `poison.py`'s Sharp edges on this same distinction),
  `cover_rate=getattr(config, "cover_rate", 0.0)`,
  `architecture=args.architecture, use_sam=args.use_sam, rho=args.rho,
  epochs=args.epochs, seed=args.seed, clean_accuracy=ca, asr=asr,
  started_at=started_at, ended_at=ended_at`; (11) prints save confirmation
  and total elapsed minutes.
- I/O: writes `args.output` (the `.pt` file) and its sibling `args.json`.
- Randomness: fully determined by `args.seed` via the single
  `seed_everything` call at the top.

### Interactions

Imports from `attacks` (registry + `attacks.generated.GeneratedConfig`
directly), `utils.config`, `utils.datasets`, `defences.detection`, `poison`,
and `train` (the shared training core) — the entrypoint that ties every
package together for the backdoor-training path. Does **not** import
`utils.datasets.build_transform` (defines its own `base_transform`
instead, per the pattern noted in that module's section).

### Sharp edges

- **`--output` is a free-form required path, not derived from a naming
  template.** Unlike `train_benign.py` (which builds `folder_name` itself
  from `architecture`/`dataset`/`benign`/sam-rho-tag), `train_backdoor.py`
  places the entire responsibility for matching CLAUDE.md's canonical
  `checkpoints/` folder-name template
  (`{architecture}_{dataset}_{attack}[_{poison_rate_tag}][_sam_rho_{rho_tag}]`)
  on whatever invokes this script (a PBS job script under `pbs/`, per the
  example in the module docstring). Nothing here validates `--output`
  against the template — a mistyped or non-conforming `--output` path
  produces a checkpoint that trains and evaluates fine but sits in
  `checkpoints/` under a name `metrics.py`/`scratch/normalize_checkpoints.py`
  wouldn't parse consistently with every other run.
- `attack.label_mode` (used for the saved `label_mode`) is read from the
  **built** `Attack` object, so it correctly reflects an alias resolution
  like `badnet_a2a → "all_to_all"`; `args.attack` itself (the saved
  `"attack"` field) stays as the literal CLI string (e.g. `"badnet_a2a"`,
  not `"badnet"`) — the two fields are deliberately not redundant, but a
  reader assuming `attack` alone determines `label_mode` via
  `utils.config.label_mode_from_folder`-style string parsing would be
  wrong for this repo's own checkpoints; that folder-name-parsing fallback
  is exclusively for `backdoor_bench_checkpoints/`, which have no
  `args.json` at all.
- `build_training_set`'s `getattr(config, "cover_rate", 0.0)` /
  `getattr(config, "source_classes", None)` duck-typing means adding a new
  attack config field with either of those exact names to any *other*
  attack's config dataclass (even accidentally, e.g. copy-pasting from
  `TactConfig`) would silently route that attack through the cover-sample
  training path, even if the attack's own trigger logic has no
  cover-sample awareness.
- `seed_everything(args.seed)` is called exactly once, before
  `build_poisoned_loaders` — this single call is what makes
  `choose_poison_indices`'/`choose_indices_with_cover`'s own internal
  `np.random.default_rng(seed)` calls redundant for *cross-run*
  reproducibility (they're separately seeded by the explicit `seed`
  parameter already), but it's what seeds anything *without* an explicit
  seed parameter downstream (e.g. `DataLoader(shuffle=True)`'s batch
  ordering for `poisoned_train`) — training-batch order is reproducible
  only because of this single global call, not because of any per-call
  seeding in the loader construction itself.

---

## `train_benign.py`

Purpose: trains benign (unpoisoned) ViT-B/16 (or Swin-S) models — the
negative controls for detection and latent-analysis experiments, models
that never saw a trigger. Also the module `metrics.py` imports
`build_clean_loaders` from for its own benign-eval loader construction
(cross-entrypoint reuse, not just a training script).

### Functions

**`working_resolution(dataset_name) -> int`** (`train_benign.py:32-35`)
- Same body as the other three copies (see `defences/checkpoint_eval.py`'s
  Sharp edges). Comment here specifically frames it as matching how
  backdoored models are trained "so their clean accuracy stays comparable"
  — an explicit cross-entrypoint consistency requirement satisfied only by
  the four copies staying byte-identical.

**`build_clean_loaders(dataset_name, raw_data_dir, batch_size,
num_workers=8) -> tuple[DataLoader, DataLoader, int]`**
(`train_benign.py:38-59`)
- Builds its own `Compose([Resize, ToTensor, Normalize])` transform
  (**with** normalization baked in, unlike `train_backdoor.py`'s
  `base_transform` — correct here since benign training never applies a
  pixel-space trigger, so there's no ordering constraint forcing
  normalization to happen separately/later); `load_clean_datasets(...)`;
  wraps both splits in `DataLoader`s (`train_loader` shuffled, `test_loader`
  not), both with `num_workers`. Returns `(train_loader, test_loader,
  spec.num_classes)`.
- Called by `train_one_benign` (`train_benign.py:70`), and by
  `metrics.py:evaluate_benign` (`metrics.py:116`) — the one place a
  training-script function is imported and reused by the separate
  `metrics.py` eval entrypoint, rather than `metrics.py` reimplementing
  clean-loader construction itself the way it reimplements the poisoned-eval
  transform pattern.

**`train_one_benign(dataset_name, args, device) -> float`**
(`train_benign.py:62-118`)
- Steps: (1) `seed_everything(args.seed)` — called **per dataset, inside
  the per-dataset function**, not once before the outer loop; comment
  explains explicitly: "so each dataset's run is reproducible independent
  of loop order or an earlier dataset's failure" (contrast with
  `train_backdoor.py`'s single top-level seed call — this file seeds
  once per iteration since `main()` loops over multiple datasets in one
  process); (2) timing/provenance start; (3)
  `build_clean_loaders(dataset_name, args.raw_data_dir, args.batch_size,
  args.num_workers)`; (4) `train_classifier(args.architecture, num_classes,
  train_loader, test_loader, device, epochs=args.epochs,
  use_sam=args.use_sam, rho=args.rho)` — here `test_loader` doubles as both
  the validation loader during training *and* the final clean-accuracy eval
  set (the same held-out split serves both roles, since there's no
  train/val/test three-way split for benign training, only train/test);
  (5) `ended_at = utc_timestamp()`; `accuracy = clean_accuracy(model,
  test_loader, device, use_bfloat16=True)` (a **second**, separate forward
  pass over the same `test_loader` the last training epoch already
  evaluated — see Sharp edges); (6) builds `folder_name` from the
  canonical template directly: `f"{architecture}_{dataset_name}_benign"`,
  appending `f"_sam_rho_{str(rho).replace('.', '_')}"` when `use_sam` (the
  underscore-before-digits convention CLAUDE.md's naming rule requires,
  implemented here via a literal `.replace('.', '_')` on the float's string
  form); (7) `output_path = f"{weights_dir}/{folder_name}/attack_result.pt"`;
  (8) `save_checkpoint(...)` with `checkpoint_metadata(dataset=dataset_name,
  attack="benign", label_mode=None, target_label=0, poison_rate=0.0,
  cover_rate=0.0, ..., asr=None, ...)` — the benign-specific metadata
  convention: `attack="benign"` is what `metrics.py:process_checkpoints_folder`
  branches on (`if args["attack"] == "benign":`) to route to
  `evaluate_benign` instead of `evaluate_attack_from_args`; (9) prints
  accuracy and elapsed time; returns `accuracy`.
- Called by `main()` (`train_benign.py:149`), once per dataset in
  `args.datasets`.

**`parse_args() -> argparse.Namespace`** (`train_benign.py:121-138`)
- `--datasets` (`nargs="+"`, default `["cifar10", "cifar100", "gtsrb",
  "tiny"]` — all four registered datasets by default, unlike
  `train_backdoor.py`'s single required `--dataset`), `--epochs 15`,
  `--batch-size 128`, `--architecture vit`, `--use-sam`, `--rho 0.1`,
  `--weights-dir "checkpoints"`, `--raw-data-dir "raw_data"`,
  `--num-workers 8`, `--seed 0`.

**`main() -> None`** (`train_benign.py:141-161`)
- Loops over `args.datasets`; for each, calls `train_one_benign` inside a
  `try/except Exception`, printing `f"FAILED {dataset_name}_benign:
  {error}"` and continuing rather than aborting the whole multi-dataset run
  — comment explains this explicitly ("one dataset failing should not
  waste the datasets after it"). Collects successful `accuracies` into a
  dict, prints a final summary. **This is the one entrypoint in the repo
  with a broad `except Exception` that swallows and continues** — every
  other entrypoint (`train_backdoor.py`, `metrics.py`) either lets
  exceptions propagate or (in `metrics.py`'s per-checkpoint loop) has an
  analogous but differently-scoped catch (see `metrics.py`'s section).

### Interactions

Imports `utils.config`, `utils.datasets`, `defences.detection.clean_accuracy`,
`train` — notably **no** `attacks`/`poison` imports at all, since benign
training never touches the attack machinery. `metrics.py` imports
`build_clean_loaders` from this file directly, making `train_benign.py` a
dependency of `metrics.py`, not merely a sibling entrypoint (documented in
the `metrics.py` section's Interactions).

### Sharp edges

- `train_one_benign`'s final `clean_accuracy(model, test_loader, ...)`
  call recomputes accuracy that `train_classifier`'s last epoch already
  computed and printed (`train.py:188`) — a second, redundant full forward
  pass over `test_loader`, done so the returned/saved value comes from a
  call outside `train_classifier`'s internal loop (which doesn't return
  its per-epoch accuracy values, only the model) rather than a cache-and-
  reuse optimization; a caller who assumed the printed per-epoch "val_acc"
  and the saved `clean_accuracy` metadata field are computed by the exact
  same call would be right about the loader and method but wrong about the
  call being deduplicated.
- **Per-dataset `seed_everything` inside a multi-dataset loop** means the
  RNG state at the start of training dataset N+1 is *not* whatever state
  dataset N's training left it in — it's freshly reset to
  `seed_everything(args.seed)` again, using the **same** `args.seed` for
  every dataset in one `--datasets` run (there is no per-dataset seed
  offset). This is deliberate (the comment explains why), but it does mean
  every dataset in a single multi-dataset invocation trains with
  identically-seeded initialization/shuffling *modulo* whatever randomness
  the dataset's own size/content introduces — not independently-seeded
  runs in the sense of using different seeds per dataset.
- The `except Exception` in `main()`'s loop catches and continues past
  **any** failure, including ones a caller might want to stop the whole
  run for (e.g. a missing `raw_data_dir`, an out-of-memory error, a
  corrupted download) — the failure is printed but the script's exit code
  is still 0 (success) even if every single dataset failed, since nothing
  re-raises or tracks failure count into a non-zero exit. A caller relying
  on this script's exit code to detect any failure (e.g. a PBS job's
  post-processing step) would need to parse stdout for `"FAILED"` lines
  instead.

---

## `analysis/analyze_latent.py`

Purpose: the worked-example CLI entrypoint tying together every other
`analysis/` module — builds paired clean/triggered versions of the same test
images, extracts per-layer CLS features from a trained checkpoint, and
reports where the trigger's representation lives (backdoor direction norm,
TAC, CKA) and how separable clean/backdoor representations are (PCA
scatter). Module docstring gives the expected qualitative outcome to sanity-
check results against: a benign checkpoint should show small TAC everywhere
(never learned a trigger); a backdoored checkpoint should show TAC and
direction norm rising at the layer carrying the backdoor. Must be run as
`python -m analysis.analyze_latent` (a module, not a bare script), since it
mixes relative imports of its own package siblings with absolute imports of
`attacks/`, `defences/`, `utils/`.

### Functions

**`working_resolution(dataset_name) -> int`** (`analysis/analyze_latent.py:43-44`)
- Fifth independent copy of the same one-liner (see
  `defences/checkpoint_eval.py`'s Sharp edges for the other three) — this
  one lacks even the explanatory comment the other three carry.

**`build_paired_loaders(dataset_name, attack, raw_data_dir, batch_size,
sample_count, seed)`** (`analysis/analyze_latent.py:47-77`)
- Steps: (1) loads the 0-to-1 base test set via a local
  `Compose([Resize, ToTensor])` transform (yet another independent instance
  of the no-normalize-yet pattern); (2)
  `rng = np.random.default_rng(seed)`; `chosen = rng.choice(len(test_clean),
  size=min(sample_count, len(test_clean)), replace=False)` — a random
  sample of test indices, capped at the dataset size; (3)
  `subset = Subset(test_clean, chosen.tolist())`; (4)
  `clean_set = PoisonedTrainingSet(subset, attack, set(), normalize, ...)`
  (empty poison set → plain normalized clean wrapper, the recurring pattern)
  and `triggered_set = PoisonedTrainingSet(subset, attack,
  set(range(len(subset))), normalize, ...)` — **poisons every index in
  `subset`** by passing the full range as the poison-index set, unlike
  `AttackSuccessSet`'s eligibility-filtered approach; (5) both wrapped in
  unshuffled `DataLoader`s.
- Returns `(clean_loader, backdoor_loader)`, index-aligned (position `i` in
  each loader is the same original test image, clean vs. triggered) —
  docstring states explicitly this pairing is what TAC and the backdoor
  direction require.
- Called by `main()` (`analysis/analyze_latent.py:165`).

**`per_layer_report(clean_features, backdoor_features) -> dict[int, float]`**
(`analysis/analyze_latent.py:80-96`)
- For each layer index (sorted, so embedding layer 0 first through block 12
  last): computes `direction = backdoor_direction(clean_features[layer],
  backdoor_features[layer])`, `tac = trigger_activated_change(...)`,
  `cka_value = debiased_linear_cka(...)`; stores
  `direction_norms[layer] = direction.norm().item()`; prints a formatted row
  `layer, direction_norm, max_tac (tac.max().item()), cka`. Side effect:
  stdout table, header printed once before the loop
  (`analysis/analyze_latent.py:86`).
- Returns `direction_norms` (only the norms, not the TAC or CKA values —
  those are printed but not returned, so a caller wanting them
  programmatically would need to recompute or restructure this function).
- Docstring: rising direction norm and TAC mark where the trigger becomes
  dominant; falling CKA marks where clean/backdoor representations diverge.
- Called by `main()` (`analysis/analyze_latent.py:186`).

**`save_pca_scatter(clean_features, backdoor_features, layer, output_path) ->
None`** (`analysis/analyze_latent.py:99-130`)
- Steps: (1) `combined = cat([clean_features[layer],
  backdoor_features[layer]], dim=0)` — clean rows first, backdoor rows
  appended; (2) `projected = pca_project(combined, num_components=2)`; (3)
  `clean_count = clean_features[layer].shape[0]`; (4) two `plt.scatter`
  calls, slicing `projected[:clean_count]` (labeled "clean") and
  `projected[clean_count:]` (labeled "backdoor") — relies on the
  concatenation order from step 1 to correctly split the projected points
  back into their original groups; (5) titles, legend, `tight_layout()`,
  `plt.savefig(output_path, dpi=200)`, prints confirmation.
- I/O: writes an image file to `output_path`. Side effect: mutates the
  global matplotlib figure state (`plt.figure(...)`, not scoped to a
  returned `Figure` object — a second call to this function in the same
  process would create a second figure via `plt.figure()`, not reuse or
  clear the first, though nothing in this script calls it twice).
- Called by `main()` (`analysis/analyze_latent.py:190`).

**`parse_args() -> argparse.Namespace`** (`analysis/analyze_latent.py:133-151`)
- Required: `--dataset`, `--attack` (any registry name, "the trigger to
  probe with, for example badnet_a2o" — explicitly framed as choosing what
  trigger to *test the model against*, independent of what the checkpoint
  was actually trained with, since a benign checkpoint has no attack of its
  own to read), `--checkpoint`. Defaults: `--architecture vit`,
  `--target-label 0`, `--samples 1000`, `--batch-size 64`, `--raw-data-dir
  "raw_data"`, `--output "latent_scatter.png"`, `--seed 0`.

**`main() -> None`** (`analysis/analyze_latent.py:154-194`)
- Steps: (1) `matplotlib.use("Agg")` — set **after** `parse_args()` but
  **before** any plotting call, comment explains this is for headless
  cluster nodes with no display; (2) `device = cuda if available else cpu`;
  (3) `image_size = working_resolution(args.dataset)`; `attack =
  build_attack(args.attack, default_config(args.attack), image_size,
  args.target_label)` — always the attack's **default** config, same
  limitation as `defences/checkpoint_eval.py:build_eval_loaders_from_checkpoint`
  (no way to probe with a non-default trigger configuration via CLI flags
  here); (4) `build_paired_loaders(...)`; (5) `model =
  load_checkpoint(args.architecture, args.checkpoint, device)`; (6)
  `reset_dropout(model, "pre_residual")` — comment: "dropout off, we want
  the clean baseline features"; hardcodes `"pre_residual"` as the
  placement argument regardless of what placement, if any, the checkpoint
  was ever evaluated under (see Sharp edges for why this is safe in
  practice); (7) `clean_features = extract_layer_features(model,
  clean_loader, device, use_bfloat16=True)`,
  `backdoor_features = extract_layer_features(model, backdoor_loader,
  device, use_bfloat16=True)` — two full passes over the model, once per
  loader; (8) `direction_norms = per_layer_report(...)`; `peak_layer =
  max(direction_norms, key=direction_norms.get)` — the layer with the
  single largest backdoor-direction norm; (9)
  `save_pca_scatter(clean_features, backdoor_features, peak_layer,
  args.output)` — only plots the peak layer, not every layer.
- Randomness: `args.seed` flows only into `build_paired_loaders`'s sample
  selection (`np.random.default_rng(seed)`) — no `seed_everything` call
  anywhere in this script, unlike both training entrypoints. Model loading
  and feature extraction are otherwise deterministic (`inference_mode`,
  dropout off).

### Interactions

Imports `attacks.{build_attack, default_config}` (absolute), its own
package siblings `.cka`, `.direction`, `.embedding`, `.features` (relative,
per CLAUDE.md's within-package convention), `utils.config`, `utils.datasets`
(absolute), `defences.dropout.reset_dropout` (absolute), `models.load_checkpoint`
(absolute), `poison.PoisonedTrainingSet` (absolute) — the only file in
`analysis/` that reaches into `poison.py` or `defences.dropout`, since it's
the only one that needs to build an actual poisoned dataset and manage
dropout state on a live model rather than operate on precomputed feature
tensors.

### Sharp edges

- **`triggered_set` in `build_paired_loaders` is built by poisoning every
  index via `PoisonedTrainingSet`, not by using `AttackSuccessSet`'s
  eligibility-filtered eval-time semantics.** For an `all_to_one` attack,
  this means already-target-class images in the sampled subset *do* get
  triggered here (and relabeled via training-time `poisoned_label`,
  a no-op label change for already-target images) — whereas
  `AttackSuccessSet` would have excluded them entirely as ineligible. This
  is harmless for this script's actual purpose (it never reads or uses the
  dataset labels, only the images, for feature extraction), but it means
  `backdoor_loader` here is **not** interchangeable with an eval-time
  `AttackSuccessSet`-based backdoor loader from `train_backdoor.py`/
  `metrics.py`/`defences/checkpoint_eval.py` — a different, larger sample
  population (every sampled index, not just eval-eligible ones).
- `reset_dropout(model, "pre_residual")`'s hardcoded placement argument is
  safe only because `load_checkpoint` always produces a **freshly
  constructed** `build_vit`/`build_swin` model with weights loaded via
  `load_state_dict` — dropout-placement wrapping
  (`defences/dropout.py:configure_post_residual_dropout`) is a runtime-only
  module-tree mutation never reflected in a saved checkpoint's state dict
  (checkpoints only ever come from `train.py:save_checkpoint`, called on an
  unwrapped model). If this script were ever changed to load a model that
  had already been through post-residual wrapping in the same process,
  hardcoding `"pre_residual"` here would skip the necessary unwrap step
  (`remove_post_residual_dropout`), leaving `PostResidualEncoderBlock`/
  `PostResidualSwinBlock` wrappers in place with only their rate zeroed,
  not actually removed.
- `per_layer_report` prints TAC's **max** value per layer
  (`tac.max().item()`) as the single scalar summary, discarding the
  per-dimension detail `trigger_activated_change` actually computes — a
  layer with one extreme outlier dimension and a layer with many
  moderately-elevated dimensions could print the same `max_tac` value,
  even though `outlier_dimensions`
  (`analysis/direction.py`, never called from this script) could
  distinguish them.
- `save_pca_scatter` only visualizes the single `peak_layer` (highest
  direction norm) — if the backdoor signal is actually spread across
  multiple layers, or peaks at a layer that isn't the global max by this
  particular metric (direction norm, not TAC or CKA), this script's one
  saved figure would miss it; a full sweep would require calling
  `save_pca_scatter` once per layer manually (not automated here).
- No `seed_everything` call anywhere in this file — only sample selection
  is explicitly seeded via `np.random.default_rng`. Any other source of
  nondeterminism in the forward pass (e.g. non-deterministic CUDA kernels,
  though dropout itself is off) is not controlled by this script the way
  `train_backdoor.py`/`train_benign.py` control theirs.

---

## `metrics.py`

Purpose: the baseline attack-success/clean-accuracy eval entrypoint for
every checkpoint under `checkpoints/` (explicitly **not**
`backdoor_bench_checkpoints/` — module docstring states it is "scoped to
`checkpoints/` only"). Decoupled from the PSBD dropout-sweep mechanism
(archived, pending rewrite): this script answers only "what is clean
accuracy (overall and per-class) for a benign model" and "what is ASR and
clean accuracy for an attacked model," writing `results/<folder>/
metrics.json`, distinct from the future `psbd_metrics.json` CLAUDE.md's
naming convention reserves for the PSBD rewrite's own output in the same
directory. Named `results/`, not `analysis/`, specifically to avoid
colliding with the `analysis/` source package.

### Module-level state

- `CHECKPOINTS_DIR = "checkpoints"` (`metrics.py:43`), `RESULTS_DIR =
  "results"` (`metrics.py:44`) — the two top-level directory constants this
  entire script operates over.

### Functions

**`list_checkpoint_folders(source_dir) -> list[str]`** (`metrics.py:47-53`)
- Returns `[]` if `source_dir` isn't a directory; else the sorted list of
  subdirectory names directly under it (not recursive).
- Called by `run` (`metrics.py:159`).

**`mirror_results_folders(results_dir, folder_names) -> None`**
(`metrics.py:56-72`)

> **Removed. This section describes a version of `metrics.py` that no longer
> exists.** `metrics.py` is now 88 lines, writes each checkpoint's `metrics.json`
> into `checkpoints/<folder_name>/` beside `attack_result.pt` and `args.json`
> (`metrics.py:77`), and touches `results/` not at all. There is no `shutil.rmtree`
> anywhere in the repo. `results/<folder_name>/` is created on demand by the sweep
> instead, so an orphaned results directory is now inert rather than deleted. The
> description below is kept only because later sections refer back to it.
- `os.makedirs(results_dir, exist_ok=True)`; computes `existing` (current
  subfolders of `results_dir`) vs. `wanted` (`set(folder_names)`); for every
  `orphan in existing - wanted`, `shutil.rmtree(...)` — **deletes** any
  `results/<name>/` with no matching `checkpoints/<name>/` (e.g. one
  `normalize_checkpoints.py` renamed away, per the docstring's example);
  then `os.makedirs` a folder for every wanted name, even ones not yet
  processed this run.
- I/O: recursive directory deletion, directory creation. This is the one
  destructive operation in the whole eval pipeline — invoked
  unconditionally at the start of every `run()` call, before any per-folder
  processing.
- Called by `run` (`metrics.py:160`).

**`write_metrics(results_dir, folder_name, metrics) -> None`**
(`metrics.py:75-79`)
- `os.makedirs(folder, exist_ok=True)`; writes `metrics` dict as
  `folder/metrics.json` (`indent=2`).
- Called by `run` (`metrics.py:168`).

**`read_args_json(checkpoint_dir) -> dict`** (`metrics.py:82-84`)
- Opens and JSON-loads `checkpoint_dir/args.json` — no existence check or
  descriptive error (unlike `defences/checkpoint_eval.py:read_checkpoint_metadata`'s
  explicit `FileNotFoundError` with a helpful message); a missing
  `args.json` here raises the raw `FileNotFoundError` from `open()`.
- Called by `process_checkpoints_folder` (`metrics.py:148`).

**`build_full_eval_loaders(dataset_name, attack, raw_data_dir, batch_size) ->
tuple[DataLoader, DataLoader]`** (`metrics.py:87-112`)
- Docstring is explicit about how this differs from
  `defences/checkpoint_eval.py`'s loaders: "clean and attack-success loaders
  over the **whole** test set, not a sweep-sized subset" — no validation
  hold-out, no per-class balancing, since this baseline metric has no need
  for either (those are sweep-specific concerns).
- Steps: local `base_transform = Compose([Resize, ToTensor])` (0-to-1,
  comment explicitly cross-references matching
  `defences/checkpoint_eval.py`'s in-memory eval path and states
  normalization happens "last, inside `PoisonedTrainingSet`/
  `AttackSuccessSet`, same as at training time" — yet another independent
  instance of the pattern, this time with an explicit comment acknowledging
  the duplication is intentional consistency, not oversight); loads the
  clean test set; `clean_eval = PoisonedTrainingSet(test_base, attack,
  set(), normalize, num_classes)` (empty-set clean wrapper, same reuse
  pattern everywhere else); `true_labels = extract_labels(test_base)`;
  `backdoor_eval = AttackSuccessSet(test_base, true_labels, attack,
  normalize, num_classes)`; both wrapped in unshuffled `DataLoader`s via a
  local `loader` closure.
- Called by `evaluate_attack_from_args` (`metrics.py:132`).

**`evaluate_benign(model, dataset_name, architecture, device, raw_data_dir,
batch_size) -> dict`** (`metrics.py:115-123`)
- `_, test_loader, num_classes = build_clean_loaders(...)` — imported from
  `train_benign.py`, the one cross-entrypoint dependency in this file
  (`metrics.py:41`); `correct, total = class_correct_and_total(model,
  test_loader, device, num_classes, use_bfloat16=True)` (single pass,
  comment: "shared between the pooled and per-class figures, not two" —
  the deliberate avoidance of a double pass noted in
  `defences/detection.py`'s Sharp edges); returns `{"architecture":
  architecture, "clean_accuracy": pooled_accuracy_from_counts(correct,
  total), "clean_accuracy_by_class": accuracy_by_class_from_counts(correct,
  total)}`.
- Called by `process_checkpoints_folder` (`metrics.py:152`), when
  `args["attack"] == "benign"`.

**`evaluate_attack_from_args(model, args, device, raw_data_dir, batch_size) ->
dict`** (`metrics.py:126-142`)
- Steps: `dataset_name = args["dataset"]`; `image_size =
  working_resolution(dataset_name)` (using the copy imported from
  `defences.checkpoint_eval`, per `metrics.py:29`); `attack =
  build_attack(args["attack"], default_config(args["attack"]), image_size,
  args["target_label"])` — **default** config again, same limitation
  discussed in `defences/checkpoint_eval.py`'s and
  `analysis/analyze_latent.py`'s sections: a checkpoint trained with
  non-default attack hyperparameters would be evaluated here against the
  wrong trigger, silently; `clean_loader, backdoor_loader =
  build_full_eval_loaders(...)`; returns a dict with `architecture,
  dataset, attack, label_mode, poison_rate, target_label` (all read
  straight from `args`, i.e. from the checkpoint's saved `args.json`, not
  recomputed) plus `asr` and `clean_accuracy` (computed fresh via
  `attack_success_rate`/`clean_accuracy`).
- Called by `process_checkpoints_folder` (`metrics.py:153`), for any
  non-benign attack.

**`process_checkpoints_folder(folder_name, device, raw_data_dir,
batch_size) -> dict`** (`metrics.py:145-153`)
- Steps: builds `checkpoint_dir`/`checkpoint_path`; `args =
  read_args_json(checkpoint_dir)`; `model =
  load_checkpoint(args["architecture"], checkpoint_path, device)`;
  branches on `args["attack"] == "benign"` to `evaluate_benign` else
  `evaluate_attack_from_args`.
- Called by `run` (`metrics.py:167`).

**`run(folder_filter, raw_data_dir, batch_size) -> None`** (`metrics.py:156-171`)
- Steps: (1) `device = cuda if available else cpu`; (2)
  `local_folders = list_checkpoint_folders(CHECKPOINTS_DIR)`; (3)
  `mirror_results_folders(RESULTS_DIR, local_folders)` — runs **before**
  filtering, so `results/` is mirrored to match *every* checkpoint folder
  on disk even if `folder_filter` restricts which ones get actually
  processed this invocation; (4) if `folder_filter is not None`, narrows
  `local_folders` to the intersection; (5) for each folder: wraps
  `process_checkpoints_folder` + `write_metrics` in a `try/except
  Exception`, printing `f"FAILED {folder_name}: {error}"` and continuing —
  same broad-catch-and-continue pattern as `train_benign.py:main`, for the
  same reason (one bad checkpoint shouldn't abort evaluating the rest);
  else prints the computed metrics dict.
- Side effects: the directory mirroring (destructive on orphans), one
  `metrics.json` write per successfully-processed folder, stdout logging.

**`parse_args() -> argparse.Namespace`** (`metrics.py:174-179`)
- `--folder` (`nargs="+"`, default `None` — process every checkpoint folder
  found), `--raw-data-dir "raw_data"`, `--batch-size 64` (default 64,
  distinct from `train_backdoor.py`'s 128 and `analyze_latent.py`'s 64 —
  eval-time batch size is independently chosen per script, not shared).

**`main() -> None`** (`metrics.py:182-184`)
- `run(args.folder, args.raw_data_dir, args.batch_size)`.

### Interactions

Imports `attacks.{build_attack, default_config}`,
`defences.checkpoint_eval.working_resolution`,
`defences.detection.{accuracy_by_class_from_counts, attack_success_rate,
class_correct_and_total, clean_accuracy, pooled_accuracy_from_counts}`,
`utils.config.DATASET_REGISTRY`, `utils.datasets.{extract_labels,
load_clean_datasets}`, `models.load_checkpoint`, `poison.{AttackSuccessSet,
PoisonedTrainingSet}`, and `train_benign.build_clean_loaders` — the single
widest-importing file in the repo, and the only one that imports from
another *entrypoint* script (`train_benign.py`) rather than only from the
shared packages, making `train_benign.py` a load-bearing dependency of
`metrics.py`, not just a sibling script.

### Sharp edges

- ~~**`mirror_results_folders` deletes `results/<name>/` directories
  unconditionally for any name in `results/` that has no matching
  `checkpoints/<name>/`, on every invocation, before any processing.**~~
  **No longer true: the function was removed and `metrics.py` never deletes
  anything.** The rest of this bullet described that removed behaviour. This
  runs even when `--folder` restricts processing to a subset — the deletion
  pass is **not** scoped by `folder_filter`, only the subsequent processing
  loop is. Renaming or temporarily moving a `checkpoints/` folder (e.g.
  mid-reorganization) and then running `metrics.py --folder
  some-other-folder` would still delete the now-orphaned `results/`
  counterpart for the moved folder, even though that folder wasn't touched
  by `--folder`'s filter.
- `evaluate_attack_from_args`'s use of `default_config(args["attack"])`
  reproduces the same silent-wrong-trigger risk documented in
  `defences/checkpoint_eval.py`'s and `analysis/analyze_latent.py`'s Sharp
  edges — a third independent site with the identical limitation (no
  recorded non-default attack config, so eval always reconstructs the
  attack's dataclass defaults regardless of what was actually used at
  training time). Since `train_backdoor.py` itself also only ever builds
  attacks from `default_config`/`resolve_config`, this is currently latent
  everywhere consistently, not actively wrong for any existing checkpoint —
  but any future code path that lets training use a non-default config
  (e.g. a `--patch-size` CLI flag) would silently desync all three of these
  eval sites at once, not just one.
- `read_args_json`'s bare `open(...)` gives a raw `FileNotFoundError` with
  Python's default message (just the path) if `args.json` is missing —
  inside `run()`'s broad `except Exception`, this becomes just another
  `FAILED {folder_name}: {error}` line, indistinguishable in the log from
  any other kind of failure (a corrupted checkpoint, an OOM, a bad attack
  name) without reading the exception text closely.
- `run()`'s per-folder `try/except Exception` means a systematic bug
  affecting every checkpoint (e.g. a broken import, a changed function
  signature) would still print one `FAILED` line per folder rather than
  failing fast on the first one — for a `checkpoints/` directory with
  hundreds of folders (559, per this repo's current `checkpoints/`
  listing), a systematic failure produces hundreds of near-identical error
  lines rather than surfacing the root cause immediately.
- `results/` folders are created for **every** checkpoint folder
  (`mirror_results_folders`, step 3 above) regardless of whether that
  folder's `metrics.json` ever actually gets written — a folder whose
  processing fails still has an empty `results/<name>/` directory sitting
  there with no `metrics.json` inside, which looks superficially like "not
  yet processed" and is indistinguishable from that state by directory
  listing alone (only the presence/absence of `metrics.json` inside
  distinguishes "failed" from "never attempted," and nothing surfaces that
  distinction outside the run's own stdout log).

---
