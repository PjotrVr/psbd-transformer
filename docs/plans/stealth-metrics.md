# Add stealth metrics (PSNR, SSIM, LPIPS) for attack checkpoints

## Context

Written before `evaluate.py`/`loaders.py` picked up `max_samples`/`seed` (`docs/plans/smoke-tests.md`) and before this pass confirmed exactly how `metrics.py`/`evaluate.py` divide responsibility. This revision fixes those references and adds one structural finding the original draft missed: stealth metrics don't vary per checkpoint, only per `(dataset, attack)`.

**Scope, confirmed**: attack checkpoints only. `evaluate.py:evaluate_checkpoint` (lines 115-151) already has two structurally separate branches — `if args["attack"] == "benign":` (122-130) returns early with no attack object in scope at all, `else:` (132-151) is where the new `stealth` block goes. There is no shared code path where a stealth computation could accidentally run for a benign checkpoint — the benign branch never builds an `Attack` object, so there is nothing to compute stealth against even by mistake.

**File/function naming fix**: the actual attack-vs-benign branching, `evaluate_attack` call, and JSON-shape construction all live in `evaluate.py`, not `metrics.py`. `metrics.py` is a thin CLI (`parse_args` + one call to `evaluate.evaluate_all_checkpoints`) with a `--folder` filter — it has no branches of its own to wire into. Every place the original plan said "metrics.py" below has been corrected to name the actual function in `evaluate.py`.

**Dependencies**: `torchmetrics>=1.9.0` and `lpips>=0.1.4` are already in `pyproject.toml`/`uv.lock` and importable in `.venv` — confirmed by import, nothing to add.

## A. The one structural fix: compute once per `(dataset, attack)`, not once per checkpoint

Checked directly rather than assumed: `attacks/badnet.py` and `attacks/lc.py`'s `apply_trigger` closures (lc is the interesting case — clean-label attacks can in principle depend on the target class) are both built purely from `config` and `image_size`; `target_label` is threaded through to the returned `Attack` object but never captured inside `apply_trigger` itself. Every other attack in scope follows the same shape (`build(config, image_size, target_label) -> Attack`, trigger closure over `config`/`image_size` only). So the pixel-space trigger — and therefore every stealth number — depends only on `(dataset_name, attack_name, attack_config, image_size)`. It does not depend on `poison_rate`, `target_label`, `seed`, `architecture`, SAM/rho, or epochs.

Concretely: `vit_cifar100_badnet_a2o_0_01`, `_0_05`, `_0_1`, and all four `_sam_rho_*` variants (8 checkpoints) share the exact same trigger and would get byte-identical stealth numbers. Across `checkpoints/`, this collapses roughly 400+ attack checkpoints down to on the order of 30-40 distinct `(dataset, attack)` pairs. LPIPS runs a real AlexNet forward pass per image — recomputing it 8-16x for what is the same trigger stamped on the same images is real, avoidable GPU time, not just a style nit.

**Design**: compute stealth once per `(dataset_name, attack_name)` the first time it's needed, cache in memory for the duration of one `metrics.py` run, and copy the same dict into every checkpoint's `metrics.json` that shares that `(dataset, attack)` pair. Duplication across each checkpoint's own JSON is intentional, not an oversight — keeps every checkpoint's metrics file self-contained (matches the "one checkpoint folder is a complete, backup-able unit" direction from the `checkpoints/`-vs-`results/` discussion), it's just computed once instead of once-per-file.

```python
_stealth_cache: dict[tuple[str, str], dict] = {}

def cached_stealth_metrics(dataset_name: str, attack_name: str, config, image_size: int, ...) -> dict:
    key = (dataset_name, attack_name)
    if key not in _stealth_cache:
        _stealth_cache[key] = compute_stealth_metrics(dataset_name, attack_name, config, image_size, ...)
    return _stealth_cache[key]
```

Module-level cache dict is fine here specifically because `metrics.py`'s loop over `checkpoints/` (§F) is a single-process, single-pass run — no concurrency, no long-lived server, process exits when the sweep finishes. The cache lives in `stealth.py`, not in whatever loops over checkpoints, so it works the same regardless of where that loop lives.

## B. Where the new code lives

Repo root, new file `stealth.py`, alongside `evaluate.py`/`metrics.py`/`poison.py`/`loaders.py` — not inside `attacks/`. Reasoning: every file currently in `attacks/` is one of the 10 registered attack implementations (`attacks/__init__.py`'s `_ATTACKS` registry) with a uniform one-file-per-attack shape; a stealth-metrics helper would be the first non-attack file there, breaking that pattern. It also doesn't fit `evaluate.py` (scoped tightly to model-behavior evaluation — "one model, one dataset, clean or under attack," per its own docstring; stealth needs no model or checkpoint at all, only an attack and raw images) or `defences/detection.py` (also model-behavior metrics, ASR/clean-accuracy, needs a forward pass). It's shared by both the attack pipeline and the eval pipeline without being owned by either — the same reasoning CLAUDE.md already gives for why `poison.py` stays at repo root.

## C. Preprocessing — grounded in the actual pipeline, not restated generically

`loaders.py:_load_test_base` already returns exactly the right raw material: 0-to-1 range float tensors, resized, **before** `Normalize` (`loaders.py:30-37` — `transforms_v2.Compose([Resize, ToTensor()])`, no `Normalize` in that pipeline). So `data_range=1.0` is a confirmed fact for this codebase, not something to verify per-call — the original plan's "confirm the actual pixel range... adjust if 0 to 255" hedge can be dropped. GTSRB's identity normalization (`mean=(0,0,0)`, `std=(1,1,1)`, CLAUDE.md) doesn't matter either way since stealth never touches `Normalize` at all.

Build pairs directly from `_load_test_base` + `attack.apply_trigger`, reusing the reproducible-subsampling helper `smoke-tests.md` already added, not `seed_everything`:

```python
def build_clean_triggered_pairs(dataset_name, attack, raw_data_dir="raw_data", max_samples=None, seed=0):
    test_base, spec = _load_test_base(dataset_name, raw_data_dir)  # loaders.py, already 0-to-1, pre-normalize
    test_base = limit_dataset(test_base, max_samples, seed)        # utils/datasets.py, isolated RNG, not seed_everything
    clean = torch.stack([test_base[i][0] for i in range(len(test_base))])
    triggered = torch.stack([attack.apply_trigger(test_base[i][0], i) for i in range(len(test_base))])
    return clean, triggered
```

**Eligibility, resolved**: unlike ASR (`AttackSuccessSet`/`is_eval_poisonable`, which restricts to samples the label mode can actually flip — e.g. `all_to_one` drops the target class), stealth measures pixel-level visibility to a human, which doesn't depend on which class an image belongs to or whether the label mode would poison it. So this pairs every sampled image with its triggered self directly against `test_base`, not through `AttackSuccessSet`'s eligibility filter — a wider, simpler, and more representative pool than the ASR-eligible one. `_load_test_base` is already `@lru_cache`d per `(dataset_name, raw_data_dir)`, so calling this repeatedly for multiple attacks on the same dataset, across however many checkpoints share that dataset, re-reads no data from disk.

- PSNR/SSIM: `data_range=1.0`, SSIM with `reduction='none'` for a real per-image std.
- LPIPS: `x * 2 - 1` to map to -1..1; images are already 3-channel here (every dataset in `DATASET_REGISTRY` is RGB), so the "repeat if grayscale" case in the original draft doesn't apply to any in-scope dataset — drop it unless a future dataset changes that.
- Batch, same device, model-free so no `use_bfloat16` concern — image-quality metrics are precision-sensitive (PSNR is directly an MSE ratio in dB), run these in float32 regardless of what dtype checkpoint evaluation uses elsewhere.

## D. Functions to add (`stealth.py`)

```python
def build_clean_triggered_pairs(dataset_name, attack, raw_data_dir="raw_data", max_samples=None, seed=0) -> tuple[Tensor, Tensor]:
    ...  # per §C

def compute_stealth_metrics(clean: Tensor, triggered: Tensor, device, lpips_backbone="alex") -> dict:
    """PSNR/SSIM/LPIPS between pair-aligned clean and triggered images, pixel space, pre-normalization."""
    return {
        "psnr_mean": ..., "psnr_std": ...,
        "ssim_mean": ..., "ssim_std": ...,
        "lpips_mean": ..., "lpips_std": ...,
        "lpips_backbone": lpips_backbone,
        "data_range": 1.0,
        "n_pairs": int,
        "max_samples": max_samples,   # new: provenance, matches checkpoint_metadata's own max_samples field
        "seed": seed,                 # new: provenance
    }

_stealth_cache: dict[tuple[str, str], dict] = {}

def cached_stealth_metrics(dataset_name, attack_name, attack, raw_data_dir, device, max_samples=None, seed=0) -> dict:
    ...  # per §A
```

## E. Wiring — `evaluate.py:evaluate_checkpoint`, both branches

`folder_name` isn't in either branch's return dict today (`evaluate_all_checkpoints` only uses it as a loop variable). Adding it is somewhat redundant once `metrics.json` lives inside the checkpoint's own folder (§F) — the folder name is already implicit in the file's path — but it's one cheap line and keeps a single `metrics.json` self-describing if it's ever read out of its directory context (aggregated across checkpoints, copied elsewhere). Decided: add it to *both* branches, derived from `checkpoint_path`, no new parameter needed. `stealth` stays attack-only.

```python
def evaluate_checkpoint(checkpoint_path, device, raw_data_dir="raw_data", batch_size=64) -> dict:
    args = read_args_json(os.path.dirname(checkpoint_path))
    model = load_checkpoint(args["architecture"], checkpoint_path, device)
    folder_name = os.path.basename(os.path.dirname(checkpoint_path))

    if args["attack"] == "benign":
        metrics = evaluate_benign(model, args["dataset"], device, raw_data_dir, batch_size)
        return {
            "folder_name": folder_name,
            "architecture": args["architecture"],
            "dataset": args["dataset"],
            **metrics,
        }

    config = default_config(args["attack"])
    attack = build_attack(args["attack"], config, DATASET_REGISTRY[args["dataset"]].image_size, args["target_label"])
    metrics = evaluate_attack(model, args["dataset"], args["attack"], config, args["target_label"], device, raw_data_dir, batch_size)
    stealth = cached_stealth_metrics(args["dataset"], args["attack"], attack, raw_data_dir, device)
    return {
        "folder_name": folder_name,
        "architecture": args["architecture"],
        "dataset": args["dataset"],
        "attack": args["attack"],
        "label_mode": args["label_mode"],
        "poison_rate": args["poison_rate"],
        "target_label": args["target_label"],
        **metrics,
        "stealth": stealth,
    }
```

This now builds an `Attack` object directly (`evaluate_checkpoint` currently doesn't — `evaluate_attack` builds its own internally and doesn't return it), so `evaluate_checkpoint` needs that one extra `build_attack` call to hand the same attack to `cached_stealth_metrics`. `evaluate_benign`/`evaluate_attack` themselves are unchanged — this is purely `evaluate_checkpoint` assembling more fields around them.

Resulting `checkpoints/vit_cifar10_badnet_a2o_0_1/metrics.json`:
```json
{
  "folder_name": "vit_cifar10_badnet_a2o_0_1",
  "architecture": "vit",
  "dataset": "cifar10",
  "attack": "badnet_a2o",
  "label_mode": "all_to_one",
  "poison_rate": 0.1,
  "target_label": 0,
  "asr": 0.98,
  "clean_accuracy": 0.91,
  "stealth": {
    "psnr_mean": 34.2, "psnr_std": 2.1,
    "ssim_mean": 0.97, "ssim_std": 0.01,
    "lpips_mean": 0.03, "lpips_std": 0.01,
    "lpips_backbone": "alex",
    "data_range": 1.0,
    "n_pairs": 2000,
    "max_samples": null,
    "seed": 0
  }
}
```

## F. Output location — `checkpoints/<folder_name>/metrics.json`, and `evaluate.py` stays atomic

Same directory as `attack_result.pt` and `args.json`, not `results/` — decided last revision, unchanged. Every `checkpoints/<folder_name>/` ends up with exactly three files: `attack_result.pt`, `args.json`, `metrics.json`.

**Bigger structural change, per your last message**: `evaluate.py` does not get a whole-directory loop at all, not even as an internal helper. `evaluate_all_checkpoints` — and `list_checkpoint_folders`/`mirror_results_folders`, which exist only to support it — are being *removed* from `evaluate.py` entirely, not repointed at a new path. Looping over "the whole thing" is orchestration; `evaluate.py`'s own docstring already scopes it to "one model, one dataset" / "one checkpoint path in, one metrics dict out," and a function that walks an entire directory doesn't belong at that altitude — that's the code smell you flagged. Confirmed by grep: `metrics.py` is the *only* caller of `evaluate_all_checkpoints`, `CHECKPOINTS_DIR`, `RESULTS_DIR`, `list_checkpoint_folders`, and `mirror_results_folders` anywhere in the tracked codebase, so nothing else needs touching.

`evaluate.py` keeps exactly: `evaluate_benign`, `evaluate_attack`, `evaluate_checkpoint` (atomic — one `checkpoint_path` in, one metrics dict out, `folder_name` on both branches, `stealth` on the attack branch only), `read_args_json`, and `save_metrics` (a plain "write this dict to this path" helper — not a loop, so it stays; it's exactly as atomic as `evaluate_checkpoint` itself, just the write-side counterpart). Its module docstring's "three layers, thinnest to widest" becomes two — `evaluate_benign`/`evaluate_attack` at the bottom, `evaluate_checkpoint` on top — with a line stating there's no third, directory-walking layer by design, not an oversight.

`metrics.py` becomes the one place that walks `checkpoints/` — a plain `for` loop written directly in `main()`, not a function reimported from `evaluate.py`:

```python
"""CLI: baseline attack-success/clean-accuracy/stealth metrics for every checkpoint.

evaluate.py stays atomic (one checkpoint path in, one metrics dict out) on
purpose. Looping over the whole checkpoints/ directory is this file's job,
not evaluate.py's.
"""

import argparse
import os

import torch

from evaluate import evaluate_checkpoint, save_metrics


def list_checkpoint_folders(checkpoints_dir: str) -> list[str]:
    if not os.path.isdir(checkpoints_dir):
        return []
    return sorted(
        name
        for name in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, name))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline attack/benign/stealth metrics for every checkpoint"
    )
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument(
        "--folder", nargs="+", default=None, help="only process these folder names"
    )
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_names = list_checkpoint_folders(args.checkpoints_dir)
    if args.folder is not None:
        folder_names = [name for name in folder_names if name in args.folder]

    for folder_name in folder_names:
        checkpoint_path = os.path.join(args.checkpoints_dir, folder_name, "attack_result.pt")
        try:
            metrics = evaluate_checkpoint(
                checkpoint_path, device, args.raw_data_dir, args.batch_size
            )
            save_metrics(
                os.path.join(args.checkpoints_dir, folder_name, "metrics.json"), metrics
            )
            print(f"{folder_name}: {metrics}")
        except Exception as error:
            print(f"FAILED {folder_name}: {error}")


if __name__ == "__main__":
    main()
```

`list_checkpoint_folders` moves here verbatim (its only consumer). `CHECKPOINTS_DIR`/`RESULTS_DIR` don't move anywhere as constants — nothing needs a shared, importable constant for a single literal string used in exactly one `argparse` default, so `"checkpoints"` is just written directly. `--results-dir` is gone from the CLI (nothing reads it, per the last revision); `--checkpoints-dir` is both the read and write root now.

From the outside, `python metrics.py --folder <name> <name>` still works the same way it did before this change — same flags minus `--results-dir`, same per-folder `FAILED {folder_name}: {error}` resilience, same progress printing. What moved is purely internal: the loop's home, not its behavior.

## G. Sharp edges (unchanged from the original, still correct)

- Pixel space, pre-normalization — now backed by a concrete function reference (`loaders._load_test_base`), not just a rule to remember.
- LPIPS is opposite direction (lower = more similar); PSNR/SSIM higher = better.
- Per-image reduction for std (`reduction='none'`), not std-of-batch-means.
- Clean and triggered must be the same images, index-aligned — guaranteed here by both being built from the same `test_base[i]` in the same loop.
- Subsampling seeding now goes through `limit_dataset`'s isolated `np.random.default_rng(seed)` (`utils/datasets.py`), **not** `seed_everything` — consistent with why `docs/plans/smoke-tests.md` moved off `seed_everything` for this exact kind of reproducible-subset need: an isolated generator never perturbs the global RNG state model init/training rely on, and is reproducible on every call independent of when in a run it's called.

## H. Verification

- Validate PSNR and SSIM against `skimage.metrics` on 5 sample pairs.
- Sanity check: identical images give PSNR very high/inf, SSIM near 1.0, LPIPS near 0. A visible trigger (badnet) should score lower SSIM / higher LPIPS than a subtler one (blend or wanet). Wrong direction means something's broken.
- Identity-trigger test: a locally-constructed `Attack` whose `apply_trigger` returns its input unchanged (`lambda image, index: image`, wrapped in `Attack("identity", ..., "all_to_one", 0)` — no need for a real registered attack) should produce PSNR at/near infinity (or the metric's max), SSIM ~1.0, LPIPS ~0.
- Confirm the cache actually caches: call `cached_stealth_metrics` twice with the same `(dataset, attack)` key and confirm the second call doesn't rerun LPIPS (e.g. patch/spy on `compute_stealth_metrics` in the test, or just time it).
- One real `python metrics.py --folder <one attack checkpoints/ folder> <one benign checkpoints/ folder>` end to end, confirm `checkpoints/<attack-folder>/metrics.json` has `folder_name` and the `stealth` block with sane numbers, `checkpoints/<benign-folder>/metrics.json` has `folder_name` but no `stealth` key, and `results/` is untouched by the run (same folder listing/contents before and after).
- `grep -n "evaluate_all_checkpoints\|CHECKPOINTS_DIR\|RESULTS_DIR\|mirror_results_folders" evaluate.py` returns nothing — confirms the whole-directory loop is fully gone from `evaluate.py`, not just unused.

## I. Build order (one commit each)

1. `stealth.py`: `build_clean_triggered_pairs`, `compute_stealth_metrics`, `cached_stealth_metrics`. Commit.
2. Remove `evaluate_all_checkpoints`/`list_checkpoint_folders`/`mirror_results_folders`/`CHECKPOINTS_DIR`/`RESULTS_DIR` from `evaluate.py`; rewrite `metrics.py:main()` as the plain `for` loop that replaces them, writing into `checkpoints_dir` directly; drop `--results-dir` from its CLI. Commit.
3. Wire `folder_name` (both branches) and `stealth` (attack branch only) into `evaluate.py:evaluate_checkpoint`. Commit.
4. Skimage-agreement, identity-trigger, and cache-hit tests. Commit.
