# Atomic dropout-position ablation for PSBD (ViT first)

## Context

CLAUDE.md's current core contribution is a binary choice: `pre_residual` (toggle the transformer's own existing `nn.Dropout` modules) separates clean from backdoor PSU on ViT-B/16, `post_residual` (wrap each block, add fresh Dropout after the residual add) mostly doesn't. That result was never decomposed further — "pre_residual" bundles two positions (before the attention-branch residual add, before the MLP-branch residual add) into one setting, and nothing between "before" and "after" was ever tested. This plan builds the infrastructure to test every individual position, on ViT first (Swin's registry is defined alongside it, per §B, but not run in this pass), to find out where in the block the known pre/post-residual gap actually originates.

**Decided across this conversation, not re-litigating:**
- **Always insert fresh, independent modules; never toggle a model's existing dropout.** Reusing a trained dropout as PSBD's noise source conflates two different things: the model's own training-time regularization (whose inverted-dropout scaling the *next* layer's weights were calibrated against) and PSBD's own injected perturbation. Every existing dropout (ViT's block-level `dropout`, both architectures' MLP-internal dropouts, Swin's `stochastic_depth`) is left exactly at its natural inference state (eval mode, i.e. identity) and never touched. This reverses `defences/dropout.py`'s current mechanism, not just extends it — `configure_pre_residual_dropout`'s "find existing `nn.Dropout`, set `.p` and `.train()`" approach is replaced entirely, not kept alongside the new one.
- **Forward hooks, not `forward()`-reimplementation.** Today's `PostResidualEncoderBlock`/`PostResidualSwinBlock` each hand-copy a block's internal control flow into a wrapper class — fragile (breaks the moment torchvision changes a block's internals) and doesn't scale to 8 positions × 2 architectures without 16 near-duplicate classes. `register_forward_pre_hook`/`register_forward_hook` inject at a named submodule boundary without touching or knowing the surrounding `forward()` logic at all, and return a `RemovableHandle` with a native `.remove()` — "plug" and "unplug" become trivial, no manual reset/restore bookkeeping needed, because we never mutated the original model in the first place.
- **Two-stage compute/analyze split**, already proven in `_archive/sweep.py` + `_archive/experiment_io.py` (archived, not resurrected in place — this plan is a clean rewrite with better names and modularity, per your request): Stage 1 (GPU, expensive) runs the forward passes and saves raw per-sample data to disk. Stage 2 (CPU, cheap) does threshold selection and TPR/FPR/AUROC from the saved data, fast enough to iterate on in a notebook without touching the GPU again.

## A. Scope for this pass

- **Architecture**: ViT only, both optimizers — Swin's position registry is defined in the same pass (§B) since the registry pattern is identical either way, but Swin is not run yet.
- **Dataset**: cifar100.
- **Attack**: WaNet, at all three poison rates already trained, Adam and SAM (rho=0.1) both — confirmed on disk, 6 checkpoints:
  - Adam: `vit_cifar100_wanet_0_01`, `vit_cifar100_wanet_0_05`, `vit_cifar100_wanet_0_1`
  - SAM (rho=0.1): `vit_cifar100_wanet_0_01_sam_rho_0_1`, `vit_cifar100_wanet_0_05_sam_rho_0_1`, `vit_cifar100_wanet_0_1_sam_rho_0_1`
- **Position sweep**: every single atomic position in isolation (§B's registry, 8 positions) is the primary sweep — this is what "atomically" means and what directly answers where the signal lives. `pre_residual` and `post_residual` as two-position combos are run too, as a direct comparison against the existing (toggle-based) result already on record for those names. 10 position-configs total.
- **Rate sweep**: keep the full 9 rates (0.1-0.9), same as `RunConfig.dropout_rates` today.

Grid for this pass: 6 checkpoints × 10 position-configs × 9 rates = 540 (checkpoint, position-config, rate) runs, each doing `forward_passes=3` stochastic passes over 3 splits (validation, clean, backdoor). Not run as one job — see §H for how this is flattened across many small PBS jobs instead.

## A2. Data split, threshold heldout set, and reproducibility (0 leakage)

A single, standardized split of each dataset's clean test set, shared by every checkpoint of that dataset — the split is a function of `(dataset test set, seed)` only, never of the attack or checkpoint, so all 6 cifar100 checkpoints see the identical split. This replaces the archived `split_validation_and_eval` (sklearn-stratified) + `balance_by_class` (per-class cap) entirely: the user wants a plain shuffle, the full analysis pool, and an explicit auditable index record.

**The split:**
- Load the full clean test set in its native, deterministic order (torchvision cifar100 test = 10000 fixed-order images), 0-to-1 range, pre-normalize (the `loaders._load_test_base` base).
- `seed_everything(PSBD_SPLIT_SEED)` then `perm = torch.randperm(n_total)` as the very first RNG consumption. `PSBD_SPLIT_SEED = 0`, one named constant, the standardized seed.
- `heldout_indices = perm[:2000]` — the clean threshold-finding set (the "validation" split). Clean only, never triggered.
- `analysis_indices = perm[2000:]` — everything else (8000 for cifar100), the PSBD analysis pool. Both the clean-analysis and backdoor-analysis splits come from these same images, so they stay paired.
- No stratification, no per-class balancing.

**Three splits, run through the full pipeline (baseline + every rate), same three-way structure as before, only the carving changes:**
- `validation`: `Subset(base, heldout_indices)`, clean, no trigger → the quantile threshold source.
- `clean`: `Subset(base, analysis_indices)`, clean → FPR and the clean PSU distribution.
- `backdoor`: `AttackSuccessSet(Subset(base, analysis_indices), ...)`, triggered, eligibility-filtered via `is_eval_poisonable` → TPR and the backdoor PSU distribution.

**0 leakage, by construction**: heldout and analysis are disjoint slices of one permutation, so the threshold is never fit on any sample it is later scored against. The threshold set is clean-only; the backdoor split is drawn only from the analysis pool.

**100% reproducible, seed-recipe plus explicit manifest (belt and suspenders):**
- Recoverable from the seed alone via the exact recipe (`seed_everything(seed); torch.randperm(n_total)`), so a Jupyter notebook that loads the same full test set and reruns those two lines recovers exactly which original test index maps to which saved tensor row. Wrap the recipe in one helper both the sweep and the notebook call, so there is one definition, not two that could drift.
- ALSO written explicitly to `results/<checkpoint_folder>/psbd/split_manifest.json`: `{seed, dataset, n_total, n_heldout, heldout_indices, analysis_clean_indices, analysis_backdoor_indices, recipe_note}` — the ground-truth record, so reproduction never has to rely on RNG-algorithm equivalence across environments. Because the split is deterministic and checkpoint-independent, all 60 jobs write byte-identical manifests (same idempotency/race argument as the baseline in §C, no correctness hazard).
- **Tensor row ordering is defined by the manifest, and every loader is `shuffle=False`** so row order equals subset order equals the manifest arrays, no hidden reordering between a tensor and its index record: row `i` of any `*_validation.pt` ↔ `heldout_indices[i]`; row `i` of any `*_clean.pt` ↔ `analysis_clean_indices[i]`; row `j` of any `*_backdoor.pt` ↔ `analysis_backdoor_indices[j]`, where `analysis_backdoor_indices = analysis_indices[eligible_positions]` (the `is_eval_poisonable` subset, in order).

## B. Position registry — `defences/dropout.py`, rewritten

Delete entirely: `configure_pre_residual_dropout`, `configure_post_residual_dropout`, `PostResidualEncoderBlock`, `PostResidualSwinBlock`, `_wrap_as_post_residual`, `_unwrap_post_residual`, `_replace_blocks`, `reset_dropout`. All superseded — there is nothing to "reset" once nothing is ever mutated in place.

```python
@dataclass(frozen=True)
class PositionSpec:
    submodule_name: str   # child module name relative to a block, "" for the block itself
    hook_type: str        # "pre" or "post"
```

**ViT-B/16** (`EncoderBlock`, confirmed from `torchvision/models/vision_transformer.py`, one per each of 12 blocks):

| position | target | hook | note |
|---|---|---|---|
| `after_embedding` | `Encoder.dropout` (once, model-level, not per-block) | pre | existing dropout stays untouched, ours runs on its input |
| `before_attention_norm` | `ln_1` | pre | |
| `before_attention` | `self_attention` | pre | `self_attention(x, x, x, ...)` — q/k/v all resolve to the same perturbed tensor |
| `before_attention_residual` | `dropout` | post | ViT's own existing dropout sits exactly here; ours runs on its output |
| `before_mlp_norm` | `ln_2` | pre | identical hook point to `after_attention_residual` — both names alias the same target, keep both names, they mean different things conceptually even though nothing sits between them |
| `before_mlp` | `mlp` | pre | |
| `before_mlp_residual` | `mlp` | post | |
| `after_mlp_residual` | the `EncoderBlock` itself | post | its return value *is* this position |

**Swin-S** (`SwinTransformerBlock` V1 — confirmed `models.build_swin` calls `swin_s`, the pre-norm V1 block, not `swin_v2_s`; depths `[2,2,18,2]` = 24 blocks total; confirmed from `torchvision/models/swin_transformer.py`):

| position | target | hook | note |
|---|---|---|---|
| `after_embedding` | `features[0]` (the whole patch-embed `Sequential(Conv2d, Permute, LayerNorm)`, once, model-level) | post | no existing dropout here at all in torchvision's swin_s, unlike ViT |
| `before_attention_norm` | `norm1` | pre | |
| `before_attention` | `attn` | pre | |
| `before_attention_residual` | `attn` | post | **not** after `stochastic_depth` — see the shared-instance note below |
| `before_mlp_norm` | `norm2` | pre | aliases `after_attention_residual`, same as ViT |
| `before_mlp` | `mlp` | pre | |
| `before_mlp_residual` | `mlp` | post | **not** after `stochastic_depth`, same reason |
| `after_mlp_residual` | the `SwinTransformerBlock` itself | post | |

**Why `before_*_residual` hooks `attn`/`mlp` directly instead of `stochastic_depth`, unlike the original plan of putting it right before the residual add**: `self.stochastic_depth` is one `StochasticDepth` instance, constructed once per block and called twice (`x + stochastic_depth(attn(...))`, then `x + stochastic_depth(mlp(...))`). A hook on it fires identically for both calls — there is no way to tell from `(module, input, output)` which branch invoked it. Hooking `attn`/`mlp` directly is unambiguous (each called exactly once per block) at the cost of a small asymmetry with ViT: on Swin, our inserted dropout runs *before* `stochastic_depth` sees the branch output, not after it. `stochastic_depth` itself is never touched — same "leave existing regularizers alone" rule as everywhere else, it just ends up downstream of our hook instead of upstream.

**Attention-weight dropout (inside `nn.MultiheadAttention` for ViT, inside the functional `shifted_window_attention` for Swin) is out of scope** — both are internal to their attention implementation (a private forward pass inside `nn.MultiheadAttention`, or a functional call with no module boundary at all for Swin), reaching either would need monkey-patching internals rather than a hook on a module boundary, and is a different, deeper kind of experiment than the block-level ablation this plan is scoped to.

**Both registries are defined now; only ViT's is run in this pass (§A).**

**Plug / unplug**:

```python
def plug_dropout(
    model: nn.Module, architecture: str, position_names: tuple[str, ...],
    dropout_factory, rate: float,
) -> list[RemovableHandle]:
    """Attach a fresh dropout_factory(rate) module at every named position, in every block."""

def unplug_dropout(handles: list[RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()
```

`dropout_factory` is the "pair of position_name and dropout module" you asked for, generalized: a `dict[str, Callable[[float], nn.Module]]` mapping position name to a constructor, defaulting every position to `nn.Dropout` but overridable per position without touching the plug/unplug mechanics at all.

**Named combos**, replacing the old `dropout_placement` string enum with an open registry:

```python
DROPOUT_CONFIGS: dict[str, tuple[str, ...]] = {
    "pre_residual": ("before_attention_residual", "before_mlp_residual"),
    "post_residual": ("after_attention_residual", "after_mlp_residual"),
    # every single position also usable directly as a one-element combo,
    # e.g. position_names=("before_attention",) needs no registry entry
}
```

## C. Baseline cache and raw per-pass data — `defences/inference.py` extended

`build_baseline_cache`/`compute_psu_and_shift` already exist and are correct; two changes:

1. **Persist the baseline to disk, and make computing it idempotent across separate PBS jobs, not just within one process.** It depends only on `(checkpoint, split)`, never on position or rate, so in principle it's computed once per checkpoint and reused across all 90 `(position-config, rate)` runs for that checkpoint (10 position-configs × 9 rates). But §H splits each checkpoint's 10 position-configs into 10 *separate* PBS jobs precisely so they can run concurrently on different GPUs — which means there is no longer one long-lived process to hold the baseline in memory across all 10. Before computing, check whether `results/<checkpoint>/psbd/baseline_<split>.pt` already exists; if so, load it instead of recomputing. Whichever of a checkpoint's 10 jobs happens to start first computes and writes it once, the other 9 just load the file. Safe under a rare near-simultaneous race (deterministic no-dropout forward pass — two jobs computing it at once just both write the same values, not a correctness issue, only a small duplicated cost in that edge case).
2. **Save raw per-pass data, not just the final reduced PSU score.** `compute_psu_and_shift` reduces `forward_passes` samples down to one score per sample internally. Saving only that throws away the ability to recompute PSU under a different aggregation (median instead of mean, a different `k`) without rerunning the GPU pass. Save the per-pass probability of the *baseline-argmax class* instead — shape `(forward_passes, num_samples)`, one float per pass per sample (not the full per-class softmax vector, which would be `num_classes`× bigger for no benefit PSU ever uses). Still small: for cifar100's 2000-sample clean validation set, `3 × 2000 × 4 bytes ≈ 24KB` per `(checkpoint, position, rate, split)` — this was never a real disk-space concern, unlike the earlier general PSBD-cache-location conversation.

## D. Save format — new module, `defences/psbd_cache.py`

```python
def save_baseline(path, probs: Tensor, labels: Tensor) -> None: ...
def load_baseline(path) -> tuple[Tensor, Tensor]: ...
def save_dropout_pass_probs(path, per_pass_probs: Tensor) -> None: ...   # (forward_passes, N)
def load_dropout_pass_probs(path) -> Tensor: ...
```

Plain `torch.save`/`torch.load` of a small dict, `.pt` as you specified (not `experiment_io.py`'s `.npz` — no reason to keep numpy in the loop when everything else here is already a `torch.Tensor`).

**Layout**, under the `results/`-is-for-analysis-config-dependent-artifacts convention from earlier this session:

```
results/<checkpoint_folder>/psbd/
    split_manifest.json         # §A2: seed + the three original-index arrays, ground truth for every tensor's row order
    baseline_validation.pt
    baseline_clean.pt
    baseline_backdoor.pt
    <position_config_name>/
        rate_0_1_validation.pt
        rate_0_1_clean.pt
        rate_0_1_backdoor.pt
        rate_0_2_validation.pt
        ...
```
`baseline_*.pt` and `split_manifest.json` sit one level above the position-config subfolders, written once and read by every one of them — not duplicated per position-config. The manifest is the single index record for the whole `psbd/` subtree (every rate/position tensor shares the one split), so there is one authoritative mapping, not a near-identical sidecar next to each of the hundreds of `.pt` files. `save_dropout_pass_probs`/`save_baseline` write only the tensors; the manifest is written once by the split step (§A2, §E).

## E. Orchestration entrypoint — new root-level script, `psbd_dropout_sweep.py`

Replaces `sweep.py`/`run_sweep.py`. Takes **one checkpoint and one position-config** as CLI arguments and loops over rates internally — the outer loop over checkpoints × position-configs is not inside this script at all, it's §H's PBS generation layer, one job per combination. This mirrors the `metrics.py` lesson from earlier in this project: keep the atomic unit of work small and let something else do the flattening, rather than one script owning both dimensions.

Reuses, rather than reimplements, what already exists and is correct:
- `defences.detection.attack_success_rate`/`clean_accuracy` for the per-checkpoint behavior numbers.
- The attack rebuild from `args.json` metadata that `checkpoint_eval.build_eval_loaders_from_checkpoint` already does (read metadata, `build_attack(default_config(...))`) — but NOT its stratified/balanced split. A new `build_psbd_loaders_from_checkpoint` (in `defences/checkpoint_eval.py`, alongside the existing one, which stays for any other caller) does the §A2 split instead and returns the manifest so it can be written once. Confirm the old function's callers before touching it; add, don't replace.

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-folder", required=True)
    parser.add_argument("--position-config", required=True, choices=tuple(DROPOUT_CONFIGS) + SINGLE_POSITION_NAMES)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_checkpoint(...)
    # §A2 split: seed_everything(PSBD_SPLIT_SEED); torch.randperm; 2000 heldout / rest analysis.
    loaders, manifest = build_psbd_loaders_from_checkpoint(checkpoint_path, PSBD_SPLIT_SEED)
    # loaders is {"validation": ..., "clean": ..., "backdoor": ...}, all shuffle=False.

    psbd_dir = os.path.join(args.results_dir, args.checkpoint_folder, "psbd")
    write_split_manifest(psbd_dir, manifest)   # idempotent, deterministic, checkpoint-independent (§A2)
    baselines = {split: load_or_build_baseline(psbd_dir, split, model, loaders[split]) for split in loaders}

    position_names = DROPOUT_CONFIGS.get(args.position_config, (args.position_config,))
    for rate in DROPOUT_RATES:                                          # 9 rates
        handles = plug_dropout(model, "vit", position_names, dropout_factory, rate)
        for split, loader in loaders.items():
            per_pass_probs = compute_dropout_pass_probs(model, loader, baselines[split], ...)
            save_dropout_pass_probs(os.path.join(psbd_dir, args.position_config), rate, split, per_pass_probs)
        unplug_dropout(handles)
```

No `reset_dropout` call needed anywhere in this loop — `unplug_dropout` already returns the model to exactly its loaded state, every time, by construction. `load_or_build_baseline` is the idempotent check from §C. `write_split_manifest` is likewise idempotent (§A2): all 60 jobs write the same bytes.

## F. Threshold / TPR / FPR / AUROC — stays a separate, cheap step

Not part of the orchestration script at all. `defences.detection`'s `threshold_from_validation`, `detection_rates`, `auroc` are unchanged and already operate on score tensors, not on a model or a GPU. A short loader (or your existing jupyter workflow) reads `load_dropout_pass_probs` + `load_baseline`, reduces to a PSU score under whatever aggregation you want, and computes threshold/TPR/FPR/AUROC at any quantile — instantly, no GPU, freely repeatable.

## G. Correctness rules this must not regress

- **Two distinct uses of `seed_everything` here, both deliberate, do not merge them.** (1) The §A2 data split reseeds then draws `torch.randperm` to fix which samples land in heldout vs analysis — this genuinely wants global-RNG reproducibility so the notebook recipe matches, and the result is frozen into explicit indices anyway. (2) `compute_psu_and_shift`'s `seed_everything(seed)` reseeds the *dropout mask sampling* (`nn.Dropout`'s mask comes from torch's global RNG) so masks are reproducible pass-to-pass and identical across the validation/clean/backdoor calls that must stay paired. Both are different from the smoke-tests `max_samples` isolated-RNG rule (that was about not perturbing global state during subsetting); here global-RNG reproducibility is exactly what is wanted, so keep `seed_everything`. The split reseed happens once at loader-build time and is captured as fixed indices before any mask sampling, so the two never interfere even at the same seed value.
- ASR/eligibility (`is_eval_poisonable`, `AttackSuccessSet`) is untouched — the new `build_psbd_loaders_from_checkpoint` applies the same eligibility filter to its analysis subset that the existing path applies to the full set.
- Uniform 15 epochs / SAM rho=0.1 are training-time rules, not touched by this (all 6 checkpoints are already-trained: 3 Adam, 3 SAM rho=0.1).

## H. PBS job flattening — one job per (checkpoint, position-config)

540 (checkpoint, position-config, rate) runs as one PBS job would mean a multi-day walltime request, a long queue wait for that large a reservation, and — critically, learned from the earlier eval sweep — only ever using one GPU at a time regardless of how the cluster's queue could otherwise parallelize it. Flattening by `(checkpoint, position-config)`, with rates swept internally per job (§E), gives:

- **6 checkpoints × 10 position-configs = 60 PBS jobs**, each requesting exactly 1 GPU and sweeping its 9 rates internally (reusing one loaded model, one baseline cache per job).
- Small per-job walltime → faster to schedule, easier for the scheduler to backfill.
- 60 independent single-GPU jobs can land on 60 different physical GPUs across the cluster simultaneously (queue and node availability permitting) — actual parallelism, unlike the shared login-node GPU from the eval-sweep discussion, where more concurrent processes just meant more contention on the one A100 available there. This is the real payoff of going through `qsub` here rather than another login-node background chain.

**Generator script, not 60 hand-written files**: `scratch/generate_psbd_pbs_jobs.py`, one template parameterized by `(checkpoint_folder, position_config_name)`, iterated over the 6×10 grid. Output layout:

```
pbs/psbd_sweep/<checkpoint_folder>/<position_config>.pbs
logs/psbd_sweep/<checkpoint_folder>/<position_config>.log
```

Each generated file's body is just:
```bash
python psbd_dropout_sweep.py \
    --checkpoint-folder <checkpoint_folder> \
    --position-config <position_config>
```
plus the standard header (proxy exports, `cd`+activate venv, echo Job ID/Node/Started, `nvidia-smi`, `echo Finished`) matching every other `.pbs` file in this repo. Walltime sized the same way as the eval sweep: one real timed run of a single `(checkpoint, position-config)` job (9 rates × 3 splits × k=3 passes) before generating all 60, not guessed.

**Submission gate**: for this pass the user has authorized submission, but only after a passing smoke test (§I) that exercises every position without crashing. Generate all 60, run the smoke test, and `qsub` only if it fully passes; if any position crashes, do not submit — report instead. After submitting, `qstat` and report the job IDs.

## I. Verification

- Unit test: `plug_dropout` on a fresh `build_vit(10)`, confirm exactly one new module is registered per targeted block per position (12 blocks × 1 position for a per-block position; the model-level `after_embedding` adds 1), confirm `unplug_dropout` leaves `model.state_dict()` byte-identical to before plugging (no weights touched, only hooks added and removed). Repeat the plug/unplug structural check on `build_swin(10)` too, since Swin's registry is defined this pass even though not swept.
- **Smoke test that every position runs without crashing (this is the submission gate, §H)**: on one real checkpoint (`vit_cifar100_wanet_0_1`) with a tiny sample budget, loop over all 8 single positions plus both combos, plug → one forward per split → unplug, and confirm none raise. This is what proves the 60 jobs won't crash on an unforeseen hook target before any `qsub`.
- **Reproducibility and 0-leakage checks (§A2)**:
  - The seed recipe reproduces the manifest exactly: independently rerun `seed_everything(PSBD_SPLIT_SEED); torch.randperm(n_total)` and assert the resulting heldout/analysis index arrays equal `split_manifest.json`'s.
  - `set(heldout_indices) & set(analysis_indices) == empty`, and their union is the full `range(n_total)` — disjoint and complete.
  - Row-mapping round-trip: for a couple of rows, confirm the image the loader served at row `i` is the same original test image at the manifest index for row `i` (load the raw test set, compare tensors), so the tensor-to-original-index mapping is provably correct, not assumed.
- Real run: one checkpoint (`vit_cifar100_wanet_0_1`), one position (`before_attention_residual`), one rate (0.5), confirm `results/vit_cifar100_wanet_0_1/psbd/before_attention_residual/rate_0_5_*.pt` exist and load back as a `(3, N)` tensor with values in `[0, 1]`, and that `split_manifest.json` was written.
- Confirm the baseline and manifest are written once and reused: run two different positions back to back for the same checkpoint, confirm `baseline_*.pt` and `split_manifest.json` mtimes don't change on the second position's run.
- Sanity check the known result still reproduces: `pre_residual` combo on `vit_cifar100_wanet_0_1` should show a clean/backdoor PSU gap in the same direction previously observed with the old toggle-based mechanism, confirming the new hook-based insertion measures the same phenomenon, not something subtly different.

## J. Build order (one commit each)

1. `defences/dropout.py` rewrite: `PositionSpec`, `VIT_POSITIONS`, `SWIN_POSITIONS`, `plug_dropout`, `unplug_dropout`, `DROPOUT_CONFIGS`. Delete the toggle-based functions. Commit.
2. `defences/checkpoint_eval.py`: add `build_psbd_loaders_from_checkpoint` (§A2 split) returning loaders + manifest; leave the existing balanced function in place. Commit.
3. `defences/inference.py`: add `compute_dropout_pass_probs` (raw per-pass, not reduced). Commit.
4. `defences/psbd_cache.py`: save/load for baseline and per-pass tensors, plus `write_split_manifest`/`read_split_manifest`. Commit.
5. `psbd_dropout_sweep.py`: the orchestration entrypoint, one `(checkpoint, position-config)` per invocation, scoped to the 6 WaNet/ViT/cifar100 checkpoints. Commit.
6. Tests per §I, including the reproducibility/leakage checks and the all-positions smoke test. Commit.
7. One real timed smoke run (one `(checkpoint, position-config)` job, all 9 rates) to size PBS walltime the same way the eval sweep was sized.
8. `scratch/generate_psbd_pbs_jobs.py` emits the 60 `pbs/psbd_sweep/<checkpoint>/<position-config>.pbs` files. Run the §I all-positions smoke test as the submission gate, then `qsub` the 60 and `qstat` to confirm they queued.
