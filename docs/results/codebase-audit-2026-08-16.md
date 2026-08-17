# Codebase Audit: 2026-08-16

Full codebase review covering training pipeline, PSBD detection mechanism,
perturbation operators, adaptive evasion, evaluation, and reporting.

## Bottom line

The core PSBD detection pipeline is correct: PSU computation, one-sided AUROC,
sigma matching, eval-set construction, and clean/backdoor pairing all work as
designed. The main detection tables (AUROC at `before_attention_norm`) are valid.

Three categories of issues surfaced:

1. **Adaptive attacker jobs (120 in flight) are misconfigured.** The ViT arm
   penalty is structurally zero. These jobs must be stopped and reconfigured
   before their results are used.
2. **Clean-label rate axis is fabricated.** `lc`/`sig` at 5% and 10% are
   silently capped to 1% on CIFAR-100, Tiny, and GTSRB because the eligible
   pool is the target class only. The three "rates" train identical models.
3. **Secondary analysis scripts have oracle leaks.** `psbd_variants.py` and
   `psbd_operating_points.py` use labeled data for threshold direction.
   These affect specific hypothesis docs, not the main pipeline.

## Fixes applied in this audit

1. **TokenMask CLS scaling** (`defences/perturbations.py:119-126`). The CLS
   token was forced to `keep=1` then multiplied by `1/(1-rate)`, giving it a
   deterministic gain instead of identity. Fixed by applying the scale factor
   before setting `keep[:, 0, :] = 1.0`. At `before_attention_norm` (our main
   position) LayerNorm absorbs this gain, so existing main-table numbers are
   unaffected. Results at non-LN positions (`before_mlp_residual`,
   `mlp_neurons`, `after_attention_residual`, etc.) carry the contamination
   and should be re-swept after this fix.

2. **inference.py finally block** (`defences/inference.py:139-161`).
   `restore_model_dropout` was not in a `finally` block. If any batch raised,
   model dropout stayed live at the elevated rate for all subsequent splits,
   silently corrupting every cache written after the failure. Wrapped in
   `try/finally`.

3. **defence_tables.py TPR column label** (`defence_tables.py:210-211`).
   TPR@5%FPR is computed from the labeled ROC curve (oracle threshold, not the
   deployable clean-validation quantile). Column header now says
   "TPR@5%FPR (oracle)" to distinguish from the deployable threshold.

4. **Sigma-matching rule alignment** (`defence_tables.py:82-96`). `cell_score`
   used a different sigma-matching rule (first ascending rate >= target) than
   `select_rate_at_matched_shift` (closest rate to target). Replaced with a
   call to `select_rate_at_matched_shift` so both scripts agree.

5. **Two-sided docstring** (`defences/psbd_metrics.py:223-233`). Docstring still
   argued for two-sided reporting as a valid diagnostic. Updated to state that
   H15 retired it and no live consumer reads it.

6. **next-steps.md job IDs**. Updated crashed job references to resubmitted IDs
   (1022128 for direction_norm, 1022129 for Swin token_mask).

## Issues requiring action

### BLOCKER: Adaptive attacker ViT penalty is zero

`token_mask` at p=0.5 applied at all 12 blocks destroys the prediction entirely.
PSU degenerates to pure confidence, and `ReLU(mean_clean - mean_poisoned)` is
structurally 0 because poisoned samples are always the most confident. Confirmed
from logs: `psu_poisoned=0.99`, `penalty=0.0000` from epoch 2 onward.

All 60 ViT evasion jobs are training ordinary backdoor models at 3-4x cost.

Fix: use the sigma-calibrated probe rate (the rate where clean validation sigma
>= 0.6) instead of the hardcoded 0.5. The rate grid already has this data in
the PSBD cache.

### BLOCKER: Evasion uses absolute PSU, defence reports fractional

`adaptive_evasion.py:98` penalizes absolute PSU (`base - dropped`). The
reporting path defaults to fractional PSU (`defence_tables.py --score fractional`),
which is the preferred score (92.8% of cells). An attack optimizing the wrong
statistic is not a fair test.

Fix: match the penalty's score form to whatever the defence reports.

### BLOCKER: Lambda=0 controls are corrupted (CIFAR-10 pilot)

The earlier control checkpoints were written as bare files, not inside folders,
because the `--output` path had no `/attack_result.pt`. The sidecars overwrote
each other at `checkpoints/args.json`. These controls cannot be read.

### HIGH: Clean-label rate axis is fabricated

`poison.choose_poison_indices` silently caps the count at the eligible pool.
For `lc`/`sig`, the eligible pool is the target class: 500 images on CIFAR-100,
500 on Tiny. So 1%, 5%, and 10% all realize the same 500-sample poisoning (1%
on CIFAR-100, 0.5% on Tiny). The three checkpoints trained with these "rates"
differ only by random seed noise, but are reported as distinct rate points.

This is live in `defence_tables.py` (PANEL includes `lc` against a 3-rate axis)
and in 16 of the 120 evasion jobs.

Fix: record `realized_poison_rate` and `n_poisoned` in checkpoint metadata.
Acknowledge the cap in any rate-axis claim for clean-label attacks.

### HIGH: Swin evasion PSU computed with stochastic depth live

`adaptive_evasion.py` computes PSU under `model.train()`, which leaves Swin's
stochastic depth active. The defence computes PSU under `model.eval()`. The
attacker optimizes a noisy, differently-distributed proxy. A null result on
Swin cannot be read as "PSBD survives an adaptive attacker."

Fix: temporarily switch to `model.eval()` for the PSU computation inside the
training loop, leaving only the injected probe modules in train mode.

### HIGH: No lambda=0 controls for the 120-job matrix

The pre-registration requires lambda=0 controls. None exist for CIFAR-100 or
Tiny. Comparing to pre-existing baselines confounds evasion with batch size
(128 vs 48).

### HIGH: Probe rate not sigma-calibrated

The evasion jobs use `--evade-rate 0.5` for both architectures and both datasets.
At that rate, the measured perturbation strength spans a 12x range across
configurations. Any architecture or operator comparison is uninterpretable.

### HIGH: Clean-label evasion penalty targets wrong population

For `sig`/`lc`, `FlaggedPoisonedSet` flags training poison indices (target-class
images), but the PSBD backdoor split scores non-target images with the trigger.
The attacker raises PSU on a group the defender never scores (24 cells).

### HIGH: STRIP blended in normalized space

`defences/baselines.py:111` adds normalized tensors directly instead of
de-normalizing, blending, clipping, and re-normalizing. The blended input
carries an extra per-channel offset. Since the paper claims PSBD outperforms
STRIP, a handicapped STRIP baseline weakens the comparison.

### HIGH: GaussianNoise magnitude is data-dependent

`defences/perturbations.py:170-174` computes noise scale from `x.detach().std()`
per batch. Clean and backdoor splits have different activation statistics, so
the perturbation strength applied to each is different. The measured separation
partially reflects how hard each set was hit, not just backdoor sensitivity.
This is the control operator for H23 and H28.

### HIGH: Placement selection is in-sample

The "27 operator/position combinations sorted by 1% AUROC" ranking (H28) picks
its winner by reading the same backdoor split it reports on. This is a
best-of-27 selection effect, not a held-out result.

### MEDIUM: PERTURBATION_POSITIONS is dead code

The operator/position compatibility constraint in `perturbations.py:395-401` is
defined but never enforced. `psbd_dropout_sweep.py` accepts any operator at any
position, and mismatches (token_mask at input_pixels, gain_scale at non-LN
positions) produce plausible but meaningless results.

### MEDIUM: activate_model_dropout differs between ViT and Swin

On ViT it covers 36 modules (block-output + MLP dropouts). On Swin it covers
only 48 MLP-internal dropouts (attention dropout is `F.dropout`, stochastic
depth is `StochasticDepth`, neither reachable by this function). The E2a
stacking experiment is therefore not architecture-comparable.

### MEDIUM: Attack configs not serialized

`checkpoint_metadata` records the attack name but none of its config (blend
alpha, SIG amplitude, WaNet grid params, etc.). Changing a config default
silently re-evaluates all checkpoints against a different trigger.

### MEDIUM: TaCT ASR measured over wrong population

ASR includes all non-target test images, but TaCT is source-specific. The
reported ASR is ~1/num_classes of the true source-class ASR, causing TaCT to
be excluded from tables for a measurement artifact.

## Issues confirmed NOT to affect main results

These were checked and are correct:
- PSU computation (`psbd_metrics.py:114`): `tracked - per_pass_probs.mean(dim=0)`
- AUROC one-sided (`psbd_metrics.py:242`): negates scores, 0=clean/1=backdoor
- ASR set construction: `is_eval_poisonable`/`attack_success_label` used everywhere
- Clean-label eligibility inversion: `== target` for training, `!= target` for eval
- Sigma matching in `psbd_metrics.py`: clean-validation only, no label leak
- Clean/backdoor pairing via manifest
- Threshold at 25th percentile of clean validation PSU
- `model.eval()` + `inference_mode()` on all eval paths
- `plug_dropout`/`unplug_dropout` with try/finally in sweep code
- `encoder.dropout` skip in `activate_model_dropout`
- SAM two-pass update (Foret et al., rho=0.1)
- Uniform 15 epochs across all training
- Trigger applied pre-normalization, identically at train and eval
- FlaggedPoisonedSet confined to attacker's training path
- Held-out validation set never accessed by attacker
