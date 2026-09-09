# PSBD-ViT

Prediction Shift Backdoor Detection adapted from ConvNets to Vision
Transformers (ViT-B/16 and Swin-S). Evaluated on CIFAR-10, CIFAR-100,
GTSRB, and Tiny ImageNet against 10 backdoor attacks, including an
adaptive attacker that knows the defence.

## Key findings

**Detection.** Token masking before the attention LayerNorm achieves mean
AUROC 0.911 on CIFAR-100 across all attacks and poison rates, with zero
inversions and worst-case floor 0.632. Position matters more than operator
choice (position variance 1.43x operator variance).

**PSBD measures decision margin.** An unstructured perturbation with no
neuron-dropping semantics matches the best structured masks, so removing capacity
is not what carries the method. A second order expansion says why: the prediction
shift is a curvature measurement, the perturbation site sets the curvature and the
operator sets only the noise covariance, and softmax curvature falls as the
decision margin grows. Four independent negative results on structured masking
follow from the same expression. See `docs/theory-perturbation-consistency.md`.

The headline Gaussian number is being re-measured. It was produced by an operator
whose noise was scaled by a batch wide statistic, and the position it most
affected is the one the previous figure came from.

**Adaptive attacker.** A hinge penalty that matches poisoned PSU onto the clean
distribution collapses the probed operator's AUROC from 0.952 to 0.322 while
preserving ASR, delta -0.004. It costs 4.8 points of clean accuracy, which is
**above the 2 point budget this project's own threat model set** as the line
between evasion and simply damaging the model. It is reported as a partial
evasion, and tuning the penalty weight down is the obvious next step.

**Multi-probe defence.** The min-rank union of k independent perturbation
operators recovers detection on evasive checkpoints, where a single probed
operator collapses to 0.322. Evasion is probe specific, and the reason is
structural: a perturbation operator sets the covariance of the noise, an attacker
minimising the gap for 1 operator constrains only the curvature it can see, and a
second operator reads a projection that was never constrained. See
`docs/theory-perturbation-consistency.md`.

The exact AUROC is being re-measured. The published 0.951 included a Gaussian
probe whose noise was scaled by a batch wide statistic, which ran the backdoor
split 13 to 32 percent hotter than the clean split it was compared against. The
operator is fixed and the re-sweep is queued. Correcting the threshold rule at the
same time raised the defence's TPR from 0.877 to 0.979 at a correctly calibrated
25 percent false positive budget, with AUROC unmoved.

## Installation

Requires Python 3.11, PyTorch 2.6.0 with CUDA 12.4.

```bash
uv sync
source .venv/bin/activate
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Quick start

### 1. Train a backdoored model

The label mode is part of the attack name (`badnet_a2o` is all-to-one,
`badnet_a2a` is all-to-all), and `--output` is the path to the checkpoint
file, whose parent directory is the folder name every later stage refers to.

```bash
python train_backdoor.py \
    --architecture vit \
    --dataset cifar100 \
    --attack badnet_a2o \
    --poison-rate 0.1 \
    --epochs 15 \
    --output checkpoints/vit_cifar100_badnet_a2o_0_1/attack_result.pt
```

### 2. Run the PSBD sweep

Stage 1 on GPU. Position and operator are separate axes: `--position-config`
is where the perturbation is injected, `--perturbation` is what is injected.
`--checkpoint-folder` takes bare folder names, resolved under
`--checkpoints-dir` (default `checkpoints`).

```bash
python psbd_dropout_sweep.py \
    --checkpoint-folder vit_cifar100_badnet_a2o_0_1 \
    --position-config before_attention_norm \
    --perturbation token_mask
```

### 3. Analyze detection metrics

Stage 2 on CPU. `--all` covers every folder under `results/` that already has
a stage-1 cache.

```bash
python psbd_analyze.py --checkpoint-folder vit_cifar100_badnet_a2o_0_1
python psbd_report.py
```

## Project structure

```
attacks/             10 attack implementations + registry
analysis/            Latent-space analysis: TAC, CKA, PCA, UMAP, Lipschitz
defences/            PSBD detection: dropout hooks, inference, metrics, cache
utils/               Dataset specs, data loading, transforms
experiments/         Hypothesis-driven experiments (one per subdirectory, each with a README)
scripts/             Repo-level tools: detection_summary, verify_results, backfill_metadata
tests/               Test suite (200+ tests)
docs/hypothesis/     42 pre-registered hypotheses with verdicts
docs/results/        Detection tables, analysis reports, protocol docs
notebooks/           Executable documentation, committed with outputs (00 is the index)
pbs/                 PBS job generators for cluster scheduling
```

### Root-level scripts

| Script | Purpose |
|---|---|
| `train_backdoor.py` | Train a backdoored ViT or Swin model |
| `train_benign.py` | Train clean negative controls |
| `psbd_dropout_sweep.py` | Stage 1: GPU sweep over positions and rates |
| `psbd_analyze.py` | Stage 2: CPU analysis, write psbd_metrics.json |
| `psbd_report.py` | Aggregate all metrics into a single table |
| `psbd_variants.py` | Ablation: published PSBD vs ViT-adapted variant |
| `psbd_operating_points.py` | Detection at deployable FPR (1%, 5%) |
| `metrics.py` | Compute ASR and clean accuracy for all checkpoints |
| `baseline_detect.py` | STRIP and confidence baseline detectors |
| `detector_comparison.py` | PSBD vs baselines comparison table |
| `detector_fusion.py` | PSBD + STRIP rank fusion |
| `defence_tables.py` | Per-dataset detection tables with coverage enforcement |
| `adaptive_evasion.py` | Adaptive attacker: hinge penalty training |
| `stealth.py` | Trigger stealth metrics: PSNR, SSIM, LPIPS |

## Attacks

BadNet (all-to-one and all-to-all), Blend, SIG, WaNet, LF, LC, BPP,
Adaptive-Blend, TaCT. All implementations in `attacks/`.

## Reproducing results

The full experiment grid runs on a PBS cluster. Job generators in `pbs/`
produce per-checkpoint job scripts. The two-stage pipeline:

1. `psbd_dropout_sweep.py` runs on GPU, writes raw per-pass probabilities
2. `psbd_analyze.py` runs on CPU, reads cached data, writes psbd_metrics.json

Results are tracked in `results/<checkpoint>/psbd_metrics.json` (versioned).

For the adaptive attacker experiments, `train_backdoor.py --evade-psbd` trains
evasive models, with `--evade-position`, `--evade-operator` and
`--evade-weight` selecting the probe it trains against. Analysis scripts in
`experiments/adaptive_attack/`, `experiments/multi_probe/`, and
`experiments/adaptive_defender/` produce the transfer, multi-probe, and forensic
identification results.

## Hypothesis register

### Findings and corrections

Defects that changed how results must be read, each with its blast radius:

| document | what it records |
|---|---|
| `docs/status-2026-09-09.md` | entry point for the 2026-09-09 work: 3 findings, the fixes, the queue, what is next |
| `docs/clean-label-rate-caps.md` | which clean-label poison rates are reachable, the GTSRB target-class fix, and the missing Label-Consistent adversarial step |
| `docs/attack-strength-and-implantation.md` | why attacks fail to implant, sorted into 4 causes, and the WaNet strength dose-response |
| `docs/checkpoint-integrity-2026-09-09.md` | 21 training runs written as unreadable files, and the guard that now catches it |
| `docs/audit-2026-09-07.md` | 21 audit findings, including the withdrawn +0.258 headline |
| `docs/gtsrb-training-split-mismatch.md` | this repo's GTSRB split is not the one the literature uses |

All 42 hypotheses are pre-registered in `docs/hypothesis/` with predictions,
methodology, and verdicts. See `docs/hypothesis/README.md` for the full index
and reporting standards.

## Results

Full detection tables, operating point analysis, and the adaptive defender
protocol are in `docs/results/`. The top-level report is
`docs/results-report.md`.
