# PSBD-ViT

Prediction Shift Backdoor Detection adapted from ConvNets to Vision
Transformers (ViT-B/16 and Swin-T). Evaluated on CIFAR-10, CIFAR-100,
GTSRB, and Tiny ImageNet against 10 backdoor attacks, including an
adaptive attacker that knows the defence.

## Key findings

**Detection.** Token masking before the attention LayerNorm achieves mean
AUROC 0.911 on CIFAR-100 across all attacks and poison rates, with zero
inversions and worst-case floor 0.632. Position matters more than operator
choice (position variance 1.43x operator variance).

**PSBD measures decision margin, not neuron bias.** Gaussian noise (an
unstructured perturbation with no neuron-dropping semantics) matches the best
structured masks at AUROC 0.950. The mechanism is perturbation of the
decision boundary, not selective disruption of backdoor neurons.

**Adaptive attacker.** A hinge penalty that matches poisoned PSU onto the
clean distribution collapses the probed operator's AUROC from 0.952 to 0.322,
while preserving ASR (delta -0.004) at a cost of 4.8 points CA.

**Multi-probe defence.** Min-rank union of k independent perturbation
operators recovers AUROC to 0.951 on evasive checkpoints. Evasion is
probe-specific: different operators produce different Jacobians, so the
attacker cannot evade all simultaneously without prohibitive memory cost.

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

```bash
python train_backdoor.py \
    --architecture vit \
    --dataset cifar100 \
    --attack badnet \
    --label_mode all_to_one \
    --poison_rate 0.1
```

### 2. Run the PSBD sweep

```bash
python psbd_dropout_sweep.py \
    --checkpoint checkpoints/vit_cifar100_badnet_a2o_0_1 \
    --positions before_attention_norm_token_mask
```

### 3. Analyze detection metrics

```bash
python psbd_analyze.py --checkpoint checkpoints/vit_cifar100_badnet_a2o_0_1
python psbd_report.py
```

## Project structure

```
attacks/             10 attack implementations + registry
analysis/            Latent-space analysis: TAC, CKA, PCA, UMAP, Lipschitz
defences/            PSBD detection: dropout hooks, inference, metrics, cache
utils/               Dataset specs, data loading, transforms
scripts/             Hypothesis-driven experiments (one per subdirectory)
tests/               Test suite (200+ tests)
docs/hypothesis/     42 pre-registered hypotheses with verdicts
docs/results/        Detection tables, analysis reports, protocol docs
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

For the adaptive attacker experiments, `train_backdoor.py --adaptive` trains
evasive models. Analysis scripts in `scripts/adaptive_attack/`,
`scripts/multi_probe/`, and `scripts/adaptive_defender/` produce the
transfer, multi-probe, and forensic identification results.

## Hypothesis register

All 42 hypotheses are pre-registered in `docs/hypothesis/` with predictions,
methodology, and verdicts. See `docs/hypothesis/README.md` for the full index
and reporting standards.

## Results

Full detection tables, operating point analysis, and the adaptive defender
protocol are in `docs/results/`. The top-level report is
`docs/results-report.md`.
