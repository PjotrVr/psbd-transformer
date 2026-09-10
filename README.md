# PSBD-ViT

Prediction Shift Backdoor Detection adapted from ConvNets to Vision
Transformers (ViT-B/16 and Swin-S). Evaluated on CIFAR-10, CIFAR-100,
GTSRB, and Tiny ImageNet against 10 backdoor attacks, including an
adaptive attacker that knows the defence.

## Key findings

Research project adapting Prediction Shift Backdoor Detection (PSBD) from ConvNets to Vision Transformers (ViT-B/16 and Swin-S). The project's founding claim, that dropout placed **before** the residual add beats placing it after, is **refuted** ([H1](docs/hypothesis/H1-pre-beats-post.md)): measured at matched shift ratio, the pre-versus-post gap is **+0.002**, indistinguishable from noise ([H20](docs/hypothesis/H20-input-side-beats-residual-adjacent.md)). What the evidence supports instead is a search result. Where the perturbation is injected dominates what is injected (position variance 1.43x operator variance), and the split that carries the effect is input-side against residual-adjacent, **+0.054** mean AUROC with bootstrap CI [+0.031, +0.080], with both pre-residual and post-residual sitting in the losing family. The recommended deployment configuration is `token_mask` at `before_attention_norm`, which gains **+0.089** mean AUROC over the published ConvNet placement at matched clean-validation shift ratio 0.6 across the full 48-cell panel, stable at +0.081 to +0.103 across shift ratios 0.2 to 0.8, and on CIFAR-100 at 1% poisoning specifically **+0.166** (n=4). Its absolute numbers are mean AUROC 0.911, worst-case floor 0.632, 0 inversions over 48/48 cells. An earlier headline of +0.258 for `gain_scale` at `mlp_norm_out` is **withdrawn**: it read the winner at shift ratio 0.95 to 0.98 and the baseline at 0.65 to 0.76, a disturbance gap the same size as the reported effect, and at matched shift ratio over the full panel that arm beats the published placement by **-0.007** ([the audit](docs/audit-2026-09-07.md)). The mechanism underneath is decision-margin estimation. The direct evidence is that no shifted clean prediction lands on the attacker's target class, against a uniform expectation of 1%, so the neuron-bias account in the original PSBD paper is not what carries the method on ViT. H23's supporting claim is restated after the batch-coupled Gaussian was corrected and its cells re-swept: gaussian at `before_attention` scores **0.897**, not the withdrawn 0.950, which places it 6th rather than 1st but still within 0.002 of `channel_mask` and above `dropout` at `pre_residual` ([audit A19](docs/audit-2026-09-07.md)). An operator removing no capacity remains competitive with ones that do, which is the claim that refutes the capacity-removal account. Validated on CIFAR-10, CIFAR-100, GTSRB and Tiny ImageNet. Full numbers in `docs/results-report.md`, verdicts in `docs/hypothesis/README.md`.

## Installation

Requires Python 3.11, PyTorch 2.6.0 with CUDA 12.4.

```bash
uv sync
source .venv/bin/activate
```

## Quick start

Every command is a module, `python -m cli.<name>`.

### 1. Train a backdoored model

The label mode is part of the attack name (`badnet_a2o` is all-to-one,
`badnet_a2a` is all-to-all), and `--output` is the path to the checkpoint
file, whose parent directory is the folder name every later stage refers to.

```bash
python -m cli.train_backdoor \
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
python -m cli.sweep \
    --checkpoint-folder vit_cifar100_badnet_a2o_0_1 \
    --position-config before_attention_norm \
    --perturbation token_mask
```

### 3. Analyze detection metrics

Stage 2 on CPU. `--all` covers every folder under `results/` that already has
a stage-1 cache under `psbd/`.

```bash
python -m cli.analyze --checkpoint-folder vit_cifar100_badnet_a2o_0_1
python -m cli.report --format markdown
```

## Project structure

```
attacks/        10 attack implementations, shared trigger patterns, poisoning eligibility, evasion
data/           dataset registry, loading, the PSBD split, the BackdoorBench PNG path
models/         ViT-B/16 and Swin-S, the probe position registry
training/       the training loop, checkpoint provenance, SAM
defences/       PSBD: operators, inference, scores, decision rules, the stage-1 cache
detectors/      competitor input detectors, 1 registry, 1 module per method
analysis/       latent-space analysis: TAC, CKA, PCA, UMAP, Lipschitz
evaluation/     attack success rate, clean accuracy, loaders, summary
utils/          numerics only
cli/            1 module per command, the only place a main() lives
scripts/        repo-level tools: the coverage ledger, table generators, the prose audit
tests/          the suite
docs/hypothesis/  47 pre-registered hypotheses with verdicts
docs/results/     detection tables, analysis reports, protocol docs
notebooks/        executable documentation, committed with outputs (00 is the index)
pbs/               PBS job generators for cluster scheduling
```

Full layout and the rule for where a new file goes: `docs/repository-layout.md`.

### Commands

| Command | Purpose |
|---|---|
| `cli.train_backdoor` | Train a backdoored ViT or Swin model |
| `cli.train_benign` | Train benign ViT models, the negative controls |
| `cli.sweep` | Stage 1: GPU sweep over dropout rates for a (checkpoint, position) pair, writes raw per-pass probabilities |
| `cli.analyze` | Stage 2: CPU analysis of a checkpoint's cache, writes psbd_metrics.json |
| `cli.summary` | Collapse every psbd_metrics.json into a single compact table |
| `cli.tables` | Per-dataset detection tables, with the coverage bar enforced in code |
| `cli.report` | Aggregate every psbd_metrics.json into a single per-checkpoint table |
| `cli.operating_points` | Detection at low false-positive rates (1%, 5%) |
| `cli.variants` | PSBD as published against the recommended ViT configuration, head to head |
| `cli.baselines` | Score the competitor detectors on exactly the splits PSBD is scored on |
| `cli.compare_detectors` | 1 table per metric, PSBD against every competitor detector |
| `cli.fuse_detectors` | Fuse PSBD and STRIP, whose failures are disjoint |
| `cli.evaluate` | Attack-success, clean-accuracy and stealth metrics for every checkpoint |
| `cli.backfill` | Recover metadata fields that are deducible from artifacts already on disk |
| `cli.lc_bases` | Generate the adversarially perturbed base images the Label-Consistent attack needs |
| `cli.head_profile` | Per-head sensitivity profile, ablating each attention head in turn |
| `cli.analyze_latent` | Latent-space analysis of a checkpoint: TAC, backdoor direction, CKA, PCA |

`scripts/` holds repo-level tools that are not commands: `coverage_ledger.py`,
`vit_config_inventory.py`, `vit_config_tables.py`, `vit_detection_tables.py`,
`vit_shift_target_compare.py`, `vit_top3_tables.py`, `verify_results.py`,
`verify_splits.py`, `prose_audit.py` and `check_prose_only.py`.

## Competitor detectors

`python -m cli.baselines` scores STRIP, SCALE-UP, IBD-PSC, TeCo, CD-L,
Beatrix, TED and SentiNet on exactly the PSBD splits and quantiles.
`python -m cli.compare_detectors` reads those records beside PSBD's own and
writes the like-for-like comparison table. See `docs/detectors/README.md`
for the registry, the on-disk layout and each port's cross-check against its
reference implementation.

## Attacks

BadNet (all-to-one and all-to-all), Blend, SIG, WaNet, LF, LC, BPP,
Adaptive-Blend, TaCT. All implementations in `attacks/`.

## Reproducing results

The full experiment grid runs on a PBS cluster. Job generators in `pbs/`
produce per-checkpoint job scripts. The two-stage pipeline:

1. `python -m cli.sweep` runs on GPU, writes raw per-pass probabilities
2. `python -m cli.analyze` runs on CPU, reads cached data, writes psbd_metrics.json

The compact, versioned record is `results/detection_summary.csv.gz`, one row
per (checkpoint, placement, rate rule), regenerated with `python -m cli.summary`.
Each checkpoint's own `results/<checkpoint>/psbd_metrics.json` is regenerable
from the stage-1 cache and is not versioned.

For the adaptive attacker experiments, `python -m cli.train_backdoor --evade-psbd`
trains evasive models, with `--evade-position`, `--evade-operator` and
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

Every hypothesis is pre-registered in `docs/hypothesis/` with predictions,
methodology, and verdicts. See `docs/hypothesis/README.md` for the full index
and reporting standards.

## Results

Full detection tables, operating point analysis, and the adaptive defender
protocol are in `docs/results/`. The top-level report is
`docs/results-report.md`.
