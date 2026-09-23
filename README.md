# PSBD-ViT

Prediction Shift Backdoor Detection (PSBD) adapted from ConvNets to Vision
Transformers. The method flags a poisoned input by how little its prediction moves
when the network is perturbed at inference. It was designed for ResNets, where the
residual stream is the one obvious place to put that perturbation. A transformer
block offers many places and many kinds of perturbation, and this project measures
which choice works.

Models are ViT-B/16 and Swin-S. The panel covers CIFAR-10, CIFAR-100, GTSRB, Tiny
ImageNet, SVHN and EuroSAT against 10 backdoor attacks, including an adaptive
attacker trained against the defense.

**The paper is the record.** `paper/` holds the draft, its generated tables and its
figures, all built from `results/` by `scripts/paper/`. No number in it is typed by
hand. `docs/open-questions.md` lists what the evidence does not yet support.

## The result

Where the perturbation goes decides whether the method works at all. Masking whole
tokens at the input of every attention block reaches a mean AUROC of **0.935**
against **0.832** for the placement the original paper used, dropout on the residual
stream after the add. That is a paired gain of **+0.103** with a 95% bootstrap
interval of [+0.048, +0.160] over the models that carry the full basis, and the gain
grows as poisoning falls, which is the regime a defender cares about most.

Position and operator both matter and neither reduces to the other. Holding the site
fixed and swapping token masking for Gaussian noise costs **0.206**. Holding the
operator fixed and moving from the attention input to the MLP input costs **0.109**.
The project's founding claim, that dropout before the residual add beats dropout
after it, is **refuted** at matched shift ratio
([H1](docs/hypothesis/H1-pre-beats-post.md)).

The reason is mechanical. A backdoor in a ViT is 1 direction in the residual stream,
written in the last third of the network and routed through attention from the
trigger's own tokens to the class token. Removing whole tokens before attention
removes that route. Noise at the same site is absorbed by the LayerNorm behind it.

The original paper's account, that perturbation shifts clean predictions toward the
attacker's target class, holds on ViT only for local triggers on datasets with few
classes. It is absent for global triggers such as Blend and TaCT, while detection
works in both cases, so that account is attack dependent on ViT rather than the
mechanism. An earlier headline of +0.258 for `gain_scale` at `mlp_norm_out` stays
**withdrawn** ([the audit](docs/audit-2026-09-07.md)).

### The panel, and what it excludes

Read from `results/coverage/COVERAGE.md`, the tracked view of the coverage ledger.

| | cells |
|---|---:|
| trained | 105 |
| diverged, clean accuracy below half the benign reference | 3 |
| below the attack success bar of 0.85 | 31 |
| clearing the bar | 71 |
| of those, carrying 18 or more basis placements | 65 |

The headline reads on those 65. 2 of them invert, both on CIFAR-10 at 10% poisoning.
Adaptive-Blend never clears the attack success bar at any rate, so no Adaptive-Blend
model enters the detection numbers despite being the 1 attack in the set built to
evade this kind of detector.

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

Stage 1 on GPU. Position and operator are separate axes: `--position`
is where the perturbation is injected, `--operator` is what is injected.
`--checkpoint-folder` takes bare folder names, resolved under
`--checkpoints-dir` (default `checkpoints`).

```bash
python -m cli.sweep \
    --checkpoint-folder vit_cifar100_badnet_a2o_0_1 \
    --position before_attention_norm \
    --operator token_mask
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
defenses/       PSBD: operators, inference, scores, decision rules, the stage-1 cache
detectors/      competitor input detectors, 1 registry, 1 module per method
analysis/       latent-space analysis: TAC, CKA, PCA, UMAP, Lipschitz
visualization/  the figures over analysis/'s statistics, 1 module per figure family
evaluation/     attack success rate, clean accuracy, loaders, summary
utils/          numerics and provenance helpers, no subject of their own
cli/            1 module per command, the only place a main() lives
scripts/        repo-level tools: the coverage ledger, scripts/paper/, the prose audit
tests/          the suite
docs/hypothesis/  pre-registered hypotheses with verdicts
docs/results/     detection tables, analysis reports, protocol docs
experiments/       hypothesis-driven experiments, 1 directory per question, indexed in docs/experiments-index.md
notebooks/          executable documentation, committed with outputs (00 is the index)
pbs/                 PBS job generators for cluster scheduling
```

Full layout and the rule for where a new file goes: `docs/repository-layout.md`.

### Commands

| Command | Purpose |
|---|---|
| `cli.train_backdoor` | Train a backdoored ViT or Swin model |
| `cli.train_benign` | Train benign ViT models, the negative controls |
| `cli.sweep` | Stage 1: GPU sweep over a (checkpoint, position, operator) triple, writes raw per-pass probabilities |
| `cli.analyze` | Stage 2: CPU analysis of a checkpoint's cache, writes psbd_metrics.json |
| `cli.baselines` | Score the competitor detectors on exactly the splits PSBD is scored on |
| `cli.compare placements report` | Aggregate every psbd_metrics.json into a single per-checkpoint table |
| `cli.compare placements summary` | Collapse every psbd_metrics.json into a single compact table |
| `cli.compare placements tables` | Per-dataset detection tables, with the coverage bar enforced in code |
| `cli.compare placements variants` | PSBD as published against the recommended ViT configuration, head to head |
| `cli.compare operating-points` | Detection at low false-positive rates (1%, 5%) |
| `cli.compare detectors` | 1 table per metric, PSBD against every competitor detector |
| `cli.compare fused` | Fuse PSBD with 1 named competitor detector by rank |
| `cli.evaluate` | Attack-success, clean-accuracy and stealth metrics for every checkpoint |
| `cli.visualize` | Render 1 registered figure tool over a checkpoint's cached statistics, 1 subcommand per tool |
| `cli.backfill` | Recover metadata fields that are deducible from artifacts already on disk |
| `cli.lc_bases` | Generate the adversarially perturbed base images the Label-Consistent attack needs |
| `cli.head_profile` | Per-head sensitivity profile, ablating each attention head in turn |
| `cli.analyze_latent` | Latent-space analysis of a checkpoint: TAC, backdoor direction, CKA, PCA |

7 reporting commands were folded into `cli.compare`'s 4 subcommands
(`placements`, `operating-points`, `detectors`, `fused`) shown above. Their
old names still run: `cli.report`, `cli.summary`, `cli.tables` and
`cli.variants` each forward to a `placements` action, `cli.operating_points`
forwards to `operating-points`, `cli.compare_detectors` forwards to
`detectors`, and `cli.fuse_detectors` forwards to `fused`. Run
`python -m cli.compare --help` and `python -m cli.compare <subcommand> --help`
for every flag.

`scripts/` holds repo-level tools that are not commands: `coverage_ledger.py`,
`scripts/paper/build_all.py` and its generators, `vit_config_inventory.py`,
`vit_config_tables.py`, `vit_detection_tables.py`, `vit_shift_target_compare.py`,
`vit_top3_tables.py`, `verify_results.py`, `verify_splits.py`, `prose_audit.py`
and `check_prose_only.py`.

## Competitor detectors

`python -m cli.baselines` scores 11 registered detectors on exactly the PSBD
splits and quantiles: `confidence`, `strip`, `scale_up`,
`scale_up_data_limited`, `ibd_psc`, `ibd_psc_calibrated`, `teco`, `cd_l`,
`beatrix`, `ted` and `sentinet`. Every one returns low for poisoned, with at
most 1 negation at its own scoring boundary. The 3 that fit per-sample
statistics on the validation split return out-of-fit scores for it and are
listed in `detectors.CROSS_FITTED`.

`python -m cli.compare detectors` reads those records beside PSBD's own and
writes the like-for-like comparison table. `docs/detectors/README.md` holds the
registry, the on-disk layout and each port's numerical cross-check against its
reference implementation.

## Attacks

10 attacks, all implemented in `attacks/` behind 1 registry
(`ATTACK_NAMES`, `build_attack`, `default_config`): BadNet in both label modes
(all-to-one and all-to-all), Blend, SIG, WaNet, LF, Label-Consistent, BPP,
Adaptive-Blend and TaCT.

SIG and Label-Consistent are clean label, so they can only poison the target
class and their reachable poison rate is that class's share of the training set.
`docs/clean-label-rate-caps.md` records the cap per dataset and why GTSRB
clean-label runs use target class 1.

## Reproducing results

The full experiment grid runs on a PBS cluster. Job generators in `pbs/`
produce per-checkpoint job scripts. The two-stage pipeline:

1. `python -m cli.sweep` runs on GPU, writes raw per-pass probabilities
2. `python -m cli.analyze` runs on CPU, reads cached data, writes psbd_metrics.json

The compact, versioned record is `results/detection_summary.csv.gz`, 1 row
per (checkpoint, placement, rate rule), regenerated with `python -m cli.summary`.
Each checkpoint's own `results/<checkpoint>/psbd_metrics.json` is regenerable
from the stage-1 cache and is not versioned.

For the adaptive attacker experiments, `python -m cli.train_backdoor --evade-psbd`
trains evasive models, with `--evade-position`, `--evade-operator` and
`--evade-weight` selecting the probe it trains against. Analysis scripts in
`experiments/adaptive_attack/`, `experiments/multi_probe/`, and
`experiments/adaptive_defender/` produce the transfer, multi-probe and forensic
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

## The paper

`paper/` is the record. Everything in it is generated from `results/` by the
generators under `scripts/paper/`, and `paper/headline.tex` holds every headline
number as a macro with its own provenance comment.

```bash
cd paper
make tables     # rerun every generator, fold the macro sidecars into headline.tex
make            # compile main.pdf with tectonic
```

`make tables` refuses to finish when a section types a digit outside a macro, a
citation, a reference, an input path or a year. That guard is what keeps the
compiled paper and the results tree from drifting apart.

tectonic is the only TeX on this cluster and it lives at
`~/.local/bin/tectonic`, fetching its packages through the proxy on first use.
There is no system LaTeX, so `pdflatex` and `latexmk` will not work.

## Checks

```bash
.venv/bin/python -m pytest tests -q          # the suite
.venv/bin/ruff format <files> && .venv/bin/ruff check <files>
.venv/bin/python scripts/prose_audit.py <paths>        # the style rules
.venv/bin/python scripts/prose_audit.py --gate <paths> # exit 1 on any hit
```

`prose_audit.py` reads Python comments and docstrings, markdown bodies and LaTeX
sections, and checks the rules in `.claude/styles/`: no semicolons, no em dashes,
no arrows, no Oxford comma, digits rather than number words and American English
spelling.

## Notebooks

`notebooks/` is the guided tour and is committed with its outputs. It follows the
paper rather than the package layout. Start at `start-here.ipynb`, and
`notebooks/README.md` gives the reading order. Run them from the repository root
so the packages import:

```bash
PYTHONPATH=. .venv/bin/jupyter lab
```

## Older reports

`docs/` holds the working record: `docs/hypothesis/` has every pre-registered
hypothesis with its verdict, `docs/runs/` has the training and sweep logs and
`docs/detectors/` documents each port. Several documents under `docs/results/`
predate the current panel and still quote the retired 48-cell numbers, so treat
`paper/` as authoritative wherever the 2 disagree.
