# Repository layout

One place for every kind of thing, and a rule for deciding where a new file goes.

```
attacks/        the 10 trigger implementations and the shared poisoning, adversarial and evasion logic.
data/           dataset registry, loading, the PSBD split and the BackdoorBench PNG path.
models/         the 2 architectures and the probe position registry.
training/       the training loop, checkpoint provenance and SAM.
defences/       PSBD itself: operators, inference, scores, decision rules and the stage-1 cache.
detectors/      the competitor input detectors, 1 registry, 1 module per method.
analysis/       latent-space statistics, no matplotlib: CKA, backdoor direction, TAC, Lipschitz, PCA, UMAP and the BackdoorBench ports.
visualization/  the figures over those statistics, 1 module per figure family, tools registered for cli.visualize.
evaluation/     attack success rate, clean accuracy, loaders and summary.
utils/          numerics only, helpers with no subject of their own.
cli/            1 module per command, the only place a main() lives.
scripts/        repo-level tooling: the coverage ledger, the table generators, the prose audit.
pbs/            cluster job generators and the job scripts they emit.
experiments/    hypothesis-driven experiments, 1 directory per question, each with a README.
notebooks/      the guided tour, numbered 00 to 14.
docs/           hypotheses, results, specs, audits.
configs/        canonical constants pinned outside code, for example psbd_basis.json.
tests/          the suite.
scratch/        gitignored. Throwaway only, never the only source of a published number.
third_party/    gitignored. Reference implementations cloned at pinned commits.
results/        per-checkpoint output, written on demand.
checkpoints/    trained models, 1 folder per run, with an args.json sidecar.
```

## Where a new file goes

Ask what the file is, not what it is about.

| the file is | it goes in | test it by |
|---|---|---|
| a function another package calls | the package named for its subject | importing it in `tests/` |
| a thing you run from a terminal | `cli/` | calling `python -m cli.<name> --help` and running it small |
| a question you asked once, with a written answer | `experiments/<question>/` | running it on real data |
| something that writes `.pbs` files | `pbs/` | asserting the generated text |
| a repo-level tool with no single package as its home | `scripts/` | running it against the current tree |
| a note nobody runs | `docs/` | reading it |
| a file you would not mind losing | `scratch/` | not at all |

The distinction that matters most is the first two. Anything in `cli/` must be
thin enough that deleting it costs nothing but the argument parsing, because
every claim in the paper is reproduced by calling into a package, never by
re-running a script whose logic lives in its own `main()`.

## The packages

Each package is named for its subject, not for which paper section it serves.

| package | responsibility |
|---|---|
| `attacks/` | the 10 attacks (`badnet`, `blend`, `sig`, `wanet`, `lf`, `lc`, `bpp`, `adaptive_blend`, `tact`, `generated`), shared trigger `patterns`, `poisoning` (training and eval eligibility, index selection, `AttackSuccessSet`), `adversarial` and `bases` (Label-Consistent's adversarial bases), and `evasion` (the adaptive attacker) |
| `data/` | `registry` (`DATASET_REGISTRY`, normalization statistics, `label_mode_from_folder`), `loading`, `splits` (the PSBD split and `read_checkpoint_metadata`), and `backdoorbench` (the PNG path for `backdoor_bench_checkpoints/`) |
| `models/` | `backbones` (ViT-B/16 and Swin-S, `load_checkpoint`, `network_core`) and `positions` (`POSITION_REGISTRY`, `plug_dropout`, `unplug_dropout`, `activate_model_dropout`) |
| `training/` | `loop` (the training loop, `save_checkpoint` and provenance) and `sam` |
| `defences/` | `operators` (what is injected), `inference` (forward passes, `forward_logits`, `frozen_parameters`), `scores` (PSU and the shift ratio), `decision` (rate rules, quantile thresholds, `detection_report`, the canonical placements), and `cache` (the stage-1 tensors under `results/<folder>/psbd/`) |
| `detectors/` | the competitor detectors behind 1 registry (`DETECTOR_NAMES`, `build_detector`, `DetectorContext`), 1 module per method, and `records` for their on-disk layout under `results/<folder>/detectors/` |
| `analysis/` | `features` (`captured_layers`), `cka`, `direction`, `embedding`, `lipschitz`, `distribution`, `stealth`, `cases`, `latent` |
| `evaluation/` | `metrics` (attack success and clean accuracy), `loaders` and `summary` |
| `utils/` | numerics only |
| `cli/` | training (`train_backdoor`, `train_benign`), PSBD (`sweep`, `analyze`, `summary`, `tables`, `report`, `operating_points`, `variants`), competitors (`baselines`, `compare_detectors`, `fuse_detectors`), and others (`evaluate`, `backfill`, `lc_bases`, `head_profile`, `analyze_latent`) |

## The import rule

Relative imports inside a package, absolute imports across packages. No
package re-exports its submodules except `attacks/__init__.py`, which is a
registry (`ATTACK_NAMES`, `build_attack`, `default_config`) so external call
sites (`from attacks import build_attack`) do not need to know which
submodule an attack lives in.

## `results/` and `checkpoints/`

`checkpoints/` holds every trained model, 1 folder per run, named by the
canonical template in `.claude/CLAUDE.md`. Every folder carries an
`args.json` provenance sidecar next to `attack_result.pt`.

`results/<folder_name>/` is not created at training time. It is written on
demand, with 2 per-checkpoint subtrees: `psbd/`, created by `cli.sweep`,
holding the raw per-pass probabilities that `cli.analyze` reads to write
`psbd_metrics.json`, and `detectors/`, created by `cli.baselines`, holding 1
record per competitor detector. An orphaned `results/<folder_name>/` with no
matching `checkpoints/<folder_name>/` is inert, not deleted.

## Testing

Every layer is reachable from a test without a GPU or a cluster:

- Each package is pure library code, so it is tested directly.
- `cli/` is tested by asserting its argument parsers and by running the cheap
  entrypoints end to end on cached data.
- `pbs/` generates text, so its output is asserted as text.
- `experiments/` reads cached tensors, so it runs on CPU on the login node.
- `tests/test_canon.py` holds the constants in `defences/decision.py` to
  `configs/psbd_basis.json`, so a placement or rate rule cannot drift between
  the code and the pinned config without a failing test.

The expensive half, PSBD's stage 1, is the only part needing a GPU. Everything
downstream reads the caches it writes.
