# Repository layout

The top-level tree, 1 line per directory and per file. Package internals and
the rule for where a new file goes are in `.claude/CLAUDE.md` and
`docs/architecture.md`.

```
analysis/                  latent-space statistics: CKA, backdoor direction, TAC, PCA, UMAP, Lipschitz, the BackdoorBench ports.
archive/                   frozen pre-reorganisation sweep output, kept for comparison, never written to again.
attacks/                   the 10 trigger implementations, the shared poisoning and evasion logic.
backdoor_bench_checkpoints/  BackdoorBench's own downloaded checkpoints, read-only, evaluated through the PNG path.
checkpoints/                every trained model, 1 folder per run, with an args.json provenance sidecar.
cli/                        1 module per command, the only place a main() lives.
configs/                    canonical constants pinned outside code, for example psbd_basis.json.
data/                       dataset registry, loading, the PSBD split, the BackdoorBench PNG path.
defences/                   PSBD itself: operators, inference, scores, decision rules, the stage-1 cache.
detectors/                  the 11 competitor input detectors behind 1 registry, 1 module per method.
docs/                       hypotheses, results, specs, audits, per-experiment writeups outside experiments/.
evaluation/                 attack success rate, clean accuracy, loaders, summary.
experiments/                hypothesis-driven experiments, 1 directory per question, most with a README.
figures/                    figure output that predates the visualization/ package's JSON sidecar convention.
logs/                       PBS job stdout and stderr, 1 subdirectory per job family, matching pbs/.
models/                     the 2 architectures and the probe position registry.
notebooks/                  the guided tour, numbered 00 to 14.
paper/                      the built paper: sections, tables, figures, headline.tex, main.pdf.
papers/                     local copies of the source papers this project reproduces or ports from.
pbs/                        cluster job generators and the job scripts they emit.
raw_data/                   downloaded dataset archives and extracted images.
results/                    per-checkpoint output, written on demand, plus results/_experiments/ and results/coverage/.
scratch/                    gitignored, throwaway only, never the only source of a published number.
scripts/                    repo-level tooling: the coverage ledger, scripts/paper/, the prose audit.
tests/                      the suite.
tmp/                        gitignored scratch output from smoke runs.
training/                   the training loop, checkpoint provenance, SAM.
utils/                      numerics and provenance helpers, no subject of their own.
visualization/              the figures over analysis/'s statistics, 1 module per figure family, JSON sidecar per figure.
.gitignore                  patterns kept out of version control: raw_data/, archive/, figures/, checkpoints/, logs/, tmp/, backdoor_bench_checkpoints/, scratch/, papers/, results/ and the rest of the generated tree.
pyproject.toml              the project's Python 3.11 and dependency pins, read by uv.
README.md                   installation, the quick start, the command table, pointers into docs/.
uv.lock                     the resolved dependency lock uv installs from.
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

The distinction that matters most is the first 2. Anything in `cli/` must be
thin enough that deleting it costs nothing but the argument parsing, because
every claim in the paper is reproduced by calling into a package, never by
re-running a script whose logic lives in its own `main()`.

## The import rule

Relative imports inside a package, absolute imports across packages. No
package re-exports its submodules except `attacks/__init__.py`, which is a
registry (`ATTACK_NAMES`, `build_attack`, `default_config`) so external call
sites (`from attacks import build_attack`) do not need to know which
submodule an attack lives in.

## Testing

Every layer is reachable from a test without a GPU or a cluster. Each package
is pure library code, so it is tested directly. `cli/` is tested by asserting
its argument parsers and by running the cheap entrypoints end to end on
cached data. `pbs/` generates text, so its output is asserted as text.
`experiments/` reads cached tensors, so it runs on CPU on the login node.
`tests/test_canon.py` holds the constants in `defences/decision.py` to
`configs/psbd_basis.json`, so a placement or rate rule cannot drift between
the code and the pinned config without a failing test.

The expensive half, PSBD's stage 1, is the only part needing a GPU. Everything
downstream reads the caches it writes.
