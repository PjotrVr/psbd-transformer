# Repository layout

One place for every kind of thing, and a rule for deciding where a new file goes.

```
psbd/           the library. Importable, tested, no side effects at import time.
cli/            thin entrypoints, one per verb. Argument parsing and orchestration only.
experiments/    research scripts, one directory per question, each with a README.
pbs/            cluster job generators and the grid flattener.
tests/          the suite.
docs/           hypotheses, results, specs, audits.
scratch/        gitignored. Throwaway only.
archive/        gitignored. Superseded output kept for reference.
```

## Where a new file goes

Ask what the file is, not what it is about.

| the file is | it goes in | test it by |
|---|---|---|
| a function other code calls | `psbd/` | importing it in `tests/` |
| a thing you run from a terminal | `cli/` | calling its `parse_args` and running it small |
| a question you asked once, with a written answer | `experiments/<question>/` | running it on real data |
| something that writes `.pbs` files | `pbs/` | asserting the generated text |
| a note nobody runs | `docs/` | reading it |
| a file you would not mind losing | `scratch/` | not at all |

The distinction that matters most is the first two. Anything in `cli/` must be
thin enough that deleting it costs nothing but the argument parsing, because
every claim in the paper is reproduced by calling into `psbd/`, never by
re-running a script whose logic lives in its own `main()`.

## The library

`psbd/` splits along what a module is responsible for, not along which paper
section it serves.

| module | responsibility |
|---|---|
| `config.py` | dataset registry, normalization statistics, run configuration |
| `data.py` | loading clean datasets, transforms, reading labels without decoding images |
| `poisoning.py` | label modes, which samples are eligible, the dataset wrappers that apply triggers |
| `splits.py` | the standardized PSBD split and the manifest that pins its row order |
| `models.py` | building and loading ViT-B/16 and Swin-S |
| `sam.py` | the sharpness-aware two pass update |
| `training.py` | the training loop and checkpoint provenance |
| `evaluation.py` | attack success rate, clean accuracy, per-class accuracy |
| `eval_loaders.py` | full test set loaders, clean and triggered |
| `backdoorbench.py` | reading BackdoorBench's pregenerated PNG triggers |
| `positions.py` | **where** a probe attaches inside a transformer block |
| `operators.py` | **what** a probe does to the activation it sees |
| `inference.py` | the no-dropout baseline and the stochastic passes |
| `scores.py` | per-sample quantities: PSU, fractional PSU, shift ratio, critical rate |
| `decision.py` | turning scores into verdicts: thresholds, rate selection, detection reports |
| `cache.py` | the on-disk layout of stage 1 output |
| `baselines.py` | STRIP, the confidence null, and the ported detectors |
| `evasion.py` | the adaptive attacker that trains against the defence |
| `stealth.py` | how visible a trigger is, in pixel space |
| `attacks/` | the 10 trigger implementations and the registry that dispatches them |
| `analysis/` | latent tools: CKA, backdoor direction, TAC, Lipschitz, PCA and UMAP |

The split between `positions.py` and `operators.py` is the one worth
understanding, because it is the structure of the research question rather than a
filing convenience. A probe is a **position** crossed with an **operator**, and
the central result is that the position matters 1.43 times more than the operator.
Keeping them in separate modules means adding a new position never touches an
operator and adding a new operator never touches a position.

`scores.py` does not import `decision.py`. A per-sample number is meaningful
without a threshold, and keeping that direction one way stops the scoring code
from acquiring opinions about false-positive budgets.

## Why there is no `utils/`

There was one, holding `config.py` and `datasets.py`. Neither was a utility. One
is the dataset registry and the other is the data loading layer, and both now sit
in `psbd/` under their real names.

A `utils/` directory is where files go when nobody has decided what they are, and
it grows monotonically because nothing is ever obviously wrong to put there. Every
file in this repository has a home that describes its responsibility, so the
question "does this belong in utils" never has to be asked. The one genuine
cross-cutting tool, the PBS grid flattener, lives in `pbs/grid.py` next to the
generators that use it.

## Testing

Every layer is reachable from a test without a GPU or a cluster:

- `psbd/` is pure library code, so it is tested directly.
- `cli/` is tested by asserting its argument parsers and by running the cheap
  entrypoints end to end on cached data.
- `pbs/` generates text, so its output is asserted as text.
- `experiments/` reads cached tensors, so it runs on CPU on the login node.

The expensive half, stage 1, is the only part needing a GPU. Everything
downstream reads the caches it writes, which is why the pipeline was split in 2
in the first place.

## The transition, and how it ends

`old/` holds the pre-rewrite tree. It is temporary and exists for 1 reason: the
equivalence tests import both trees at once and assert they produce identical
results, so the old code has to stay importable until that proof is no longer
needed.

```
old/            the pre-rewrite tree. Deleted by the cleanup branch.
conftest.py     puts old/ on sys.path so its modules import unchanged.
```

The old modules import each other by their original top-level names, for example
`from poison import Attack`. Rather than rewrite those imports in code that is
scheduled for deletion, `conftest.py` appends `old/` to `sys.path` and every one
of them resolves as it always did. There is no collision with the new package,
since the old names are `poison`, `models` and `attacks` while the new ones are
`psbd.poisoning`, `psbd.models` and `psbd.attacks`.

3 test files are transitional and die with `old/`:

| file | what it proves |
|---|---|
| `tests/test_rewrite_equivalence.py` | split manifests, loaders, poisoning, models and SAM are bit identical |
| `tests/test_rewrite_probes.py` | every probe position and operator is bit identical, and the cache paths match |
| `tests/test_rewrite_library.py` | attacks, evaluation, baselines, evasion and analysis are bit identical |

### The cleanup branch

When the rewrite has been exercised for long enough to trust it, a branch does
exactly 4 things and nothing else:

1. delete `old/`
2. delete `conftest.py`
3. delete the 3 transitional test files above
4. record their final results in `audit-2026-09-07.md` as the permanent evidence

Keeping that branch to those 4 deletions is deliberate. A cleanup that also
changes behaviour cannot be reviewed as a cleanup, and the whole value of the
equivalence tests is that the diff which removes them contains no other change.
