# Notebooks

Executable documentation. Every notebook runs top to bottom against the real
repository, calls the same library functions a script or a test would, and stores
its outputs so it can be read without running it.

They exist for 2 reasons. They teach the codebase by using it, and they are the
strictest test of whether the library is actually modular: anything that needs a
command line, a config file, or a working directory to be useful will not fit in a
cell.

## Conventions

Every notebook opens with the same setup cell, which anchors at the repository
root so the library's default relative paths (`checkpoints/`, `raw_data/`,
`results/`) resolve exactly as they do from a script:

```python
import os
import sys
from pathlib import Path

REPO_ROOT = next(
    parent for parent in [Path.cwd(), *Path.cwd().parents]
    if (parent / "pyproject.toml").exists()
)
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
```

Beyond that:

- Call library functions from `psbd.*`. A notebook that defines its own analysis
  is a notebook whose numbers cannot be reproduced by anything else.
- Plot with helpers that return a `Figure`. Nothing writes an image to disk.
- Tables are `pandas.DataFrame`, built from the plain dicts the library returns.
- Prose explains why a cell exists and what its output means. A notebook whose
  markdown only restates the code is not documentation.
- Notebooks are committed **with** their outputs, so the numbers are readable
  without a GPU and a change in them shows up in a diff.

## Reading order

`00-start-here.ipynb` is the index. It lists every notebook with the question it
answers, separates what is settled from what was withdrawn or downgraded, and
states the known gaps rather than leaving them to be discovered.

If you read only 2, read **06** for the method and **12** for whether to believe
any of the numbers.

## Running them

```bash
uv run jupyter lab                     # interactive
uv run python -m nbconvert --to notebook --execute --inplace notebooks/NAME.ipynb
```

Notebooks marked GPU need a CUDA device and a few minutes. Notebooks marked CPU
read cached results under `results/` and run anywhere in seconds.
