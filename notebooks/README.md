# Notebooks

Executable documentation, one notebook per section of `paper/`. Every notebook
runs top to bottom against the real repository, calls the same library
functions `scripts/paper/` and the `cli.*` commands call, and is committed
with its outputs so its numbers are readable without a GPU. See
`docs/notebooks.md` for what each notebook shows and how long it runs, and
`00-start-here.ipynb` for the vocabulary and the reading order.

```bash
PYTHONPATH=. .venv/bin/python -m jupyter lab                      # interactive
PYTHONPATH=. .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/NAME.ipynb --ExecutePreprocessor.timeout=1800       # re-execute one
```
