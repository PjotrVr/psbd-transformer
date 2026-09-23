# Notebooks

Executable documentation, 1 notebook per section of `paper/` plus 4 short
figure notebooks. Every notebook runs top to bottom against the real
repository, calls the same library functions `scripts/paper/` and the `cli.*`
commands call, and is committed with its outputs so its numbers are readable
without a GPU. See `docs/notebooks.md` for what each notebook shows and how
long it runs.

## Reading order

1. `start-here.ipynb`, the vocabulary and the 2 commands behind every detection number.
2. `data-and-attacks.ipynb`, every trigger and the poisoning protocol.
3. `psbd-end-to-end.ipynb`, PSBD-TM and PSBD-RD run on 1 checkpoint (GPU).
4. `placement-walk.ipynb`, the staircase tables that separate the site from the operator.
5. `mechanism.ipynb`, why the ranking looks the way it does (GPU).
6. `swin-and-robustness.ipynb`, Swin-S, an adaptive attacker and the cost of the defense.
7. `reproducing-the-paper.ipynb`, how `paper/` is regenerated from `results/`.
8. `detectors-per-attack.ipynb`, every defense's AUROC per attack as a heatmap.
9. `attention-probes.ipynb`, every attention-side probe against PSBD-TM.
10. `swin-per-attack.ipynb`, PSBD-TM against PSBD-RD per attack on Swin-S.
11. `depth-bands.ipynb`, PSBD-TM restricted to 4 blocks against all 12.

The last 4 read cached JSON only and draw 1 or 2 figures each, so they run on
a login node in seconds.

```bash
PYTHONPATH=. .venv/bin/python -m jupyter lab                      # interactive
PYTHONPATH=. .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/NAME.ipynb --ExecutePreprocessor.timeout=1800       # re-execute one
```
