# Balanced panels

Four conclusions in this project inverted when rebuilt on a common set of
checkpoints. All four failed identically: a mean was taken over whatever cells
happened to exist, so the groups being compared had been measured on different
problems, and the placements with the most coverage looked worst because their extra
checkpoints were the hard ones.

`audit.py` finds the largest complete (group x unit) block and reports the naive
answer beside the balanced one, flagging disagreement. It is meant to be run before
any table in `docs/hypothesis/` or `.claude/docs/` is believed.

```bash
PYTHONPATH=. python experiments/balanced_panels/audit.py --architecture vit
PYTHONPATH=. python experiments/balanced_panels/audit.py --group rho --metric oracle
```

What it found on first run: the ViT placement ranking disagrees between the naive
and balanced tables, and the balanced version supports a family-level conclusion
([H20](../../docs/hypothesis/H20-input-side-beats-residual-adjacent.md)) rather than
a single winning position.
