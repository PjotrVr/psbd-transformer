# The paper folder

Everything the paper needs, in 1 place, generated from `results/` by the scripts under
`scripts/paper/`. No number here is typed by hand. A table or figure names its generator,
its inputs, the commit and the time in its first line, a headline number is a LaTeX macro in
`headline.tex` whose provenance is in `headline.json`, and a finding is 1 entry in
`findings.md` with its grade, its evidence path and where the draft puts it.

## Regenerating

```
PYTHONPATH=. python scripts/paper/build_all.py --results-dir results
```

`build_all.py` runs every generator in order, folds the macro sidecars into `headline.tex`,
renders `findings.rendered.md`, and refuses to finish when a chapter carries a digit literal
outside a macro, a citation or a reference, or names a macro `headline.tex` does not define.
Every generator also runs alone with the same flags.

## The criticality scale

A grade is a claim about evidence, so each has criteria a reader can check against a number.

| grade | criteria, all required | where it goes |
|---|---|---|
| CRITICAL | carries the paper. A detection claim spans at least 30 cells on the basis panel across at least 3 datasets, read at matched shift ratio, paired within cell, with a 5000-resample bootstrap 95% interval excluding 0, positive under leave-one-attack-out, every number generated. A mechanism claim rests on a causal intervention with a random-control null at 0 and a benign control at chance, replicated on at least 10 checkpoints across at least 2 trigger families | main text, a numbered figure or table |
| STRONG | at least 12 cells or paired units, or at least 2 datasets, interval excluding 0, at least 1 held-out or replication check, may be single seed. A mechanism claim: a causal or zero-parameter prediction confirmed on at least 3 checkpoints across at least 2 attack families with a benign control | main text, may share a table |
| SUPPORTING | consistent with a CRITICAL or STRONG finding, at least 3 cells, the interval may include 0 or be uncomputed, may be 1 dataset | 1 sentence in the main text, table in the appendix |
| WEAK | single checkpoint or single cell, no interval, no benign control, or read at unmatched strength. Labelled provisional | appendix only, or omitted |
| NEGATIVE | a founding or pre-registered claim refuted, or a number withdrawn. The evidence class of the refutation is stated beside it, since a refutation can rest on CRITICAL evidence (H1) or on WEAK evidence (H29) | the failures chapter, then limitations or appendix |

## Layout

- `headline.tex`, `headline.json`: every headline number as a macro, with provenance.
- `findings.md`: the registry, every finding once. `findings.rendered.md` is the same with
  macros substituted and every evidence path checked.
- `chapters/`: 1 `.tex` per chapter, opening with a generated block of its findings.
- `tables/`: 1 `.tex` per table, plus the `.macros.json` sidecar a generator contributes.
- `figures/`: 1 `.pdf` per figure with a `.json` sidecar of the plotted numbers.
- `draft/`: `main.tex`, `preamble.tex`, `references.bib`, the preliminary paper.

The raw artefacts every table reads live under `results/`: per-checkpoint files at
`results/<folder>/` and cross-checkpoint files at `results/_experiments/<slug>/`.
