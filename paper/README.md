# The paper folder

Everything the paper needs, in 1 place, generated from `results/` by the scripts under
`scripts/paper/`. No number here is typed by hand. A table or figure names its generator,
its inputs, the commit and the time in its first line, a headline number is a LaTeX macro in
`headline.tex` whose provenance is in `headline.json`, and a finding is 1 entry in
`findings.md` with its grade, its evidence path and where the draft puts it. The chapter
text under `chapters/` is the primary record of every result and `findings.md` is its
machine index.

## Regenerating

```
PYTHONPATH=. python scripts/paper/build_all.py \
    --results-dir /lustre/home/pstika/projects/PSBD-ViT/results --paper-dir paper
PYTHONPATH=. python scripts/paper/build_all.py --check-only --paper-dir paper
```

`build_all.py` runs every generator in name order (`tab_*`, `fig_*`, `mech_*`, `app_*`),
folds the macro sidecars into `headline.tex`, renders `findings.rendered.md`, runs the
ledgers (`ledger_*`) that read the folded headline, and refuses to finish when a chapter
carries a digit literal outside a macro, a citation, a reference, a proper name or a year,
or names a macro `headline.tex` does not define. Every generator also runs alone with the
same flags. Generators that read `checkpoints/<folder>/args.json` take `--checkpoints-dir`.

The per-layer TAC figure reads a record a GPU pass writes, and that pass is not part of the
build:

```
PYTHONPATH=. python scripts/paper/run_tac_layers.py --checkpoints-dir checkpoints
```

It writes `results/_experiments/tac_layers/tac_layers.json` in the worktree, about a minute
per checkpoint on the login-node A100, and `mech_tac_layers.py` prints a pending table when
the record is absent so the build never depends on it.

## Compiling

No LaTeX installation exists on the cluster (`which pdflatex latexmk` and `module avail`
return nothing). The draft was compiled with a standalone `tectonic` binary fetched through
the proxy into the session's scratch directory, from inside `paper/`:

```
cd paper && /path/to/tectonic --keep-logs main.tex
```

Any chapter compiles alone through the `subfiles` class, `tectonic chapters/05-headline.tex`,
which is what the `\documentclass[../main.tex]{subfiles}` line at the top of every chapter is
for. Without a compiler, `build_all.py --check-only` is the validation.

## The criticality scale

A grade is a claim about evidence, so each has criteria a reader can check against a number.

| grade | criteria, all required | where it goes |
|---|---|---|
| CRITICAL | carries the paper. A detection claim spans at least 30 cells on the basis panel across at least 3 datasets, read at matched shift ratio, paired within cell, with a 5000-resample bootstrap 95% interval excluding 0, positive under leave-one-attack-out, every number generated. A mechanism claim rests on a causal intervention with a random-control null at 0 and a benign control at chance, replicated on at least 10 checkpoints across at least 2 trigger families | main text, a numbered figure or table |
| STRONG | at least 12 cells or paired units, or at least 2 datasets, interval excluding 0, at least 1 held-out or replication check, may be single seed. A mechanism claim: a causal or zero-parameter prediction confirmed on at least 3 checkpoints across at least 2 attack families with a benign control | main text, may share a table |
| SUPPORTING | consistent with a CRITICAL or STRONG finding, at least 3 cells, the interval may include 0 or be uncomputed, may be 1 dataset | 1 sentence in the main text, table in the appendix |
| WEAK | single checkpoint or single cell, no interval, no benign control, or read at unmatched strength. Labelled provisional | appendix only, or omitted |
| NEGATIVE | a founding or pre-registered claim refuted, or a number withdrawn. The evidence class of the refutation is stated beside it, since a refutation can rest on CRITICAL evidence (H1) or on WEAK evidence (H29) | the failures chapter, then limitations or appendix |

In the draft a grade appears as a margin tag, `\finding{GRADE}{Fnn}`, beside the sentence
that states the claim, and each chapter opens with a `claim` block giving the grade, the
finding id and what the chapter's experiment would refute. Both are defined in
`preamble.tex` and both print nothing when `\draftnotesfalse` is set.

## Layout

- `main.tex`: the root, `\input{preamble}`, `\input{headline}`, then 1 `\subfile` per chapter.
- `preamble.tex`: packages and the `\finding`, `claim`, `\pending` and `\code` macros.
- `references.bib`: the bibliography. Author lists are the verified ones, an unverified list
  is written as `and others`.
- `headline.tex`, `headline.json`: every headline number as a macro, with provenance.
- `findings.md`: the registry, every finding once. `findings.rendered.md` is the same with
  macros substituted and every evidence path checked.
- `chapters/`: 1 `.tex` per chapter, `<nn>-<slug>.tex`, each compilable alone.
- `tables/`: 1 `.tex` per table, plus the `.macros.json` sidecar a generator contributes.
- `figures/`: 1 `.pdf` per figure with a `.json` sidecar of the plotted numbers.
- `draft/`: retired, a pointer to this structure.

The raw artefacts every table reads live under `results/`: per-checkpoint files at
`results/<folder>/` and cross-checkpoint files at `results/_experiments/<slug>/`. One table,
`detector_smoke.tex`, reads a run entry under `docs/runs/` because the records it
transcribes were written under an untracked directory that has since been cleared, and its
first line says so.

## Chapters

| file | subject |
|---|---|
| `00-abstract` | the abstract |
| `01-introduction` | the port as a search over position and operator, the 4 contributions |
| `02-background` | PSBD as its authors state it, quoted, the threat model, the architectures |
| `03-method` | PSU and the shift ratio in source notation, positions, operators, the 2 rate rules, the threshold |
| `04-setup` | the panel, training, naming, coverage and provenance |
| `05-headline` | transfer, the recommended placement, leave-one-attack-out, seed replicates |
| `06-placement` | the founding claim and the family split refuted, the site and operator effects, position over operator with its condition, per-sample agreement |
| `07-depth-bands` | banding helps residual dropout and hurts the input-side mask, the early band pending |
| `08-prediction-shift` | the shift-to-target phenomenon per attack and dataset, removal against disturbance, the confidence null |
| `09-latent` | the rank-1 direction and its ablation, crystallisation and the per-layer TAC pass, activation patching, routing and sinks, prediction depth |
| `10-competitors` | the detector registry and the smoke run, panel pending |
| `11-swin` | what exists on Swin-S |
| `12-clean-label` | the rate cap, SIG on GTSRB, the diverged runs, Label-Consistent, multi-target SIG, SVHN and EuroSAT |
| `13-planned-mechanism` | specifications of the mechanism experiments, done or pending with their commands |
| `14-missing` | what each headline claim still lacks, and the macro ledger |
| `15-limitations` | all-to-all, the adaptive attacker, withdrawn and refuted claims, scope |
| `16-appendix` | the full placement basis, supplementary tables, the hypothesis ledger |
