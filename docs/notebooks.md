# Notebooks

`notebooks/` is the guided tour of PSBD-ViT. It follows `paper/`, not the
package layout, so each notebook answers the question its matching section of
the paper asks and reads the same files the paper's own table and figure
generators under `scripts/paper/` read. Every notebook is committed with its
outputs, runs from the repository root with `PYTHONPATH=.` set, and stays
under 10 minutes on the A100 the project trains on. Re-execute one with:

```bash
PYTHONPATH=. .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/<name>.ipynb --ExecutePreprocessor.timeout=1800
```

## 00-start-here

The vocabulary every other notebook assumes: site, perturbation, placement,
rate, sweep and analysis, the basis, a band, PSBD-TM and PSBD-RD, and the
prediction shift uncertainty statistic itself. It shows the two commands,
`cli.sweep` and `cli.analyze`, that produce every detection number in the
paper, states which file on disk each command reads and writes, and gives the
reading order for the notebooks that follow. It runs no model and reads no
large file, under 10 seconds.

## 01-data-and-attacks

Loads CIFAR-100 and builds every one of the 9 distinct triggers from the same
attack registry the training code uses, plots each trigger against a clean
image with its pixel statistics, and confirms which triggers stamp identical
pixels and which merely agree at evaluation time. The second half works
through the poisoning protocol: why a clean-label attack's training-eligible
and evaluation-eligible pools are opposites of each other, and why a poison
rate above a class's share of the training set trains the identical index set
as the cap itself. Reads `paper/sections/13-attacks.tex` for the full
parameter and attack-success record. CPU only, under 15 seconds.

## 02-psbd-end-to-end

Runs the method itself on one checkpoint, `vit_cifar100_badnet_a2o_0_01`:
attaches PSBD-TM and PSBD-RD in turn, sweeps a rate ladder for each, and
reads off the adaptive, matched and oracle rate rules from
`defences.decision`. Draws the PSU histograms `scripts/paper/fig_psu_histograms.py`
builds from the cache on disk, computed here directly instead, with the
clean-validation threshold marked, and checks the false-positive rate at
every quantile against the quantile itself. Closes with the shift-target
histogram the original paper's neuron-bias explanation predicts. Needs a GPU,
about 75 seconds.

## 03-placement-walk

Reads the same 4 staircase tables `scripts/paper/tab_staircase.py` writes to
`paper/tables/`, recomputed here from `results/<folder>/psbd_metrics.json`
rather than the generated LaTeX: the residual stream against PSBD-RD, dropout
moved across every site, the operator held fixed against the site held fixed,
and a band of blocks against acting in every block. Closes with the full
basis ranked by mean AUROC, the same figure `scripts/paper/fig_basis_ranking.py`
draws. Establishes that position and operator both move the detection result
and neither reduces to the other. CPU only, reads `results/coverage/coverage.json`
and roughly 70 models' cached metrics, about 80 seconds.

## 04-mechanism

Explains the ranking notebook 03 reads. Traces the backdoor direction's depth
profile and tests it causally by steering clean activations toward it,
reads `results/<folder>/activation_patching.json` to show where the network's
decision causally sits at each depth, reads `results/<folder>/cls_routing.json`
to show attention moving the trigger's tokens toward the class token in the
second half of the network, and checks directly on a synthetic tensor why a
LayerNorm removes more of an additive noise perturbation's norm than a
token-masking perturbation's. Needs a GPU for the direction and steering
sections, about 35 seconds.

## 05-swin-and-robustness

Reads every `swin_*` checkpoint's `psbd_metrics.json` to check whether the
ViT ranking transfers to windowed attention, reads
`results/adaptive_attacker_analysis.json` and `results/multi_probe_analysis.json`
to show an attacker trained against one probe collapsing that probe while
leaving two untrained-against probes standing, and a rank-union defence
restoring most of what was lost, and reads `results/_experiments/psbd_cost/cost.json`
for the wall-clock cost of the defence at several pass counts. CPU only,
reads cached JSON exclusively, about 10 seconds.

## 06-reproducing-the-paper

Shows how `scripts/paper/build_all.py` regenerates every table and figure in
`paper/` from `results/` in one pass, and how it refuses a chapter that
carries a number no generator produced. Reads `results/coverage/coverage.json`
to show the attack-success and clean-accuracy bar that decides which of the
105 trained models enter a detection table, and works through two kinds of
mistakes that have previously produced a wrong published number, a comparison
between two placements read at unmatched disturbance and a stage-2 record
left stale by a regenerated stage-1 cache, together with the checks now in
place against each. States what in the paper is still single-seed and what
is still running. CPU only, under 10 seconds.

## What executed and what did not

All 7 notebooks execute cleanly, start to finish, with zero errors, against
the repository's current `results/` and `checkpoints/` on disk. Nothing was
skipped or stubbed out. The one place a number in a notebook can drift from
the paper's own published figure is notebook 03's model count: the paper's
headline table is frozen at a fixed commit, 65 models, while `results/`
keeps growing between that commit and whenever the notebook next runs, so a
later run can read a slightly larger set without the ranking changing. The
notebook states this rather than pretending to match the frozen count
exactly.
