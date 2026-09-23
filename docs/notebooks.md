# Notebooks

`notebooks/` is the guided tour of PSBD-ViT. It follows `paper/`, not the
package layout, so each notebook answers the question its matching section of
the paper asks and reads the same files the paper's own table and figure
generators under `scripts/paper/` read. Every notebook is committed with its
outputs, runs from the repository root with `PYTHONPATH=.` set and stays
under 10 minutes on the A100 the project trains on. Re-execute a notebook with:

```bash
PYTHONPATH=. .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/<name>.ipynb --ExecutePreprocessor.timeout=1800
```

## start-here

The vocabulary every other notebook assumes: site, perturbation, placement,
rate, sweep and analysis, the basis, a band, PSBD-TM and PSBD-RD, as well as the
prediction shift uncertainty statistic itself. It shows the 2 commands,
`cli.sweep` and `cli.analyze`, that produce every detection number in the
paper, states which file on disk each command reads and writes, and gives the
reading order for the notebooks that follow, which `notebooks/README.md`
also lists. It runs no model and reads no
large file, under 10 seconds.

## data-and-attacks

Loads CIFAR-100 and builds every one of the 9 distinct triggers from the same
attack registry the training code uses, plots each trigger against a clean
image with its pixel statistics, and confirms which triggers stamp identical
pixels and which merely agree at evaluation time. The second half works
through the poisoning protocol: why a clean-label attack's training-eligible
and evaluation-eligible pools are opposites of each other, and why a poison
rate above a class's share of the training set trains the identical index set
as the cap itself. Reads `paper/sections/13-attacks.tex` for the full
parameter and attack-success record. CPU only, under 15 seconds.

## psbd-end-to-end

Runs the method itself on 1 checkpoint, `vit_cifar100_badnet_a2o_0_01`:
attaches PSBD-TM and PSBD-RD in turn, sweeps a rate ladder for each, and
reads off the adaptive, matched and oracle rate rules from
`defenses.decision`. Draws the PSU histograms `scripts/paper/fig_psu_histograms.py`
builds from the cache on disk, computed here directly instead, with the
clean-validation threshold marked, and checks the false-positive rate at
every quantile against the quantile itself. Closes with the shift-target
histogram the original paper's neuron-bias explanation predicts. Needs a GPU,
about 75 seconds.

## placement-walk

Reads the same 4 staircase tables `scripts/paper/tab_staircase.py` writes to
`paper/tables/`, recomputed here from `results/<folder>/psbd_metrics.json`
rather than the generated LaTeX: the residual stream against PSBD-RD, dropout
moved across every site, the operator held fixed against the site held fixed,
and a band of blocks against acting in every block. Closes with the full
basis ranked by mean AUROC, the same figure `scripts/paper/fig_basis_ranking.py`
draws. Establishes that position and operator both move the detection result
and neither reduces to the other. CPU only, reads `results/coverage/coverage.json`
and roughly 70 models' cached metrics, about 80 seconds.

## mechanism

Explains the ranking `placement-walk` reads. Traces the backdoor direction's depth
profile and tests it causally by steering clean activations toward it,
reads `results/<folder>/activation_patching.json` to show where the network's
decision causally sits at each depth, reads `results/<folder>/cls_routing.json`
to show attention moving the trigger's tokens toward the class token in the
second half of the network, and checks directly on a synthetic tensor why a
LayerNorm removes more of an additive noise perturbation's norm than a
token-masking perturbation's. Needs a GPU for the direction and steering
sections, about 35 seconds.

## swin-and-robustness

Reads every `swin_*` checkpoint's `psbd_metrics.json` to check whether the
ViT ranking transfers to windowed attention, reads
`results/adaptive_attacker_analysis.json` and `results/multi_probe_analysis.json`
to show an attacker trained against 1 probe collapsing that probe while
leaving 2 untrained-against probes standing, and a rank-union defense
restoring most of what was lost, and reads `results/_experiments/psbd_cost/cost.json`
for the wall-clock cost of the defense at several pass counts. CPU only,
reads cached JSON exclusively, about 10 seconds.

## reproducing-the-paper

Shows how `scripts/paper/build_all.py` regenerates every table and figure in
`paper/` from `results/` in 1 pass, and how it refuses a chapter that
carries a number no generator produced. Reads `results/coverage/coverage.json`
to show the attack-success and clean-accuracy bar that decides which of the
105 trained models enter a detection table, and works through 2 kinds of
mistakes that have previously produced a wrong published number, a comparison
between 2 placements read at unmatched disturbance and a stage-2 record
left stale by a regenerated stage-1 cache, together with the checks now in
place against each. States what in the paper is still single-seed and what
is still running. CPU only, under 10 seconds.

## detectors-per-attack

Breaks the paper's detector comparison out by attack. Reads the same
population `scripts/paper/tab_detectors.py` builds its tables from, the
clearing models that carry a reading from every defense, and draws 1 heatmap
of mean AUROC with attacks as rows and the 13 defenses as columns, PSBD-TM and
PSBD-RD first. A table beside it names the leading defense per attack and the
rank of both PSBD placements. CPU only, reads cached JSON exclusively, under
10 seconds.

## attention-probes

Every attention-side probe the sweeps cover at full depth, inside the declared
basis and outside it, following `scripts/paper/tab_attention.py`. The figure
sets each probe's unpaired mean AUROC beside its paired gap to PSBD-TM over
the models carrying both, with a bootstrap interval, so a probe swept on a
small subset of the panel cannot pass for a better one. CPU only, reads cached
JSON exclusively, under 10 seconds.

## swin-per-attack

PSBD-TM against PSBD-RD on Swin-S per attack, over the `results/swin_*`
models `scripts/paper/tab_swin.py` reads. The figure shows both placements'
paired means per attack with every model behind them as a faint point, and a
table counts the models on which the adaptive rule found no rate. CPU only,
reads cached JSON exclusively, under 10 seconds.

## depth-bands

PSBD-TM restricted to blocks 1 to 4, 5 to 8 and 9 to 12 against acting in
every block, read at the matched rule the way `scripts/paper/tab_depth_bands.py`
reads it. A first table shows why the adaptive rule cannot compare bands, a
figure sets each band's mean beside its paired gap to all blocks with a
bootstrap interval. CPU only, reads cached JSON exclusively, under 10 seconds.

## What executed and what did not

The 7 section notebooks executed cleanly, start to finish, with 0 errors,
against the `results/` and `checkpoints/` on disk when they were committed,
and the 4 figure notebooks did the same when they were added. Nothing was
skipped or stubbed out. The one place a number in a notebook can drift from
the paper's own published figure is the model count in `placement-walk`: the
paper's headline table is frozen at a fixed commit, 65 models, while
`results/` keeps growing between that commit and whenever the notebook next
runs, so a later run can read a slightly larger set without the ranking
changing. The notebook states this rather than pretending to match the frozen
count exactly.

The 7 section notebooks import `scripts.paper._style`, which selects the Agg
backend for the paper build, so their `plt.show()` calls embed no image and
their committed outputs carry tables and printed numbers only. The 4 figure
notebooks switch back to the inline backend after the same import, and their
committed outputs carry the figures.
