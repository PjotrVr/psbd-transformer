# Operator and position ranking at matched disturbance (H17)

## Question

The published PSBD configuration is dropout at `pre_residual`, and it collapses at
1% poisoning. Is that a limit of prediction-shift detection, or a bad operating
point? The 7 scripts here answer it from one cached surface, so no 2 of them are
allowed to disagree about what a cell means.

Every comparison is at a matched clean-validation shift ratio (sigma) rather than
at a shared dropout rate. Comparing at a shared rate would only measure which
operator perturbs hardest. Every score is one-sided: low PSU means poisoned, and a
value below 0.5 is printed as the failure it is rather than flipped.

## The 7 scripts

| script | what it produces |
| --- | --- |
| `build_surface.py` | the `(folder, placement, rate) -> (sigma, AUROC)` table every other script slices, cached to `scratch/surface.json` |
| `analyze_surface.py` | the defender-legal configuration search, ranked by mean AUROC per poison rate with the benign control alongside |
| `head_to_head.py` | the published configuration against the best defender-legal one, per checkpoint |
| `sigma_target.py` | which sigma target the rate rule should aim at, scored by the AUROC each target would have selected |
| `lowrate_diagnosis.py` | the full (placement, rate) surface for 1 checkpoint, to separate a broken premise from a broken rate rule |
| `operator_ranking.py` | operators ranked per attack at matched sigma, with an unreachable operating point reported as `--` rather than dropped |
| `full_stack_1pct.py` | what every accumulated change buys at 1% poisoning and 5% FPR, both deployable and ROC |

`build_surface.py` writes its cache into `scratch/` on purpose. It is a derived
table of about 2600 cells, regenerable from `results/<folder>/psbd/` in 1 command,
so it is data rather than a result.

## Running it

Repo root has to be the working directory, because the cache paths are relative to
it.

    PYTHONPATH=. python experiments/operator_ranking/build_surface.py <folders>
    PYTHONPATH=. python experiments/operator_ranking/analyze_surface.py
    PYTHONPATH=. python experiments/operator_ranking/head_to_head.py
    PYTHONPATH=. python experiments/operator_ranking/sigma_target.py
    PYTHONPATH=. python experiments/operator_ranking/operator_ranking.py
    PYTHONPATH=. python experiments/operator_ranking/full_stack_1pct.py

All of them are CPU only and read cached per-pass probabilities. No GPU job is
needed once the sweep has run.

## Finding

The low-poison-rate failure is an operating point, not a limit. On CIFAR-100 at 1%,
moving from dropout at `pre_residual` to `token_mask` at `before_attention_norm`
gains 0.162 mean AUROC at the swept rate, and 0.166 at matched shift ratio. The
`gain_scale` at `mlp_norm_out` figure of 0.258 is WITHDRAWN: that arm is read at
shift ratio 0.95 to 0.98 against a baseline at 0.65 to 0.76, and at matched shift
ratio over the full 48-cell panel it gains -0.007. The
published configuration reads 0.687 mean where `gain_scale` reads 0.945.

`head_to_head.py` shows where the gain comes from: `badnet_a2o` at 1% moves from
0.297 to 0.839, which is an inverted detector becoming a working one. The benign
control stays at 0.494 against 0.504, so the candidate configuration is not simply
reading confidence. All-to-all still fails under both, which is H5 and a separate
problem.

Hypothesis doc: `docs/hypothesis/H17-low-poison-rate-is-a-placement-artifact.md`.
Published tables: `docs/results/operator-position-ranking.md`,
`docs/results/detection-operating-points.md`.
