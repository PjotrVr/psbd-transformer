# Experiments

1 directory per question, each holding its scripts and a `README.md` stating the
question, how to run it, and what was found. Everything here is tracked, because a
number in `docs/` that cannot be traced back to a commit is not a result.

Throwaway work goes in `scratch/`, which is gitignored and holds smoke tests, job
generators, already-run migrations, and regenerable caches. Nothing in `scratch/`
may be the only source of a published number.

Repo-level tools are not experiments and stay in `scripts/`: `detection_summary.py`,
`verify_results.py`, and `backfill_metadata.py`.

## Conventions

Run everything from the repo root. Data paths are relative to it, so the working
directory is load bearing even for the scripts that bootstrap `sys.path`
themselves.

    PYTHONPATH=. python experiments/<question>/<script>.py

Analysis output goes to `results/`, either as `results/<name>.json` for a
cross-checkpoint result or `results/<checkpoint>/<name>.json` for a per-checkpoint
one. Which of the 2 a script uses is stated in its own README.

Scoring is one-sided everywhere: low PSU means poisoned, and an AUROC below 0.5 is
reported as the method failing on that attack rather than flipped into a win. That
rule is H15 and it is not negotiable. Operator and placement comparisons are made at
a matched clean-validation shift ratio rather than at a shared rate, or they would
only measure which operator perturbs hardest.

## Index

| directory | question | hypothesis | cited by |
| --- | --- | --- | --- |
| `adaptive_attack/` | does evasion trained against 1 operator transfer to others? | H25 | `docs/results/adaptive-attacker-analysis.md` |
| `adaptive_defender/` | which operator was evaded, and what protocol should the defender run? | H42 | `docs/results/adaptive-defender-protocol.md` |
| `attack_viability/` | which (attack, dataset, rate) combinations produced a backdoor that actually fires? | gate for every sweep | `pbs/generate_psbd_jobs.py` |
| `attention_heads/` | do a few heads route the trigger, and does their entropy detect it? | H31, H40 | `docs/hypothesis/H31-*.md`, `docs/hypothesis/H40-*.md` |
| `backdoor_direction/` | what geometry does the backdoor direction have in the residual stream? | H28, H29, H30, H36, H38 | `docs/results/direction-norm-analysis.md` |
| `backdoor_direction_layers/` | at which layer does each attack write its direction into the CLS token? | H4 | `docs/runs/2026-08-13-h10-out-of-sample.md` |
| `backdoor_neurons/` | which units carry the backdoor, and does deleting them remove it? | H16 | `docs/hypothesis/H16-*.md` |
| `balanced_panels/` | is each comparison in the ledger made over a balanced set of checkpoints? | H19, H20 | `docs/hypothesis/README.md` |
| `cache_integrity/` | is the cached sweep every headline number rests on internally consistent? | audit | `docs/audit-2026-09-07.md` |
| `detector_ensemble/` | does rank averaging across placements beat betting on 1 placement? | H17, H18, H41 | `docs/results/adaptive-defender-protocol.md` |
| `dropout_kills_direction/` | does dropout at each placement destroy the backdoor direction or spare it? | H3, H20 | `docs/hypothesis/H3-*.md` |
| `gaussian_batch_coupling/` | did the batch-wide std in GaussianNoise bias the split comparison? | audit A2 | `docs/audit-2026-09-07.md` |
| `head_profile/` | does the shape of a per-unit sensitivity profile detect what its mean cannot? | H18, H35 | `docs/hypothesis/H18-*.md`, `docs/hypothesis/H35-*.md` |
| `monte_carlo_passes/` | is the 1% failure estimator noise, fixable by raising k? | H24 | `docs/hypothesis/H24-*.md` |
| `multi_probe/` | can a min-rank rule over several operators recover detection after evasion? | H41 | `docs/hypothesis/H41-*.md` |
| `operator_ranking/` | is the low-poison-rate failure a bad operating point rather than a limit? | H17 | `docs/results/operator-position-ranking.md`, `docs/results/detection-operating-points.md` |
| `psu_vs_confidence/` | is PSU just measuring baseline confidence? | H12 | `docs/hypothesis/H12-*.md` |
| `removal_defences/` | can the backdoor be removed from a trained checkpoint without retraining? | H34, H39 | `docs/hypothesis/H34-*.md`, `docs/hypothesis/H39-*.md` |
| `sam_backdoor_effect/` | does SAM amplify the backdoor on ViT as it does on ResNet18? | H6 | `docs/hypothesis/H6-*.md` |
| `shift_in_latent_space/` | do clean samples under dropout move toward the target class in latent space? | H7 | `docs/hypothesis/H7-*.md` |
| `stealth/` | how visible is each attack's trigger, so the panel can be compared fairly? | panel setup | `docs/results/stealth-metrics.md` |
| `tail_selection/` | can the detection tail be chosen without reading poison labels? | H15 | `docs/hypothesis/H15-*.md` |
| `token_structure/` | does the per-token direction localize the trigger and name the attack family? | H32, H37 | `docs/hypothesis/H32-*.md`, `docs/hypothesis/H37-*.md` |
| `weight_structure/` | is the backdoor a low-rank perturbation in weight space? | H33 | `docs/hypothesis/H33-*.md` |

`backdoor_direction_layers/measure.py` exports `build_paired_loaders`, which 6 other
experiments import as `from experiments.backdoor_direction_layers.measure import
build_paired_loaders`. That is the only cross-experiment import in the tree, and it
is why `experiments/` and `experiments/backdoor_direction_layers/` carry an
`__init__.py`. The `__init__.py` files in `backdoor_neurons/`, `balanced_panels/`,
`dropout_kills_direction/` and `shift_in_latent_space/` are left over from when
those directories were imported as packages and nothing reads them now.
