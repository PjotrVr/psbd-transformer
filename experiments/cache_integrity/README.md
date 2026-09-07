# PSBD cache integrity audit

## Question

Every headline PSBD number is computed by `psbd_analyze.py` from tensors written
once by `psbd_dropout_sweep.py` and never recomputed. If a cached tensor is
truncated, misaligned with its manifest, or records a probe that silently never
fired, stage 2 still produces a complete and plausible-looking result. Nothing in
the pipeline had ever checked that the 605927 cached files under
`results/<folder>/psbd/` are internally consistent, so this answers whether the
cache the paper rests on is trustworthy.

## Running it

Repo root has to be importable, matching the rest of `experiments/`.

    PYTHONPATH=. python experiments/cache_integrity/check.py --sample 20
    PYTHONPATH=. python experiments/cache_integrity/check.py --full --workers 128

`--sample N` draws N checkpoints seeded through `lightning.seed_everything`, so
the same `--seed` returns the same N and any failure it reports is reproducible.
`--full` audits every checkpoint carrying a `psbd/` subtree. Exactly 1 of the 2
is required, because the full pass reads about 73 GB.

The report goes to `results/cache_integrity.json` by default: per-check pass and
fail counts, the complete failure list grouped by check, and a per-checkpoint
record of the resolved operator, stored k, and rate coverage for every position
config. Stdout carries the same summary with failing paths capped at
`--max-shown` (default 50).

The audit is read only. It opens cache files with `torch.load(...,
weights_only=True)` and writes nothing except the report, which lives at the top
of `results/` and never inside a `psbd/` subtree. Work is distributed 1
checkpoint per task over a process pool with `OMP_NUM_THREADS=1` per worker,
since each worker only does elementwise arithmetic on small `(k, N)` tensors and
128 of them each spawning a thread pool would only contend.

## Checks

Scope is the unit the check is evaluated on, which is also the denominator in the
report's `attempted` column.

| id | scope | invariant |
| --- | --- | --- |
| 00 | checkpoint | the audit of this subtree ran to completion |
| 01 | checkpoint | `split_manifest.json` parses |
| 02 | checkpoint | `heldout_indices` and `analysis_clean_indices` are disjoint and together cover `n_total` |
| 03 | checkpoint | `n_heldout` equals `len(heldout_indices)` |
| 04 | checkpoint | `analysis_backdoor_indices` is a subset of `analysis_clean_indices` |
| 05 | checkpoint | `label_mode` matches what `attacks.build_attack` derives for `probe_attack` |
| 06 | baseline file | loads |
| 07 | baseline file | `probs`, `labels`, `loader_labels` row counts agree |
| 08 | baseline file | row count matches the manifest for that split |
| 09 | baseline file | `probs` is finite and every row sums to 1 within 1e-3 |
| 10 | baseline file | `labels` equals `probs.argmax(dim=1)` bitwise |
| 11 | baseline file | `probs` column count equals the dataset's `num_classes` |
| 12 | rate file | loads, with both tensors present |
| 13 | rate file | `per_pass_probs` and `per_pass_argmax` have the same shape |
| 14 | rate file | row count (dim 1) matches that split's baseline |
| 15 | rate file | `per_pass_probs` is finite and lies in [0, 1] |
| 16 | rate file | `per_pass_argmax` lies in [0, `num_classes`) |
| 17 | position config | a single k (dim 0) across every rate and split |
| 18 | rate file | a stochastic operator's k passes are not all identical |

Check 18 is the one that earns its keep. PSU is an expectation over the k
passes, so k bit-identical rows make it identically 0, and stage 2 reports that
as a placement with no effect rather than as a probe that never fired. Identical
passes are correct only for the operators in
`defences.perturbations.DETERMINISTIC_PERTURBATIONS`, which the check exempts.

Check 04 must hold because the backdoor split is built as the attack-eligible
subset of the analysis pool (`AttackSuccessSet` over the analysis subset in
`defences.checkpoint_eval.build_psbd_loaders_from_checkpoint`), so a backdoor
index outside the pool would mean the 2 splits were never paired.

Check 17 is a report as much as a check: `k_value_counts` in the summary is the k
actually stored, which is not always the k the folder name advertises.

Rate coverage (check 17's companion) reuses
`defences.psbd_metrics.complete_rates` for the definition of complete, so what
this audit calls complete is exactly what `psbd_analyze.py` will agree to load.
Anything short of all 3 splits is listed per config under `partial_rates`.

## Resolving which operator wrote a position config

Check 18 needs to know whether a folder's operator is deterministic. The audit
resolves it in 3 steps, and records which step answered in `resolved_from`:

1. `run_<config>.json`, the provenance sidecar `psbd_dropout_sweep.py` writes,
   which records `perturbation` verbatim.
2. The folder name, after stripping the `_k<n>` and `_pmodel<rate>` suffixes
   `cache_config_name` appends after the operator.
3. Bare name means dropout, which is the convention that keeps pre-perturbation
   caches addressable.

## Result, full run over 798 checkpoints

Commit `30c5936`, 128 workers, 42 seconds.

    checkpoints checked     798
    files checked           605927
    position configs        20408
    rate coverage           200897 complete, 30 partial
    stored k values         {3: 600443, 20: 2292}

0 failures on all 18 checks. Every denominator was full: 798 manifests, 2394
baseline files, 602735 rate files, 20408 position configs, and 582051 rate files
eligible for the stochastic-passes check.

The 30 partial rates are each a single rate at the end of a position config,
missing only the backdoor split, or both the backdoor and clean splits. That is
the signature of a sweep interrupted mid-rate rather than of damage, and
`complete_rates` already excludes them, so no analysis has ever read them.

20684 rate files did have k identical passes. All of them sit under a
`gain_scale` or `scale_up` folder, where that is the correct and expected
behaviour. No stochastic operator produced identical passes anywhere in the tree.

Operator coverage resolved as: token_mask 4683 configs, gaussian 4675,
channel_mask 4665, dropout 3713, droppath 1341, gain_scale 707, scale_up 177,
head_mask 447. The 4649 `*_gaussian_batchstd` folders carry no provenance
sidecar and were resolved from their folder name to gaussian, correctly
classified as stochastic.

## Verification

The checker was validated against a synthetic tree carrying 1 deliberate defect
per check, built in a scratch directory rather than the repo because
`--results-dir` and `--checkpoints-dir` point anywhere. All 18 content checks
fired on it, check 00 correctly stayed silent because nothing crashed, and a
deterministic `scale_up` folder with identical passes was correctly not flagged.
A separately written script then re-read 120 baselines and 1356 rate files drawn
from 40 random checkpoints and agreed with the audit: 0 bad baselines, 66
identical-pass files all under deterministic operators, 0 under stochastic ones.

Denominators exclude what they cannot cover. A file that fails its load check
counts only toward that check, never toward the checks below it, so a passed
count is always over files that were genuinely readable.
