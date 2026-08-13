---
name: git-workflow
description: Branching model, commit conventions, result traceability, and merge/rebase discipline for a machine learning research repository. Use this skill whenever work involves version control in any way - creating a branch, committing changes, writing a commit message, merging, rebasing, resolving conflicts, freezing code for a submission, tagging, or cleaning up branches. Also use it before starting any multi-file code change in a repo that has a .git directory, and before launching any training run, because which commit a run is launched from determines whether its numbers can ever be traced back. Trigger on phrases like "start a new experiment", "commit this", "merge dev into main", "freeze for submission", "why do these numbers not reproduce", or any request that ends with code being written or a job being launched inside a repo.
---

# Git Workflow for Research

## The two contracts

A software repo protects 1 property. A research repo protects 2, and the second one is the reason this document differs from a normal git workflow.

**Contract 1, takeover.** At any commit, on any branch, a human can pull the repo and continue working without archaeology.

**Contract 2, traceability.** Every number you have ever reported traces to a commit that still exists, and checking out that commit reproduces that number.

Contract 2 is stronger than it sounds. It means a commit that has produced a result is frozen forever. It cannot be rebased, amended, squashed, or garbage collected. In a software repo a feature branch is disposable scaffolding you rewrite freely until it merges. In a research repo, the moment you launch a job from a commit, that commit becomes evidence.

Every rule below exists to protect 1 of those 2 contracts. Nothing here is style for its own sake.

Two audiences read this file: the person maintaining the repo, and any agent committing into it. Both follow the same rules. Where an agent has narrower permissions than a human, section 10 says so explicitly.

Commits should not be completely atomic, but also should not be huge. Big enough to be a meaningful unit, small enough to revert one part without undoing everything.

## 1. Branch model

There is no "release" in a research repo. The thing you ship is a claim backed by numbers, so `main` holds code that produced reported results, and the freeze point is a submission rather than a version.

```
main      A-----------------M1--------------M2    tags: submit/venue-2026, camera/venue-2026
             \             /                /
submission    \      S1---/          S2----/
               \    /               /
dev        B---C---D-------E---F---G
            \     /         \     /
exp          e1--e2          f1--f2
```

| Branch | Lifetime | Created from | Merges into | Contains |
|---|---|---|---|---|
| `main` | permanent | (initial) | nothing | code that produced results you have reported to anyone |
| `dev` | permanent | `main` | `submission/*` | integrated pipeline code that builds, passes tests, and has a LAUNCH verdict |
| `exp/<slug>` | temporary | `dev` | `dev` | one experimental question |
| `repro/<paper>` | temporary | `dev` | `dev` | reproducing a published method before extending it |
| `fix/<slug>` | temporary | `dev` | `dev` | a bug fix in the pipeline |
| `submission/<venue>-<year>` | long-lived, frozen | `dev` | `main` and back into `dev` | the exact code behind one submission, plus rebuttal experiments |
| `hotfix/<slug>` | temporary | `main` | `main` and back into `dev` | a correctness bug discovered in code that has already produced reported numbers |

`submission/*` outlives a normal release branch. It stays alive through review, rebuttal, and camera-ready, because reviewers ask for experiments that must run against the submitted code and not against whatever `dev` has become since.

`hotfix/*` means something different here than in software. It is not "production is down". It is "a number I have already put in a table is wrong", which is more serious and is handled in section 8.

### Naming

Lowercase, hyphen separated. The slug names the question or the outcome, not the activity.

```
exp/dropout-position-ablation
exp/asr-vs-poison-rate
repro/psbd-cifar10
fix/trigger-applied-post-normalization
submission/neurips-2026
```

Bad: `exp/test`, `exp/new-idea`, `exp/experiment-2`, `fix/bug`, `exp/petar-stuff`.

An experiment branch named after a question survives contact with your future self. One named `exp/test2` does not.

## 2. Branch invariants

Violating one is a defect regardless of whether the code runs.

**`main`**
- Every commit is a merge of a `submission/*` or `hotfix/*` branch, or the initial commit.
- Every merge is tagged with an annotated tag naming the venue and milestone.
- Never committed to directly, never rebased, never force-pushed.
- Any checkout of `main` builds, installs, passes the full test suite, and reproduces the results recorded in `results/` at the recorded seeds.

**`dev`**
- Always builds and passes the test suite.
- Never rebased, never force-pushed.
- Receives experiment and fix branches through no-fast-forward merges.
- A pipeline change only reaches `dev` after `research-reviewer` returns LAUNCH. Broken code on `dev` does not just block a colleague, it gets launched onto the cluster by someone who assumed `dev` was safe.

**`exp/*`, `repro/*`, `fix/*`**
- One question per branch. A second question gets a second branch.
- Rebasable onto `dev` **only while no run has been launched from any of its commits.** Once a job has been submitted from a commit on this branch, the branch is frozen against history rewriting for the rest of its life. This is the single most important rule in this document.
- Deleted after merge only if no result artifact references a SHA unique to that branch. If one does, the branch is not deleted, or the SHA is preserved by a tag first. See section 5.

**`submission/*`**
- Accepts rebuttal experiments and correctness fixes. No new research directions.
- Shared and referenced by external reviewers' expectations, therefore never rebased, never deleted after merge. It stays.

## 3. Commit messages

The message explains the change, not the process of producing it.

### Format

```
<type>(<scope>): <imperative summary>

<optional body, only when the "why" is not visible in the diff>

Numbers: <unchanged|changed|not-applicable>
<optional additional footer>
```

Rules:
- Subject in imperative mood, lowercase after the colon, no trailing period, hard cap 72 characters, target 50.
- Scope is the module or directory touched. Omit only when genuinely repo-wide.
- Body wrapped at 72 columns, blank line after the subject, maximum 5 lines.
- No tool attribution, no co-author trailers, no emoji, no file lists, no "as requested".

### The `Numbers:` footer

This is the research-specific addition, and it is mandatory on every commit that touches anything under the pipeline: data loading, trigger injection, model definition, training loop, evaluation, or config defaults.

| Value | Meaning | How it is verified |
|---|---|---|
| `unchanged` | this commit provably does not move any result | logit hash on the reference batch is byte-identical to the parent commit |
| `changed` | this commit is expected to move results | body states which metric, which direction, and roughly how much |
| `not-applicable` | commit touches no pipeline code | docs, plotting, CI, gitignore |

`unknown` is not a permitted value. A pipeline commit whose effect on the numbers you cannot state is a commit you do not understand well enough to merge. If you cannot tell, run the reference config and find out. That takes minutes and saves the week you would otherwise spend bisecting a metric drift 3 months from now.

A `changed` commit that turns out to move a metric in the opposite direction from what the body predicted is a bug, not a surprise. Investigate before merging.

### Types

| Type | Use for | Typical `Numbers:` |
|---|---|---|
| `exp` | new experimental capability: an attack, a defense, an ablation axis | `changed` or `not-applicable` |
| `fix` | corrected defect in the pipeline | `changed`, with expected direction |
| `refactor` | behavior-preserving restructuring | `unchanged`, always |
| `data` | dataset handling, splits, transforms, trigger injection | `changed` or `unchanged` |
| `eval` | metric definitions, evaluation protocol, aggregation | almost always `changed` |
| `perf` | speed or memory, no numerical effect | `unchanged` |
| `test` | tests only | `not-applicable` |
| `viz` | plotting scripts, tables, figures | `not-applicable` |
| `docs` | documentation, including SPEC.md and PLAN.md | `not-applicable` |
| `build` | dependencies, environment, packaging | `unchanged`, and verify it, since a library bump can move numbers |
| `ci` | pipeline configuration | `not-applicable` |
| `chore` | housekeeping touching no source | `not-applicable` |
| `revert` | reverting a previous commit, footer names the SHA | inherit from the reverted commit |

`build` deserves attention. A PyTorch or torchvision upgrade can change interpolation, default initialization, or kernel selection, and it will move your numbers silently. Treat a dependency bump as a pipeline change and verify the hash.

**Good**

```
fix(eval): exclude target-class samples from attack success rate

ASR was computed over the full test set, so samples already labeled as
the target class counted as successes. Expected drop of roughly the
class prior, about 10 points on CIFAR-10.

Numbers: changed
```

```
refactor(data): split trigger injection out of the dataset wrapper

Numbers: unchanged
```

```
exp(defense): add pre-residual dropout variant

Numbers: not-applicable
```

```
build: pin torchvision to 0.19

Resize default interpolation changed upstream, which moved clean
accuracy by 0.4 points. Pinning until the sweep is finished.

Numbers: unchanged
```

**Bad**

```
fix: fixed the ASR calculation and also cleaned up the dataloader and
renamed some variables
```
Three commits pretending to be one, and the rename hides the fix.

```
exp(defense): tune dropout rate

Numbers: unknown
```
A tuning change whose effect on the numbers is unknown is a change you have not measured.

```
WIP
```
Never reaches a shared branch.

### Commit granularity

One logical change per commit. The test: it can be reverted on its own without breaking anything unrelated.

- Never mix a refactor with a behavior change. Split into 2 commits, refactor first, and the refactor's `Numbers: unchanged` is verified before the behavior change lands on top.
- Never mix a formatting sweep with logic.
- Never mix a config default change with a code change. A moved default is a silent experiment.
- Tests ship in the same commit as the code they cover, or the commit immediately after.
- Every commit builds, and every pipeline commit is runnable end to end on the reference config. A commit that only runs when combined with the next one breaks `git bisect`, and bisecting a metric regression is the main reason clean history is worth anything here.

## 4. Rebase or merge

**Rebase to bring changes into your branch. Merge to give your branch to a shared branch.** Then the research override on top of it.

| Situation | Action |
|---|---|
| `dev` moved ahead, no run launched from this branch yet | `git rebase dev` |
| `dev` moved ahead, a run has been launched from this branch | `git merge dev` into the branch, never rebase |
| Experiment finished and reviewed | `git merge --no-ff` into `dev` |
| Freezing for a submission | `git merge --no-ff` into `main`, then tag |
| Submission or hotfix landed on `main` | `git merge --no-ff main` into `dev` |
| Branch has 8 noise commits, 1 idea, and no runs launched | `git merge --squash` into `dev` |
| Branch has 8 noise commits, 1 idea, and runs launched | plain `--no-ff` merge, no squash, the noise stays |

The golden rule of rebasing still holds: never rebase a branch anyone else may have pulled. The research addition is stronger and applies even to a branch nobody else has ever seen.

**Never rebase, squash, or amend a commit that a run was launched from.** Rebasing rewrites SHAs. Every result directory, checkpoint, and log that recorded the old SHA now points at a commit that does not exist, and the numbers become untraceable. They are not wrong, they are unverifiable, which for a paper is the same thing. Squash merging has the identical effect and is therefore also forbidden once runs exist.

Practical consequence: decide early whether a branch is exploratory or load-bearing. Explore freely and rebase freely up to the moment you `sbatch`. After that, merge only.

Never resolve a conflict with a blanket `-X ours` or `-X theirs`. Resolve by intent and run the tests before continuing.

## 5. What is committed and what is not

Getting this wrong is how a research repo becomes a 40 GB object nobody can clone.

**Committed, always**

- All source, configs, and environment specification (`pyproject.toml` and the lockfile)
- `experiments/<slug>/SPEC.md`, `PLAN.md`, `TEST_REPORT.md`
- `results/<run-id>/metrics.json`, the resolved config, the seed, the commit SHA, the environment fingerprint. These are small, they are the traceability record, and losing them costs you the run.
- The reference logit hash used for `Numbers: unchanged` verification
- Plotting scripts and generated `.tex` tables, since a table is regenerable output but you want the diff when a number moves
- `.claude/agents/`, `.claude/skills/`, `.claude/agent-memory/`, `scripts/agent-guards/`

**Never committed**

- Checkpoints, model weights, optimizer state
- Datasets, including poisoned datasets and generated triggers. Commit the generation script and its seed instead.
- `wandb/`, `lightning_logs/`, TensorBoard event files, raw per-step logs
- Figure image files that a committed script regenerates, unless a journal needs the exact bytes
- Anything above roughly 5 MB, ever, without a deliberate decision

**The pointer rule.** For any artifact too large to commit, commit a small file recording where it is, how it was produced, and its hash. A checkpoint on the cluster whose provenance exists only in your head is not reproducible and will be gone when the scratch filesystem is cleaned.

**Preserving evidence SHAs.** Before deleting a merged `exp/*` branch, check whether any `results/` entry names a SHA that only exists on that branch. If so, tag it first:

```bash
git tag -a run/2026-03-14-psbd-ablation <sha> -m "source commit for ablation sweep"
```

Tags are refs, so the commit survives garbage collection and the traceability chain holds. A `run/*` tag is cheap and permanent, which is exactly the trade you want.

## 6. The standard experiment cycle

This maps onto the agent loop. Each stage is a commit, which is what makes a failed stage cheap to reset.

```bash
# 1. Start from current dev
git switch dev
git pull --ff-only
git switch -c exp/dropout-position-ablation

# 2. Spec and plan, committed BEFORE the builder runs, because a builder
#    working in an isolated worktree branches from the default branch and
#    will not see uncommitted files
git add experiments/dropout-position/SPEC.md
git commit -m "docs(exp): extract spec for dropout position ablation

Numbers: not-applicable"

git add experiments/dropout-position/PLAN.md
git commit -m "docs(exp): plan for dropout position ablation

Numbers: not-applicable"

# 3. Implementation, one commit per logical unit
git add src/models/dropout_variants.py
git commit -m "exp(models): add pre-residual dropout variant

Numbers: not-applicable"

# 4. Tests
git add tests/
git commit -m "test(models): pin determinism and shapes for dropout variants

Numbers: not-applicable"

# 5. Stay current. Rebase is still allowed here, no run has been launched
git fetch origin
git rebase origin/dev
python -m pytest

# 6. Review gate. Do not proceed without LAUNCH from research-reviewer.

# 7. Launch. From this commit onward the branch is frozen against rewriting.
git log -1 --format=%H > experiments/dropout-position/LAUNCHED_FROM
sbatch scripts/sweep.sh

# 8. Record results, on the same branch
git add results/2026-03-14-dropout-position/
git commit -m "exp(results): dropout position ablation, 3 seeds

Numbers: not-applicable"

# 9. Integrate
git switch dev
git pull --ff-only
git merge --no-ff exp/dropout-position-ablation
python -m pytest
git push origin dev

# 10. Preserve the launch SHA, then clean up
git tag -a run/2026-03-14-dropout-position $(cat experiments/dropout-position/LAUNCHED_FROM) \
  -m "source commit for dropout position ablation"
git push origin --tags
git branch -d exp/dropout-position-ablation
```

Step 7 writing the SHA to a file is not ceremony. It is the thing you will look for in 4 months when a reviewer asks how a number was produced.

## 7. Freezing for a submission

`main` never receives work directly from `dev`. It receives a submission branch, which exists so that stabilization and rebuttal work happen somewhere other than the branch you develop on.

### The gate

Nothing moves toward `main` until all of these hold.

1. `git log --oneline main..dev` reviewed, every commit accounted for.
2. Full test suite green on a **clean checkout**, not an incremental working tree.
3. `results-auditor` returns clean: every number in the paper traces to a run directory, config, commit SHA, and seed.
4. Every SHA referenced in `results/` is reachable from a branch or tag.
5. Multi-seed results report a seed count and a spread. No single-seed number in a table without being labeled as such.
6. The environment lockfile is committed and a clean environment build reproduces the reference logit hash.
7. No `Numbers: unknown` anywhere in `main..dev`, and no commit missing the footer.
8. Every figure and table in the paper regenerates from committed scripts and committed results.

### Cut, stabilize, freeze

```bash
git switch dev
git pull --ff-only
git switch -c submission/neurips-2026

git commit -m "docs: freeze results for NeurIPS 2026 submission

Numbers: not-applicable"

git switch main
git pull --ff-only
git merge --no-ff submission/neurips-2026 -m "submission: NeurIPS 2026"
git tag -a submit/neurips-2026 -m "NeurIPS 2026 submission"
git push origin main --follow-tags

git switch dev
git merge --no-ff main -m "chore: merge NeurIPS 2026 submission back into dev"
git push origin dev
```

Do not delete `submission/neurips-2026`. Rebuttal experiments branch from it, not from `dev`, because a reviewer's question is about the submitted code and `dev` has moved. Camera-ready gets a second merge and a `camera/neurips-2026` tag.

Tags are annotated, never lightweight, because only annotated tags carry an author and date and are described by `git describe`.

## 8. When a reported number is wrong

This is the research equivalent of a production outage and it is handled with the same urgency.

```bash
git switch main
git pull --ff-only
git switch -c hotfix/asr-target-class-leak

git commit -m "fix(eval): exclude target-class samples from attack success rate

ASR counted samples already labeled as the target class as successes.
All ASR numbers in submit/neurips-2026 are inflated by roughly the
class prior.

Numbers: changed"
```

Then, before merging: re-run every affected configuration, replace the affected entries in `results/`, and regenerate every table and figure that used them. The code fix is the small part. The correction of the record is the point.

```bash
git switch main
git merge --no-ff hotfix/asr-target-class-leak -m "fix: correct ASR computation"
git tag -a fix/asr-target-class-leak -m "corrects ASR in submit/neurips-2026"
git push origin main --follow-tags

git switch dev
git merge --no-ff main -m "chore: merge ASR correction back into dev"
git push origin dev
```

The back-merge is mandatory. A correctness fix that is not merged back into `dev` will be reintroduced as a regression by the next experiment.

Never rewrite history to hide a wrong number. The old commit stays, the tag pointing at it stays, and the fix sits on top with the body explaining what was wrong. That record is what makes the corrected number credible.

## 9. Recovery

Nothing here is unrecoverable as long as the work was committed at least once.

| Problem | Fix |
|---|---|
| Rebase went wrong mid-flight | `git rebase --abort` |
| Merge went wrong mid-flight | `git merge --abort` |
| Bad merge, not yet pushed | `git reset --hard ORIG_HEAD` |
| Bad merge, already pushed | `git revert -m 1 <merge-sha>` |
| Commits seem to have vanished | `git reflog`, then `git reset --hard <sha>` or `git cherry-pick <sha>` |
| Rebased a branch that runs were launched from | `git reflog` to find the pre-rebase SHAs, tag each one as `run/*` immediately, then never do this again |
| A `results/` entry names a SHA that no longer exists | search `git reflog` and every remote. If it is genuinely gone, the run is untraceable and must be re-run. Do not report it. |
| Metric drifted and you do not know which commit did it | `git bisect` with a script that runs the reference config and compares the logit hash. See below. |
| Wrong branch, changes not committed | `git stash`, switch, `git stash pop` |
| Last commit message is wrong, not pushed, no run launched from it | `git commit --amend` |
| Last commit message is wrong, and a run was launched from it | leave it, fix in the next commit |

### Bisecting a metric change

The reason `Numbers: unchanged` is verified rather than asserted is that it makes this work.

```bash
git bisect start
git bisect bad HEAD
git bisect good run/2026-02-01-baseline
git bisect run python scripts/check_reference_hash.py
```

`check_reference_hash.py` runs the fastest real configuration at a fixed seed and exits non-zero if the logit hash differs from the recorded reference. Keep that configuration under 5 minutes, or bisecting a 20-commit range becomes an afternoon.

Enable `git config --global rerere.enabled true` once. It replays conflict resolutions automatically, which matters on long-lived `submission/*` branches.

## 10. Rules for agents

Agents follow everything above, plus these restrictions.

| Agent | May commit | May merge | May tag | May launch |
|---|---|---|---|---|
| `paper-spec-extractor` | no | no | no | no |
| `ml-architect` | no | no | no | no |
| `ml-builder` | yes, on `exp/*`, `repro/*`, `fix/*` only | no | no | no |
| `ml-test-automator` | yes, tests only | no | no | no |
| `research-reviewer` | no | no | no | no |
| `run-monitor` | no | no | no | no |
| `results-auditor` | no | no | no | no |
| `results-writer` | yes, under `paper/` and `scripts/plots/` only | no | no | no |

Absolute prohibitions for every agent, without a human explicitly asking in that turn:

- Never commit to `main`, `dev`, or `submission/*`
- Never merge anything into anything
- Never create or move a tag
- Never rebase, squash, amend, or force-push any branch
- Never `git clean`, `git reset --hard`, or `git restore .` across the tree
- Never delete a branch
- Never `git add -A` when the tree contains changes the agent did not make
- Never `git commit --no-verify`
- Never submit a cluster job

An agent that believes a merge or a rebase is needed says so and stops. Integration is a human decision because it is the point where 2 pieces of verified work become 1 piece of unverified work.

### Worktree isolation

`ml-builder` runs with `isolation: worktree`, which branches from the default branch rather than the parent session HEAD. Two consequences:

1. Commit `SPEC.md` and `PLAN.md` before invoking the builder, or it starts blind.
2. Review the worktree's commits before they reach your working branch. The worktree is a proposal, not a merge.

### Commit scope for agents

An agent commits exactly what it was asked to change. The instruction "fix finding 4, change nothing else, do not refactor surrounding code" is not redundant. Left alone, a builder will tidy adjacent code, and tidying is what turns a 1-line fix into a diff you cannot bisect.

## 11. Takeover contract

The state the repo is in after every unit of work:

- `HEAD` is on a named branch. Never detached.
- No rebase, merge, cherry-pick, or bisect in progress.
- `git status` is clean, or contains only changes explicitly announced as in progress.
- The last commit on the current branch builds and runs the reference config.
- Every branch that exists has an obvious purpose from its name.
- Nothing is stashed silently. A stash is invisible to someone taking over.
- Every SHA referenced by anything in `results/` is reachable from a branch or a tag.
- No running or queued cluster job was launched from a commit that is not pushed.

That last one is the research-specific addition and it matters. A job launched from an unpushed commit produces results nobody else can trace, and if your local clone dies before you push, the results die with it.

### Useful defaults

```bash
git config --global pull.ff only
git config --global rebase.autoStash true
git config --global rerere.enabled true
git config --global fetch.prune true
git config --global tag.sort -v:refname
```

## 12. Scaling down

The full model is right for a project heading to a venue. For a quick exploratory study, drop parts in this order:

1. Drop `submission/*`. Merge `dev` into `main` directly, still `--no-ff`, still tagged, still behind the section 7 gate.
2. Drop `hotfix/*`. Use `fix/*` from `main` and merge back into both.
3. Drop `dev`. Experiment branches merge into `main`.

Do not drop, at any size:

- 1 question per branch
- the commit message format and the `Numbers:` footer
- the rule that a commit a run was launched from is never rewritten
- `run/*` tags on launch SHAs
- committing `results/` metadata
- the takeover contract

Those 6 cost nothing and they are the difference between a repo you can defend in review and a directory of scripts.

## Quick reference

```bash
# start an experiment
git switch dev && git pull --ff-only && git switch -c exp/<slug>

# stay current, before any run is launched
git fetch origin && git rebase origin/dev

# stay current, after a run has been launched
git fetch origin && git merge origin/dev

# record the launch SHA before submitting
git log -1 --format=%H > experiments/<slug>/LAUNCHED_FROM

# preserve the launch SHA permanently
git tag -a run/<date>-<slug> $(cat experiments/<slug>/LAUNCHED_FROM) -m "source commit"

# integrate
git switch dev && git pull --ff-only && git merge --no-ff exp/<slug>

# freeze for a submission
git switch -c submission/<venue>-<year> && git switch main && \
  git merge --no-ff submission/<venue>-<year> && git tag -a submit/<venue>-<year>

# back-merge, always
git switch dev && git merge --no-ff main

# find which commit moved a metric
git bisect start && git bisect bad HEAD && git bisect good <tag> && \
  git bisect run python scripts/check_reference_hash.py

# inspect before touching anything
git status --short --branch && git log --oneline --graph -15
```

## 13. This repo, concretely

The model above assumes conventions this repo does not have. These are the real ones.

### Where traceability actually lives

`checkpoints/`, `results/`, and `logs/` are all gitignored, and `checkpoints/` alone
is 336 GB. So the chain from a reported number back to a commit runs through three
committed artifacts, not through the run directories:

| artifact | tracked | holds |
|---|---|---|
| `checkpoints/<folder>/args.json` | no, but per-checkpoint | dataset, attack, label_mode, target_label, poison_rate, architecture, optimizer, rho, epochs, seed, git_commit |
| `results/<folder>/psbd_metrics.json` | **yes**, un-ignored explicitly | every detection number, per placement and rate |
| `docs/runs/<date>-<slug>.md` | yes | launch SHA, the grid, the job IDs |

`.gitignore` un-ignores `results/**/psbd_metrics.json` through the
`results/**` + `!results/**/` + `!results/**/psbd_metrics.json` idiom, because git
cannot re-include a file whose parent directory is excluded. Raw `.pt` tensors stay
out. Any future `<defense>_metrics.json` should be un-ignored the same way.

Note the `runs/` pattern must stay anchored as `/runs/`. Unanchored it also swallows
`docs/runs/`, which is where the launch SHAs live.

### There is no experiments/<slug>/ tree

Section 6 writes the launch SHA to `experiments/<slug>/LAUNCHED_FROM`. Here it goes
in `docs/runs/<date>-<slug>.md` along with the grid and the job IDs, committed before
`qsub`. Same purpose, one file instead of two.

### Provenance gap in the existing checkpoints

Every checkpoint currently on disk has `seed: null`, `git_commit: null`, and null
timestamps in its `args.json`. They were backfilled from folder names by
`scratch/normalize_checkpoints.py`, not recorded at training time, so `label_mode`
and `poison_rate` there are *inferred* rather than observed. This cannot be repaired
retroactively. State it in any writeup that leans on those fields, and check that
newly trained checkpoints get real values.

### Numbers: on an eval-only change

Most work here re-evaluates existing checkpoints rather than training. A commit that
changes how a checkpoint is *scored* is still `Numbers: changed` even though no model
moved, because every reported number moves. `Numbers: unchanged` on a scoring change
is the easiest way to lose a week later.

### The hypothesis ledger is part of the record

`docs/hypothesis/` carries one file per claim, with its prediction, the evidence, and
a verdict. A commit that lands evidence updates the verdict in the same commit. A
refuted hypothesis keeps its file; deleting it destroys the record of what was ruled
out, which is most of what a ledger is for.

Preliminary verdicts get labelled as preliminary. Two hypotheses here were written up
as SUPPORTED from a 64-sample smoke run of a single checkpoint, and the full grid cut
one effect by 4x and refuted the other outright.

### Never edit a script that in-flight jobs read

PBS jobs here run `python psbd_dropout_sweep.py` against the **working tree**, not
against a snapshot of the commit they were submitted from. So editing an entrypoint
while jobs are queued changes the code those jobs will execute, mid-sweep.

This has already cost 50 jobs once: a half-applied edit left `write_run_provenance`
referencing an argparse field that had not been added yet, and every job that started
after the edit died with `AttributeError` while the ones that started before it
succeeded. The result directory was left in a state where a job's success depended on
when it happened to be scheduled.

Rules:

- Before editing any file under a running sweep's import graph, check `qstat`. If jobs
  are queued or running, either wait, or copy the tree and point the remaining jobs at
  the copy.
- Verify an edit actually applied before moving on. String-replacement patches against
  a `ruff format`-ed file fail silently when the target text has been re-wrapped, which
  is exactly how the half-applied state above arose. Run `--help` or import the module.
- A sweep whose jobs did not all run the same code is not one experiment. Identify the
  failed jobs by their log signature, resubmit them, and confirm they used the fixed
  code before analysing anything.
