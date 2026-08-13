# Git Workflow

This document enforces 1 contract:

**At any commit, on any branch, a person can pull the repository and continue work with no
archaeology.**

Every rule below exists to protect that property. Nothing here is style for its own sake.

2 audiences read this file: the person who maintains the repository, and any agent that commits
into it. Both follow the same rules.

**Size a commit between 2 limits.** A commit must be large enough to be a meaningful change, and
small enough that you can revert 1 part without the loss of everything else.

## 1. The branch model

```
main      A-----------------M1--------------M2       tags: v1.0.0, v1.1.0
             \             /                /
release       \      R1---/          R2----/
               \    /               /
dev        B---C---D-------E---F---G
            \     /         \     /
feature      f1--f2          g1--g2
```

| Branch | Lifetime | Created from | Merges into | Contains |
|---|---|---|---|---|
| `main` | Permanent | The initial commit | Nothing | Released, tagged code that is known to be good |
| `dev` | Permanent | `main` | `release/*` | Integrated work that builds and passes the tests |
| `feature/<slug>` | Temporary | `dev` | `dev` | 1 feature, 1 concern |
| `fix/<slug>` | Temporary | `dev` | `dev` | A bug fix that is not urgent |
| `release/<version>` | Temporary | `dev` | `main`, then back into `dev` | A version bump, and fixes that block the release |
| `hotfix/<version>` | Temporary | `main` | `main`, then back into `dev` | A fix for a broken production build that cannot wait for `dev` |

### Naming a branch

Use lowercase and hyphens. Do not use a personal name. Do not use a ticket number alone.

Good:

```
feature/card-shuffle-animation
feature/clone
fix/deck-underflow-on-reshuffle
release/1.4.0
hotfix/1.4.1
```

Bad: `feature/petar-stuff`, `fix/bug`, `feature/JIRA-1234`, `new-feature-2`.

**The slug names the outcome, not the activity.** `feature/response-cache` is better than
`feature/add-caching`, because the existence of the branch already implies "add".

## 2. Branch invariants

These invariants are what make a takeover possible. To violate 1 of them is a defect, whether or
not the code works.

### `main`

- Every commit is a merge of a `release/*` branch or a `hotfix/*` branch, or it is the initial
  commit. Nothing else.
- An annotated semver tag marks every merge commit.
- Never commit to `main` directly. Never rebase it. Never force push it.
- Any checkout of `main` builds, installs, and passes the full test suite.

### `dev`

- `dev` always builds and passes the tests. If a merge breaks it, the fix or the revert lands
  immediately. It does not land "later".
- Never rebase `dev` and never force push it. Other branches are cut from it, so to rewrite it
  corrupts everything downstream.
- `dev` receives feature branches and fix branches through merges with `--no-ff`. Each feature then
  stays visible as a unit.

### `feature/*` and `fix/*`

- 1 concern for each branch. If a second concern appears in the middle of a branch, give it its own
  branch.
- Rebase onto `dev` before you integrate. Never rebase after a merge.
- You can force push with `--force-with-lease` while the branch belongs to 1 person.
- Delete the branch after the merge, on this machine and on the remote.

### `release/*`

- A release branch accepts a version bump and fixes for defects that release testing found. It
  accepts nothing else. A request for a new feature during a release cycle goes to `dev` and ships
  in the next version.
- Release notes live on GitHub, written on the release that you create from the tag.
- A release branch is shared. Therefore never rebase it.

## 3. Commit messages

The message explains the change. It does not explain the process that produced the change.

A person who reads `git log --oneline` must be able to reconstruct what happened in the repository
without a single diff.

### Format

```
<type>(<scope>): <imperative summary>

<optional body, only when the diff does not show the reason>

<optional footer>
```

5 rules:

1. Write the subject in the imperative mood. Write "add", not "added" or "adds".
2. Use lowercase after the colon. Do not end with a period.
3. Cap the subject at 72 characters. Target 50.
4. The scope is the module, package or directory that you touched. Omit the scope only when the
   change is genuinely repository wide.
5. Wrap the body at 72 columns, with a blank line after the subject. Most commits have no body.

WARNING: DO NOT ADD TOOL ATTRIBUTION, A CO-AUTHOR TRAILER, AN EMOJI, A FILE LIST, OR THE PHRASE
"AS REQUESTED".

### Types

| Type | Use for |
|---|---|
| `feat` | New behaviour that a user can see |
| `fix` | A defect that you corrected |
| `refactor` | Restructuring that preserves behaviour |
| `perf` | An improvement in speed or memory |
| `test` | Tests only |
| `docs` | Documentation only |
| `build` | The build system, dependencies, packaging |
| `ci` | Pipeline configuration |
| `chore` | Housekeeping that touches no source, such as gitignore or a licence |
| `revert` | A revert of an earlier commit. The footer names the SHA |

### When to write a body

Write a body only if a reviewer 6 months from now would ask "why like this?"

4 good reasons: a constraint that is not obvious, an alternative that you rejected, the root cause
behind a fix, or a performance number.

**Good examples:**

```
fix(engine): reshuffle the discard pile before a draw when the deck is empty

The draw handler assumed a deck that is not empty, because the printed
rules guarantee one. A custom card count breaks that assumption.
```

```
perf(ui): stream the card images instead of a buffered sprite sheet

Peak memory dropped from 480 MB to 90 MB, which keeps the game usable on
a device with 4 GB.
```

```
feat(bots): score every legal action by the position it leads to
```

```
refactor(engine): replace the turn queue with an index into the player list
```

**Bad examples:**

```
Updated gameState.ts to add a new function called reshuffleDiscard which
takes the discard pile as an argument and returns a shuffled deck, and
also modified drawCard to call it when the deck is empty.
```

This narrates the diff. The diff already says all of it.

```
fix: bug fix
```

This says nothing.

```
feat(game): add reshuffle, fix lint errors, bump deps, rename Player.hp
```

These are 4 commits that pretend to be 1.

```
WIP
```

This must never reach a shared branch.

### Commit granularity

1 logical change for each commit.

The test for correct scope: **you can revert the commit on its own, and nothing unrelated breaks.**

4 rules follow:

1. Never mix a refactor with a change in behaviour. Split them into 2 commits, and put the refactor
   first.
2. Never mix a formatting sweep with logic. A reformat commit that also fixes a bug hides the bug.
3. Ship the tests in the same commit as the code that they cover, or in the commit immediately
   after.
4. Every commit builds. A commit that compiles only when combined with the next commit breaks
   `git bisect`, and `git bisect` is the main reason that a clean history is worth anything.

## 4. Rebase or merge

1 rule covers 95% of cases:

**Rebase to bring changes into your branch. Merge to give your branch to a shared branch.**

| Situation | Action |
|---|---|
| `dev` moved ahead while you worked | Run `git rebase dev` on your feature branch |
| The feature is finished and reviewed | Run `git merge --no-ff` into `dev` |
| The release is ready | Run `git merge --no-ff` into `main`, then tag |
| A release or a hotfix landed on `main` | Run `git merge --no-ff main` into `dev` |
| The feature branch holds 8 noise commits and 1 idea | Run `git merge --squash` into `dev`, and write 1 clean message |

WARNING: NEVER REBASE A BRANCH THAT ANOTHER PERSON COULD HAVE PULLED. `main`, `dev` AND `release/*`
ARE SHARED BY DEFINITION, SO NEVER REBASE THEM. A PRIVATE FEATURE BRANCH IS SAFE UNTIL ANOTHER
PERSON CHECKS IT OUT.

Why rebase first and then merge with `--no-ff`, instead of 1 or the other:

- A rebase puts the commits of the feature on top of the current `dev`. Each commit was therefore
  tested against the code that it will really live with.
- The `--no-ff` merge keeps 1 commit that marks the boundary of the feature. So
  `git log --first-parent dev` reads as a list of features, not as a list of individual edits.
- The result is linear inside a feature and structured between features. That is the format that
  survives a takeover.

CAUTION: NEVER RESOLVE A CONFLICT WITH A BLANKET `-X ours` OR `-X theirs`. THAT SILENTLY DISCARDS
THE WORK OF ANOTHER PERSON. RESOLVE BY INTENT, AND RUN THE TESTS BEFORE YOU CONTINUE.

## 5. Procedure: the standard feature cycle

```bash
# 1. Start from the current dev
git switch dev
git pull --ff-only
git switch -c feature/response-cache

# 2. Work. Commit in logical units
git add ui/cache.ts ui/cache.test.ts
git commit -m "feat(ui): add a response cache with a TTL"

# 3. Stay current while dev moves. Repeat as needed
git fetch origin
git rebase origin/dev
# Run the tests after every rebase, not only at the end
npm test

# 4. Publish
git push --force-with-lease -u origin feature/response-cache

# 5. Integrate, once the tests pass on the rebased branch
git switch dev
git pull --ff-only
git merge --no-ff feature/response-cache
npm test        # dev must be green before you push it
git push origin dev

# 6. Clean up
git branch -d feature/response-cache
git push origin --delete feature/response-cache
```

Step 5 is not bureaucracy. To run the suite on `dev` after the merge is the only thing that catches
an interaction bug between 2 features that each passed on their own.

## 6. Procedure: promote `dev` to `main`

`main` never receives work directly from `dev`. It receives a release branch. That branch exists so
that stabilization happens somewhere other than the branch that everybody develops on.

### The gate

Nothing moves toward `main` until all 7 of these hold. If 1 fails, fix it on `dev` or on the
release branch, and start the checklist again.

1. You reviewed `git log --oneline main..dev`, and every commit is accounted for and intentional.
2. The full test suite is green on a **clean checkout**, not on an incremental working tree.
3. The build and the lint are green.
4. This cycle introduced no debug logging, no commented out code, and no `TODO` or `FIXME` without
   a reference to an issue.
5. The lockfiles are committed and agree with the manifest.
6. You ran a manual smoke test of the primary user path.
7. No unmerged feature branch is expected to be part of this version.

### Cut, stabilize, release

```bash
# Cut the release branch from dev
git switch dev
git pull --ff-only
git switch -c release/1.4.0

# The version bump lives here, not on dev
git commit -m "build: bump the version to 1.4.0"

# From here on, only fixes that block the release
git commit -m "fix(ui): evict expired entries on a read, not on a timer"

# Merge into main and tag
git switch main
git pull --ff-only
git merge --no-ff release/1.4.0 -m "release: 1.4.0"
git tag -a v1.4.0 -m "1.4.0"
git push origin main --follow-tags

# Bring the stabilization work back into dev
git switch dev
git merge --no-ff main -m "chore: merge 1.4.0 back into dev"
git push origin dev

git branch -d release/1.4.0
```

WARNING: THE BACK MERGE IN THE LAST STEP IS THE STEP THAT PEOPLE FORGET. IF YOU SKIP IT, THE FIXES
THAT YOU MADE DURING STABILIZATION DISAPPEAR AT THE NEXT RELEASE, WHEN `dev` OVERWRITES THEM.

Use an annotated tag with `-a`. Never use a lightweight tag. Only an annotated tag carries an author
and a date, and only an annotated tag is described by `git describe`.

## 7. Procedure: a hotfix

Use this path for a defect in production that cannot wait for the current `dev` cycle.

```bash
git switch main
git pull --ff-only
git switch -c hotfix/1.4.1

git commit -m "fix(network): clear the token cache on logout"

git switch main
git merge --no-ff hotfix/1.4.1 -m "release: 1.4.1"
git tag -a v1.4.1 -m "1.4.1"
git push origin main --follow-tags

git switch dev
git merge --no-ff main -m "chore: merge 1.4.1 back into dev"
git push origin dev

git branch -d hotfix/1.4.1
```

WARNING: A HOTFIX THAT YOU DO NOT MERGE BACK INTO `dev` COMES BACK AS A REGRESSION AT THE NEXT
RELEASE. THIS STEP IS MANDATORY.

## 8. Recovery

Nothing here is unrecoverable, as long as the work was committed at least once. Commit early for
exactly this reason.

| Problem | Fix |
|---|---|
| A rebase went wrong in the middle | `git rebase --abort` |
| A merge went wrong in the middle | `git merge --abort` |
| A bad merge that you have not pushed | `git reset --hard ORIG_HEAD` |
| A bad merge that you already pushed | `git revert -m 1 <merge-sha>` |
| Commits seem to have vanished | `git reflog`, then `git reset --hard <sha>` or `git cherry-pick <sha>` |
| The wrong branch, with changes not committed | `git stash`, switch, then `git stash pop` |
| The last commit message is wrong, not pushed | `git commit --amend` |
| The last commit message is wrong, already pushed to a shared branch | Leave it. Fix it in the next commit |

Run `git config --global rerere.enabled true` once. Git then records how you resolved a conflict
and replays that resolution on the next rebase. This saves real time on a branch that lives a long
time.

## 9. The takeover contract

Leave the repository in this state after every unit of work, so that a person can take over with no
context:

1. `HEAD` is on a named branch. `HEAD` is never detached.
2. No rebase, merge, cherry-pick or bisect is in progress.
3. `git status` is clean, or it holds only changes that you announced as work in progress.
4. The last commit on the current branch builds.
5. Every branch that exists has an obvious purpose from its name, and a first commit that explains
   it.
6. Nothing is stashed in silence. A stash is invisible to the person who takes over.

### Never do these without explicit approval

| Command | Why not |
|---|---|
| `git push --force` on `main`, `dev` or `release/*` | It destroys the history of other people |
| `git reset --hard` with changes in the tree that you did not make | It deletes uncommitted work permanently |
| `git clean -fd` | It deletes untracked files, and that includes local config and scratch work |
| `git restore .` or `git checkout .` across the whole tree | The same loss, in silence |
| `git commit --no-verify` | It bypasses the hooks that guard the invariants above |
| `git rebase` on a shared branch | It rewrites history that other people depend on |
| `git add -A` when the tree holds changes that you did not make | It sweeps unrelated work into your commit |
| `git commit --amend` on a commit that you pushed | It rewrites published history |
| Deleting a branch that you did not create | It could be the parked work of another person |

`--force-with-lease` is the only acceptable force, and only on a private branch. It refuses to
overwrite a commit that you have not seen.

### Useful defaults

```bash
git config --global pull.ff only          # never create a surprise merge commit on a pull
git config --global rebase.autoStash true # rebase with no manual stash
git config --global rerere.enabled true   # remember how you resolved a conflict
git config --global fetch.prune true      # drop remote tracking branches that somebody deleted
```

## 10. Scaling the model down

The full model is right for anything with users. For a solo project or an experiment, drop the
parts that do not earn their keep, in this order:

1. **Drop `release/*`.** Merge `dev` into `main` directly. Still use `--no-ff`, still tag, and still
   apply the gate in section 6.
2. **Drop `hotfix/*`.** Use `fix/*` from `main`, and merge back into both branches.
3. **Drop `dev`.** Feature branches then merge into `main`. This is the smallest structure that
   still lets another person take over.

Do not drop these 4 at any size, because they cost nothing and they are the parts that make a
history worth having:

1. 1 concern for each branch.
2. The commit message format.
3. The rule that every commit builds.
4. The takeover contract.

## Quick reference

```bash
# start work
git switch dev && git pull --ff-only && git switch -c feature/<slug>

# stay current
git fetch origin && git rebase origin/dev

# integrate
git switch dev && git pull --ff-only && git merge --no-ff feature/<slug>

# release
git switch -c release/<version> && ... && git switch main && \
  git merge --no-ff release/<version> && git tag -a v<version> -m "<version>"

# back merge, always
git switch dev && git merge --no-ff main

# inspect before you touch anything
git status --short --branch && git log --oneline --graph -15
```
