---
name: feedback-shared-game-checkout
description: In the game clone several agents share one working tree; commit your own paths early because another agent's stash or checkout will discard your uncommitted edits
metadata:
  type: feedback
---

In `/lustre/home/pstika/projects/exploding_kittens` (branch `solver-integration`) there are no
worktrees, only the one clone, and several agents edit it at once. Commit your own paths as soon as
a change stands on its own, and keep every multi-file edit in a replayable script under the
scratchpad until it is committed.

**Why:** on 2026-09-17 another agent's stash handling ran `git checkout -- <file>` and a `git reset`
in that tree and silently discarded a whole batch of my uncommitted edits (scripts, settings,
registry) while leaving untracked new files alone. Nothing committed was lost. Re-doing the work
from memory cost about an hour; re-running a saved script would have cost a minute.

**How to apply:** write edits as `python3 - <<PY` / heredoc scripts saved to the scratchpad, run
them, `git add <my paths>` and commit locally as you go. Push only when the chunk is finished and
green (the user's rule, 2026-09-17), not after every commit. Re-check `git status` before each
commit: a file you edited an hour ago may have been reverted under you.
Related: [[project-model-exports-and-the-pin]]
