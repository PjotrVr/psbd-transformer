---
name: feedback-push-when-chunk-done
description: Commit locally as you go, but push the working branch only when a chunk of work is finished and green
metadata:
  type: feedback
---

Commit locally as work proceeds, and push the working branch (`git push origin <branch>`) when a
CHUNK is finished — the fix plus its checker and tests green — rather than after every single
commit. This revises the earlier "push after every commit" rule, which the user changed on
2026-09-17 while the game's legacy-record compatibility work was in flight.

**Why:** the user still reviews from another machine, but a stream of pushes mid-chunk is noise and
can publish a half-finished state into a tree several agents share.

**How to apply:** in the game clone (`solver-integration`) and in solver worktrees alike, batch the
pushes at chunk boundaries. The final landing push still happens as specified by the task. See
[[feedback-no-banner-comments]].
