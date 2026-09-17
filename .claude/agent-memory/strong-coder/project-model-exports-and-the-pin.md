---
name: project-model-exports-and-the-pin
description: Exporting game models on the cluster — BUILT_FROM's solver_commit must equal SOLVER_REV, exports take seconds, per-model position fixtures take minutes to half an hour
metadata:
  type: project
---

Exporting a checkpoint into the game (`./solver/promote.sh`, which runs
`solver/dmc/export.py --for-game`) stamps `BUILT_FROM: solver_commit` from the git HEAD of the
checkout the exporter RAN FROM, and `ui/solverBotFreshness.test.ts` fails unless that equals the
game's `SOLVER_REV`. So export from a checkout sitting exactly at the pinned rev — normally the
game's own `solver/ek_solver` clone, with `SOLVER_PYTHON` pointing at the main checkout's venv,
since the clone has no venv of its own. A git worktree of the clone at the pinned rev works when
the clone has moved on.

**Why:** the pin moved twice under me in one afternoon (a993e20 then e4a7ca6, another agent
bumping it), and each bump invalidated every model exported before it. Re-exporting is cheap and
re-dumping fixtures is not.

**How to apply:** check `cat SOLVER_REV` immediately before exporting, and re-check before
committing. Costs measured 2026-09-17 on the login node: an export with its 2000 node parity gate
is ~10 seconds per checkpoint; an `ek-play --dump-positions 500` fixture is ~2.5 minutes on
duel-deck and ~25 minutes on `boardgamearena-2p`, at roughly 10 cores whatever `--threads` says.
Plan the slow half first and do the code while it runs. `solver/fixtures.sh` must name
`ORT_LIB_LOCATION` absolutely or the ek-play link fails with `undefined symbol: OrtGetApiBase`.
Related: [[feedback-shared-game-checkout]]
