---
name: leak-audit-test-gaps
description: How to run the ek_solver blindness suite, the audit protocol quirks that have burned me, and which coverage gaps are currently open vs closed
metadata:
  type: project
---

The blindness suite and how to run it. Always run it in the worktree under audit, never in the main
checkout at `/lustre/home/pstika/projects/ek_solver` (PBS jobs read that tree):

    cd <worktree>/engine && RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 cargo test -j 16 -p ek-obs --test no_leak
    ... same env, -p ek-obs --test seats_block, -p ek-env --test targets, -p ek-env --test trio
    <worktree>/.venv/bin/python -m pytest tests/test_leak.py -q     (check `pgrep -fa pytest` first)

All of them build against the existing `engine/target` and finish in seconds. Never `uv run`.

**Audit the working tree, not the committed diff.** Feature worktrees on this project routinely
carry uncommitted work on top of the branch tip. On the `trio` audit of 2026-09-13, `git diff
BASE..HEAD` showed a complete-looking feature and `git status` showed 6 more modified files plus an
untracked test, containing a whole second feature (`MixedGames`) and a rewritten `_outcomes`. Run
`git status --short` and `git diff` (unstaged) before reading anything. The initial gitStatus in the
system prompt describes the MAIN checkout, not the worktree you were pointed at.

**The complement rule bites on appended blocks.** `ek-env/tests/targets.rs::targets_differ_across_hidden_halves`
is the test that proves the targets array is not accidentally empty or view-derived. It compares only
the blocks its helpers slice. Every time a label block is APPENDED to the targets row, that test
silently stops covering the new columns while continuing to pass. Check which offsets its helpers
actually read before crediting it. Value tests that assert a block's TOTAL (sum of counts equals hand
size, sum of slots equals kittens dealt) do not substitute: they pass on a block that put the right
total in the wrong or view-derivable slot.

**Why:** a leak test only proves something if the two worlds it compares genuinely differ, and an
empty or view-derived targets array satisfies every equality assertion in the suite.

Gap status as of 2026-09-13 end of day, branch `trio` at 09f0d68:
- CLOSED: `encode_action` has `the_action_encoding_is_blind_to_the_hidden_half`; the forbidden-field
  grep covers 9 files including `ledger_v2.rs` and `seats.rs`; `seats_block.rs` has
  `the_block_is_blind_to_the_deck_cut` and checks all 7 scalars of a slot at 2 seats.
- OPEN, the one blocking item: the appended per-seat-hand (targets offset 111, width 264) and
  bomb-count (offset 375, width 66) label blocks have no differ-across-hidden-halves assertion.
  `targets_differ_across_hidden_halves` compares only `bomb()` and `opponent_hand()` and runs only on
  2 seat decks. Until it is extended, `--seat-beliefs` must stay off. Both trio arms already leave it
  off, which is why they were safe to run anyway.

**When a teammate agent reports "your findings are fixed", check which ones.** On the trio audit the
3 items fixed were my non-blocking observations and the blocking violation was untouched, while the
message numbered them as if they were my findings list. Re-read the actual test bodies rather than
trusting the summary, and do not flip the verdict without it.

Related: [[leak-audit-surface]]
