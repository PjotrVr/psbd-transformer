---
name: project-multi-agent-concurrent-repo
description: PSBD-ViT repo often has multiple agents/sessions working and testing concurrently on the same checkout
metadata:
  type: project
---

The PSBD-ViT repo (`/lustre/home/pstika/projects/PSBD-ViT`) is regularly worked on by more than
one agent session at once, on the same checkout. During a 2026-09-23 review I observed 3+
concurrent full `pytest tests` invocations already running before I started my own, and a file
I was actively reviewing (`scripts/prose_audit.py`) changed on disk mid-session from an
uncommitted edit that was not part of the commit range I was asked to review.

**Why:** the user runs long test suites and parallel work sessions routinely; this is expected,
not an anomaly to flag.

**How to apply:** when reviewing a specific commit range, freeze the committed version with
`git show <ref>:<path> > /tmp/.../frozen_copy.py` and test against that frozen copy rather than
the live working tree, so concurrent edits by another session don't contaminate the review. When
running the full test suite, expect it to be slow due to GPU/CPU contention from other concurrent
runs (observed ~20% progress after 10+ minutes) — a `pytest --collect-only` pass is a fast,
useful substitute for confirming "nothing broke at import time" when the full run can't finish
in the review window. Report the full run as still-in-progress rather than blocking indefinitely.
