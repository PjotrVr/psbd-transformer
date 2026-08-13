---
name: feedback-review-style
description: How this user wants ML-security audits delivered - read-only, exact file:line, severity-ranked, no praise padding, no report files
metadata:
  type: feedback
---

Audits in this repo are **read-only**. Never edit a file, never write a findings/report `.md`.
Findings go in the final assistant message as a severity-ranked list, each with an exact
`file:line`, a concrete failure scenario (inputs/state -> wrong output), and a suggested fix
the user will apply themselves.

**Why:** the user runs these as a separate skeptical pass over code that already passes tests
and is often about to be launched on the cluster; a reviewer that edits code destroys the
audit trail, and a reviewer that pads with praise buries the one finding that matters.

**How to apply:** when the user names a scope and lists "context you must accept as
established", do not re-derive or re-report those items. Answer their numbered questions
directly and in order, then add anything else found. State plainly when a category has no
findings rather than manufacturing one. End with a single `LAUNCH` / `DO NOT LAUNCH` line.
Prefer verifying claims by reading the actual torch/torchvision source in `.venv/` over
reasoning from memory of the API.
