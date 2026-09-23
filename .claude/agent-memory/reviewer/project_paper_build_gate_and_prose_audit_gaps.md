---
name: project-paper-build-gate-and-prose-audit-gaps
description: scripts/paper/build_all.py and scripts/prose_audit.py have zero test coverage and had long-silent bugs; check both by actually running them, not by reading
metadata:
  type: project
---

As of the 2026-09-23 review of commits `pre-cleanup-2026-09-23..HEAD`, neither
`scripts/paper/build_all.py` nor `scripts/prose_audit.py` (nor `app_basis.py`,
`fig_forward_passes.py`, `_style.py`) is referenced anywhere under `tests/`
(`grep -rl "build_all|prose_audit|app_basis|fig_forward_passes" tests/` is empty). Bugs in this
family live silently for a long time: `build_all.py`'s section-checking guard globbed a
nonexistent `paper/chapters/*.tex` and therefore checked 0 files and always passed, for an
unknown but presumably long period, until fixed to glob `paper/sections/*.tex` on 2026-09-23.
The moment the fix landed, running `python scripts/paper/build_all.py --check-only` started
failing with 153 problems (143 bare/un-macroed numbers across every paper section, plus 10
undefined-macro hits for `\FloatBarrier`/`\IfFileExists` missing from the `LATEX_COMMANDS`
whitelist) — debt that had accumulated invisibly the whole time the guard was a no-op.

Separately, `scripts/prose_audit.py`'s LaTeX extractor (`tex_prose`) has a real, currently-active
bug: it splits a line on the first literal `%` to strip comments, which also matches inside `\%`
(escaped percent sign, used ~55 times across `paper/sections/*.tex` for poison rates and
detection rates), silently truncating most of the sentence that follows. It also strips a whole
macro argument (not just the command name) via `TEX_NOISE`, so `\caption{...}` and
`\section{...}` text is invisible to every style rule.

**Why:** this matters for future reviews because reading the code for these files is not enough
to judge correctness — the failure modes only show up when the tool is actually run against real
paper content, since the bugs are about what silently never reaches the rule engine, not about
what the rule engine does wrong once it sees text.

**How to apply:** when reviewing any change to `scripts/paper/*.py` or `scripts/prose_audit.py`,
always (a) run `PYTHONPATH=. python scripts/paper/build_all.py --results-dir results
--check-only` and check the exit code, not just skim the diff, and (b) run the relevant extractor
(`tex_prose`/`markdown_prose`) against a handful of real files with `\%`, `\caption{}` and list
continuations and diff what comes out against what should. See
[[project-multi-agent-concurrent-repo]] for how to do this safely when the repo is being edited
concurrently.
