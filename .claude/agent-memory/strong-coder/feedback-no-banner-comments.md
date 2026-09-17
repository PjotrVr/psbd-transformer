---
name: feedback-no-banner-comments
description: No banner or step-narration comments in committed code; a comment gives a reason, never a heading
metadata:
  type: feedback
---

Never commit banner comments (`# --- section ---`) or step narration ("step 1: ...", numbered
comments in a docs code block). A comment states a reason or a non-obvious fact.

**Why:** `.claude/styles/coding-style.md` line 14 forbids decorative dividers and placeholder
comments, and the user re-stated it as a rule for everything committed.

**How to apply:** do not copy the banner style even when the file already has it (`tools/cli.py`
has old `# --- name ---` dividers; new blocks added to it must not). Rewrite numbered comments in
docs shell blocks as statements of what the command is for.
