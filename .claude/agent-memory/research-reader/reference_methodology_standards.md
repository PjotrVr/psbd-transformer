---
name: methodology-standards-memo
description: docs/security-ml-research-standards.md holds the evaluation-validity checklist (Arp pitfalls, base rate arithmetic, adaptive-attack and artifact standards) with 40+ rows and per-row verdicts
metadata:
  type: reference
---

`docs/security-ml-research-standards.md`, written 2026-09-23, is the project's evaluation
methodology reference. It extracts Arp et al.'s 10 pitfalls (USENIX Security 2022) with
prevalence and remedies, Axelsson's base rate argument, Carlini's TPR-at-low-FPR frame with
the AUROC counter-arguments, the Carlini 2019 and Tramer 2020 adaptive checklists, Abad et
al.'s SoK on 183 backdoor defense papers (arXiv 2511.13143), multiple-comparison reporting
standards and the ACM, USENIX and NeurIPS artifact criteria. It closes with a 44-row
requirement table carrying a per-row verdict read from the repository and a ranked list of
the 5 mistakes this project is most at risk of.

**Why:** the user said plainly "I do not want to make stupid mistakes" while finishing the
paper, so the memo is the held-to standard rather than background reading. It cross-references
`docs/open-questions.md` by Q number and deliberately does not restate those entries.

**How to apply:** before asserting a methodology gap is new, check this file and
`docs/open-questions.md`. The reusable facts it establishes: `paper/headline.tex` already
carries `\HeadlineTprAtOnePercent` at 0.602 so a 1% FPR column needs no compute,
`defenses.decision.detection_report` returns a realized `fpr` that no paper table prints,
`threshold_diagnostics` is called only from `cli/baselines.py`, and `configs/psbd_basis.json`
declares `auprc` and `achieved_fpr` as required metrics that nothing in `defenses/` computes.
See [[literature-folder-layout]] and [[nonadaptive-attack-request]].
