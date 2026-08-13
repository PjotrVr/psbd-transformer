---
name: user-role
description: Petar Stika, ML security researcher running PSBD-ViT; wants skeptical, read-only, severity-ranked audits with exact file:line refs
metadata:
  type: user
---

Petar Stika is running the PSBD-ViT research project (adapting Prediction Shift Backdoor Detection from ConvNets to ViT-B/16 and Swin) on the Supek/SRCE PBS cluster.

Working style he asks for in review tasks:
- Read-only audits. Never edit files during a review; report findings only.
- Findings must carry exact `file:line`, a concrete failure scenario, a suggested fix, and a severity rank (BLOCKER / HIGH / MEDIUM / LOW).
- Explicitly asks not to pad: "be skeptical; do not pad." Saying "no blockers found" is preferred over manufacturing findings.
- He pre-declares already-known issues and asks them not to be re-reported, so check the prompt's exclusion list before writing anything up.
- He is comfortable with dense technical prose and paper-level detail (LaTeX sources of the papers he reproduces live under `papers/`).

See [[psbd-vit-research-validity]] for standing correctness facts about the project.
