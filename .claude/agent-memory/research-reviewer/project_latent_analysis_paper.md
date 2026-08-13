---
name: project-latent-analysis-paper
description: The analysis/ latent toolkit is being prepared to back mechanistic claims in a paper, so its correctness bar is publication-level, not exploratory
metadata:
  type: project
---

As of 2026-08-13, the `analysis/` subpackage (features, direction, cka, embedding,
lipschitz, analyze_latent) is being readied to produce **mechanistic claims for a
paper**, on the `latent-analysis` branch.

**Why:** these are not exploratory plots. Numbers such as "the trigger enters the
residual stream at layer L", TAC curves, CKA, backdoor-direction norms, and the
weight-based head detector are intended as publishable evidence, so a silent
numerical artifact (dtype noise floor, confounded layer selection, unpaired
features) is a publication-invalidating bug rather than a nuisance.

**How to apply:** when reviewing anything under `analysis/`, hold it to the standard
of "would this number survive a reviewer asking for the control?" Specifically ask
for: a clean-vs-clean null control for any paired-difference metric, a
scale-normalized version of any depth-varying quantity, and provenance (which
checkpoint, which attack config) persisted next to every reported number.

Related: [[reference-backdoor-directions-paper]], [[user-role]].
