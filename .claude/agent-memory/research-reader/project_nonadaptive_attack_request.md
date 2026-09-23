---
name: nonadaptive-attack-request
description: Mentor asked (2026-09-23) for a NON-adaptive attack that breaks PSBD on both ResNet and ViT; adaptive-loss attacks are explicitly out of scope
metadata:
  type: project
---

The user's mentor wants a general-purpose backdoor attack that defeats prediction-shift
detection as a side effect of its trigger or its training, never by referencing the defense.
Hard constraint: no term in the attacker's loss may mention PSBD, the prediction shift,
dropout or any probe. The adaptive case is already studied and is rejected on sight.

**Why:** the project's contribution is PSBD-TM at `before_attention_norm_token_mask`, and a
defense paper needs a threat model it does not trivially survive. An adaptive-loss attack
proves nothing new, because PSBD's own appendix already builds one.

**How to apply:** when asked about attacks against this project, classify every candidate by
whether its objective references a stability statistic. Survey memo with the ranked shortlist
is at `literature/README-attack-survey-2026-09-23.md`. The 5 ranked directions are
probabilistic relabeling to a low margin, reverse training to a weakened trigger-target
association, clean-label triggers aligned to target-class latent geometry, multi-trigger
parallel poisoning and sharpness-inducing training. See [[literature-folder-layout]].
