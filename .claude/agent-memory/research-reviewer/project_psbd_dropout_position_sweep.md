---
name: psbd-dropout-position-sweep
description: In-flight PSBD-ViT dropout-position sweep - known post_residual hook bug, its planned fix, and the open confounds flagged in the 2026-08-13 audit
metadata:
  type: project
---

The current line of work is the hook-based dropout-position sweep (`defences/dropout.py`,
`defences/inference.py`, `psbd_dropout_sweep.py`), run as 60 single-GPU PBS jobs
(6 checkpoints x 10 position-configs, 9 rates each).

Known and already scheduled for a fix (do not re-report as a new finding):
`DROPOUT_CONFIGS["post_residual"]`'s `after_attention_residual` member is a pre-hook on
`ln_2`, which perturbs only the MLP branch input, not the residual stream reused in
`return x + y`. The planned fix is a removable per-instance forward wrapper rather than a
hook.

**Why:** the project's core contribution is pre-residual vs post-residual dropout placement
in ViT/Swin, so `post_residual` has to actually land on the residual stream for the headline
comparison to mean anything.

**How to apply:** when reviewing this area, skip the `ln_2` finding and focus on the
remaining confounds raised on 2026-08-13: (1) shift ratio sigma is not saved by
`compute_dropout_pass_probs`, so PSBD's paper-specified adaptive p selection cannot be run
from stage-1 artifacts; (2) a fixed p grid is not a comparable axis across positions,
because some positions are immediately renormalized by a downstream LayerNorm and some
compound over 12 (ViT) or 24 (Swin) blocks; (3) the eval split is built from the test set,
so this measures backdoor-input detection, not the paper's poisoned-training-sample
filtering. See [[psbd-paper-fidelity-deviations]].
