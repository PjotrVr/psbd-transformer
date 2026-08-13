# H5 — PSBD degrades on all-to-all, which has no single target class

**Status: OPEN** (detection pending; a supporting mechanism is already measured)

## Claim

PSBD's stated mechanism is "neuron bias": under heavy dropout the model falls back
on its strongest learned association, so clean samples collapse onto the target
class `y_t` while backdoor samples stay put. All-to-all backdoors map class `y` to
`(y+1) mod K`, so there is no single `y_t` to collapse onto. If the mechanism is
real, detection should degrade badly on `badnet_a2a` relative to `badnet_a2o`,
which uses an identical trigger and differs *only* in label geometry.

## Prediction

`badnet_a2a` AUROC substantially below `badnet_a2o` AUROC at matched poison rate,
despite both having ASR above 0.9 and the same trigger pattern.

Refuted if a2a detects as well as a2o, which would mean PSU works for a reason
other than the neuron-bias story the paper tells.

## Why it is interesting

It is a clean, controlled falsification test. Same trigger, same architecture,
same poison rate, same everything except which label the poisoned samples carry.
Almost nothing else in backdoor research isolates the label geometry that cleanly.

If a2a detects *well*, the paper's mechanism explanation is wrong even though the
method works, which is a more interesting result than confirmation.

## Confound that has to be controlled

`AttackSuccessSet` for all-to-all admits every analysis sample, because every
class is eligible. So the positive class includes triggered images the trigger
never actually flipped. On CIFAR-10 a2a ASR is 0.93 to 0.96, so this is a ~5%
effect and second-order; on CIFAR-100 a2a it is ASR 0.50 and would dominate
entirely. `psbd_analyze.py` therefore reports `detection_captured_only` alongside
the unconditioned numbers, and **the captured-only figure is the one this
hypothesis is judged on**. CIFAR-100 a2a is excluded from the grid for this reason.

## Evidence

Detection numbers pending; `vit_cifar10_badnet_a2a_{0_01,0_05,0_1}` are in the
phase-3a grid.

**A supporting mechanism is already measured, independently of PSBD.**
`scripts/backdoor_direction_layers/` finds that `badnet_a2a`'s backdoor direction
is roughly half the magnitude of `badnet_a2o`'s at every layer, despite an
identical trigger and comparable ASR (0.96 vs 1.00), and that its clean-vs-triggered
CKA at layer 12 is **0.953** where `badnet_a2o` is 0.419. Clean and triggered
representations stay nearly indistinguishable.

The reason is structural rather than empirical. A backdoor direction is a *mean* of
paired differences. All-to-all pushes each source class somewhere different, so the
mean partially cancels: there is no single direction because the attack does not
implement one. That predicts failure for any method assuming a single target,
including the companion paper's steering and orthogonalization, not just PSBD.

## Subquestions

1. Does the shift-target histogram for a2a spread across many classes where a2o
   spikes on class 0? That is the mechanism claim measured directly, and it is
   the same evidence [H7](H7-clean-shifts-to-target.md) needs.
2. If a2a fails, can PSU be repaired for it, e.g. by scoring against the
   *per-sample* expected shifted class `(y+1) mod K` rather than a global one?
3. Do other multi-target attacks (TaCT's source-specific mapping) fail the same
   way? TaCT does not train successfully on ViT here, so this may not be testable.
