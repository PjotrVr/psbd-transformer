# H6 — SAM training makes backdoors more detectable

**Status: OPEN**

## Claim

Models trained with SAM (Sharpness-Aware Minimization) on top of AdamW yield
higher PSBD AUROC than Adam-trained models of the same attack and poison rate,
and detectability varies systematically with rho.

## Rationale

SAM seeks flat minima. The intuition is that a backdoor is a sharp, narrow
feature in the loss landscape, so flattening pressure should either weaken it
(lower ASR, less to detect) or force it into a more separable, more distinct
representation (higher detectability). Which of the two happens is the question,
and the measured ASR already says the first does *not* happen: on CIFAR-10 ViT,
ASR is unchanged across rho for every candidate attack, and clean accuracy
actually improves by 1 to 2 points.

So SAM does not remove these backdoors. That makes the detectability question
clean: same ASR, same trigger, different loss geometry.

## Prediction

AUROC increases monotonically, or at least consistently, with rho over
{Adam, 0.05, 0.1, 0.15, 0.2}. Refuted if AUROC is flat in rho, which would say
loss-landscape flatness is irrelevant to prediction-shift detectability.

## Why it is interesting

If it holds, it is an actionable and slightly odd recommendation: a defender who
controls training can make backdoors *easier to find later* by choosing the
optimizer, without needing to know a backdoor exists. That is a cheap intervention
compared to any dedicated defence.

If it does not hold, it usefully bounds the "flat minima help with backdoors"
intuition, which is widely assumed and rarely measured against a detection metric.

## Evidence

Pending. Phase 3c, 4 rho values x 5 attacks x 3 poison rates + 4 SAM benign
controls, on whichever placements phases 3a and 3b show to be worth the compute.
Deliberately not run before the placement question is settled: running the rho
sweep on a placement that turns out not to work would waste most of the budget.

Baseline ASR / clean accuracy across rho are already measured and confirm SAM
does not degrade either.

## Subquestions

1. Does SAM change the *shift ratio* curve, i.e. does it change how much
   perturbation a given `p` causes? If so, comparisons across rho need the same
   matched-sigma treatment as [H9](H9-strength-not-position.md).
2. Does SAM move the backdoor direction's entry layer? Measurable directly with
   `analysis/direction.py` and would connect this to [H4](H4-placement-is-attack-dependent.md).
3. Is the effect on the clean side or the backdoor side? SAM improves clean
   accuracy, so higher AUROC might come from clean samples being more confident
   rather than backdoor samples being more distinct. The PSU means separate these.
