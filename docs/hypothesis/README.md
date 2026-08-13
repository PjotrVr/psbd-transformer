# Hypothesis ledger

One file per hypothesis. Each states what it predicts in falsifiable terms, what
evidence would settle it, and the verdict once evidence lands.

**Status vocabulary.** `OPEN` (no evidence yet), `SUPPORTED` (evidence points
this way, stated with the numbers), `REFUTED` (evidence points the other way),
`INCONCLUSIVE` (evidence gathered and it does not decide). A hypothesis is never
quietly deleted; a refuted one keeps its file, because knowing what was ruled out
is most of the value.

Every file names the script or artifact that produced its evidence. If a claim
here has no runnable path back to a number, it is not evidence, it is a guess,
and it says so.

## The question all of these serve

PSBD (`papers/PSBD/`) detects backdoors by turning dropout on at inference and
measuring how far the model's confidence in its own no-dropout prediction falls.
Clean samples lose a lot of confidence; backdoor samples barely move, because the
trigger-to-target path is the most robust thing the model learned. On ResNet-18
the paper places dropout **after each residual add**. On ViT that placement is
reported to fail, and **pre-residual** (on each branch, just before the add) to
work. This ledger is about localizing why, and turning the observation into a
mechanism.

The companion paper (`papers/backdoor_directions/`) supplies the candidate
mechanism: in ViT the backdoor is a linear direction in the residual stream, and
when it reaches the `[CLS]` token is attack-dependent (final few layers for
static patch triggers, layers 5 to 6 for distributed ones). A perturbation should
only suppress the backdoor if it lands where that direction is still being
written.

## Index

| ID | Claim | Status |
|---|---|---|
| [H1](H1-pre-beats-post.md) | Pre-residual beats post-residual on ViT | SUPPORTED (preliminary) |
| [H2](H2-psbd-transfers-to-vit.md) | PSBD works on ViT at all, and not on benign models | OPEN |
| [H3](H3-why-post-residual-fails.md) | Post-residual fails by saturating: it destroys clean and backdoor evidence alike | SUPPORTED (preliminary) |
| [H4](H4-placement-is-attack-dependent.md) | The best placement tracks where the backdoor direction enters `[CLS]` | OPEN |
| [H5](H5-all-to-all-breaks-psbd.md) | PSBD degrades on all-to-all, which has no single target class | OPEN |
| [H6](H6-sam-improves-detectability.md) | SAM makes backdoors more detectable | OPEN |
| [H7](H7-clean-shifts-to-target.md) | Clean samples under dropout shift specifically to the target class | OPEN |
| [H8](H8-detection-scales-with-poison-rate.md) | Detection improves with poison rate | OPEN |
| [H9](H9-strength-not-position.md) | The pre/post gap is a perturbation-strength artifact, not a placement effect | OPEN (adversarial) |

H9 is deliberately the adversarial one. It is the reviewer's objection stated as a
hypothesis, and the whole study is worthless if it cannot be refuted.

## Protocol notes that apply to every hypothesis

- **Seeding.** `lightning.seed_everything`, seeds `0, 1, 2, 3, ...`. Split seed
  and dropout-mask seed are separate and both fixed.
- **Threshold.** 25th percentile of clean-validation PSU, the paper's choice in
  every experiment. Quantiles 0.10 / 0.15 / 0.20 are swept alongside.
- **FPR is nearly uninformative here, by construction.** The threshold is a
  quantile of clean *test* PSU and FPR is measured on clean *test* PSU from the
  same distribution, so FPR lands near the quantile whatever the placement does.
  AUROC and TPR carry the signal; FPR is reported for completeness, not as
  evidence.
- **Deviation from the paper's protocol.** PSBD scores the poisoned *training*
  set; this scores a held-out *test* pool (clean images vs the same images
  triggered). So these numbers answer "can PSBD flag a triggered input" and are
  not directly comparable to the paper's tables. Stated once here rather than
  hedged in every file.
