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
| [H1](H1-pre-beats-post.md) | Pre-residual beats post-residual on ViT | **REFUTED** as a general claim |
| [H2](H2-psbd-transfers-to-vit.md) | PSBD works on ViT at all, and not on benign models | **SUPPORTED** |
| [H3](H3-why-post-residual-fails.md) | Post-residual fails by saturating | **REFUTED** as stated |
| [H4](H4-placement-is-attack-dependent.md) | The best placement tracks where the backdoor direction enters `[CLS]` | **SUPPORTED** |
| [H5](H5-all-to-all-breaks-psbd.md) | PSBD degrades on all-to-all, which has no single target class | **SUPPORTED, strongly** |
| [H6](H6-sam-improves-detectability.md) | SAM makes backdoors more detectable | OPEN (phase 3c) |
| [H7](H7-clean-shifts-to-target.md) | Clean samples under dropout shift specifically to the target class | **PARTIALLY REFUTED** |
| [H8](H8-detection-scales-with-poison-rate.md) | Detection improves with poison rate | INCONCLUSIVE |
| [H9](H9-strength-not-position.md) | The pre/post gap is a perturbation-strength artifact, not a placement effect | **SUPPORTED** for 3 of 4 attacks |
| [H10](H10-depth-band-placement.md) | Aiming dropout at the depth where an attack's direction lives beats spreading it over all blocks | OPEN (mechanism built, confound identified) |
| [H11](H11-adaptive-rate-overshoots.md) | PSBD's adaptive rate rule overshoots on ViT | **SUPPORTED**, 12/12, and free to fix |

## Where this stands after phases 3a and 3b

**The founding claim did not survive.** Pre-residual does not beat post-residual on
ViT in general. Swept over its own rate window (p = 0.005 to 0.09, which the grid
inherited from the ConvNet paper never reached), post-residual **wins on three of the
four working attacks**: blend 0.989 against 0.978, bpp 0.991 against 0.977, lf 0.969
against 0.947. Pre-residual leads only on the static patch trigger, and there by a
large margin (0.889 against 0.747).

**And pre-residual is not the best placement anyway.** It ranks 4th of 11. The top
three are `before_mlp_residual` (0.969 mean), `before_attention_norm` (0.965) and
`before_attention` (0.962), against pre-residual's 0.948. `pre_residual` even scores
below `before_mlp_residual` alone, which is one of its own two components: adding the
attention-branch perturbation actively dilutes the MLP-branch one.

**The most useful result is the cheapest.** PSBD's adaptive rate rule targets a
clean-validation shift ratio of 0.8; on ViT the optimum sits near 0.7, and the rule
overshoots on 12 of 12 checkpoints, costing about 0.09 AUROC that retuning one
constant recovers for free ([H11](H11-adaptive-rate-overshoots.md)).

The two results that would change a paper:

- **H7.** `blend` is the best-detected attack in the grid (AUROC 0.978 to 0.986) and
  shows *no* shift-to-target effect (0.052 at 10% poisoning, below the 0.10 chance
  line). PSBD's published mechanism is measurably absent exactly where PSBD works
  best, so on ViT the method and its explanation come apart.
- **H5.** All-to-all is undetectable (AUROC 0.510 at 10% poisoning, against a benign
  floor of 0.506) while its backdoor fires on 96% of inputs. Identical trigger to
  `badnet_a2o`, which reaches 0.889. Label geometry, not trigger, decides.

Corrections worth being loud about. **Three verdicts have now been overturned by
their own follow-ups**: H3 (post-residual does not fail after all), H1 (pre-residual
does not generally win), and the earlier reading of H9 (the adversarial hypothesis was
largely right). In every case the error ran the same way, and it is worth naming:
a placement was compared against another placement **outside its own operating
range**, and the resulting gap was read as a property of the position.

The through-line of this whole study is therefore methodological rather than about
any particular position: **what transfers badly from ConvNets to transformers is the
rate grid and the rate-selection constant, not the placement.**

H9 is deliberately the adversarial one. It is the reviewer's objection stated as a
hypothesis, and the whole study is worthless if it cannot be refuted.

The confound H9 names has now appeared twice: once for pre versus post residual, and
again for early versus late block bands (H10), where an early band is a stronger
intervention than a late one at the same rate because it propagates through more
blocks. Treat it as the default hazard of this whole line of work: **no comparison
across placements is valid at a shared dropout rate.** Match on clean-validation
shift ratio instead.

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
