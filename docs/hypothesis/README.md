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

## Reporting standard: the coverage bar

**No conclusion is reported unless it has been measured across all three axes at
once:**

| axis | required |
|---|---|
| datasets | CIFAR-10, CIFAR-100, GTSRB, Tiny ImageNet, all four |
| poison rates | 1%, 5%, 10%, all three |
| attacks | `badnet_a2o`, `wanet`, `lc`, `blend`, `adaptive_blend`, every one that implants |

A number that covers two of the three axes is not a partial result, it is a lead,
and it is labelled PROVISIONAL. `defence_tables.py` enforces this in code: it
refuses to emit a row whose coverage is incomplete and prints what is missing
instead, so an under-covered claim cannot reach a table by accident.

**No conclusion is reported unless it has been measured on the full panel:
`badnet_a2o`, `wanet`, `lc`, `blend`, `adaptive_blend`.** A result on a single
attack, `badnet_a2o` above all, is a lead and is labelled PROVISIONAL. It is
never a finding, never a headline, and never goes in a table without the other
four beside it.

The reason is not pedantry. `badnet_a2o` is a static patch trigger, the easiest
attack to implant (ASR 0.997 to 1.000 at every poison rate on every dataset) and
the one whose detection behaviour is least like the others. Three results in this
ledger already show the panel disagreeing with itself: `token_mask` is best on the
patch trigger and worst on `wanet` ([H27](H27-token-mask-trigger-locality.md)),
`channel_mask` loses everywhere except the patch trigger at 1%
([H26](H26-channel-mask-structured-vs-elementwise.md)), and `lc` fails under
configurations that carry every other attack. Generalising from `badnet_a2o` to
"the method" is exactly the error those three results warn about.

**Where the panel cannot be completed, say so rather than substituting.**
Attack success at ASR >= 0.5, ViT, no SAM:

| attack | 1% | 5% |
|---|---|---|
| `badnet_a2o` | 4/4 datasets | 4/4 |
| `blend` | 4/4 | 4/4 |
| `adaptive_blend` | 2/4 (cifar10, cifar100) | 4/4 |
| `lc` | 1/4 (cifar100 only) | 2/4 (cifar100, tiny) |
| **`wanet`** | **0/4** | 4/4 |

`wanet` does not implant at 1% on any dataset (peak ASR 0.379 on Tiny). So at 1%
the panel is `badnet_a2o`, `blend`, `adaptive_blend` and `lc` where viable, and
`wanet` is reported as **attack failed to implant**, not as a detection failure
and not silently omitted. Any 1% claim states which of the five it rests on.

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
| [H5](H5-all-to-all-breaks-psbd.md) | PSBD degrades on all-to-all, which has no single target class | **SUPPORTED**: PSBD does not cover all-to-all. The signal is present with the sign reversed, recorded as a diagnostic only, never as detection |
| [H6](H6-sam-improves-detectability.md) | SAM makes backdoors more detectable | **DROPPED**, out of scope. The effect is +0.009 AUROC, which needs 10 to 89 seeds per cell to establish. SAM checkpoints are excluded from every reporting path by default |
| [H7](H7-clean-shifts-to-target.md) | Clean samples under dropout shift specifically to the target class | **PARTIALLY REFUTED** |
| [H8](H8-detection-scales-with-poison-rate.md) | Detection improves with poison rate | INCONCLUSIVE |
| [H9](H9-strength-not-position.md) | The pre/post gap is a perturbation-strength artifact, not a placement effect | **SUPPORTED** for 3 of 4 attacks |
| [H10](H10-depth-band-placement.md) | Aiming dropout at the depth where an attack's direction lives beats spreading it over all blocks | band premise **SUPPORTED** 6/6; onset-based selection **REFUTED** out of sample 0/2 |
| [H11](H11-adaptive-rate-overshoots.md) | PSBD's adaptive rate rule overshoots on ViT | **SUPPORTED**, 12/12, and free to fix |
| [H12](H12-psu-is-not-just-confidence.md) | PSU is just a proxy for baseline confidence | **REFUTED** (and yielded a free improvement) |
| [H13](H13-combined-variant.md) | The accumulated changes combine into a materially better defence | **SUPPORTED**: +0.110 derivation, **+0.151 held out (2/2)** |
| [H14](H14-fusion.md) | PSBD and STRIP fuse into something better than either | **SUPPORTED**: 0.614 TPR at 1% FPR, 93% of oracle-max |
| [H15](H15-one-sided-rules-are-the-common-weakness.md) | One-sided decision rules are a systematic weakness across detectors | **RETIRED as a method**, kept as a recorded negative result. Inversion is a symptom of a broken assumption, never a decision rule; nothing downstream uses it |
| [H16](H16-where-the-backdoor-neurons-are.md) | The backdoor is one late-layer linear direction, not a set of neurons | **SUPPORTED**: post-LayerNorm rank-1 removal takes ASR 1.00 to 0.00 on every checkpoint; "neurons" framing **REFUTED** (300 of 768 coords does nothing); SAM sub-claim **REFUTED** as a LayerNorm artifact |
| [H17](H17-low-poison-rate-is-a-placement-artifact.md) | PSBD's low-poison-rate failure is a placement artifact, not a limit of the method | **SUPPORTED** (full panel, 48/48): position/operator search gains **+0.166** AUROC at matched shift ratio at 1% on CIFAR-100 over the published configuration (the +0.258 gain_scale figure is withdrawn as unmatched, see the audit) |
| [H18](H18-sensitivity-profile-over-units.md) | The shape of a per-unit sensitivity profile beats its mean | **REFUTED**: 144-head leave-one-out carries no signal (mean 0.28-0.61, concentration 0.12-0.44, benign clean at 0.497). Third failed attempt to exploit *where* the backdoor sits |
| [H19](H19-placement-ranking-is-rate-selection.md) | The placement ranking is mostly about which placement the adaptive rate rule can aim at | **Swin half INCONCLUSIVE** (point estimate 0.110 to 0.015 to 0.045 as n grew 6 to 11 to 16; CI contains zero throughout); holds weakly on ViT. `blocks_9_12` still has the project's best oracle AUROC and **0/20 deployable operating points** |
| [H20](H20-input-side-beats-residual-adjacent.md) | What matters is sub-layer input versus residual stream, not pre- versus post-residual | **SUPPORTED**, stress-tested: **+0.054**, bootstrap CI [+0.031, +0.080], positive on every leave-one-out and leave-one-attack-out refit. Pre-versus-post, the founding question, is **+0.002** |
| [H21](H21-droppath-residual-native.md) | DropPath is the residual-native perturbation | **CONFIRMED**, including the prediction that it would lose (0.860). The unit that matters is the feature, not the computation |
| [H22](H22-head-mask-attention-units.md) | Attention heads are the transformer's own unit | **REFUTED**: 0.886, below dropout. Heads are redundant, the backdoor is not in them, and one head (block 1 head 6) dominates all inputs equally |
| [H23](H23-gaussian-noise-control.md) | Does PSBD need capacity removed, or merely disturbed? | **REFUTED, most consequential result**: gaussian noise scores 0.950, ahead of every mask. Removal is not required, so the neuron-bias mechanism is not what carries the method on ViT |
| [H24](H24-monte-carlo-passes.md) | k=3 Monte Carlo passes is the noise floor at low poison rate | **CONFIRMED**, both predictions: k=20 gives +0.028 at 1% vs +0.011 at 10%, `badnet_a2o` 1% +0.046, benign unmoved |
| [H25](H25-adaptive-attacker.md) | An adaptive attacker can hide from PSBD, but only from the probe it trained against | **CONFIRMED**: probed AUROC collapses 0.952 to 0.322, transfer operators still detect at 0.887 mean |
| [H28](H28-perturbation-consistency-is-margin-estimation.md) | PSBD, SCALE-UP, IBD-PSC and STRIP are one method: perturbation-consistency measures decision margin, and the operator only sets the Jacobian | **PARTIALLY SUPPORTED**: prediction 4 (interchangeability) confirmed at 1.43x position/family ratio with Kendall tau 0.700; prediction 3 (only stream positions invert) **REFUTED** (input-side inversions 9.4% vs stream 4.1%); predictions 1, 2 untested (need GPU) |
| [H26](H26-channel-mask-structured-vs-elementwise.md) | Masking whole channels beats thinning every channel a little | **REFUTED as stated** (panel-backed): loses to dropout at every matched position at 10%. The `badnet_a2o` 1% win is a single cell, **PROVISIONAL** |
| [H27](H27-token-mask-trigger-locality.md) | Removing whole tokens separates local triggers from distributed ones | **SUPPORTED** (attack ordering: patch 0.985, warp 0.747) and stronger overall than predicted. No operating point at 1% yet, which is itself a limitation of the operator |
| [H29](H29-cross-attack-direction-universality.md) | Different attacks targeting the same class produce parallel backdoor directions | **REFUTED**: off-diagonal cosine 0.023 to 0.053 (indistinguishable from random). Each attack learns its own direction |
| [H30](H30-residual-persistence-phase-transition.md) | The backdoor direction persists uniformly through the residual stream | **SUPPORTED with qualification**: phase transition, not uniform persistence. Crystallizes at layers 8 to 10, blend earlier than badnet |
| [H31](H31-attention-divergence-backdoor-heads.md) | Backdoored inputs produce distinctive attention patterns in identifiable heads | **SUPPORTED**: 3 heads (L5H0, L6H3, L5H10) diverge across all attacks. BadNet recruits additional late heads (L9H7, L10H9) |
| [H32](H32-token-localization-spatial.md) | Per-token direction norms localize triggers spatially | **SUPPORTED**: BadNet top token at (13,13) matches trigger position. Blend and WaNet show diffuse patterns. SIG shows row-0 concentration |
| [H33](H33-weight-spectral-signature.md) | The weight difference (backdoor minus benign) is low-rank | **REFUTED**: encoder weight matrices show 2 to 5% top-1 concentration. The backdoor is rank-1 in activation space but NOT in weight space |
| [H34](H34-direction-erasure-defense.md) | Weight orthogonalization removes backdoors | **PARTIALLY SUPPORTED**: works on weak attacks (LC blind: 0.626 to 0.006), fails on strong (badnet, blend stay at 1.000). Residual stream persistence explains the gap |
| [H35](H35-targeted-head-psbd.md) | Masking the 3 backdoor heads as a PSBD operator outperforms random head masking | **REFUTED**: 0.580 mean AUROC (3-head), only +0.04 over random head masking (0.539), far below dropout (0.911). Head masking as a class is too weak |
| [H36](H36-cone-geometry-backdoor-directions.md) | Backdoor directions cluster in a cone around the readout weight | **REFUTED**: 9 of 10 attacks have angle 87 to 91 degrees to the readout weight (cosine near 0). Only badnet_a2o aligns (33 to 43 degrees). No cone exists |
| [H37](H37-token-concentration-attack-classifier.md) | Token concentration ratio classifies attack family (localized vs global) | **REFUTED**: 62.5% accuracy, separation gap -0.98. adaptive_blend and lf have high concentration despite being global attacks |
| [H38](H38-crystallization-depth-vs-placement.md) | Crystallization depth predicts optimal perturbation placement | **SUPPORTED** (crystallization confirmed: blend at layer 8.2, badnet at layer 10.4), **INCONCLUSIVE** (placement correlation untested, no block-band PSBD data) |
| [H39](H39-skip-scaling-defense.md) | Skip connection scaling at layers 10 to 11 removes backdoors | **PARTIALLY SUPPORTED**: works on wanet and lc (alpha=0.3 to 0.5), blend has a sharp threshold (alpha=0.1). BadNet survives even alpha=0.0 at 5% (ASR=0.995). No universal alpha |
| [H40](H40-attention-entropy-detection.md) | Per-sample attention entropy in backdoor heads detects backdoor samples | **REFUTED**: AUROC 0.48 to 0.54 for most attacks (random). Blend shows strong inverted signal (higher entropy, AUROC 0.000 under one-sided convention) |
| [H41](H41-multi-probe-defence.md) | Multi-probe PSBD defeats the adaptive attacker | **SUPPORTED**: min-rank union of k probes recovers AUROC 0.951 from single-probed 0.322. 48/56 above 0.90 |
| [H42](H42-evasion-identification.md) | The evaded operator can be identified without poison labels | **REFUTED**: 8.9% accuracy by val PSU std, below 25% chance. Inherent operator differences dominate. Identification not needed: multi-probe works without it |
| [H43](H43-entropy-covers-all-to-all.md) | The case PSBD cannot cover is covered by its own discarded tensor | **SUPPORTED**: entropy 0.761 against PSBD's 0.411 on all-to-all, benign at chance, from 1 cached forward pass. A label-free sign rule (mean PSU of the suspect pool minus clean validation) selects between them for **+0.045**, CI [+0.017, +0.079], 98% of oracle-max, hurting 0 of 64 cells. Rests on 9 all-to-all cells and is 0.000 at 1% poisoning |

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
- **H5.** All-to-all is undetectable by PSBD, and the reason is mechanistic rather
  than statistical: its backdoor samples are 4x *more* fragile under dropout than
  clean ones, where all-to-one's are more robust. There is no single target class
  for the perturbed prediction to collapse onto, so the premise PSBD rests on does
  not hold. The separation is measurable with the sign reversed, but **that is
  reported as a diagnostic, not as detection** (see the retirement note on
  [H15](H15-one-sided-rules-are-the-common-weakness.md)): choosing which tail to
  flag needs the poison labels the detector exists to predict, so a two-sided
  number is not a result. All-to-all stands as a case PSBD does not cover.

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

## The failure mode this ledger keeps hitting

Three distinct versions of one mistake, each of which inverted a conclusion:

1. **Comparing outside a placement's operating range** (H9, H1, H3). Fixed by
   matching on clean-validation shift ratio, never on rate.
2. **Averaging over unequal coverage** (H19, H20). Different groups were swept over
   different checkpoints, so the means compared different problems. Run through
   `experiments/balanced_panels/audit.py`, **4 of 6 comparisons here invert**, the worst
   at 45x imbalance. The 2 that survive are the 2 with imbalance under 1.5x.
3. **Measuring across a normalization layer with a scale-dependent statistic**
   (H16 sections 9 and 10). A pre-LayerNorm ablation produced a clean, monotone,
   benign-controlled and entirely false result that survived 2 rounds of follow-up.

4. **Re-reading a comparison every time the data grows.** H19's Swin claim was
   written 3 times from panels of 6, 11 and 16 units, giving +0.110 ("supported"),
   +0.015 ("refuted") and +0.045 ("inconclusive"), with a bootstrap interval
   containing zero throughout. Each re-read was a fresh chance to over-interpret
   noise, and the middle verdict repeated the first error with the sign flipped. The
   fix is a **pre-registered n**: compute the panel size needed for 80% power, record
   it, and do not re-open the question below it. H19 now carries such a threshold.

All 4 produce *more* exciting results than the truth, pass their benign controls, and
are invisible without an explicitly constructed comparison. Treat any table here that
was not built on a balanced panel, at matched shift ratio, with a scale-invariant
statistic, and at a pre-committed sample size as unverified.

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
