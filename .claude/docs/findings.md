# PSBD on Vision Transformers: findings

State after the ViT CIFAR-10 grid, the baseline comparison, and the SAM control.
The CIFAR-100, Swin and low-poison-rate sweeps are still running. Per-hypothesis
detail with full numbers is in `docs/hypothesis/` (H1 to H15).

## The short version

PSBD transfers to ViT, but almost every specific claim the project started with turned
out to be wrong, and the useful results are the ones that replaced them.

The founding claim (pre-residual beats post-residual) is **refuted**. What actually
fails to transfer from ConvNets is the **rate grid and the rate-selection constant**,
not the placement. PSBD's published mechanism is **measurably absent** on the attack it
detects best. And the single most useful finding is not about PSBD at all: **both PSBD
and STRIP are one-tailed detectors that run backwards on the attacks their assumptions
do not cover**, which is worth up to +0.98 AUROC.

## What works

**PSBD detects backdoors on ViT, and the control holds.** Benign model probed with the
same trigger: AUROC 0.502 to 0.508 across all 14 placements, TPR 0.016 at 1% FPR.
Backdoored models reach 0.89 to 0.99 on four of five attacks.

**The best placement is not the one the project was built on.** Ranked over the four
working attacks: `before_mlp_residual` 0.969, `before_attention_norm` 0.965,
`before_attention` 0.962, `pre_residual` 0.948. Restricting `pre_residual` to a band of
blocks beats applying it to all twelve on **6 of 6** attacks (+0.019 to +0.136) at a
third of the perturbation cost; **blocks 5-8** is the robust default (best mean rank,
never worse than second).

**Four changes combine into a materially better defence** (H13): band placement,
fractional PSU, a retuned rate target, and a two-sided rule. Against published PSBD:
**+0.110 mean AUROC on the derivation set (14/15 wins)** and **+0.151 on two held-out
attacks (2/2)**, all with the rate chosen on clean validation data only. On `lc` the
published configuration sits at chance (0.515) and the adapted one reaches 0.786.

**Rate selection is mistuned for ViT** (H11). The paper targets a clean-validation
shift ratio of 0.8; on ViT the optimum is near **0.70**, and the rule overshoots on
**12 of 12** checkpoints, costing about 0.09 AUROC that retuning one constant recovers
for free.

**PSU is not a confidence proxy** (H12). The fractional form, which divides the
starting confidence out entirely, wins on **92.8% of 624 grid cells** with a worst
regression of -0.0025. Confidence alone scores 0.012 mean TPR at 1% FPR.

## What broke

**Post-residual does not fail.** Given its own rate window (p = 0.005 to 0.09, an
order of magnitude below the inherited grid), it beats pre-residual on three of four
attacks. ResNet-18 has 8 residual adds and ViT-B/16 has 24, so the same nominal rate
compounds three times as hard and the ConvNet grid starts past the operating point.

**PSBD's published mechanism is absent where PSBD works best** (H7). The paper says
clean samples under dropout collapse onto the attacker's target class. On ViT the
*concentration* is real (one class takes 0.27 to 0.79 of shifted predictions, against a
0.10 chance line) but the class is usually **not** the target. `blend`, the best-detected
attack at 0.978 to 0.986, sends only 0.052 of its shifted clean predictions to `y_t`.
Worse, the **benign** model's own fallback is class 0, which is the `y_t` every attack
here uses, so the convention confounds its own mechanism test.

**Six verdicts were overturned by their own follow-ups**: H1, H3, H5, H9's first
reading, H10's predicted direction, and H6. Five were mine. The recurring error was
comparing a placement against another **outside its own operating range**.

## The two results that generalize beyond this project

**One-sided decision rules are the shared weakness** (H15). PSBD inverts on all-to-all;
STRIP, which shares no machinery with it, inverts on **nine checkpoints**, up to
AUROC 0.010 recovering to 0.990. Both invert on exactly the cases their stated
mechanism does not cover: low-amplitude full-image triggers that do not survive
superimposition, and all-to-all which has no single target class. A label-free tail
choice recovers most of the gain, but only with a tail quantile matched to a rare
contaminant: at the default 5th percentile it is a coin flip when 1% of the pool is
poisoned, and at the 0.2nd percentile it reaches 79%.

**The detectors are complementary, and fusing them beats both** (H14). STRIP is
near-perfect on the static patch trigger (0.97 to 1.00 at 1% FPR) and exactly 0.000 on
`adaptive_blend` and `badnet_a2a`; PSBD is the reverse. No checkpoint defeats both.
Taking the more suspicious of the two verdicts gives **0.614 mean TPR at 1% FPR**
against 0.507 for STRIP and 0.379 for PSBD, recovering **93% of an oracle** that picks
the better detector per checkpoint. Its value is not that it beats both (only 6 of 23)
but that it never collapses: median shortfall against the oracle is -0.017.

## SAM (H6)

SAM does amplify the backdoor on ViT, monotonically in rho (top-2 TAC 3.25 to 6.07 for
`badnet_a2o`), so the mechanism the SAM paper describes is present in our models. But
**separability does not improve** (silhouette moves ±0.02, where their ResNet18 went
0.19 to 0.32) because a pretrained ViT already starts at 0.45 to 0.51, leaving no
headroom. And the side effect their stage-2 feature scaling exists to cancel is present
and uncancelled: clean intra-class variance rises with rho, which raises clean PSU and
closes the gap PSU thresholds on.

SAM hands a prediction-space detector the cost without the benefit. Their paper
validates on five feature-space detectors, never cites PSBD, and its only
prediction-space detector (STRIP) appears solely in **commented-out** tables where it
is the one method SAM does not help.

## Honest limitations

- One dataset, one architecture, one seed, k=3 passes so far.
- The evaluation pool is a held-out **test** split, not the poisoned **training** set
  PSBD was designed to filter, so none of these numbers are directly comparable to the
  published tables.
- FPR at the paper's 25th-percentile threshold is uninformative here by construction:
  threshold and FPR come from the same clean test distribution.
- PSBD is given a per-checkpoint placement and rate search that the baselines do not
  get; correcting that would lower its column.
- The benign control cannot fail structurally: a benign model's clean and triggered
  inputs agree on 98.5% of predictions, so any paired comparison returns ~0.5. It
  proves the probe is not reacting to the trigger itself, nothing more.
- Every existing checkpoint has `seed: null` and `git_commit: null` in its `args.json`,
  backfilled by a migration rather than recorded at training time.
