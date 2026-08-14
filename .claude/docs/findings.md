# PSBD on Vision Transformers: findings

State after the ViT CIFAR-10 grid, the baseline comparison, the SAM control, the
latent-mechanism study, and the first Swin results. The CIFAR-100, GTSRB, Tiny
ImageNet and remaining Swin sweeps are still running. Per-hypothesis detail with
full numbers is in `docs/hypothesis/` (H1 to H20).

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

**The best placement is not the one the project was built on, and it is
`before_attention`.** An earlier version of this section named `before_mlp_residual`
on the strength of 1 slice: oracle AUROC at 10% poisoning. Both qualifiers matter,
because the ranking is not stable across either. Mean over the 4 working attacks:

| slice | `before_attention` | `before_attention_norm` | `before_mlp_residual` | `pre_residual` |
|---|---|---|---|---|
| oracle, 10% poisoning | 0.962 | 0.965 | **0.969** | 0.948 |
| oracle, 5% | **0.963** | 0.956 | 0.952 | 0.948 |
| oracle, 1% | **0.909** | 0.793 | 0.824 | 0.797 |
| oracle, all rates | **0.945** | 0.871 | 0.880 | 0.861 |
| **deployable, all rates** | **0.905** | 0.812 | 0.718 | 0.722 |
| matched shift 0.6, all rates | **0.909** | 0.822 | 0.817 | 0.832 |

`before_attention` wins 5 of 6 slices, and its lead grows exactly where the problem
gets hard: +0.085 at 1% poisoning and +0.093 on the deployable metric.
`before_mlp_residual` wins only the top row.

**But naming any single placement overstates it** (H20). On a fully balanced 13x12
panel the top 4 are within 0.024 of each other and the ordering inside that group is
noise (`before_attention_norm` over `before_attention` is 0.41 SE, winning 5 of 12).
What is real is the **family**: perturbing what a sub-layer *reads* beats perturbing
what the residual stream *carries* by **+0.054, 4.0 standard errors, on 11 of 12
units**. And pre-residual against post-residual, the question this project was founded
on, is **+0.002**, both sitting 0.06 below the winning family.

**Oracle against deployable is the distinction to keep** (H19). *Oracle* picks the
dropout rate by maximizing AUROC, which reads the labels, so it is an upper bound
and not a method. *Deployable* picks it from clean validation data only. The gap is
placement-dependent, from 0.025 to 0.116, so it reorders the table rather than
shifting it. The sharpest case: `pre_residual_blocks_9_12` has the best oracle AUROC
in the project (0.997 on `blend`) and **no deployable operating point on any of 20
checkpoints**, because 4 blocks of dropout never move the clean-validation shift
ratio past 0.171 against a target of 0.7.

Restricting `pre_residual` to a band of blocks still beats applying it to all 12, on
**15 of 19** checkpoints at the deployable rate (it was 6 of 6 at the oracle), at a
third of the perturbation cost. **Blocks 5-8** is the deployable band default;
blocks 9-12, which the oracle prefers, cannot be selected at all.

**Four changes combine into a materially better defence** (H13): band placement,
fractional PSU, a retuned rate target, and a two-sided rule. Against published PSBD:
**+0.110 mean AUROC on the derivation set (14/15 wins)** and **+0.151 on two held-out
attacks (2/2)**, all with the rate chosen on clean validation data only. On `lc` the
published configuration sits at chance (0.515) and the adapted one reaches 0.786.

**And it generalizes off CIFAR-10**, where it was derived: **+0.145 over 33
backdoored checkpoints on 3 other datasets, winning 31 of 33**, including a clean
**21 of 21 on CIFAR-100** at +0.206. Its placement was also challenged directly using
[H20](../../docs/hypothesis/H20-input-side-beats-residual-adjacent.md)'s result and
defended: swapping to the winning placement family is +0.017 at 1.1 SE on the
derivation set and worse on the held-out pair.

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

SAM also **relocates** the backdoor without weakening it (H16). The peak layer moves
earlier monotonically in rho for all 5 attacks (12 to 10, 12 to 9, 11 to 8) while the
benign control stays at 8, and the top-20 TAC coordinates at rho 0.2 overlap Adam's at
0.03 to 0.08, against a chance floor of 0.014 and a split-half reproducibility ceiling
of 0.87. ASR stays 0.96 to 1.00 and clean accuracy rises about 2 points. It does not
change the backdoor's *rank*: after the final LayerNorm, removing 1 direction takes
every SAM checkpoint to ASR 0.000.

## Where the backdoor is (H16)

**A direction, not neurons.** Removing the rank-1 backdoor direction, measured after
the final LayerNorm where the head reads, takes ASR from 1.00 to 0.000 on all 4
single-target attacks for 0.02 to 0.07 clean accuracy. Zeroing TAC-ranked
*coordinates* never works, up to 300 of 768. Random rank-1 directions and the benign
model are unaffected. The backdoor is linear and simply not axis-aligned, which is
why both data-free localizers failed (they rank coordinates) and why a 2-component
PCA already separates clean from triggered at 10-NN purity 0.996 to 1.000.

**Attacks are disjoint.** Cross-attack Jaccard of top-20 TAC dimensions is 0.00 to
0.08 at a chance floor of 0.014. `badnet_a2o` and `badnet_a2a` use the identical
trigger image and overlap at exactly 0.00, so the label mapping sets the dimensions,
not the trigger's appearance.

**A methodological result that may outlast the mechanistic one.** Projecting a
direction out at a block output is the obvious ablation in a residual network and is
**not sound in a pre-norm transformer**: the LayerNorm that follows rescales whatever
survives, partly undoing the deletion. It produced a clean, monotone, benign-controlled
and entirely false result (that SAM makes the backdoor un-removable) which survived 2
rounds of follow-up before an independent measurement caught it. The rule: across a
normalization layer, scale-invariant statistics transfer and absolute ones do not.
`scripts/dropout_kills_direction/` survives the same check precisely because it
reports a standardized separation.

## Swin (first results, 11 checkpoints)

PSBD transfers to a second architecture. `badnet_a2o` reaches **0.926 deployable
AUROC** (`before_attention_norm`), and `badnet_a2a` sits at 0.470 to 0.532 across all
3 placements, reproducing H5's all-to-all failure on Swin.

At the oracle rate the 3 placements tested are indistinguishable (spread
**0.011**), which is the one clear Swin placement result. An earlier version of this
section also reported them differing by **0.11** at the deployable rate; that was
measured on 6 checkpoints and **did not survive** the sweep reaching 11, where the
difference is 0.015 at 0.3 SE. So on Swin, placement appears not to matter much
either way, and the ranking claim is withdrawn (H19).

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
