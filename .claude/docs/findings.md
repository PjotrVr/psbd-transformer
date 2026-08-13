# PSBD on Vision Transformers: what the data says so far

State after phases 3a and 3b: 192 jobs, 11 placements x 16 checkpoints, CIFAR-10
ViT, full 10000-image test split, 5088 cached tensors, zero missing. The depth-band
(48) and SAM (256) sweeps are running. Hypothesis-by-hypothesis detail lives in
`docs/hypothesis/`.

## The one-paragraph version

PSBD transfers to ViT and the negative control is clean: a benign model probed with
the same trigger scores AUROC 0.502 to 0.508 at all 11 placements, while backdoored
models reach 0.89 to 0.99 on four of five attacks. But the study's founding claim did
not survive. **Pre-residual does not beat post-residual on ViT.** Given its own rate
window, post-residual wins on three of the four working attacks, and pre-residual is
only the 4th best of 11 placements anyway. What fails to transfer from ConvNets is
not the placement but **the rate grid and the rate-selection constant**. Separately,
PSBD's published *mechanism* is measurably absent on the attack it detects best, so
on transformers the method works for a reason its authors did not give.

## What holds

**PSBD works, and the control proves it is reading a backdoor.** Benign model,
probed with the same BadNet trigger: AUROC 0.506 (post-residual), 0.503
(pre-residual). Backdoored, best placement: `blend` 0.978 to 0.986, `bpp` 0.964 to
0.995, `lf` 0.887 to 0.947, `badnet_a2o` 0.686 to 0.889. The latent measurement
agrees independently: the benign model's relative backdoor-direction norm peaks at
0.089 and decays with depth, against 1.0 to 2.2 growing monotonically for every
backdoored model.

**Placement matters, but only for one trigger family.** With every placement swept
over its own operating range, the ranking over the 4 working attacks at 10% poisoning:

| rank | placement | mean AUROC |
|---|---|---|
| 1 | `before_mlp_residual` | 0.969 |
| 2 | `before_attention_norm` | 0.965 |
| 3 | `before_attention` | 0.962 |
| 4 | `pre_residual` | 0.948 |
| 5 | `before_mlp` | 0.947 |
| 6 | `post_residual` | 0.924 |
| 11 | `after_embedding` | 0.867 |

Head to head at each placement's own best rate, `post_residual` wins on `blend`
(0.989 against 0.978), `bpp` (0.991 against 0.977) and `lf` (0.969 against 0.947),
and loses only on `badnet_a2o` (0.747 against 0.889). That one exception is the
static patch trigger, whose backdoor direction reaches `[CLS]` only at layers 9 to 12
while the other three arrive by layer 8.

Two structural notes from the ranking. `pre_residual` scores **below
`before_mlp_residual` alone**, which is one of its own two components: adding the
attention-branch perturbation actively dilutes the MLP-branch one, so the
two-position combos that framed this study were never the right unit of analysis.
And the top three placements are not residual positions at all.

**All-to-all is undetectable, and the reason is structural.** `badnet_a2a` scores
0.510 at 10% poisoning, against a benign floor of 0.506, while its backdoor fires on
96% of inputs. Identical trigger to `badnet_a2o` at 0.889. A backdoor direction is a
*mean* of paired differences, and all-to-all sends each source class somewhere
different, so the mean cancels: its direction is half the magnitude of
`badnet_a2o`'s at every layer, and its clean-vs-triggered CKA at layer 12 is 0.953
against 0.419. There is no single direction because the attack implements none. This
should break any single-target method, not only PSBD.

## What broke

**"Post-residual cannot work on transformers" is wrong, and so was the founding
claim built on it.** Swept over its own window (p = 0.005 to 0.09), post-residual
gains 0.053 to 0.080 over its phase-3a scores and overtakes pre-residual on three of
four attacks. The window was found by measuring how much of the backdoor direction
survives: 0.91 retained at p=0.005, 0.44 at p=0.03, 0.015 at p=0.10 where the
standard grid *starts*. ResNet-18 has 8 residual adds and ViT-B/16 has 24, so the
same nominal rate compounds three times as hard and the inherited grid begins past
the operating point.

**The adaptive rate rule is mistuned for ViT, and fixing it is free.** PSBD selects
its rate as the smallest `p` whose clean-validation shift ratio reaches 0.8. On ViT
the optimum sits at a shift ratio near **0.70**, and the rule overshoots on **12 of
12** checkpoints for the two best-known placements, costing about 0.09 AUROC. The
shift ratio is measured on clean data only, so retargeting the constant is exactly as
defender-legal as the original. This is the most directly actionable result here.

**PSBD's mechanism does not transfer, though the method does.** The paper's story is
that under dropout clean samples collapse onto the attacker's target class. On ViT:

- The *concentration* is real. Every model collapses onto one dominant class with a
  0.27 to 0.79 share, against a 0.10 chance line. Prediction shift is not diffuse.
- The class is usually **not** the target. `blend`, the best-detected attack in the
  grid, sends 0.052 of its shifted clean predictions to `y_t` at 10% poisoning,
  *below* chance, while scoring AUROC 0.978.
- The **benign** model's dominant fallback is class 0, which is the `y_t` every
  attack in this project uses. So a shift toward class 0 cannot be attributed to
  poisoning when the un-poisoned model does it too, and the backdoored models mostly
  shift *away* from it onto other classes.

The `y_t = 0` convention, taken from the PSBD paper unchanged, confounds the
mechanism test. Fixing it needs retraining with a different target class and is the
one open question compute cannot shortcut.

## Two process notes worth keeping

**Three verdicts were overturned by their own follow-ups, and all three failed the
same way.** H3, H1, and the earlier reading of H9. In every case a placement was
compared against another placement **outside its own operating range**, and the
resulting gap was read as a property of the position. The 64-sample smoke run that
started it showed pre-residual 0.903 against post-residual 0.701 on the single most
favourable checkpoint in the grid.

The through-line is methodological: **no comparison across placements is valid at a
shared dropout rate.** The confound has now appeared three times (pre versus post,
early versus late block band, and the adaptive constant itself).

**An adversarial verification pass caught a real bug in a metric I had added.** The
captured-only detection metric restricted the backdoor side to samples the trigger
actually flipped but left the clean side as the whole pool. Captured samples are
systematically the low-confidence ones, so it compared hard images against easy ones:
it reported AUROC **0.921 for the benign control**. With both sides subset, the same
control gives 0.510. The confound was largest exactly where the metric had been
introduced to help.

**The negative control cannot fail, and that is a limitation worth stating.** A
benign model's clean and triggered inputs agree on 98.5% of predictions, so any
paired comparison of them returns ~0.5 by construction. The control proves a BadNet
patch is invisible to a model that never saw it; it does not rule out the confound it
was meant to (that PSU tracks baseline confidence). A confidence-only detector with
no dropout scores 0.514 on it, beating PSBD's 0.506. The missing control is a
confidence-matched resampling on the backdoored models.

**Three parallel read-only audits before any GPU time paid for themselves.** They
caught, among others: the benign control could not run at all
(`build_attack("benign")` raises); the shift ratio was unrecoverable from what stage
1 saved, so the paper's own rate-selection rule was uncomputable and the only
alternative would have read the labels the defence is meant to predict; and the
`badnet_a2a` positive class included triggered images the trigger never flipped.
Each would have produced plausible numbers rather than an error.

## Open

- **H6, SAM.** Not started. Deliberately after the placement question, so the rho
  sweep does not run on a placement that turns out not to matter.
- **H9, the strength confound.** Whether pre-residual still wins once both placements
  are compared at matched clean-validation shift ratio. The fine-rate jobs are the
  decider; until they land, the defensible claim is "pre beats post on the standard
  grid", not "pre beats post".
- **Block-restricted placement.** The sharpest available test of H4: restrict dropout
  to blocks 1-4 / 5-8 / 9-12 and predict, per attack, which band wins from its
  measured onset layer. Small change to `_resolve_targets`, not yet written.
- **Swin.** The registry is defined and tested, nothing has been run. Its 24 blocks
  should place the post-residual window at roughly half ViT's rate, which turns the
  depth-compounding observation into a testable formula.
