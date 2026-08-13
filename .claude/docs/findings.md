# PSBD on Vision Transformers: what the data says so far

State as of the completed phase 3a (32 jobs, CIFAR-10 ViT, full test split).
Phase 3b (144 single-position jobs) and the fine-rate follow-up (16 jobs) are
running. Hypothesis-by-hypothesis detail lives in `docs/hypothesis/`.

## The one-paragraph version

PSBD transfers to ViT and the negative control is clean: a benign model probed with
the same trigger scores AUROC 0.506, while backdoored models reach 0.89 to 0.99 on
three of five attacks. But two of the study's founding assumptions did not survive
contact with the full grid. Post-residual dropout does **not** fail on ViT, it just
has an operating window an order of magnitude below the rate grid inherited from the
ConvNet paper. And PSBD's published *mechanism* is measurably absent on the attack it
detects best, so on transformers the method works for a reason its authors did not
give.

## What holds

**PSBD works, and the control proves it is reading a backdoor.** Benign model,
probed with the same BadNet trigger: AUROC 0.506 (post-residual), 0.503
(pre-residual). Backdoored, best placement: `blend` 0.978 to 0.986, `bpp` 0.964 to
0.995, `lf` 0.887 to 0.947, `badnet_a2o` 0.686 to 0.889. The latent measurement
agrees independently: the benign model's relative backdoor-direction norm peaks at
0.089 and decays with depth, against 1.0 to 2.2 growing monotonically for every
backdoored model.

**Placement matters, but only for one trigger family.** The pre-residual advantage
tracks where the trigger's backdoor direction reaches the `[CLS]` token:

| attack | direction onset | pre-post AUROC gap |
|---|---|---|
| `blend` | layer 5 | +0.055 |
| `bpp` | layer 6 | +0.024 |
| `lf` | layer 8 | +0.030 |
| `badnet_a2o` | layer 9 | **+0.226** |

Attacks whose direction is established by layer 8 are caught by either placement.
The static patch trigger, which only arrives at `[CLS]` in the last few blocks, is
where post-residual collapses to near chance.

**All-to-all is undetectable, and the reason is structural.** `badnet_a2a` scores
0.510 at 10% poisoning, against a benign floor of 0.506, while its backdoor fires on
96% of inputs. Identical trigger to `badnet_a2o` at 0.889. A backdoor direction is a
*mean* of paired differences, and all-to-all sends each source class somewhere
different, so the mean cancels: its direction is half the magnitude of
`badnet_a2o`'s at every layer, and its clean-vs-triggered CKA at layer 12 is 0.953
against 0.419. There is no single direction because the attack implements none. This
should break any single-target method, not only PSBD.

## What broke

**"Post-residual cannot work on transformers" is wrong.** It reaches 0.92 to 0.98 on
`blend` and `bpp`. Its usable window is p = 0.01 to 0.03, measured by how much of the
backdoor direction survives: 0.91 retained at p=0.005, 0.44 at p=0.03, 0.015 at
p=0.10 where the standard grid *starts*. ResNet-18 has 8 residual adds and ViT-B/16
has 24, so the compounding differs by an order of magnitude and the ConvNet rate grid
does not transfer. Every post-residual number on record was measured at or past the
edge of its own range; 16 fine-rate jobs are fixing that.

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

**The 64-sample smoke run was misleading and nearly became a finding.** It showed
pre-residual 0.903 against post-residual 0.701 on one checkpoint, and both H1 and H3
were written up as SUPPORTED on that basis. The full grid cut the effect to 0.02-0.06
on three of four attacks and refuted H3 outright. That checkpoint was the most
favourable case in the grid. Preliminary verdicts are now labelled as such.

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
