# Where and when is the answer decided? (CLS versus all tokens)

## Question

Every detector in the PSBD family perturbs the network and measures how far the
prediction moves (H28). This asks something algebraically different: not how robust the
prediction is, but **at what depth, and from which patches, it was already decided**.

A ViT hands over a per-token latent state for free, and H32 already showed that per-token
direction norms localize a trigger (BadNet's top token sits at (13,13), the trigger's own
position). PSBD reads none of that: it collapses everything to one scalar from the CLS
prediction. The question is whether the discarded structure carries detection signal.

## Method

A logit lens: read every encoder block through the network's **own** final LayerNorm and
head,

    g(h) = head(LN_final(h))

so every depth is read at the scale the head was trained for, and the statistic is an
argmax, which is invariant to positive rescaling. That matters because reading
intermediate activations with a scale-sensitive statistic is failure mode 3 in the
ledger, and it once produced a clean, monotone, benign-controlled and entirely false
result that survived 2 rounds of follow-up.

Four raw statistics per sample, one forward pass, no fitting and no calibration against
the split being scored:

| statistic | reads | a priori direction |
|---|---|---|
| `depth_cls` | CLS only | a trigger is a strong simple feature, so it resolves early: LOW depth |
| `depth_token_min` | earliest any patch locks on | a localized trigger commits almost immediately: LOW |
| `token_agreement` | fraction of patches agreeing with the final answer | see below |
| `token_max_prob` | largest final-class probability any single patch assigns | see below |

Plus a `_dev` variant of each: distance from the clean-validation distribution in either
direction, `-2 |rank_val(s) - 0.5|`.

## Result so far

`vit_cifar10_badnet_a2o_0_1`, against PSBD at its own best placement:

| | AUROC | TPR@1%FPR | TPR@5%FPR | forward passes |
|---|---|---|---|---|
| PSBD (`token_mask @ before_attention_norm`) | 0.961 | **0.000** | 0.710 | 30 |
| `token_agreement` | **0.982** | **0.287** | **0.929** | 1 |
| `depth_cls` (CLS only) | 0.826 | 0.000 | - | 1 |

Benign control 0.500 on both token statistics, 0.473 on `depth_cls`. So all-token beats
CLS-only by 0.156 AUROC, and beats PSBD at 30x lower cost on the operating point PSBD
fails at.

**And then the direction flips.** On a global trigger it reverses:

| cell | trigger | clean agreement | backdoor agreement | AUROC |
|---|---|---|---|---|
| cifar10 badnet_a2o | local patch | 0.429 | **0.064** | 0.982 |
| cifar100 adaptive_blend | global blend | 0.019 | **0.222** | 0.213 |

This is mechanical, not noise. A patch trigger drives the answer from 1 or 2 patches
while the other 194 keep voting their own class, so agreement falls. A global blend
paints every patch with the trigger, so every patch votes the target and agreement
rises. Both are equally far from clean, in opposite directions.

That is what the `_dev` variants are for, and it is **not** the two-sided rule H15
retired. H15's rule picked a tail by reading the AUROC, which needs the poison labels the
detector exists to predict. This ranks against the clean validation split, which the
threat model already grants, and applies 1 fixed rule everywhere: far from clean in
either direction is suspicious. The threshold stays a quantile of the validation
deviation, so the false-positive budget is set exactly as before.

## The headline: same AUROC, but it works at a budget PSBD cannot

Judged the way it should be judged. BadNet and Blend excluded, because BadNet implants at
ASR 0.997 to 1.000 everywhere and the ledger already records that "its detection behaviour
is least like the others". Poison rates 1% and 5% only. Attacks that actually implanted
(ASR >= 0.5) only. Paired against PSBD on the same cells, same splits, same pairing.

**n = 32 (adaptive_blend, WaNet, LC, SIG, LF, Bpp), 1% and 5%, full 217-cell panel:**

| | AUROC | **TPR at 1% FPR** | forward passes |
|---|---|---|---|
| `depth_soft` | 0.831 | **0.675** | **1** |
| PSBD | 0.821 | **0.428** | 30 |
| delta | +0.011, CI [-0.028, +0.044] | **+0.248, CI [+0.090, +0.402]**, wins 24/32 | |

(An earlier read on 31 of these cells gave +0.284; the full panel gives +0.248. The
conclusion is unchanged and the interval still excludes zero.)

**AUROC is a tie and the confidence interval says so.** The result is entirely at the
operating point: at a 1% false-positive budget it catches 70% where PSBD catches 41%,
winning 23 of 31 cells, from a thirtieth of the compute.

That is not a coincidence. `experiments/low_fpr_audit/` measured that 13 of 56 cells have
AUROC >= 0.85 with TPR@1%FPR < 0.05, i.e. PSBD's binding failure is the extreme tail rather
than the average. This lands exactly there.

By attack, TPR at 1% FPR:

| attack | n | `depth_soft` | PSBD |
|---|---|---|---|
| LF | 8 | **0.884** | 0.507 |
| Bpp | 8 | **0.859** | 0.524 |
| WaNet | 4 | **0.769** | 0.141 |
| adaptive_blend | 8 | 0.465 | **0.521** |
| LC | 3 | **0.269** | 0.145 |
| SIG | 1 | 0.062 | **0.280** |

**It LOSES on adaptive_blend and on SIG**, which are the 2 hardest cases in the panel and
the ones a defence most needs to win. On the full panel adaptive_blend flips from a tie to
a loss. The wins are WaNet, Bpp and LF, and they are large. So the honest reading is that
this buys a large low-FPR gain on the mid-difficulty attacks and nothing on the hardest.

**Benign controls, all 4 datasets:** 0.481, 0.481, 0.501, 0.498, mean **0.491**, with TPR
at the 1% budget reading 0.010 to 0.012, i.e. exactly nominal. The signal is not an
artifact of applying a trigger.

### What has to be settled before this is a claim

- **Prior art.** TED (S&P 2024) and TED++ do layer-wise trajectory analysis with clean-only
  calibration for backdoor input detection, above 0.95 AUROC, on ResNets. That is the number
  to beat, not a citation. Orion (IJCAI 2023) uses internal-readout-disagrees-with-final-
  answer as a per-input poisoned score.
- The AUROC delta is not significant. The claim is a low-FPR claim and must be stated as one.
- Mostly single seed. A few cells have `_seed_1` and `_seed_2` replicates and those should
  set the pre-registered n.

## The earlier verdict, retained: on AUROC alone it loses

BadNet is the wrong thing to judge this on. It implants at ASR 0.997 to 1.000 everywhere
and the ledger already records that "its detection behaviour is least like the others".
Restricted to the attacks that are actually hard (adaptive_blend, WaNet, LC, SIG, LF, Bpp,
TaCT), over 41 cells at every poison rate:

| | mean AUROC | inversions |
|---|---|---|
| `depth_cls` | **0.720** | 10/41 |
| PSBD | **0.796** | |

So the earlier "0.983 against PSBD's 0.974 on GTSRB" was GTSRB **including BadNet and
Blend at 10%**, and it does not survive the cut. The single worst case is the attack
designed to defeat detectors:

| cell | `depth_cls` | PSBD |
|---|---|---|
| `vit_gtsrb_adaptive_blend_0_05` | 0.695 | **0.986** |
| `vit_cifar10_adaptive_blend_0_1` | 0.643 | **0.924** |
| `vit_gtsrb_lc_0_1` | 0.421 | 0.522 |
| `vit_gtsrb_sig_0_1` | 0.427 | 0.451 |
| `vit_cifar10_tact_0_05` | 0.034 | - |

**Where it genuinely wins is WaNet**, and only WaNet:

| cell | `depth_cls` | PSBD |
|---|---|---|
| `vit_cifar10_wanet_0_1` | **0.848** | 0.408 |
| `vit_cifar100_wanet_0_1` | **0.817** | 0.702 |
| `vit_gtsrb_wanet_0_1` | **0.933** | 0.854 |

WaNet is a warp, and H27 already recorded that `token_mask` is worst on exactly that
attack (0.747 against 0.985 on the patch trigger). A depth statistic reading the residual
stream appears to catch what a token-space perturbation misses. That is a narrow, specific
and checkable claim, and it is the only one this experiment currently supports.

## The earlier CIFAR-10 result, retained as a caution

## The panel verdict on the token statistic

The CIFAR-10 cell above is not representative, and GTSRB (43 classes) shows why. Same
protocol, 10% poisoning, `token_agreement` read in the direction its own mechanism
predicts for that class count:

| GTSRB group | n | token agreement | PSBD |
|---|---|---|---|
| dirty-label (`all_to_one`) | 6 | 0.948 | **0.974** |
| clean-label (`sig`, `lc`) | 2 | 0.496 | 0.487 |
| all-to-all | 1 | 0.449 | 0.306 |
| benign control | 1 | **0.503** | 0.500 |

Per cell on the dirty-label group: blend 0.987, bpp 0.983, adaptive_blend 0.975, lf 0.956,
wanet 0.947, badnet 0.838. Real signal, clean benign control, and still **below PSBD**,
whose TPR at 1% FPR on those same cells is 0.966 to 1.000.

It also fails exactly where PSBD fails: the 2 clean-label attacks and all-to-all. A
clean-label trigger reinforces the true class rather than overriding it, so the agreement
never jumps, which is the mechanism working as stated and predicting its own failure.

**And the sign is not stable.** It depends on the class count, not only on the trigger's
spatial extent:

| cell | raw AUROC | clean -> backdoor agreement |
|---|---|---|
| cifar10 badnet_a2o | 0.982 | 0.429 -> 0.064 |
| gtsrb badnet_a2o | 0.162 | 0.071 -> 0.033 |

With 10 classes a clean patch token often resolves the true class (agreement 0.429), so a
local trigger LOWERS agreement. With 43 classes clean agreement is already near chance
(1/43), so the confident trigger patches RAISE it. The same attack inverts between
datasets. A sign router built on the suspect-pool-minus-validation deviation recovers 7 of
9 cells (0.423 raw, 0.577 flipped, 0.731 routed) but fails wherever that deviation is
near zero, which is exactly the GTSRB badnet case.

## Where it does win

Only where PSBD has a low-FPR collapse. On `vit_cifar10_badnet_a2o_0_1`, PSBD reads AUROC
0.961 with TPR **0.000** at 1% FPR, and token agreement reads 0.982 with **0.287**. That
is 1 of the 6 cells identified this session where PSBD has AUROC >= 0.85 and TPR@1%FPR
< 0.05. Whether the per-token reading is systematically strong on exactly those cells is
the one open question worth the GPU time, and the 217-cell panel answers it.

## Methodological note

Pilot on GTSRB, not CIFAR-10. The 10-class result was flattering by a wide margin, and the
statistic's chance level is 1/K, so a class-count-sensitive statistic looks far stronger
there than it is.

## Running it

    PYTHONPATH=. python experiments/prediction_depth/measure.py --checkpoint-folder <folder>

Writes `results/<folder>/prediction_depth.json` and, unlike `cli/baselines.py`, also
`prediction_depth_scores.pt` with the per-sample tensors, so any new scoring rule is a
CPU rescore rather than another GPU pass.
