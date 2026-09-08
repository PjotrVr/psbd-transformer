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

## Known limitation

On CIFAR-100 the raw agreement fractions are ~0.01, which is chance for 100 classes, so
individual patch tokens rarely resolve the true class through the lens at all. Whether
the statistic degenerates as the class count grows is open, and Tiny (200 classes) is the
test.

## Running it

    PYTHONPATH=. python experiments/prediction_depth/measure.py --checkpoint-folder <folder>

Writes `results/<folder>/prediction_depth.json` and, unlike `cli/baselines.py`, also
`prediction_depth_scores.pt` with the per-sample tensors, so any new scoring rule is a
CPU rescore rather than another GPU pass.
