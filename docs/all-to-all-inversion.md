# Why PSBD fails on all-to-all, and a label-free way to know

Status: 2026-09-07. ViT evidence complete. **The ResNet-18 replication confirms the
limitation belongs to the method, not to the ViT adaptation.**

`badnet_a2a` is the only attack in the panel where detection scores *below* chance:
AUROC 0.31 to 0.47 across all four datasets and all three poison rates. That is not weak
detection, it is anti-detection, and the cause is structural rather than a tuning failure.

## It is PSBD's limitation, not the ViT port's

All-to-all was added to the authors' own code as `badnet_a2a` and run through their exact
recipe: ResNet-18, 100 epochs SGD, detection at epoch 95, 25th-percentile threshold.

| CIFAR-10, ResNet-18, 10% | TPR | FPR | CA | ASR |
|---|---:|---:|---:|---:|
| all-to-one (reproduced) | 1.000 | 0.103 | 0.847 | 1.000 |
| **all-to-all** | **0.210** | **0.192** | 0.860 | 0.842 |

TPR 0.210 against FPR 0.192: poisoned and clean data are flagged at nearly the same rate, so
there is no detection at all. The paper marks TPR below 0.8 as a failed case; this is far
below that, on the paper's own architecture.

No probe rate rescues it. Sweeping the rate by hand on the same checkpoint, TPR never
exceeds 0.216 while FPR runs 0.166 to 0.291:

| rate | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| TPR | 0.100 | 0.173 | 0.216 | 0.210 | 0.195 | 0.200 |
| FPR | 0.291 | 0.207 | 0.174 | 0.192 | 0.177 | 0.166 |
| AUROC | 0.314 | 0.461 | 0.500 | 0.541 | 0.557 | 0.561 |

The ResNet-18 inverts at low rates (0.314) and is weakly positive at high ones (0.561); the
ViT inverts at the matched rate instead. The direction of the residual differs, the outcome
does not: no rate on either architecture produces a usable detector, one-sided or two-sided.

The attack itself is healthy, with clean accuracy 0.860 and ASR 0.842. The ASR shortfall
relative to all-to-one's 1.000 is itself predicted: with a rotation the poisoned logit must
beat the source class rather than an arbitrary runner-up, so the margin is contested and
empirical risk minimisation stops as soon as it is satisfied. WaNet reports the same effect
in its own all-to-all mode, roughly 78% against 99% for the single-target version.

## The measurement

CIFAR-10, 10% poisoning, `token_mask @ before_attention_norm`, at the rate whose
clean-validation shift ratio is closest to 0.8. Shift ratio is how often the prediction
*moves* when the probe is applied.

| attack | validation | clean train | backdoor | AUROC |
|---|---:|---:|---:|---:|
| badnet_a2o | 0.806 | 0.804 | **0.005** | 0.991 |
| badnet_a2a | 0.791 | 0.786 | **0.903** | 0.440 |

Under all-to-one the backdoor samples are essentially immune to the probe: their prediction
moves 0.5% of the time against 80% for clean data. That gap *is* the method.

Under all-to-all the ordering reverses. Backdoor samples move **more** than clean ones,
0.903 against 0.786, so their PSU is higher and the ranking inverts. The same reversal holds
on every dataset:

| dataset | clean shift | backdoor shift | AUROC |
|---|---:|---:|---:|
| CIFAR-10 | 0.786 | 0.903 | 0.440 |
| CIFAR-100 | 0.868 | 0.964 | 0.377 |
| GTSRB | 0.863 | 0.923 | 0.388 |
| Tiny | 0.843 | 0.956 | 0.392 |

## Why

PSBD's premise is that the trigger creates a *constant, content-independent* shortcut. With
all-to-one, the trigger maps to a fixed class regardless of the image:

    y_poisoned = t

Under the probe the image features degrade, but the shortcut does not depend on them, so the
prediction stays pinned to the target. Clean predictions, having lost their features, drift,
and they drift *onto that same target class*: 84.4% of shifted clean predictions land on the
target on CIFAR-10, against 10% under uniform chance. The two populations separate.

All-to-all rotates each class instead:

    y_poisoned = (y + 1) mod K

The poisoned output now **depends on the source class**, so there is no content-independent
shortcut to survive the probe. The model must still read the image to know which class to
increment from, which means the backdoor pathway inherits the fragility of the clean pathway.
It is also strictly more work than clean classification (recognise the trigger, recognise the
source class, apply the rotation), so it degrades faster. Hence backdoor samples shift more
than clean ones, not less.

The shift destinations confirm there is no target to collapse onto. At the same rate, the
fraction of shifted clean predictions landing on the attacker's target class is 0.844 for
all-to-one and **0.0002** for all-to-all. Under all-to-all, clean and backdoor predictions
also drift to the *same* attractor classes (clean: 8 at 54%, 6 at 28%; backdoor: 6 at 58%,
8 at 32%), so the populations are not separable by where they move either.

## The inversion is not exploitable as-is

`direction` in `psbd_metrics.json` is set from `auroc < 0.5`, and AUROC needs labels. A
defender cannot know a cell is inverted without already knowing which samples are poisoned,
so the sign cannot simply be flipped in deployment. This is the same reason `auroc_two_sided`
was retired (audit H15).

## A label-free validity check the method already half-computes

PSBD's own rate rule maximises the gap between the clean-validation shift ratio and the shift
ratio of the whole suspect training set. Both quantities are available to a defender: clean
validation data is assumed by the threat model, and the suspect training set is the thing
being screened. No labels are involved. Only its *magnitude* is currently unused.

max over rates of (validation shift - suspect-training-set shift):

| dataset | all-to-one | all-to-all |
|---|---:|---:|
| CIFAR-10 | 0.086 | 0.004 |
| CIFAR-100 | 0.100 | 0.004 |
| GTSRB | 0.100 | **-0.010** |
| Tiny | 0.096 | 0.002 |

A 20-fold separation, computed without labels. The interpretation is exactly the premise:
the suspect set is supposed to contain samples that resist the probe *more* than clean
validation data does. When the best achievable gap is ~0, no such samples exist, and PSBD's
output on that dataset carries no information regardless of what its TPR happens to be.

**Proposed use.** Report this gap alongside every detection result and treat a near-zero
value as "the prediction-shift premise does not hold here", rather than reporting a TPR that
looks like a measurement but is noise. This costs nothing: the sweep already computes every
term.

## What this does not claim

The reversal is consistent and large, but this is one operator at one position on one
architecture family. A ResNet-18 all-to-all run using the original authors' code is in
flight; if it inverts too, the limitation belongs to the method rather than to the ViT
adaptation. `scratch/psbd-upstream/run_a2a.sh`, with all-to-all added to their BadNet as
`badnet_a2a`.
