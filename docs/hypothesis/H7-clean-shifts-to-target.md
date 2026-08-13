# H7 — Clean samples under dropout shift specifically to the target class

**Status: PARTIALLY REFUTED, and the exception is the most important result in the
ledger.**

## Claim

PSBD's central mechanistic observation on ResNet-18 is that among clean samples
whose prediction changes under dropout, "almost all clean data shifts to the target
class `y_t`". This is the *explanation* the method rests on. It has never been
checked on a transformer.

## Evidence

Fraction of shifted clean predictions landing on `y_t`, CIFAR-10 ViT, pre-residual
placement, at a non-saturating rate (clean shift ratio 0.15 to 0.45, so the
histogram reflects a real bias rather than universal collapse). Chance is 0.10.

| attack | 1% | 5% | 10% | AUROC at 10% |
|---|---|---|---|---|
| `bpp` | 0.275 | **0.514** | 0.226 | 0.977 |
| `badnet_a2o` | 0.052 | 0.098 | **0.294** | 0.889 |
| `lf` | 0.125 | 0.216 | 0.247 | 0.947 |
| `blend` | 0.116 | 0.070 | **0.052** | **0.978** |
| `badnet_a2a` | 0.040 | 0.016 | 0.018 | 0.510 |

**The effect is real for `bpp`, `lf`, and `badnet_a2o` at higher poisoning** (2 to 5
times chance), and it strengthens with poison rate for the two `badnet` variants,
exactly as the neuron-bias story predicts.

**It is absent for `blend`** (0.052 at 10% poisoning, *below* chance) — and `blend`
is the **best-detected attack in the entire grid**, at AUROC 0.978 to 0.986.

## Why this is the important one

PSBD works on `blend` better than on anything else, while the mechanism PSBD's
authors give for why it works is measurably absent there. So on ViT the method and
its published explanation come apart.

Whatever is driving PSU separation for `blend` is not "clean samples collapse onto
the attacker's target class". Something else produces a large, reliable
confidence gap. Candidates, none yet tested:

- Clean samples shift to a *different* consistent class (an ImageNet-pretraining
  prior rather than the poisoned target). Directly checkable: the full histogram is
  already cached, only the argmax over classes has not been taken.
- The backdoor feature is simply far more robust to perturbation than clean
  features, and PSU measures that robustness gap without any class-collapse being
  involved. This would make PSU a *stability* measure rather than a bias measure,
  which is a different and simpler story than the paper's.

`badnet_a2a` at 0.016 to 0.040, well *below* chance, is consistent with
[H5](H5-all-to-all-breaks-psbd.md): there is no single `y_t` to collapse onto, and
detection there is also at chance. So where the mechanism is absent *and* detection
fails, the story holds together; `blend` is the case where detection succeeds
without it.

## Caveat on how this was measured

At the adaptive rate the clean shift ratio is 0.83 to 0.87, i.e. nearly everything
shifts, and the histogram then measures the model's unconditional fallback rather
than a backdoor-driven bias. The table above deliberately uses a lower rate. Both
are in `psbd_metrics.json`; reading the saturated one would have made `bpp` look
like 0.75 and inflated the whole effect.

## Independent verification in latent space

Everything above is prediction-space: it counts which class *label* the model outputs.
That is one technique, and the neuron-bias claim is really about representations, so it
was checked again with a different measurement entirely
(`scripts/shift_in_latent_space/`).

For clean samples whose prediction shifts under dropout, measure where the CLS feature
actually *moves*: the cosine between the dropout-induced displacement and the direction
from the sample's own class centroid to the target class centroid. `pre_residual`
blocks 5-8, p=0.5, layer 12, 800 samples, fp32. `cos->landed` is the control, the same
cosine but toward whichever class the prediction actually went to.

| checkpoint | landed on y_t | cos -> target | cos -> landed | excess | proj on backdoor dir |
|---|---|---|---|---|---|
| `badnet_a2o` | 0.374 | 0.4724 | 0.3986 | +0.074 | 0.622 |
| `blend` | 0.145 | 0.4301 | 0.3437 | +0.087 | 1.461 |
| `bpp` | 0.300 | 0.4630 | 0.3784 | +0.085 | 1.859 |
| `lf` | 0.214 | 0.5086 | 0.4606 | +0.048 | 1.354 |
| **benign control** | 0.295 | **0.4781** | 0.4028 | +0.075 | 0.422 |

**Clean features do drift toward the target class centroid, and the benign model drifts
just as much.** Backdoored mean 0.4685 against a benign control of **0.4781**, i.e. the
control is marginally *higher*. The excess over the landed-class control is +0.074 for
`badnet_a2o` and +0.075 for benign: identical.

So a second, independent technique reaches the same conclusion the label histogram did,
and supplies the control the label version lacked. The drift toward class 0 is a
property of the perturbation and the pretrained backbone, not of the poisoning.

Note the one place the two measurements diverge, which is informative: displacement
along the *backdoor direction* is clearly larger for backdoored models (0.62 to 1.86)
than for benign (0.42). So dropout does push clean samples along the trigger's
direction more in a poisoned model. It simply does not push them far enough, or in the
right way, to land on the target class. The mechanism is present in the representation
and absent in the decision.

## Techniques used, and not used

Used for this hypothesis: the per-pass argmax histogram (prediction space) and the
class-centroid displacement cosine plus backdoor-direction projection (latent space).

**Not used**: UMAP and the Lipschitz tooling. Both exist in `analysis/` and neither has
been run. UMAP would add a qualitative picture of the same displacement and is worth
doing; the Lipschitz tools are weight-space and data-free, so they bear on detector
design rather than on this claim.

## Reproduce

```bash
# prediction space
# results/<folder>/psbd_metrics.json, clean_shift_to_target_fraction and
# shift_target_histogram.clean, per rate

# latent space
PYTHONPATH=. python scripts/shift_in_latent_space/measure.py \
    --checkpoint-folder vit_cifar10_blend_0_1 --samples 800 --rate 0.5
```

## Follow-up: subquestions 1 and 2, answered

Taking the argmax over the cached histograms rather than only reading the `y_t`
column changes the reading substantially. Dominant shift class per model,
pre-residual, same non-saturating rate, `y_t` = 0 = airplane throughout:

| checkpoint | top-1 shift class | share | `-> y_t` |
|---|---|---|---|
| `benign` (control) | **airplane (0)** | 0.333 | n/a |
| `blend_0_01` | cat (3) | 0.386 | 0.116 |
| `blend_0_05` | cat (3) | **0.787** | 0.070 |
| `blend_0_1` | deer (4) | 0.379 | 0.052 |
| `badnet_a2o_0_05` | dog (5) | 0.581 | 0.098 |
| `badnet_a2o_0_1` | deer (4) | 0.530 | 0.294 |
| `bpp_0_05` | **airplane (0)** | 0.514 | 0.514 |
| `lf_0_05` | deer (4) | 0.496 | 0.216 |
| `badnet_a2a_0_05` | cat (3) | 0.682 | 0.016 |

**The concentration half of PSBD's mechanism holds, robustly.** Every model collapses
onto one dominant class with a share of 0.27 to 0.79, against a 0.10 chance line.
Prediction shift under dropout is emphatically not diffuse on ViT.

**The target-class half does not.** The dominant class is usually *not* `y_t`. Only
`bpp_0_05` picks it.

**And the control explains why this was hard to see.** The *benign* model's dominant
fallback is airplane, i.e. class 0, i.e. exactly the `y_t` every attack in this
project uses. So on a benign ViT, clean samples already collapse onto class 0 with
no backdoor present at all.

## What this means

The `y_t = 0` convention, inherited from the PSBD paper and used unchanged here,
**confounds the mechanism test**: a shift toward class 0 cannot be attributed to
neuron bias from poisoning when the un-poisoned model does it too.

Worse for the published story, the backdoored models mostly shift *away* from class 0
onto other classes. Poisoning appears to *displace* the pretrained model's natural
fallback rather than redirect it toward the attacker's target.

So the honest statement for ViT is: dropout-induced prediction shift concentrates on
a single class, that class is a property of the model rather than of the attacker's
target, and PSU separates clean from backdoor samples anyway. The method survives;
the explanation does not transfer.

## Subquestion that follows, and it is a design fix

Rerun a small grid with `y_t` chosen to be a class the benign model does *not* fall
back to (anything but airplane). If shift-to-target then tracks the attack, the
neuron-bias story is fine and only the class-0 convention was hiding it. If it still
does not, the story is genuinely wrong for transformers. Every checkpoint here uses
`target_label = 0`, so this needs new training and is the one experiment in this
ledger that compute cannot shortcut.

## Remaining subquestion

- Does PSU separation survive if the shift-to-target effect is regressed out? That
  isolates how much of PSBD's performance the published mechanism explains at all.
