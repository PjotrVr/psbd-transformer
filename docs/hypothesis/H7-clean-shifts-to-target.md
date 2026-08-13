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

## Reproduce

`results/<folder>/psbd_metrics.json`, keys `clean_shift_to_target_fraction` and
`shift_target_histogram.clean`, per rate.

## Subquestions, in priority order

1. **Which class do `blend`'s clean samples actually shift to?** One argmax over a
   histogram that is already on disk. If it is a single consistent non-target class,
   that is a publishable finding about pretrained-backbone priors.
2. Does the benign control shift to that same class? If yes, the fallback is a
   property of the pretrained model, not of the poisoning, and the neuron-bias story
   needs restating for transformers.
3. Does PSU separation survive if the shift-to-target effect is regressed out? That
   would isolate how much of PSBD's performance the published mechanism explains.
