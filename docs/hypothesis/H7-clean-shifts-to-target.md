# H7 — Clean samples under dropout shift specifically to the target class

**Status: OPEN**

## Claim

The PSBD paper's central mechanistic observation on ResNet-18 is that among clean
samples whose prediction changes under dropout, "almost all clean data shifts to
the target class `y_t`". This claim is what makes the whole method make sense.
It has never been checked on a transformer.

## Prediction

For an all-to-one attack, the histogram of classes that shifted clean predictions
land on has a large spike at `y_t` (class 0 here), well above what the class prior
would give (0.1 on CIFAR-10). Refuted if the histogram is flat, or spikes on some
other class.

## Why it is interesting

It is the most direct test available of *why* PSBD works, and it is nearly free:
the per-pass argmax classes are already cached, so it needs no GPU.

The paper itself reports a caveat worth checking: on Tiny ImageNet the shift
classes "exhibit a predominant inclination towards a certain class rather than the
target class". So the effect is already known to be dataset-dependent in the
original work. If ViT behaves like their Tiny ImageNet case rather than their
CIFAR-10 case, that is a real finding about what the pretrained backbone
contributes.

A pretrained ViT is a genuinely different starting point from a
trained-from-scratch ResNet: it arrives with strong ImageNet priors, so its
fallback class under heavy dropout might be dictated by pretraining rather than by
the poisoning. That would be a clean explanation for a null result here.

## Evidence

Pending, but the machinery is in place: `shift_target_histogram` and
`clean_shift_to_target_fraction` are computed for every (placement, rate) and
written into `psbd_metrics.json`.

Note the interaction with [H3](H3-why-post-residual-fails.md): at a saturating
placement, essentially everything shifts, so the histogram measures the model's
unconditional fallback rather than any backdoor-driven bias. This hypothesis is
only meaningful at placements and rates with a non-degenerate shift ratio.

## Subquestions

1. Is the fallback class the same for a benign model? If `vit_cifar10_benign`
   also collapses onto class 0, the spike is not about the backdoor at all. This
   is the control that makes or breaks the reading.
2. Does the fallback class match the ImageNet-pretrained prior rather than the
   poisoned target?
3. For all-to-all, does each source class shift to its own `(y+1)` target, or do
   all classes shift to one arbitrary class? Directly relevant to [H5](H5-all-to-all-breaks-psbd.md).
