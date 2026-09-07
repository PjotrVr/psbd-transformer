"""Every detector must point the right way, checked against a known answer.

This exists because 3 detectors were wired backwards in one sitting and none of
them raised. IBD-PSC scored 0.043 and TeCo scored 0.055 where both should have
scored above 0.94. Nothing failed, nothing warned, and the numbers looked like
numbers. A reviewer finding that in a submitted paper is the outcome this check
is built to prevent.

The convention every detector in psbd.detectors returns, and which
psbd.decision.detection_report assumes, is LOW MEANS POISONED. There is exactly
one negation per detector, at the boundary, and each says so in its docstring.
The hazard is that STRIP's raw statistic already points the right way while
SCALE-UP's, IBD-PSC's and TeCo's do not, so a reader who negates uniformly gets 3
of 4 wrong and a reader who negates none gets 1 of 4 wrong.

The test is on a model whose backdoor we installed, so a failure means the code is
wrong rather than the checkpoint being unusual. Runs on CPU in under a minute,
which is why it belongs in the test suite and not in a cluster job.

Run:
    python -m experiments.preflight.check_signs
"""

import sys

import torch

from experiments.preflight.synthetic import (
    TARGET_CLASS,
    attack_success_rate,
    build_backdoored_model,
    build_splits,
)
from psbd.decision import HEADLINE_QUANTILE, detection_report
from psbd.detectors import DETECTOR_NAMES, DetectorContext, build_detector

# The synthetic backdoor is unmissable by construction, so any detector that is
# wired correctly clears this comfortably. A sign inversion lands near 1 minus the
# true value, far below it, which is what makes the check unambiguous.
MINIMUM_AUROC = 0.60

# A detector can only be judged here if this case gives it something to detect,
# and the failure mode is subtler than a small mean difference.
#
# SCALE-UP forced this. Its statistic is a fraction over 5 amplification scales,
# so it takes 6 possible values, and a barely-trained synthetic model keeps almost
# every clean prediction stable under amplification. Clean piles up at the maximum,
# backdoor sits at the maximum too, and AUROC is 0.51 not because the sign is
# wrong but because nearly every pair is tied. The difference of means still reads
# 0.218, so a mean-based premise check passes it through and then calls it broken.
#
# The check that catches it is concentration: if the clean population is piled on
# 1 value, there is no clean baseline to rank against, whatever the sign.
MAXIMUM_CLEAN_CONCENTRATION = 0.9

# TeCo costs 71 forward passes per input against 6 for the next most expensive, so
# the check buys a cheaper answer with a reduced corruption set. That is visible
# here rather than hidden behind a flag, because a reduced set is not the setting
# any reported number should use.
CHECK_CORRUPTIONS = ("gaussian_noise", "defocus_blur", "brightness", "contrast")

SAMPLES = 256


def main():
    device = torch.device("cpu")
    model = build_backdoored_model()
    loaders = build_splits(num_samples=SAMPLES)

    # If the synthetic backdoor does not work, nothing below means anything, so
    # the premise is checked before the thing that rests on it.
    asr = attack_success_rate(model, loaders, device)
    if asr < 0.99:
        print(f"FAIL premise: synthetic backdoor ASR is {asr:.3f}, expected 1.000")
        return 1
    print(f"synthetic backdoor ASR {asr:.3f}, target class {TARGET_CLASS}")

    context = DetectorContext(
        model=model,
        device=device,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        validation_loader=loaders["validation"],
        num_classes=10,
        use_bfloat16=False,
        teco_corruptions=CHECK_CORRUPTIONS,
    )

    failures = []
    unexercised = []
    print(f"\n{'detector':24s} {'auroc':>8s}  verdict")
    for name in DETECTOR_NAMES:
        score = build_detector(name, context)
        scores = {
            split: score(model, loader, device) for split, loader in loaders.items()
        }
        report = detection_report(
            scores["validation"], scores["clean"], scores["backdoor"], HEADLINE_QUANTILE
        )
        auroc = report["auroc"]
        # What fraction of the clean population sits on its single most common
        # score. Near 1 means the ranking is almost all ties and no sign can be
        # read off it.
        clean = scores["clean"]
        _values, counts = clean.unique(return_counts=True)
        concentration = (counts.max() / clean.numel()).item()

        if concentration > MAXIMUM_CLEAN_CONCENTRATION:
            verdict = f"NOT EXERCISED, {concentration:.0%} of clean scores tied"
            unexercised.append(name)
        elif auroc >= MINIMUM_AUROC:
            verdict = "ok"
        elif auroc <= 1.0 - MINIMUM_AUROC:
            verdict = f"INVERTED, 1 - auroc = {1.0 - auroc:.3f}"
            failures.append(name)
        else:
            verdict = f"WEAK, {concentration:.0%} clean tied but auroc near chance"
            failures.append(name)
        print(f"{name:24s} {auroc:8.4f}  {verdict}")

    if failures:
        print(
            f"\nFAIL: {len(failures)} detector(s) did not point the right way: {failures}"
        )
        print("A detector below chance on a backdoor this obvious is wired backwards.")
        return 1

    judged = len(DETECTOR_NAMES) - len(unexercised)
    print(f"\nPASS: all {judged} exercised detectors point the right way")
    if unexercised:
        print(
            f"NOT JUDGED here: {unexercised}. This case does not separate for them, so "
            "their sign is unverified and must be confirmed on a real checkpoint before "
            "any number they produce is reported."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
