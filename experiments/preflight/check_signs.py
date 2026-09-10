"""Every detector must point the right way, checked against a known answer.

The convention every detector in detectors returns, and which
defences.decision.detection_report assumes, is LOW MEANS POISONED. There is
exactly 1 negation per detector, at the boundary, and each says so in its
docstring. The hazard is that some raw statistics already point the right way
while others do not, so a reader who negates uniformly gets some wrong and a
reader who negates none gets the others wrong, and nothing raises either way.

The test is on a model whose backdoor we installed, so a failure means the code
is wrong rather than the checkpoint being unusual. The judging rule lives in
experiments.preflight.gate and tests/test_preflight.py applies the same one.

Run:
    python -m experiments.preflight.check_signs
"""

import sys

import torch

from detectors import DETECTOR_NAMES, build_detector
from experiments.preflight.gate import NOT_JUDGEABLE, fixture_context, judge
from experiments.preflight.synthetic import (
    TARGET_CLASS,
    attack_success_rate,
    build_backdoored_model,
    build_splits,
)

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

    context = fixture_context(model, loaders, device)
    failures = []
    unexercised = []
    print(f"\n{'detector':24s} {'auroc':>8s}  verdict")
    for name in DETECTOR_NAMES:
        score = build_detector(name, context)
        scores = {
            split: score(model, loader, device) for split, loader in loaders.items()
        }
        verdict, auroc, concentration = judge(name, scores)
        if verdict == "not judgeable":
            text = f"NOT JUDGED by construction, {NOT_JUDGEABLE[name]}"
        elif verdict == "not exercised":
            text = f"NOT EXERCISED, {concentration:.0%} of clean scores tied"
            unexercised.append(name)
        elif verdict == "ok":
            text = "ok"
        elif verdict == "inverted":
            text = f"INVERTED, 1 - auroc = {1.0 - auroc:.3f}"
            failures.append(name)
        else:
            text = f"WEAK, {concentration:.0%} clean tied but auroc near chance"
            failures.append(name)
        print(f"{name:24s} {auroc:8.4f}  {text}")

    if failures:
        print(
            f"\nFAIL: {len(failures)} detector(s) did not point the right way: {failures}"
        )
        print("A detector below chance on a backdoor this obvious is wired backwards.")
        return 1

    judged = len(DETECTOR_NAMES) - len(unexercised) - len(NOT_JUDGEABLE)
    print(f"\nPASS: all {judged} exercised detectors point the right way")
    skipped = sorted(set(unexercised) | set(NOT_JUDGEABLE))
    if skipped:
        print(
            f"NOT JUDGED here: {skipped}. Their sign is confirmed on a real checkpoint "
            "before any number they produce is reported."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
