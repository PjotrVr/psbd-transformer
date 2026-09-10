"""The 1 judging rule the synthetic sign gate applies, shared by its test and its script.

A detector is judged on the synthetic backdoor only if the fixture can exercise
it. NOT_JUDGEABLE names the ones it cannot, with the reason, so a verdict there
is never mistaken for a failure and never mistaken for a pass. Everything else is
held to MINIMUM_AUROC, and a clean population piled on 1 value is reported as
NOT EXERCISED rather than judged, since no sign can be read off a tie.
"""

import torch

from defences.decision import HEADLINE_QUANTILE, detection_report
from detectors import DETECTOR_NAMES, DetectorContext

# The synthetic backdoor is unmissable by construction, so a correctly wired
# detector clears this comfortably and an inverted detector lands near 1 minus it.
MINIMUM_AUROC = 0.60

# If the clean population is piled on 1 value there is no clean baseline to
# rank against, whatever the sign. SCALE-UP forced this: its statistic takes 6
# values and a barely trained model keeps almost every clean prediction stable.
MAXIMUM_CLEAN_CONCENTRATION = 0.9

# TeCo costs 71 forward passes per input, so the gate buys a cheaper answer with
# a reduced corruption set, visible here rather than hidden behind a flag.
CHEAP_CORRUPTIONS = ("gaussian_noise", "defocus_blur", "brightness", "contrast")

# Detectors the fixture cannot exercise, by construction of the fixture.
NOT_JUDGEABLE: dict[str, str] = {
    "sentinet": (
        "the fixture's gate reads the trigger straight from the pixels, so the "
        "backdoor never passes through the tokens Grad-CAM reads and the transplant "
        "statistic sees a random model"
    ),
    "scale_up": "6-valued statistic, almost every clean prediction tied at the maximum",
    "scale_up_data_limited": "the same 6-valued statistic after standardisation",
}


def fixture_context(model, loaders, device) -> DetectorContext:
    """The context every gate run shares, with the fixture's cheap knobs stated."""
    context = DetectorContext(
        model=model,
        device=device,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        validation_loader=loaders["validation"],
        num_classes=10,
        use_bfloat16=False,
        teco_corruptions=CHEAP_CORRUPTIONS,
        cd_l_steps=30,
        # The fixture's ViT has 2 blocks, so the ViT-B/16 default layer 9 does not exist.
        beatrix_layer=1,
        ted_reduction="cls",
        sentinet_overlays=8,
    )
    return context


def judgeable_names() -> tuple[str, ...]:
    names = tuple(name for name in DETECTOR_NAMES if name not in NOT_JUDGEABLE)
    return names


def clean_concentration(clean_scores: torch.Tensor) -> float:
    """The share of the clean population on its single most common score."""
    _values, counts = clean_scores.unique(return_counts=True)
    concentration = (counts.max() / clean_scores.numel()).item()
    return concentration


def judge(name: str, scores: dict[str, torch.Tensor]) -> tuple[str, float, float]:
    """(verdict, auroc, clean concentration) for 1 detector's scores on the 3 splits.

    verdict is "not judgeable", "not exercised", "ok", "inverted" or "weak".
    """
    report = detection_report(
        scores["validation"], scores["clean"], scores["backdoor"], HEADLINE_QUANTILE
    )
    auroc = report["auroc"]
    concentration = clean_concentration(scores["clean"])

    if name in NOT_JUDGEABLE:
        verdict = "not judgeable"
    elif concentration > MAXIMUM_CLEAN_CONCENTRATION:
        verdict = "not exercised"
    elif auroc >= MINIMUM_AUROC:
        verdict = "ok"
    elif auroc <= 1.0 - MINIMUM_AUROC:
        verdict = "inverted"
    else:
        verdict = "weak"
    return verdict, auroc, concentration
