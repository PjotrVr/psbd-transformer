"""The competitor detectors' smoke run, read from the run entry that records it.

The smoke records were written under scratch/, which is untracked and has since
been cleared, so the run entry docs/runs/2026-09-10-detector-smoke.md is the
surviving record and this generator names it as its input. 3 GTSRB checkpoints,
1 benign and 2 backdoored at the middle panel rate, every registered detector,
acceptance only. The panel sweep over the clearing cells is pending and prints
as such.

    PYTHONPATH=. python scripts/paper/tab_detector_smoke.py --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from detectors import DETECTOR_NAMES, FORWARD_PASSES_PER_INPUT  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    build_parser,
    fmt,
    markdown_table_rows,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_detector_smoke.py"
SMOKE_ENTRY = os.path.join("docs", "runs", "2026-09-10-detector-smoke.md")
MODEL_ORDER = ("benign", "BadNet 5%", "Blend 5%")


def main() -> None:
    args = build_parser(__doc__).parse_args()
    rows = markdown_table_rows(SMOKE_ENTRY, "model")
    if not rows:
        raise SystemExit(f"no acceptance rows found in {SMOKE_ENTRY}")
    # The follow-up rows rerun 1 detector with 1 setting changed. They are kept
    # in the second table and excluded from the acceptance table.
    acceptance = [row for row in rows if row["model"] in MODEL_ORDER]
    followups = [row for row in rows if row["model"] not in MODEL_ORDER]

    by_detector: dict[str, dict[str, dict]] = {}
    for row in acceptance:
        by_detector.setdefault(row["detector"], {})[row["model"]] = row
    table_rows = []
    for detector in sorted(by_detector, key=lambda name: DETECTOR_NAMES.index(name) if name in DETECTOR_NAMES else 99):
        cells = by_detector[detector]
        table_rows.append(
            [
                detector,
                str(FORWARD_PASSES_PER_INPUT.get(detector, "--")),
                *[cells.get(model, {}).get("AUROC q0.25", "--") for model in MODEL_ORDER],
                *[cells.get(model, {}).get("TPR q0.01", "--") for model in MODEL_ORDER[1:]],
                cells.get("BadNet 5%", {}).get("s per input", "--"),
            ]
        )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "detector_smoke.tex"),
        generator=GENERATOR,
        inputs=[SMOKE_ENTRY],
        caption=(
            "Competitor detectors on the login-node smoke run: 3 GTSRB checkpoints at "
            "the middle panel rate, AUROC at the headline quantile on the benign "
            "reference and the 2 backdoored models, TPR at the smallest budget on "
            "the backdoored ones, forward passes per input and seconds per input. "
            "Acceptance only, the panel sweep is pending."
        ),
        label="tab:detector-smoke",
        header=[
            "detector",
            "passes",
            "AUROC benign",
            "AUROC BadNet",
            "AUROC Blend",
            "TPR@1% BadNet",
            "TPR@1% Blend",
            "s per input",
        ],
        rows=table_rows,
        align="lrrrrrrr",
    )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "detector_smoke_followups.tex"),
        generator=GENERATOR,
        inputs=[SMOKE_ENTRY],
        caption=(
            "The smoke follow-ups, each changing 1 setting against the BadNet row: "
            "the full validation split for the 2 detectors that fit per-class "
            "statistics, a larger batch for TeCo and full precision for CD-L."
        ),
        label="tab:detector-smoke-followups",
        header=["model", "detector", "n val", "AUROC", "TPR@1%", "precision", "batch"],
        rows=[
            [row["model"], row["detector"], row["n val"], row["AUROC q0.25"], row["TPR q0.01"], row["precision"], row["batch"]]
            for row in followups
        ],
        align="lllrrll",
    )

    def auroc(detector: str, model: str) -> float | None:
        text = by_detector.get(detector, {}).get(model, {}).get("AUROC q0.25")
        value = float(text) if text not in (None, "--") else None
        return value

    benign_values = [auroc(detector, "benign") for detector in by_detector]
    macros = {
        "smoke_checkpoints": (str(len(MODEL_ORDER)), "checkpoints in the smoke run"),
        "smoke_backdoored_checkpoints": (str(len(MODEL_ORDER) - 1), "backdoored checkpoints in the smoke run"),
        "smoke_followups": (str(len(followups)), "follow-up runs in the smoke entry"),
        "smoke_detectors": (str(len(by_detector)), "detectors in the smoke run"),
        "smoke_benign_auroc_min": (fmt(min(value for value in benign_values if value is not None)), "lowest benign-reference AUROC across smoke detectors"),
        "smoke_benign_auroc_max": (fmt(max(value for value in benign_values if value is not None)), "highest benign-reference AUROC across smoke detectors"),
        "smoke_ibd_psc_badnet_auroc": (fmt(auroc("ibd_psc", "BadNet 5%")), "IBD-PSC AUROC on the smoke BadNet cell"),
        "smoke_sentinet_badnet_auroc": (fmt(auroc("sentinet", "BadNet 5%")), "SentiNet AUROC on the smoke BadNet cell"),
        "smoke_beatrix_badnet_auroc": (fmt(auroc("beatrix", "BadNet 5%")), "Beatrix AUROC on the smoke BadNet cell at the truncated split"),
        "smoke_strip_badnet_auroc": (fmt(auroc("strip", "BadNet 5%")), "STRIP AUROC on the smoke BadNet cell"),
        "smoke_ted_blend_auroc": (fmt(auroc("ted", "Blend 5%")), "TED AUROC on the smoke Blend cell"),
        "smoke_confidence_blend_auroc": (fmt(auroc("confidence", "Blend 5%")), "confidence-null AUROC on the smoke Blend cell"),
        "smoke_beatrix_badnet_full_auroc": (
            next((row["AUROC q0.25"] for row in followups if row["detector"] == "beatrix"), "--"),
            "Beatrix AUROC on the smoke BadNet cell at the full validation split",
        ),
        "smoke_ibd_psc_badnet_full_auroc": (
            next((row["AUROC q0.25"] for row in followups if row["detector"] == "ibd_psc"), "--"),
            "IBD-PSC AUROC on the smoke BadNet cell at the full validation split",
        ),
        "registry_detectors": (str(len(DETECTOR_NAMES)), "detectors registered for the panel"),
        "cd_l_passes": (str(FORWARD_PASSES_PER_INPUT["cd_l"]), "forward passes per input for CD-L"),
        "teco_passes": (str(FORWARD_PASSES_PER_INPUT["teco"]), "forward passes per input for TeCo"),
        "sentinet_passes": (str(FORWARD_PASSES_PER_INPUT["sentinet"]), "forward passes per input for SentiNet"),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "detector_smoke.macros.json"), GENERATOR, [SMOKE_ENTRY], macros
    )
    print(f"detector smoke: {len(by_detector)} detectors, {len(followups)} follow-ups")


if __name__ == "__main__":
    main()
