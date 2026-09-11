"""Whether SAM training helps PSBD on ViT and Swin, matched against Adam.

Reads results/_experiments/sam_reading/sam_reading.json, written by
experiments/sam_reading/measure.py from every (architecture, dataset, attack,
poison rate) combination with both an Adam checkpoint and at least 1 SAM checkpoint
(badnet_a2a excluded). 1 row per rho, with Adam as the reference row: mean
attack success and clean accuracy, mean AUROC and TPR at the q0.10 FPR budget
for the token-mask placement at the attention input, mean AUROC for the
dropout placement after the residual add, and the paired AUROC difference for
the token-mask placement against the matched Adam checkpoints, with its
bootstrap interval. Every number here is read from that file, never
recomputed from results/<folder>/psbd_metrics.json directly.

    PYTHONPATH=. python scripts/paper/tab_sam.py --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    build_parser,
    ci_text,
    fmt,
    load_json,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_sam.py"
SAM_READING_PATH = os.path.join(
    "results", "_experiments", "sam_reading", "sam_reading.json"
)

TOKEN_MASK_HEADER = "PSBD-TM AUROC"
TOKEN_MASK_TPR_HEADER = "PSBD-TM TPR@10\\%"
RESIDUAL_DROPOUT_HEADER = "PSBD-RD AUROC"
DELTA_HEADER = "PSBD-TM, SAM minus Adam"


def adam_row(sam_reading: dict) -> list[str]:
    """The Adam reference row: no rho, no delta, just the baseline the SAM rows are read against."""
    reference = sam_reading["adam_reference"]
    token_mask = reference["placements"]["token_mask"]
    residual_dropout = reference["placements"]["residual_dropout"]
    row = [
        "Adam",
        str(reference["n"]),
        fmt(reference["asr_mean"]),
        fmt(reference["clean_accuracy_mean"]),
        fmt(token_mask["auroc_mean"]),
        fmt(token_mask["tpr_at_10_mean"]),
        fmt(residual_dropout["auroc_mean"]),
        "--",
        "--",
    ]
    return row


def rho_row(entry: dict) -> list[str]:
    """1 SAM rho against the Adam reference: its own means, plus the paired token-mask delta."""
    token_mask = entry["placements"]["token_mask"]
    residual_dropout = entry["placements"]["residual_dropout"]
    ci = token_mask["delta_auroc_ci"]
    ci_low, ci_high = (ci[0], ci[1]) if ci is not None else (float("nan"), float("nan"))
    row = [
        entry["rho"],
        str(entry["n_pairs"]),
        fmt(entry["sam_asr_mean"]),
        fmt(entry["sam_clean_accuracy_mean"]),
        fmt(token_mask["sam_auroc_mean"]),
        fmt(token_mask["sam_tpr_at_10_mean"]),
        fmt(residual_dropout["sam_auroc_mean"]),
        fmt(token_mask["delta_auroc_mean"], signed=True),
        ci_text(ci_low, ci_high),
    ]
    return row


def rho_macro_stem(rho: str) -> str:
    """A rho value ("0.15") as a macro-name-safe stem ("rho_0_15")."""
    stem = f"rho_{rho.replace('.', '_')}"
    return stem


def best_and_worst_rho(sam_reading: dict) -> tuple[dict, dict]:
    """The rho rows with the largest and smallest token-mask delta against Adam."""
    scored = [
        entry
        for entry in sam_reading["aggregate_by_rho"]
        if entry["placements"]["token_mask"]["delta_auroc_mean"] is not None
    ]
    best = max(
        scored, key=lambda entry: entry["placements"]["token_mask"]["delta_auroc_mean"]
    )
    worst = min(
        scored, key=lambda entry: entry["placements"]["token_mask"]["delta_auroc_mean"]
    )
    return best, worst


def main() -> None:
    args = build_parser(__doc__).parse_args()
    sam_reading = load_json(SAM_READING_PATH)
    if sam_reading is None:
        raise SystemExit(
            f"{SAM_READING_PATH} does not exist, run experiments/sam_reading/measure.py first"
        )

    inputs = [SAM_READING_PATH]

    rows = [adam_row(sam_reading)] + [
        rho_row(entry) for entry in sam_reading["aggregate_by_rho"]
    ]

    write_table(
        path=os.path.join(args.paper_dir, "tables", "sam.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "PSBD on models trained with sharpness-aware minimisation against their Adam-trained counterparts, 1 row per sharpness radius."
        ),
        label="tab:sam",
        header=[
            "Rho",
            "Pairs",
            "ASR",
            "CA",
            TOKEN_MASK_HEADER,
            TOKEN_MASK_TPR_HEADER,
            RESIDUAL_DROPOUT_HEADER,
            DELTA_HEADER,
            "95% CI",
        ],
        rows=rows,
        align="lrrrrrrrl",
    )

    best, worst = best_and_worst_rho(sam_reading)
    macros = {
        "sam_pairs": (
            str(sam_reading["n_cells"]),
            "matched architecture, dataset, attack and poison-rate combinations with "
            "both an Adam and a SAM checkpoint, badnet_a2a excluded",
        ),
        "sam_adam_reference_auroc_token_mask": (
            fmt(
                sam_reading["adam_reference"]["placements"]["token_mask"]["auroc_mean"]
            ),
            "mean AUROC of the token-mask placement at the attention input on "
            "the Adam checkpoints in the matched grid",
        ),
        "sam_adam_reference_auroc_residual_dropout": (
            fmt(
                sam_reading["adam_reference"]["placements"]["residual_dropout"][
                    "auroc_mean"
                ]
            ),
            "mean AUROC of the dropout placement after the residual add on the "
            "Adam checkpoints in the matched grid",
        ),
        "sam_max_rho_token_mask_delta": (
            fmt(best["placements"]["token_mask"]["delta_auroc_mean"], signed=True),
            "highest paired AUROC delta of the token-mask placement, SAM minus "
            f"Adam, over any rho, reached at rho {best['rho']}, n="
            f"{best['placements']['token_mask']['n']}",
        ),
        "sam_min_rho_token_mask_delta": (
            fmt(worst["placements"]["token_mask"]["delta_auroc_mean"], signed=True),
            "lowest paired AUROC delta of the token-mask placement, SAM minus "
            f"Adam, over any rho, reached at rho {worst['rho']}, n="
            f"{worst['placements']['token_mask']['n']}",
        ),
    }
    for entry in sam_reading["aggregate_by_rho"]:
        stem = rho_macro_stem(entry["rho"])
        token_mask = entry["placements"]["token_mask"]
        residual_dropout = entry["placements"]["residual_dropout"]
        ci = token_mask["delta_auroc_ci"]
        ci_low, ci_high = (
            (ci[0], ci[1]) if ci is not None else (float("nan"), float("nan"))
        )
        macros[f"sam_{stem}_token_mask_delta"] = (
            fmt(token_mask["delta_auroc_mean"], signed=True),
            f"mean paired AUROC delta of the token-mask placement, SAM rho "
            f"{entry['rho']} minus Adam, over the {token_mask['n']} combinations "
            "both sides swept",
        )
        macros[f"sam_{stem}_token_mask_delta_ci"] = (
            ci_text(ci_low, ci_high),
            f"95% bootstrap interval on sam_{stem}_token_mask_delta",
        )
        macros[f"sam_{stem}_residual_dropout_delta"] = (
            fmt(residual_dropout["delta_auroc_mean"], signed=True),
            f"mean paired AUROC delta of the dropout placement after the "
            f"residual add, SAM rho {entry['rho']} minus Adam, over the "
            f"{residual_dropout['n']} combinations both sides swept",
        )

    write_macros(
        os.path.join(args.paper_dir, "tables", "sam.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"sam: {sam_reading['n_cells']} matched combinations, {len(sam_reading['aggregate_by_rho'])} rho rows"
    )


if __name__ == "__main__":
    main()
