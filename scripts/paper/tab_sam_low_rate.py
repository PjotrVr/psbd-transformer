"""Whether SAM's PSBD gain on ViT survives at low poison rates, by rate and rho.

Reads results/_experiments/sam_low_rate/sam_low_rate.json, written by
experiments/sam_low_rate/measure.py from the same matched Adam-versus-SAM
cells experiments/sam_reading/measure.py builds, re-sliced by poison rate. 1
row per (poison rate, sharpness radius): the matched pair count, mean AUROC
for the token-mask placement at the attention input and the dropout placement
after the residual add on each side with the paired delta and its bootstrap
interval, and the adaptive rule's own mean selected dropout rate on each side
for the token-mask placement, so a delta can be read against whether the rule
even chose the same disturbance on both sides.

    PYTHONPATH=. python scripts/paper/tab_sam_low_rate.py --paper-dir paper
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

GENERATOR = "scripts/paper/tab_sam_low_rate.py"
SAM_LOW_RATE_PATH = os.path.join(
    "results", "_experiments", "sam_low_rate", "sam_low_rate.json"
)
RATE_TOKENS_IN_ORDER = ("0_01", "0_05", "0_1")


def rate_rho_row(rate_label: str, rho: str, block: dict) -> list[str]:
    """1 (poison rate, rho) row: pair count, both placements' AUROC and delta, both
    sides' mean token-mask selected rate."""
    token_mask = block["placements"]["token_mask"]
    residual_dropout = block["placements"]["residual_dropout"]
    tm_ci = token_mask["delta_auroc_ci"]
    tm_low, tm_high = tm_ci if tm_ci is not None else (float("nan"), float("nan"))
    rd_ci = residual_dropout["delta_auroc_ci"]
    rd_low, rd_high = rd_ci if rd_ci is not None else (float("nan"), float("nan"))
    calibration = block["calibration"]["token_mask"]
    row = [
        rate_label,
        rho,
        str(block["n_pairs"]),
        fmt(token_mask["adam_auroc_mean"]),
        fmt(token_mask["sam_auroc_mean"]),
        fmt(token_mask["delta_auroc_mean"], signed=True),
        ci_text(tm_low, tm_high),
        fmt(residual_dropout["adam_auroc_mean"]),
        fmt(residual_dropout["sam_auroc_mean"]),
        fmt(residual_dropout["delta_auroc_mean"], signed=True),
        ci_text(rd_low, rd_high),
        fmt(calibration["adam_selected_rate_mean"]),
        fmt(calibration["sam_selected_rate_mean"]),
    ]
    return row


def build_rows(sam_low_rate: dict) -> list[list[str]]:
    """Every (poison rate, rho) row, rates in ascending order, rhos as sam_low_rate wrote them."""
    rows = []
    for rate_token in RATE_TOKENS_IN_ORDER:
        entry = sam_low_rate["by_rate"][rate_token]
        for rho, block in entry["by_rho"].items():
            rows.append(rate_rho_row(entry["label"], rho, block))
    return rows


def excluded_row(record: dict) -> list[str]:
    """1 dropped pair: which combination, both sides' ASR, why it was dropped."""
    row = [
        record["rate"],
        record["rho"],
        record["dataset_label"],
        record["attack_label"],
        record["architecture"],
        fmt(record["adam_asr"]),
        fmt(record["sam_asr"]),
    ]
    return row


def build_excluded_rows(sam_low_rate: dict) -> list[list[str]]:
    """Every pair dropped for either side falling below the ASR gate."""
    rows = [excluded_row(record) for record in sam_low_rate["excluded_asr_gate"]]
    return rows


def low_rate_coverage_macro(sam_low_rate: dict) -> tuple[str, str]:
    """How many (dataset, attack, architecture) combinations were matched at 1% and 5%."""
    n_low = len(sam_low_rate["low_rate_pairs"]["0_01"])
    n_mid = len(sam_low_rate["low_rate_pairs"]["0_05"])
    value = f"{n_low}/{n_mid}"
    meaning = (
        "matched (dataset, attack, architecture) combinations with both an Adam "
        "checkpoint and at least 1 SAM checkpoint, at 1% and 5% poisoning"
    )
    return value, meaning


def main() -> None:
    args = build_parser(__doc__).parse_args()
    sam_low_rate = load_json(SAM_LOW_RATE_PATH)
    if sam_low_rate is None:
        raise SystemExit(
            f"{SAM_LOW_RATE_PATH} does not exist, run experiments/sam_low_rate/measure.py first"
        )

    inputs = [SAM_LOW_RATE_PATH]
    rows = build_rows(sam_low_rate)

    write_table(
        path=os.path.join(args.paper_dir, "tables", "sam_low_rate.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Sharpness-aware minimisation against Adam on the matched cells at "
            "each poison rate and sharpness radius, both placements' AUROC and "
            "the adaptive rule's mean selected dropout rate for the token-mask "
            "placement."
        ),
        label="tab:sam-low-rate",
        header=[
            "Rate",
            "Rho",
            "Pairs",
            "TM AUROC Adam",
            "TM AUROC SAM",
            "TM delta",
            "TM 95% CI",
            "RD AUROC Adam",
            "RD AUROC SAM",
            "RD delta",
            "RD 95% CI",
            "TM rate Adam",
            "TM rate SAM",
        ],
        rows=rows,
        align="llrrrrlrrrlrr",
    )

    excluded_rows = build_excluded_rows(sam_low_rate)
    write_table(
        path=os.path.join(args.paper_dir, "tables", "sam_low_rate_excluded.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Pairs dropped from every sharpness-aware minimisation comparison "
            "because the Adam or the SAM checkpoint fell below the attack-success bar, "
            "so a detection loss is never read from a broken attack."
        ),
        label="tab:sam-low-rate-excluded",
        header=[
            "Rate",
            "Rho",
            "Dataset",
            "Attack",
            "Architecture",
            "Adam ASR",
            "SAM ASR",
        ],
        rows=excluded_rows,
        align="llllrrr",
    )

    macros = {}
    macros["sam_low_rate_asr_gate"] = (
        fmt(sam_low_rate["asr_clears_threshold"]),
        "the attack-success rate below which either side of a pair is dropped "
        "from every SAM-versus-Adam comparison in this table",
    )
    macros["sam_low_rate_n_excluded_asr_gate"] = (
        str(len(excluded_rows)),
        "pairs dropped from the per-rate table because the Adam or the SAM "
        "checkpoint fell below the attack-success bar",
    )
    value, meaning = low_rate_coverage_macro(sam_low_rate)
    macros["sam_low_rate_coverage"] = (value, meaning)
    for rate_token in RATE_TOKENS_IN_ORDER:
        entry = sam_low_rate["by_rate"][rate_token]
        stem = f"sam_rate_{rate_token}"
        for rho, block in entry["by_rho"].items():
            rho_stem = f"{stem}_rho_{rho.replace('.', '_')}"
            token_mask = block["placements"]["token_mask"]
            ci = token_mask["delta_auroc_ci"]
            low, high = ci if ci is not None else (float("nan"), float("nan"))
            macros[f"{rho_stem}_token_mask_delta"] = (
                fmt(token_mask["delta_auroc_mean"], signed=True),
                f"mean paired token-mask AUROC delta, SAM minus Adam, at "
                f"{entry['label']} poisoning and rho {rho}, n={block['n_pairs']}",
            )
            macros[f"{rho_stem}_token_mask_delta_ci"] = (
                ci_text(low, high),
                f"95% bootstrap interval on {rho_stem}_token_mask_delta",
            )

    write_macros(
        os.path.join(args.paper_dir, "tables", "sam_low_rate.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"sam_low_rate: {len(rows)} rows written, "
        f"{len(excluded_rows)} pairs excluded for ASR below "
        f"{sam_low_rate['asr_clears_threshold']}"
    )


if __name__ == "__main__":
    main()
