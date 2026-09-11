"""Whole-network backdoor-direction erasure, read from experiments/whole_network_erasure.

Reads every results/_experiments/whole_network_erasure/<folder>.json written by
experiments/whole_network_erasure/measure.py, 1 file per backdoored ViT-B/16
checkpoint at 10% poisoning. Each record holds the baseline attack success rate
(ASR) and clean accuracy (CA), the layer Eq. 1 of Karayalcin et al. selects from
class-token steering (steer_cls_eq1_layer), and, at that same layer, ASR and CA
after orthogonalising every residual-writing weight matrix against the layer's
backdoor direction (whole_weights), ASR after the same edit restricted to blocks
10 and 11 (h34_blocks_10_11), and ASR after orthogonalising a random unit
direction against every weight (random_direction_whole, layer-independent). A
sweep may still be writing files, so this generator reads whatever is present
and is safe to rerun.

    PYTHONPATH=. python scripts/paper/tab_erasure.py --paper-dir paper
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    attack_label,
    build_parser,
    dataset_label,
    fmt,
    mean_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_erasure.py"
SLUG = "whole_network_erasure"


def load_reports(results_dir: str) -> list[dict]:
    """Every erasure record on disk, in whatever order glob returns them."""
    pattern = os.path.join(results_dir, "_experiments", SLUG, "*.json")
    reports = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            reports.append(json.load(handle))
    return reports


def attack_cell(report: dict) -> str:
    """The attack label, with the poison rate appended when it departs from 10%.

    Every panel checkpoint trains at 10% poisoning except the CIFAR-100 Blend
    replicate at 5%. The table carries no separate rate column, so the
    deviating row names its own rate to stay unambiguous.
    """
    label = attack_label(report["attack"])
    rate = report.get("poison_rate")
    if rate is not None and abs(rate - 0.1) > 1e-9:
        label = f"{label} ({rate:.0%})"
    return label


def table_row(report: dict) -> list[str]:
    """1 checkpoint at its Eq. 1 layer: the baseline, the all-writes edit, the blocks 10 and 11 edit and the random-direction control."""
    layer = report["steer_cls_eq1_layer"]
    all_writes = report["whole_weights"][layer]
    blocks_10_11 = report["h34_blocks_10_11"][layer]
    random_direction = report["random_direction_whole"]
    row = [
        dataset_label(report["dataset"]),
        attack_cell(report),
        fmt(report["baseline"]["asr"]),
        fmt(report["baseline"]["ca"]),
        str(layer),
        fmt(all_writes["asr"]),
        fmt(all_writes["ca"]),
        fmt(blocks_10_11["asr"]),
        fmt(random_direction["asr"]),
    ]
    return row


def main() -> None:
    args = build_parser(__doc__).parse_args()
    reports = load_reports(args.results_dir)
    if not reports:
        raise SystemExit(
            f"no records found under {args.results_dir}/_experiments/{SLUG}/"
        )
    reports.sort(key=lambda report: (report["dataset"], report["attack"]))

    inputs = [
        f"{args.results_dir}/_experiments/{SLUG}/<folder>.json ({len(reports)} checkpoints)"
    ]
    write_table(
        path=os.path.join(args.paper_dir, "tables", "erasure.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Attack success and clean accuracy after removing the backdoor "
            "direction from every weight that writes to the residual stream, "
            "on ViT-B/16 at 10\\% poisoning, or at the rate named in the attack "
            "column where it differs."
        ),
        label="tab:erasure",
        header=[
            "Dataset",
            "Attack",
            "ASR",
            "CA",
            "Layer",
            "ASR, all writes",
            "CA, all writes",
            "ASR, blocks 10 and 11",
            "ASR, random direction",
        ],
        rows=[table_row(report) for report in reports],
        align="llrrrrrrr",
    )

    asr_before = [report["baseline"]["asr"] for report in reports]
    ca_before = [report["baseline"]["ca"] for report in reports]
    asr_all_writes = [
        report["whole_weights"][report["steer_cls_eq1_layer"]]["asr"]
        for report in reports
    ]
    ca_all_writes = [
        report["whole_weights"][report["steer_cls_eq1_layer"]]["ca"]
        for report in reports
    ]
    asr_blocks_10_11 = [
        report["h34_blocks_10_11"][report["steer_cls_eq1_layer"]]["asr"]
        for report in reports
    ]
    asr_random = [report["random_direction_whole"]["asr"] for report in reports]

    macros = {
        "erasure_checkpoints": (
            str(len(reports)),
            "backdoored ViT-B/16 checkpoints in the whole-network erasure record",
        ),
        "erasure_asr_before_mean": (
            fmt(mean_or_none(asr_before)),
            "mean baseline attack success rate before erasure",
        ),
        "erasure_ca_before_mean": (
            fmt(mean_or_none(ca_before)),
            "mean baseline clean accuracy before erasure",
        ),
        "erasure_asr_after_all_writes_mean": (
            fmt(mean_or_none(asr_all_writes)),
            "mean attack success rate after orthogonalising every residual-writing "
            "weight against the Eq. 1 layer's backdoor direction",
        ),
        "erasure_asr_after_all_writes_max": (
            fmt(max(asr_all_writes)) if asr_all_writes else "--",
            "largest attack success rate remaining after the whole-network edit, "
            "across checkpoints",
        ),
        "erasure_ca_after_all_writes_mean": (
            fmt(mean_or_none(ca_all_writes)),
            "mean clean accuracy after the whole-network edit",
        ),
        "erasure_asr_after_blocks_10_11_mean": (
            fmt(mean_or_none(asr_blocks_10_11)),
            "mean attack success rate after restricting the same edit to blocks "
            "10 and 11",
        ),
        "erasure_asr_after_random_direction_mean": (
            fmt(mean_or_none(asr_random)),
            "mean attack success rate after orthogonalising a random unit "
            "direction against every residual-writing weight",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "erasure.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(f"erasure: {len(reports)} checkpoints")


if __name__ == "__main__":
    main()
