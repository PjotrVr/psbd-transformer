"""Seed replicates: how much the headline comparison moves when a cell is retrained.

Every panel cell is seed 0. The seed replicates under results/ carry the
recommended and the published placement only, on CIFAR-100 and Tiny, at seeds 1
and 2, so this reads the 3 seeds of every replicated cell whose seed-0 folder is
a clearing panel cell and reports the per-cell spread of each placement's AUROC
and of their paired gain. The number a reader wants is the seed standard
deviation, since it says whether a single-seed effect of a given size could be
retraining noise.

    PYTHONPATH=. python scripts/paper/tab_seeds.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defences.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    build_parser_with_checkpoints,
    clearing_cells,
    fmt,
    load_args_json,
    load_coverage,
    load_psbd_metrics,
    mean_or_none,
    std_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_seeds.py"
REPLICATE_SEEDS = (1, 2)
RULE = "adaptive"


def seed_folder(folder: str, seed: int) -> str:
    name = folder if seed == 0 else f"{folder}_seed_{seed}"
    return name


def auroc_at(results_dir: str, folder: str, placement: str) -> float | None:
    report = load_psbd_metrics(results_dir, folder)
    block = psbd_values(report, placement, RULE)
    if block is None or HEADLINE_KEY not in block:
        return None
    value = block[HEADLINE_KEY]["auroc"]
    return value


def replicated_cells(results_dir: str, cells: list[dict]) -> list[dict]:
    """Clearing cells whose every replicate carries both placements at the rule."""
    replicated = []
    for cell in cells:
        seeds = {}
        for seed in (0, *REPLICATE_SEEDS):
            folder = seed_folder(cell["folder_name"], seed)
            recommended = auroc_at(results_dir, folder, RECOMMENDED_PLACEMENT)
            published = auroc_at(results_dir, folder, PUBLISHED_PLACEMENT)
            if recommended is None or published is None:
                break
            seeds[seed] = {"recommended": recommended, "published": published}
        if len(seeds) == 1 + len(REPLICATE_SEEDS):
            replicated.append({**cell, "seeds": seeds})
    return replicated


def replicate_asr(checkpoints_dir: str, folder: str) -> list[float]:
    values = []
    for seed in REPLICATE_SEEDS:
        sidecar = load_args_json(checkpoints_dir, seed_folder(folder, seed)) or {}
        if sidecar.get("asr") is not None:
            values.append(float(sidecar["asr"]))
    return values


def main() -> None:
    args = build_parser_with_checkpoints(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    cells = replicated_cells(args.results_dir, clearing_cells(load_coverage(args.results_dir)))
    inputs = [
        coverage_path,
        f"{args.results_dir}/<folder>[_seed_N]/psbd_metrics.json ({len(cells)} cells)",
        f"{args.checkpoints_dir}/<folder>_seed_N/args.json",
    ]

    rows = []
    recommended_sds, published_sds, gain_sds, gain_means = [], [], [], []
    sign_flips = 0
    for cell in cells:
        recommended = [cell["seeds"][seed]["recommended"] for seed in sorted(cell["seeds"])]
        published = [cell["seeds"][seed]["published"] for seed in sorted(cell["seeds"])]
        gains = [rec - pub for rec, pub in zip(recommended, published)]
        recommended_sds.append(std_or_none(recommended))
        published_sds.append(std_or_none(published))
        gain_sds.append(std_or_none(gains))
        gain_means.append(mean_or_none(gains))
        if min(gains) < 0 < max(gains):
            sign_flips += 1
        asr = replicate_asr(args.checkpoints_dir, cell["folder_name"])
        rows.append(
            [
                cell["dataset"],
                cell["attack"],
                f"{cell['poison_rate']:g}",
                fmt(min(asr)) if asr else "--",
                " / ".join(fmt(value) for value in recommended),
                fmt(std_or_none(recommended)),
                " / ".join(fmt(value) for value in published),
                fmt(std_or_none(published)),
                fmt(mean_or_none(gains), signed=True),
                fmt(std_or_none(gains)),
            ]
        )

    write_table(
        path=os.path.join(args.paper_dir, "tables", "seeds.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Seed replicates of the headline comparison at the adaptive rule and "
            "the headline quantile: AUROC of the recommended and the published "
            "placement at seeds 0, 1 and 2 of every replicated clearing cell, the "
            "sample standard deviation over seeds, and the paired gain's mean and "
            "spread. The ASR column is the lowest replicate's attack success."
        ),
        label="tab:seeds",
        header=[
            "dataset",
            "attack",
            "rate",
            "min ASR",
            "recommended s0 / s1 / s2",
            "sd",
            "published s0 / s1 / s2",
            "sd",
            "gain mean",
            "gain sd",
        ],
        rows=rows,
        align="lllrlrlrrr",
    )

    macros = {
        "seeds_per_cell": (str(1 + len(REPLICATE_SEEDS)), "training seeds per replicated cell"),
        "seed_cells": (str(len(cells)), "clearing cells with 3 training seeds at both placements"),
        "seed_sd_recommended": (
            fmt(mean_or_none([value for value in recommended_sds if value is not None])),
            "mean over replicated cells of the recommended placement's AUROC "
            "standard deviation across 3 seeds",
        ),
        "seed_sd_published": (
            fmt(mean_or_none([value for value in published_sds if value is not None])),
            "mean over replicated cells of the published placement's AUROC "
            "standard deviation across 3 seeds",
        ),
        "seed_sd_gain": (
            fmt(mean_or_none([value for value in gain_sds if value is not None])),
            "mean over replicated cells of the paired gain's standard deviation across 3 seeds",
        ),
        "seed_sd_gain_max": (
            fmt(max(value for value in gain_sds if value is not None)) if gain_sds else "--",
            "largest per-cell standard deviation of the paired gain across 3 seeds",
        ),
        "seed_gain_mean": (
            fmt(mean_or_none([value for value in gain_means if value is not None]), signed=True),
            "mean over replicated cells of the seed-averaged paired gain, "
            "recommended minus published at the adaptive rule",
        ),
        "seed_gain_sign_flips": (
            str(sign_flips),
            "replicated cells where the paired gain changes sign across seeds",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "seeds.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"seeds: {len(cells)} cells, sd recommended {macros['seed_sd_recommended'][0]}, "
        f"gain {macros['seed_gain_mean'][0]}, flips {sign_flips}"
    )


if __name__ == "__main__":
    main()
