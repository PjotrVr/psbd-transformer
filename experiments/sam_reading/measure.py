"""Does SAM training help PSBD on ViT and Swin, and does it amplify the backdoor.

H6 dropped the naive Adam-versus-SAM comparison as confounded: the SAM
checkpoints swept so far were not the same (architecture, dataset, attack,
poison rate) cells as the Adam checkpoints they were compared against, so the
sign of the difference could flip once the comparison is matched. This script
builds the matched comparison directly.

For every (architecture, dataset, attack, poison rate) cell that has both an
Adam checkpoint and at least 1 SAM checkpoint (badnet_a2a excluded, since it is
the atypical inverted attack the earlier confounded aggregate over-weighted),
it reads ASR and clean accuracy from checkpoints/<folder>/metrics.json, and,
where a sweep has reached both sides, the PSBD-TM (before_attention_norm
token_mask) and PSBD-RD (post_residual) AUROC and TPR at the q0.10 and q0.20
FPR budgets from results/<folder>/psbd_metrics.json, at the deployable
adaptive rate. Placement values are never recomputed here: they come from
cli.compare_detectors.psbd_values, the same reader every paper table uses.

Per rho, the paired AUROC difference (SAM minus Adam) is bootstrapped over the
matched cells, separately for the 2 placements. The SAM paper's own
amplification metrics (top2_tac, silhouette, clean_intra_class_variance),
already computed in experiments/sam_backdoor_effect/measure.py, are folded in
the same way wherever they are on disk, with their coverage stated plainly
since it does not span this grid.

    PYTHONPATH=. python experiments/sam_reading/measure.py
"""

import json
import math
import os
import re
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defences.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    HEADLINE_KEY,
    bootstrap_ci,
    mean_or_none,
)
from utils.provenance import current_git_commit, utc_timestamp  # noqa: E402

CHECKPOINTS_DIR = "checkpoints"
RESULTS_DIR = "results"
OUTPUT_PATH = os.path.join("results", "_experiments", "sam_reading", "sam_reading.json")
AMPLIFICATION_SOURCE = os.path.join("results", "sam_backdoor_effect.json")
AMPLIFICATION_COVERAGE_NOTE = (
    "vit only, cifar10 only, poison rate 0.1 only, attacks badnet_a2o, blend, "
    "bpp and lf (no wanet), rho in 0.05, 0.1, 0.15, 0.2. Does not span the grid "
    "this script reads for PSBD, so it is reported separately and never mixed "
    "into aggregate_by_rho."
)

# badnet_a2a is excluded: it is the atypical inverted attack that made up 13 of
# the 18 matched cells behind H6's earlier "SAM helps" correction, so it alone
# cannot settle the question either.
ATTACKS = ("badnet_a2o", "blend", "bpp", "lf", "wanet")
ARCHITECTURES = ("vit", "swin")
RATE_TOKENS = ("0_01", "0_05", "0_1")
RHO_TOKENS = ("0_05", "0_1", "0_15", "0_2", "0_5")
PLACEMENTS = {
    "token_mask": RECOMMENDED_PLACEMENT,
    "residual_dropout": PUBLISHED_PLACEMENT,
}
AMPLIFICATION_METRICS = ("top2_tac", "silhouette", "clean_intra_class_variance")


def rho_label(rho_token: str) -> str:
    """The folder's rho token as a plain decimal string, "0_15" to "0.15"."""
    label = rho_token.replace("_", ".")
    return label


def load_json(path: str) -> dict | None:
    """A JSON file, or None when it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        loaded = json.load(handle)
    return loaded


def discover_cells() -> list[dict]:
    """Every (architecture, dataset, attack, rate) cell with an Adam folder and >=1 SAM folder."""
    checkpoint_names = set(os.listdir(CHECKPOINTS_DIR))
    cells = []
    for architecture in ARCHITECTURES:
        for attack in ATTACKS:
            adam_pattern = re.compile(
                rf"^{architecture}_([a-z0-9]+)_{attack}_(0_01|0_05|0_1)$"
            )
            for name in sorted(checkpoint_names):
                match = adam_pattern.match(name)
                if match is None:
                    continue
                dataset, rate_token = match.groups()
                sam_folders = {
                    rho_token: f"{name}_sam_rho_{rho_token}"
                    for rho_token in RHO_TOKENS
                    if f"{name}_sam_rho_{rho_token}" in checkpoint_names
                }
                if not sam_folders:
                    continue
                cells.append(
                    {
                        "architecture": architecture,
                        "dataset": dataset,
                        "attack": attack,
                        "rate_token": rate_token,
                        "adam_folder": name,
                        "sam_folders": sam_folders,
                    }
                )
    return cells


def read_psbd_summary(folder: str, placement_id: str) -> dict | None:
    """AUROC and TPR@10%/20% of 1 placement at the adaptive rate, or None unswept."""
    report = load_json(os.path.join(RESULTS_DIR, folder, "psbd_metrics.json"))
    if report is None:
        return None
    block = psbd_values(report, placement_id, "adaptive")
    if block is None:
        return None
    summary = {
        "rate": block["_rate"],
        "auroc": block[HEADLINE_KEY]["auroc"],
        "tpr_at_10": block["q0.10"]["tpr"],
        "tpr_at_20": block["q0.20"]["tpr"],
    }
    return summary


def build_row(
    folder: str, architecture: str, dataset: str, attack: str, rho: float | None
) -> dict | None:
    """1 checkpoint's ASR, clean accuracy and both placements' PSBD summary. None if unevaluated."""
    metrics = load_json(os.path.join(CHECKPOINTS_DIR, folder, "metrics.json"))
    if metrics is None:
        return None
    row = {
        "architecture": architecture,
        "dataset": dataset,
        "attack": attack,
        "poison_rate": metrics["poison_rate"],
        "rho": rho,
        "folder": folder,
        "asr": metrics["asr"],
        "clean_accuracy": metrics["clean_accuracy"],
        "psbd": {
            name: read_psbd_summary(folder, placement_id)
            for name, placement_id in PLACEMENTS.items()
        },
    }
    return row


def read_rho(folder: str) -> float | None:
    """The rho a SAM checkpoint trained at, from its args.json sidecar."""
    args = load_json(os.path.join(CHECKPOINTS_DIR, folder, "args.json"))
    rho = args.get("rho") if args is not None else None
    return rho


def attach_rows(cells: list[dict]) -> list[dict]:
    """Every row this script will ever look at, and the adam/sam rows filed back onto each cell."""
    rows = []
    for cell in cells:
        adam_row = build_row(
            cell["adam_folder"],
            cell["architecture"],
            cell["dataset"],
            cell["attack"],
            None,
        )
        cell["adam_row"] = adam_row
        cell["sam_rows"] = {}
        if adam_row is not None:
            rows.append(adam_row)
        for rho_token, sam_folder in cell["sam_folders"].items():
            sam_row = build_row(
                sam_folder,
                cell["architecture"],
                cell["dataset"],
                cell["attack"],
                read_rho(sam_folder),
            )
            if sam_row is None:
                continue
            rows.append(sam_row)
            cell["sam_rows"][rho_token] = sam_row
    return rows


def ci_or_none(bounds: tuple[float, float]) -> list[float] | None:
    """A bootstrap interval as a 2-element list, None where too few pairs made it undefined."""
    low, high = bounds
    if math.isnan(low) or math.isnan(high):
        return None
    return [low, high]


def matched_pairs(cells: list[dict], rho_token: str) -> list[tuple[dict, dict]]:
    """(adam_row, sam_row) for every cell that reached this rho on both sides."""
    pairs = [
        (cell["adam_row"], cell["sam_rows"][rho_token])
        for cell in cells
        if cell["adam_row"] is not None and rho_token in cell["sam_rows"]
    ]
    return pairs


def common_coverage(pairs: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    """Pairs where every placement in PLACEMENTS was swept on both the Adam and the SAM side.

    A row's ASR, clean accuracy, AUROC and TPR are all read over this same set,
    so 1 pair count describes every column in the row rather than a different,
    looser count for whichever metric happened to need less coverage.
    """
    covered = [
        (adam, sam)
        for adam, sam in pairs
        if all(
            adam["psbd"][name] is not None and sam["psbd"][name] is not None
            for name in PLACEMENTS
        )
    ]
    return covered


def aggregate_placement(covered: list[tuple[dict, dict]], placement_name: str) -> dict:
    """SAM against Adam for 1 placement, over the common-coverage pairs of its row."""
    scored = [
        (adam["psbd"][placement_name], sam["psbd"][placement_name])
        for adam, sam in covered
    ]
    deltas = [sam["auroc"] - adam["auroc"] for adam, sam in scored]
    aggregate = {
        "n": len(scored),
        "adam_auroc_mean": mean_or_none([adam["auroc"] for adam, _ in scored]),
        "sam_auroc_mean": mean_or_none([sam["auroc"] for _, sam in scored]),
        "delta_auroc_mean": mean_or_none(deltas),
        "delta_auroc_ci": ci_or_none(
            bootstrap_ci(deltas, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
        ),
        "adam_tpr_at_10_mean": mean_or_none([adam["tpr_at_10"] for adam, _ in scored]),
        "sam_tpr_at_10_mean": mean_or_none([sam["tpr_at_10"] for _, sam in scored]),
        "adam_tpr_at_20_mean": mean_or_none([adam["tpr_at_20"] for adam, _ in scored]),
        "sam_tpr_at_20_mean": mean_or_none([sam["tpr_at_20"] for _, sam in scored]),
    }
    return aggregate


def aggregate_rho(cells: list[dict], rho_token: str) -> dict:
    """ASR, clean accuracy and both placements' PSBD comparison, SAM against Adam, at 1 rho.

    n_checkpoint_pairs is every cell that trained both sides at this rho.
    n_pairs, the count every column in the row is actually averaged over, is
    the smaller common-coverage subset that also swept both placements.
    """
    pairs = matched_pairs(cells, rho_token)
    covered = common_coverage(pairs)
    aggregate = {
        "rho": rho_label(rho_token),
        "n_checkpoint_pairs": len(pairs),
        "n_pairs": len(covered),
        "adam_asr_mean": mean_or_none([adam["asr"] for adam, _ in covered]),
        "sam_asr_mean": mean_or_none([sam["asr"] for _, sam in covered]),
        "adam_clean_accuracy_mean": mean_or_none(
            [adam["clean_accuracy"] for adam, _ in covered]
        ),
        "sam_clean_accuracy_mean": mean_or_none(
            [sam["clean_accuracy"] for _, sam in covered]
        ),
        "placements": {
            placement_name: aggregate_placement(covered, placement_name)
            for placement_name in PLACEMENTS
        },
    }
    return aggregate


def adam_reference(cells: list[dict]) -> dict:
    """The Adam baseline row: ASR, clean accuracy and PSBD summary over the cells swept for both placements."""
    all_adam_rows = [cell["adam_row"] for cell in cells if cell["adam_row"] is not None]
    adam_rows = [
        row
        for row in all_adam_rows
        if all(row["psbd"][name] is not None for name in PLACEMENTS)
    ]
    reference = {
        "n": len(adam_rows),
        "asr_mean": mean_or_none([row["asr"] for row in adam_rows]),
        "clean_accuracy_mean": mean_or_none(
            [row["clean_accuracy"] for row in adam_rows]
        ),
        # adam_rows already covers every placement in PLACEMENTS, so 1 pair count
        # and 1 subset serves every column below, the same common-coverage
        # discipline aggregate_rho applies to the SAM rows.
        "placements": {
            placement_name: {
                "n": len(adam_rows),
                "auroc_mean": mean_or_none(
                    [row["psbd"][placement_name]["auroc"] for row in adam_rows]
                ),
                "tpr_at_10_mean": mean_or_none(
                    [row["psbd"][placement_name]["tpr_at_10"] for row in adam_rows]
                ),
                "tpr_at_20_mean": mean_or_none(
                    [row["psbd"][placement_name]["tpr_at_20"] for row in adam_rows]
                ),
            }
            for placement_name in PLACEMENTS
        },
    }
    return reference


def load_amplification_rows() -> list[dict] | None:
    """The SAM paper's own amplification metrics, per checkpoint. None if never measured."""
    payload = load_json(AMPLIFICATION_SOURCE)
    rows = payload["rows"] if payload is not None else None
    return rows


def aggregate_amplification(rows: list[dict]) -> dict:
    """SAM against Adam per rho, for top2_tac, silhouette and clean_intra_class_variance.

    Matched by attack only, since the source file holds a single (architecture,
    dataset, rate) cell per attack: it never varied those 3 dimensions.
    """
    rows_by_attack: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_attack.setdefault(row["attack"], []).append(row)

    deltas_by_rho: dict[str, dict[str, list[float]]] = {}
    for attack_rows in rows_by_attack.values():
        adam_row = next((row for row in attack_rows if row["rho"] is None), None)
        if adam_row is None:
            continue
        for row in attack_rows:
            if row["rho"] is None:
                continue
            rho_key = f"{row['rho']:g}"
            bucket = deltas_by_rho.setdefault(
                rho_key, {metric: [] for metric in AMPLIFICATION_METRICS}
            )
            for metric in AMPLIFICATION_METRICS:
                bucket[metric].append(row[metric] - adam_row[metric])

    aggregate = {
        rho_key: {
            metric: {
                "n": len(deltas),
                "mean_delta": mean_or_none(deltas),
                "ci": ci_or_none(
                    bootstrap_ci(deltas, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
                ),
            }
            for metric, deltas in deltas_by_metric.items()
        }
        for rho_key, deltas_by_metric in deltas_by_rho.items()
    }
    return aggregate


def build_amplification_section() -> dict:
    """The amplification-metrics block of the output, or a plain statement that it is not on disk."""
    rows = load_amplification_rows()
    if rows is None:
        section = {
            "available": False,
            "reason": f"{AMPLIFICATION_SOURCE} does not exist",
        }
        return section
    section = {
        "available": True,
        "source": AMPLIFICATION_SOURCE,
        "coverage": AMPLIFICATION_COVERAGE_NOTE,
        "aggregate_by_rho": aggregate_amplification(rows),
    }
    return section


def delta_text(delta: float | None) -> str:
    """A signed 3-place number for the console summary, a dash where nothing paired up."""
    text = "--" if delta is None else f"{delta:+.3f}"
    return text


def print_summary(aggregate_by_rho: list[dict]) -> None:
    print(f"{'rho':>6} {'n':>4} {'TM AUROC delta':>15} {'RD AUROC delta':>15}")
    for entry in aggregate_by_rho:
        tm = entry["placements"]["token_mask"]
        rd = entry["placements"]["residual_dropout"]
        print(
            f"{entry['rho']:>6} {entry['n_pairs']:>4} "
            f"{delta_text(tm['delta_auroc_mean']):>15} "
            f"{delta_text(rd['delta_auroc_mean']):>15}"
        )


def main() -> None:
    cells = discover_cells()
    attach_rows(cells)

    rows = [cell["adam_row"] for cell in cells if cell["adam_row"] is not None]
    for cell in cells:
        rows.extend(cell["sam_rows"].values())

    present_rho_tokens = sorted(
        {rho_token for cell in cells for rho_token in cell["sam_rows"]},
        key=lambda token: float(rho_label(token)),
    )
    aggregate_by_rho = [
        aggregate_rho(cells, rho_token) for rho_token in present_rho_tokens
    ]

    payload = {
        "generator": "experiments/sam_reading/measure.py",
        "written_at": utc_timestamp(),
        "git_commit": current_git_commit(),
        "inputs": [
            "checkpoints/<folder>/metrics.json",
            "checkpoints/<folder>/args.json",
            "results/<folder>/psbd_metrics.json",
            AMPLIFICATION_SOURCE,
        ],
        "n_cells": len(cells),
        "rows": rows,
        "adam_reference": adam_reference(cells),
        "aggregate_by_rho": aggregate_by_rho,
        "amplification_metrics": build_amplification_section(),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(
        f"{len(cells)} matched (architecture, dataset, attack, rate) cells, {len(rows)} rows"
    )
    print_summary(aggregate_by_rho)
    print(f"written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
