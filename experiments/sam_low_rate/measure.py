"""Does SAM's PSBD gain survive at low poison rates, and is it separation or calibration.

experiments/sam_reading/measure.py already built the matched Adam-versus-SAM
comparison and found a small token-mask gain and a residual-dropout loss, pooled
over every poison rate together. This script splits that same matched grid by
poison rate, to answer 3 further questions the pooled number cannot: whether the
gain survives at 1% and 5% poisoning, where a real attacker sits, whether SAM
changes the adaptive rule's calibration (the rate it selects and the
clean-validation shift ratio it reaches at each nominal rate) rather than the
separation at a fixed disturbance, and whether SAM's advantage holds across the
whole disturbance ladder or only at 1 rung of it.

Cells, rows and the matched-pair machinery are imported unchanged from
experiments/sam_reading/measure.py (discover_cells, attach_rows, matched_pairs,
common_coverage, ci_or_none): this script never recomputes what combinations
exist or which psbd_metrics.json a placement's adaptive-rate summary comes
from, it only re-slices the same matched pairs by poison rate and reads the
full per-rate ladder those functions do not carry.

SAM sometimes breaks implantation rather than merely changing detectability
(WaNet on CIFAR-100 at 5% falls from ASR 0.650 under Adam to 0.007 under SAM),
so every paired comparison here drops a pair where either side's ASR is below
ASR_CLEARS_THRESHOLD before computing a delta: a "loss" that is really a broken
attack is not evidence about detection at all. The dropped pairs are counted
per (rate, rho) and listed in full under excluded_asr_gate, so a reader can
see that the reported deltas never hide a SAM-broke-the-attack case as a
detection result.

    PYTHONPATH=. python experiments/sam_low_rate/measure.py
"""

import json
import os
import sys

sys.path.insert(0, os.getcwd())

from defences.decision import HEADLINE_QUANTILE  # noqa: E402
from experiments.sam_reading.measure import (  # noqa: E402
    ATTACKS,
    PLACEMENTS,
    RATE_TOKENS,
    RESULTS_DIR,
    attach_rows,
    ci_or_none,
    common_coverage,
    discover_cells,
    load_json,
    matched_pairs,
    rho_label,
)
from scripts.paper._common import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    attack_label,
    bootstrap_ci,
    dataset_label,
    mean_or_none,
)
from utils.provenance import current_git_commit, utc_timestamp  # noqa: E402

OUTPUT_PATH = os.path.join(
    "results", "_experiments", "sam_low_rate", "sam_low_rate.json"
)
RATE_TOKEN_LABELS = {"0_01": "1%", "0_05": "5%", "0_1": "10%"}
HEADLINE_KEY = f"q{HEADLINE_QUANTILE:.2f}"

# The ladder rung the whole-curve comparison (part 3) is read at: rho 0.1 held
# the tightest bootstrap interval in experiments/sam_reading/README.md, so it is
# the 1 rho used to keep the ladder comparison to a single, readable curve pair
# per placement rather than 4 overlapping ones.
LADDER_RHO_TOKEN = "0_1"

# The coverage ledger's own implantation bar (CLAUDE.md): a pair where either
# side falls below this ASR is a broken attack rather than a detection result
# and is excluded from every delta rather than averaged in as a loss.
ASR_CLEARS_THRESHOLD = 0.85


def filter_cells_by_rate(cells: list[dict], rate_token: str) -> list[dict]:
    """The subset of matched cells trained at exactly this poison-rate token."""
    filtered = [cell for cell in cells if cell["rate_token"] == rate_token]
    return filtered


def excluded_record(adam_row: dict, sam_row: dict, rate_label: str, rho: str) -> dict:
    """1 dropped pair's identity and both sides' ASR, for the excluded-pairs table."""
    record = {
        "dataset": adam_row["dataset"],
        "dataset_label": dataset_label(adam_row["dataset"]),
        "attack": adam_row["attack"],
        "attack_label": attack_label(adam_row["attack"]),
        "architecture": adam_row["architecture"],
        "rate": rate_label,
        "rho": rho,
        "adam_folder": adam_row["folder"],
        "sam_folder": sam_row["folder"],
        "adam_asr": adam_row["asr"],
        "sam_asr": sam_row["asr"],
    }
    return record


def apply_asr_gate(
    covered: list[tuple[dict, dict]],
) -> tuple[list[tuple[dict, dict]], list[tuple[dict, dict]]]:
    """Matched pairs where both sides cleared ASR_CLEARS_THRESHOLD, and the rest.

    A pair whose Adam or SAM side never implanted the backdoor answers no
    question about detection, so it is split off here rather than folded into
    a delta as a spurious loss.
    """
    kept = [
        (adam, sam)
        for adam, sam in covered
        if adam["asr"] >= ASR_CLEARS_THRESHOLD and sam["asr"] >= ASR_CLEARS_THRESHOLD
    ]
    excluded = [
        (adam, sam)
        for adam, sam in covered
        if adam["asr"] < ASR_CLEARS_THRESHOLD or sam["asr"] < ASR_CLEARS_THRESHOLD
    ]
    return kept, excluded


def paired_metric_deltas(
    covered: list[tuple[dict, dict]], placement_name: str, field: str
) -> list[float]:
    """SAM minus Adam for 1 psbd summary field, over the paired rows of covered."""
    deltas = [
        sam["psbd"][placement_name][field] - adam["psbd"][placement_name][field]
        for adam, sam in covered
    ]
    return deltas


def aggregate_placement_at_rate(
    covered: list[tuple[dict, dict]], placement_name: str
) -> dict:
    """AUROC, TPR@10% and TPR@20%, Adam against SAM, paired and bootstrapped, at 1 rate."""
    fields = {"auroc": "auroc", "tpr_at_10": "tpr_at_10", "tpr_at_20": "tpr_at_20"}
    aggregate = {"n": len(covered)}
    for output_name, field in fields.items():
        adam_values = [adam["psbd"][placement_name][field] for adam, _ in covered]
        sam_values = [sam["psbd"][placement_name][field] for _, sam in covered]
        deltas = paired_metric_deltas(covered, placement_name, field)
        aggregate[f"adam_{output_name}_mean"] = mean_or_none(adam_values)
        aggregate[f"sam_{output_name}_mean"] = mean_or_none(sam_values)
        aggregate[f"delta_{output_name}_mean"] = mean_or_none(deltas)
        aggregate[f"delta_{output_name}_ci"] = ci_or_none(
            bootstrap_ci(deltas, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
        )
    return aggregate


def calibration_at_rate(covered: list[tuple[dict, dict]], placement_name: str) -> dict:
    """The adaptive rule's own outputs, Adam against SAM: the rate it picked and the
    clean-validation shift ratio it reached there.

    A placement's "gain" at a nominal poison rate can be the adaptive rule
    landing on a different dropout rate on the SAM side rather than the same
    disturbance separating classes better, so the rate and the achieved shift
    ratio are reported beside the AUROC delta, never folded into it.
    """
    adam_rates = [adam["psbd"][placement_name]["rate"] for adam, _ in covered]
    sam_rates = [sam["psbd"][placement_name]["rate"] for _, sam in covered]
    calibration = {
        "adam_selected_rate_mean": mean_or_none(adam_rates),
        "sam_selected_rate_mean": mean_or_none(sam_rates),
    }
    return calibration


def read_placement_ladder(folder: str, placement_id: str) -> dict[float, dict] | None:
    """Every swept rate's clean-validation shift ratio and headline AUROC for 1
    placement of 1 checkpoint's psbd_metrics.json, keyed by rate. None unswept.

    Read straight from the cache rather than through
    cli.compare_detectors.psbd_values, since that reader returns only the 1 rate
    the adaptive rule picked and this comparison needs the whole ladder.
    """
    report = load_json(os.path.join(RESULTS_DIR, folder, "psbd_metrics.json"))
    if report is None:
        return None
    block = report.get("placements", {}).get(placement_id)
    if block is None:
        return None
    ladder = {}
    for row in block.get("rates", []):
        headline = row.get("detection_psu_ratio", {}).get(HEADLINE_KEY)
        if headline is None:
            continue
        ladder[row["rate"]] = {
            "shift_ratio": row["shift_ratio"]["validation"],
            "auroc": headline["auroc"],
        }
    return ladder or None


def rate_ladder_points(cells: list[dict], rho_token: str) -> dict[str, list[dict]]:
    """Every (shift ratio, AUROC) point on the disturbance ladder, Adam and SAM
    separately, for both placements, over 1 rho's matched cells.

    Points are pooled across every matched cell rather than averaged onto a
    shared rate axis, since Adam and SAM do not reach the same shift ratio at
    the same nominal rate (that mismatch is exactly what calibration_at_rate
    reports): the scatter itself is what shows whether 1 curve sits above the
    other everywhere or only at 1 disturbance level. Pairs where either side
    never implanted the backdoor (ASR below ASR_CLEARS_THRESHOLD) are dropped
    before the points are read, the same gate every other comparison in this
    module applies.
    """
    pairs = matched_pairs(cells, rho_token)
    covered, _ = apply_asr_gate(common_coverage(pairs))
    points_by_side_and_placement: dict[str, list[dict]] = {
        f"{side}_{placement_name}": []
        for side in ("adam", "sam")
        for placement_name in PLACEMENTS
    }
    for adam_row, sam_row in covered:
        for placement_name, placement_id in PLACEMENTS.items():
            adam_ladder = read_placement_ladder(adam_row["folder"], placement_id)
            sam_ladder = read_placement_ladder(sam_row["folder"], placement_id)
            if adam_ladder is not None:
                for point in adam_ladder.values():
                    points_by_side_and_placement[f"adam_{placement_name}"].append(point)
            if sam_ladder is not None:
                for point in sam_ladder.values():
                    points_by_side_and_placement[f"sam_{placement_name}"].append(point)
    return points_by_side_and_placement


def aggregate_by_rate(cells: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    """Part 1 and 2: per poison-rate-token, per rho, both placements' AUROC and
    TPR deltas plus the adaptive rule's calibration, over the pairs that
    cleared ASR_CLEARS_THRESHOLD on both sides, plus the flat list of every
    pair this rate x rho grid dropped for failing that bar."""
    by_rate = {}
    all_excluded = []
    for rate_token in RATE_TOKENS:
        rate_cells = filter_cells_by_rate(cells, rate_token)
        present_rho_tokens = sorted(
            {rho_token for cell in rate_cells for rho_token in cell["sam_rows"]},
            key=lambda token: float(rho_label(token)),
        )
        by_rho = {}
        for rho_token in present_rho_tokens:
            pairs = matched_pairs(rate_cells, rho_token)
            covered, excluded = apply_asr_gate(common_coverage(pairs))
            rho = rho_label(rho_token)
            all_excluded.extend(
                excluded_record(adam, sam, RATE_TOKEN_LABELS[rate_token], rho)
                for adam, sam in excluded
            )
            by_rho[rho] = {
                "n_pairs": len(covered),
                "n_excluded_asr_gate": len(excluded),
                "placements": {
                    placement_name: aggregate_placement_at_rate(covered, placement_name)
                    for placement_name in PLACEMENTS
                },
                "calibration": {
                    placement_name: calibration_at_rate(covered, placement_name)
                    for placement_name in PLACEMENTS
                },
            }
        by_rate[rate_token] = {
            "label": RATE_TOKEN_LABELS[rate_token],
            "n_cells": len(rate_cells),
            "by_rho": by_rho,
        }
    return by_rate, all_excluded


def list_low_rate_pairs(cells: list[dict]) -> dict[str, list[dict]]:
    """Every (dataset, attack, architecture) combination matched at 1% and 5%
    poisoning, with the rhos it reached, so coverage can be stated plainly
    rather than inferred from a bare pair count.
    """
    listing = {}
    for rate_token in ("0_01", "0_05"):
        rate_cells = filter_cells_by_rate(cells, rate_token)
        entries = [
            {
                "dataset": cell["dataset"],
                "dataset_label": dataset_label(cell["dataset"]),
                "attack": cell["attack"],
                "attack_label": attack_label(cell["attack"]),
                "architecture": cell["architecture"],
                "rhos": sorted(
                    (rho_label(token) for token in cell["sam_rows"]),
                    key=float,
                ),
            }
            for cell in rate_cells
            if cell["adam_row"] is not None and cell["sam_rows"]
        ]
        listing[rate_token] = entries
    return listing


def aggregate_by_attack(cells: list[dict]) -> dict[str, dict]:
    """Part 4: per attack, per rho, the paired token-mask and residual-dropout
    AUROC delta, pooled over every poison rate and dataset that attack reached,
    since splitting by attack AND rate AND rho at once would leave too few
    pairs in most cells to bootstrap."""
    by_attack = {}
    for attack in ATTACKS:
        attack_cells = [cell for cell in cells if cell["attack"] == attack]
        present_rho_tokens = sorted(
            {rho_token for cell in attack_cells for rho_token in cell["sam_rows"]},
            key=lambda token: float(rho_label(token)),
        )
        by_rho = {}
        for rho_token in present_rho_tokens:
            pairs = matched_pairs(attack_cells, rho_token)
            covered, _ = apply_asr_gate(common_coverage(pairs))
            by_rho[rho_label(rho_token)] = {
                "n_pairs": len(covered),
                "placements": {
                    placement_name: aggregate_placement_at_rate(covered, placement_name)
                    for placement_name in PLACEMENTS
                },
            }
        by_attack[attack] = {"label": attack_label(attack), "by_rho": by_rho}
    return by_attack


def main() -> None:
    cells = discover_cells()
    attach_rows(cells)

    by_rate, excluded_asr_gate = aggregate_by_rate(cells)
    payload = {
        "generator": "experiments/sam_low_rate/measure.py",
        "written_at": utc_timestamp(),
        "git_commit": current_git_commit(),
        "inputs": [
            "experiments/sam_reading/measure.py",
            "checkpoints/<folder>/metrics.json",
            "checkpoints/<folder>/args.json",
            "results/<folder>/psbd_metrics.json",
        ],
        "n_cells": len(cells),
        "asr_clears_threshold": ASR_CLEARS_THRESHOLD,
        "low_rate_pairs": list_low_rate_pairs(cells),
        "by_rate": by_rate,
        "rate_ladder": {
            "rho": rho_label(LADDER_RHO_TOKEN),
            "points": rate_ladder_points(cells, LADDER_RHO_TOKEN),
        },
        "by_attack": aggregate_by_attack(cells),
        "excluded_asr_gate": excluded_asr_gate,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)

    for rate_token in RATE_TOKENS:
        entry = payload["by_rate"][rate_token]
        print(f"{entry['label']}: {entry['n_cells']} matched cells")
        for rho, block in entry["by_rho"].items():
            tm = block["placements"]["token_mask"]
            rd = block["placements"]["residual_dropout"]
            print(
                f"  rho {rho}: n={block['n_pairs']} excluded={block['n_excluded_asr_gate']} "
                f"TM delta={tm['delta_auroc_mean']} RD delta={rd['delta_auroc_mean']}"
            )
    print(
        f"{len(excluded_asr_gate)} pairs dropped for ASR below {ASR_CLEARS_THRESHOLD}"
    )
    print(f"written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
