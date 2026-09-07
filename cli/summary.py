"""Collapse every psbd_metrics.json into one compact, versionable table.

The full stage-2 record is about 2 MB per checkpoint and 1.2 GB across the tree,
which is regenerable from the cached tensors and far too large to keep in git.
What every table in the paper actually reads is a handful of numbers per
(checkpoint, placement): the operating point, the rate that produced it, and how
well it separated. That is what this writes.

One row per (checkpoint, placement, rate-selection rule). The 3 rules answer
different questions and are all kept, because reporting only one of them has
already produced 2 wrong conclusions in this project:

  adaptive  the rate the paper's own rule picks from clean validation data. What
            a defender actually gets.
  oracle    the best rate in hindsight. An upper bound, since picking it reads
            the poison labels the defence exists to predict.
  matched   the rate whose clean-validation shift ratio sits nearest a target.
            The only rule under which 2 placements are being asked the same
            question, since a shared nominal rate is a different intervention at
            every position.

achieved_shift_ratio travels with every matched row. A cell whose grid never
reaches the target is still selected by the nearest rule, so without the achieved
value a table cannot tell a matched cell from an unmatched one.

Example
    python -m cli.summary --output results/detection_summary.csv
"""

import argparse
import csv
import glob
import json
import os

SELECTION_RULES = ("adaptive", "oracle")

# Longest names first, so channel_mask is not read as a position ending in mask.
KNOWN_OPERATORS = (
    "channel_mask",
    "gain_scale",
    "token_mask",
    "head_mask",
    "droppath",
    "gaussian",
    "scale_up",
)

FIELDS = (
    "folder",
    "architecture",
    "dataset",
    "attack",
    "label_mode",
    "poison_rate",
    "realized_poison_rate",
    "poison_rate_capped",
    "optimizer",
    "rho",
    "asr",
    "clean_accuracy",
    "target_label",
    "placement",
    "position",
    "operator",
    "rule",
    "rate",
    "achieved_shift_ratio",
    "auroc",
    "tpr",
    "fpr",
    "threshold",
    "quantile",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--output", default="results/detection_summary.csv")
    parser.add_argument(
        "--include-sam",
        action="store_true",
        help=(
            "keep SAM-trained checkpoints. Off by default. SAM is 69 percent of the "
            "checkpoint set and its whole effect on detection is +0.009 mean AUROC, "
            "which needs 10 to 89 seeds per cell to establish. Carrying it dilutes "
            "every panel and reintroduces the unequal-coverage failure mode that "
            "inverted 4 conclusions in this project"
        ),
    )
    return parser.parse_args()


def read_json(path: str) -> dict | None:
    """One JSON file, or None when it does not exist."""
    if not os.path.exists(path):
        return None

    with open(path) as handle:
        return json.load(handle)


def split_operator(placement: str, known_operators: tuple[str, ...]) -> tuple[str, str]:
    """(position, operator) from a placement name.

    A bare name is the paper's dropout, which is exactly why the naming keeps it
    bare. known_operators must be ordered longest first so a compound name is not
    matched by its own suffix.
    """
    for operator in known_operators:
        if placement.endswith(f"_{operator}"):
            return placement[: -len(operator) - 1], operator
    return placement, "dropout"


def rows_for_checkpoint(folder: str, report: dict, metadata: dict) -> list[dict]:
    """Every (placement, rule) row this checkpoint contributes."""
    base = {
        "folder": folder,
        "architecture": metadata.get("architecture"),
        "dataset": report.get("dataset"),
        "attack": report.get("attack"),
        "label_mode": report.get("label_mode"),
        "poison_rate": report.get("poison_rate"),
        "realized_poison_rate": metadata.get("realized_poison_rate"),
        "poison_rate_capped": metadata.get("poison_rate_capped"),
        "optimizer": report.get("optimizer"),
        "rho": report.get("rho"),
        "asr": metadata.get("asr"),
        "clean_accuracy": metadata.get("clean_accuracy"),
        "target_label": report.get("target_label"),
    }

    rows = []
    for placement, block in sorted(report.get("placements", {}).items()):
        position, operator = split_operator(placement, KNOWN_OPERATORS)
        shared = {
            **base,
            "placement": placement,
            "position": position,
            "operator": operator,
        }

        for rule in SELECTION_RULES:
            detection = block.get(rule)
            if detection is None:
                continue
            rows.append(
                {
                    **shared,
                    "rule": rule,
                    "rate": block.get(f"{rule}_rate"),
                    "achieved_shift_ratio": None,
                    "auroc": detection.get("auroc"),
                    "tpr": detection.get("tpr"),
                    "fpr": detection.get("fpr"),
                    "threshold": detection.get("threshold"),
                    "quantile": detection.get("quantile"),
                }
            )

        for target_key, matched in sorted((block.get("matched_shift") or {}).items()):
            if matched is None:
                continue
            rows.append(
                {
                    **shared,
                    "rule": f"matched_{target_key}",
                    "rate": matched.get("rate"),
                    "achieved_shift_ratio": matched.get("achieved_shift_ratio"),
                    "auroc": matched.get("auroc"),
                    "tpr": matched.get("tpr"),
                    "fpr": matched.get("fpr"),
                    "threshold": matched.get("threshold"),
                    "quantile": matched.get("quantile"),
                }
            )

    return rows


def collect_rows(
    results_dir: str, checkpoints_dir: str, include_sam: bool
) -> tuple[list[dict], int, int]:
    """Every row in the tree, plus how many checkpoints were skipped and why."""
    all_rows: list[dict] = []
    skipped = 0
    excluded_sam = 0

    for path in sorted(glob.glob(os.path.join(results_dir, "*", "psbd_metrics.json"))):
        folder = os.path.basename(os.path.dirname(path))
        report = read_json(path)
        metadata = read_json(os.path.join(checkpoints_dir, folder, "args.json")) or {}
        if report is None:
            skipped += 1
            continue
        if metadata.get("optimizer") == "sam" and not include_sam:
            excluded_sam += 1
            continue
        all_rows.extend(rows_for_checkpoint(folder, report, metadata))

    return all_rows, skipped, excluded_sam


def write_csv(output_path: str, rows: list[dict]) -> None:
    """Write the summary table, one line per (checkpoint, placement, rule)."""
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows, skipped, excluded_sam = collect_rows(
        args.results_dir, args.checkpoints_dir, args.include_sam
    )
    write_csv(args.output, rows)

    size_mb = os.path.getsize(args.output) / 1048576
    checkpoints = len({row["folder"] for row in rows})
    placements = len({row["placement"] for row in rows})
    print(f"rows:        {len(rows)}")
    print(f"checkpoints: {checkpoints}")
    print(f"placements:  {placements}")
    print(f"skipped:     {skipped}")
    print(f"sam excluded: {excluded_sam}")
    print(f"wrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
