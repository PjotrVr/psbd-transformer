"""Aggregate every psbd_metrics.json into one per-checkpoint table.

One row per (checkpoint, placement). ASR and clean accuracy are read from the
checkpoint's own metrics.json rather than recomputed, because that is the
already-measured, eligibility-correct number and recomputing it here would be a
second implementation of the same thing that could disagree.

Three detection columns, deliberately, because they answer different questions:

  adaptive   the rate a defender could actually have chosen, using the paper's
             rule on clean validation data only. This is the honest headline.
  oracle     the best rate in hindsight. An upper bound, not a result: choosing
             it reads the poison labels the defence is supposed to predict. The
             gap to adaptive is the cost of not knowing the right rate.
  matched    every placement at the same measured disturbance (clean-validation
             shift ratio), which is the only way to compare placements without
             the comparison collapsing into "which one perturbs harder".

FPR is printed but carries little information here: the threshold is a quantile
of clean test PSU and FPR is measured on clean test PSU, so it sits near the
quantile whatever the placement does. TPR and AUROC are the signal.

Example
    python psbd_report.py --format markdown
    python psbd_report.py --format csv > results/psbd_report.csv
"""

import argparse
import csv
import json
import os
import sys

HEADLINE = "q0.25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument(
        "--matched-shift",
        default="sigma0.8",
        help="which matched-disturbance column to print (sigma0.2 .. sigma0.8)",
    )
    return parser.parse_args()


def read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def collect_rows(results_dir: str, checkpoints_dir: str, matched_key: str) -> list[dict]:
    """One row per (checkpoint, placement), sorted for a stable diff."""
    rows = []
    for folder in sorted(os.listdir(results_dir)):
        report = read_json(os.path.join(results_dir, folder, "psbd_metrics.json"))
        if report is None:
            continue
        baseline = read_json(os.path.join(checkpoints_dir, folder, "metrics.json")) or {}

        for placement, block in sorted(report["placements"].items()):
            adaptive = block.get("adaptive") or {}
            oracle = block.get("oracle") or {}
            matched = (block.get("matched_shift") or {}).get(matched_key) or {}
            rows.append(
                {
                    "folder": folder,
                    "attack": report.get("attack"),
                    "poison_rate": report.get("poison_rate"),
                    "optimizer": report.get("optimizer"),
                    "rho": report.get("rho"),
                    "asr": baseline.get("asr"),
                    "clean_accuracy": baseline.get("clean_accuracy"),
                    "placement": placement,
                    "adaptive_rate": block.get("adaptive_rate"),
                    "adaptive_tpr": adaptive.get("tpr"),
                    "adaptive_fpr": adaptive.get("fpr"),
                    "adaptive_auroc": adaptive.get("auroc"),
                    "oracle_rate": block.get("oracle_rate"),
                    "oracle_tpr": oracle.get("tpr"),
                    "oracle_auroc": oracle.get("auroc"),
                    "matched_rate": matched.get("rate"),
                    "matched_auroc": matched.get("auroc"),
                    "n_backdoor": report.get("split_sizes", {}).get("backdoor"),
                }
            )
    return rows


def _cell(value, spec: str = ".3f") -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:{spec}}"
    return str(value)


def render_markdown(rows: list[dict], matched_key: str) -> str:
    header = (
        "| checkpoint | attack | pr | opt | ASR | CA | placement | "
        "adapt p | adapt TPR | adapt FPR | adapt AUROC | oracle p | oracle AUROC | "
        f"{matched_key} p | {matched_key} AUROC |"
    )
    divider = "|" + "---|" * 15
    lines = [header, divider]
    for row in rows:
        optimizer = row["optimizer"] or "--"
        if row["rho"]:
            optimizer = f"{optimizer} {row['rho']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["folder"],
                    str(row["attack"]),
                    _cell(row["poison_rate"], ".3f"),
                    optimizer,
                    _cell(row["asr"]),
                    _cell(row["clean_accuracy"]),
                    row["placement"],
                    _cell(row["adaptive_rate"], ".2f"),
                    _cell(row["adaptive_tpr"]),
                    _cell(row["adaptive_fpr"]),
                    _cell(row["adaptive_auroc"]),
                    _cell(row["oracle_rate"], ".2f"),
                    _cell(row["oracle_auroc"]),
                    _cell(row["matched_rate"], ".2f"),
                    _cell(row["matched_auroc"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_csv(rows: list[dict]) -> None:
    if not rows:
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def summarize_by_placement(rows: list[dict]) -> str:
    """Mean AUROC per placement, the one number the whole study is about.

    Rows whose AUROC is missing are counted and reported rather than dropped
    silently, because a placement that failed to produce a number on half the
    grid should not look like a placement that won on the half it managed.
    """
    by_placement: dict[str, list[float]] = {}
    missing: dict[str, int] = {}
    for row in rows:
        value = row["adaptive_auroc"]
        if value is None:
            missing[row["placement"]] = missing.get(row["placement"], 0) + 1
        else:
            by_placement.setdefault(row["placement"], []).append(value)

    lines = ["", "## Mean AUROC by placement (adaptive rate, backdoored models only)", ""]
    lines.append("| placement | n | mean AUROC | min | max | no adaptive rate |")
    lines.append("|---|---|---|---|---|---|")
    for placement in sorted(set(list(by_placement) + list(missing))):
        values = by_placement.get(placement, [])
        if not values:
            lines.append(f"| {placement} | 0 | -- | -- | -- | {missing.get(placement, 0)} |")
            continue
        lines.append(
            f"| {placement} | {len(values)} | {sum(values) / len(values):.3f} | "
            f"{min(values):.3f} | {max(values):.3f} | {missing.get(placement, 0)} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.results_dir, args.checkpoints_dir, args.matched_shift)
    if not rows:
        raise SystemExit(f"no psbd_metrics.json found under {args.results_dir}")

    if args.format == "csv":
        render_csv(rows)
        return

    backdoored = [row for row in rows if row["attack"] != "benign"]
    benign = [row for row in rows if row["attack"] == "benign"]

    print(render_markdown(rows, args.matched_shift))
    print(summarize_by_placement(backdoored))
    if benign:
        print("")
        print("## Negative control (benign model, same trigger)")
        print("")
        print("Detection here should sit near 0.5. Anything much above it means the")
        print("probe responds to the trigger perturbation rather than to a backdoor.")
        print("")
        for row in benign:
            print(
                f"- {row['folder']} / {row['placement']}: "
                f"AUROC {_cell(row['adaptive_auroc'])} at p={_cell(row['adaptive_rate'], '.2f')}"
            )


if __name__ == "__main__":
    main()
