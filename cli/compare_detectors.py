"""1 table per metric: PSBD pinned to the declared placement against every detector.

Reads results/<folder>/psbd_metrics.json and the per-detector records under
results/<folder>/detectors/, which were produced on the identical split,
threshold rule and quantile grid, so the only difference between the columns is
the score being thresholded.

PSBD appears in 3 columns and never at a per-cell best. psbd_adaptive is the
paper's own deployable rule, the smallest rate whose clean-validation shift ratio
reaches 0.8, at the recommended placement. psbd_matched06 is the same placement at
the rate whose shift ratio sits nearest 0.6, the comparison device the placement
study used. psbd_published is the ConvNet placement, dropout after the residual
add, at the adaptive rule, so the size of this project's placement selection is
visible beside the competitors. Every per-quantile number is read from the
chosen rate's detection_psu_ratio block, since the adaptive and matched_shift
summaries in psbd_metrics.json hold the absolute form at 1 quantile only.

Aggregates run over the common-coverage cell set, the cells where every column
with any data has a value, and print their n. A detector that failed on 10 cells
is never averaged over 55 cells against another's 65, and a column with no
records yet does not empty the table for the others. Benign checkpoints get
their own table: a flag rate on trigger-stamped clean images is a false-alarm
rate, never a TPR.

Example
    python -m cli.compare_detectors
    python -m cli.compare_detectors --fpr 0.01 0.05 --markdown docs/detectors/comparison.md
    python -m cli.compare_detectors --per-detector-dir docs/detectors --csv results/detector_summary.csv.gz
"""

import argparse
import csv
import gzip
import json
import os
import random
import statistics
import sys

from defences.decision import (
    ADAPTIVE_SHIFT_TARGET,
    EASY_ATTACKS,
    HARD_ATTACKS,
    HEADLINE_QUANTILE,
    PRIMARY_DATASETS,
    PLACEMENT_MATCH_TARGET,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    shift_key,
)
from detectors import (
    DATA_REQUIREMENT,
    DETECTOR_NAMES,
    EXPERIMENTAL_DETECTOR_NAMES,
    FORWARD_PASSES_PER_INPUT,
)
from detectors.records import (
    STATUS_FAILED,
    STATUS_SCORED,
    legacy_report_present,
    load_report,
    report_path,
)
from training.loop import current_git_commit, utc_timestamp

PSBD_COLUMNS = ("psbd_adaptive", "psbd_matched06", "psbd_published")
DEFAULT_FPRS = (0.01, 0.05, 0.10, 0.25)
# Below this many triggered images a per-cell TPR moves in steps a reader must
# see, so the row is marked.
SMALL_POSITIVE_SET = 500
DEFAULT_BOOTSTRAP = 5000
RESULTS_BLOCK_BEGIN = "<!-- results:begin -->"
RESULTS_BLOCK_END = "<!-- results:end -->"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--coverage", default=None, help="default <results-dir>/coverage/coverage.json"
    )
    parser.add_argument("--declaration", default="configs/psbd_basis.json")
    parser.add_argument("--fpr", nargs="*", type=float, default=list(DEFAULT_FPRS))
    parser.add_argument("--include-sam", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--markdown", default=None, help="write every table to this file"
    )
    parser.add_argument(
        "--per-detector-dir",
        default=None,
        help="rewrite the results block of <dir>/<detector>.md for every detector",
    )
    parser.add_argument(
        "--csv", default=None, help="1 row per (cell, column, quantile), gzipped"
    )
    return parser


def read_json(path: str) -> dict | None:
    """A JSON file, or None when it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        loaded = json.load(handle)
    return loaded


def panel_cells(coverage: dict, declaration: dict, include_sam: bool) -> list[dict]:
    """The cells a table covers: attacks that cleared the ASR bar, plus the benign references."""
    cells = []
    for cell in coverage["cells"]:
        if cell.get("asr_class") != "clears":
            continue
        if "sam_rho" in cell["folder_name"] and not include_sam:
            continue
        cells.append(
            {
                "folder": cell["folder_name"],
                "dataset": cell["dataset"],
                "attack": cell["attack"],
                "poison_rate": cell["poison_rate"],
                "asr": cell.get("asr"),
                "kind": "attack",
            }
        )
    for dataset, folder in declaration["benign_reference"].items():
        if dataset.startswith("_"):
            continue
        cells.append(
            {
                "folder": folder,
                "dataset": dataset,
                "attack": "benign",
                "poison_rate": None,
                "asr": None,
                "kind": "benign",
            }
        )
    return cells


def psbd_rate(placement_block: dict, rule: str) -> float | None:
    """The rate a PSBD rule chose for a placement, or None when it chose nothing."""
    if rule == "adaptive":
        return placement_block.get("adaptive_rate")
    matched = placement_block.get("matched_shift", {}).get(
        shift_key(PLACEMENT_MATCH_TARGET)
    )
    rate = matched.get("rate") if matched else None
    return rate


def psbd_values(report: dict | None, placement: str, rule: str) -> dict | None:
    """Every quantile's fractional-PSU report at the rate the rule picked, or None."""
    if report is None:
        return None
    block = report.get("placements", {}).get(placement)
    if block is None:
        return None
    rate = psbd_rate(block, rule)
    if rate is None:
        return None
    for row in block.get("rates", []):
        if row.get("rate") == rate:
            values = dict(row.get("detection_psu_ratio", {}))
            values["_rate"] = rate
            values["_n_backdoor"] = row.get("n_samples", {}).get("backdoor")
            return values
    return None


def detector_values(
    results_dir: str, folder: str, name: str, notes: dict
) -> dict | None:
    """A detector's scored detection blocks on a cell, or None with the reason noted."""
    record = load_report(report_path(results_dir, folder, name))
    if record is None:
        return None
    if record.get("status") == STATUS_FAILED:
        notes["failed"].append((folder, name))
        return None
    if record.get("status") != STATUS_SCORED:
        return None
    if record["provenance"].get("max_samples") is not None:
        notes["smoke"].append((folder, name))
        return None
    values = dict(record["detection"])
    values["_n_backdoor"] = record["provenance"]["split"]["n_backdoor"]
    return values


def collect_columns(
    cells: list[dict], args: argparse.Namespace, notes: dict
) -> tuple[list[str], dict]:
    """Column names, plus the quantile blocks that exist per (folder, column)."""
    detector_columns = list(DETECTOR_NAMES)
    for name in EXPERIMENTAL_DETECTOR_NAMES:
        if any(
            load_report(report_path(args.results_dir, cell["folder"], name)) is not None
            for cell in cells
        ):
            detector_columns.append(name)

    values: dict[tuple[str, str], dict] = {}
    for cell in cells:
        folder = cell["folder"]
        report = read_json(os.path.join(args.results_dir, folder, "psbd_metrics.json"))
        for column, placement, rule in (
            ("psbd_adaptive", RECOMMENDED_PLACEMENT, "adaptive"),
            ("psbd_matched06", RECOMMENDED_PLACEMENT, "matched"),
            ("psbd_published", PUBLISHED_PLACEMENT, "adaptive"),
        ):
            block = psbd_values(report, placement, rule)
            if block is None and report is not None:
                notes["psbd_none"].append((folder, column))
            if block is not None:
                values[(folder, column)] = block
        for name in detector_columns:
            block = detector_values(args.results_dir, folder, name, notes)
            if block is not None:
                values[(folder, name)] = block
        if legacy_report_present(args.results_dir, folder):
            notes["legacy"] += 1

    columns = list(PSBD_COLUMNS) + detector_columns
    return columns, values


def metric(block: dict | None, key: str, quantile_key: str) -> float | None:
    """1 number out of a quantile block, or None when the column has no value."""
    if block is None or quantile_key not in block:
        return None
    value = block[quantile_key].get(key)
    return value


def cell_text(value: float | None, places: int = 3) -> str:
    """A table cell, with a placeholder for a column that produced no number."""
    text = "--" if value is None or value != value else f"{value:.{places}f}"
    return text


def bootstrap_interval(
    values: list[float], resamples: int, seed: int
) -> tuple[float, float]:
    """A 95% interval on the mean by resampling with replacement, seeded."""
    if len(values) < 3 or resamples <= 0:
        return float("nan"), float("nan")
    generator = random.Random(seed)
    draws = sorted(
        statistics.mean(generator.choices(values, k=len(values)))
        for _ in range(resamples)
    )
    interval = (draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1])
    return interval


def subsets(cells: list[dict]) -> list[tuple[str, list[dict]]]:
    """The aggregate rows every metric table ends with."""
    attacks = [cell for cell in cells if cell["kind"] == "attack"]
    groups = [
        ("all cells", attacks),
        ("hard attacks", [c for c in attacks if c["attack"] in HARD_ATTACKS]),
        ("easy attacks", [c for c in attacks if c["attack"] in EASY_ATTACKS]),
        ("primary datasets", [c for c in attacks if c["dataset"] in PRIMARY_DATASETS]),
    ]
    for dataset in sorted({c["dataset"] for c in attacks}):
        groups.append(
            (f"dataset {dataset}", [c for c in attacks if c["dataset"] == dataset])
        )
    for rate in sorted({c["poison_rate"] for c in attacks}):
        groups.append(
            (f"poison rate {rate:g}", [c for c in attacks if c["poison_rate"] == rate])
        )
    return groups


def active_columns(cells: list[dict], columns: list[str], values: dict) -> list[str]:
    """The columns with a value on at least 1 of these cells."""
    active = [
        column
        for column in columns
        if any((cell["folder"], column) in values for cell in cells)
    ]
    return active


def common_coverage(
    cells: list[dict], columns: list[str], values: dict, key: str, quantile_key: str
) -> list[dict]:
    """The cells where every active column has this metric, the only set a mean may span."""
    active = active_columns(cells, columns, values)
    covered = [
        cell
        for cell in cells
        if all(
            metric(values.get((cell["folder"], column)), key, quantile_key) is not None
            for column in active
        )
    ]
    return covered


def metric_table(
    title: str,
    cells: list[dict],
    columns: list[str],
    values: dict,
    key: str,
    quantile_key: str,
    args: argparse.Namespace,
) -> list[str]:
    """1 metric over every attack cell, then the aggregate rows, as markdown lines."""
    lines = [f"### {title}", ""]
    lines.append(
        "| dataset | attack | rate | ASR | n_bd | " + " | ".join(columns) + " |"
    )
    lines.append("|---|---|---:|---:|---:|" + "---:|" * len(columns))

    attacks = [cell for cell in cells if cell["kind"] == "attack"]
    for cell in sorted(
        attacks, key=lambda c: (c["dataset"], c["attack"], c["poison_rate"])
    ):
        blocks = [values.get((cell["folder"], column)) for column in columns]
        n_backdoor = next(
            (b["_n_backdoor"] for b in blocks if b and b.get("_n_backdoor")), None
        )
        mark = "*" if n_backdoor is not None and n_backdoor < SMALL_POSITIVE_SET else ""
        row = [
            cell["dataset"],
            cell["attack"],
            f"{cell['poison_rate']:g}",
            cell_text(cell["asr"], 2),
            f"{n_backdoor}{mark}" if n_backdoor is not None else "--",
        ]
        row += [cell_text(metric(block, key, quantile_key)) for block in blocks]
        lines.append("| " + " | ".join(row) + " |")

    active = active_columns(attacks, columns, values)
    lines.append("")
    lines.append(
        f"Aggregates over the common-coverage cells of the {len(active)} columns with "
        f"data, {key} at {quantile_key}. Macro is the mean over attack means."
    )
    lines.append("")
    lines.append("| subset | n | " + " | ".join(columns) + " |")
    lines.append("|---|---:|" + "---:|" * len(columns))
    for name, subset in subsets(cells):
        covered = common_coverage(subset, columns, values, key, quantile_key)
        if not covered:
            continue
        means = [
            statistics.mean(
                metric(values[(c["folder"], column)], key, quantile_key)
                for c in covered
            )
            if column in active
            else None
            for column in columns
        ]
        lines.append(
            f"| {name} | {len(covered)} | "
            + " | ".join(cell_text(m) for m in means)
            + " |"
        )
        if name == "all cells":
            macro = []
            for column in columns:
                if column not in active:
                    macro.append(None)
                    continue
                per_attack = [
                    statistics.mean(
                        metric(values[(c["folder"], column)], key, quantile_key)
                        for c in covered
                        if c["attack"] == attack
                    )
                    for attack in sorted({c["attack"] for c in covered})
                ]
                macro.append(statistics.mean(per_attack))
            lines.append(
                f"| macro over attacks | {len(covered)} | "
                + " | ".join(cell_text(m) for m in macro)
                + " |"
            )

    lines += paired_deltas(cells, columns, values, key, quantile_key, args)
    lines.append("")
    return lines


def paired_deltas(cells, columns, values, key, quantile_key, args) -> list[str]:
    """Each column minus psbd_adaptive on the same cells, with a bootstrap interval."""
    reference = "psbd_adaptive"
    if reference not in columns:
        return []
    attacks = [cell for cell in cells if cell["kind"] == "attack"]
    lines = [
        "",
        f"Paired deltas against {reference}, {args.bootstrap} resamples, 95% interval.",
        "",
        "| column | n | mean delta | interval |",
        "|---|---:|---:|---|",
    ]
    for column in columns:
        if column == reference:
            continue
        deltas = []
        for cell in attacks:
            ours = metric(values.get((cell["folder"], column)), key, quantile_key)
            theirs = metric(values.get((cell["folder"], reference)), key, quantile_key)
            if ours is not None and theirs is not None:
                deltas.append(ours - theirs)
        if not deltas:
            continue
        low, high = bootstrap_interval(deltas, args.bootstrap, args.seed)
        lines.append(
            f"| {column} | {len(deltas)} | {statistics.mean(deltas):+.3f} | "
            f"[{low:+.3f}, {high:+.3f}] |"
        )
    return lines


def benign_table(
    cells: list[dict], columns: list[str], values: dict, quantile_keys: list[str]
) -> list[str]:
    """Benign references: clean FPR and the flag rate on trigger-stamped clean images."""
    benign = [cell for cell in cells if cell["kind"] == "benign"]
    lines = ["### Benign references", ""]
    lines.append(
        "A benign model has no backdoor, so the rate at which it flags trigger-stamped "
        "clean images is a false-alarm rate on a patch, never a TPR, and AUROC should "
        "sit at 0.5."
    )
    lines.append("")
    lines.append(
        "| dataset | column | "
        + " | ".join(f"clean FPR {q} | stamped flag {q}" for q in quantile_keys)
        + " | AUROC |"
    )
    lines.append("|---|---|" + "---:|---:|" * len(quantile_keys) + "---:|")
    headline = f"q{HEADLINE_QUANTILE:.2f}"
    for cell in benign:
        for column in columns:
            block = values.get((cell["folder"], column))
            if block is None:
                continue
            row = [cell["dataset"], column]
            for q in quantile_keys:
                row += [
                    cell_text(metric(block, "fpr", q)),
                    cell_text(metric(block, "tpr", q)),
                ]
            row.append(cell_text(metric(block, "auroc", headline)))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def cost_table(columns: list[str], declaration: dict) -> list[str]:
    """Forward passes per input and clean data needed, so a column's price is visible."""
    passes = declaration["panel"]["forward_passes"]
    lines = [
        "### Cost and data",
        "",
        "| column | forwards per input | clean data |",
        "|---|---:|---|",
    ]
    for column in columns:
        if column in PSBD_COLUMNS:
            lines.append(
                f"| {column} | {passes + 1} | the clean validation split, unlabelled, "
                "for the rate rule |"
            )
        else:
            lines.append(
                f"| {column} | {FORWARD_PASSES_PER_INPUT[column]} | "
                f"{DATA_REQUIREMENT[column]} |"
            )
    lines.append("")
    return lines


def footer(
    cells: list[dict], columns: list[str], values: dict, notes: dict
) -> list[str]:
    """What is missing, so an aggregate is never read as complete when it is not."""
    lines = ["### Coverage", ""]
    for column in columns:
        missing = [c["folder"] for c in cells if (c["folder"], column) not in values]
        line = f"- {column}: {len(cells) - len(missing)} of {len(cells)} cells"
        if missing:
            line += (
                f", missing {', '.join(missing)}"
                if len(missing) <= 12
                else f", missing {len(missing)}"
            )
        lines.append(line)
    if notes["failed"]:
        lines.append(
            f"- failed records: {len(notes['failed'])}: "
            + ", ".join(f"{f}/{n}" for f, n in notes["failed"][:12])
        )
    if notes["smoke"]:
        lines.append(
            f"- smoke records ignored (max_samples set): {len(notes['smoke'])}"
        )
    if notes["psbd_none"]:
        lines.append(
            f"- PSBD rule chose no rate: {len(notes['psbd_none'])}: "
            + ", ".join(f"{f}/{c}" for f, c in notes["psbd_none"][:12])
        )
    lines.append(f"- superseded baseline_metrics.json files ignored: {notes['legacy']}")
    lines.append("")
    return lines


def render(cells, columns, values, notes, args, declaration) -> list[str]:
    """Every table, as markdown lines."""
    quantile_keys = [f"q{value:.2f}" for value in args.fpr]
    headline = f"q{HEADLINE_QUANTILE:.2f}"
    lines = [
        "# Detector comparison",
        "",
        f"Generated by `python -m cli.compare_detectors` at {utc_timestamp()} from "
        f"`{args.results_dir}`, commit {current_git_commit()}. Do not edit by hand.",
        "",
        f"PSBD columns: recommended placement `{RECOMMENDED_PLACEMENT}` at the adaptive "
        f"rule (shift ratio {ADAPTIVE_SHIFT_TARGET}) and at the matched rule (nearest "
        f"{PLACEMENT_MATCH_TARGET}), and the published placement `{PUBLISHED_PLACEMENT}` "
        "at the adaptive rule. Fractional PSU throughout. Every detector shares the "
        "split, the clean-validation quantile threshold and the pairing. A `*` on n_bd "
        f"marks fewer than {SMALL_POSITIVE_SET} triggered images.",
        "",
    ]
    lines += metric_table("AUROC", cells, columns, values, "auroc", headline, args)
    for q in quantile_keys:
        lines += metric_table(
            f"TPR at FPR budget {q}", cells, columns, values, "tpr", q, args
        )
        lines += metric_table(
            f"Achieved FPR at budget {q}", cells, columns, values, "fpr", q, args
        )
    lines += benign_table(cells, columns, values, quantile_keys)
    lines += cost_table(columns, declaration)
    lines += footer(cells, columns, values, notes)
    return lines


def write_csv(path: str, cells, columns, values, quantile_keys) -> None:
    """1 row per (cell, column, quantile) with every field of the block."""
    fields = (
        "auroc",
        "tpr",
        "fpr",
        "threshold",
        "tie_share_at_threshold",
        "tpr_interpolated",
    )
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["folder", "dataset", "attack", "poison_rate", "kind", "column", "quantile"]
            + list(fields)
        )
        for cell in cells:
            for column in columns:
                block = values.get((cell["folder"], column))
                if block is None:
                    continue
                for q in quantile_keys:
                    if q not in block:
                        continue
                    report = block[q]
                    writer.writerow(
                        [
                            cell["folder"],
                            cell["dataset"],
                            cell["attack"],
                            cell["poison_rate"],
                            cell["kind"],
                            column,
                            q,
                        ]
                        + [report.get(field) for field in fields]
                    )


def rewrite_results_block(path: str, body: list[str]) -> None:
    """Replace the results block of a detector doc, refusing a doc without the markers."""
    with open(path) as handle:
        text = handle.read()
    begin = text.find(RESULTS_BLOCK_BEGIN)
    end = text.find(RESULTS_BLOCK_END)
    if begin < 0 or end < 0 or end < begin:
        raise ValueError(
            f"{path} has no {RESULTS_BLOCK_BEGIN} ... {RESULTS_BLOCK_END} block"
        )
    replacement = RESULTS_BLOCK_BEGIN + "\n" + "\n".join(body) + "\n"
    rewritten = text[:begin] + replacement + text[end:]
    with open(path, "w") as handle:
        handle.write(rewritten)


def per_detector_blocks(cells, columns, values, args) -> dict[str, list[str]]:
    """The results block of each detector doc: its column beside psbd_adaptive."""
    headline = f"q{HEADLINE_QUANTILE:.2f}"
    blocks = {}
    for name in columns:
        if name in PSBD_COLUMNS:
            continue
        pair = ["psbd_adaptive", name]
        body = [
            f"Generated by `python -m cli.compare_detectors --per-detector-dir` at "
            f"{utc_timestamp()}.",
            "",
        ]
        body += metric_table("AUROC", cells, pair, values, "auroc", headline, args)
        for value in args.fpr:
            body += metric_table(
                f"TPR at FPR budget q{value:.2f}",
                cells,
                pair,
                values,
                "tpr",
                f"q{value:.2f}",
                args,
            )
        blocks[name] = body
    return blocks


def main() -> int:
    args = build_parser().parse_args()
    coverage_path = args.coverage or os.path.join(
        args.results_dir, "coverage", "coverage.json"
    )
    coverage = read_json(coverage_path)
    declaration = read_json(args.declaration)
    if coverage is None or declaration is None:
        raise SystemExit(f"need {coverage_path} and {args.declaration}")

    cells = panel_cells(coverage, declaration, args.include_sam)
    notes = {"failed": [], "smoke": [], "psbd_none": [], "legacy": 0}
    columns, values = collect_columns(cells, args, notes)
    lines = render(cells, columns, values, notes, args, declaration)
    print("\n".join(lines))

    quantile_keys = [f"q{value:.2f}" for value in args.fpr]
    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
        with open(args.markdown, "w") as handle:
            handle.write("\n".join(lines) + "\n")
    if args.csv:
        write_csv(args.csv, cells, columns, values, quantile_keys)
    if args.per_detector_dir:
        for name, body in per_detector_blocks(cells, columns, values, args).items():
            rewrite_results_block(
                os.path.join(args.per_detector_dir, f"{name}.md"), body
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
