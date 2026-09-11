"""1 entrypoint, 4 subcommands, folding 7 reporting commands into 1.

Every subcommand below used to be its own `cli.<name>` module. Folding them here
means 1 place to add a flag to the reporting layer instead of 7, while every flag,
every default and every output path stays exactly what it was. The 7 old modules
still work: each is now a 2-line shim that forwards its argv to the matching
subcommand here, so PBS scripts and doc examples that spell `python -m cli.report`
keep running unchanged.

  placements          what `cli.report`, `cli.summary`, `cli.tables` and
                       `cli.variants` did for positions and operators. A further
                       action picks which of the 4: `report`, `summary`,
                       `tables`, `variants`.
  operating-points     what `cli.operating_points` did: detection at low
                       false-positive rates.
  detectors            what `cli.compare_detectors` did: PSBD against every
                       competitor detector.
  fused                what `cli.fuse_detectors` did: PSBD fused with 1
                       competitor by rank.

Example
    python -m cli.compare placements report --format markdown
    python -m cli.compare placements summary --output results/detection_summary.csv
    python -m cli.compare placements tables --coverage-only
    python -m cli.compare placements variants --held-out vit_cifar10_sig_0_1
    python -m cli.compare operating-points --fpr 0.01 0.05
    python -m cli.compare detectors --markdown docs/detectors/comparison.md
    python -m cli.compare fused --detector strip --fpr 0.01 0.05 0.25
"""

import argparse
import csv
import glob
import gzip
import json
import os
import random
import re
import statistics
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from data.splits import SPLITS, read_checkpoint_metadata
from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import (
    ADAPTIVE_SHIFT_TARGET,
    EASY_ATTACKS,
    HARD_ATTACKS,
    HEADLINE_QUANTILE,
    PLACEMENT_MATCH_TARGET,
    PRIMARY_DATASETS,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    SHIFT_MATCH_TARGETS,
    attack_success_mask,
    complete_rates,
    pair_clean_to_backdoor,
    select_rate_adaptively,
    select_rate_at_matched_shift,
    shift_key,
    threshold_at_quantile,
)
from defences.scores import psu_from_cache, psu_ratio_from_cache, shift_ratio, to_rank
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
    load_scores,
    report_path,
    scores_path,
)
from models.positions import DROPOUT_CONFIGS, POSITION_REGISTRY
from utils.provenance import current_git_commit, utc_timestamp


def read_json(path: str) -> dict | None:
    """A JSON file, or None when it does not exist.

    Shared by every subcommand below that reads a psbd_metrics.json or a
    detector record, so a read that comes back empty means the same thing
    everywhere: the file was never written, not that it failed to parse.
    """
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        loaded = json.load(handle)
    return loaded


# placements report
# Ported from cli/report.py. Aggregates every psbd_metrics.json into a single
# per-checkpoint table.
#
# A row per (checkpoint, placement). ASR and clean accuracy are read from the
# checkpoint's own metrics.json rather than recomputed, since that is the measured,
# eligibility-correct number and a second implementation could disagree with it.
#
# 3 detection columns, because they answer different questions:
#
#   adaptive   the rate a defender could actually have chosen, using the paper's
#              rule on clean validation data only. This is the honest headline.
#   oracle     the best rate in hindsight. An upper bound, not a result: choosing
#              it reads the poison labels the defence is supposed to predict. The
#              gap to adaptive is the cost of not knowing the right rate.
#   matched    every placement at the same measured disturbance (clean-validation
#              shift ratio), which is the only way to compare placements without
#              the comparison collapsing into "which one perturbs harder".
#
# FPR is printed but carries little information here: the threshold is a quantile
# of clean test PSU and FPR is measured on clean test PSU, so it sits near the
# quantile whatever the placement does. TPR and AUROC are the signal.

REPORT_HELP = """Aggregate every psbd_metrics.json into a single per-checkpoint table.

A row per (checkpoint, placement). ASR and clean accuracy are read from the
checkpoint's own metrics.json rather than recomputed, since that is the measured,
eligibility-correct number and a second implementation could disagree with it.

3 detection columns, because they answer different questions:

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
"""

REPORT_HEADLINE_QUANTILE_KEY = "q0.25"
REPORT_MARKDOWN_COLUMN_COUNT = 15


def report_collect_rows(
    results_dir: str, checkpoints_dir: str, matched_key: str
) -> list[dict]:
    """A row per (checkpoint, placement), sorted for a stable diff."""
    rows = []
    for folder in sorted(os.listdir(results_dir)):
        report = read_json(os.path.join(results_dir, folder, "psbd_metrics.json"))
        if report is None:
            continue
        baseline = (
            read_json(os.path.join(checkpoints_dir, folder, "metrics.json")) or {}
        )

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


def report_cell(value, spec: str = ".3f") -> str:
    """A table cell, with a placeholder for a value the report never produced."""
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:{spec}}"
    return str(value)


def report_render_markdown(rows: list[dict], matched_key: str) -> str:
    """The full per-(checkpoint, placement) table as a markdown block."""
    header = (
        "| checkpoint | attack | pr | opt | ASR | CA | placement | "
        "adapt p | adapt TPR | adapt FPR | adapt AUROC | oracle p | oracle AUROC | "
        f"{matched_key} p | {matched_key} AUROC |"
    )
    divider = "|" + "---|" * REPORT_MARKDOWN_COLUMN_COUNT
    lines = [header, divider]

    for row in rows:
        optimizer = row["optimizer"] or "--"
        if row["rho"]:
            optimizer = f"{optimizer} {row['rho']}"
        cells = [
            row["folder"],
            str(row["attack"]),
            report_cell(row["poison_rate"], ".3f"),
            optimizer,
            report_cell(row["asr"]),
            report_cell(row["clean_accuracy"]),
            row["placement"],
            report_cell(row["adaptive_rate"], ".2f"),
            report_cell(row["adaptive_tpr"]),
            report_cell(row["adaptive_fpr"]),
            report_cell(row["adaptive_auroc"]),
            report_cell(row["oracle_rate"], ".2f"),
            report_cell(row["oracle_auroc"]),
            report_cell(row["matched_rate"], ".2f"),
            report_cell(row["matched_auroc"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    return table


def report_render_csv(rows: list[dict]) -> None:
    """Write every row to stdout as CSV, for a spreadsheet or a later join."""
    if not rows:
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def report_summarize_by_placement(rows: list[dict]) -> str:
    """Mean AUROC per placement, the single number the whole study is about.

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

    lines = [
        "",
        "## Mean AUROC by placement (adaptive rate, backdoored models only)",
        "",
        "| placement | n | mean AUROC | min | max | no adaptive rate |",
        "|---|---|---|---|---|---|",
    ]
    for placement in sorted(set(list(by_placement) + list(missing))):
        values = by_placement.get(placement, [])
        if not values:
            lines.append(
                f"| {placement} | 0 | -- | -- | -- | {missing.get(placement, 0)} |"
            )
            continue
        lines.append(
            f"| {placement} | {len(values)} | {sum(values) / len(values):.3f} | "
            f"{min(values):.3f} | {max(values):.3f} | {missing.get(placement, 0)} |"
        )

    summary = "\n".join(lines)
    return summary


def report_print_negative_control(benign_rows: list[dict]) -> None:
    """The benign models, whose detection must sit near chance."""
    print("")
    print("## Negative control (benign model, same trigger)")
    print("")
    print("Detection here should sit near 0.5. Anything much above it means the")
    print("probe responds to the trigger perturbation rather than to a backdoor.")
    print("")
    for row in benign_rows:
        print(
            f"- {row['folder']} / {row['placement']}: "
            f"AUROC {report_cell(row['adaptive_auroc'])} at "
            f"p={report_cell(row['adaptive_rate'], '.2f')}"
        )


def add_report_parser(subparsers) -> None:
    sub = subparsers.add_parser(
        "report",
        help="aggregate every psbd_metrics.json into a per-checkpoint table",
        description=REPORT_HELP,
    )
    sub.add_argument("--results-dir", default="results")
    sub.add_argument("--checkpoints-dir", default="checkpoints")
    sub.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    sub.add_argument(
        "--matched-shift",
        type=float,
        default=0.8,
        choices=SHIFT_MATCH_TARGETS,
        help="which matched-disturbance shift ratio to print",
    )
    sub.set_defaults(func=run_report)


def run_report(args: argparse.Namespace) -> int:
    matched_key = shift_key(args.matched_shift)
    rows = report_collect_rows(args.results_dir, args.checkpoints_dir, matched_key)
    if not rows:
        raise SystemExit(f"no psbd_metrics.json found under {args.results_dir}")

    if args.format == "csv":
        report_render_csv(rows)
        return 0

    backdoored = [row for row in rows if row["attack"] != "benign"]
    benign = [row for row in rows if row["attack"] == "benign"]

    print(report_render_markdown(rows, matched_key))
    print(report_summarize_by_placement(backdoored))
    if benign:
        report_print_negative_control(benign)
    return 0


# placements summary
# Ported from cli/summary.py. Collapses every psbd_metrics.json into a single
# compact, versionable table.
#
# The full stage-2 record is about 2 MB per checkpoint and 1.2 GB across the tree,
# which is regenerable from the cached tensors and far too large to keep in git.
# What every table in the paper actually reads is a handful of numbers per
# (checkpoint, placement): the operating point, the rate that produced it and how
# well it separated. That is what this writes.

SUMMARY_HELP = """Collapse every psbd_metrics.json into a single compact, versionable table.

A row per (checkpoint, placement, rate-selection rule). The 3 rules answer
different questions and are all kept, because any 1 of them alone hides what the
other 2 show:

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
value a table cannot tell a matched cell from an unmatched cell.
"""

SELECTION_RULES = ("adaptive", "oracle")

# Every position name a bare placement may legitimately carry, across both
# architectures plus the named multi-position configs.
VALID_POSITION_NAMES: frozenset[str] = frozenset(
    name for positions in POSITION_REGISTRY.values() for name in positions
) | frozenset(DROPOUT_CONFIGS)

# Suffixes that qualify a placement without naming its operator. Stripped before
# the operator is read, and recorded so a variant can be filtered rather than
# silently pooled with the plain measurement.
VARIANT_SUFFIXES = (
    # The superseded batch-coupled Gaussian. Its caches were archived out of
    # results/, but the psbd_metrics.json records they produced were not, so they
    # still reach this script and must be labelled.
    ("_gaussian_batchstd", "gaussian", "batch_coupled_superseded"),
)

# A trailing _k<digits> is a Monte Carlo pass-count variant, not an operator.
PASS_COUNT_SUFFIX = re.compile(r"_k(\d+)$")

# A trailing _pmodel<rate> marks a run with the model's OWN dropouts activated,
# the deliberate conflation study. It qualifies the run, not the operator.
MODEL_DROPOUT_SUFFIX = re.compile(r"_pmodel([0-9_.]+)$")

# A trailing _blocks_<first>_<last> restricts a block-scope position to a span of
# blocks. It qualifies the position rather than naming an operator, and the depth
# band sweep is entirely made of these.
BLOCK_RANGE_SUFFIX = re.compile(r"_blocks_(\d+)_(\d+)$")

# A trailing _seed<digits> is a different draw of the same stochastic estimator,
# written by --mask-seed. Seed 0 keeps the bare name, so only replicates carry it.
# Without this rule a seeded placement parses as an unknown operator and its rows
# leave the summary entirely.
MASK_SEED_SUFFIX = re.compile(r"_seed(\d+)$")

KNOWN_OPERATORS = (
    "channel_mask",
    "gain_scale",
    "token_mask",
    "head_mask",
    "droppath",
    "gaussian",
    "rademacher",
    "scale_up",
)

UNKNOWN_OPERATOR = "unknown"

# Every suffix that qualifies a run without naming an operator, with the variant
# label it produces. Order here does not matter, since the parser loops until no
# pattern matches, so stacked qualifiers come off whatever order they were written in.
QUALIFIER_SUFFIXES = (
    (MODEL_DROPOUT_SUFFIX, "model_dropout_{0}"),
    (BLOCK_RANGE_SUFFIX, "blocks_{0}_{1}"),
    (PASS_COUNT_SUFFIX, "passes_{0}"),
    (MASK_SEED_SUFFIX, "mask_seed_{0}"),
)


def split_operator(
    placement: str, known_operators: tuple[str, ...]
) -> tuple[str, str, str | None]:
    """(position, operator, variant) from a placement name.

    A bare name is the paper's dropout, which is why the naming keeps it bare.
    Longest operator names are tried first so channel_mask is not read as a
    position ending in mask.

    An unrecognized placement is labelled UNKNOWN_OPERATOR rather than falling
    back to dropout. Dropout is the baseline every other operator is compared
    against, so a suffix silently attributed to it would inflate the thing every
    margin is measured from. Labelled, it can be found and excluded.
    """
    variant = None
    remaining = placement

    for suffix, operator, variant_name in VARIANT_SUFFIXES:
        if remaining.endswith(suffix):
            position = remaining[: -len(suffix)]
            return position, operator, variant_name

    # Qualifiers stack, and the sweep writes them in whatever order the flags were
    # given, so a single pass in a fixed order cannot work. Every pattern is
    # anchored at the end of the string, which means an outer qualifier hides an
    # inner qualifier: pre_residual_blocks_9_16_seed1 does not match the block-range
    # pattern at all until _seed1 comes off. Strip repeatedly until nothing more
    # matches, and keep every qualifier found rather than letting the last one win.
    qualifiers = []
    stripping = True
    while stripping:
        stripping = False
        for pattern, template in QUALIFIER_SUFFIXES:
            found = pattern.search(remaining)
            if found is not None:
                qualifiers.append(template.format(*found.groups()))
                remaining = remaining[: found.start()]
                stripping = True
    if qualifiers:
        # Innermost first, so the name reads in the order the suffixes appear.
        variant = "+".join(reversed(qualifiers))

    for operator in known_operators:
        if remaining.endswith(f"_{operator}"):
            position = remaining[: -len(operator) - 1]
            return position, operator, variant

    if remaining in VALID_POSITION_NAMES:
        return remaining, "dropout", variant

    return remaining, UNKNOWN_OPERATOR, variant


def placement_is_cache_backed(results_dir: str, folder: str, placement: str) -> bool:
    """Whether a stage-2 record still has its stage-1 tensors under results/.

    A psbd_metrics.json entry outlives the cache it was computed from. When a
    superseded operator's caches are archived out of results/, the stage-2 records
    they produced stay behind and read as plain measurements, and the suffix tag in
    VARIANT_SUFFIXES only catches placements literally named for the variant.

    Checking for the directory is the rule that does not depend on a naming
    convention. A record whose tensors are gone cannot be recomputed, verified or
    trusted, whatever it is called.
    """
    cache_directory = os.path.join(results_dir, folder, "psbd", placement)
    return os.path.isdir(cache_directory)


# The PSU variants each per-rate block records. The absolute form is the paper's,
# the ratio form divides by the starting confidence, and captured-only restricts
# BOTH splits to the images the trigger actually flipped.
PSU_VARIANTS = {
    "absolute": "detection",
    "fractional": "detection_psu_ratio",
    "captured_only": "detection_captured_only",
}

SUMMARY_OPERATING_POINT_FIELDS = (
    "folder",
    "architecture",
    "dataset",
    "attack",
    "poison_rate",
    "realized_poison_rate",
    "poison_rate_capped",
    "asr",
    "placement",
    "position",
    "operator",
    "variant",
    "cache_backed",
    "psu_variant",
    "rate",
    "sigma_validation",
    "quantile",
    "tpr",
    "fpr",
    "auroc",
)


def summary_operating_point_rows(folder, report, metadata, results_dir="results"):
    """A row per placement, rate, PSU variant and quantile.

    The compact summary keeps only the headline 0.25 quantile, which is a 25
    percent clean loss and not an operating point any deployment would run at. A
    security venue asks for TPR at 1 and 5 percent FPR, and every per-rate block
    already records all 6 quantiles, so this reads them out rather than
    recomputing anything.

    Shares split_operator and placement_is_cache_backed with the compact summary
    deliberately. A second parser is how 2 tables come to disagree.
    """
    base = {
        "folder": folder,
        "architecture": metadata.get("architecture"),
        "dataset": metadata.get("dataset"),
        "attack": metadata.get("attack"),
        # The requested rate is a request. A clean-label attack is eligible only on
        # the target class, so it saturates and 3 folder names can describe 1 run.
        # The compact summary already carries both. Without them here every
        # consumer of operating_points.csv reads a rate that was never applied.
        "poison_rate": metadata.get("poison_rate"),
        "realized_poison_rate": metadata.get("realized_poison_rate"),
        "poison_rate_capped": metadata.get("poison_rate_capped"),
        "asr": metadata.get("asr"),
    }

    rows = []
    for placement, block in sorted(report.get("placements", {}).items()):
        position, operator, variant = split_operator(placement, KNOWN_OPERATORS)
        shared = {
            **base,
            "placement": placement,
            "position": position,
            "operator": operator,
            "variant": variant,
            "cache_backed": placement_is_cache_backed(results_dir, folder, placement),
        }
        for entry in block.get("rates", []):
            # The clean-validation shift ratio is what makes 2 placements
            # comparable, so it travels with every row rather than being looked
            # up separately later.
            sigma = (entry.get("shift_ratio") or {}).get("validation")
            for variant_name, key in PSU_VARIANTS.items():
                detection = entry.get(key)
                if not detection:
                    continue
                for quantile_key, values in detection.items():
                    rows.append(
                        {
                            **shared,
                            "psu_variant": variant_name,
                            "rate": entry.get("rate"),
                            "sigma_validation": sigma,
                            "quantile": values.get("quantile"),
                            "tpr": values.get("tpr"),
                            "fpr": values.get("fpr"),
                            "auroc": values.get("auroc"),
                        }
                    )
    return rows


def summary_rows_for_checkpoint(
    folder: str, report: dict, metadata: dict, results_dir: str = "results"
) -> list[dict]:
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
        position, operator, variant = split_operator(placement, KNOWN_OPERATORS)
        shared = {
            **base,
            "placement": placement,
            "position": position,
            "operator": operator,
            # None for a plain measurement. Anything else marks a row that must
            # not be pooled with the plain measurement without saying so.
            "variant": variant,
            # False means the stage-1 tensors this row came from are no longer
            # under results/, so the row cannot be recomputed or verified.
            "cache_backed": placement_is_cache_backed(results_dir, folder, placement),
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


SUMMARY_FIELDS = (
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
    "variant",
    "cache_backed",
    "rule",
    "rate",
    "achieved_shift_ratio",
    "auroc",
    "tpr",
    "fpr",
    "threshold",
    "quantile",
)


def add_summary_parser(subparsers) -> None:
    sub = subparsers.add_parser(
        "summary",
        help="collapse every psbd_metrics.json into a compact, versionable CSV",
        description=SUMMARY_HELP,
    )
    sub.add_argument("--results-dir", default="results")
    sub.add_argument("--checkpoints-dir", default="checkpoints")
    sub.add_argument("--output", default="results/detection_summary.csv")
    sub.add_argument(
        "--operating-points",
        action="store_true",
        help=(
            "emit one row per placement, rate, PSU variant and quantile instead of "
            "the compact table, so TPR at 1 and 5 percent FPR is available rather "
            "than only the 25 percent headline"
        ),
    )
    sub.add_argument(
        "--include-sam",
        action="store_true",
        help=(
            "keep SAM-trained checkpoints. Off by default: SAM is a training-time "
            "change under a separate question, and carrying it dilutes every panel "
            "and reintroduces unequal coverage between groups"
        ),
    )
    sub.set_defaults(func=run_summary)


def run_summary(args: argparse.Namespace) -> int:
    all_rows = []
    skipped = 0
    excluded_sam = 0

    for path in sorted(
        glob.glob(os.path.join(args.results_dir, "*", "psbd_metrics.json"))
    ):
        folder = os.path.basename(os.path.dirname(path))
        report = read_json(path)
        metadata = (
            read_json(os.path.join(args.checkpoints_dir, folder, "args.json")) or {}
        )
        if report is None:
            skipped += 1
            continue
        if metadata.get("optimizer") == "sam" and not args.include_sam:
            excluded_sam += 1
            continue
        if args.operating_points:
            all_rows.extend(
                summary_operating_point_rows(folder, report, metadata, args.results_dir)
            )
        else:
            all_rows.extend(
                summary_rows_for_checkpoint(folder, report, metadata, args.results_dir)
            )

    with open(args.output, "w", newline="") as handle:
        fields = (
            SUMMARY_OPERATING_POINT_FIELDS if args.operating_points else SUMMARY_FIELDS
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    size_mb = os.path.getsize(args.output) / 1048576
    checkpoints = len({row["folder"] for row in all_rows})
    placements = len({row["placement"] for row in all_rows})
    print(f"rows:        {len(all_rows)}")
    print(f"checkpoints: {checkpoints}")
    print(f"placements:  {placements}")
    print(f"skipped:     {skipped}")
    print(f"sam excluded: {excluded_sam}")
    print(f"wrote {args.output} ({size_mb:.1f} MB)")
    return 0


# placements tables
# Ported from cli/tables.py. Per-dataset detection tables, with the coverage bar
# enforced in code.

TABLES_HELP = """Per-dataset detection tables, with the coverage bar enforced in code.

The reporting standard (docs/hypothesis/README.md) is that no conclusion is
reported unless it spans all 3 axes at once: 4 datasets, 3 poison rates and the
full attack panel. This refuses to emit a row that does not, and prints what is
missing instead. A number from 1 dataset does not generalise, and enforcing the
bar mechanically is more reliable than remembering to.

A cell is required only if its attack actually implants (ASR >= --min-asr). An
attack that failed to implant is reported as such and does not block coverage,
because there is no backdoor there to detect.

One-sided throughout: low score means poisoned, and a value below 0.5 is printed
as the failure it is, never re-signed.

Every target FPR is reported at 2 thresholds, and the gap between them is the
part that matters (the same distinction the operating-points subcommand draws):

  deployable  the threshold is the q-quantile of CLEAN VALIDATION score, with q
              set to the target FPR, exactly the rule the PSBD paper prescribes.
              This is what a defender can actually build, since it needs no
              poisoned data. The FPR it achieves on the paired clean analysis
              pool is printed beside it, because nothing guarantees the
              validation quantile transfers to the analysis pool.

  oracle      the target FPR read off the labelled ROC curve of the analysis
              pool. Not deployable: it places the threshold using the very
              clean/poison split the detector is supposed to find. Reported as an
              upper bound only.

The dropout rate is chosen by matching the clean-validation shift ratio to
--sigma, but the rate grid is not guaranteed to contain a rate that reaches the
target. The achieved sigma is therefore printed per cell and any cell further
than --sigma-tolerance from the target is marked, so a strength-matched claim can
never be made from a cell that was never matched.
"""

TABLES_DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
TABLES_POISON_TAGS = (("0_01", "1%"), ("0_05", "5%"), ("0_1", "10%"))
TABLES_PANEL = ("badnet_a2o", "blend", "wanet", "lc", "adaptive_blend")

TABLES_DEFAULT_FPRS = (0.01, 0.05, 0.10, 0.25)

# Which rate a cell is read at. These are 2 different protocols and the choice
# moves the numbers a long way, because the nearest rate to a mid-ladder sigma can
# sit well below the rate that actually separates.
#
#   deployable  PSBD's own rule: the smallest rate reaching the target. This is
#               what a defender executes and what a detection number must be read
#               at. Returns None when the grid never reaches the target, which is
#               a real answer and not a gap to be filled by the nearest rate.
#
#   matched     the nearest rate to the target, for comparing a placement
#               against another at equal disturbance. It always returns something,
#               so it can silently report a cell that was never matched, which is
#               what --sigma-tolerance marks.
#
# This subcommand emits DETECTION tables, so it defaults to deployable. Pass
# --rate-rule matched only when the table's question is about placement.
TABLES_RATE_RULES = {
    "deployable": select_rate_adaptively,
    "matched": select_rate_at_matched_shift,
}
TABLES_DEFAULT_RATE_RULE = "deployable"

# Each rule has its own target, and pairing a rule with the other's target is the
# specific mistake this mapping exists to prevent.
TABLES_DEFAULT_SIGMA = {
    "deployable": ADAPTIVE_SHIFT_TARGET,
    "matched": PLACEMENT_MATCH_TARGET,
}

# Appended to the achieved sigma of any cell the rate grid could not match. A
# marked cell is still printed, because dropping it would hide that the grid is
# too coarse for this operator, but it must never be read as strength-matched.
TABLES_SIGMA_MISMATCH_MARKER = "*"

# Below this ASR the backdoor is present but unreliable, so the row is labelled
# rather than silently averaged in with the ones that implanted cleanly.
TABLES_WEAK_ASR = 0.8

TABLES_MAX_MISSING_SHOWN = 12


def tables_is_mismatched(result: dict, args: argparse.Namespace) -> bool:
    """Whether a cell's achieved sigma disqualifies it from a matched comparison.

    Only the "matched" rule can produce such a cell. The "deployable" rule selects
    on sigma >= target and returns None when nothing qualifies, so every cell it
    does return satisfies the constraint by construction and overshoot is the rule
    working as specified. Marking those would flag every correct cell.
    """
    if args.rate_rule != "matched":
        return False
    return abs(result["achieved_sigma"] - args.sigma) > args.sigma_tolerance


def tables_read_cell_metadata(checkpoints_dir: str, folder: str) -> dict:
    """ASR and poison-rate facts for a checkpoint, preferring args.json.

    args.json is authoritative and metrics.json is not. Every metrics.json on disk
    predates the source-restricted eval set, so its TaCT and Adaptive-Blend rows
    disagree with args.json by enough to decide whether a cell clears the ASR bar.
    args.json's value is corroborated independently by asr_from_cache, read back
    from the PSBD baseline cache.

    The requested rate is also a request. A clean-label attack is eligible only on
    the target class, so it saturates and 3 folder names can name 1 run. The
    realized rate says which.
    """
    args_path = os.path.join(checkpoints_dir, folder, "args.json")
    metrics_path = os.path.join(checkpoints_dir, folder, "metrics.json")

    metadata = {}
    for path in (metrics_path, args_path):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as handle:
                metadata.update(json.load(handle))
        except Exception:
            continue

    if not metadata:
        return {}
    return {
        "asr": metadata.get("asr"),
        "realized_poison_rate": metadata.get("realized_poison_rate"),
        "poison_rate_capped": metadata.get("poison_rate_capped"),
    }


def tables_read_asr(checkpoints_dir: str, folder: str) -> float | None:
    """The measured ASR, or None when unavailable."""
    return tables_read_cell_metadata(checkpoints_dir, folder).get("asr")


def tables_required_cells(
    checkpoints_dir: str, architecture: str, min_asr: float
) -> tuple[list[tuple], list[tuple]]:
    """Every (dataset, rate, attack) the bar demands, split into required and dead.

    Dead cells are carried rather than dropped so the tables can say "the attack
    failed to implant" instead of leaving a silent blank that reads like a
    detection failure.
    """
    required, dead = [], []
    for dataset in TABLES_DATASETS:
        for tag, label in TABLES_POISON_TAGS:
            for attack in TABLES_PANEL:
                folder = f"{architecture}_{dataset}_{attack}_{tag}"
                asr = tables_read_asr(checkpoints_dir, folder)
                if asr is None:
                    continue
                entry = (dataset, label, attack, folder, asr)
                (required if asr >= min_asr else dead).append(entry)

    return required, dead


def tables_cell_score(
    psbd_dir: str,
    name: str,
    kind: str,
    sigma_target: float,
    target_fprs: list[float],
    rate_rule: str = TABLES_DEFAULT_RATE_RULE,
) -> dict | None:
    """One-sided AUROC and both TPRs per target FPR, at the sigma-matched rate.

    Returns None when the position config holds no complete rate on disk.
    Otherwise returns the chosen rate, the clean-validation shift ratio that rate
    actually achieved, the one-sided AUROC and an operating point per entry of
    target_fprs in the order given.

    achieved_sigma is part of the contract because the "matched" rule returns the
    nearest rate unconditionally: it never fails, so the caller, not this
    function, has to decide whether the match was close enough to report as matched.
    The "deployable" rule returns None instead when the grid never reaches the
    target, so a None here means 2 different things depending on the rule and the
    caller has to know which one it asked for.
    """
    manifest = read_split_manifest(psbd_dir)

    # The validation baseline depends on the checkpoint alone, never on the rate,
    # so it is loaded once for the whole rate scan rather than inside it.
    validation_probs, validation_labels, _ = load_baseline(
        baseline_path(psbd_dir, "validation")
    )

    shift_by_rate = {}
    for rate in complete_rates(psbd_dir, name):
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, name, rate, "validation")
        )
        shift_by_rate[rate] = shift_ratio(validation_labels, argmax)

    chosen = TABLES_RATE_RULES[rate_rule](shift_by_rate, sigma_target)
    if chosen is None:
        return None
    achieved_sigma = shift_by_rate[chosen]

    build = psu_ratio_from_cache if kind == "fractional" else psu_from_cache

    # Validation has to be scored with the same rule as clean and backdoor, or the
    # deployable threshold is read off a distribution the detector never produces.
    validation_pass_probs, _ = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, name, chosen, "validation")
    )
    validation_score = build(
        validation_probs, validation_labels, validation_pass_probs
    )  # (n_validation,)

    scored = {}
    for split in ("clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, name, chosen, split)
        )
        scored[split] = build(probs, labels, per_pass)
    clean = pair_clean_to_backdoor(scored["clean"], manifest).float().numpy()
    backdoor = scored["backdoor"].float().numpy()
    assert len(clean) == len(backdoor), (
        "pairing must leave a clean row per backdoor row"
    )

    truth = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])

    # Negated once, because low score is the poisoned evidence and roc_auc_score
    # wants higher to mean more positive. Never negated a second time: an AUROC
    # below 0.5 is the detector failing, and re-signing it would hide that.
    ranking = np.concatenate([-clean, -backdoor])
    auroc = float(roc_auc_score(truth, ranking))
    curve_fpr, curve_tpr, _ = roc_curve(truth, ranking)

    operating_points = []
    for target in target_fprs:
        deployable_threshold = threshold_at_quantile(validation_score, target)
        operating_points.append(
            {
                "target_fpr": float(target),
                "threshold": deployable_threshold,
                "tpr_deployable": float((backdoor < deployable_threshold).mean()),
                "fpr_achieved": float((clean < deployable_threshold).mean()),
                "tpr_oracle": float(np.interp(target, curve_fpr, curve_tpr)),
            }
        )

    result = {
        "rate": chosen,
        "achieved_sigma": float(achieved_sigma),
        "auroc": auroc,
        "operating_points": operating_points,
    }
    return result


def tables_placement_name(position: str, operator: str) -> str:
    """The results/ subfolder a (position, operator) pair was cached under.

    The paper's dropout keeps the bare position name, which is what cli.sweep
    writes, so it is the one case with no operator suffix.
    """
    if operator == "dropout":
        return position
    return f"{position}_{operator}"


def tables_table_columns(target_fprs: list[float]) -> list[str]:
    """Column headings, in the order every row must fill them.

    %g rather than a percent format so a sub-1% target keeps its digits instead
    of collapsing to "0%".
    """
    columns = ["poison", "attack", "ASR", "sigma", "rate", "AUROC"]
    for target in target_fprs:
        columns.append(f"TPR@{target * 100:g}% depl (FPR)")
        columns.append(f"TPR@{target * 100:g}% oracle")
    return columns


def tables_score_every_cell(
    args: argparse.Namespace, name: str, required: list[tuple]
) -> tuple[dict, list[tuple]]:
    """Score every required cell that has a cache, and list the ones that do not."""
    have, missing = {}, []
    for dataset, label, attack, folder, asr in required:
        psbd_dir = os.path.join(args.results_dir, folder, "psbd")
        result = (
            tables_cell_score(
                psbd_dir, name, args.score, args.sigma, args.fpr, args.rate_rule
            )
            if os.path.isdir(os.path.join(psbd_dir, name))
            else None
        )
        if result is None:
            missing.append((dataset, label, attack))
        else:
            have[(dataset, label, attack)] = (result, asr)

    return have, missing


def tables_print_coverage(
    args: argparse.Namespace,
    required: list[tuple],
    dead: list[tuple],
    have: dict,
    missing: list[tuple],
) -> None:
    """How much of the bar this configuration actually meets, and where it falls short."""
    print(
        f"configuration: {args.operator} @ {args.position} "
        f"({args.score} PSU, sigma>={args.sigma})"
    )
    print(f"coverage: {len(have)}/{len(required)} required cells")
    print(
        f"  {len(dead)} cells excluded: attack did not implant at ASR>={args.min_asr}"
    )

    by_dataset = {dataset: [0, 0] for dataset in TABLES_DATASETS}
    for dataset, label, attack, _folder, _asr in required:
        by_dataset[dataset][1] += 1
        if (dataset, label, attack) in have:
            by_dataset[dataset][0] += 1
    print(
        "  per dataset: "
        + "  ".join(f"{d}={a}/{b}" for d, (a, b) in by_dataset.items())
    )

    unmatched = [
        key
        for key, (result, _asr) in have.items()
        if tables_is_mismatched(result, args)
    ]
    if unmatched:
        print(
            f"  {len(unmatched)} cells NOT strength-matched: achieved sigma is "
            f"further than {args.sigma_tolerance} from {args.sigma}, marked "
            f"'{TABLES_SIGMA_MISMATCH_MARKER}' in the tables"
        )

    if missing and not args.coverage_only:
        shown = missing[:TABLES_MAX_MISSING_SHOWN]
        print(f"\n  MISSING ({len(missing)}):")
        for dataset, label, attack in shown:
            print(f"    {dataset:9} {label:>4} {attack}")
        if len(missing) > len(shown):
            print(f"    ... and {len(missing) - len(shown)} more")


def tables_render_row(
    label: str, attack: str, result: dict, asr: float, marker: str
) -> str:
    """A table row: the cell's provenance, its AUROC and both TPRs per target FPR."""
    weak = " (weak)" if asr < TABLES_WEAK_ASR else ""
    cells = [
        label,
        f"`{attack}`{weak}",
        f"{asr:.3f}",
        f"{result['achieved_sigma']:.3f}{marker}",
        f"{result['rate']:g}",
        f"{result['auroc']:.3f}",
    ]
    for point in result["operating_points"]:
        cells.append(f"{point['tpr_deployable']:.3f} ({point['fpr_achieved']:.3f})")
        cells.append(f"{point['tpr_oracle']:.3f}")

    row = "| " + " | ".join(cells) + " |"
    return row


def tables_print_dataset_table(
    dataset: str,
    args: argparse.Namespace,
    required: list[tuple],
    dead: list[tuple],
    have: dict,
    columns: list[str],
) -> None:
    """A dataset's table, plus the did-not-implant rows underneath it."""
    rows = [
        (label, attack, have[(dataset, label, attack)])
        for d, label, attack, _folder, _asr in required
        if d == dataset and (dataset, label, attack) in have
    ]
    if not rows:
        return

    rule_note = (
        f"smallest rate reaching sigma {args.sigma}"
        if args.rate_rule == "deployable"
        else f"nearest rate to sigma {args.sigma}"
    )
    print(f"\n### {dataset}   (one-sided, {args.score} PSU, {rule_note})\n")
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")

    marked_any = False
    for label, attack, (result, asr) in rows:
        mismatched = tables_is_mismatched(result, args)
        marked_any = marked_any or mismatched
        marker = TABLES_SIGMA_MISMATCH_MARKER if mismatched else ""
        print(tables_render_row(label, attack, result, asr, marker))

    for d, label, attack, _folder, asr in dead:
        if d != dataset:
            continue
        cells = [label, f"`{attack}`", f"{asr:.3f}", "--", "--", "did not implant"]
        cells += ["--"] * (2 * len(args.fpr))
        print("| " + " | ".join(cells) + " |")

    if marked_any:
        print(
            f"\n{TABLES_SIGMA_MISMATCH_MARKER} the rate grid holds no rate landing "
            f"within {args.sigma_tolerance} of sigma {args.sigma} for this "
            "operator, so the nearest achieved sigma is shown instead (it may "
            "under or overshoot). These cells are NOT strength-matched and "
            "must not be compared against matched cells."
        )


def validate_tables_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Post-parse checks tables needs, run from a live parser so errors read the same.

    A target outside (0, 1) is not a false-positive rate. np.quantile would raise
    deep inside the scoring loop instead of here at the boundary.
    """
    for target in args.fpr:
        if not 0.0 < target < 1.0:
            parser.error(f"--fpr values must lie in (0, 1), got {target}")
    if args.sigma_tolerance < 0.0:
        parser.error("--sigma-tolerance must not be negative")
    if args.sigma is None:
        args.sigma = TABLES_DEFAULT_SIGMA[args.rate_rule]


def add_tables_parser(subparsers) -> None:
    sub = subparsers.add_parser(
        "tables",
        help="per-dataset detection tables, with the coverage bar enforced in code",
        description=TABLES_HELP,
    )
    sub.add_argument("--results-dir", default="results")
    sub.add_argument("--checkpoints-dir", default="checkpoints")
    sub.add_argument("--architecture", default="vit")
    sub.add_argument("--operator", default="dropout")
    sub.add_argument("--position", default="before_attention_norm")
    sub.add_argument(
        "--score", default="fractional", choices=("fractional", "absolute")
    )
    sub.add_argument(
        "--rate-rule",
        default=TABLES_DEFAULT_RATE_RULE,
        choices=tuple(TABLES_RATE_RULES),
        help="deployable (PSBD's own: smallest rate reaching --sigma) or matched "
        "(nearest rate to --sigma, for placement comparison only).",
    )
    sub.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="target clean-validation shift ratio. Defaults to the target that "
        f"belongs to --rate-rule: {TABLES_DEFAULT_SIGMA}.",
    )
    sub.add_argument(
        "--sigma-tolerance",
        type=float,
        default=0.1,
        help="how far the achieved clean-validation shift ratio may sit from "
        "--sigma before the cell is marked as not strength-matched.",
    )
    sub.add_argument(
        "--fpr",
        nargs="+",
        type=float,
        default=list(TABLES_DEFAULT_FPRS),
        help="target false-positive rates, one column pair per value.",
    )
    sub.add_argument("--min-asr", type=float, default=0.5)
    sub.add_argument(
        "--include-sam",
        action="store_true",
        help="keep SAM checkpoints. Off by default, see the placements summary action",
    )
    sub.add_argument("--coverage-only", action="store_true")
    sub.add_argument(
        "--allow-partial",
        action="store_true",
        help="print the table even when the bar is not met. Anything produced "
        "this way is PROVISIONAL and must be labelled so wherever it is used.",
    )
    sub.set_defaults(func=run_tables, _validate=validate_tables_args)


def run_tables(args: argparse.Namespace) -> int:
    name = tables_placement_name(args.position, args.operator)

    required, dead = tables_required_cells(
        args.checkpoints_dir, args.architecture, args.min_asr
    )
    have, missing = tables_score_every_cell(args, name, required)
    tables_print_coverage(args, required, dead, have, missing)

    if args.coverage_only:
        return 0
    if missing and not args.allow_partial:
        print(
            "\nREFUSED: the coverage bar is not met, so no table is emitted.\n"
            "Rerun once the missing cells land, or pass --allow-partial and label\n"
            "every number it produces PROVISIONAL."
        )
        raise SystemExit(1)

    print(
        "\ndepl = deployable: threshold at the target-FPR quantile of clean "
        "validation score, needing no poisoned data, with the FPR it actually "
        "achieved on the paired clean pool in brackets. oracle = the same target "
        "read off the labelled ROC curve, an upper bound no defender can reach. "
        "Deployable can exceed oracle only by overspending the clean budget, which "
        "the bracketed FPR makes visible."
    )

    columns = tables_table_columns(args.fpr)
    for dataset in TABLES_DATASETS:
        tables_print_dataset_table(dataset, args, required, dead, have, columns)
    return 0


# placements variants
# Ported from cli/variants.py. Head-to-head: PSBD as published against the
# recommended ViT configuration.

VARIANTS_HELP = """Head-to-head: PSBD as published against the recommended ViT configuration.

Every other result measures a single knob in isolation. This assembles the knobs
into 2 complete, runnable defences and scores them the same way, which is the
comparison a reader needs.

Neither configuration sees a poison label. Both pick their rate by the paper's
adaptive rule on clean validation data, never by best AUROC, and both flag low
scores below the same clean-validation quantile. The oracle numbers reported
elsewhere are upper bounds and do not appear here.

  psbd_paper      the method as published, ported to ViT
                    placement  post_residual dropout, every block, the ConvNet
                               placement (defences.decision.PUBLISHED_PLACEMENT)
                    score      absolute PSU, P_c - mean_k(P_c_dropout)
                    rate       smallest p with clean-validation shift ratio >= 0.8
                    threshold  25th percentile of clean-validation score

  psbd_vit        the same method with the 2 changes this project recommends
                    placement  token_mask at before_attention_norm, every block
                               (defences.decision.RECOMMENDED_PLACEMENT)
                    score      fractional PSU, 1 - mean_k(P_c_dropout)/P_c
                    rate       the same 0.8 rule
                    threshold  unchanged, 25th percentile

The recommended placement was selected on this panel, so psbd_vit's numbers on
the panel are optimistic and labelled as such. A checkpoint named --held-out was
not used to choose anything and is reported separately.
"""

PLACEMENT_VARIANTS = {
    "psbd_paper": {
        "placement": PUBLISHED_PLACEMENT,
        "score": "absolute",
        "shift_target": ADAPTIVE_SHIFT_TARGET,
    },
    "psbd_vit": {
        "placement": RECOMMENDED_PLACEMENT,
        "score": "fractional",
        "shift_target": ADAPTIVE_SHIFT_TARGET,
    },
}

VARIANTS_TABLE_WIDTH = 118


def variants_score_split(
    psbd_dir: str, placement: str, rate: float, split: str, kind: str
):
    """A split's per-sample score and its shift ratio, at a rate."""
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    per_pass, argmax = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, placement, rate, split)
    )
    build = psu_ratio_from_cache if kind == "fractional" else psu_from_cache

    return build(probs, labels, per_pass), shift_ratio(labels, argmax)


def variants_select_rate(psbd_dir: str, placement: str, kind: str, shift_target: float):
    """The smallest rate whose clean-validation shift ratio reaches the target.

    Chosen on clean validation data alone, exactly as the paper's rule prescribes,
    and with the same target for both variants.
    """
    for rate in complete_rates(psbd_dir, placement):
        _, sigma = variants_score_split(psbd_dir, placement, rate, "validation", kind)
        if sigma is not None and sigma >= shift_target:
            return rate
    return None


def variants_evaluate(psbd_dir: str, config: dict) -> dict | None:
    """A complete defence run on a checkpoint, or None if it cannot run."""
    placement, kind = config["placement"], config["score"]
    if not complete_rates(psbd_dir, placement):
        return None
    manifest = read_split_manifest(psbd_dir)

    chosen = variants_select_rate(psbd_dir, placement, kind, config["shift_target"])
    if chosen is None:
        return {"rate": None, "reason": "no swept rate reaches the shift-ratio target"}

    scores = {}
    for split in SPLITS:
        scores[split], _ = variants_score_split(
            psbd_dir, placement, chosen, split, kind
        )
    clean = pair_clean_to_backdoor(scores["clean"], manifest)
    backdoor = scores["backdoor"]

    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    ranking = np.concatenate([-clean.float().numpy(), -backdoor.float().numpy()])
    auroc = float(roc_auc_score(labels, ranking))

    # The same rule for both variants: flag a score below the clean-validation quantile.
    # An AUROC under 0.5 is reported as inverted, never rescued by flipping the rule.
    threshold = threshold_at_quantile(scores["validation"], HEADLINE_QUANTILE)
    tpr = float((backdoor < threshold).float().mean())
    fpr = float((clean < threshold).float().mean())
    direction = "inverted" if auroc < 0.5 else "as_expected"

    result = {
        "rate": chosen,
        "auroc": auroc,
        "tpr": tpr,
        "fpr": fpr,
        "direction": direction,
    }
    return result


def variants_collect_rows(results_dir: str, checkpoints_dir: str) -> list[tuple]:
    """Both variants evaluated on every checkpoint with a stage-1 cache.

    SAM checkpoints are skipped: SAM is a training-time change under a separate
    question, and carrying it here would double every row of this comparison.
    """
    folders = sorted(
        os.path.basename(os.path.dirname(path))
        for path in glob.glob(os.path.join(results_dir, "*", "psbd"))
    )

    rows = []
    for folder in folders:
        if "sam_rho" in folder:
            continue
        meta_path = os.path.join(checkpoints_dir, folder, "metrics.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        result = {
            name: variants_evaluate(psbd_dir, cfg)
            for name, cfg in PLACEMENT_VARIANTS.items()
        }
        if not any(result.values()):
            continue
        rows.append((folder, meta, result))

    return rows


def variants_variant_cells(result: dict | None, with_direction: bool) -> str:
    """A variant's 4 numbers, or placeholders when it could not run."""
    if result and result.get("auroc") is not None:
        cells = (
            f" {result['rate']:>5g} {result['auroc']:>7.3f} "
            f"{result['tpr']:>6.3f} {result['fpr']:>6.3f}"
        )
        if with_direction:
            cells += f" {result['direction'][:9]:>9}"
        return cells

    cells = f" {'--':>5} {'--':>7} {'--':>6} {'--':>6}"
    if with_direction:
        cells += f" {'--':>9}"
    return cells


def variants_show(title: str, subset: list[tuple]) -> None:
    """A block of the comparison, plus the mean delta over its backdoored rows."""
    if not subset:
        return

    print(f"\n## {title}\n")
    print(
        f"{'checkpoint':32} {'ASR':>5} | {'p':>5} {'AUROC':>7} {'TPR':>6} {'FPR':>6} "
        f"| {'p':>5} {'AUROC':>7} {'TPR':>6} {'FPR':>6} {'dir':>9} {'dAUROC':>7}"
    )
    print("-" * VARIANTS_TABLE_WIDTH)

    deltas = []
    for folder, meta, result in subset:
        paper, adapted = result.get("psbd_paper"), result.get("psbd_vit")
        asr = meta.get("asr")
        asr_cell = "--" if asr is None else f"{asr:.2f}"

        line = f"{folder:32} {asr_cell:>5} |"
        line += variants_variant_cells(paper, with_direction=False) + " |"
        line += variants_variant_cells(adapted, with_direction=True)
        if (
            paper
            and adapted
            and paper.get("auroc") is not None
            and adapted.get("auroc") is not None
        ):
            delta = adapted["auroc"] - paper["auroc"]
            deltas.append((folder, delta))
            line += f" {delta:>+7.3f}"
        print(line)

    backdoored = [delta for folder, delta in deltas if "benign" not in folder]
    if backdoored:
        wins = sum(1 for delta in backdoored if delta > 0)
        print(
            f"\n  mean delta over {len(backdoored)} backdoored: "
            f"{sum(backdoored) / len(backdoored):+.3f}"
            f"   wins {wins}/{len(backdoored)}"
        )


def add_variants_parser(subparsers) -> None:
    sub = subparsers.add_parser(
        "variants",
        help="PSBD as published against the recommended ViT configuration",
        description=VARIANTS_HELP,
    )
    sub.add_argument("--results-dir", default="results")
    sub.add_argument("--checkpoints-dir", default="checkpoints")
    sub.add_argument(
        "--placement",
        default=None,
        help="override psbd_vit's placement, the default is the recommended one",
    )
    sub.add_argument(
        "--held-out",
        nargs="*",
        default=[],
        help="checkpoints not used to choose any hyperparameter, reported separately",
    )
    sub.set_defaults(func=run_variants)


def run_variants(args: argparse.Namespace) -> int:
    held_out = set(args.held_out)
    if args.placement:
        PLACEMENT_VARIANTS["psbd_vit"] = {
            **PLACEMENT_VARIANTS["psbd_vit"],
            "placement": args.placement,
        }
        print(f"psbd_vit placement overridden to {args.placement}\n")

    rows = variants_collect_rows(args.results_dir, args.checkpoints_dir)

    print("PSBD as published, against the recommended ViT configuration.")
    print("Both choose their rate on clean validation data only. No oracle.")
    variants_show(
        "Panel (the recommended placement was selected here, so these numbers are "
        "optimistic)",
        [row for row in rows if row[0] not in held_out and "benign" not in row[0]],
    )
    variants_show(
        "HELD OUT (nothing was fitted on these)",
        [row for row in rows if row[0] in held_out],
    )
    variants_show(
        "Negative control (both must sit near 0.5)",
        [row for row in rows if "benign" in row[0]],
    )
    return 0


# operating-points
# Ported from cli/operating_points.py. Detection at low false-positive rates, the
# operating points a defender lives at.

LOWFPR_HELP = """Detection at low false-positive rates, the operating points a defender lives at.

Everything reported at the headline used the PSBD paper's 25th-percentile
threshold, which throws away a quarter of clean data. No deployment tolerates
that. This reports the same detectors at 1% and 5% false positives.

2 thresholds per operating point, and the gap between them is the interesting part:

  deployable   the threshold is the q-quantile of CLEAN VALIDATION score, exactly as
               the paper prescribes, with q set to the target FPR. This is what a
               defender can actually build, since it needs no poisoned data. The FPR
               it achieves on the analysis pool is reported next to it, because
               nothing guarantees the validation quantile transfers.

  oracle_fpr   the threshold placed directly on the clean analysis pool to hit the
               target FPR exactly. Not deployable, since it needs the very clean/poison
               split the defence is trying to find. Reported as the ceiling, so the
               cost of thresholding on validation instead is visible.

TPR is also reported at ASR-conditioned form where available: a triggered image the
trigger never flipped is behaviourally clean, so counting it as a missed detection
charges the detector for the attack's failure.
"""

LOWFPR_DEFAULT_FPRS = (0.01, 0.05, 0.10)


def lowfpr_scores_at(
    psbd_dir: str, placement: str, rate: float
) -> tuple[dict, float | None]:
    """Fractional PSU per split, plus the clean-validation shift ratio."""
    scores, sigma = {}, None
    for split in SPLITS:
        probs, labels, targets = load_baseline(baseline_path(psbd_dir, split))
        per_pass, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        scores[split] = psu_ratio_from_cache(probs, labels, per_pass)
        if split == "validation":
            sigma = shift_ratio(labels, argmax)
        if split == "backdoor":
            scores["captured"] = attack_success_mask(labels, targets)

    return scores, sigma


def lowfpr_tpr_at(
    clean: torch.Tensor, backdoor: torch.Tensor, threshold: float, inverted: bool
) -> tuple[float, float]:
    """(TPR, FPR) at a threshold, flagging the tail the detector actually separates."""
    if inverted:
        return (
            float((backdoor > threshold).float().mean()),
            float((clean > threshold).float().mean()),
        )
    return (
        float((backdoor < threshold).float().mean()),
        float((clean < threshold).float().mean()),
    )


def lowfpr_operating_points(
    scores: dict, manifest: dict, target_fprs: list[float]
) -> tuple[dict, bool]:
    """TPR at each target FPR, thresholded 2 ways."""
    validation = scores["validation"].numpy()
    clean = pair_clean_to_backdoor(scores["clean"], manifest)
    backdoor = scores["backdoor"]
    captured = scores.get("captured")

    # Decide the tail once, from the pooled ranking, so both thresholds agree.
    inverted = float(backdoor.mean()) > float(clean.mean())

    rows = {}
    for target in target_fprs:
        # Deployable: quantile of clean validation. Inverted rule flags the upper
        # tail, so it needs the complementary quantile to cost the same clean budget.
        quantile = 1.0 - target if inverted else target
        deployable = float(np.quantile(validation, quantile))
        tpr_deployable, fpr_deployable = lowfpr_tpr_at(
            clean, backdoor, deployable, inverted
        )

        # Oracle: threshold placed on the clean analysis pool itself.
        oracle = float(np.quantile(clean.numpy(), quantile))
        tpr_oracle, fpr_oracle = lowfpr_tpr_at(clean, backdoor, oracle, inverted)

        row = {
            "target_fpr": target,
            "tpr_deployable": tpr_deployable,
            "fpr_deployable": fpr_deployable,
            "tpr_oracle_fpr": tpr_oracle,
            "fpr_oracle": fpr_oracle,
        }
        if captured is not None and bool(captured.any()):
            row["tpr_captured_only"] = lowfpr_tpr_at(
                clean[captured], backdoor[captured], deployable, inverted
            )[0]
        rows[f"fpr{target:.2f}"] = row

    return rows, inverted


def lowfpr_best_placement(
    psbd_dir: str,
    placements: list[str],
    target_fprs: list[float],
    shift_target: float,
) -> tuple | None:
    """The placement and rate with the highest deployable TPR at the tightest FPR."""
    tightest = min(target_fprs)
    best = None
    for placement in placements:
        for rate in complete_rates(psbd_dir, placement):
            try:
                scores, sigma = lowfpr_scores_at(psbd_dir, placement, rate)
            except Exception:
                continue
            if sigma is None or sigma < shift_target:
                continue
            manifest = read_split_manifest(psbd_dir)
            rows, inverted = lowfpr_operating_points(scores, manifest, target_fprs)
            value = rows[f"fpr{tightest:.2f}"]["tpr_deployable"]
            if best is None or value > best[0]:
                best = (value, placement, rate, rows, inverted)
            break  # first rate meeting the shift target, per the adaptive rule

    return best


def lowfpr_discover_folders(results_dir: str) -> list[str]:
    """Every results/<folder> carrying a stage-1 cache, in sorted order."""
    folders = sorted(
        os.path.basename(os.path.dirname(path))
        for path in glob.glob(os.path.join(results_dir, "*", "psbd"))
    )
    return folders


def lowfpr_build_header(target_fprs: list[float]) -> str:
    """The fixed-width header, a column pair per target FPR.

    Architecture and dataset are in the row because the sweep now spans 2 of the
    first and 4 of the second. Without them the same attack appears several times
    with different numbers and reads as a bug or, worse, gets averaged.
    """
    header = (
        f"{'arch':5} {'dataset':14} {'attack':16} {'pr':>5} {'ASR':>5} "
        f"{'best placement':>26} {'p':>4}"
    )
    for target in target_fprs:
        header += f" | {f'TPR@{target:.0%}':>9} {'(FPR)':>7}"
    return header


def lowfpr_render_row(
    meta: dict, placement: str, rate: float, rows: dict, target_fprs: list[float]
) -> str:
    """A checkpoint's line: its provenance, its best placement and its TPRs."""
    asr = meta.get("asr")
    line = (
        f"{meta.get('architecture') or '?':5} "
        f"{meta.get('dataset') or '?':14} {meta.get('attack') or 'benign':16} "
        f"{meta.get('poison_rate') or 0:>5.3f} "
        f"{'--' if asr is None else f'{asr:.2f}':>5} {placement:>26} {rate:>4g}"
    )
    for target in target_fprs:
        row = rows[f"fpr{target:.2f}"]
        line += f" | {row['tpr_deployable']:>9.3f} {row['fpr_deployable']:>7.3f}"
    return line


def add_operating_points_parser(subparsers) -> None:
    sub = subparsers.add_parser(
        "operating-points",
        help="detection at low false-positive rates (1%%, 5%%)",
        description=LOWFPR_HELP,
    )
    sub.add_argument("--results-dir", default="results")
    sub.add_argument("--checkpoints-dir", default="checkpoints")
    sub.add_argument("--fpr", nargs="*", type=float, default=list(LOWFPR_DEFAULT_FPRS))
    sub.add_argument(
        "--placement",
        nargs="*",
        default=None,
        help="restrict to these placements. The default is every placement on disk",
    )
    sub.add_argument(
        "--poison-rate",
        type=float,
        default=None,
        help="report only this poison rate",
    )
    sub.add_argument("--shift-target", type=float, default=ADAPTIVE_SHIFT_TARGET)
    sub.set_defaults(func=run_operating_points)


def run_operating_points(args: argparse.Namespace) -> int:
    print("Detection at low false-positive rates. Score: fractional PSU.")
    print(
        f"Rate chosen by the adaptive rule (clean-validation shift ratio >= "
        f"{args.shift_target}), placement chosen by best deployable TPR at "
        f"{min(args.fpr):.0%} FPR.\n"
    )
    header = lowfpr_build_header(args.fpr)
    print(header)
    print("-" * len(header))

    for folder in lowfpr_discover_folders(args.results_dir):
        if "sam_rho" in folder:
            continue
        meta_path = os.path.join(args.checkpoints_dir, folder, "metrics.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        if args.poison_rate is not None and meta.get("poison_rate") != args.poison_rate:
            continue

        psbd_dir = os.path.join(args.results_dir, folder, "psbd")
        placements = args.placement or sorted(
            name
            for name in os.listdir(psbd_dir)
            if os.path.isdir(os.path.join(psbd_dir, name))
        )
        best = lowfpr_best_placement(psbd_dir, placements, args.fpr, args.shift_target)
        if best is None:
            continue

        _value, placement, rate, rows, inverted = best
        line = lowfpr_render_row(meta, placement, rate, rows, args.fpr)
        print(line + ("  [inverted]" if inverted else ""))
    return 0


# detectors
# Ported from cli/compare_detectors.py. 1 table per metric: PSBD pinned to the
# declared placement against every detector.

DETECTORS_HELP = """1 table per metric: PSBD pinned to the declared placement against every detector.

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
"""

DETECTORS_PSBD_COLUMNS = ("psbd_adaptive", "psbd_matched06", "psbd_published")
DETECTORS_DEFAULT_FPRS = (0.01, 0.05, 0.10, 0.25)
# Below this many triggered images a per-cell TPR moves in steps a reader must
# see, so the row is marked.
DETECTORS_SMALL_POSITIVE_SET = 500
DETECTORS_DEFAULT_BOOTSTRAP = 5000
DETECTORS_RESULTS_BLOCK_BEGIN = "<!-- results:begin -->"
DETECTORS_RESULTS_BLOCK_END = "<!-- results:end -->"


def detectors_panel_cells(
    coverage: dict, declaration: dict, include_sam: bool
) -> list[dict]:
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


def detectors_psbd_rate(placement_block: dict, rule: str) -> float | None:
    """The rate a PSBD rule chose for a placement, or None when it chose nothing."""
    if rule == "adaptive":
        return placement_block.get("adaptive_rate")
    matched = placement_block.get("matched_shift", {}).get(
        shift_key(PLACEMENT_MATCH_TARGET)
    )
    rate = matched.get("rate") if matched else None
    return rate


def detectors_psbd_values(
    report: dict | None, placement: str, rule: str
) -> dict | None:
    """Every quantile's fractional-PSU report at the rate the rule picked, or None."""
    if report is None:
        return None
    block = report.get("placements", {}).get(placement)
    if block is None:
        return None
    rate = detectors_psbd_rate(block, rule)
    if rate is None:
        return None
    for row in block.get("rates", []):
        if row.get("rate") == rate:
            values = dict(row.get("detection_psu_ratio", {}))
            values["_rate"] = rate
            values["_n_backdoor"] = row.get("n_samples", {}).get("backdoor")
            return values
    return None


def detectors_detector_values(
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


def detectors_collect_columns(
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
            block = detectors_psbd_values(report, placement, rule)
            if block is None and report is not None:
                notes["psbd_none"].append((folder, column))
            if block is not None:
                values[(folder, column)] = block
        for name in detector_columns:
            block = detectors_detector_values(args.results_dir, folder, name, notes)
            if block is not None:
                values[(folder, name)] = block
        if legacy_report_present(args.results_dir, folder):
            notes["legacy"] += 1

    columns = list(DETECTORS_PSBD_COLUMNS) + detector_columns
    return columns, values


def detectors_metric(block: dict | None, key: str, quantile_key: str) -> float | None:
    """1 number out of a quantile block, or None when the column has no value."""
    if block is None or quantile_key not in block:
        return None
    value = block[quantile_key].get(key)
    return value


def detectors_cell_text(value: float | None, places: int = 3) -> str:
    """A table cell, with a placeholder for a column that produced no number."""
    text = "--" if value is None or value != value else f"{value:.{places}f}"
    return text


def detectors_bootstrap_interval(
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


def detectors_subsets(cells: list[dict]) -> list[tuple[str, list[dict]]]:
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


def detectors_active_columns(
    cells: list[dict], columns: list[str], values: dict
) -> list[str]:
    """The columns with a value on at least 1 of these cells."""
    active = [
        column
        for column in columns
        if any((cell["folder"], column) in values for cell in cells)
    ]
    return active


def detectors_common_coverage(
    cells: list[dict], columns: list[str], values: dict, key: str, quantile_key: str
) -> list[dict]:
    """The cells where every active column has this metric, the only set a mean may span."""
    active = detectors_active_columns(cells, columns, values)
    covered = [
        cell
        for cell in cells
        if all(
            detectors_metric(values.get((cell["folder"], column)), key, quantile_key)
            is not None
            for column in active
        )
    ]
    return covered


def detectors_metric_table(
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
        mark = (
            "*"
            if n_backdoor is not None and n_backdoor < DETECTORS_SMALL_POSITIVE_SET
            else ""
        )
        row = [
            cell["dataset"],
            cell["attack"],
            f"{cell['poison_rate']:g}",
            detectors_cell_text(cell["asr"], 2),
            f"{n_backdoor}{mark}" if n_backdoor is not None else "--",
        ]
        row += [
            detectors_cell_text(detectors_metric(block, key, quantile_key))
            for block in blocks
        ]
        lines.append("| " + " | ".join(row) + " |")

    active = detectors_active_columns(attacks, columns, values)
    lines.append("")
    lines.append(
        f"Aggregates over the common-coverage cells of the {len(active)} columns with "
        f"data, {key} at {quantile_key}. Macro is the mean over attack means."
    )
    lines.append("")
    lines.append("| subset | n | " + " | ".join(columns) + " |")
    lines.append("|---|---:|" + "---:|" * len(columns))
    for name, subset in detectors_subsets(cells):
        covered = detectors_common_coverage(subset, columns, values, key, quantile_key)
        if not covered:
            continue
        means = [
            statistics.mean(
                detectors_metric(values[(c["folder"], column)], key, quantile_key)
                for c in covered
            )
            if column in active
            else None
            for column in columns
        ]
        lines.append(
            f"| {name} | {len(covered)} | "
            + " | ".join(detectors_cell_text(m) for m in means)
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
                        detectors_metric(
                            values[(c["folder"], column)], key, quantile_key
                        )
                        for c in covered
                        if c["attack"] == attack
                    )
                    for attack in sorted({c["attack"] for c in covered})
                ]
                macro.append(statistics.mean(per_attack))
            lines.append(
                f"| macro over attacks | {len(covered)} | "
                + " | ".join(detectors_cell_text(m) for m in macro)
                + " |"
            )

    lines += detectors_paired_deltas(cells, columns, values, key, quantile_key, args)
    lines.append("")
    return lines


def detectors_paired_deltas(
    cells, columns, values, key, quantile_key, args
) -> list[str]:
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
            ours = detectors_metric(
                values.get((cell["folder"], column)), key, quantile_key
            )
            theirs = detectors_metric(
                values.get((cell["folder"], reference)), key, quantile_key
            )
            if ours is not None and theirs is not None:
                deltas.append(ours - theirs)
        if not deltas:
            continue
        low, high = detectors_bootstrap_interval(deltas, args.bootstrap, args.seed)
        lines.append(
            f"| {column} | {len(deltas)} | {statistics.mean(deltas):+.3f} | "
            f"[{low:+.3f}, {high:+.3f}] |"
        )
    return lines


def detectors_benign_table(
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
                    detectors_cell_text(detectors_metric(block, "fpr", q)),
                    detectors_cell_text(detectors_metric(block, "tpr", q)),
                ]
            row.append(detectors_cell_text(detectors_metric(block, "auroc", headline)))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def detectors_cost_table(columns: list[str], declaration: dict) -> list[str]:
    """Forward passes per input and clean data needed, so a column's price is visible."""
    passes = declaration["panel"]["forward_passes"]
    lines = [
        "### Cost and data",
        "",
        "| column | forwards per input | clean data |",
        "|---|---:|---|",
    ]
    for column in columns:
        if column in DETECTORS_PSBD_COLUMNS:
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


def detectors_footer(
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


def detectors_render(cells, columns, values, notes, args, declaration) -> list[str]:
    """Every table, as markdown lines."""
    quantile_keys = [f"q{value:.2f}" for value in args.fpr]
    headline = f"q{HEADLINE_QUANTILE:.2f}"
    lines = [
        "# Detector comparison",
        "",
        "Generated by `python -m cli.compare detectors` at "
        f"{utc_timestamp()} from `{args.results_dir}`, commit "
        f"{current_git_commit()}. Do not edit by hand.",
        "",
        f"PSBD columns: recommended placement `{RECOMMENDED_PLACEMENT}` at the adaptive "
        f"rule (shift ratio {ADAPTIVE_SHIFT_TARGET}) and at the matched rule (nearest "
        f"{PLACEMENT_MATCH_TARGET}), and the published placement `{PUBLISHED_PLACEMENT}` "
        "at the adaptive rule. Fractional PSU throughout. Every detector shares the "
        "split, the clean-validation quantile threshold and the pairing. A `*` on n_bd "
        f"marks fewer than {DETECTORS_SMALL_POSITIVE_SET} triggered images.",
        "",
    ]
    lines += detectors_metric_table(
        "AUROC", cells, columns, values, "auroc", headline, args
    )
    for q in quantile_keys:
        lines += detectors_metric_table(
            f"TPR at FPR budget {q}", cells, columns, values, "tpr", q, args
        )
        lines += detectors_metric_table(
            f"Achieved FPR at budget {q}", cells, columns, values, "fpr", q, args
        )
    lines += detectors_benign_table(cells, columns, values, quantile_keys)
    lines += detectors_cost_table(columns, declaration)
    lines += detectors_footer(cells, columns, values, notes)
    return lines


def detectors_write_csv(path: str, cells, columns, values, quantile_keys) -> None:
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


def detectors_rewrite_results_block(path: str, body: list[str]) -> None:
    """Replace the results block of a detector doc, refusing a doc without the markers."""
    with open(path) as handle:
        text = handle.read()
    begin = text.find(DETECTORS_RESULTS_BLOCK_BEGIN)
    end = text.find(DETECTORS_RESULTS_BLOCK_END)
    if begin < 0 or end < 0 or end < begin:
        raise ValueError(
            f"{path} has no {DETECTORS_RESULTS_BLOCK_BEGIN} ... "
            f"{DETECTORS_RESULTS_BLOCK_END} block"
        )
    replacement = DETECTORS_RESULTS_BLOCK_BEGIN + "\n" + "\n".join(body) + "\n"
    rewritten = text[:begin] + replacement + text[end:]
    with open(path, "w") as handle:
        handle.write(rewritten)


# Detectors documented in 1 file share its results block.
DETECTORS_DOC_OF = {
    "scale_up_data_limited": "scale_up",
    "ibd_psc_calibrated": "ibd_psc",
}


def detectors_per_detector_blocks(cells, columns, values, args) -> dict[str, list[str]]:
    """The results block of each detector doc: its columns beside psbd_adaptive."""
    headline = f"q{HEADLINE_QUANTILE:.2f}"
    blocks = {}
    for name in columns:
        if name in DETECTORS_PSBD_COLUMNS or name in DETECTORS_DOC_OF:
            continue
        shared = [
            other
            for other, doc in DETECTORS_DOC_OF.items()
            if doc == name and other in columns
        ]
        pair = ["psbd_adaptive", name] + shared
        body = [
            "Generated by `python -m cli.compare detectors --per-detector-dir` at "
            f"{utc_timestamp()}.",
            "",
        ]
        body += detectors_metric_table(
            "AUROC", cells, pair, values, "auroc", headline, args
        )
        for value in args.fpr:
            body += detectors_metric_table(
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


def add_detectors_parser(subparsers) -> None:
    sub = subparsers.add_parser(
        "detectors",
        help="1 table per metric, PSBD against every competitor detector",
        description=DETECTORS_HELP,
    )
    sub.add_argument("--results-dir", default="results")
    sub.add_argument(
        "--coverage", default=None, help="default <results-dir>/coverage/coverage.json"
    )
    sub.add_argument("--declaration", default="configs/psbd_basis.json")
    sub.add_argument(
        "--fpr", nargs="*", type=float, default=list(DETECTORS_DEFAULT_FPRS)
    )
    sub.add_argument("--include-sam", action="store_true")
    sub.add_argument("--bootstrap", type=int, default=DETECTORS_DEFAULT_BOOTSTRAP)
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--markdown", default=None, help="write every table to this file")
    sub.add_argument(
        "--per-detector-dir",
        default=None,
        help="rewrite the results block of <dir>/<detector>.md for every detector",
    )
    sub.add_argument(
        "--csv", default=None, help="1 row per (cell, column, quantile), gzipped"
    )
    sub.set_defaults(func=run_detectors)


def run_detectors(args: argparse.Namespace) -> int:
    coverage_path = args.coverage or os.path.join(
        args.results_dir, "coverage", "coverage.json"
    )
    coverage = read_json(coverage_path)
    declaration = read_json(args.declaration)
    if coverage is None or declaration is None:
        raise SystemExit(f"need {coverage_path} and {args.declaration}")

    cells = detectors_panel_cells(coverage, declaration, args.include_sam)
    notes = {"failed": [], "smoke": [], "psbd_none": [], "legacy": 0}
    columns, values = detectors_collect_columns(cells, args, notes)
    lines = detectors_render(cells, columns, values, notes, args, declaration)
    print("\n".join(lines))

    quantile_keys = [f"q{value:.2f}" for value in args.fpr]
    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
        with open(args.markdown, "w") as handle:
            handle.write("\n".join(lines) + "\n")
    if args.csv:
        detectors_write_csv(args.csv, cells, columns, values, quantile_keys)
    if args.per_detector_dir:
        for name, body in detectors_per_detector_blocks(
            cells, columns, values, args
        ).items():
            detectors_rewrite_results_block(
                os.path.join(args.per_detector_dir, f"{name}.md"), body
            )
    return 0


# fused
# Ported from cli/fuse_detectors.py. Fuse PSBD with a recorded competitor
# detector by rank, where their failures are disjoint.

FUSED_HELP = """Fuse PSBD with a recorded competitor detector by rank, where their failures are disjoint.

The first comparison table showed PSBD and STRIP failing on opposite attacks:
STRIP is near perfect on a static patch trigger and fails outright on
adaptive_blend and badnet_a2a, while PSBD is the reverse. No checkpoint in that
grid defeated both, which is the textbook case for combining them. This command
reads both detectors from disk, the PSBD stage-1 cache under
results/<folder>/psbd/ and the competitor's record under
results/<folder>/detectors/ written by cli.baselines, and never runs a model,
so the fusion is a CPU read over exactly the rows both were scored on.

Fusion is by rank, not by score. The 2 scores are on incompatible scales, a
probability drop against an entropy in nats, so any weighted sum would be
dominated by whichever has the larger spread. Converting each to its rank within
a shared reference makes them commensurable without fitting anything. The
reference is the clean validation split, the only distribution the defender
holds and the one the threshold is drawn from. Ranking each split against itself
would destroy the method, since within-split ranks span [0, 1] for every split by
construction and a threshold at the 1st percentile of validation rank would flag
exactly the bottom 1% of the backdoor split whatever its scores are.

2 rules, both needing no poisoned data:

  mean_rank   average of the 2 normalized ranks. Balanced, the natural choice
              when neither detector is known to be reliable in advance.
  min_rank    the more suspicious of the 2 verdicts. The right rule if the
              failures really are disjoint, since a sample only escapes when both
              detectors consider it clean.

The threshold is the quantile of clean-validation fused rank, so the defender
never touches poisoned data and the false-positive budget is set exactly as
before.
"""

FUSED_COLUMNS = ("psbd", "detector", "mean", "min")
FUSED_DEFAULT_FPRS = (0.01, 0.05, 0.25)


def fused_psbd_scores(
    psbd_dir: str, placement: str, shift_target: float
) -> tuple[dict[str, torch.Tensor], float] | None:
    """Fractional PSU per split at the adaptive rate, or None when the cache lacks it.

    The adaptive rule is the smallest cached rate whose clean-validation shift
    ratio reaches the target, the same rule the analyze command applies.
    """
    if not os.path.isdir(os.path.join(psbd_dir, placement)):
        return None

    for rate in complete_rates(psbd_dir, placement):
        _, labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "validation")
        )
        sigma = shift_ratio(labels, argmax)
        if sigma is None or sigma < shift_target:
            continue

        scores = {}
        for split in SPLITS:
            probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
            per_pass, _ = load_dropout_pass_probs(
                dropout_pass_path(psbd_dir, placement, rate, split)
            )
            scores[split] = psu_ratio_from_cache(probs, labels, per_pass)  # (n_split,)
        return scores, rate

    return None


def fused_detector_scores(
    results_dir: str, folder: str, name: str
) -> dict[str, torch.Tensor] | None:
    """The recorded per-split scores of 1 detector, or None without a scored record."""
    report = load_report(report_path(results_dir, folder, name))
    if report is None or report.get("status") != STATUS_SCORED:
        return None
    scores = {
        split: load_scores(scores_path(results_dir, folder, name, split))
        for split in SPLITS
    }
    return scores


def fused_fuse(psu: dict, other: dict) -> dict[str, dict[str, torch.Tensor]]:
    """The 4 comparable columns per split: both components and both fusion rules.

    Both detectors become percentiles of the same reference, the clean validation
    split, so a fused score means the same thing in every split and the validation
    threshold transfers.
    """
    fused = {}
    for split in SPLITS:
        psu_rank = to_rank(psu[split], psu["validation"])  # (n_split,)
        other_rank = to_rank(other[split], other["validation"])  # (n_split,)
        fused[split] = {
            "psbd": psu[split],
            "detector": other[split],
            "mean": (psu_rank + other_rank) / 2,
            "min": torch.minimum(psu_rank, other_rank),
        }
    return fused


def fused_tpr_at_fpr(
    validation: torch.Tensor,
    clean: torch.Tensor,
    backdoor: torch.Tensor,
    target: float,
) -> tuple[float, float]:
    """TPR at a threshold set from clean validation, plus the FPR it achieves."""
    threshold = float(np.quantile(validation.numpy(), target))
    tpr = float((backdoor < threshold).float().mean())
    fpr = float((clean < threshold).float().mean())
    return tpr, fpr


def fused_discover_folders(args: argparse.Namespace) -> list[str]:
    """Folders of the architecture with a PSBD cache and a scored record of the detector."""
    folders = sorted(
        os.path.basename(os.path.dirname(path))
        for path in glob.glob(os.path.join(args.results_dir, "*", "psbd"))
    )
    selected = []
    for folder in folders:
        if not folder.startswith(f"{args.architecture}_") or "sam_rho" in folder:
            continue
        if args.dataset and folder.split("_")[1] not in args.dataset:
            continue
        if fused_detector_scores(args.results_dir, folder, args.detector) is None:
            continue
        selected.append(folder)
    return selected


def fused_folder_row(folder: str, args: argparse.Namespace) -> dict | None:
    """1 checkpoint's TPR per column and FPR budget, or None when a source is missing."""
    psbd_dir = os.path.join(args.results_dir, folder, "psbd")
    scored = fused_psbd_scores(psbd_dir, args.placement, args.shift_target)
    other = fused_detector_scores(args.results_dir, folder, args.detector)
    if scored is None or other is None:
        return None
    psu, rate = scored
    if any(psu[split].numel() != other[split].numel() for split in SPLITS):
        return None

    manifest = read_split_manifest(psbd_dir)
    metadata = read_checkpoint_metadata(
        os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    )
    fused = fused_fuse(psu, other)
    tprs = {}
    for target in args.fpr:
        for name in FUSED_COLUMNS:
            # The clean side is paired down to the backdoor split's images, so all
            # 4 columns are compared on a single population.
            clean = pair_clean_to_backdoor(fused["clean"][name], manifest)
            tpr, _ = fused_tpr_at_fpr(
                fused["validation"][name], clean, fused["backdoor"][name], target
            )
            tprs[(target, name)] = tpr
    row = {
        "folder": folder,
        "attack": metadata["attack"],
        "poison_rate": metadata.get("poison_rate") or 0.0,
        "rate": rate,
        "tprs": tprs,
    }
    return row


def fused_column_label(name: str, detector: str) -> str:
    label = detector.upper() if name == "detector" else name.upper()
    return label


def fused_render_table(rows: list[dict], args: argparse.Namespace) -> list[str]:
    """A markdown table, 1 row per checkpoint and a mean over the backdoored ones."""
    header = ["folder", "attack", "rate"]
    for target in args.fpr:
        header += [
            f"{fused_column_label(name, args.detector)}@{target:.0%}"
            for name in FUSED_COLUMNS
        ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["folder"], row["attack"], f"{row['poison_rate']:.3f}"]
        for target in args.fpr:
            cells += [f"{row['tprs'][(target, name)]:.3f}" for name in FUSED_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    backdoored = [row for row in rows if row["attack"] != "benign"]
    if backdoored:
        cells = [f"mean over {len(backdoored)} backdoored", "", ""]
        for target in args.fpr:
            cells += [
                f"{statistics.mean(row['tprs'][(target, name)] for row in backdoored):.3f}"
                for name in FUSED_COLUMNS
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def add_fused_parser(subparsers) -> None:
    sub = subparsers.add_parser(
        "fused",
        help="fuse PSBD and a competitor detector by rank",
        description=FUSED_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub.add_argument("--results-dir", default="results")
    sub.add_argument("--checkpoints-dir", default="checkpoints")
    sub.add_argument("--detector", choices=DETECTOR_NAMES, default="strip")
    sub.add_argument("--placement", default=RECOMMENDED_PLACEMENT)
    sub.add_argument("--shift-target", type=float, default=ADAPTIVE_SHIFT_TARGET)
    sub.add_argument("--architecture", default="vit")
    sub.add_argument("--dataset", nargs="*", default=None)
    sub.add_argument("--fpr", type=float, nargs="+", default=list(FUSED_DEFAULT_FPRS))
    sub.add_argument("--markdown", default=None, help="also write the table here")
    sub.set_defaults(func=run_fused)


def run_fused(args: argparse.Namespace) -> int:
    folders = fused_discover_folders(args)
    rows = [
        row for row in (fused_folder_row(folder, args) for folder in folders) if row
    ]
    if not rows:
        print(
            f"no folder carries both a PSBD cache at {args.placement} and a scored "
            f"{args.detector} record under {args.results_dir}"
        )
        return 1

    lines = fused_render_table(rows, args)
    print(
        f"PSBD ({args.placement}, shift target {args.shift_target}) fused with "
        f"{args.detector} by rank, threshold from clean validation only, "
        f"{len(rows)} checkpoints\n"
    )
    print("\n".join(lines))
    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
        with open(args.markdown, "w") as handle:
            handle.write("\n".join(lines) + "\n")
    return 0


# Top-level parser and dispatch


def build_parser() -> argparse.ArgumentParser:
    """The full `cli.compare` parser: 4 subcommands, `placements` holding 4 more."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    placements_parser = subparsers.add_parser(
        "placements",
        help="positions and operators: report, summary, tables, variants",
    )
    placements_actions = placements_parser.add_subparsers(dest="action", required=True)
    add_report_parser(placements_actions)
    add_summary_parser(placements_actions)
    add_tables_parser(placements_actions)
    add_variants_parser(placements_actions)

    add_operating_points_parser(subparsers)
    add_detectors_parser(subparsers)
    add_fused_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    validate = getattr(args, "_validate", None)
    if validate is not None:
        validate(parser, args)

    exit_code = args.func(args)
    return 0 if exit_code is None else exit_code


if __name__ == "__main__":
    sys.exit(main())
