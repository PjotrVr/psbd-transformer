"""Collapse every psbd_metrics.json into a single compact, versionable table.

The full stage-2 record is about 2 MB per checkpoint and 1.2 GB across the tree,
which is regenerable from the cached tensors and far too large to keep in git.
What every table in the paper actually reads is a handful of numbers per
(checkpoint, placement): the operating point, the rate that produced it and how
well it separated. That is what this writes.

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

import argparse
import csv
import glob
import json
import os
import re

from models.positions import DROPOUT_CONFIGS, POSITION_REGISTRY

SELECTION_RULES = ("adaptive", "oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--output", default="results/detection_summary.csv")
    parser.add_argument(
        "--operating-points",
        action="store_true",
        help=(
            "emit one row per placement, rate, PSU variant and quantile instead of "
            "the compact table, so TPR at 1 and 5 percent FPR is available rather "
            "than only the 25 percent headline"
        ),
    )
    parser.add_argument(
        "--include-sam",
        action="store_true",
        help=(
            "keep SAM-trained checkpoints. Off by default: SAM is a training-time "
            "change under a separate question, and carrying it dilutes every panel "
            "and reintroduces unequal coverage between groups"
        ),
    )
    return parser.parse_args()


def read_json(path: str) -> dict | None:
    """A JSON file, or None when it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


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

OPERATING_POINT_FIELDS = (
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


def operating_point_rows(folder, report, metadata, results_dir="results"):
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
        "poison_rate": metadata.get("poison_rate"),
        # The requested rate is a request. A clean-label attack is eligible only on
        # the target class, so it saturates and 3 folder names can describe 1 run.
        # The compact summary already carries both. Without them here every
        # consumer of operating_points.csv reads a rate that was never applied.
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


def rows_for_checkpoint(
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


def main() -> None:
    args = parse_args()
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
                operating_point_rows(folder, report, metadata, args.results_dir)
            )
        else:
            all_rows.extend(
                rows_for_checkpoint(folder, report, metadata, args.results_dir)
            )

    with open(args.output, "w", newline="") as handle:
        fields = OPERATING_POINT_FIELDS if args.operating_points else FIELDS
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


if __name__ == "__main__":
    main()
