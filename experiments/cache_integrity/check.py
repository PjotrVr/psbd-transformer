"""Read-only integrity audit of the stage-1 PSBD cache under results/<folder>/psbd/.

Every headline number in the paper is computed by psbd_analyze.py from tensors it
never recomputes, so a silently corrupt cache is indistinguishable from a real
result. This script re-derives the invariants that stage 1 establishes and stage 2
assumes, over the whole tree, and writes results/_experiments/cache_integrity/cache_integrity.json.

The audit is grouped into 4 scopes, and the check identifiers below match the
numbering used in the accompanying README:

    manifest   split_manifest.json is a well-formed index of a disjoint 3-way split
    baseline   baseline_<split>.pt is finite, normalized, and its own argmax
    rate       rate_<tag>_<split>.pt agrees in shape and range with that baseline
    position   a position config stores a single k and a known rate coverage

The single highest-value check is 18. A stochastic operator whose k passes are
bit-identical never fired, which makes PSU identically 0 and reads downstream as
"this placement had no effect" rather than as a broken run. Identical passes are
correct and expected only for the operators in
defences.perturbations.DETERMINISTIC_PERTURBATIONS.

Nothing here opens a file for writing except the final report.

Example
    python experiments/cache_integrity/check.py --sample 20
    python experiments/cache_integrity/check.py --full --workers 128
"""

import os

# Each worker only ever does elementwise arithmetic on small (k, N) tensors, so
# intra-op threading buys nothing and 128 workers each spawning a thread pool
# oversubscribe the login node badly. OpenMP reads this at load time, so it has to
# be set before torch is imported anywhere.
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import datetime
import json
import multiprocessing
import random
import re
import subprocess
import traceback

import torch
from lightning import seed_everything

from attacks import build_attack, default_config
from defences.operators import DETERMINISTIC_OPERATORS, OPERATORS
from defences.decision import complete_rates
from data.registry import DATASET_REGISTRY
from defences.cache import read_run_provenance
from experiments._paths import experiment_result_path

SPLITS: tuple[str, ...] = ("validation", "clean", "backdoor")

# Softmax rows are accumulated in float32, so the residual is nearer 1e-6 than the
# tolerance. 1e-3 is loose enough that no honest row trips it and tight enough to
# catch a truncated or half-written tensor.
PROBABILITY_SUM_TOLERANCE = 1e-3

# Operators that can be named in a cache folder. PERTURBATIONS is the live registry,
# scale_up is built separately because it needs the dataset normalization constants,
# and dropout is the unmarked default that carries no suffix at all.
KNOWN_PERTURBATIONS: tuple[str, ...] = tuple(
    sorted(set(OPERATORS) | {"scale_up"}, key=len, reverse=True)
)

# Suffixes cache_config_name appends AFTER the operator name, which have to come off
# before the folder name can be matched against an operator.
TRAILING_SUFFIX_PATTERN = re.compile(r"(_k\d+|_pmodel[0-9_]+)+$")

CHECK_IDS: tuple[str, ...] = (
    "00_checkpoint_audit_completes",
    "01_manifest_parses",
    "02_manifest_partitions_n_total",
    "03_manifest_n_heldout_matches",
    "04_manifest_backdoor_subset_of_clean",
    "05_manifest_label_mode_matches_attack",
    "06_baseline_loads",
    "07_baseline_row_counts_agree",
    "08_baseline_rows_match_manifest",
    "09_baseline_probs_finite_and_normalized",
    "10_baseline_labels_are_argmax",
    "11_baseline_num_classes_matches_dataset",
    "12_rate_loads",
    "13_rate_shapes_agree",
    "14_rate_rows_match_baseline",
    "15_rate_probs_in_unit_range",
    "16_rate_argmax_in_class_range",
    "17_position_k_consistent",
    "18_stochastic_passes_differ",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--full",
        action="store_true",
        help="audit every checkpoint with a psbd/ subtree",
    )
    scope.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="audit a random N checkpoints, seeded so the same N come back next time",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument(
        "--output",
        default=None,
        help="default: <results-dir>/_experiments/cache_integrity/cache_integrity.json",
    )
    parser.add_argument("--seed", type=int, default=0, help="seeds the --sample draw")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(128, os.cpu_count() or 1),
        help="worker processes",
    )
    parser.add_argument(
        "--max-shown",
        type=int,
        default=50,
        help="failing paths printed per check, with the full list always in the JSON",
    )
    parsed = parser.parse_args()
    return parsed


def current_git_commit() -> str | None:
    """The commit the audit ran at, so a report can be traced back to a tree state."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None
    return commit


def discover_checkpoint_folders(results_dir: str) -> list[str]:
    """Every results/<folder> carrying a stage-1 cache, in sorted order."""
    if not os.path.isdir(results_dir):
        return []
    folders = sorted(
        name
        for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name, "psbd"))
    )
    return folders


def select_folders(folders: list[str], sample_size: int | None, seed: int) -> list[str]:
    """The audit set: all folders, or a reproducible random subset of them.

    Seeded through lightning so a --sample run is rerunnable and a reported failure
    can be reproduced from the report's own seed and sample size.
    """
    if sample_size is None:
        return folders
    seed_everything(seed, workers=False, verbose=False)
    drawn = sorted(random.sample(folders, min(sample_size, len(folders))))
    return drawn


_LABEL_MODE_CACHE: dict[tuple[str, str, int], str] = {}


def expected_label_mode(attack_name: str, dataset: str, target_label: int) -> str:
    """The label mode the attack registry builds for this (attack, dataset, target).

    Memoized because the registry is consulted once per checkpoint but only ever
    returns 1 of a handful of distinct answers, and some builders synthesize a
    trigger image on construction.
    """
    key = (attack_name, dataset, target_label)
    if key not in _LABEL_MODE_CACHE:
        attack = build_attack(
            attack_name,
            default_config(attack_name),
            DATASET_REGISTRY[dataset].image_size,
            target_label,
        )
        _LABEL_MODE_CACHE[key] = attack.label_mode
    return _LABEL_MODE_CACHE[key]


def read_dataset_name(
    checkpoints_dir: str, folder: str, manifest: dict
) -> tuple[str | None, str]:
    """(dataset name, where it came from). args.json is the training-time record.

    The manifest copy is only a fallback, because it was written by the sweep from
    the same args.json and so cannot independently confirm it.
    """
    args_path = os.path.join(checkpoints_dir, folder, "args.json")
    if os.path.exists(args_path):
        try:
            with open(args_path) as handle:
                metadata = json.load(handle)
            if metadata.get("dataset"):
                return metadata["dataset"], "args.json"
        except (OSError, json.JSONDecodeError):
            pass
    if manifest.get("dataset"):
        return manifest["dataset"], "split_manifest.json"
    return None, "unresolved"


def resolve_operator(psbd_dir: str, placement: str) -> tuple[str, str]:
    """(operator name, how it was resolved) for one position-config folder.

    The run provenance sidecar records the operator verbatim and is authoritative.
    Caches written before that sidecar existed, and folders renamed by hand after
    the fact, have to be read off the folder name instead: cache_config_name appends
    the operator to the position, leaving the paper's dropout as the bare name.
    """
    recorded = read_run_provenance(psbd_dir, placement).get("operator")
    if recorded:
        return recorded, "run_json"

    stem = TRAILING_SUFFIX_PATTERN.sub("", placement)
    for name in KNOWN_PERTURBATIONS:
        if stem == name or stem.endswith(f"_{name}") or f"_{name}_" in stem:
            return name, "folder_name"
    return "dropout", "bare_name_default"


def audit_manifest(
    psbd_dir: str, manifest: dict, dataset: str | None
) -> tuple[list[dict], dict[str, int]]:
    """Checks 02 to 05 over one parsed manifest, plus the expected rows per split.

    Returns the failures and the row count each split's tensors must have, which is
    the manifest's whole job: it is the ground-truth row order for every tensor in
    the subtree below it.
    """
    failures: list[dict] = []
    manifest_file = os.path.join(psbd_dir, "split_manifest.json")

    heldout = manifest.get("heldout_indices", [])
    analysis_clean = manifest.get("analysis_clean_indices", [])
    analysis_backdoor = manifest.get("analysis_backdoor_indices", [])
    n_total = manifest.get("n_total")

    heldout_set = set(heldout)
    clean_set = set(analysis_clean)
    overlap = heldout_set & clean_set
    union_size = len(heldout_set | clean_set)
    if overlap or union_size != n_total:
        failures.append(
            {
                "check": "02_manifest_partitions_n_total",
                "path": manifest_file,
                "detail": (
                    f"heldout and analysis_clean overlap on {len(overlap)} indices and "
                    f"cover {union_size} of n_total={n_total} "
                    f"(len heldout {len(heldout)}, len clean {len(analysis_clean)})"
                ),
            }
        )

    if manifest.get("n_heldout") != len(heldout):
        failures.append(
            {
                "check": "03_manifest_n_heldout_matches",
                "path": manifest_file,
                "detail": f"n_heldout={manifest.get('n_heldout')} but {len(heldout)} indices",
            }
        )

    stray = set(analysis_backdoor) - clean_set
    if stray:
        failures.append(
            {
                "check": "04_manifest_backdoor_subset_of_clean",
                "path": manifest_file,
                "detail": (
                    f"{len(stray)} backdoor indices are outside the analysis pool, "
                    f"first few {sorted(stray)[:5]}"
                ),
            }
        )

    attack_name = manifest.get("probe_attack")
    recorded_mode = manifest.get("label_mode")
    if attack_name and dataset:
        try:
            derived = expected_label_mode(
                attack_name, dataset, manifest.get("probe_target_label") or 0
            )
            if derived != recorded_mode:
                failures.append(
                    {
                        "check": "05_manifest_label_mode_matches_attack",
                        "path": manifest_file,
                        "detail": f"records {recorded_mode!r} but {attack_name!r} builds {derived!r}",
                    }
                )
        except Exception as error:
            failures.append(
                {
                    "check": "05_manifest_label_mode_matches_attack",
                    "path": manifest_file,
                    "detail": f"could not build {attack_name!r} on {dataset!r}: {error!r}",
                }
            )

    expected_rows = {
        "validation": len(heldout),
        "clean": len(analysis_clean),
        "backdoor": len(analysis_backdoor),
    }
    return failures, expected_rows


def audit_one_baseline(
    path: str, split: str, expected_rows: int, num_classes: int | None
) -> tuple[list[dict], int | None]:
    """Checks 06 to 11 over one baseline_<split>.pt, returning its row count.

    The row count is what every rate tensor under this checkpoint is measured
    against, so a baseline that fails to load takes its whole split with it.
    """
    failures: list[dict] = []
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        probs = blob["probs"]  # (N, num_classes) float32
        labels = blob["labels"]  # (N,) int64
        loader_labels = blob["loader_labels"]  # (N,) int64
    except Exception as error:
        failures.append(
            {"check": "06_baseline_loads", "path": path, "detail": repr(error)}
        )
        return failures, None

    rows = probs.shape[0]
    if not (rows == labels.shape[0] == loader_labels.shape[0]):
        failures.append(
            {
                "check": "07_baseline_row_counts_agree",
                "path": path,
                "detail": (
                    f"probs {tuple(probs.shape)}, labels {tuple(labels.shape)}, "
                    f"loader_labels {tuple(loader_labels.shape)}"
                ),
            }
        )

    if rows != expected_rows:
        failures.append(
            {
                "check": "08_baseline_rows_match_manifest",
                "path": path,
                "detail": f"{rows} rows but the manifest's {split} split has {expected_rows}",
            }
        )

    if probs.numel():
        finite = bool(torch.isfinite(probs).all())
        worst_residual = float((probs.float().sum(dim=1) - 1.0).abs().max())
        if not finite or worst_residual > PROBABILITY_SUM_TOLERANCE:
            failures.append(
                {
                    "check": "09_baseline_probs_finite_and_normalized",
                    "path": path,
                    "detail": f"all_finite={finite}, worst row sum residual {worst_residual:.3e}",
                }
            )

        # build_baseline_cache stores labels as probs.argmax(dim=1) of this exact
        # tensor, so anything but bitwise equality means the two were written by
        # different forward passes and PSU is being read off a mismatched class.
        recomputed = probs.argmax(dim=1)  # (N,)
        if not torch.equal(labels, recomputed):
            if labels.shape == recomputed.shape:
                detail = (
                    f"{int((labels != recomputed).sum())} of {rows} labels differ from "
                    "probs.argmax(dim=1)"
                )
            else:
                detail = (
                    f"labels {tuple(labels.shape)} cannot be compared to "
                    f"probs.argmax(dim=1) {tuple(recomputed.shape)}"
                )
            failures.append(
                {
                    "check": "10_baseline_labels_are_argmax",
                    "path": path,
                    "detail": detail,
                }
            )

    if num_classes is not None and probs.ndim == 2 and probs.shape[1] != num_classes:
        failures.append(
            {
                "check": "11_baseline_num_classes_matches_dataset",
                "path": path,
                "detail": f"{probs.shape[1]} columns but the dataset has {num_classes} classes",
            }
        )

    return failures, rows


def audit_one_rate_file(
    path: str,
    split: str,
    baseline_rows: int | None,
    num_classes: int | None,
    deterministic: bool,
) -> tuple[list[dict], int | None]:
    """Checks 12 to 16 and 18 over one rate_<tag>_<split>.pt, returning its stored k."""
    failures: list[dict] = []
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        per_pass_probs = blob["per_pass_probs"]  # (k, N) float32
        per_pass_argmax = blob["per_pass_argmax"]  # (k, N) int16
    except Exception as error:
        failures.append({"check": "12_rate_loads", "path": path, "detail": repr(error)})
        return failures, None

    if per_pass_probs.shape != per_pass_argmax.shape:
        failures.append(
            {
                "check": "13_rate_shapes_agree",
                "path": path,
                "detail": (
                    f"per_pass_probs {tuple(per_pass_probs.shape)} but per_pass_argmax "
                    f"{tuple(per_pass_argmax.shape)}"
                ),
            }
        )

    passes = per_pass_probs.shape[0] if per_pass_probs.ndim == 2 else None
    rows = per_pass_probs.shape[1] if per_pass_probs.ndim == 2 else None
    if baseline_rows is not None and rows != baseline_rows:
        failures.append(
            {
                "check": "14_rate_rows_match_baseline",
                "path": path,
                "detail": f"{rows} rows but baseline_{split}.pt has {baseline_rows}",
            }
        )

    if per_pass_probs.numel():
        probs_float = per_pass_probs.float()
        finite = bool(torch.isfinite(probs_float).all())
        lowest = float(probs_float.min())
        highest = float(probs_float.max())
        if not finite or lowest < 0.0 or highest > 1.0:
            failures.append(
                {
                    "check": "15_rate_probs_in_unit_range",
                    "path": path,
                    "detail": f"all_finite={finite}, range [{lowest:.6g}, {highest:.6g}]",
                }
            )

        # Identical passes are the signature of a probe that never fired: PSU is an
        # expectation over k, so k identical rows collapse it to a constant and the
        # placement reads as inert rather than as unmeasured.
        if passes and passes > 1 and not deterministic:
            if bool((per_pass_probs == per_pass_probs[0]).all()):
                failures.append(
                    {
                        "check": "18_stochastic_passes_differ",
                        "path": path,
                        "detail": f"all {passes} passes are bit-identical under a stochastic operator",
                    }
                )

    if num_classes is not None and per_pass_argmax.numel():
        lowest_class = int(per_pass_argmax.min())
        highest_class = int(per_pass_argmax.max())
        if lowest_class < 0 or highest_class >= num_classes:
            failures.append(
                {
                    "check": "16_rate_argmax_in_class_range",
                    "path": path,
                    "detail": (
                        f"argmax range [{lowest_class}, {highest_class}] outside "
                        f"[0, {num_classes})"
                    ),
                }
            )

    return failures, passes


def split_rate_filename(name: str) -> tuple[float, str] | None:
    """("rate_0_05_clean.pt") becomes (0.05, "clean"), or None if it is not one."""
    if not name.startswith("rate_") or not name.endswith(".pt"):
        return None
    tag, split = name[len("rate_") : -len(".pt")].rsplit("_", 1)
    try:
        rate = float(tag.replace("_", "."))
    except ValueError:
        return None
    return rate, split


def audit_position_config(
    psbd_dir: str,
    placement: str,
    baseline_rows: dict[str, int | None],
    num_classes: int | None,
) -> dict:
    """Every rate tensor under one position config, plus checks 17 and the coverage.

    Returns a record carrying the resolved operator, the k values actually stored,
    the complete versus partial rate coverage, and the failures found below it.
    """
    folder = os.path.join(psbd_dir, placement)
    operator, resolved_from = resolve_operator(psbd_dir, placement)
    deterministic = operator in DETERMINISTIC_OPERATORS

    try:
        entries = sorted(os.listdir(folder))
    except OSError as error:
        unreadable = {
            "operator": operator,
            "resolved_from": resolved_from,
            "deterministic": deterministic,
            "rate_files": 0,
            "rate_files_loaded": 0,
            "k_values": {},
            "complete_rates": [],
            "partial_rates": {},
            "failures": [
                {"check": "12_rate_loads", "path": folder, "detail": repr(error)}
            ],
        }
        return unreadable

    failures: list[dict] = []
    passes_seen: dict[int, int] = {}
    splits_by_rate: dict[float, list[str]] = {}
    rate_files = 0
    rate_files_loaded = 0

    for name in entries:
        parsed = split_rate_filename(name)
        if parsed is None:
            continue
        rate, split = parsed
        splits_by_rate.setdefault(rate, []).append(split)
        rate_files += 1
        file_failures, passes = audit_one_rate_file(
            os.path.join(folder, name),
            split,
            baseline_rows.get(split),
            num_classes,
            deterministic,
        )
        failures.extend(file_failures)
        if not any(failure["check"] == "12_rate_loads" for failure in file_failures):
            rate_files_loaded += 1
        if passes is not None:
            passes_seen[passes] = passes_seen.get(passes, 0) + 1

    if len(passes_seen) > 1:
        failures.append(
            {
                "check": "17_position_k_consistent",
                "path": folder,
                "detail": f"mixed forward-pass counts across rates and splits: {passes_seen}",
            }
        )

    complete = complete_rates(psbd_dir, placement)
    partial = {
        rate: sorted(splits)
        for rate, splits in sorted(splits_by_rate.items())
        if rate not in set(complete)
    }

    record = {
        "operator": operator,
        "resolved_from": resolved_from,
        "deterministic": deterministic,
        "rate_files": rate_files,
        "rate_files_loaded": rate_files_loaded,
        "k_values": passes_seen,
        "complete_rates": complete,
        "partial_rates": {f"{rate:g}": splits for rate, splits in partial.items()},
        "failures": failures,
    }
    return record


def audit_one_checkpoint(task: tuple[str, str, str]) -> dict:
    """Everything under one results/<folder>/psbd/ subtree, as a plain dict.

    Wrapped so a single unreadable subtree is reported rather than killing the pool
    and losing the other 797 results.
    """
    folder, results_dir, checkpoints_dir = task
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    record: dict = {
        "folder": folder,
        "psbd_dir": psbd_dir,
        "files_checked": 0,
        "failures": [],
        "position_configs": {},
    }

    try:
        with open(os.path.join(psbd_dir, "split_manifest.json")) as handle:
            manifest = json.load(handle)
    except Exception as error:
        record["failures"].append(
            {
                "check": "01_manifest_parses",
                "path": os.path.join(psbd_dir, "split_manifest.json"),
                "detail": repr(error),
            }
        )
        return record

    record["files_checked"] += 1
    dataset, dataset_source = read_dataset_name(checkpoints_dir, folder, manifest)
    num_classes = (
        DATASET_REGISTRY[dataset].num_classes if dataset in DATASET_REGISTRY else None
    )
    record["dataset"] = dataset
    record["dataset_source"] = dataset_source
    record["num_classes"] = num_classes
    record["probe_attack"] = manifest.get("probe_attack")

    try:
        manifest_failures, expected_rows = audit_manifest(psbd_dir, manifest, dataset)
        record["failures"].extend(manifest_failures)

        baseline_rows: dict[str, int | None] = {}
        baseline_files_seen = 0
        baseline_files_loaded = 0
        for split in SPLITS:
            path = os.path.join(psbd_dir, f"baseline_{split}.pt")
            if not os.path.exists(path):
                continue
            baseline_failures, rows = audit_one_baseline(
                path, split, expected_rows[split], num_classes
            )
            record["failures"].extend(baseline_failures)
            baseline_rows[split] = rows
            baseline_files_seen += 1
            if not any(
                failure["check"] == "06_baseline_loads" for failure in baseline_failures
            ):
                baseline_files_loaded += 1
            record["files_checked"] += 1
        record["baseline_rows"] = baseline_rows
        record["baseline_files_seen"] = baseline_files_seen
        record["baseline_files_loaded"] = baseline_files_loaded
        record["expected_rows"] = expected_rows

        for name in sorted(os.listdir(psbd_dir)):
            if not os.path.isdir(os.path.join(psbd_dir, name)):
                continue
            config_record = audit_position_config(
                psbd_dir, name, baseline_rows, num_classes
            )
            record["failures"].extend(config_record.pop("failures"))
            record["files_checked"] += config_record["rate_files"]
            record["position_configs"][name] = config_record
    except Exception:
        record["failures"].append(
            {
                "check": "00_checkpoint_audit_completes",
                "path": psbd_dir,
                "detail": traceback.format_exc(),
            }
        )

    return record


def summarize(records: list[dict]) -> dict:
    """Pass and fail counts per check, plus the cache-wide coverage tallies.

    A check's denominator is how many times it was actually evaluated, so a file
    that failed to load does not silently inflate the pass count of the checks that
    could never run on it.
    """
    attempted: dict[str, int] = {check: 0 for check in CHECK_IDS}
    failed: dict[str, int] = {check: 0 for check in CHECK_IDS}

    files_checked = 0
    placements = 0
    complete_rate_slots = 0
    partial_rate_slots = 0
    identical_pass_by_operator: dict[str, int] = {}
    operator_counts: dict[str, int] = {}
    k_counts: dict[str, int] = {}

    for record in records:
        files_checked += record.get("files_checked", 0)
        baselines_seen = record.get("baseline_files_seen", 0)
        baselines_loaded = record.get("baseline_files_loaded", 0)
        configs = record.get("position_configs", {})
        rates_seen = sum(config["rate_files"] for config in configs.values())
        rates_loaded = sum(config["rate_files_loaded"] for config in configs.values())
        config_count = len(configs)
        placements += config_count

        attempted["00_checkpoint_audit_completes"] += 1
        attempted["01_manifest_parses"] += 1
        # A checkpoint whose manifest never parsed cannot have had the manifest
        # content checks run on it, so it must not count toward their denominator.
        manifest_parsed = "dataset" in record
        for check in (
            "02_manifest_partitions_n_total",
            "03_manifest_n_heldout_matches",
            "04_manifest_backdoor_subset_of_clean",
            "05_manifest_label_mode_matches_attack",
        ):
            attempted[check] += 1 if manifest_parsed else 0
        # A file that never opened cannot have had the checks below the load check
        # run on it, so it counts only toward the load check's denominator.
        attempted["06_baseline_loads"] += baselines_seen
        for check in (
            "07_baseline_row_counts_agree",
            "08_baseline_rows_match_manifest",
            "09_baseline_probs_finite_and_normalized",
            "10_baseline_labels_are_argmax",
            "11_baseline_num_classes_matches_dataset",
        ):
            attempted[check] += baselines_loaded
        attempted["12_rate_loads"] += rates_seen
        for check in (
            "13_rate_shapes_agree",
            "14_rate_rows_match_baseline",
            "15_rate_probs_in_unit_range",
            "16_rate_argmax_in_class_range",
        ):
            attempted[check] += rates_loaded
        attempted["17_position_k_consistent"] += config_count

        for config in configs.values():
            operator = config["operator"]
            operator_counts[operator] = operator_counts.get(operator, 0) + 1
            for passes, count in config["k_values"].items():
                k_counts[str(passes)] = k_counts.get(str(passes), 0) + count
            if not config["deterministic"]:
                attempted["18_stochastic_passes_differ"] += config["rate_files_loaded"]
            complete_rate_slots += len(config["complete_rates"])
            partial_rate_slots += len(config["partial_rates"])

        for failure in record.get("failures", []):
            check = failure["check"]
            failed[check] = failed.get(check, 0) + 1

    for record in records:
        by_config = {
            name: config for name, config in record.get("position_configs", {}).items()
        }
        for failure in record.get("failures", []):
            if failure["check"] != "18_stochastic_passes_differ":
                continue
            config_name = os.path.basename(os.path.dirname(failure["path"]))
            operator = by_config.get(config_name, {}).get("perturbation", "unknown")
            identical_pass_by_operator[operator] = (
                identical_pass_by_operator.get(operator, 0) + 1
            )

    summary = {
        "checkpoints_checked": len(records),
        "files_checked": files_checked,
        "position_configs_checked": placements,
        "complete_rate_slots": complete_rate_slots,
        "partial_rate_slots": partial_rate_slots,
        "operator_counts": dict(sorted(operator_counts.items())),
        "k_value_counts": dict(sorted(k_counts.items(), key=lambda item: int(item[0]))),
        "identical_pass_failures_by_operator": dict(
            sorted(identical_pass_by_operator.items())
        ),
        "checks": {
            check: {
                "attempted": attempted[check],
                "failed": failed[check],
                "passed": max(attempted[check] - failed[check], 0),
            }
            for check in CHECK_IDS
        },
    }
    return summary


def collect_failures(records: list[dict]) -> dict[str, list[dict]]:
    """Every failure in the run, grouped by check identifier."""
    grouped: dict[str, list[dict]] = {}
    for record in records:
        for failure in record.get("failures", []):
            grouped.setdefault(failure["check"], []).append(
                {"folder": record["folder"], **failure}
            )
    return grouped


def print_report(
    summary: dict, grouped: dict[str, list[dict]], max_shown: int, output: str
) -> None:
    """The stdout half of the deliverable. The JSON always carries the full lists."""
    print()
    print("PSBD cache integrity")
    print(f"  checkpoints checked     {summary['checkpoints_checked']}")
    print(f"  files checked           {summary['files_checked']}")
    print(f"  position configs        {summary['position_configs_checked']}")
    print(
        f"  rate coverage           {summary['complete_rate_slots']} complete, "
        f"{summary['partial_rate_slots']} partial"
    )
    print(f"  stored k values         {summary['k_value_counts']}")
    print()
    print(f"{'check':<44}{'attempted':>12}{'passed':>12}{'failed':>10}")
    for check, counts in summary["checks"].items():
        print(
            f"{check:<44}{counts['attempted']:>12}{counts['passed']:>12}{counts['failed']:>10}"
        )

    total_failed = sum(counts["failed"] for counts in summary["checks"].values())
    print()
    if total_failed == 0:
        print("No failures.")
    else:
        print(f"{total_failed} failures across {len(grouped)} checks")
        for check in CHECK_IDS:
            failures = grouped.get(check, [])
            if not failures:
                continue
            print()
            print(f"  {check}: {len(failures)} failing")
            for failure in failures[:max_shown]:
                print(f"    {failure['path']}")
                print(f"      {failure['detail'].splitlines()[0]}")
            if len(failures) > max_shown:
                print(f"    ... {len(failures) - max_shown} more, see {output}")

    if summary["identical_pass_failures_by_operator"]:
        print()
        print("  identical-pass anomalies by operator:")
        for operator, count in summary["identical_pass_failures_by_operator"].items():
            print(f"    {operator:<24}{count}")
    print()
    print(f"report written to {output}")


def write_report(path: str, payload: dict) -> None:
    """The only write this script performs, and never into the cache tree itself."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output = args.output or experiment_result_path(
        "cache_integrity", "cache_integrity.json", args.results_dir
    )

    folders = discover_checkpoint_folders(args.results_dir)
    selected = select_folders(folders, args.sample, args.seed)
    print(
        f"auditing {len(selected)} of {len(folders)} checkpoints with {args.workers} workers",
        flush=True,
    )

    started = datetime.datetime.now()
    tasks = [(folder, args.results_dir, args.checkpoints_dir) for folder in selected]
    records: list[dict] = []
    with multiprocessing.Pool(
        processes=args.workers, initializer=torch.set_num_threads, initargs=(1,)
    ) as pool:
        for done, record in enumerate(
            pool.imap_unordered(audit_one_checkpoint, tasks, chunksize=1), 1
        ):
            records.append(record)
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} checkpoints", flush=True)
    records.sort(key=lambda record: record["folder"])

    summary = summarize(records)
    grouped = collect_failures(records)
    payload = {
        "generated_at": started.isoformat(timespec="seconds"),
        "elapsed_seconds": round(
            (datetime.datetime.now() - started).total_seconds(), 1
        ),
        "git_commit": current_git_commit(),
        "mode": "full" if args.sample is None else "sample",
        "sample_size": args.sample,
        "seed": args.seed,
        "results_dir": args.results_dir,
        "checkpoints_dir": args.checkpoints_dir,
        "summary": summary,
        "failures_by_check": grouped,
        "checkpoints": {record["folder"]: record for record in records},
    }
    write_report(output, payload)
    print_report(summary, grouped, args.max_shown, output)


if __name__ == "__main__":
    main()
