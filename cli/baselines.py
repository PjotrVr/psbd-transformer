"""Score the competitor detectors on exactly the splits PSBD is scored on.

Writes 1 record per detector, results/<folder>/detectors/<name>_metrics.json,
beside that detector's per-split score tensors, the layout detectors.records
describes. A job that runs 1 detector never touches another's file, finished
work is skipped per (checkpoint, detector), and every quantile, pairing or
fusion is recomputable from the saved scores on a CPU.

Fairness is the whole point of this command, so the shared parts are shared
literally: the same build_psbd_loaders_from_checkpoint, the same clean-validation
split for thresholding, the same pair_clean_to_backdoor subsetting, the same
quantile grid and the same detection_report. The only thing that differs between
methods is the score. A folder is refused unless the split this run builds is
the split PSBD's cache was built on, so a detector column can never sit beside
a PSBD column that scored different rows.

Every detector returns low for poisoned, matching PSU, so nothing downstream
special-cases them. A detector that fits per-sample statistics on the validation
split returns out-of-fit scores for it (detectors.CROSS_FITTED), so its threshold
is never read from scores it was fitted to.

Example
    python -m cli.baselines --checkpoint-folder vit_gtsrb_badnet_a2o_0_05
    python -m cli.baselines --all --detectors strip teco --skip-existing
    python -m cli.baselines --checkpoint-folder vit_gtsrb_benign \\
        --probe-attack badnet_a2o --probe-target-label 0
    python -m cli.baselines --checkpoint-folder vit_gtsrb_badnet_a2o_0_05 \\
        --detectors cd_l --max-samples 200 \\
        --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
"""

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.registry import DATASET_REGISTRY
from data.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.cache import (
    baseline_path,
    load_baseline,
    manifest_path,
    read_split_manifest,
)
from defences.decision import (
    HEADLINE_QUANTILE,
    PSBD_QUANTILES,
    attack_success_mask,
    detection_report,
    pair_clean_to_backdoor,
    threshold_diagnostics,
)
from detectors import (
    CROSS_FITTED,
    DATA_REQUIREMENT,
    DETECTOR_NAMES,
    EXPERIMENTAL_DETECTOR_NAMES,
    FORWARD_PASSES_PER_INPUT,
    NEEDS_FITTING,
    DetectorContext,
    build_detector,
    effective_hyperparameters,
    effective_precision,
)
from detectors.records import (
    SPLITS,
    STATUS_FAILED,
    STATUS_SCORED,
    manifest_digest,
    report_path,
    save_report,
    save_scores,
    scored_detectors,
    scores_path,
)
from models.backbones import load_checkpoint
from training.loop import current_git_commit, utc_timestamp

# The trigger a benign checkpoint is probed with when the caller names none, so
# its numbers sit on the same footing as its PSBD numbers.
BENIGN_PROBE_ATTACK = "badnet_a2o"

DEFAULT_RESULTS_DIR = "results"
MANIFEST_INDEX_KEYS = (
    "heldout_indices",
    "analysis_clean_indices",
    "analysis_backdoor_indices",
)


def build_parser() -> argparse.ArgumentParser:
    """The command's parser, exposed so a job generator can validate its flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=DETECTOR_NAMES + EXPERIMENTAL_DETECTOR_NAMES,
        default=list(DETECTOR_NAMES),
        help="which detectors to score, experimental ones only when named",
    )
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument("--probe-attack", default=None)
    parser.add_argument("--probe-target-label", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="truncate every split, smoke runs only, needs an explicit --results-dir",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a detector whose record on a folder already reached the scored status",
    )
    parser.add_argument(
        "--allow-missing-psbd-cache",
        action="store_true",
        help="score a folder with no results/<folder>/psbd/ cache, smoke runs only",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.results_dir == DEFAULT_RESULTS_DIR:
        parser.error(
            "--max-samples writes truncated records, so it needs an explicit "
            "--results-dir away from the real results tree"
        )
    return args


@dataclass(frozen=True)
class ScoringCase:
    """Everything a detector needs on 1 checkpoint, loaded once and shared."""

    folder: str
    metadata: dict
    loaders: dict[str, DataLoader]
    manifest: dict
    model: nn.Module
    context: DetectorContext
    psbd_dir: str
    manifest_matches_cache: bool | None
    captured: torch.Tensor | None


def discover_folders(results_dir: str) -> list[str]:
    """Folders that already have a PSBD cache, so the comparison is like for like."""
    if not os.path.isdir(results_dir):
        return []

    folders = sorted(
        name
        for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name, "psbd"))
    )
    return folders


def resolve_probe(
    metadata: dict, args: argparse.Namespace
) -> tuple[str | None, int | None]:
    """The trigger to build the eval sets with, and its target label."""
    probe = args.probe_attack
    if probe is None and metadata["attack"] == "benign":
        probe = BENIGN_PROBE_ATTACK
    target = args.probe_target_label if probe else None
    return probe, target


def manifest_matches_psbd_cache(manifest: dict, psbd_dir: str) -> bool | None:
    """Whether this run's split equals the one PSBD's cache was written on.

    None when no cache exists to compare against. A truncated smoke split never
    equals a full cache, and that is a truthful False rather than a comparison
    to skip.
    """
    if not os.path.exists(manifest_path(psbd_dir)):
        return None
    cached = read_split_manifest(psbd_dir)
    matches = all(manifest[key] == cached[key] for key in MANIFEST_INDEX_KEYS)
    return matches


def captured_mask_from_cache(psbd_dir: str, n_backdoor: int) -> torch.Tensor | None:
    """Which backdoor rows the trigger actually flipped, read from PSBD's baseline.

    The cached no-perturbation argmax and the attack-success label give the
    captured mask defences.decision.attack_success_mask defines, the same mask
    cli.analyze uses for PSBD's captured-only view. None when the cache is absent
    or was written on a different split, since then its rows do not line up.
    """
    path = baseline_path(psbd_dir, "backdoor")
    if not os.path.exists(path):
        return None
    _, labels, loader_labels = load_baseline(path)
    captured = attack_success_mask(labels, loader_labels)
    if captured is None or captured.numel() != n_backdoor:
        return None
    return captured


def load_case(
    folder: str, args: argparse.Namespace, device: torch.device
) -> ScoringCase:
    """Load a checkpoint, build its 3 PSBD splits and check them against the cache."""
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    probe, probe_target = resolve_probe(metadata, args)

    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        probe_attack=probe,
        probe_target_label=probe_target,
    )

    psbd_dir = os.path.join(args.results_dir, folder, "psbd")
    matches = manifest_matches_psbd_cache(manifest, psbd_dir)
    if matches is None and not args.allow_missing_psbd_cache:
        raise FileNotFoundError(
            f"{psbd_dir} has no split manifest, so this run cannot be checked against "
            "PSBD's split. Point --results-dir at the tree holding the PSBD caches, or "
            "pass --allow-missing-psbd-cache for a smoke run."
        )
    if matches is False and args.max_samples is None:
        raise ValueError(
            f"the split built for {folder} differs from the one in {psbd_dir}, so its "
            "detector scores would not line up row for row with PSBD's"
        )

    model = load_checkpoint(metadata["architecture"], checkpoint_path, device)
    spec = DATASET_REGISTRY[metadata["dataset"]]
    context = DetectorContext(
        model=model,
        device=device,
        mean=spec.mean,
        std=spec.std,
        validation_loader=loaders["validation"],
        num_classes=spec.num_classes,
        use_bfloat16=not args.no_bfloat16,
        seed=PSBD_SPLIT_SEED,
    )
    captured = (
        captured_mask_from_cache(psbd_dir, len(loaders["backdoor"].dataset))
        if matches
        else None
    )

    case = ScoringCase(
        folder=folder,
        metadata=metadata,
        loaders=loaders,
        manifest=manifest,
        model=model,
        context=context,
        psbd_dir=psbd_dir,
        manifest_matches_cache=matches,
        captured=captured,
    )
    return case


def check_scores(name: str, split: str, scores: torch.Tensor, expected: int) -> None:
    """Refuse a score tensor that could be written and read as a number."""
    if scores.ndim != 1 or scores.numel() != expected:
        raise ValueError(
            f"{name} returned shape {tuple(scores.shape)} on the {split} split, "
            f"expected ({expected},)"
        )
    if not torch.isfinite(scores).all():
        raise ValueError(f"{name} returned a non-finite score on the {split} split")


def score_detector(
    name: str, case: ScoringCase, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Per-split scores of 1 detector, each (n_split,), and the seconds each took.

    Fitting happens inside build_detector and is timed under the "fit" key, so
    the fixed cost a deployment pays once is separable from the per-input cost.
    """
    started = time.perf_counter()
    detector = build_detector(name, case.context)
    runtimes = {"fit": time.perf_counter() - started}

    scores = {}
    for split in SPLITS:
        started = time.perf_counter()
        values = detector(case.model, case.loaders[split], device)  # (n_split,)
        runtimes[split] = time.perf_counter() - started
        check_scores(name, split, values, len(case.loaders[split].dataset))
        scores[split] = values.detach().cpu().float()

    return scores, runtimes


def detection_blocks(
    scores: dict[str, torch.Tensor], case: ScoringCase
) -> tuple[dict, dict | None]:
    """detection_report at every PSBD quantile, on all triggered rows and on captured ones.

    The clean side is paired to the backdoor split's images before anything is
    thresholded, and the captured-only view subsets both sides by the same mask,
    the 2 rules cli.analyze applies to PSBD.
    """
    paired_clean = pair_clean_to_backdoor(
        scores["clean"], case.manifest
    )  # (n_backdoor,)
    backdoor = scores["backdoor"]  # (n_backdoor,)
    assert paired_clean.shape == backdoor.shape, (paired_clean.shape, backdoor.shape)

    def blocks(clean_side: torch.Tensor, backdoor_side: torch.Tensor) -> dict:
        reports = {}
        for quantile in PSBD_QUANTILES:
            report = detection_report(
                scores["validation"], clean_side, backdoor_side, quantile
            )
            report.update(
                threshold_diagnostics(
                    scores["validation"], clean_side, backdoor_side, quantile
                )
            )
            reports[f"q{quantile:.2f}"] = report
        return reports

    detection = blocks(paired_clean, backdoor)
    captured_only = None
    if case.captured is not None and bool(case.captured.any()):
        captured_only = blocks(paired_clean[case.captured], backdoor[case.captured])
    return detection, captured_only


def score_summary(scores: dict[str, torch.Tensor]) -> dict:
    """Per-split moments and range, enough to spot a degenerate score at a glance."""
    summary = {
        split: {
            "n": int(values.numel()),
            "mean": float(values.mean()),
            "std": float(values.std()) if values.numel() > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
        for split, values in scores.items()
    }
    return summary


def build_record(
    name: str,
    case: ScoringCase,
    scores: dict[str, torch.Tensor],
    runtimes: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """The record detectors.records stores for 1 detector on 1 checkpoint."""
    detection, captured_only = detection_blocks(scores, case)
    n_backdoor = int(scores["backdoor"].numel())
    record = {
        "detector": name,
        "status": STATUS_SCORED,
        "folder_name": case.folder,
        "dataset": case.metadata["dataset"],
        "attack": case.metadata["attack"],
        "poison_rate": case.metadata.get("poison_rate"),
        "architecture": case.metadata["architecture"],
        "probe_attack": case.manifest.get("probe_attack"),
        "probe_target_label": case.manifest.get("probe_target_label"),
        "label_mode": case.manifest.get("label_mode"),
        "headline_quantile": HEADLINE_QUANTILE,
        "detection": detection,
        "detection_captured_only": captured_only,
        "provenance": {
            "git_commit": current_git_commit(),
            "scored_at": utc_timestamp(),
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "precision": effective_precision(name, case.context),
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "forward_passes_per_input": FORWARD_PASSES_PER_INPUT[name],
            "data_requirement": DATA_REQUIREMENT[name],
            "needs_fitting": name in NEEDS_FITTING,
            "cross_fitted": name in CROSS_FITTED,
            "hyperparameters": effective_hyperparameters(name, case.context),
            "runtime_seconds": runtimes,
            "split": {
                "seed": case.manifest["seed"],
                "n_validation": int(scores["validation"].numel()),
                "n_clean": int(scores["clean"].numel()),
                "n_backdoor": n_backdoor,
                "n_backdoor_captured": int(case.captured.sum())
                if case.captured is not None
                else None,
                "manifest_sha256": manifest_digest(case.manifest),
                "manifest_matches_psbd_cache": case.manifest_matches_cache,
            },
            "score_summary": score_summary(scores),
        },
    }
    return record


def failed_record(name: str, case_folder: str, error: BaseException) -> dict:
    """A record saying the detector did not run, so the gap is visible and retried."""
    record = {
        "detector": name,
        "status": STATUS_FAILED,
        "folder_name": case_folder,
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
        "provenance": {
            "git_commit": current_git_commit(),
            "scored_at": utc_timestamp(),
        },
    }
    return record


def write_scored(
    args: argparse.Namespace, name: str, case: ScoringCase, scores: dict, record: dict
) -> None:
    """Persist the score tensors first, then the record that summarises them."""
    for split, values in scores.items():
        save_scores(scores_path(args.results_dir, case.folder, name, split), values)
    save_report(report_path(args.results_dir, case.folder, name), record)


def pending_detectors(args: argparse.Namespace, folder: str) -> list[str]:
    """The requested detectors still to score on a folder."""
    if not args.skip_existing:
        return list(args.detectors)
    done = scored_detectors(args.results_dir, folder)
    pending = [name for name in args.detectors if name not in done]
    return pending


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folders = discover_folders(args.results_dir) if args.all else args.checkpoint_folder
    if not folders:
        raise SystemExit("nothing to score: pass --checkpoint-folder or --all")

    headline = f"q{HEADLINE_QUANTILE:.2f}"
    failures: list[tuple[str, str, str]] = []
    for folder in folders:
        pending = pending_detectors(args, folder)
        if not pending:
            print(
                f"[skip] {folder}: every requested detector already scored", flush=True
            )
            continue

        try:
            case = load_case(folder, args, device)
        except Exception as error:
            traceback.print_exc()
            failures.extend(
                (folder, name, f"{type(error).__name__}: {error}") for name in pending
            )
            continue

        summaries = []
        for name in pending:
            try:
                scores, runtimes = score_detector(name, case, device)
                record = build_record(name, case, scores, runtimes, args, device)
                write_scored(args, name, case, scores, record)
            except Exception as error:
                traceback.print_exc()
                save_report(
                    report_path(args.results_dir, folder, name),
                    failed_record(name, folder, error),
                )
                failures.append((folder, name, f"{type(error).__name__}: {error}"))
                continue
            auroc = record["detection"][headline]["auroc"]
            seconds = sum(runtimes.values())
            summaries.append(f"{name} auroc={auroc:.3f} ({seconds:.0f}s)")
        print(f"{folder}  " + "  ".join(summaries), flush=True)

    if failures:
        print("\nFAILED", file=sys.stderr)
        for folder, name, message in failures:
            print(f"  {folder}  {name}  {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
