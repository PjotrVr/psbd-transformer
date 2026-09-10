"""Where a detector's numbers live on disk, and the helpers every reader shares.

1 record per detector per checkpoint, results/<folder>/detectors/<name>_metrics.json,
beside the per-split score tensors <name>_scores_{validation,clean,backdoor}.pt. A
record is the product of exactly 1 job, so 4 detector groups scoring the same
checkpoint on 4 nodes never write the same file, and skipping finished work is a
stat on a path. The score tensors are the evidence: any quantile, pairing or
fusion is recomputable from them on a CPU without touching a model.

baseline_metrics.json, the earlier layout that held every detector in 1 file, is
never read again. Readers count it so a table can say how many superseded files
it ignored.
"""

import hashlib
import json
import os

import torch

RECORDS_DIR = "detectors"
LEGACY_REPORT = "baseline_metrics.json"
SPLITS = ("validation", "clean", "backdoor")

# A record's status. Only a scored record counts as finished work, so a failed
# one is retried by the next run instead of silently skipped.
STATUS_SCORED = "scored"
STATUS_FAILED = "failed"


def records_dir(results_dir: str, folder: str) -> str:
    """The directory holding a checkpoint's detector records."""
    path = os.path.join(results_dir, folder, RECORDS_DIR)
    return path


def report_path(results_dir: str, folder: str, name: str) -> str:
    """The metrics record of 1 detector on 1 checkpoint."""
    path = os.path.join(records_dir(results_dir, folder), f"{name}_metrics.json")
    return path


def scores_path(results_dir: str, folder: str, name: str, split: str) -> str:
    """The raw per-sample scores of 1 detector on 1 split, in the loader's order."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    path = os.path.join(records_dir(results_dir, folder), f"{name}_scores_{split}.pt")
    return path


def legacy_report_present(results_dir: str, folder: str) -> bool:
    """Whether the superseded single-file record exists for this checkpoint."""
    present = os.path.exists(os.path.join(results_dir, folder, LEGACY_REPORT))
    return present


def load_report(path: str) -> dict | None:
    """A record, or None when the file does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        report = json.load(handle)
    return report


def scored_detectors(results_dir: str, folder: str) -> set[str]:
    """The detectors whose record on this checkpoint reached the scored status."""
    directory = records_dir(results_dir, folder)
    if not os.path.isdir(directory):
        return set()

    names = set()
    for entry in os.listdir(directory):
        if not entry.endswith("_metrics.json"):
            continue
        report = load_report(os.path.join(directory, entry))
        if report is not None and report.get("status") == STATUS_SCORED:
            names.add(report["detector"])
    return names


def save_report(path: str, report: dict) -> None:
    """Write a record through a private temporary file, then rename.

    A reader arriving mid-write sees either the old file or the complete new
    one, never a truncated one, because os.replace is atomic on 1 filesystem.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(report, handle, indent=2)
    os.replace(temporary, path)


def save_scores(path: str, scores: torch.Tensor) -> None:
    """Write a (N,) float32 score tensor atomically, the same way as save_report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(scores.detach().cpu().float(), temporary)
    os.replace(temporary, path)


def load_scores(path: str) -> torch.Tensor:
    """A saved (N,) score tensor on the CPU."""
    scores = torch.load(path, map_location="cpu")
    return scores


def manifest_digest(manifest: dict) -> str:
    """A sha256 over the 3 index lists that fix every row order under a checkpoint.

    2 runs whose digests agree scored the same images in the same order, so
    their per-row tensors line up. That is the check a table makes before it
    puts a detector's column beside PSBD's.
    """
    payload = json.dumps(
        [
            manifest["heldout_indices"],
            manifest["analysis_clean_indices"],
            manifest["analysis_backdoor_indices"],
        ]
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return digest
