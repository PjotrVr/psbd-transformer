"""On-disk layout for the two-stage PSBD sweep.

Stage 1 (GPU, expensive) writes the no-dropout baseline and the raw per-pass
dropout probabilities here. Stage 2 (CPU, cheap) reads them back to pick
thresholds and compute TPR/FPR/AUROC at any quantile, with no GPU rerun.

Layout under results/<checkpoint_folder>/psbd/:

    split_manifest.json          the one index record for the whole subtree
    baseline_<split>.pt          no-dropout probs and argmax labels, per split
    <position_config>/
        rate_<tag>_<split>.pt    (forward_passes, N) tracked-class probs

The baseline and manifest sit one level above the position-config folders because
they depend only on the checkpoint (and the split), not on position or rate, so
they are written once and read by every one of the checkpoint's jobs.
"""

import json
import os

import torch

from .inference import build_baseline_cache


def _rate_tag(rate: float) -> str:
    """0.1 becomes "0_1", matching the underscore-for-decimal folder convention."""
    return f"{rate:.1f}".replace(".", "_")


def baseline_path(psbd_dir: str, split: str) -> str:
    return os.path.join(psbd_dir, f"baseline_{split}.pt")


def dropout_pass_path(
    psbd_dir: str, position_config: str, rate: float, split: str
) -> str:
    return os.path.join(psbd_dir, position_config, f"rate_{_rate_tag(rate)}_{split}.pt")


def manifest_path(psbd_dir: str) -> str:
    return os.path.join(psbd_dir, "split_manifest.json")


def save_baseline(path: str, probs: torch.Tensor, labels: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"probs": probs, "labels": labels}, path)


def load_baseline(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    blob = torch.load(path, map_location="cpu")
    return blob["probs"], blob["labels"]


def save_dropout_pass_probs(path: str, per_pass_probs: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"per_pass_probs": per_pass_probs}, path)


def load_dropout_pass_probs(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu")["per_pass_probs"]


def write_split_manifest(psbd_dir: str, manifest: dict) -> None:
    """Write the split manifest once, idempotent for an identical split.

    All of a checkpoint's jobs derive the same deterministic split, so they build
    byte-identical manifests. Rewriting is skipped when an identical file is
    already present, which keeps its mtime stable for the reuse check and is safe
    under the 60-job race. A file that exists but differs is a real conflict, not
    a race: a truncated max_samples run or a different seed wrote into this
    results dir. That raises rather than silently keeping stale indices, since
    every tensor under this subtree is ordered by whichever manifest wins.
    """
    os.makedirs(psbd_dir, exist_ok=True)
    path = manifest_path(psbd_dir)
    if os.path.exists(path):
        existing = read_split_manifest(psbd_dir)
        if existing != manifest:
            raise ValueError(
                f"{path} already describes a different split (n_heldout "
                f"{existing.get('n_heldout')} vs {manifest.get('n_heldout')}). A "
                "truncated max_samples or different-seed run likely wrote it. Delete "
                "it and rerun, or point --results-dir somewhere separate."
            )
        return
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)


def read_split_manifest(psbd_dir: str) -> dict:
    with open(manifest_path(psbd_dir)) as handle:
        return json.load(handle)


def load_or_build_baseline(
    psbd_dir: str,
    split: str,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_bfloat16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The no-dropout baseline for one split, computed once and reused across jobs.

    The baseline depends only on (checkpoint, split), never on position or rate,
    but §H runs a checkpoint's 10 position-configs as 10 separate jobs, so there
    is no long-lived process to hold it in memory across them. Whichever job runs
    first writes baseline_<split>.pt, the rest just load it. Safe under a rare
    near-simultaneous race: the no-dropout forward pass is deterministic, so two
    jobs computing it at once write the same values.
    """
    path = baseline_path(psbd_dir, split)
    if os.path.exists(path):
        probs, labels = load_baseline(path)
        expected = len(loader.dataset)
        if probs.shape[0] != expected:
            raise ValueError(
                f"cached {path} has {probs.shape[0]} rows but the loader serves "
                f"{expected}. A truncated max_samples run likely wrote it into this "
                "results dir. Delete it and rerun, or use a separate --results-dir."
            )
        return probs, labels

    cache = build_baseline_cache(model, loader, device, use_bfloat16)
    probs = torch.cat([row["probs"] for row in cache]) if cache else torch.empty(0)
    labels = (
        torch.cat([row["labels"] for row in cache])
        if cache
        else torch.empty(0, dtype=torch.long)
    )
    save_baseline(path, probs, labels)
    return probs, labels
