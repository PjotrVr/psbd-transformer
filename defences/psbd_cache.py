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


def _atomic_save(payload: dict, path: str) -> None:
    """Write through a private temp file, then rename.

    A checkpoint's position-config jobs run concurrently and share the baseline
    and manifest, so a reader can arrive mid-write. os.replace is atomic within a
    filesystem, so a reader sees either the old file or the complete new one,
    never a truncated one.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rate_tag(rate: float) -> str:
    """0.1 becomes "0_1", matching the underscore-for-decimal folder convention.

    %g rather than a fixed decimal count, because the sub-0.1 rates a
    residual-stream position needs (0.01 to 0.09, where (1-p)^12 has not yet
    collapsed) would all round to "0_0" at one decimal place and silently
    overwrite each other's files. %g keeps the existing "0_1" spelling for the
    main grid while staying injective below it.
    """
    return f"{rate:g}".replace(".", "_")


def baseline_path(psbd_dir: str, split: str) -> str:
    return os.path.join(psbd_dir, f"baseline_{split}.pt")


def dropout_pass_path(
    psbd_dir: str, position_config: str, rate: float, split: str
) -> str:
    return os.path.join(psbd_dir, position_config, f"rate_{_rate_tag(rate)}_{split}.pt")


def manifest_path(psbd_dir: str) -> str:
    return os.path.join(psbd_dir, "split_manifest.json")


def save_baseline(
    path: str,
    probs: torch.Tensor,
    labels: torch.Tensor,
    loader_labels: torch.Tensor,
) -> None:
    """The no-dropout state of one split: probabilities, prediction, and target.

    loader_labels is what the loader asked for. On the backdoor split that is the
    attack-success label, so labels == loader_labels is the per-sample record of
    whether the trigger actually worked on this image.
    """
    _atomic_save(
        {"probs": probs, "labels": labels, "loader_labels": loader_labels}, path
    )


def load_baseline(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blob = torch.load(path, map_location="cpu")
    loader_labels = blob.get("loader_labels", torch.empty(0, dtype=torch.long))
    return blob["probs"], blob["labels"], loader_labels


def save_dropout_pass_probs(
    path: str, per_pass_probs: torch.Tensor, per_pass_argmax: torch.Tensor
) -> None:
    """Both raw per-pass tensors for one (position, rate, split), shaped (k, N)."""
    _atomic_save(
        {"per_pass_probs": per_pass_probs, "per_pass_argmax": per_pass_argmax}, path
    )


def load_dropout_pass_probs(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (probs, argmax). argmax is absent from files written before it was
    saved, so it comes back empty rather than raising, and stage 2 reports sigma
    as unavailable for those instead of failing the whole checkpoint."""
    blob = torch.load(path, map_location="cpu")
    argmax = blob.get("per_pass_argmax", torch.empty(0, 0, dtype=torch.int16))
    return blob["per_pass_probs"], argmax


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
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temporary, path)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        probs, labels, loader_labels = load_baseline(path)
        expected = len(loader.dataset)
        if probs.shape[0] != expected:
            raise ValueError(
                f"cached {path} has {probs.shape[0]} rows but the loader serves "
                f"{expected}. A truncated max_samples run likely wrote it into this "
                "results dir. Delete it and rerun, or use a separate --results-dir."
            )
        return probs, labels, loader_labels

    cache = build_baseline_cache(model, loader, device, use_bfloat16)

    def stack(key: str, empty_dtype) -> torch.Tensor:
        if not cache:
            return torch.empty(0, dtype=empty_dtype)
        return torch.cat([row[key] for row in cache])

    probs = stack("probs", torch.float32)
    labels = stack("labels", torch.long)
    loader_labels = stack("loader_labels", torch.long)
    save_baseline(path, probs, labels, loader_labels)
    return probs, labels, loader_labels
