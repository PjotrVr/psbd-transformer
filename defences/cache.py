"""On-disk layout for the 2-stage PSBD sweep.

Stage 1 (GPU, expensive) writes the no-perturbation baseline and the raw
per-pass probabilities here. Stage 2 (CPU, cheap) reads them back to pick
thresholds and compute TPR/FPR/AUROC at any quantile, with no GPU rerun.

Layout under results/<checkpoint_folder>/psbd/:

    split_manifest.json          the one index record for the whole subtree
    baseline_<split>.pt          no-perturbation probs and argmax labels, per split
    <position_config>/
        rate_<tag>_<split>.pt    (passes, n) tracked-class probs

The baseline and manifest sit a level above the position-config folders because
they depend only on the checkpoint (and the split), not on position or rate, so
they are written once and read by every one of the checkpoint's jobs.

Hundreds of thousands of these files already exist, so every path spelling and
every payload key here is a fixed on-disk contract.
"""

import json
import os

import torch

from .inference import build_baseline_cache


def baseline_path(psbd_dir: str, split: str) -> str:
    """Where a split's no-perturbation baseline lives."""
    path = os.path.join(psbd_dir, f"baseline_{split}.pt")
    return path


def dropout_pass_path(
    psbd_dir: str, position_config: str, rate: float, split: str
) -> str:
    """Where the raw per-pass tensors for a (position, rate, split) live."""
    path = os.path.join(psbd_dir, position_config, f"rate_{_rate_tag(rate)}_{split}.pt")
    return path


def manifest_path(psbd_dir: str) -> str:
    """Where the subtree's single split manifest lives."""
    path = os.path.join(psbd_dir, "split_manifest.json")
    return path


def run_provenance_path(psbd_dir: str, placement: str) -> str:
    """Where cli.sweep records the commit, position, operator and device of a placement."""
    path = os.path.join(psbd_dir, f"run_{placement}.json")
    return path


# Caches written before the vocabulary settled on position and operator carry
# the older key names. 1016 of them are on disk, so the reader translates rather
# than every consumer carrying a fallback.
LEGACY_PROVENANCE_KEYS = {"position_config": "position", "perturbation": "operator"}


def read_run_provenance(psbd_dir: str, placement: str) -> dict:
    """The run provenance of a placement, empty when it predates the record or is unreadable."""
    path = run_provenance_path(psbd_dir, placement)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    for old, new in LEGACY_PROVENANCE_KEYS.items():
        if old in payload and new not in payload:
            payload[new] = payload.pop(old)
    return payload


def save_baseline(
    path: str,
    probs: torch.Tensor,
    labels: torch.Tensor,
    loader_labels: torch.Tensor,
) -> None:
    """Save the no-perturbation state of a split: probabilities, prediction and target.

    probs is the (n, num_classes) softmax, labels its (n,) argmax and
    loader_labels the (n,) label the loader asked for. On the backdoor split that
    is the attack-success label, so labels == loader_labels is the per-sample
    record of whether the trigger actually worked on this image.
    """
    _atomic_save(
        {"probs": probs, "labels": labels, "loader_labels": loader_labels}, path
    )


def load_baseline(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(probs, labels, loader_labels) from a baseline file, the last empty if unsaved.

    probs is (n, num_classes) and labels (n,). loader_labels is (n,) too, except
    from a file written before loader labels were saved, where it is (0,).
    """
    blob = torch.load(path, map_location="cpu")
    probs = blob["probs"]  # (n, num_classes)
    labels = blob["labels"]  # (n,)
    loader_labels = blob.get("loader_labels", torch.empty(0, dtype=torch.long))
    return probs, labels, loader_labels


def save_dropout_pass_probs(
    path: str, per_pass_probs: torch.Tensor, per_pass_argmax: torch.Tensor
) -> None:
    """Save both raw per-pass tensors for a (position, rate, split).

    per_pass_probs and per_pass_argmax are both (passes, n).
    """
    _atomic_save(
        {"per_pass_probs": per_pass_probs, "per_pass_argmax": per_pass_argmax}, path
    )


def load_dropout_pass_probs(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(probs, argmax), both (passes, n).

    argmax is absent from files written before it was saved, so it comes back as
    an empty (0, 0) tensor rather than raising. Stage 2 then reports sigma as
    unavailable for those instead of failing the whole checkpoint.
    """
    blob = torch.load(path, map_location="cpu")
    probs = blob["per_pass_probs"]  # (passes, n)
    argmax = blob.get("per_pass_argmax", torch.empty(0, 0, dtype=torch.int16))
    return probs, argmax


def write_split_manifest(psbd_dir: str, manifest: dict) -> None:
    """Write the split manifest once, idempotent for an identical split.

    All of a checkpoint's jobs derive the same deterministic split, so they build
    byte-identical manifests. Rewriting is skipped when an identical file is
    already present, which keeps its mtime stable for the reuse check and is safe
    under the 60-job race. A file that exists but differs is a real conflict: a
    truncated max_samples run or a different seed wrote into this results dir.
    That raises rather than silently keeping stale indices, since every tensor
    under this subtree is ordered by whichever manifest wins.
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
    """The subtree's split manifest."""
    with open(manifest_path(psbd_dir)) as handle:
        manifest = json.load(handle)
    return manifest


def load_or_build_baseline(
    psbd_dir: str,
    split: str,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_bfloat16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The no-perturbation baseline for a split, computed once and reused.

    Returns (probs, labels, loader_labels): the (n, num_classes) softmax, its
    (n,) argmax and the (n,) label the loader asked for, in loader order.

    The baseline depends only on (checkpoint, split), never on position or rate,
    but a checkpoint's 10 position-configs run as 10 separate jobs, so there is no
    long-lived process to hold it in memory across them. Whichever job runs first
    writes baseline_<split>.pt, the rest just load it. Safe under a rare
    near-simultaneous race: the no-perturbation forward pass is deterministic, so
    2 jobs computing it at once write the same values.
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
            empty = torch.empty(0, dtype=empty_dtype)
            return empty
        stacked = torch.cat([row[key] for row in cache])  # batches joined along n
        return stacked

    probs = stack("probs", torch.float32)  # (n, num_classes)
    labels = stack("labels", torch.long)  # (n,)
    loader_labels = stack("loader_labels", torch.long)  # (n,)

    save_baseline(path, probs, labels, loader_labels)
    return probs, labels, loader_labels


def _rate_tag(rate: float) -> str:
    """0.1 becomes "0_1", matching the underscore-for-decimal folder convention.

    %g rather than a fixed decimal count, because the sub-0.1 rates a
    residual-stream position needs (0.01 to 0.09, where (1-p)^12 has not yet
    collapsed) would all round to "0_0" at 1 decimal place and silently overwrite
    each other's files. %g keeps the existing "0_1" spelling for the main grid
    while staying injective below it.
    """
    tag = f"{rate:g}".replace(".", "_")
    return tag


def _atomic_save(payload: dict, path: str) -> None:
    """Write through a private temp file, then rename.

    A checkpoint's position-config jobs run concurrently and share the baseline
    and manifest, so a reader can arrive mid-write. os.replace is atomic within a
    filesystem, so a reader sees either the old file or the complete new file,
    never a truncated file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)
