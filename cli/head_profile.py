"""Per-head sensitivity profile: ablate each of the 144 attention heads in turn.

PSBD summarises a random perturbation into a single number per sample. This measures
the whole profile instead: for every attention head, how far the model's
confidence in its own unperturbed prediction falls when exactly that head is
removed.

    original form (PSBD Equation 2, with the expectation replaced by an ablation)
        s_u(x) = P_c(x; theta) - P_c(x; theta \\ u),  c = argmax_j P_j(x; theta)

    descriptive form
        for each head u, delete that head and record how much the confidence in
        the sample's own no-ablation predicted class drops. A sample becomes a
        144-long vector instead of a scalar.

The mask is deterministic, so a single forward pass per head measures it exactly
and no Monte Carlo averaging is needed. 144 passes replace the 3 PSU uses, not
144 x 3.

The head study shows the backdoor occupies few dimensions, that those dimensions
are disjoint across attacks, and that data-free attempts to locate them failed. A
profile sidesteps location entirely: it records the response of every unit and
leaves the summary statistic to analysis, which is CPU-only and can be revisited
without rerunning the GPU work.

Stage 1 only. This writes raw (144, N) tensors per split and computes no metric.

Example
    python -m cli.head_profile --checkpoint-folder vit_cifar10_badnet_a2o_0_01
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from defences.cache import load_or_build_baseline, write_split_manifest
from defences.inference import forward_probs
from defences.operators import fixed_head_mask
from models.positions import plug_dropout, unplug_dropout
from data.splits import PSBD_SPLIT_SEED
from training.loop import current_git_commit

from .sweep import load_model_and_loaders

VIT_BLOCKS = 12
VIT_HEADS_PER_BLOCK = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", required=True, nargs="+")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument("--probe-attack", default=None)
    parser.add_argument("--probe-target-label", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def profile_path(psbd_dir: str, split: str) -> str:
    """Where a split's (144, N) profile tensor lives."""
    return os.path.join(psbd_dir, "head_profile", f"{split}.pt")


def tracked_probs_with_head_ablated(
    model: torch.nn.Module,
    loader: DataLoader,
    baseline_labels: torch.Tensor,
    block: int,
    head: int,
    device: torch.device,
    use_bfloat16: bool,
) -> torch.Tensor:
    """Confidence in each sample's baseline class with a single head deleted, shape (N,).

    block_range restricts the attachment to a single block, so the head index is
    unambiguous: without it the same index would be masked in all 12 blocks at
    once and the profile would have 12 entries, not 144.
    """
    handles = plug_dropout(
        model,
        "vit",
        ("attention_heads",),
        {"attention_heads": fixed_head_mask(head)},
        rate=0.0,
        block_range=(block, block),
    )
    try:
        collected, offset = [], 0
        with torch.no_grad():
            for images, _ in loader:
                probs = forward_probs(model, images, device, use_bfloat16)
                labels = baseline_labels[offset : offset + len(probs)].to(probs.device)
                collected.append(probs.gather(1, labels.unsqueeze(1)).squeeze(1).cpu())
                offset += len(probs)

        tracked = torch.cat(collected)  # (N,)
        return tracked
    finally:
        unplug_dropout(handles)


def profile_one_split(
    model: torch.nn.Module,
    loader: DataLoader,
    baseline_labels: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
) -> torch.Tensor:
    """(144, N) tracked-class confidences, a row per head, in block-major order."""
    rows = []
    for block in range(1, VIT_BLOCKS + 1):
        for head in range(VIT_HEADS_PER_BLOCK):
            rows.append(
                tracked_probs_with_head_ablated(
                    model, loader, baseline_labels, block, head, device, use_bfloat16
                )
            )
        print(f"    block {block}/{VIT_BLOCKS} done", flush=True)

    profile = torch.stack(rows)  # (144, N)
    return profile


def save_profile(path: str, profile: torch.Tensor) -> None:
    """Write a split's profile through a temp file, so a reader never sees a partial."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    torch.save({"profile": profile}, temporary)
    os.replace(temporary, path)


def already_complete(psbd_dir: str, splits) -> bool:
    """Whether every split's profile tensor is already on disk."""
    complete = all(os.path.exists(profile_path(psbd_dir, split)) for split in splits)
    return complete


def write_run_provenance(psbd_dir: str, folder: str, use_bfloat16: bool) -> None:
    """The commit, head count and GPU behind this profile, written next to it."""
    payload = {
        "folder": folder,
        "heads": VIT_BLOCKS * VIT_HEADS_PER_BLOCK,
        "split_seed": PSBD_SPLIT_SEED,
        "git_commit": current_git_commit(),
        "bfloat16": use_bfloat16,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(os.path.join(psbd_dir, "run_head_profile.json"), "w") as handle:
        json.dump(payload, handle, indent=2)


def run_one_checkpoint(
    args: argparse.Namespace, folder: str, device: torch.device
) -> None:
    """Every split's 144-head profile for a checkpoint, over a single loaded model."""
    psbd_dir = os.path.join(args.results_dir, folder, "psbd")
    model, architecture, loaders, manifest, _metadata = load_model_and_loaders(
        args, folder, device
    )
    if architecture != "vit":
        raise ValueError(f"{folder}: head profiling is implemented for ViT only")
    if args.skip_existing and already_complete(psbd_dir, loaders):
        print(f"[skip] {folder} head_profile", flush=True)
        return

    write_split_manifest(psbd_dir, manifest)
    use_bfloat16 = not args.no_bfloat16
    for split, loader in loaders.items():
        # Reuses baseline_<split>.pt when the dropout sweep already wrote it, so
        # the no-ablation pass is not recomputed here.
        _, baseline_labels, _ = load_or_build_baseline(
            psbd_dir, split, model, loader, device, use_bfloat16
        )
        print(f"  {folder} / {split} ({len(baseline_labels)} samples)", flush=True)
        profile = profile_one_split(
            model, loader, baseline_labels, device, use_bfloat16
        )
        save_profile(profile_path(psbd_dir, split), profile)

    write_run_provenance(psbd_dir, folder, use_bfloat16)
    print(f"[ok] {folder} head_profile", flush=True)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for folder in args.checkpoint_folder:
        try:
            run_one_checkpoint(args, folder, device)
        except Exception as error:
            # A bad checkpoint must not cost the batch its remaining hours.
            print(f"[fail] {folder}: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
