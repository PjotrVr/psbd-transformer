"""PSBD dropout-position sweep for one (checkpoint, position-config), all 9 rates.

One invocation is the atomic unit of work: a single checkpoint, a single
position-config, sweeping the 9 dropout rates internally over one loaded model
and one baseline cache. The outer grid over checkpoints and position-configs is
flattened into separate PBS jobs (pbs/generate_psbd_jobs.py), so many single-GPU
jobs run concurrently rather than one long serial job.

Stage 1 only: this writes the raw per-pass probabilities, the per-pass argmax
classes, and the baseline to disk (defences.psbd_cache). Threshold, TPR, FPR,
AUROC, and the shift ratio are a separate cheap CPU step (psbd_analyze.py) that
reads those back, so this script never touches the GPU for anything but forward
passes.

A benign checkpoint has no attack of its own, so probing it needs --probe-attack
to name the trigger. That is the sweep's negative control and chance-level
detection is the expected result.
"""

import argparse
import json
import os
import subprocess

import torch

from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.dropout import (
    DROPOUT_CONFIGS,
    SINGLE_POSITION_NAMES,
    plug_dropout,
    unplug_dropout,
)
from defences.inference import compute_dropout_pass_probs
from defences.psbd_cache import (
    dropout_pass_path,
    load_or_build_baseline,
    save_dropout_pass_probs,
    write_split_manifest,
)
from models import detect_architecture, load_checkpoint

# The 9 dropout rates 0.1 to 0.9, the same grid the archived sweep used.
DROPOUT_RATES: tuple[float, ...] = tuple(i / 10.0 for i in range(1, 10))

# Reseeds the dropout mask sampling (see defences.inference), distinct from the
# data-split seed. Kept fixed so a rerun reproduces the same masks exactly.
PSBD_MASK_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", required=True)
    parser.add_argument(
        "--position-config",
        required=True,
        choices=tuple(DROPOUT_CONFIGS) + SINGLE_POSITION_NAMES,
    )
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--forward-passes", type=int, default=3)
    # A smoke and timing knob only. The real jobs leave it None for the full split.
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument(
        "--probe-attack",
        default=None,
        help="trigger to probe a benign checkpoint with (the negative control)",
    )
    parser.add_argument("--probe-target-label", type=int, default=None)
    parser.add_argument(
        "--rates",
        nargs="*",
        type=float,
        default=None,
        help=(
            "override the 0.1-to-0.9 grid. A residual-stream position masks the "
            "whole stream once per block, so (1-p)^12 survives the ViT stack and "
            "even p=0.1 is already saturating; reaching a comparable disturbance "
            "there needs rates an order of magnitude smaller."
        ),
    )
    return parser.parse_args()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def resolve_architecture(checkpoint_path: str, metadata: dict) -> str:
    """The architecture, cross-checked against the checkpoint's own weights.

    args.json's architecture field was backfilled from folder names by a
    migration, not recorded at training time, so it is inference rather than
    observation. detect_architecture reads the state_dict keys instead. Building
    the wrong architecture would now raise on the strict load anyway, but failing
    here names the actual problem instead of dumping a key mismatch.
    """
    recorded = metadata["architecture"]
    detected = detect_architecture(checkpoint_path)
    if recorded != detected:
        raise ValueError(
            f"{checkpoint_path}: args.json says architecture={recorded!r} but its "
            f"state_dict looks like {detected!r}"
        )
    return detected


def load_model_and_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, str, dict, dict]:
    """Side-effecting setup: read the checkpoint, build the model and the splits."""
    checkpoint_path = os.path.join(
        args.checkpoints_dir, args.checkpoint_folder, "attack_result.pt"
    )
    metadata = read_checkpoint_metadata(checkpoint_path)
    architecture = resolve_architecture(checkpoint_path, metadata)
    model = load_checkpoint(architecture, checkpoint_path, device)
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        probe_attack=args.probe_attack,
        probe_target_label=args.probe_target_label,
    )
    return model, architecture, loaders, manifest


def run_one_rate(
    model: torch.nn.Module,
    loaders: dict,
    baselines: dict,
    psbd_dir: str,
    position_config: str,
    rate: float,
    device: torch.device,
    forward_passes: int,
    use_bfloat16: bool,
) -> None:
    """Every split at one rate, with the position already plugged."""
    for split, loader in loaders.items():
        _, baseline_labels, _ = baselines[split]
        per_pass_probs, per_pass_argmax = compute_dropout_pass_probs(
            model,
            loader,
            baseline_labels,
            device,
            forward_passes,
            use_bfloat16,
            PSBD_MASK_SEED,
        )
        save_dropout_pass_probs(
            dropout_pass_path(psbd_dir, position_config, rate, split),
            per_pass_probs,
            per_pass_argmax,
        )


def sweep_rates(
    model: torch.nn.Module,
    architecture: str,
    position_config: str,
    loaders: dict,
    baselines: dict,
    psbd_dir: str,
    device: torch.device,
    forward_passes: int,
    use_bfloat16: bool,
    rates: tuple[float, ...] = DROPOUT_RATES,
) -> None:
    """For each rate: plug the position, run every split, save, unplug.

    unplug_dropout returns the model to exactly its loaded state, so no reset is
    needed between rates: nothing about the model was ever mutated. The finally
    is not decoration. Without it, a failure mid-rate leaves that rate's dropouts
    attached, and the next rate stacks on top of them for an effective rate of
    1 - (1-p1)(1-p2). Every file would still be written and every number would be
    quietly wrong.
    """
    position_names = DROPOUT_CONFIGS.get(position_config, (position_config,))
    for rate in rates:
        handles = plug_dropout(model, architecture, position_names, {}, rate)
        try:
            run_one_rate(
                model,
                loaders,
                baselines,
                psbd_dir,
                position_config,
                rate,
                device,
                forward_passes,
                use_bfloat16,
            )
        finally:
            unplug_dropout(handles)


def write_run_provenance(psbd_dir: str, args: argparse.Namespace, device) -> None:
    """The commit, config, and GPU behind this cache, next to the cache itself.

    bfloat16 logits have an 8-bit mantissa, so a near-tie can put the baseline
    argmax on a different class on a different GPU model. Recording which device
    produced a cache makes that reproducible rather than mysterious.
    """
    payload = {
        "git_commit": current_git_commit(),
        "position_config": args.position_config,
        "dropout_rates": list(args.rates) if args.rates else list(DROPOUT_RATES),
        "forward_passes": args.forward_passes,
        "mask_seed": PSBD_MASK_SEED,
        "split_seed": PSBD_SPLIT_SEED,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "use_bfloat16": not args.no_bfloat16,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
    }
    path = os.path.join(psbd_dir, f"run_{args.position_config}.json")
    os.makedirs(psbd_dir, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    device = resolve_device()
    use_bfloat16 = not args.no_bfloat16

    model, architecture, loaders, manifest = load_model_and_loaders(args, device)

    psbd_dir = os.path.join(args.results_dir, args.checkpoint_folder, "psbd")
    write_split_manifest(psbd_dir, manifest)
    write_run_provenance(psbd_dir, args, device)
    baselines = {
        split: load_or_build_baseline(
            psbd_dir, split, model, loader, device, use_bfloat16
        )
        for split, loader in loaders.items()
    }

    sweep_rates(
        model,
        architecture,
        args.position_config,
        loaders,
        baselines,
        psbd_dir,
        device,
        args.forward_passes,
        use_bfloat16,
        rates=tuple(args.rates) if args.rates else DROPOUT_RATES,
    )


if __name__ == "__main__":
    main()
