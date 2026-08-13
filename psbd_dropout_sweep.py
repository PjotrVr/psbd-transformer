"""PSBD dropout-position sweep for one (checkpoint, position-config), all 9 rates.

One invocation is the atomic unit of work: a single checkpoint, a single
position-config, sweeping the 9 dropout rates internally over one loaded model
and one baseline cache. The outer grid over 6 checkpoints and 10 position-configs
is flattened into separate PBS jobs (scratch/generate_psbd_pbs_jobs.py), so 60
single-GPU jobs can run concurrently rather than one long serial job.

Stage 1 only: this writes the raw per-pass probabilities and the baseline to disk
(defences.psbd_cache). Threshold, TPR, FPR, and AUROC are a separate cheap CPU
step (defences.detection) that reads those back, so this script never touches the
GPU for anything but the forward passes.
"""

import argparse
import os

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
from models import load_checkpoint

# The 9 dropout rates 0.1 to 0.9, the same grid the archived sweep used.
DROPOUT_RATES: tuple[float, ...] = tuple(i / 10.0 for i in range(1, 10))

# Reseeds the dropout mask sampling (see defences.inference), distinct from the
# data-split seed. Kept fixed so masks are reproducible and paired across splits.
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
    return parser.parse_args()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, str, dict, dict]:
    """Side-effecting setup: read the checkpoint, build the model and the splits."""
    checkpoint_path = os.path.join(
        args.checkpoints_dir, args.checkpoint_folder, "attack_result.pt"
    )
    architecture = read_checkpoint_metadata(checkpoint_path)["architecture"]
    model = load_checkpoint(architecture, checkpoint_path, device)
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    return model, architecture, loaders, manifest


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
) -> None:
    """For each rate: plug the position, run every split, save, unplug.

    unplug_dropout returns the model to exactly its loaded state, so no reset is
    needed between rates: nothing about the model was ever mutated.
    """
    position_names = DROPOUT_CONFIGS.get(position_config, (position_config,))
    for rate in DROPOUT_RATES:
        handles = plug_dropout(model, architecture, position_names, {}, rate)
        for split, loader in loaders.items():
            _, baseline_labels = baselines[split]
            per_pass_probs = compute_dropout_pass_probs(
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
            )
        unplug_dropout(handles)


def main() -> None:
    args = parse_args()
    device = resolve_device()
    use_bfloat16 = not args.no_bfloat16

    model, architecture, loaders, manifest = load_model_and_loaders(args, device)

    psbd_dir = os.path.join(args.results_dir, args.checkpoint_folder, "psbd")
    write_split_manifest(psbd_dir, manifest)
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
    )


if __name__ == "__main__":
    main()
