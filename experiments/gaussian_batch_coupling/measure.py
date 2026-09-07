"""Does GaussianNoise's batch-wide std put the three splits at different noise levels?

GaussianNoise scaled its noise by x.std() reduced over batch, tokens and channels
together, so a sample's perturbation magnitude depended on its batch neighbours.
The validation, clean and backdoor splits hold different image populations, so
that coupling could put the three at three different noise levels while every
comparison between them assumes one.

This measures the size of that effect directly, with no perturbation applied. For
each split it records, at each position of interest, the batch-wide std the old
code would have used and the per-sample std the fixed code uses. If the split
means agree to within about 1 percent the confound is negligible and the cached
gaussian results stand. If they do not, the reported gaussian cells need a
re-sweep.

The sign matters as much as the size. Triggered images plausibly carry a larger
activation std, which would have given the backdoor split MORE noise, pushed
backdoor PSU up, and dragged AUROC down. That direction would make the published
gaussian numbers conservative rather than inflated.
"""

import argparse
import json
import os

import torch

from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.dropout import POSITION_REGISTRY, _resolve_targets, BLOCK_TYPES
from models import detect_architecture, load_checkpoint, network_core

POSITIONS = ("before_mlp", "before_attention_norm")
MAX_BATCHES = 40


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", required=True, nargs="+")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="results/gaussian_batch_coupling.json")
    return parser.parse_args()


def collect_stds(model, loader, targets, device, max_batches):
    """Batch-wide and per-sample activation std at one position, over one split.

    Returns (batch_stds, per_sample_stds). The first is what the old operator
    multiplied its noise by, one value per batch. The second is what the fixed
    operator uses, one value per sample.
    """
    batch_stds = []
    per_sample_stds = []

    def hook(_module, args):
        x = args[0]
        batch_stds.append(float(x.detach().float().std()))
        non_batch_axes = tuple(range(1, x.dim()))
        per_sample_stds.extend(x.detach().float().std(dim=non_batch_axes).tolist())

    handles = [target.register_forward_pre_hook(hook) for target in targets]
    try:
        with torch.inference_mode():
            for index, (images, _) in enumerate(loader):
                if index >= max_batches:
                    break
                model(images.to(device))
    finally:
        for handle in handles:
            handle.remove()

    return batch_stds, per_sample_stds


def mean(values):
    average = sum(values) / len(values) if values else float("nan")
    return average


def measure_one(folder, args, device):
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    architecture = detect_architecture(checkpoint_path)
    model = load_checkpoint(architecture, checkpoint_path, device)
    model.eval()

    loaders, _ = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=2,
    )

    core = network_core(model)
    report = {"folder": folder, "attack": metadata["attack"], "positions": {}}

    for position in POSITIONS:
        spec = POSITION_REGISTRY[architecture][position]
        targets = _resolve_targets(model, core, spec, BLOCK_TYPES[architecture])
        per_split = {}
        for split, loader in loaders.items():
            batch_stds, sample_stds = collect_stds(
                model, loader, targets, device, MAX_BATCHES
            )
            per_split[split] = {
                "batch_std_mean": mean(batch_stds),
                "sample_std_mean": mean(sample_stds),
                "n_batches": len(batch_stds),
                "n_samples": len(sample_stds),
            }

        clean_batch = per_split["clean"]["batch_std_mean"]
        report["positions"][position] = {
            "splits": per_split,
            # The quantity that decides whether a re-sweep is needed: how far the
            # backdoor and validation splits sat from the clean split's noise level.
            "backdoor_vs_clean_pct": 100.0
            * (per_split["backdoor"]["batch_std_mean"] - clean_batch)
            / clean_batch,
            "validation_vs_clean_pct": 100.0
            * (per_split["validation"]["batch_std_mean"] - clean_batch)
            / clean_batch,
        }

    del model
    torch.cuda.empty_cache()
    return report


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reports = [measure_one(folder, args, device) for folder in args.checkpoint_folder]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(reports, handle, indent=2)

    print(
        f"\n{'checkpoint':<32} {'position':<24} {'bd vs clean':>12} {'val vs clean':>13}"
    )
    for report in reports:
        for position, block in report["positions"].items():
            print(
                f"{report['folder']:<32} {position:<24} "
                f"{block['backdoor_vs_clean_pct']:>11.3f}% "
                f"{block['validation_vs_clean_pct']:>12.3f}%"
            )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
