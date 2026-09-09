"""Latent-space analysis of one trained checkpoint: TAC, backdoor direction, CKA, PCA.

Builds paired clean and triggered copies of the same test images, extracts per-layer
CLS features from both, and reports where in the network the trigger lives and how
separable the 2 representations are. A PCA scatter is saved at the layer whose
backdoor direction norm is largest.

Run it on a benign checkpoint and a backdoored one and compare. A benign model
should show small TAC at every layer, because it never learned the trigger; a
backdoored one should show TAC and the direction norm rising at the layer carrying
the backdoor.

All the work lives in analysis.latent. This file only parses arguments.

Example
    python -m cli.analyze_latent --dataset cifar100 --attack badnet_a2o \
        --checkpoint checkpoints/vit_cifar100_badnet_a2o_0_1/attack_result.pt
"""

import argparse

from analysis.latent import analyze_latent
from data.registry import DATASET_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latent-space analysis of a checkpoint"
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_REGISTRY), required=True)
    parser.add_argument(
        "--attack",
        required=True,
        help="the trigger to probe with, for example badnet_a2o",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architecture", choices=("vit", "swin"), default="vit")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--output", default="latent_scatter.png")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_latent(
        dataset=args.dataset,
        attack_name=args.attack,
        checkpoint=args.checkpoint,
        architecture=args.architecture,
        target_label=args.target_label,
        samples=args.samples,
        batch_size=args.batch_size,
        raw_data_dir=args.raw_data_dir,
        output=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
