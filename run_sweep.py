"""Run a PSBD dropout sweep over BackdoorBench attack folders.

Run from inside this directory so the flat module imports resolve.

Example
    python run_sweep.py --placement pre_residual --architecture vit \
        --attacks cifar10_badnet_0_1 cifar10_wanet_0_1
"""

import argparse

import torch

from config import ARCHITECTURES, DROPOUT_PLACEMENTS, RunConfig
from sweep import run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSBD dropout sweep")
    parser.add_argument("--placement", choices=DROPOUT_PLACEMENTS, default="pre_residual")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="vit")
    parser.add_argument("--attacks", nargs="+", required=True, help="BackdoorBench folder names")
    parser.add_argument("--weights-dir", default="backdoor_bench_checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-root", default="experiments")
    parser.add_argument("--forward-passes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--float32", action="store_true", help="Disable bfloat16 autocast")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        batch_size=args.batch_size,
        forward_passes=args.forward_passes,
        architecture=args.architecture,
        dropout_placement=args.placement,
        weights_dir=args.weights_dir,
        raw_data_dir=args.raw_data_dir,
        results_root=args.results_root,
        use_bfloat16=not args.float32,
        attack_folders=tuple(args.attacks),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, architecture: {config.architecture}, placement: {config.dropout_placement}")
    run_sweep(config, device)


if __name__ == "__main__":
    main()
