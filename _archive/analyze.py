"""Render PSBD figures for every swept attack under an experiments root.

Run from inside this directory so the flat module imports resolve.

Example
    python analyze.py --experiments-root experiments/pre_residual --no-show
"""

import argparse

import matplotlib

from plotting import analyze_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PSBD figures")
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--quantile", type=float, default=0.25)
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening a window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    analyze_all(args.experiments_root, quantile=args.quantile, plot=True)


if __name__ == "__main__":
    main()
