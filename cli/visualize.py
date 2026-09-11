"""Run 1 visualisation tool against 1 checkpoint and write its figure with a sidecar.

The tools mirror BackdoorBench's analysis module, re-implemented over this
project's loaders and hooks (docs/visualization/README.md). Every tool reads the
same loaded case: the checkpoint's model, the 3 PSBD splits (validation, paired
clean and backdoor) built by the standard permutation, its args.json metadata,
and the dataset's normalisation statistics. Output lands under
results/<folder>/visual/<tool>.{pdf,png,json,csv} with the git commit, layer,
view and sample count in the sidecar.

    python -m cli.visualize tac --folder vit_gtsrb_badnet_a2o_0_05
    python -m cli.visualize gradcam --folder vit_gtsrb_badnet_a2o_0_05 --view paired
    python -m cli.visualize --all-cheap --folder vit_gtsrb_benign
"""

import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.registry import DATASET_REGISTRY
from data.splits import (
    BENIGN_PROBE_ATTACK,
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from models.backbones import load_checkpoint
from visualization import cheap_tools, expensive_tools

VIEWS = ("clean_test", "bd_test", "mixed", "paired")
TOOLS = {**cheap_tools.TOOLS, **expensive_tools.TOOLS}


@dataclass(frozen=True)
class VisualCase:
    """Everything a tool may read: 1 checkpoint, its splits and its provenance."""

    folder: str
    metadata: dict
    model: nn.Module
    loaders: dict[str, DataLoader]
    manifest: dict
    device: torch.device
    mean: tuple[float, ...]
    std: tuple[float, ...]
    num_classes: int
    image_size: int
    architecture: str
    out_dir: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tool", nargs="?", choices=sorted(TOOLS), default=None)
    parser.add_argument(
        "--all-cheap", action="store_true", help="run every tool in cheap_tools"
    )
    parser.add_argument("--folder", required=True, help="checkpoint folder name")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--layer", type=int, default=None, help="block index, tool default when omitted"
    )
    parser.add_argument("--view", choices=VIEWS, default="paired")
    parser.add_argument(
        "--samples", type=int, default=2000, help="images per split the tool reads"
    )
    parser.add_argument(
        "--classes", type=int, default=10, help="classes drawn in a categorical figure"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-attack", default=None)
    parser.add_argument("--probe-target-label", type=int, default=None)
    parser.add_argument("--no-bfloat16", action="store_true")
    return parser


def resolve_probe(
    metadata: dict, args: argparse.Namespace
) -> tuple[str | None, int | None]:
    """A benign checkpoint is probed with the standard attack so it has a backdoor split."""
    probe = args.probe_attack
    if probe is None and metadata["attack"] == "benign":
        probe = BENIGN_PROBE_ATTACK
    target = args.probe_target_label if probe else None
    return probe, target


def load_case(args: argparse.Namespace, device: torch.device) -> VisualCase:
    """Load the checkpoint and build its PSBD splits, the same rows every detector saw."""
    checkpoint_path = os.path.join(
        args.checkpoints_dir, args.folder, "attack_result.pt"
    )
    metadata = read_checkpoint_metadata(checkpoint_path)
    probe, probe_target = resolve_probe(metadata, args)
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.samples,
        probe_attack=probe,
        probe_target_label=probe_target,
    )
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device)
    spec = DATASET_REGISTRY[metadata["dataset"]]
    out_dir = os.path.join(args.results_dir, args.folder, "visual")
    os.makedirs(out_dir, exist_ok=True)
    case = VisualCase(
        folder=args.folder,
        metadata=metadata,
        model=model,
        loaders=loaders,
        manifest=manifest,
        device=device,
        mean=spec.mean,
        std=spec.std,
        num_classes=spec.num_classes,
        image_size=spec.image_size,
        architecture=metadata["architecture"],
        out_dir=out_dir,
    )
    return case


def selected_tools(args: argparse.Namespace) -> list[str]:
    if args.all_cheap:
        return sorted(cheap_tools.TOOLS)
    if args.tool is None:
        raise SystemExit("name a tool or pass --all-cheap")
    return [args.tool]


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case = load_case(args, device)
    failures = 0
    for name in selected_tools(args):
        print(f"[{name}] {case.folder}", flush=True)
        try:
            TOOLS[name](case, args)
        except Exception as error:  # 1 broken tool must not hide the others' figures
            failures += 1
            print(f"[{name}] FAILED: {error!r}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
