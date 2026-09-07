"""Score the baseline detectors on exactly the splits PSBD is scored on.

Writes results/<folder>/baseline_metrics.json, a sibling of psbd_metrics.json, so a
comparison table reads both without either method's numbers being recomputed under
different conditions.

Fairness is the whole point of this script, so the shared parts are shared literally:
the same build_psbd_loaders_from_checkpoint, the same clean-validation split for
thresholding, the same pair_clean_to_backdoor subsetting, the same quantile, and the
same detection_report. The only thing that differs between methods is the score.

Every detector here returns LOW for poisoned, matching PSU, so nothing downstream
special-cases them.

Example
    python -m cli.baselines --checkpoint-folder vit_cifar10_badnet_a2o_0_1
    python -m cli.baselines --all
"""

import argparse
import json
import os

import torch

from psbd.config import DATASET_REGISTRY
from psbd.detectors import DETECTOR_NAMES, DetectorContext, build_detector
from psbd.decision import (
    HEADLINE_QUANTILE,
    PSBD_QUANTILES,
    detection_report,
    pair_clean_to_backdoor,
)
from psbd.models import load_checkpoint
from psbd.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)

STRIP_OVERLAYS = 8

# The trigger a benign checkpoint is probed with when the caller names none, so
# its baseline numbers sit on the same footing as its PSBD numbers.
BENIGN_PROBE_ATTACK = "badnet_a2o"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument("--probe-attack", default=None)
    parser.add_argument("--probe-target-label", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip folders whose baseline_metrics.json is already written",
    )
    return parser.parse_args()


def discover_folders(results_dir: str) -> list[str]:
    """Folders that already have a PSBD cache, so the comparison is like for like."""
    if not os.path.isdir(results_dir):
        return []

    folders = sorted(
        name
        for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name, "psbd"))
    )
    return folders


def score_checkpoint(
    folder: str, args: argparse.Namespace, device: torch.device
) -> dict:
    """Both baseline detectors on one checkpoint, reported at every PSBD quantile."""
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)

    probe = args.probe_attack
    if probe is None and metadata["attack"] == "benign":
        probe = BENIGN_PROBE_ATTACK

    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        probe_attack=probe,
        probe_target_label=args.probe_target_label if probe else None,
    )
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device)
    use_bfloat16 = not args.no_bfloat16

    # Every detector is built through psbd.detectors.build_detector rather than
    # wired by hand here. That registry already resolves the 2 things this file
    # got wrong when it did wire them by hand: SCALE-UP's per-class statistics are
    # fitted on TRUE labels, which is what Eq. (3) defines them by, not on the
    # model's predictions; and the data-free and data-limited variants are 2
    # separate entries rather than one entry silently carrying the data-limited
    # numbers under the data-free name.
    #
    # It also owns the sign convention. Every builder returns low-is-poisoned,
    # which is what detection_report assumes, so nothing here negates anything.
    spec = DATASET_REGISTRY[metadata["dataset"]]
    context = DetectorContext(
        model=model,
        device=device,
        mean=spec.mean,
        std=spec.std,
        validation_loader=loaders["validation"],
        num_classes=spec.num_classes,
        use_bfloat16=use_bfloat16,
        seed=PSBD_SPLIT_SEED,
    )

    # Fitting runs forward passes over the validation split, so it happens once
    # per model here rather than once per split scored.
    score_functions = {name: build_detector(name, context) for name in DETECTOR_NAMES}

    detectors = {}
    for name, score in score_functions.items():
        scores = {
            split: score(model, loader, device) for split, loader in loaders.items()
        }
        paired_clean = pair_clean_to_backdoor(scores["clean"], manifest)
        detectors[name] = {
            f"q{quantile:.2f}": detection_report(
                scores["validation"], paired_clean, scores["backdoor"], quantile
            )
            for quantile in PSBD_QUANTILES
        }

    report = {
        "folder_name": folder,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "poison_rate": metadata.get("poison_rate"),
        "architecture": metadata["architecture"],
        "probe_attack": manifest.get("probe_attack"),
        "headline_quantile": HEADLINE_QUANTILE,
        "strip_overlays": STRIP_OVERLAYS,
        "detectors": detectors,
    }
    return report


def save_report(results_dir: str, folder: str, report: dict) -> None:
    """Write one checkpoint's baseline record beside its PSBD one."""
    path = os.path.join(results_dir, folder, "baseline_metrics.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folders = discover_folders(args.results_dir) if args.all else args.checkpoint_folder
    if not folders:
        raise SystemExit("nothing to score: pass --checkpoint-folder or --all")

    headline = f"q{HEADLINE_QUANTILE:.2f}"
    for folder in folders:
        written = os.path.join(args.results_dir, folder, "baseline_metrics.json")
        if args.skip_existing and os.path.exists(written):
            print(f"[skip] {folder}", flush=True)
            continue

        try:
            report = score_checkpoint(folder, args, device)
        except Exception as error:
            print(f"FAILED {folder}: {type(error).__name__}: {error}")
            continue

        save_report(args.results_dir, folder, report)
        summary = "  ".join(
            f"{name} auroc={block[headline]['auroc']:.3f}"
            for name, block in report["detectors"].items()
        )
        print(f"{folder}  {summary}")


if __name__ == "__main__":
    main()
