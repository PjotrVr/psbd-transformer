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
    python baseline_detect.py --checkpoint-folder vit_cifar10_badnet_a2o_0_1
    python baseline_detect.py --all
"""

import argparse
import json
import os

import torch

from defences.baselines import collect_overlay_batch, confidence_scores, strip_scores
from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.psbd_metrics import (
    HEADLINE_QUANTILE,
    PSBD_QUANTILES,
    detection_report,
    pair_clean_to_backdoor,
)
from models import load_checkpoint

STRIP_OVERLAYS = 8


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
    return parser.parse_args()


def discover_folders(results_dir: str) -> list[str]:
    """Folders that already have a PSBD cache, so the comparison is like for like."""
    if not os.path.isdir(results_dir):
        return []
    return sorted(
        name
        for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name, "psbd"))
    )


def score_checkpoint(folder: str, args: argparse.Namespace, device) -> dict:
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    probe = args.probe_attack
    if probe is None and metadata["attack"] == "benign":
        # The benign control is probed with the same trigger the sweep used, so its
        # baseline numbers sit on the same footing as its PSBD numbers.
        probe = "badnet_a2o"
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

    # STRIP's overlays come from the clean validation split, the same data PSBD is
    # allowed for thresholding, so neither method sees more than the other.
    overlays = collect_overlay_batch(
        loaders["validation"], STRIP_OVERLAYS, PSBD_SPLIT_SEED
    )

    detectors = {}
    for name, build in (
        (
            "confidence",
            lambda loader: confidence_scores(model, loader, device, use_bfloat16),
        ),
        (
            "strip",
            lambda loader: strip_scores(
                model,
                loader,
                overlays,
                device,
                use_bfloat16,
                PSBD_SPLIT_SEED,
                STRIP_OVERLAYS,
            ),
        ),
    ):
        scores = {split: build(loader) for split, loader in loaders.items()}
        paired_clean = pair_clean_to_backdoor(scores["clean"], manifest)
        detectors[name] = {
            f"q{quantile:.2f}": detection_report(
                scores["validation"], paired_clean, scores["backdoor"], quantile
            )
            for quantile in PSBD_QUANTILES
        }

    return {
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


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folders = discover_folders(args.results_dir) if args.all else args.checkpoint_folder
    if not folders:
        raise SystemExit("nothing to score: pass --checkpoint-folder or --all")

    for folder in folders:
        try:
            report = score_checkpoint(folder, args, device)
        except Exception as error:
            print(f"FAILED {folder}: {type(error).__name__}: {error}")
            continue
        path = os.path.join(args.results_dir, folder, "baseline_metrics.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2)
        headline = f"q{HEADLINE_QUANTILE:.2f}"
        summary = "  ".join(
            f"{name} auroc={block[headline]['auroc']:.3f}"
            for name, block in report["detectors"].items()
        )
        print(f"{folder}  {summary}")


if __name__ == "__main__":
    main()
