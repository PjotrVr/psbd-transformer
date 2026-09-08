"""Fuse PSBD and STRIP, whose failures are disjoint.

The comparison table shows the two detectors failing on opposite attacks: STRIP
reaches 0.97 to 1.00 at 1% FPR on the static patch trigger and exactly 0.000 on
adaptive_blend and badnet_a2a, while PSBD is the reverse. No checkpoint in the grid
defeats both. That is the textbook case for combining them.

Fusion is by **rank**, not by score. The two scores are on incompatible scales (a
probability drop against an entropy in nats) and have different distributions, so any
weighted sum would be dominated by whichever happens to have the larger spread.
Converting each to its rank within a shared reference first makes them commensurable
without fitting anything.

The reference must be the SAME set for every split, and clean validation is the
natural choice: it is the only distribution the defender is assumed to hold, and it
is what the threshold is already drawn from. Ranking each split against itself
instead is the obvious mistake and it silently destroys the method. Within-split
ranks span [0, 1] for every split by construction, so a threshold at the 1st
percentile of validation rank flags exactly the bottom 1% of the backdoor split no
matter how extreme its scores are, pinning TPR to the false-positive rate. That
produced TPR 0.010 at 1% FPR on every checkpoint, including ones where a component
detector scored 1.000. This is the explanation psbd.scores.to_rank points at.

Two rules, both requiring no poisoned data:

  mean_rank   average of the two normalized ranks. Balanced, and the natural choice
              when neither detector is known to be the reliable one in advance.
  min_rank    the more suspicious of the two verdicts. This is the right rule if the
              failures really are disjoint: a sample only escapes when BOTH detectors
              consider it clean.

The threshold is still the quantile of clean-validation fused rank, so the defender
never touches poisoned data and the false-positive budget is set exactly as before.

Example
    python -m cli.fuse_detectors --fpr 0.01 0.05
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from psbd.baselines import collect_overlay_batch, strip_scores
from psbd.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)
from psbd.decision import complete_rates, pair_clean_to_backdoor
from psbd.models import load_checkpoint
from psbd.scores import psu_ratio_from_cache, shift_ratio, to_rank
from psbd.config import DATASET_REGISTRY
from psbd.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)

STRIP_OVERLAYS = 8

SPLITS = ("validation", "clean", "backdoor")
COLUMNS = ("psbd", "strip", "mean", "min")

# This table was built on the CIFAR-10 ViT grid, where the disjoint-failure pattern
# was measured. Widening it means re-reading that pattern first, so the restriction
# is explicit rather than left to whatever happens to be on disk.
FOLDER_PREFIX = "vit_cifar10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--fpr", nargs="*", type=float, default=[0.01, 0.05])
    parser.add_argument("--shift-target", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--placement", default="pre_residual_blocks_5_8")
    return parser.parse_args()


def psbd_scores(
    psbd_dir: str, placement: str, shift_target: float
) -> tuple[dict, float] | None:
    """Fractional PSU at the adaptive rate, per split. None if unavailable."""
    if not os.path.isdir(os.path.join(psbd_dir, placement)):
        return None

    for rate in complete_rates(psbd_dir, placement):
        _, labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "validation")
        )
        sigma = shift_ratio(labels, argmax)
        if sigma is None or sigma < shift_target:
            continue

        scores = {}
        for split in SPLITS:
            probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
            per_pass, _ = load_dropout_pass_probs(
                dropout_pass_path(psbd_dir, placement, rate, split)
            )
            scores[split] = psu_ratio_from_cache(probs, labels, per_pass)
        return scores, rate

    return None


def strip_scores_per_split(
    checkpoint_path: str,
    metadata: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, dict]:
    """STRIP entropy per split, plus the split manifest the loaders were built from."""
    probe = "badnet_a2o" if metadata["attack"] == "benign" else None
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        probe_attack=probe,
        probe_target_label=0 if probe else None,
    )
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device)
    overlays = collect_overlay_batch(
        loaders["validation"], STRIP_OVERLAYS, PSBD_SPLIT_SEED
    )
    spec = DATASET_REGISTRY[metadata["dataset"]]

    scores = {
        split: strip_scores(
            model,
            loader,
            overlays,
            spec.mean,
            spec.std,
            device,
            True,
            PSBD_SPLIT_SEED,
            STRIP_OVERLAYS,
        )
        for split, loader in loaders.items()
    }
    return scores, manifest


def fuse(psu: dict, strip: dict) -> dict:
    """The 4 comparable columns per split: both components and both fusion rules.

    Both detectors become percentiles of the SAME reference, the clean validation
    split, so a fused score means the same thing in every split and the validation
    threshold transfers.
    """
    fused = {}
    for split in SPLITS:
        psu_rank = to_rank(psu[split], psu["validation"])
        strip_rank = to_rank(strip[split], strip["validation"])
        fused[split] = {
            "psbd": psu[split],
            "strip": strip[split],
            "mean": (psu_rank + strip_rank) / 2,
            "min": torch.minimum(psu_rank, strip_rank),
        }
    return fused


def tpr_at_fpr(
    validation: torch.Tensor,
    clean: torch.Tensor,
    backdoor: torch.Tensor,
    target: float,
) -> tuple[float, float]:
    """TPR at a threshold set from clean validation, plus the FPR it achieves."""
    threshold = float(np.quantile(validation.numpy(), target))
    return (
        float((backdoor < threshold).float().mean()),
        float((clean < threshold).float().mean()),
    )


def build_header(target_fprs: list[float]) -> str:
    """The fixed-width header, one column quadruple per target FPR."""
    header = f"{'attack':16} {'pr':>5}"
    for target in target_fprs:
        header += (
            f" | {f'PSBD@{target:.0%}':>9} {f'STRIP@{target:.0%}':>10} "
            f"{f'MEAN@{target:.0%}':>9} {f'MIN@{target:.0%}':>9}"
        )
    return header


def discover_folders(results_dir: str) -> list[str]:
    """The CIFAR-10 ViT folders with a stage-1 cache, excluding SAM runs."""
    folders = sorted(
        os.path.basename(os.path.dirname(path))
        for path in glob.glob(os.path.join(results_dir, "*", "psbd"))
    )
    selected = [
        folder
        for folder in folders
        if "sam_rho" not in folder and folder.startswith(FOLDER_PREFIX)
    ]
    return selected


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("PSBD and STRIP fused by rank. Threshold from clean validation only.\n")
    header = build_header(args.fpr)
    print(header)
    print("-" * len(header))

    totals = {name: {target: [] for target in args.fpr} for name in COLUMNS}

    for folder in discover_folders(args.results_dir):
        psbd_dir = os.path.join(args.results_dir, folder, "psbd")
        scored = psbd_scores(psbd_dir, args.placement, args.shift_target)
        if scored is None:
            continue
        psu, _rate = scored

        checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
        metadata = read_checkpoint_metadata(checkpoint_path)
        strip, manifest = strip_scores_per_split(
            checkpoint_path, metadata, args, device
        )
        fused = fuse(psu, strip)

        meta = json.load(
            open(os.path.join(args.checkpoints_dir, folder, "metrics.json"))
        )
        line = f"{metadata['attack']:16} {meta.get('poison_rate') or 0:>5.3f}"
        for target in args.fpr:
            for name in COLUMNS:
                # The clean side is paired down to the backdoor split's images, so
                # all 4 columns are compared on one population.
                clean = pair_clean_to_backdoor(fused["clean"][name], manifest)
                tpr, _fpr = tpr_at_fpr(
                    fused["validation"][name], clean, fused["backdoor"][name], target
                )
                line += f" {tpr:>9.3f}" if name != "psbd" else f" | {tpr:>9.3f}"
                if metadata["attack"] != "benign":
                    totals[name][target].append(tpr)
        print(line, flush=True)

    print("\nmean TPR over backdoored checkpoints:")
    for name in COLUMNS:
        cells = []
        for target in args.fpr:
            values = totals[name][target]
            cells.append(
                f"{target:.0%}: {sum(values) / len(values):.3f}"
                if values
                else f"{target:.0%}: --"
            )
        print(f"  {name:6} " + "   ".join(cells))


if __name__ == "__main__":
    main()
