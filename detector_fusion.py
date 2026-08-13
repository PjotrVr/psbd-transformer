"""Fuse PSBD and STRIP, whose failures are disjoint.

The comparison table shows the two detectors failing on opposite attacks: STRIP
reaches 0.97 to 1.00 at 1% FPR on the static patch trigger and exactly 0.000 on
adaptive_blend and badnet_a2a, while PSBD is the reverse. No checkpoint in the grid
defeats both. That is the textbook case for combining them.

Fusion is by **rank**, not by score. The two scores are on incompatible scales (a
probability drop against an entropy in nats) and have different distributions, so any
weighted sum would be dominated by whichever happens to have the larger spread.
Converting each to its within-split rank first makes them commensurable without
fitting anything.

Two rules, both requiring no poisoned data:

  mean_rank   average of the two normalized ranks. Balanced, and the natural choice
              when neither detector is known to be the reliable one in advance.
  min_rank    the more suspicious of the two verdicts. This is the right rule if the
              failures really are disjoint: a sample only escapes when BOTH detectors
              consider it clean.

The threshold is still the quantile of clean-validation fused rank, so the defender
never touches poisoned data and the false-positive budget is set exactly as before.

Example
    PYTHONPATH=. python detector_fusion.py --fpr 0.01 0.05
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from defences.baselines import collect_overlay_batch, strip_scores
from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.psbd_cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)
from defences.psbd_metrics import (
    complete_rates,
    pair_clean_to_backdoor,
    psu_ratio_from_cache,
    shift_ratio,
)
from models import load_checkpoint

STRIP_OVERLAYS = 8


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


def to_rank(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Each score as its percentile within a shared reference distribution.

    The reference must be the SAME set for every split, and clean validation is the
    natural choice: it is the only distribution the defender is assumed to hold, and
    it is what the threshold is already drawn from.

    Ranking each split against itself instead is the obvious mistake and it silently
    destroys the method. Within-split ranks span [0, 1] for every split by
    construction, so a threshold at the 1st percentile of validation rank flags
    exactly the bottom 1% of the backdoor split no matter how extreme its scores are,
    pinning TPR to the false-positive rate. That produced TPR 0.010 at 1% FPR on
    every checkpoint, including ones where a component detector scored 1.000.

    Percentile against a common reference keeps the two detectors commensurable
    without fitting anything, and leaves an actually-extreme score extreme.
    """
    sorted_reference = reference.sort().values
    positions = torch.searchsorted(sorted_reference, values.contiguous())
    return positions.float() / max(len(sorted_reference), 1)


def psbd_scores(psbd_dir: str, placement: str, shift_target: float):
    """Fractional PSU at the adaptive rate, per split. None if unavailable."""
    folder = os.path.join(psbd_dir, placement)
    if not os.path.isdir(folder):
        return None
    rates = complete_rates(psbd_dir, placement)
    for rate in rates:
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "validation")
        )
        sigma = shift_ratio(labels, argmax)
        if sigma is None or sigma < shift_target:
            continue
        out = {}
        for split in ("validation", "clean", "backdoor"):
            probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
            per_pass, _ = load_dropout_pass_probs(
                dropout_pass_path(psbd_dir, placement, rate, split)
            )
            out[split] = psu_ratio_from_cache(probs, labels, per_pass)
        return out, rate
    return None


def tpr_at_fpr(validation, clean, backdoor, target: float) -> tuple[float, float]:
    """TPR at a threshold set from clean validation, plus the FPR it achieves."""
    threshold = float(np.quantile(validation.numpy(), target))
    return (
        float((backdoor < threshold).float().mean()),
        float((clean < threshold).float().mean()),
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("PSBD and STRIP fused by rank. Threshold from clean validation only.\n")
    header = f"{'attack':16} {'pr':>5}"
    for target in args.fpr:
        header += f" | {f'PSBD@{target:.0%}':>9} {f'STRIP@{target:.0%}':>10} {f'MEAN@{target:.0%}':>9} {f'MIN@{target:.0%}':>9}"
    print(header)
    print("-" * len(header))

    totals = {
        name: {t: [] for t in args.fpr} for name in ("psbd", "strip", "mean", "min")
    }

    for path in sorted(glob.glob(os.path.join(args.results_dir, "*", "psbd"))):
        folder = os.path.basename(os.path.dirname(path))
        if "sam_rho" in folder or not folder.startswith("vit_cifar10"):
            continue
        psbd = psbd_scores(path, args.placement, args.shift_target)
        if psbd is None:
            continue
        psu, rate = psbd

        metadata = read_checkpoint_metadata(
            os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
        )
        probe = "badnet_a2o" if metadata["attack"] == "benign" else None
        loaders, manifest = build_psbd_loaders_from_checkpoint(
            os.path.join(args.checkpoints_dir, folder, "attack_result.pt"),
            seed=PSBD_SPLIT_SEED,
            raw_data_dir=args.raw_data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            probe_attack=probe,
            probe_target_label=0 if probe else None,
        )
        model = load_checkpoint(
            metadata["architecture"],
            os.path.join(args.checkpoints_dir, folder, "attack_result.pt"),
            device,
        )
        overlays = collect_overlay_batch(
            loaders["validation"], STRIP_OVERLAYS, PSBD_SPLIT_SEED
        )
        strip = {
            split: strip_scores(
                model, loader, overlays, device, True, PSBD_SPLIT_SEED, STRIP_OVERLAYS
            )
            for split, loader in loaders.items()
        }

        # Both detectors become percentiles of the SAME reference, the clean
        # validation split, so a fused score means the same thing in every split and
        # the validation threshold transfers. The clean side is paired down to the
        # backdoor split's images afterwards, so all four columns are compared on one
        # population.
        fused = {}
        for split in ("validation", "clean", "backdoor"):
            a = to_rank(psu[split], psu["validation"])
            b = to_rank(strip[split], strip["validation"])
            fused[split] = {
                "psbd": psu[split],
                "strip": strip[split],
                "mean": (a + b) / 2,
                "min": torch.minimum(a, b),
            }

        meta = json.load(
            open(os.path.join(args.checkpoints_dir, folder, "metrics.json"))
        )
        line = f"{metadata['attack']:16} {meta.get('poison_rate') or 0:>5.3f}"
        for target in args.fpr:
            for name in ("psbd", "strip", "mean", "min"):
                clean = pair_clean_to_backdoor(fused["clean"][name], manifest)
                tpr, _ = tpr_at_fpr(
                    fused["validation"][name], clean, fused["backdoor"][name], target
                )
                line += f" {tpr:>9.3f}" if name != "psbd" else f" | {tpr:>9.3f}"
                if metadata["attack"] != "benign":
                    totals[name][target].append(tpr)
        print(line, flush=True)

    print("\nmean TPR over backdoored checkpoints:")
    for name in ("psbd", "strip", "mean", "min"):
        cells = [
            f"{t:.0%}: {sum(v) / len(v):.3f}"
            if (v := totals[name][t])
            else f"{t:.0%}: --"
            for t in args.fpr
        ]
        print(f"  {name:6} " + "   ".join(cells))


if __name__ == "__main__":
    main()
