"""Fuse PSBD with a recorded competitor detector by rank, where their failures are disjoint.

The first comparison table showed PSBD and STRIP failing on opposite attacks:
STRIP is near perfect on a static patch trigger and fails outright on
adaptive_blend and badnet_a2a, while PSBD is the reverse. No checkpoint in that
grid defeated both, which is the textbook case for combining them. This command
reads both detectors from disk, the PSBD stage-1 cache under
results/<folder>/psbd/ and the competitor's record under
results/<folder>/detectors/ written by cli.baselines, and never runs a model,
so the fusion is a CPU read over exactly the rows both were scored on.

Fusion is by rank, not by score. The 2 scores are on incompatible scales, a
probability drop against an entropy in nats, so any weighted sum would be
dominated by whichever has the larger spread. Converting each to its rank within
a shared reference makes them commensurable without fitting anything. The
reference is the clean validation split, the only distribution the defender
holds and the one the threshold is drawn from. Ranking each split against itself
would destroy the method, since within-split ranks span [0, 1] for every split by
construction and a threshold at the 1st percentile of validation rank would flag
exactly the bottom 1% of the backdoor split whatever its scores are.

2 rules, both needing no poisoned data:

  mean_rank   average of the 2 normalized ranks. Balanced, the natural choice
              when neither detector is known to be reliable in advance.
  min_rank    the more suspicious of the 2 verdicts. The right rule if the
              failures really are disjoint, since a sample only escapes when both
              detectors consider it clean.

The threshold is the quantile of clean-validation fused rank, so the defender
never touches poisoned data and the false-positive budget is set exactly as
before.

    python -m cli.fuse_detectors --detector strip --fpr 0.01 0.05 0.25
"""

import argparse
import glob
import os
import statistics

import numpy as np
import torch

from data.splits import SPLITS, read_checkpoint_metadata
from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import (
    ADAPTIVE_SHIFT_TARGET,
    RECOMMENDED_PLACEMENT,
    complete_rates,
    pair_clean_to_backdoor,
)
from defences.scores import psu_ratio_from_cache, shift_ratio, to_rank
from detectors import DETECTOR_NAMES
from detectors.records import (
    STATUS_SCORED,
    load_report,
    load_scores,
    report_path,
    scores_path,
)

COLUMNS = ("psbd", "detector", "mean", "min")
DEFAULT_FPRS = (0.01, 0.05, 0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--detector", choices=DETECTOR_NAMES, default="strip")
    parser.add_argument("--placement", default=RECOMMENDED_PLACEMENT)
    parser.add_argument("--shift-target", type=float, default=ADAPTIVE_SHIFT_TARGET)
    parser.add_argument("--architecture", default="vit")
    parser.add_argument("--dataset", nargs="*", default=None)
    parser.add_argument("--fpr", type=float, nargs="+", default=list(DEFAULT_FPRS))
    parser.add_argument("--markdown", default=None, help="also write the table here")
    return parser.parse_args()


def psbd_scores(
    psbd_dir: str, placement: str, shift_target: float
) -> tuple[dict[str, torch.Tensor], float] | None:
    """Fractional PSU per split at the adaptive rate, or None when the cache lacks it.

    The adaptive rule is the smallest cached rate whose clean-validation shift
    ratio reaches the target, the same rule cli.analyze applies.
    """
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
            scores[split] = psu_ratio_from_cache(probs, labels, per_pass)  # (n_split,)
        return scores, rate

    return None


def detector_scores(
    results_dir: str, folder: str, name: str
) -> dict[str, torch.Tensor] | None:
    """The recorded per-split scores of 1 detector, or None without a scored record."""
    report = load_report(report_path(results_dir, folder, name))
    if report is None or report.get("status") != STATUS_SCORED:
        return None
    scores = {
        split: load_scores(scores_path(results_dir, folder, name, split))
        for split in SPLITS
    }
    return scores


def fuse(psu: dict, other: dict) -> dict[str, dict[str, torch.Tensor]]:
    """The 4 comparable columns per split: both components and both fusion rules.

    Both detectors become percentiles of the same reference, the clean validation
    split, so a fused score means the same thing in every split and the validation
    threshold transfers.
    """
    fused = {}
    for split in SPLITS:
        psu_rank = to_rank(psu[split], psu["validation"])  # (n_split,)
        other_rank = to_rank(other[split], other["validation"])  # (n_split,)
        fused[split] = {
            "psbd": psu[split],
            "detector": other[split],
            "mean": (psu_rank + other_rank) / 2,
            "min": torch.minimum(psu_rank, other_rank),
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
    tpr = float((backdoor < threshold).float().mean())
    fpr = float((clean < threshold).float().mean())
    return tpr, fpr


def discover_folders(args: argparse.Namespace) -> list[str]:
    """Folders of the architecture with a PSBD cache and a scored record of the detector."""
    folders = sorted(
        os.path.basename(os.path.dirname(path))
        for path in glob.glob(os.path.join(args.results_dir, "*", "psbd"))
    )
    selected = []
    for folder in folders:
        if not folder.startswith(f"{args.architecture}_") or "sam_rho" in folder:
            continue
        if args.dataset and folder.split("_")[1] not in args.dataset:
            continue
        if detector_scores(args.results_dir, folder, args.detector) is None:
            continue
        selected.append(folder)
    return selected


def folder_row(folder: str, args: argparse.Namespace) -> dict | None:
    """1 checkpoint's TPR per column and FPR budget, or None when a source is missing."""
    psbd_dir = os.path.join(args.results_dir, folder, "psbd")
    scored = psbd_scores(psbd_dir, args.placement, args.shift_target)
    other = detector_scores(args.results_dir, folder, args.detector)
    if scored is None or other is None:
        return None
    psu, rate = scored
    if any(psu[split].numel() != other[split].numel() for split in SPLITS):
        return None

    manifest = read_split_manifest(psbd_dir)
    metadata = read_checkpoint_metadata(
        os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    )
    fused = fuse(psu, other)
    tprs = {}
    for target in args.fpr:
        for name in COLUMNS:
            # The clean side is paired down to the backdoor split's images, so all
            # 4 columns are compared on a single population.
            clean = pair_clean_to_backdoor(fused["clean"][name], manifest)
            tpr, _ = tpr_at_fpr(
                fused["validation"][name], clean, fused["backdoor"][name], target
            )
            tprs[(target, name)] = tpr
    row = {
        "folder": folder,
        "attack": metadata["attack"],
        "poison_rate": metadata.get("poison_rate") or 0.0,
        "rate": rate,
        "tprs": tprs,
    }
    return row


def column_label(name: str, detector: str) -> str:
    label = detector.upper() if name == "detector" else name.upper()
    return label


def render_table(rows: list[dict], args: argparse.Namespace) -> list[str]:
    """A markdown table, 1 row per checkpoint and a mean over the backdoored ones."""
    header = ["folder", "attack", "rate"]
    for target in args.fpr:
        header += [
            f"{column_label(name, args.detector)}@{target:.0%}" for name in COLUMNS
        ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["folder"], row["attack"], f"{row['poison_rate']:.3f}"]
        for target in args.fpr:
            cells += [f"{row['tprs'][(target, name)]:.3f}" for name in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    backdoored = [row for row in rows if row["attack"] != "benign"]
    if backdoored:
        cells = [f"mean over {len(backdoored)} backdoored", "", ""]
        for target in args.fpr:
            cells += [
                f"{statistics.mean(row['tprs'][(target, name)] for row in backdoored):.3f}"
                for name in COLUMNS
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    args = parse_args()
    folders = discover_folders(args)
    rows = [row for row in (folder_row(folder, args) for folder in folders) if row]
    if not rows:
        print(
            f"no folder carries both a PSBD cache at {args.placement} and a scored "
            f"{args.detector} record under {args.results_dir}"
        )
        return 1

    lines = render_table(rows, args)
    print(
        f"PSBD ({args.placement}, shift target {args.shift_target}) fused with "
        f"{args.detector} by rank, threshold from clean validation only, "
        f"{len(rows)} checkpoints\n"
    )
    print("\n".join(lines))
    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
        with open(args.markdown, "w") as handle:
            handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
