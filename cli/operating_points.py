"""Detection at low false-positive rates, the operating points a defender lives at.

Everything reported so far used the PSBD paper's 25th-percentile threshold, which
throws away a quarter of clean data. No deployment tolerates that. This reports the
same detectors at 1% and 5% false positives.

2 thresholds per operating point, and the gap between them is the interesting part:

  deployable   the threshold is the q-quantile of CLEAN VALIDATION score, exactly as
               the paper prescribes, with q set to the target FPR. This is what a
               defender can actually build, since it needs no poisoned data. The FPR
               it achieves on the analysis pool is reported next to it, because
               nothing guarantees the validation quantile transfers.

  oracle_fpr   the threshold placed directly on the clean analysis pool to hit the
               target FPR exactly. Not deployable, since it needs the very clean/poison
               split the defence is trying to find. Reported as the ceiling, so the
               cost of thresholding on validation instead is visible.

TPR is also reported at ASR-conditioned form where available: a triggered image the
trigger never flipped is behaviourally clean, so counting it as a missed detection
charges the detector for the attack's failure.

Example
    python -m cli.operating_points --poison-rate 0.01
    python -m cli.operating_points --fpr 0.01 0.05 --placement before_attention_norm_token_mask
"""

import argparse
import glob
import json
import os
from data.splits import SPLITS
from defences.decision import ADAPTIVE_SHIFT_TARGET

import numpy as np
import torch

from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import (
    attack_success_mask,
    complete_rates,
    pair_clean_to_backdoor,
)
from defences.scores import psu_ratio_from_cache, shift_ratio

DEFAULT_FPRS = (0.01, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--fpr", nargs="*", type=float, default=list(DEFAULT_FPRS))
    parser.add_argument(
        "--placement",
        nargs="*",
        default=None,
        help="restrict to these placements. The default is every placement on disk",
    )
    parser.add_argument(
        "--poison-rate",
        type=float,
        default=None,
        help="report only this poison rate",
    )
    parser.add_argument("--shift-target", type=float, default=ADAPTIVE_SHIFT_TARGET)
    return parser.parse_args()


def scores_at(psbd_dir: str, placement: str, rate: float) -> tuple[dict, float | None]:
    """Fractional PSU per split, plus the clean-validation shift ratio."""
    scores, sigma = {}, None
    for split in SPLITS:
        probs, labels, targets = load_baseline(baseline_path(psbd_dir, split))
        per_pass, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        scores[split] = psu_ratio_from_cache(probs, labels, per_pass)
        if split == "validation":
            sigma = shift_ratio(labels, argmax)
        if split == "backdoor":
            scores["captured"] = attack_success_mask(labels, targets)

    return scores, sigma


def tpr_at(
    clean: torch.Tensor, backdoor: torch.Tensor, threshold: float, inverted: bool
) -> tuple[float, float]:
    """(TPR, FPR) at a threshold, flagging the tail the detector actually separates."""
    if inverted:
        return (
            float((backdoor > threshold).float().mean()),
            float((clean > threshold).float().mean()),
        )
    return (
        float((backdoor < threshold).float().mean()),
        float((clean < threshold).float().mean()),
    )


def operating_points(
    scores: dict, manifest: dict, target_fprs: list[float]
) -> tuple[dict, bool]:
    """TPR at each target FPR, thresholded 2 ways."""
    validation = scores["validation"].numpy()
    clean = pair_clean_to_backdoor(scores["clean"], manifest)
    backdoor = scores["backdoor"]
    captured = scores.get("captured")

    # Decide the tail once, from the pooled ranking, so both thresholds agree.
    inverted = float(backdoor.mean()) > float(clean.mean())

    rows = {}
    for target in target_fprs:
        # Deployable: quantile of clean validation. Inverted rule flags the upper
        # tail, so it needs the complementary quantile to cost the same clean budget.
        quantile = 1.0 - target if inverted else target
        deployable = float(np.quantile(validation, quantile))
        tpr_deployable, fpr_deployable = tpr_at(clean, backdoor, deployable, inverted)

        # Oracle: threshold placed on the clean analysis pool itself.
        oracle = float(np.quantile(clean.numpy(), quantile))
        tpr_oracle, fpr_oracle = tpr_at(clean, backdoor, oracle, inverted)

        row = {
            "target_fpr": target,
            "tpr_deployable": tpr_deployable,
            "fpr_deployable": fpr_deployable,
            "tpr_oracle_fpr": tpr_oracle,
            "fpr_oracle": fpr_oracle,
        }
        if captured is not None and bool(captured.any()):
            row["tpr_captured_only"] = tpr_at(
                clean[captured], backdoor[captured], deployable, inverted
            )[0]
        rows[f"fpr{target:.2f}"] = row

    return rows, inverted


def best_placement(
    psbd_dir: str,
    placements: list[str],
    target_fprs: list[float],
    shift_target: float,
) -> tuple | None:
    """The placement and rate with the highest deployable TPR at the tightest FPR."""
    tightest = min(target_fprs)
    best = None
    for placement in placements:
        for rate in complete_rates(psbd_dir, placement):
            try:
                scores, sigma = scores_at(psbd_dir, placement, rate)
            except Exception:
                continue
            if sigma is None or sigma < shift_target:
                continue
            manifest = read_split_manifest(psbd_dir)
            rows, inverted = operating_points(scores, manifest, target_fprs)
            value = rows[f"fpr{tightest:.2f}"]["tpr_deployable"]
            if best is None or value > best[0]:
                best = (value, placement, rate, rows, inverted)
            break  # first rate meeting the shift target, per the adaptive rule

    return best


def discover_folders(results_dir: str) -> list[str]:
    """Every results/<folder> carrying a stage-1 cache, in sorted order."""
    folders = sorted(
        os.path.basename(os.path.dirname(path))
        for path in glob.glob(os.path.join(results_dir, "*", "psbd"))
    )
    return folders


def build_header(target_fprs: list[float]) -> str:
    """The fixed-width header, a column pair per target FPR.

    Architecture and dataset are in the row because the sweep now spans 2 of the
    first and 4 of the second. Without them the same attack appears several times
    with different numbers and reads as a bug or, worse, gets averaged.
    """
    header = (
        f"{'arch':5} {'dataset':14} {'attack':16} {'pr':>5} {'ASR':>5} "
        f"{'best placement':>26} {'p':>4}"
    )
    for target in target_fprs:
        header += f" | {f'TPR@{target:.0%}':>9} {'(FPR)':>7}"
    return header


def render_row(
    meta: dict, placement: str, rate: float, rows: dict, target_fprs: list[float]
) -> str:
    """A checkpoint's line: its provenance, its best placement and its TPRs."""
    asr = meta.get("asr")
    line = (
        f"{meta.get('architecture') or '?':5} "
        f"{meta.get('dataset') or '?':14} {meta.get('attack') or 'benign':16} "
        f"{meta.get('poison_rate') or 0:>5.3f} "
        f"{'--' if asr is None else f'{asr:.2f}':>5} {placement:>26} {rate:>4g}"
    )
    for target in target_fprs:
        row = rows[f"fpr{target:.2f}"]
        line += f" | {row['tpr_deployable']:>9.3f} {row['fpr_deployable']:>7.3f}"
    return line


def main() -> None:
    args = parse_args()

    print("Detection at low false-positive rates. Score: fractional PSU.")
    print(
        f"Rate chosen by the adaptive rule (clean-validation shift ratio >= "
        f"{args.shift_target}), placement chosen by best deployable TPR at "
        f"{min(args.fpr):.0%} FPR.\n"
    )
    header = build_header(args.fpr)
    print(header)
    print("-" * len(header))

    for folder in discover_folders(args.results_dir):
        if "sam_rho" in folder:
            continue
        meta_path = os.path.join(args.checkpoints_dir, folder, "metrics.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        if args.poison_rate is not None and meta.get("poison_rate") != args.poison_rate:
            continue

        psbd_dir = os.path.join(args.results_dir, folder, "psbd")
        placements = args.placement or sorted(
            name
            for name in os.listdir(psbd_dir)
            if os.path.isdir(os.path.join(psbd_dir, name))
        )
        best = best_placement(psbd_dir, placements, args.fpr, args.shift_target)
        if best is None:
            continue

        _value, placement, rate, rows, inverted = best
        line = render_row(meta, placement, rate, rows, args.fpr)
        print(line + ("  [inverted]" if inverted else ""))


if __name__ == "__main__":
    main()
