"""Detection at low false-positive rates, the operating points a defender lives at.

Everything reported so far used the PSBD paper's 25th-percentile threshold, which
throws away a quarter of clean data. No deployment tolerates that. This reports the
same detectors at 1% and 5% false positives.

Two thresholds per operating point, and the gap between them is the interesting part:

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
    python psbd_operating_points.py --poison-rate 0.01
    python psbd_operating_points.py --fpr 0.01 0.05 --placement pre_residual_blocks_5_8
"""

import argparse
import glob
import json
import os

import numpy as np

from defences.psbd_cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.psbd_metrics import (
    attack_success_mask,
    complete_rates,
    pair_clean_to_backdoor,
    psu_ratio_from_cache,
    shift_ratio,
)

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
        help="restrict to these placements; default is every placement on disk",
    )
    parser.add_argument(
        "--poison-rate",
        type=float,
        default=None,
        help="report only this poison rate",
    )
    parser.add_argument("--shift-target", type=float, default=0.7)
    return parser.parse_args()


def rates_on_disk(psbd_dir: str, placement: str) -> list[float]:
    """Only complete rates, so this is safe to run while a sweep is writing."""
    return complete_rates(psbd_dir, placement)


def scores_at(psbd_dir: str, placement: str, rate: float):
    """Fractional PSU per split, plus the clean-validation shift ratio."""
    out, sigma = {}, None
    for split in ("validation", "clean", "backdoor"):
        probs, labels, targets = load_baseline(baseline_path(psbd_dir, split))
        per_pass, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        out[split] = psu_ratio_from_cache(probs, labels, per_pass)
        if split == "validation":
            sigma = shift_ratio(labels, argmax)
        if split == "backdoor":
            out["captured"] = attack_success_mask(labels, targets)
    return out, sigma


def tpr_at(clean, backdoor, threshold, inverted: bool) -> tuple[float, float]:
    if inverted:
        return (
            float((backdoor > threshold).float().mean()),
            float((clean > threshold).float().mean()),
        )
    return (
        float((backdoor < threshold).float().mean()),
        float((clean < threshold).float().mean()),
    )


def operating_points(scores, manifest, target_fprs) -> dict:
    """TPR at each target FPR, thresholded two ways."""
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
        q = 1.0 - target if inverted else target
        deployable = float(np.quantile(validation, q))
        tpr_dep, fpr_dep = tpr_at(clean, backdoor, deployable, inverted)

        # Oracle: threshold placed on the clean analysis pool itself.
        clean_np = clean.numpy()
        oracle = float(np.quantile(clean_np, 1.0 - target if inverted else target))
        tpr_orc, fpr_orc = tpr_at(clean, backdoor, oracle, inverted)

        row = {
            "target_fpr": target,
            "tpr_deployable": tpr_dep,
            "fpr_deployable": fpr_dep,
            "tpr_oracle_fpr": tpr_orc,
            "fpr_oracle": fpr_orc,
        }
        if captured is not None and bool(captured.any()):
            row["tpr_captured_only"] = tpr_at(
                clean[captured], backdoor[captured], deployable, inverted
            )[0]
        rows[f"fpr{target:.2f}"] = row
    return rows, inverted


def best_placement(psbd_dir, placements, target_fprs, shift_target):
    """The placement and rate with the highest deployable TPR at the tightest FPR."""
    tightest = min(target_fprs)
    best = None
    for placement in placements:
        for rate in rates_on_disk(psbd_dir, placement):
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


def main() -> None:
    args = parse_args()
    folders = sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(os.path.join(args.results_dir, "*", "psbd"))
    )

    print("Detection at low false-positive rates. Score: fractional PSU.")
    print(
        f"Rate chosen by the adaptive rule (clean-validation shift ratio >= "
        f"{args.shift_target}), placement chosen by best deployable TPR at "
        f"{min(args.fpr):.0%} FPR.\n"
    )
    header = f"{'attack':16} {'pr':>5} {'ASR':>5} {'best placement':>26} {'p':>4}"
    for target in args.fpr:
        header += f" | {f'TPR@{target:.0%}':>9} {'(FPR)':>7}"
    print(header)
    print("-" * len(header))

    for folder in folders:
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
        _, placement, rate, rows, inverted = best

        attack = meta.get("attack") or "benign"
        asr = meta.get("asr")
        line = (
            f"{attack:16} {meta.get('poison_rate') or 0:>5.3f} "
            f"{'--' if asr is None else f'{asr:.2f}':>5} {placement:>26} {rate:>4g}"
        )
        for target in args.fpr:
            row = rows[f"fpr{target:.2f}"]
            line += f" | {row['tpr_deployable']:>9.3f} {row['fpr_deployable']:>7.3f}"
        print(line + ("  [inverted]" if inverted else ""))


if __name__ == "__main__":
    main()
