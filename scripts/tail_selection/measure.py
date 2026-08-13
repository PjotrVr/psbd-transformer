"""Pick the detection tail without labels, so the two-sided gain becomes deployable.

H15 showed both PSBD and STRIP invert on the attacks their stated mechanism does not
cover, and that allowing the sign to flip recovers up to +0.98 AUROC. But
max(AUROC, 1-AUROC) reads the labels, so it is an upper bound, not a method.

This tests a rule that reads no labels. The defender holds two things: a clean
validation split, and a suspicious pool that is mostly clean with some unknown
fraction poisoned. Poisoned samples pile up in ONE tail of the score distribution.
So compare the pool's tails against validation's:

    low_deviation  = quantile(validation, q)     - quantile(pool, q)
    high_deviation = quantile(pool, 1 - q)       - quantile(validation, 1 - q)

whichever is larger names the tail the anomaly sits in. Both are measured against the
defender's own clean reference, and the pool is used only through its unlabelled
score distribution.

The comparison that matters is not the AUROC this achieves. It is whether the selected
tail AGREES with the tail the labels would have chosen. If agreement is high, the
two-sided gain is real and deployable; if not, H15 stays an upper bound.

Example
    PYTHONPATH=. python scripts/tail_selection/measure.py --detector strip
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from defences.psbd_cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.psbd_metrics import (
    complete_rates,
    pair_clean_to_backdoor,
    psu_ratio_from_cache,
    shift_ratio,
)

TAIL_QUANTILE = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--placement", default="pre_residual_blocks_5_8")
    parser.add_argument("--shift-target", type=float, default=0.7)
    parser.add_argument("--tail-quantile", type=float, default=TAIL_QUANTILE)
    return parser.parse_args()


def select_tail(validation: np.ndarray, pool: np.ndarray, q: float) -> str:
    """Which tail the anomaly sits in, from unlabelled scores alone.

    Returns "low" or "high". Symmetric by construction: each side measures how far
    the pool's tail has moved AWAY from the clean reference in the direction that
    would make it more extreme.
    """
    low = float(np.quantile(validation, q) - np.quantile(pool, q))
    high = float(np.quantile(pool, 1 - q) - np.quantile(validation, 1 - q))
    return "low" if low >= high else "high"


def auroc_for_tail(clean: torch.Tensor, backdoor: torch.Tensor, tail: str) -> float:
    from sklearn.metrics import roc_auc_score

    sign = -1.0 if tail == "low" else 1.0
    scores = np.concatenate(
        [sign * clean.float().numpy(), sign * backdoor.float().numpy()]
    )
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    return float(roc_auc_score(labels, scores))


def psbd_at_adaptive_rate(psbd_dir: str, placement: str, shift_target: float):
    for rate in complete_rates(psbd_dir, placement):
        _, labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "validation")
        )
        sigma = shift_ratio(labels, argmax)
        if sigma is None or sigma < shift_target:
            continue
        out = {}
        for split in ("validation", "clean", "backdoor"):
            probs, lab, _ = load_baseline(baseline_path(psbd_dir, split))
            per_pass, _ = load_dropout_pass_probs(
                dropout_pass_path(psbd_dir, placement, rate, split)
            )
            out[split] = psu_ratio_from_cache(probs, lab, per_pass)
        return out
    return None


def main() -> None:
    args = parse_args()
    print("Choosing the detection tail from unlabelled scores alone.")
    print("The question is whether the label-free choice AGREES with the oracle.\n")
    print(
        f"{'checkpoint':32} {'one-sided':>10} {'oracle 2s':>10} "
        f"{'chosen':>7} {'auto':>8} {'agrees':>7}"
    )
    print("-" * 82)

    agree = total = 0
    one_sided_sum = auto_sum = oracle_sum = 0.0
    for path in sorted(
        glob.glob(os.path.join(args.results_dir, "vit_cifar10_*", "psbd"))
    ):
        folder = os.path.basename(os.path.dirname(path))
        if "sam_rho" in folder or "benign" in folder:
            continue
        scores = psbd_at_adaptive_rate(path, args.placement, args.shift_target)
        if scores is None:
            continue
        manifest = read_split_manifest(path)
        clean = pair_clean_to_backdoor(scores["clean"], manifest)
        backdoor = scores["backdoor"]

        # What a defender actually sees: one unlabelled pool.
        pool = torch.cat([clean, backdoor]).numpy()
        chosen = select_tail(scores["validation"].numpy(), pool, args.tail_quantile)

        low = auroc_for_tail(clean, backdoor, "low")
        oracle_tail = "low" if low >= 0.5 else "high"
        oracle = max(low, 1.0 - low)
        auto = low if chosen == "low" else 1.0 - low

        ok = chosen == oracle_tail
        agree += ok
        total += 1
        one_sided_sum += low
        auto_sum += auto
        oracle_sum += oracle
        print(
            f"{folder:32} {low:>10.3f} {oracle:>10.3f} "
            f"{oracle_tail:>7} {auto:>8.3f} {'yes' if ok else 'NO':>7}"
        )

    if total:
        print(
            f"\nagreement with the oracle tail: {agree}/{total} ({100 * agree / total:.0f}%)"
        )
        print(
            f"mean AUROC  one-sided {one_sided_sum / total:.3f}   "
            f"label-free two-sided {auto_sum / total:.3f}   "
            f"oracle two-sided {oracle_sum / total:.3f}"
        )
        print(
            f"the label-free rule captures "
            f"{100 * (auto_sum - one_sided_sum) / max(oracle_sum - one_sided_sum, 1e-9):.0f}% "
            "of the available gain"
        )


if __name__ == "__main__":
    main()
