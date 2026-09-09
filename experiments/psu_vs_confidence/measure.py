"""Does PSU beat a detector that just reads the no-dropout confidence?

The skeptical reading of PSBD: a backdoored model is extremely confident on triggered
inputs, PSU subtracts a dropout-perturbed confidence from the no-dropout one, and a
sample that starts near probability 1 has more room to fall than one that starts at
0.6. If that is the whole story, then PSU is an elaborate way of measuring baseline
confidence, and a defender could skip the k stochastic forward passes entirely.

This runs the comparison, from the cached tensors only, no GPU:

  psu             P_c(x) - mean over k dropout passes of P_c(x)   [the method]
  confidence      P_c(x), the no-dropout probability alone         [free baseline]
  psu_ratio       1 - mean_dropout / P_c(x), i.e. the FRACTIONAL drop

The third is the interesting one. If PSU only works because confident samples fall
further in absolute terms, normalising by the starting confidence should destroy it.
If the fractional drop separates just as well, then PSU is measuring robustness and
the confidence story is wrong.

AUROC is computed with backdoor as the positive class. PSU and the fractional drop
are negated (low means poisoned); confidence is NOT negated, because the claim there
is that backdoor samples are MORE confident.

Example
    PYTHONPATH=. python experiments/psu_vs_confidence/measure.py
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import pair_clean_to_backdoor
from defences.scores import psu_from_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--placement", default="before_mlp_residual")
    parser.add_argument("--poison-rate-tag", default="0_1")
    return parser.parse_args()


def auroc(clean: torch.Tensor, backdoor: torch.Tensor, negate: bool) -> float:
    """Backdoor is the positive class. negate when LOW means poisoned."""
    sign = -1.0 if negate else 1.0
    scores = np.concatenate(
        [sign * clean.float().numpy(), sign * backdoor.float().numpy()]
    )
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    return float(roc_auc_score(labels, scores))


def tracked_confidence(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """P_c(x): the no-dropout probability of the class the model predicted."""
    return probs.gather(1, labels.view(-1, 1).long()).squeeze(1).float()


def scores_for_split(psbd_dir: str, placement: str, rate: float, split: str):
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    per_pass, _ = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, placement, rate, split)
    )
    confidence = tracked_confidence(probs, labels)
    psu = psu_from_cache(probs, labels, per_pass)
    # Fractional drop rather than absolute. Clamped because a confidence of exactly 0
    # would make the ratio meaningless, not because it is expected to occur.
    ratio = psu / confidence.clamp_min(1e-6)
    return confidence, psu, ratio


def best_rate(psbd_dir: str, placement: str) -> float:
    """The rate whose PSU separates best, so the comparison is at PSBD's own best."""
    folder = os.path.join(psbd_dir, placement)
    rates = sorted(
        {
            float(name[len("rate_") :].rsplit("_", 1)[0].replace("_", "."))
            for name in os.listdir(folder)
            if name.startswith("rate_") and name.endswith(".pt")
        }
    )
    manifest = read_split_manifest(psbd_dir)
    best, best_auroc = rates[0], -1.0
    for rate in rates:
        _, clean_psu, _ = scores_for_split(psbd_dir, placement, rate, "clean")
        _, backdoor_psu, _ = scores_for_split(psbd_dir, placement, rate, "backdoor")
        value = auroc(pair_clean_to_backdoor(clean_psu, manifest), backdoor_psu, True)
        if value > best_auroc:
            best, best_auroc = rate, value
    return best


def main() -> None:
    args = parse_args()
    print(f"placement {args.placement}, PSU's own best rate per checkpoint\n")
    print(
        f"{'checkpoint':30} {'rate':>5} {'PSU':>7} {'confidence':>11} "
        f"{'frac drop':>10} {'PSU - conf':>11}"
    )
    print("-" * 80)

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*", "psbd"))):
        folder = os.path.basename(os.path.dirname(path))
        if args.poison_rate_tag not in folder and "benign" not in folder:
            continue
        if not os.path.isdir(os.path.join(path, args.placement)):
            continue
        manifest = read_split_manifest(path)
        rate = best_rate(path, args.placement)

        clean = scores_for_split(path, args.placement, rate, "clean")
        backdoor = scores_for_split(path, args.placement, rate, "backdoor")
        paired = [pair_clean_to_backdoor(column, manifest) for column in clean]

        values = {
            "psu": auroc(paired[1], backdoor[1], True),
            # Not negated: the skeptical claim is that backdoor samples are MORE
            # confident, so higher confidence should mean poisoned.
            "confidence": auroc(paired[0], backdoor[0], False),
            "ratio": auroc(paired[2], backdoor[2], True),
        }
        rows.append((folder, values))
        print(
            f"{folder:30} {rate:>5g} {values['psu']:>7.3f} {values['confidence']:>11.3f} "
            f"{values['ratio']:>10.3f} {values['psu'] - values['confidence']:>+11.3f}"
        )

    backdoored = [(f, v) for f, v in rows if "benign" not in f]
    if backdoored:
        print(f"\nmean over {len(backdoored)} backdoored checkpoints:")
        for key in ("psu", "confidence", "ratio"):
            mean = sum(v[key] for _, v in backdoored) / len(backdoored)
            print(f"  {key:12} {mean:.3f}")
        wins = sum(1 for _, v in backdoored if v["psu"] > v["confidence"])
        print(f"  PSU beats confidence on {wins}/{len(backdoored)} checkpoints")

    output = os.path.join(args.results_dir, "psu_vs_confidence.json")
    with open(output, "w") as handle:
        json.dump({"placement": args.placement, "rows": dict(rows)}, handle, indent=2)
    print(f"\nwritten to {output}")


if __name__ == "__main__":
    main()
