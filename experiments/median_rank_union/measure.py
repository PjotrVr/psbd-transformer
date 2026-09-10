"""Does the median-rank union survive a real adaptive attacker, or only a synthetic one?

The multi-probe union is this project's answer to an adaptive attacker: evading
one probe should not evade the detector. Measured on the evasive checkpoints, the
attacker's best move turns out not to be neutralizing the probed position but
INVERTING it, and a min-rank union takes the most extreme evidence across probes,
so it adopts the inversion rather than ignoring it.

A median-rank union needs a majority to be wrong instead. That fix was validated
on synthetic probes with a known answer, which is enough to establish the
mechanism and not enough to claim the defence works. This script runs the same
comparison on the 120 evasive checkpoints that actually exist, using cached PSU
only, no GPU and no model.

Run:
    python experiments/median_rank_union/measure.py
"""

import argparse
import glob
import json
import os
from data.splits import SPLITS

import pandas as pd

from defences.cache import load_baseline, load_dropout_pass_probs
from defences.decision import (
    HEADLINE_QUANTILE,
    detection_report,
    multi_probe_detection,
    pair_clean_to_backdoor,
)
from defences.scores import psu_from_cache, shift_ratio

# The shift ratio every probe is read at, so probes are compared at a matched
# disturbance rather than at a shared nominal rate.
MATCHED_SHIFT = 0.6


def rate_tags(placement_dir):
    """Every rate this placement has a complete set of 3 splits for."""
    tags = []
    for path in sorted(glob.glob(os.path.join(placement_dir, "rate_*_clean.pt"))):
        tag = os.path.basename(path).replace("_clean.pt", "")
        if all(
            os.path.exists(os.path.join(placement_dir, f"{tag}_{split}.pt"))
            for split in SPLITS
        ):
            tags.append(tag)
    return tags


def probe_at_matched_shift(psbd_dir, placement):
    """PSU for all 3 splits at the rate whose clean-validation shift is nearest target.

    Returns None when the placement has no usable rate, which happens for a
    deterministic operator whose grid never brackets the target and for caches
    written before argmax was saved.
    """
    placement_dir = os.path.join(psbd_dir, placement)
    baselines = {}
    for split in SPLITS:
        path = os.path.join(psbd_dir, f"baseline_{split}.pt")
        if not os.path.exists(path):
            return None
        probs, labels, _loader_labels = load_baseline(path)
        baselines[split] = (probs, labels)

    best = None
    for tag in rate_tags(placement_dir):
        pass_probs, argmax = load_dropout_pass_probs(
            os.path.join(placement_dir, f"{tag}_validation.pt")
        )
        sigma = shift_ratio(baselines["validation"][1], argmax)
        if sigma is None:
            continue
        distance = abs(sigma - MATCHED_SHIFT)
        if best is None or distance < best[0]:
            best = (distance, tag, sigma)

    if best is None:
        return None

    _distance, tag, sigma = best
    psu = {}
    for split in SPLITS:
        pass_probs, _argmax = load_dropout_pass_probs(
            os.path.join(placement_dir, f"{tag}_{split}.pt")
        )
        psu[split] = psu_from_cache(
            baselines[split][0], baselines[split][1], pass_probs
        )

    return {"placement": placement, "rate_tag": tag, "sigma": sigma, "psu": psu}


def measure_checkpoint(folder, results_dir):
    """Single-probe and union results for one checkpoint, or None if unusable."""
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    placements = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(psbd_dir, "*"))
        if os.path.isdir(p)
    )

    probes = [probe_at_matched_shift(psbd_dir, name) for name in placements]
    probes = [p for p in probes if p is not None]
    if len(probes) < 3:
        return None

    with open(os.path.join(psbd_dir, "split_manifest.json")) as handle:
        manifest = json.load(handle)

    # The clean split covers the whole analysis pool and the backdoor split only the
    # eligible subset, so comparing them as served contrasts different populations and
    # measures which classes were dropped as much as it measures the defence. Every
    # other consumer in the tree pairs by original index; this one did not.
    for probe in probes:
        probe["psu"]["clean"] = pair_clean_to_backdoor(probe["psu"]["clean"], manifest)

    singles = [
        detection_report(
            p["psu"]["validation"],
            p["psu"]["clean"],
            p["psu"]["backdoor"],
            HEADLINE_QUANTILE,
        )["auroc"]
        for p in probes
    ]

    unions = {}
    for reduction in ("min", "median"):
        unions[reduction] = multi_probe_detection(
            [p["psu"]["validation"] for p in probes],
            [p["psu"]["clean"] for p in probes],
            [p["psu"]["backdoor"] for p in probes],
            HEADLINE_QUANTILE,
            reduction=reduction,
        )

    return {
        "folder": folder,
        "num_probes": len(probes),
        "best_single": max(singles),
        "worst_single": min(singles),
        # An inverted probe is the case the union has to survive, so count them.
        "inverted_probes": sum(1 for a in singles if a < 0.5),
        "union_min": unions["min"]["auroc"],
        "union_median": unions["median"]["auroc"],
        "union_min_tpr": unions["min"]["tpr"],
        "union_median_tpr": unions["median"]["tpr"],
        "union_min_fpr": unions["min"]["fpr"],
        "union_median_fpr": unions["median"]["fpr"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*evade*")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--output", default="experiments/median_rank_union/union_on_evasive.csv"
    )
    arguments = parser.parse_args()

    folders = sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(
            os.path.join(arguments.results_dir, arguments.pattern, "psbd")
        )
    )
    print(f"{len(folders)} checkpoints matching {arguments.pattern!r}")

    rows = []
    for position, folder in enumerate(folders, start=1):
        try:
            row = measure_checkpoint(folder, arguments.results_dir)
        except Exception as error:
            print(
                f"[{position}/{len(folders)}] {folder}: FAILED {type(error).__name__}: {error}"
            )
            continue
        if row is None:
            continue
        rows.append(row)
        print(
            f"[{position}/{len(folders)}] {folder}: probes {row['num_probes']} "
            f"inverted {row['inverted_probes']} best_single {row['best_single']:.3f} "
            f"min {row['union_min']:.3f} median {row['union_median']:.3f}"
        )

    measured = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    measured.to_csv(arguments.output, index=False)
    print(f"\nwrote {arguments.output} with {len(measured)} rows")


if __name__ == "__main__":
    main()
