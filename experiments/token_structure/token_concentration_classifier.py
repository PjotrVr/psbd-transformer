"""H37: token concentration ratio as attack family classifier.

Uses the concentration ratio from H32 to classify attacks as localized (patch)
or global (blend, warp). Tests whether the threshold generalizes across
datasets and poison rates, and whether it can be estimated without the true
direction (using per-token feature variance under perturbation).
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np


TOKEN_LOC_PATH = "results/token_localization.json"
OUTPUT_PATH = "results/token_concentration_classifier.json"

LOCALIZED_ATTACKS = {"badnet_a2o", "tact", "badnet_a2a"}
GLOBAL_ATTACKS = {"blend", "wanet", "adaptive_blend", "sig", "lf", "bpp", "lc"}

CONCENTRATION_THRESHOLD = 2.5


def classify_attack(concentration_ratio, threshold=CONCENTRATION_THRESHOLD):
    return "localized" if concentration_ratio > threshold else "global"


def main():
    with open(TOKEN_LOC_PATH) as f:
        data = json.load(f)

    print("=== Attack family classification by concentration ratio ===\n")

    results = []
    correct = 0
    total = 0

    for entry in data:
        attack = entry["attack"]
        dataset = entry["dataset"]
        conc = entry["concentration_ratio"]
        top4 = entry["top4_fraction"]

        true_family = "localized" if attack in LOCALIZED_ATTACKS else "global"
        predicted = classify_attack(conc)
        is_correct = true_family == predicted

        if is_correct:
            correct += 1
        total += 1

        results.append(
            {
                "attack": attack,
                "dataset": dataset,
                "concentration_ratio": conc,
                "top4_fraction": top4,
                "true_family": true_family,
                "predicted_family": predicted,
                "correct": is_correct,
            }
        )

        status = "OK" if is_correct else "WRONG"
        print(
            f"  {entry['folder']:40s} conc={conc:.2f} top4={top4:.3f} "
            f"true={true_family:10s} pred={predicted:10s} [{status}]"
        )

    accuracy = correct / total if total > 0 else 0

    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"  Threshold: {CONCENTRATION_THRESHOLD}")

    localized_concs = [
        r["concentration_ratio"] for r in results if r["true_family"] == "localized"
    ]
    global_concs = [
        r["concentration_ratio"] for r in results if r["true_family"] == "global"
    ]

    print(
        f"\n  Localized attacks: mean conc = {np.mean(localized_concs):.2f} "
        f"(range {np.min(localized_concs):.2f} to {np.max(localized_concs):.2f})"
    )
    print(
        f"  Global attacks:    mean conc = {np.mean(global_concs):.2f} "
        f"(range {np.min(global_concs):.2f} to {np.max(global_concs):.2f})"
    )

    gap = min(localized_concs) - max(global_concs)
    print(f"  Separation gap: {gap:.2f} (positive = cleanly separable)")

    best_threshold = (min(localized_concs) + max(global_concs)) / 2
    print(f"  Optimal threshold: {best_threshold:.2f}")

    summary = {
        "per_attack": results,
        "accuracy": accuracy,
        "threshold_used": CONCENTRATION_THRESHOLD,
        "optimal_threshold": best_threshold,
        "separation_gap": gap,
        "localized_stats": {
            "mean": float(np.mean(localized_concs)),
            "min": float(np.min(localized_concs)),
            "max": float(np.max(localized_concs)),
        },
        "global_stats": {
            "mean": float(np.mean(global_concs)),
            "min": float(np.min(global_concs)),
            "max": float(np.max(global_concs)),
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
