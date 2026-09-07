"""H38: does crystallization depth predict optimal perturbation placement?

Cross-references H30's direction persistence data with existing PSBD sweep
results to test whether the layer where the backdoor direction crystallizes
(alignment-to-final > 0.5) matches the best detection depth.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np


RESULTS_DIR = "results"
PERSISTENCE_PATH = "results/direction_persistence.json"
OUTPUT_PATH = "results/crystallization_vs_placement.json"


def crystallization_depth(entry, threshold=0.5):
    """Find the earliest layer where alignment-to-final exceeds threshold."""
    alignment = entry["alignment_to_final"]
    for layer_idx, val in enumerate(alignment):
        if val is not None and val > threshold:
            return layer_idx + 1
    return None


def load_psbd_sweep_results(folder_name):
    """Load per-position AUROC from a PSBD sweep if available."""
    psbd_dir = os.path.join(RESULTS_DIR, folder_name, "psbd")
    if not os.path.exists(psbd_dir):
        return None

    position_aurocs = {}
    for config_dir in os.listdir(psbd_dir):
        config_path = os.path.join(psbd_dir, config_dir)
        if not os.path.isdir(config_path):
            continue

        for rate_dir in os.listdir(config_path):
            metrics_path = os.path.join(config_path, rate_dir, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                auroc = metrics.get("auroc")
                if auroc is not None:
                    key = config_dir
                    if key not in position_aurocs or auroc > position_aurocs[key]:
                        position_aurocs[key] = auroc
            except (json.JSONDecodeError, KeyError):
                continue

    return position_aurocs if position_aurocs else None


def extract_block_band_aurocs(folder_name):
    """Look for block-band specific results (blocks_1_4, blocks_5_8, etc.)."""
    psbd_dir = os.path.join(RESULTS_DIR, folder_name, "psbd")
    if not os.path.exists(psbd_dir):
        return None

    band_aurocs = {}
    for config_dir in os.listdir(psbd_dir):
        if "blocks_" not in config_dir:
            continue
        config_path = os.path.join(psbd_dir, config_dir)
        if not os.path.isdir(config_path):
            continue

        best_auroc = 0
        for rate_dir in os.listdir(config_path):
            metrics_path = os.path.join(config_path, rate_dir, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                auroc = metrics.get("auroc", 0)
                best_auroc = max(best_auroc, auroc)
            except (json.JSONDecodeError, KeyError):
                continue

        if best_auroc > 0:
            band_aurocs[config_dir] = best_auroc

    return band_aurocs if band_aurocs else None


def main():
    with open(PERSISTENCE_PATH) as f:
        persistence_data = json.load(f)

    print("=== Crystallization depths (alignment-to-final > 0.5) ===\n")

    all_results = []

    for entry in persistence_data:
        dataset = entry["dataset"]
        attack = entry["attack"]
        rate = entry["poison_rate"]
        folder = entry["folder"]

        crystal_depth = crystallization_depth(entry)
        alignment = entry["alignment_to_final"]

        print(f"{folder}:")
        print(f"  crystallization layer: {crystal_depth}")
        af_str = [f"{v:.3f}" if v is not None else "n/a" for v in alignment]
        print(f"  alignment to final: {' '.join(af_str)}")

        sweep_results = load_psbd_sweep_results(folder)
        band_results = extract_block_band_aurocs(folder)

        result = {
            "folder": folder,
            "dataset": dataset,
            "attack": attack,
            "poison_rate": rate,
            "crystallization_layer": crystal_depth,
            "alignment_to_final": alignment,
        }

        if sweep_results:
            best_position = max(sweep_results, key=sweep_results.get)
            result["best_psbd_position"] = best_position
            result["best_psbd_auroc"] = sweep_results[best_position]
            result["all_position_aurocs"] = sweep_results
            print(
                f"  best PSBD position: {best_position} (AUROC={sweep_results[best_position]:.3f})"
            )

        if band_results:
            best_band = max(band_results, key=band_results.get)
            result["block_band_aurocs"] = band_results
            result["best_band"] = best_band
            result["best_band_auroc"] = band_results[best_band]
            print(
                f"  best block band: {best_band} (AUROC={band_results[best_band]:.3f})"
            )

        all_results.append(result)
        print()

    print("\n=== Summary: crystallization vs detection ===\n")
    print(f"{'folder':45s} {'crystal':>8} {'best band':>15} {'AUROC':>8}")
    for r in all_results:
        band = r.get("best_band", "n/a")
        auroc = r.get("best_band_auroc", 0)
        crystal = r.get("crystallization_layer", "n/a")
        print(f"{r['folder']:45s} {str(crystal):>8} {band:>15} {auroc:8.3f}")

    badnet_crystals = [
        r["crystallization_layer"]
        for r in all_results
        if r["attack"] == "badnet_a2o" and r["crystallization_layer"] is not None
    ]
    blend_crystals = [
        r["crystallization_layer"]
        for r in all_results
        if r["attack"] == "blend" and r["crystallization_layer"] is not None
    ]

    if badnet_crystals:
        print(f"\n  badnet crystallization: mean layer {np.mean(badnet_crystals):.1f}")
    if blend_crystals:
        print(f"  blend crystallization: mean layer {np.mean(blend_crystals):.1f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
