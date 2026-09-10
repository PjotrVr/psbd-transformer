"""Does latent geometry predict how well PSBD detects, without running PSBD?

The curvature account in docs/theory-perturbation-consistency.md says PSU
estimates the curvature of the predicted class probability, and that curvature
falls as the decision margin grows. It does not say where a large margin comes
from.

The latent measurements suggest an answer. A backdoored model routes triggered
inputs onto a lower dimensional subspace than clean data occupies, measured as
rank_ratio below 1, while a benign model's representation is displaced by the
trigger but keeps its shape. Inside a low dimensional attractor there are few
competing class directions, so the margin is large and the curvature is near 0,
which is exactly the regime PSU reads as poisoned.

If that link is real, then rank_ratio predicts PSBD's AUROC across checkpoints,
and it does so without perturbing the model at all. This script measures both
quantities on every checkpoint that has a cached detection result and writes one
row per checkpoint.

Run:
    python experiments/latent_geometry_predicts_detection/measure.py --samples 500
"""

import argparse
import gc
import os

import pandas as pd
import torch

from analysis.cases import case_distribution_table, load_latent_case
from data.splits import BENIGN_PROBE_ATTACK

# Probed on a benign checkpoint, which has no trigger of its own. Chance level
# detection is the expected result and it is the control the correlation needs at
# the low end.

# SAM checkpoints are excluded everywhere in this project's reporting, so they are
# excluded here too rather than silently widening the sample.
EXCLUDED_FOLDER_SUBSTRING = "sam_rho"


def cached_detection_auroc(summary_path):
    """Best deployable AUROC per checkpoint, from the cached sweep.

    The oracle rule places its threshold on the split it is detecting, so it is
    not a deployable number and is dropped before taking the maximum.
    """
    summary = pd.read_csv(summary_path)
    deployable = summary[summary["rule"] != "oracle"]

    best = (
        deployable.groupby(
            ["folder", "architecture", "dataset", "attack", "poison_rate"], dropna=False
        )["auroc"]
        .max()
        .reset_index()
        .rename(columns={"auroc": "best_deployable_auroc"})
    )
    return best


def usable_checkpoints(best, checkpoints_dir):
    """Rows whose checkpoint is on disk, has metadata, and is in scope."""
    has_metadata = best["folder"].apply(
        lambda folder: os.path.exists(
            os.path.join(checkpoints_dir, folder, "args.json")
        )
    )
    in_scope = ~best["folder"].str.contains(EXCLUDED_FOLDER_SUBSTRING)

    usable = best[has_metadata & in_scope].reset_index(drop=True)
    return usable


def geometry_for_checkpoint(folder, attack, samples):
    """Latent geometry summary for one checkpoint, or an error row.

    Layer 0 is dropped before summarizing. On ViT under the cls reduction it is a
    learned constant, so its statistics are undefined and would otherwise be the
    minimum of every column by default.
    """
    probe = BENIGN_PROBE_ATTACK if attack == "benign" else None
    case = load_latent_case(folder, samples=samples, probe_attack=probe)
    table = [row for row in case_distribution_table(case) if row["layer"] > 0]

    frame = pd.DataFrame(table)
    depth = frame["layer"].max()
    deepest = frame.iloc[-1]

    geometry = {
        "num_layers": int(depth),
        "min_rank_ratio": float(frame["rank_ratio"].min()),
        "final_rank_ratio": float(deepest["rank_ratio"]),
        "min_cka": float(frame["cka"].min()),
        "final_cka": float(deepest["cka"]),
        "max_separation_auroc": float(frame["separation_auroc"].max()),
        "final_separation_auroc": float(deepest["separation_auroc"]),
        "max_target_alignment": float(frame["target_alignment"].max()),
        # LID carries the opposite sign in the 2 nearest published results, so it
        # is recorded beside the rank ratio rather than instead of it. Ma et al.
        # report adversarial inputs at about 4.36 against 1.53 for normal, and
        # COLLIDER filters on clean samples having LOW LID. Both say corrupted
        # inputs are locally HIGHER dimensional.
        "median_lid_clean": float(frame["lid_clean"].median()),
        "median_lid_backdoor": float(frame["lid_backdoor"].median()),
        "median_lid_ratio": float(frame["lid_ratio"].median()),
        "min_lid_ratio": float(frame["lid_ratio"].min()),
        "final_direction_norm": float(deepest["direction_norm"]),
        # Where the collapse is deepest, as a fraction of the stack, so ViT's 12
        # blocks and Swin's 24 are comparable.
        "rank_ratio_depth": float(
            frame.loc[frame["rank_ratio"].idxmin(), "layer"] / depth
        ),
    }

    del case
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--summary", default="results/detection_summary.csv")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument(
        "--output",
        default="experiments/latent_geometry_predicts_detection/geometry_vs_detection.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    # Sharding so the work splits across several single-GPU jobs rather than 1
    # long serial one, matching how every other sweep here is scheduled.
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    arguments = parser.parse_args()

    candidates = usable_checkpoints(
        cached_detection_auroc(arguments.summary), arguments.checkpoints_dir
    )
    if arguments.limit is not None:
        candidates = candidates.head(arguments.limit)
    if arguments.num_shards > 1:
        # Strided rather than blocked, so each shard gets a mix of architectures
        # and datasets and one slow shard does not hold up the merge.
        candidates = candidates.iloc[arguments.shard :: arguments.num_shards]
    print(f"{len(candidates)} checkpoints to measure at {arguments.samples} samples")

    rows = []
    for position, record in enumerate(candidates.to_dict("records"), start=1):
        folder = record["folder"]
        try:
            geometry = geometry_for_checkpoint(
                folder, record["attack"], arguments.samples
            )
        except Exception as error:
            # One unreadable checkpoint must not lose the other 281 measurements,
            # so the failure is recorded as a row rather than raised.
            print(
                f"[{position}/{len(candidates)}] {folder}: FAILED {type(error).__name__}: {error}"
            )
            rows.append({**record, "error": f"{type(error).__name__}: {error}"})
            continue

        rows.append({**record, **geometry, "error": None})
        print(
            f"[{position}/{len(candidates)}] {folder}: "
            f"auroc {record['best_deployable_auroc']:.3f}  "
            f"min_rank_ratio {geometry['min_rank_ratio']:.3f}  "
            f"min_cka {geometry['min_cka']:.3f}"
        )

    measured = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    measured.to_csv(arguments.output, index=False)
    print(
        f"wrote {arguments.output} with {len(measured)} rows, {measured['error'].notna().sum()} failures"
    )


if __name__ == "__main__":
    main()
