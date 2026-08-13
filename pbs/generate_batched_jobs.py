"""Pack the PSBD grid into a few long jobs instead of thousands of short ones.

The earlier generators emitted one job per (checkpoint, placement). Each ran about
4 minutes, of which a real fraction was pure overhead: scheduler dispatch, venv
activation, torch import, dataset construction, ImageNet weight load. Thousands of
those both waste that overhead once per config and swamp the queue.

Here one job carries many (checkpoint, placement) pairs and runs them in a single
process, so:

- the clean test set is built once per job rather than once per config, since
  loaders keeps it lru_cached per process;
- torch import and venv activation are paid once per job;
- the scheduler sees tens of jobs instead of thousands, which is the part that
  annoys everyone else on the cluster.

Jobs are packed by checkpoint, never splitting a checkpoint across two jobs, because
the three no-dropout baseline tensors are computed once per checkpoint and shared by
all of its placements. Splitting one checkpoint across jobs would either recompute
them or race on the same files.

--skip-existing is passed to every job, so a resubmitted batch costs nothing for work
already on disk. That makes these safe to regenerate and resubmit at any point.

Example
    python pbs/generate_batched_jobs.py --dry-run
    python pbs/generate_batched_jobs.py --architecture vit --hours 4
"""

import argparse
import json
import os

BASE = "/lustre/home/pstika/projects/PSBD-ViT"

ATTACKS = (
    "badnet_a2o",
    "badnet_a2a",
    "blend",
    "bpp",
    "lf",
    "wanet",
    "adaptive_blend",
    "sig",
    "lc",
    "tact",
)
POISON_TAGS = ("0_005", "0_01", "0_05", "0_1")
SAM_RHOS = ("0_05", "0_1", "0_15", "0_2")
MIN_ASR = 0.8

PLACEMENTS = (
    "post_residual",
    "pre_residual",
    "before_mlp_residual",
    "before_attention_norm",
)
BLOCK_COUNT = {"vit": 12, "swin": 24}
BAND_FRACTIONS = ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0))

FINE_RATES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09)
MAIN_RATES = tuple(i / 10 for i in range(1, 10))

# Measured: one (checkpoint, placement) over the 9-rate grid and the full 10000-image
# split takes about 4 minutes on an A100. post_residual carries 16 rates rather than
# 9, so it is weighted accordingly when packing.
MINUTES_PER_PLACEMENT = 4.0
MINUTES_PER_FINE_PLACEMENT = 7.0

BENIGN_PROBE_ATTACK = "badnet_a2o"
BENIGN_PROBE_TARGET_LABEL = 0

TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N psbd_{architecture}_{index:03d}
#PBS -o {base}/logs/psbd_batch/{architecture}_{index:03d}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Batch:   {n_checkpoints} checkpoints, {n_configs} configs, est {estimate} min"
nvidia-smi --query-gpu=name --format=csv,noheader

BASE={base}
cd $BASE
source .venv/bin/activate

{commands}
echo "Finished: $(date)"
exit 0
"""

# One command per placement group, each covering every checkpoint in the batch. Split
# this way rather than one command per checkpoint because --position-config takes a
# list and the rate grid differs between post_residual and the rest.
COMMAND = """python psbd_dropout_sweep.py \\
    --checkpoint-folder {folders} \\
    --position-config {positions}{band}{rates} \\
    --skip-existing
"""


def bands_for(architecture: str):
    blocks = BLOCK_COUNT[architecture]
    return [
        (int(round(low * blocks)) + 1, int(round(high * blocks)))
        for low, high in BAND_FRACTIONS
    ]


def viable_checkpoints(architecture, datasets, with_sam, checkpoints_dir):
    """Checkpoints whose backdoor actually fires, plus the benign controls."""
    kept = []
    for dataset in datasets:
        names = []
        for attack in ATTACKS:
            for tag in POISON_TAGS:
                stem = f"{architecture}_{dataset}_{attack}_{tag}"
                names.append(stem)
                if with_sam:
                    names.extend(f"{stem}_sam_rho_{rho}" for rho in SAM_RHOS)
        names.append(f"{architecture}_{dataset}_benign")
        if with_sam:
            names.extend(
                f"{architecture}_{dataset}_benign_sam_rho_{rho}" for rho in SAM_RHOS
            )
        for folder in names:
            path = os.path.join(checkpoints_dir, folder, "attack_result.pt")
            if not os.path.exists(path):
                continue
            if "benign" in folder:
                kept.append(folder)
                continue
            metrics_path = os.path.join(checkpoints_dir, folder, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as handle:
                asr = json.load(handle).get("asr")
            if asr is not None and asr >= MIN_ASR:
                kept.append(folder)
    return kept


def minutes_per_checkpoint(architecture: str) -> float:
    plain = len(PLACEMENTS) - 1  # post_residual costed separately
    bands = len(bands_for(architecture))
    return (plain + bands) * MINUTES_PER_PLACEMENT + MINUTES_PER_FINE_PLACEMENT


def pack(checkpoints, architecture, target_minutes):
    """Greedy fixed-size packing; every checkpoint stays whole."""
    per = minutes_per_checkpoint(architecture)
    per_batch = max(1, int(target_minutes // per))
    return [
        checkpoints[i : i + per_batch] for i in range(0, len(checkpoints), per_batch)
    ]


def build_commands(folders, architecture):
    """The command lines for one batch, grouped so each rate grid is issued once."""
    benign = [f for f in folders if "benign" in f]
    attacked = [f for f in folders if "benign" not in f]
    lines = []

    for group in (attacked, benign):
        if not group:
            continue
        probe = (
            f" \\\n    --probe-attack {BENIGN_PROBE_ATTACK}"
            f" \\\n    --probe-target-label {BENIGN_PROBE_TARGET_LABEL}"
            if group is benign
            else ""
        )
        joined = " ".join(group)

        lines.append(
            COMMAND.format(
                folders=joined,
                positions=" ".join(p for p in PLACEMENTS if p != "post_residual"),
                band="",
                rates=probe,
            )
        )
        lines.append(
            COMMAND.format(
                folders=joined,
                positions="post_residual",
                band="",
                rates=probe
                + " \\\n    --rates "
                + " ".join(f"{r:g}" for r in FINE_RATES + MAIN_RATES),
            )
        )
        for first, last in bands_for(architecture):
            lines.append(
                COMMAND.format(
                    folders=joined,
                    positions="pre_residual",
                    band=f" \\\n    --block-range {first} {last}",
                    rates=probe,
                )
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", nargs="*", default=["vit"])
    parser.add_argument("--dataset", nargs="*", default=["cifar10", "cifar100"])
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument(
        "--hours", type=float, default=4.0, help="target runtime per job"
    )
    parser.add_argument(
        "--walltime-hours",
        type=float,
        default=6.0,
        help="requested walltime, comfortably above --hours",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    walltime = f"{int(args.walltime_hours):02d}:00:00"
    os.makedirs(os.path.join(BASE, "logs", "psbd_batch"), exist_ok=True)
    out_dir = os.path.join(BASE, "pbs", "psbd_batched")
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for architecture in args.architecture:
        index = 0
        checkpoints = viable_checkpoints(
            architecture, args.dataset, not args.no_sam, args.checkpoints_dir
        )
        batches = pack(checkpoints, architecture, args.hours * 60)
        per = minutes_per_checkpoint(architecture)
        print(
            f"{architecture}: {len(checkpoints)} checkpoints, "
            f"{per:.0f} min each, {len(batches)} jobs of "
            f"~{len(batches[0]) if batches else 0} checkpoints "
            f"(~{per * (len(batches[0]) if batches else 0) / 60:.1f} h)"
        )
        for folders in batches:
            index += 1
            body = TEMPLATE.format(
                walltime=walltime,
                base=BASE,
                architecture=architecture,
                index=index,
                n_checkpoints=len(folders),
                n_configs=len(folders)
                * (len(PLACEMENTS) + len(bands_for(architecture))),
                estimate=int(per * len(folders)),
                commands=build_commands(folders, architecture),
            )
            path = os.path.join(out_dir, f"{architecture}_{index:03d}.pbs")
            if not args.dry_run:
                with open(path, "w") as handle:
                    handle.write(body)
            written.append(path)

    print(f"\n{len(written)} jobs, walltime {walltime} each")
    if args.dry_run:
        print("(dry run, nothing written)")
    else:
        print(f"wrote to {out_dir}")


if __name__ == "__main__":
    main()
