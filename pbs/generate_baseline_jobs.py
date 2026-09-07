"""Generate PBS jobs for the STRIP and confidence-null baselines.

baseline_detect.py has only ever been run on vit_cifar10, so every cross-detector
comparison in this project is currently one dataset wide. This closes that: STRIP
and the confidence-only null on every viable checkpoint, so the comparison table
covers the same grid the perturbation study does.

STRIP is cheap relative to a perturbation sweep (8 overlays over 3 splits, one
pass each, against 10 rates x 3 passes x 3 splits), so many checkpoints pack into
one job.

Example
    python pbs/generate_baseline_jobs.py --dataset cifar10 cifar100 gtsrb tiny --dry-run
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_batched_jobs import viable_checkpoints  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BENIGN_PROBE_ATTACK = "badnet_a2o"
BENIGN_PROBE_TARGET_LABEL = 0

# 8 overlays over roughly 18k images across the 3 splits, one forward pass each,
# at the ~2130 img/s measured for this model on an A100.
# Measured, not estimated. The Aug run did confidence and strip only, 9 forward
# passes per input, at about 1.2 minutes per checkpoint on an A100. The full set
# is 98 passes per input (confidence 1, strip 8, scale_up 6, scale_up_data_limited
# 6, ibd_psc 6, teco 71), so roughly 11 times that.
#
# TeCo alone is 71 of the 98 and 2 of its corruptions are not GPU bound:
# glass_blur runs a Python triple loop over pixels and jpeg_compression round
# trips each image through PIL on CPU. The measured figure will be worse than the
# pass ratio implies, so this is deliberately generous rather than tight.
MINUTES_PER_CHECKPOINT = 20.0

TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N psbd_{prefix}_{index:03d}
#PBS -o {base}/logs/psbd_baseline/{prefix}_{index:03d}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Work:    {n} checkpoints, est {estimate} min"
nvidia-smi --query-gpu=name --format=csv,noheader

BASE={base}
cd $BASE
source .venv/bin/activate

{commands}
echo "Finished: $(date)"
exit 0
"""

COMMAND = """python -m cli.baselines \\
    --checkpoint-folder {folders}{probe} \\
    --skip-existing
"""


def build_commands(folders: list[str]) -> str:
    """Benign controls are issued separately because they need a probe trigger."""
    benign = sorted(f for f in folders if "benign" in f)
    attacked = sorted(f for f in folders if "benign" not in f)
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
        lines.append(COMMAND.format(folders=" ".join(group), probe=probe))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default="vit")
    parser.add_argument("--dataset", nargs="*", default=["cifar10"])
    parser.add_argument("--only-tag", nargs="*", default=[])
    parser.add_argument("--with-sam", action="store_true")
    parser.add_argument("--min-asr", type=float, default=0.5)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--walltime-hours", type=float, default=6.0)
    parser.add_argument("--prefix", default="base")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folders = viable_checkpoints(
        args.architecture,
        args.dataset,
        with_sam=args.with_sam,
        checkpoints_dir=os.path.join(REPO, "checkpoints"),
        min_asr=args.min_asr,
        only_tags=args.only_tag or None,
    )
    if not folders:
        raise SystemExit("no checkpoints matched")

    per_job = max(1, int((args.hours * 60) // MINUTES_PER_CHECKPOINT))
    batches = [folders[i : i + per_job] for i in range(0, len(folders), per_job)]
    print(
        f"{len(folders)} checkpoints, {MINUTES_PER_CHECKPOINT:g} min each "
        f"-> {len(batches)} jobs of up to {per_job}"
    )
    if args.dry_run:
        print("(dry run, nothing written)")
        return

    out_dir = os.path.join(REPO, "pbs", "psbd_baseline")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(REPO, "logs", "psbd_baseline"), exist_ok=True)
    hours, minutes = divmod(int(round(args.walltime_hours * 60)), 60)
    for index, batch in enumerate(batches, start=1):
        path = os.path.join(out_dir, f"{args.prefix}_{index:03d}.pbs")
        with open(path, "w") as handle:
            handle.write(
                TEMPLATE.format(
                    walltime=f"{hours:02d}:{minutes:02d}:00",
                    prefix=args.prefix,
                    index=index,
                    base=REPO,
                    n=len(batch),
                    estimate=int(len(batch) * MINUTES_PER_CHECKPOINT),
                    commands=build_commands(batch),
                )
            )
        print(f"  wrote {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    main()
