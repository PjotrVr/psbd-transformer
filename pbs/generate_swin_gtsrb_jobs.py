"""Close the Swin/GTSRB coverage gap, which is a sweep gap and not a training gap.

`results/detection_summary.csv` has rows for swin on cifar10, cifar100 and tiny
but none for gtsrb, and the audit records that as 15 missing cells. The cause is
narrower and cheaper than it looks: **40 non-SAM, non-evade swin_gtsrb
checkpoints are trained and on disk, and not one has ever been swept.** Nothing
needs retraining.

Every claim that reports a mean over datasets is currently averaging over 3
datasets for swin and 4 for vit, which is exactly the unequal-coverage shape that
has inverted conclusions in this project before.

The placement set mirrors what vit_gtsrb carries, so the 2 architectures become
comparable on the same grid rather than on whatever each happens to have.

Run:
    python pbs/generate_swin_gtsrb_jobs.py --dry-run
    python pbs/generate_swin_gtsrb_jobs.py
"""

import argparse
import glob
import os

REPO = "/lustre/home/pstika/projects/PSBD-ViT"
JOB_DIRECTORY = "pbs/psbd_swin_gtsrb"
LOG_DIRECTORY = "logs/psbd_swin_gtsrb"

# The placements the panel and the ranking need, mirroring vit_gtsrb's grid. The
# archived gaussian_batchstd placements it also carries are deliberately not here:
# that operator is superseded and re-running it would recreate audit finding A2.
PLACEMENTS = [
    ("pre_residual", "dropout"),
    ("post_residual", "dropout"),
    ("before_attention_norm", "dropout"),
    ("before_attention_norm", "token_mask"),
    ("before_attention", "channel_mask"),
    ("before_mlp", "dropout"),
    ("before_mlp", "token_mask"),
    ("after_embedding", "token_mask"),
    ("after_attention_residual", "token_mask"),
    ("after_mlp_residual", "token_mask"),
]

# Measured at about 0.42 minutes per rate on an A100, 9 rates per placement.
MINUTES_PER_PLACEMENT = 4.0

# Generous rather than tight, which is this project's rule for walltime.
WALLTIME_HOURS = 6

TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime:02d}:00:00
#PBS -N psbd_swgts_{tag}
#PBS -o {repo}/{log_directory}/{tag}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"

cd {repo}
source .venv/bin/activate
export PYTHONPATH={repo}

{commands}

echo "Finished: $(date)"
"""


def eligible_checkpoints(checkpoints_dir):
    folders = []
    for path in sorted(glob.glob(os.path.join(checkpoints_dir, "swin_gtsrb_*"))):
        folder = os.path.basename(path)
        if "sam_rho" in folder or "evade" in folder:
            continue
        if not os.path.exists(os.path.join(path, "args.json")):
            continue
        folders.append(folder)
    return folders


def sweep_command(folder, position, operator):
    command = (
        f"python -m cli.sweep --checkpoint-folder {folder} \\\n"
        f"    --position-config {position} --perturbation {operator} --skip-existing"
    )
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--per-job", type=int, default=18)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    folders = eligible_checkpoints(arguments.checkpoints_dir)
    units = [
        sweep_command(folder, position, operator)
        for folder in folders
        for position, operator in PLACEMENTS
    ]
    minutes = len(units) * MINUTES_PER_PLACEMENT
    batches = [
        units[start : start + arguments.per_job]
        for start in range(0, len(units), arguments.per_job)
    ]

    print(f"{len(folders)} swin_gtsrb checkpoints, {len(PLACEMENTS)} placements each")
    print(
        f"{len(units)} sweeps, about {minutes / 60:.0f} GPU hours, {len(batches)} jobs"
    )
    if arguments.dry_run:
        print("\ndry run, nothing written")
        print("first job would run:")
        for command in batches[0][:3]:
            print("  " + command.replace("\\\n    ", " "))
        return

    os.makedirs(JOB_DIRECTORY, exist_ok=True)
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    for index, batch in enumerate(batches, start=1):
        tag = f"{index:03d}"
        path = os.path.join(JOB_DIRECTORY, f"swgts_{tag}.pbs")
        with open(path, "w") as handle:
            handle.write(
                TEMPLATE.format(
                    walltime=WALLTIME_HOURS,
                    tag=tag,
                    repo=REPO,
                    log_directory=LOG_DIRECTORY,
                    commands="\n\n".join(batch),
                )
            )
    print(f"wrote {len(batches)} job files to {JOB_DIRECTORY}/")


if __name__ == "__main__":
    main()
