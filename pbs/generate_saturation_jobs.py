"""Extend the dropout-rate grid past 0.9 for the 2 attacks that never saturate.

The standard grid stops at 0.9 (cli.sweep.DROPOUT_RATES). At the recommended
deployment config, 7 of 9 attacks peak strictly inside that grid, so 0.9 is a
ceiling nobody was pushing against. wanet and badnet_a2a peak AT 0.9, which means
the grid, not the method, is what bounds their measured AUROC. wanet reads 0.905
at the grid edge and rising; the number would be reported as its optimum without
ever having been shown to be one.

The rates here go to 0.98 rather than 1.0 because a rate of 1.0 masks everything
and the score stops depending on the input at all.

Writes to pbs/psbd_saturation/, which is gitignored like every other generated
job directory.
"""

import os

RATES = (0.92, 0.94, 0.96, 0.98)

# The recommended deployment config, the one whose reported number this decides.
POSITION = "before_attention_norm"
OPERATOR = "token_mask"

DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
POISON_RATES = ("0_005", "0_01", "0_05", "0_1")
UNSATURATED_ATTACKS = ("wanet", "badnet_a2a")

CHECKPOINTS_PER_JOB = 4

JOB_DIRECTORY = "pbs/psbd_saturation"
LOG_DIRECTORY = "logs/psbd_saturation"

TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime=04:00:00
#PBS -N psbd_sat_{index:03d}
#PBS -o {base}/{log_directory}/sat_{index:03d}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

BASE={base}
cd $BASE
source .venv/bin/activate

python psbd_dropout_sweep.py \\
    --checkpoint-folder {folders} \\
    --position-config {position} \\
    --perturbation {operator} \\
    --rates {rates} \\
    --forward-passes 3 \\
    --skip-existing

echo "Finished: $(date)"
exit 0
"""


def existing_checkpoints():
    """Plain checkpoints only: SAM, evasive and custom-dropout runs answer other questions."""
    folders = []
    for attack in UNSATURATED_ATTACKS:
        for dataset in DATASETS:
            for poison_rate in POISON_RATES:
                folder = f"vit_{dataset}_{attack}_{poison_rate}"
                if os.path.isdir(os.path.join("checkpoints", folder)):
                    folders.append(folder)
    return folders


def batched(items, size):
    return [items[start : start + size] for start in range(0, len(items), size)]


def write_jobs(batches, base):
    os.makedirs(JOB_DIRECTORY, exist_ok=True)
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    paths = []
    for index, folders in enumerate(batches, start=1):
        path = os.path.join(JOB_DIRECTORY, f"sat_{index:03d}.pbs")
        with open(path, "w") as handle:
            handle.write(
                TEMPLATE.format(
                    index=index,
                    base=base,
                    log_directory=LOG_DIRECTORY,
                    folders=" ".join(folders),
                    position=POSITION,
                    operator=OPERATOR,
                    rates=" ".join(str(rate) for rate in RATES),
                )
            )
        paths.append(path)
    return paths


def main():
    base = os.path.abspath(".")
    folders = existing_checkpoints()
    batches = batched(folders, CHECKPOINTS_PER_JOB)
    paths = write_jobs(batches, base)
    print(f"{len(folders)} checkpoints, {len(paths)} jobs, rates {RATES}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
