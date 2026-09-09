"""Does widening a clean-label attack's target set make it implant?

A clean-label attack poisons only images that already carry a target label, so 1
target caps it at 1/K of a balanced training set. That is 1% on CIFAR-100 and
0.5% on Tiny, and SIG implants at neither. Widening the set to a few adjacent
classes is the only way to lift the ceiling without changing the dataset.

The batch is built to separate the 2 explanations for why SIG fails:

  cifar100  the rate is held at 1%, which m=1 already reaches, while m=2 and m=3
            take the poisoned count from 500 to 1000 and 1500. Only the count
            moves, so if SIG implants here the binding constraint was count.
  tiny      m=1 cannot reach 1% at all, so its clean-label cell does not exist
            today. m=2 makes it reachable and m=3 gives headroom.

SIG only, because it needs no adversarial bases and is the attack that actually
fails. Label-Consistent follows if this looks worth pursuing, and would need
bases generated for every class in the set.

1 seed per cell. The point is a direction, not an interval.

    python pbs/generate_multitarget_jobs.py
"""

import argparse
import os

PROJECT_ROOT = "/lustre/home/pstika/projects/PSBD-ViT"

# Median observed minutes at 15 epochs, from the args.json timestamps.
TRAIN_MINUTES = {"cifar100": 167, "tiny": 332}

# (dataset, target count, poison rate). Every rate here is one the widened set
# can actually deliver, checked against |target set| / |train set|.
CELLS = (
    ("cifar100", 2, 0.01),
    ("cifar100", 3, 0.01),
    ("tiny", 2, 0.01),
    ("tiny", 3, 0.01),
)

JOB_TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N {name}
#PBS -o {root}/logs/{batch}/{name}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"; echo "Node: $(hostname)"; echo "Started: $(date)"
cd {root}
source .venv/bin/activate

{body}
echo "Finished: $(date)"
exit 0
"""

TRAIN_CALL = """echo "=== {folder} ==="
python -m cli.train_backdoor \\
    --dataset {dataset} \\
    --attack sig \\
    --poison-rate {rate} \\
    --target-label 0 \\
    --architecture vit \\
    --epochs 15 \\
    --seed 0 \\
    --attack-override num_targets={num_targets} \\
    --output checkpoints/{folder}/attack_result.pt
"""


def rate_tag(rate):
    return f"{rate:g}".replace(".", "_")


def folder_name(dataset, num_targets, rate):
    """`_m{n}` marks the target-set size, and 1 target keeps the bare name.

    The tag has to avoid every substring the panel filters on, so it cannot reuse
    `a2m`, which already means the content-dependent all-to-m label map and is a
    different thing entirely.
    """
    name = f"vit_{dataset}_sig_{rate_tag(rate)}"
    if num_targets > 1:
        name += f"_m{num_targets}"
    return name


def build_runs():
    return [
        (
            TRAIN_MINUTES[dataset],
            TRAIN_CALL.format(
                folder=folder_name(dataset, num_targets, rate),
                dataset=dataset,
                rate=rate,
                num_targets=num_targets,
            ),
        )
        for dataset, num_targets, rate in CELLS
    ]


def pack(runs, budget):
    """Greedy bin-pack by predicted minutes, never splitting a run."""
    bundles, current, spent = [], [], 0
    for minutes, body in runs:
        if current and spent + minutes > budget:
            bundles.append((current, spent))
            current, spent = [], 0
        current.append(body)
        spent += minutes
    if current:
        bundles.append((current, spent))
    return bundles


def write_jobs(batch, bundles):
    job_dir = os.path.join("pbs", batch)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(os.path.join("logs", batch), exist_ok=True)
    submit = []
    for index, (bodies, spent) in enumerate(bundles, start=1):
        name = f"{batch}_{index}"
        # A job killed at the wall loses every run it had not written, so the
        # request carries 2x headroom over the prediction.
        hours = min(48, max(6, int(spent / 60 * 2.0) + 2))
        with open(os.path.join(job_dir, f"{name}.pbs"), "w") as handle:
            handle.write(
                JOB_TEMPLATE.format(
                    walltime=f"{hours}:00:00",
                    name=name,
                    root=PROJECT_ROOT,
                    batch=batch,
                    body="\n".join(bodies),
                )
            )
        submit.append(f"qsub {PROJECT_ROOT}/{job_dir}/{name}.pbs")
    path = os.path.join(job_dir, "submit_all.sh")
    with open(path, "w") as handle:
        handle.write("#!/bin/bash\n" + "\n".join(submit) + "\n")
    os.chmod(path, 0o755)
    print(
        f"\n[ok] pbs/{batch}/  {len(bundles)} jobs, {sum(s for _, s in bundles)} minutes"
    )
    print(f"submit with: bash pbs/{batch}/submit_all.sh")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes-per-job", type=int, default=400)
    parser.add_argument("--batch", default="vit_multitarget")
    args = parser.parse_args()

    for dataset, num_targets, rate in CELLS:
        print(
            f"  {folder_name(dataset, num_targets, rate)}: "
            f"{num_targets} target classes at {rate:.0%}"
        )
    write_jobs(args.batch, pack(build_runs(), args.minutes_per_job))


if __name__ == "__main__":
    main()
