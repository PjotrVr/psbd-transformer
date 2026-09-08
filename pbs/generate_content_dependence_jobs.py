"""Generate the PBS jobs for the content-dependence law.

PSBD's premise is that a trigger is a CONSTANT, content-independent shortcut. The
perturbed prediction stays pinned because the shortcut never has to read the image,
while clean predictions lose their features and drift. all_to_one satisfies that
premise exactly. all_to_all violates it exactly: (y + 1) mod K forces the model to
recognise the source class before it can increment, so the backdoor pathway inherits
and then exceeds the clean pathway's fragility, and the detector inverts.

Those are 2 points. This sweep fills in the line between them.

The all_to_m family maps a poisoned sample to (y + 1) mod m, so m is the number of
distinct classes the trigger lands on and the backdoor map must encode log2(m) bits
about the image. m = 1 reproduces all_to_one on target 0 exactly and m = num_classes
reproduces all_to_all exactly, so both poles are already trained and are read from
the existing badnet_a2o and badnet_a2a checkpoints rather than duplicated here. Only
the interior is new.

Prediction under test: PSBD's AUROC falls monotonically in log2(m), crossing chance
somewhere strictly inside the range. If it does, "PSBD fails on all-to-all" stops
being a special case and becomes a law about content dependence, and the crossing
point is the minimum content dependence an attacker needs to buy immunity.

Powers of 2 because the axis is log2(m). m is capped at each dataset's class count,
since m > num_classes emits an out-of-range label.

    python pbs/generate_content_dependence_jobs.py && bash pbs/vit_content/submit_all.sh
"""

import os

BASE = "/lustre/home/pstika/projects/PSBD-ViT"
JOB_DIR = os.path.join(BASE, "pbs", "vit_content")
LOG_DIR = os.path.join(BASE, "logs", "vit_content")

REGISTERED_M = (2, 4, 8, 16, 32, 64, 128)
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "gtsrb": 43, "tiny": 200}

# CIFAR-100 and Tiny are the primary panel and carry both rates. CIFAR-10 and GTSRB
# are completion only, so they get the rate where both poles reliably implant.
RATES = {
    "cifar100": ((0.10, "0_1"), (0.05, "0_05")),
    "tiny": ((0.10, "0_1"), (0.05, "0_05")),
    "cifar10": ((0.10, "0_1"),),
    "gtsrb": ((0.10, "0_1"),),
}

# Tiny has twice the images so its cells run roughly twice as long; jobs are sized to
# keep every one inside the same wall-time band rather than to equalise cell counts.
JOBS_PER_DATASET = {"cifar10": 1, "cifar100": 3, "gtsrb": 1, "tiny": 4}

HEADER = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N {name}
#PBS -o {log_dir}/{name}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

cd {base}
source .venv/bin/activate
"""

TRAIN = """
echo "=== train {folder} ==="
python train_backdoor.py \\
    --dataset {dataset} --attack {attack} --poison-rate {rate} \\
    --target-label 0 --architecture vit --epochs 15 --seed 0 \\
    --output checkpoints/{folder}/attack_result.pt
"""

# No --skip-existing. Every cell here is a brand new checkpoint, so there is nothing
# legitimate to skip, and a skip would analyse an absent or unrelated cache. The
# subtree is removed anyway, because a resubmitted job must not inherit the cache its
# own earlier attempt left: a baseline is reused whenever its row count matches, which
# it always does, so a half-finished attempt would be read as this model's baseline.
SWEEP = """
echo "=== sweep {folder} ==="
rm -rf results/{folder}/psbd
python psbd_dropout_sweep.py \\
    --checkpoint-folder {folder} \\
    --position-config before_attention_norm \\
    --perturbation token_mask \\
    --rates 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \\
    --forward-passes 3
"""


def m_values(dataset):
    """Registered m strictly below the class count; m = num_classes is all_to_all."""
    return [m for m in REGISTERED_M if m < NUM_CLASSES[dataset]]


def cells(dataset):
    return [
        (f"badnet_a2m{m}", rate, f"vit_{dataset}_badnet_a2m{m}_{tag}")
        for rate, tag in RATES[dataset]
        for m in m_values(dataset)
    ]


def write_job(name, walltime, body):
    path = os.path.join(JOB_DIR, f"{name}.pbs")
    with open(path, "w") as handle:
        handle.write(
            HEADER.format(walltime=walltime, name=name, log_dir=LOG_DIR, base=BASE)
        )
        handle.write(body)
        handle.write('\necho "Finished: $(date)"\nexit 0\n')
    return path


def analyze(folders):
    return (
        "\necho '=== analyze ==='\npython psbd_analyze.py --checkpoint-folder "
        + " ".join(folders)
        + "\n"
    )


def main():
    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    written = []
    total = 0

    for dataset in ("cifar100", "tiny", "cifar10", "gtsrb"):
        todo = [c for c in cells(dataset) if not os.path.isdir(f"checkpoints/{c[2]}")]
        total += len(todo)
        n = JOBS_PER_DATASET[dataset]
        for index in range(n):
            chunk = todo[index::n]
            if not chunk:
                continue
            body = "".join(
                TRAIN.format(dataset=dataset, attack=a, rate=r, folder=f)
                + SWEEP.format(folder=f)
                for a, r, f in chunk
            )
            body += analyze([f for _, _, f in chunk])
            written.append(
                write_job(f"content_{dataset}_{index + 1}", "20:00:00", body)
            )

    submit = os.path.join(JOB_DIR, "submit_all.sh")
    with open(submit, "w") as handle:
        handle.write("#!/bin/bash\n")
        for path in written:
            handle.write(f"qsub {path}\n")
    os.chmod(submit, 0o755)

    print(f"{total} new cells -> {len(written)} jobs")
    for dataset in ("cifar100", "tiny", "cifar10", "gtsrb"):
        print(
            f"  {dataset:9s} m in {m_values(dataset)}  rates {[t for _, t in RATES[dataset]]}"
        )
    print(f"submit with: bash {submit}")


if __name__ == "__main__":
    main()
