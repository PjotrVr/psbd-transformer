"""Generate the PBS jobs for the rebuilt attacks, the TaCT re-sweep, and the sig sweeps.

Three kinds of work, all independent:

  retrain  wanet, adaptive_blend and bpp were missing the mechanism that makes them
           stealthy (noise mode, correctly scaled cover, negative samples), so their
           checkpoints do not represent the published attacks and are retrained.

  tact     needs no retraining: the attack is fine, but its cached backdoor split was
           every non-target image when only the source class is ever flipped, so both
           its ASR and its detection numbers were measured over a population that is
           mostly not backdoored. Re-sweeping with the corrected split fixes both.

  sig      the 24 trajectory snapshots whose sweeps never ran, because they were
           submitted without a dependency and exited on their wait guard.

    python pbs/generate_rebuild_jobs.py && bash pbs/vit_rebuild/submit_all.sh
"""

import os

BASE = "/lustre/home/pstika/projects/PSBD-ViT"
JOB_DIR = os.path.join(BASE, "pbs", "vit_rebuild")
LOG_DIR = os.path.join(BASE, "logs", "vit_rebuild")

RETRAIN_ATTACKS = ("wanet", "adaptive_blend", "bpp")
DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
RATES = ((0.10, "0_1"), (0.05, "0_05"), (0.01, "0_01"))

# Tiny has twice the images, so its cells run roughly twice as long and are spread
# over more jobs to keep every job in the same band.
JOBS_PER_DATASET = {"cifar10": 3, "cifar100": 3, "gtsrb": 2, "tiny": 4}

SIG_RUN = "vit_cifar10_sig_0_1_ep40run"
SIG_EPOCHS = list(range(1, 21)) + [25, 30, 35, 40]
SIG_SWEEP_JOBS = 6

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
python -m cli.train_backdoor \\
    --dataset {dataset} --attack {attack} --poison-rate {rate} \\
    --target-label 0 --architecture vit --epochs 15 --seed 0 \\
    --output checkpoints/{folder}/attack_result.pt
"""

# No --skip-existing anywhere in this file. Every cell here either has a NEW
# checkpoint (retrain) or a stale cache built from the wrong split (tact), so a
# skip would silently analyse the old sweep against the new model. That is the same
# failure mode as the sig sweeps, and it produces plausible numbers rather than an
# error, which is what makes it dangerous.
#
# The whole psbd/ subtree goes, not just the placement directory. Removing only the
# placement leaves baseline_<split>.pt behind, and load_or_build_baseline reuses any
# baseline whose row count matches, which it always does. PSU would then be the OLD
# model's confidence minus the NEW model's dropout passes. It also leaves every other
# placement's tensors behind, and psbd_analyze reads every subdirectory, so one
# psbd_metrics.json would mix 2 different models under 1 checkpoint name. Both are
# silent.
SWEEP = """
echo "=== sweep {folder} ==="
rm -rf results/{folder}/psbd
python -m cli.sweep \\
    --checkpoint-folder {folder} \\
    --position-config before_attention_norm \\
    --perturbation token_mask \\
    --rates 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \\
    --forward-passes 3
"""


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
        "\necho '=== analyze ==='\npython -m cli.analyze --checkpoint-folder "
        + " ".join(folders)
        + "\n"
    )


def main():
    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    written = []

    # Retrain: train then sweep each cell in the same job, so a cell is never swept
    # against a checkpoint that does not exist yet. That failure is what left the sig
    # sweeps empty last time.
    for dataset in DATASETS:
        cells = [
            (attack, rate, f"vit_{dataset}_{attack}_{tag}")
            for attack in RETRAIN_ATTACKS
            for rate, tag in RATES
        ]
        n = JOBS_PER_DATASET[dataset]
        for index in range(n):
            chunk = cells[index::n]
            if not chunk:
                continue
            body = "".join(
                TRAIN.format(dataset=dataset, attack=a, rate=r, folder=f)
                + SWEEP.format(folder=f)
                for a, r, f in chunk
            )
            body += analyze([f for _, _, f in chunk])
            written.append(
                write_job(f"rebuild_{dataset}_{index + 1}", "20:00:00", body)
            )

    # TaCT: re-sweep only. The cached split is stale, so --skip-existing would keep it.
    tact = [f"vit_{d}_tact_{tag}" for d in DATASETS for _, tag in RATES]
    for index in range(2):
        chunk = tact[index::2]
        body = "".join(SWEEP.format(folder=f) for f in chunk)
        body += analyze(chunk)
        written.append(write_job(f"tact_resweep_{index + 1}", "08:00:00", body))

    # Sig trajectory sweeps: the checkpoints already exist, so no wait guard is needed.
    snaps = [f"{SIG_RUN}_ep{e:02d}" for e in SIG_EPOCHS]
    for index in range(SIG_SWEEP_JOBS):
        chunk = snaps[index::SIG_SWEEP_JOBS]
        body = "".join(SWEEP.format(folder=f) for f in chunk) + analyze(chunk)
        written.append(write_job(f"sig_traj_{index + 1}", "06:00:00", body))

    submit = os.path.join(JOB_DIR, "submit_all.sh")
    with open(submit, "w") as handle:
        handle.write(
            "#!/bin/bash\n# All jobs are independent; each retrain job sweeps its\n"
            "# own cells, so nothing can sweep a checkpoint that does not exist.\n"
        )
        for path in written:
            handle.write(f"qsub {path}\n")
    os.chmod(submit, 0o755)

    retrain = len([p for p in written if "rebuild_" in p])
    print(
        f"{len(RETRAIN_ATTACKS) * len(DATASETS) * len(RATES)} cells to retrain -> {retrain} jobs"
    )
    print(f"{len(tact)} tact cells to re-sweep -> 2 jobs")
    print(f"{len(snaps)} sig snapshots -> {SIG_SWEEP_JOBS} jobs")
    print(f"{len(written)} jobs in {JOB_DIR}")
    print(f"submit with: bash {submit}")


if __name__ == "__main__":
    main()
