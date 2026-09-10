"""Generate the PBS jobs that fill the token_mask @ before_attention_norm panel.

Two kinds of job.

The gap-fill jobs sweep the (checkpoint, placement) pairs that were never run.
Which pairs those are is discovered here rather than hard-coded, by checking for
the raw cache directory, so re-running the generator after a partial batch emits
only what is still outstanding.

The sig jobs train one CIFAR-10 SIG model with dense checkpointing and then sweep
its snapshots. The sweeps are split across several jobs and submitted with a
dependency on the training job, so they run in parallel once training finishes
instead of serialising behind it.

    python pbs/generate_gapfill_jobs.py && bash pbs/vit_gapfill/submit_all.sh
"""

import os

BASE = "/lustre/home/pstika/projects/PSBD-ViT"
JOB_DIR = os.path.join(BASE, "pbs", "vit_gapfill")
LOG_DIR = os.path.join(BASE, "logs", "vit_gapfill")

DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
ATTACKS = (
    "badnet_a2o",
    "badnet_a2a",
    "blend",
    "wanet",
    "adaptive_blend",
    "lc",
    "sig",
    "lf",
    "bpp",
    "tact",
)
RATE_TAGS = {0.01: "0_01", 0.05: "0_05", 0.10: "0_1"}
PLACEMENT = "before_attention_norm_token_mask"

# Tiny has twice the training images, so its cells run roughly twice as long and
# get spread over more jobs to keep every job in the same 1-to-1.5 hour band.
JOBS_PER_DATASET = {"cifar10": 2, "cifar100": 2, "gtsrb": 3, "tiny": 4}

SIG_FOLDER = "vit_cifar10_sig_0_1"
SIG_EPOCHS = 40
SIG_DENSE_UNTIL = 20
SIG_FREQ = 5
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

SWEEP = """
echo "=== sweep {folder} ==="
python -m cli.sweep \\
    --checkpoint-folder {folder} \\
    --position-config before_attention_norm \\
    --perturbation token_mask \\
    --rates 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \\
    --forward-passes 3 --skip-existing
"""


def snapshot_epochs(total, dense_until, freq):
    chosen = set(range(1, min(dense_until, total) + 1))
    chosen |= {e for e in range(1, total + 1) if e % freq == 0 and e > dense_until}
    return sorted(chosen)


def missing_cells():
    """Cells whose raw sweep cache for the target placement does not exist yet."""
    missing = []
    for rate, tag in RATE_TAGS.items():
        for dataset in DATASETS:
            for attack in ATTACKS:
                folder = f"vit_{dataset}_{attack}_{tag}"
                if not os.path.isdir(os.path.join(BASE, "checkpoints", folder)):
                    continue
                cache = os.path.join(BASE, "results", folder, "psbd", PLACEMENT)
                if not os.path.isdir(cache):
                    missing.append((dataset, folder))
    return missing


def write_job(name, walltime, body):
    path = os.path.join(JOB_DIR, f"{name}.pbs")
    with open(path, "w") as handle:
        handle.write(
            HEADER.format(walltime=walltime, name=name, log_dir=LOG_DIR, base=BASE)
        )
        handle.write(body)
        handle.write('\necho "Finished: $(date)"\nexit 0\n')
    return path


def write_sweep_job(name, folders, walltime):
    body = "".join(SWEEP.format(folder=f) for f in folders)
    body += (
        "\necho '=== analyze ==='\npython -m cli.analyze --checkpoint-folder "
        + " ".join(folders)
        + "\n"
    )
    return write_job(name, walltime, body)


def main():
    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    gaps = missing_cells()
    by_dataset = {d: [f for ds, f in gaps if ds == d] for d in DATASETS}
    written = []

    for dataset, folders in by_dataset.items():
        if not folders:
            continue
        n = min(JOBS_PER_DATASET[dataset], len(folders))
        chunks = [folders[i::n] for i in range(n)]
        for index, chunk in enumerate(chunks, start=1):
            if not chunk:
                continue
            written.append(write_sweep_job(f"gap_{dataset}_{index}", chunk, "06:00:00"))

    # The sig trajectory run. Walltime is generous on purpose: 40 epochs is ~3 h
    # and the 24 snapshots add their own train-accuracy passes on top.
    train_body = f"""
echo "=== train {SIG_FOLDER} for {SIG_EPOCHS} epochs, dense checkpointing ==="
python -m cli.train_backdoor \\
    --dataset cifar10 --attack sig --poison-rate 0.10 \\
    --target-label 0 --architecture vit --epochs {SIG_EPOCHS} --seed 0 \\
    --checkpoint-dense-until {SIG_DENSE_UNTIL} --checkpoint-freq {SIG_FREQ} \\
    --output checkpoints/{SIG_FOLDER}_ep{SIG_EPOCHS}run/attack_result.pt
"""
    written.append(write_job("sig_train", "10:00:00", train_body))

    epochs = snapshot_epochs(SIG_EPOCHS, SIG_DENSE_UNTIL, SIG_FREQ)
    snaps = [f"{SIG_FOLDER}_ep{SIG_EPOCHS}run_ep{e:02d}" for e in epochs]
    for index in range(1, SIG_SWEEP_JOBS + 1):
        chunk = snaps[index - 1 :: SIG_SWEEP_JOBS]
        if chunk:
            written.append(write_sweep_job(f"sig_sweep_{index}", chunk, "06:00:00"))

    submit = os.path.join(JOB_DIR, "submit_all.sh")
    with open(submit, "w") as handle:
        handle.write("#!/bin/bash\n")
        handle.write("# Gap-fill jobs are independent and go straight in.\n")
        for path in written:
            name = os.path.basename(path)
            if name.startswith("gap_"):
                handle.write(f"qsub {path}\n")
        handle.write(
            "\n# The sig sweeps cannot start before the model exists, so they\n"
        )
        handle.write("# carry an afterok dependency and then run in parallel.\n")
        handle.write(f"SIG=$(qsub {os.path.join(JOB_DIR, 'sig_train.pbs')})\n")
        handle.write('echo "sig_train -> $SIG"\n')
        for index in range(1, SIG_SWEEP_JOBS + 1):
            p = os.path.join(JOB_DIR, f"sig_sweep_{index}.pbs")
            if os.path.exists(p):
                handle.write(f"qsub -W depend=afterok:$SIG {p}\n")
    os.chmod(submit, 0o755)

    print(
        f"{len(gaps)} gap cells -> {len([p for p in written if 'gap_' in p])} gap jobs"
    )
    print(f"sig: {len(epochs)} snapshots {epochs} -> {SIG_SWEEP_JOBS} sweep jobs")
    print(f"{len(written)} jobs in {JOB_DIR}")
    print(f"submit with: bash {submit}")


if __name__ == "__main__":
    main()
