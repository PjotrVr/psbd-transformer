"""Generate the PBS jobs for the prediction-depth panel.

The statistic is 1 forward pass per sample, but the logit lens reads every block over
every patch token, so a cell still costs a few minutes and the full panel is hours
serially. These jobs run it in parallel instead.

Every non-SAM, non-evade, non-snapshot ViT checkpoint is covered, at all 3 poison rates
plus the benign controls, because the open question is exactly whether the per-token
signal survives away from the patch trigger it was first measured on.

    python pbs/generate_prediction_depth_jobs.py && bash pbs/vit_depth/submit_all.sh
"""

import os
import re

BASE = "/lustre/home/pstika/projects/PSBD-ViT"
JOB_DIR = os.path.join(BASE, "pbs", "vit_depth")
LOG_DIR = os.path.join(BASE, "logs", "vit_depth")
N_JOBS = 8

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
export PYTHONPATH={base}
"""


def panel_folders() -> list[str]:
    """Every ViT cell the panel covers, excluding the arms reporting excludes anyway.

    SAM is excluded because every reporting path in this repo excludes it by default,
    and evade and _ep snapshots because they answer different questions.
    """
    folders = []
    for name in sorted(os.listdir(os.path.join(BASE, "checkpoints"))):
        if not name.startswith("vit_"):
            continue
        if "sam_rho" in name or "evade" in name or re.search(r"_ep\d+$", name):
            continue
        if not os.path.exists(
            os.path.join(BASE, "checkpoints", name, "attack_result.pt")
        ):
            continue
        folders.append(name)
    return folders


def main() -> None:
    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    folders = panel_folders()
    written = []
    for index in range(N_JOBS):
        chunk = folders[index::N_JOBS]
        if not chunk:
            continue
        name = f"depth_{index + 1}"
        body = (
            "\necho '=== prediction depth ==='\n"
            "python experiments/prediction_depth/measure.py \\\n"
            "    --checkpoint-folder " + " ".join(chunk) + "\n"
        )
        path = os.path.join(JOB_DIR, f"{name}.pbs")
        with open(path, "w") as handle:
            handle.write(
                HEADER.format(
                    walltime="10:00:00", name=name, log_dir=LOG_DIR, base=BASE
                )
            )
            handle.write(body)
            handle.write('\necho "Finished: $(date)"\nexit 0\n')
        written.append(path)

    submit = os.path.join(JOB_DIR, "submit_all.sh")
    with open(submit, "w") as handle:
        handle.write("#!/bin/bash\n")
        for path in written:
            handle.write(f"qsub {path}\n")
    os.chmod(submit, 0o755)
    print(f"{len(folders)} cells -> {len(written)} jobs")
    print(f"submit with: bash {submit}")


if __name__ == "__main__":
    main()
