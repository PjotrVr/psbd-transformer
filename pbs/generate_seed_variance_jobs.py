"""Re-sweep the 2 candidate configurations under additional defence seeds.

The PSBD probe is stochastic: dropout masks are drawn from a seeded generator, and every
number this project has reported comes from ONE draw, PSBD_MASK_SEED = 0. A single draw
says nothing about the estimator's spread, so a difference between 2 configurations cannot
currently be separated from seed noise.

This re-sweeps 2 configurations under seeds 1 and 2:

  before_attention_norm  + token_mask   the deployed configuration
  before_attention_residual + token_mask  the only combination in the top 5 at ALL 3
                                          poison rates, and 1st on the overall average

The second matters because a configuration that wins only at some poison rates is not
usable: a defender can guess the attack but can NEVER know the poison rate. pre_residual
blocks 5 to 8 is exactly that trap, 1st at 5% and 10% and 8th at 1%.

Scope is all-to-one and clean-label only, every dataset, all 3 rates, attacks that actually
implanted. Seed 0 already exists and keeps its bare cache directory.

    python pbs/generate_seed_variance_jobs.py && bash pbs/vit_seedvar/submit_all.sh
"""

import glob
import json
import os

BASE = "/lustre/home/pstika/projects/PSBD-ViT"
JOB_DIR = os.path.join(BASE, "pbs", "vit_seedvar")
LOG_DIR = os.path.join(BASE, "logs", "vit_seedvar")

CONFIGS = (
    ("before_attention_norm", "token_mask"),
    ("before_attention_residual", "token_mask"),
)
SEEDS = (1, 2)
N_JOBS = 10

HEADER = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N {name}
#PBS -o {log_dir}/{name}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"; echo "Node: $(hostname)"; echo "Started: $(date)"
cd {base}
source .venv/bin/activate
"""

SWEEP = """
echo "=== {folder} :: {position} {operator} seed {seed} ==="
python psbd_dropout_sweep.py \\
    --checkpoint-folder {folder} \\
    --position-config {position} \\
    --perturbation {operator} \\
    --mask-seed {seed} \\
    --rates 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \\
    --forward-passes 3
"""


def target_cells() -> list[str]:
    """All-to-one and clean-label cells whose attack actually implanted."""
    found = []
    for path in sorted(glob.glob(os.path.join(BASE, "results", "vit_*"))):
        folder = os.path.basename(path)
        if any(
            t in folder for t in ("sam_rho", "evade", "_ep", "a2m", "a2a", "benign")
        ):
            continue
        metrics = os.path.join(path, "psbd_metrics.json")
        if not os.path.exists(metrics):
            continue
        record = json.load(open(metrics))
        if record.get("label_mode") not in ("all_to_one", "clean_label"):
            continue
        args_path = os.path.join(BASE, "checkpoints", folder, "args.json")
        meta = json.load(open(args_path)) if os.path.exists(args_path) else {}
        asr = meta.get("asr")
        # A model with no working backdoor is not evidence about a detector.
        if not (isinstance(asr, float) and asr >= 0.5):
            continue
        found.append(folder)
    return found


def main() -> None:
    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    cells = target_cells()
    tasks = [
        (folder, position, operator, seed)
        for folder in cells
        for position, operator in CONFIGS
        for seed in SEEDS
        if not os.path.isdir(
            os.path.join(
                BASE, "results", folder, "psbd", f"{position}_{operator}_seed{seed}"
            )
        )
    ]
    written = []
    for index in range(N_JOBS):
        chunk = tasks[index::N_JOBS]
        if not chunk:
            continue
        name = f"seedvar_{index + 1}"
        body = "".join(
            SWEEP.format(folder=f, position=p, operator=o, seed=s)
            for f, p, o, s in chunk
        )
        folders = sorted({f for f, *_ in chunk})
        body += (
            "\necho '=== analyze ==='\npython psbd_analyze.py --checkpoint-folder "
            + " ".join(folders)
            + "\n"
        )
        path = os.path.join(JOB_DIR, f"{name}.pbs")
        with open(path, "w") as handle:
            handle.write(
                HEADER.format(
                    walltime="20:00:00", name=name, log_dir=LOG_DIR, base=BASE
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
    print(f"{len(cells)} cells, {len(tasks)} sweeps -> {len(written)} jobs")
    print(f"submit with: bash {submit}")


if __name__ == "__main__":
    main()
