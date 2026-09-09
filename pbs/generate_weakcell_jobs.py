"""Sweep the placements that actually win on hard attacks, over the cells that lack them.

PSBD-ViT deploys token_mask @ before_attention_norm. Ranked over the hard attacks
(adaptive_blend, LC, SIG, WaNet) on every cell that already carries a full grid, that
configuration comes **5th**:

    before_mlp_gaussian                     0.866
    both_sublayer_inputs_token_mask         0.865
    before_attention_residual_token_mask    0.861
    before_mlp_norm_token_mask              0.860
    before_attention_norm_token_mask        0.830   <- deployed

The weak cells cannot benefit from that, because the adaptive_blend, wanet and bpp cells
were retrained on 2026-09-08 and their caches were cleared, so they carry exactly 1
placement. Their "best over placements" is therefore the deployed one by construction, and
the question of whether a better placement exists for them has never been asked.

This sweeps the 4 leaders over those cells. No --skip-existing: the point is the missing
configurations.

    python pbs/generate_weakcell_jobs.py && bash pbs/vit_weakcell/submit_all.sh
"""

import glob
import json
import os

BASE = "/lustre/home/pstika/projects/PSBD-ViT"
JOB_DIR = os.path.join(BASE, "pbs", "vit_weakcell")
LOG_DIR = os.path.join(BASE, "logs", "vit_weakcell")

# (position, operator, cache directory name)
PLACEMENTS = (
    ("before_mlp", "gaussian", "before_mlp_gaussian"),
    ("both_sublayer_inputs", "token_mask", "both_sublayer_inputs_token_mask"),
    ("before_attention_residual", "token_mask", "before_attention_residual_token_mask"),
    ("before_mlp_norm", "token_mask", "before_mlp_norm_token_mask"),
)
WEAK_ATTACKS = ("adaptive_blend", "wanet", "lc", "sig", "bpp")
N_JOBS = 6

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
echo "=== {folder} :: {cache} ==="
python -m cli.sweep \\
    --checkpoint-folder {folder} \\
    --position-config {position} \\
    --perturbation {operator} \\
    --rates 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \\
    --forward-passes 3
"""


def weak_cells() -> list[str]:
    """Cells on a weak attack whose cache is missing at least 1 of the 4 leaders."""
    found = []
    for path in sorted(glob.glob(os.path.join(BASE, "results", "vit_*"))):
        folder = os.path.basename(path)
        if any(token in folder for token in ("sam_rho", "evade", "_ep", "a2m", "a2a")):
            continue
        metrics = os.path.join(path, "psbd_metrics.json")
        if not os.path.exists(metrics):
            continue
        record = json.load(open(metrics))
        if record.get("attack") not in WEAK_ATTACKS:
            continue
        args_path = os.path.join(BASE, "checkpoints", folder, "args.json")
        meta = json.load(open(args_path)) if os.path.exists(args_path) else {}
        asr = meta.get("asr")
        # An attack that never implanted is not evidence either way, so it is not swept.
        if not (isinstance(asr, float) and asr >= 0.5):
            continue
        missing = [
            p for p in PLACEMENTS if not os.path.isdir(os.path.join(path, "psbd", p[2]))
        ]
        if missing:
            found.append((folder, missing))
    return found


def main() -> None:
    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    work = weak_cells()
    tasks = [(folder, p) for folder, missing in work for p in missing]
    written = []
    for index in range(N_JOBS):
        chunk = tasks[index::N_JOBS]
        if not chunk:
            continue
        name = f"weak_{index + 1}"
        body = "".join(
            SWEEP.format(folder=f, position=p[0], operator=p[1], cache=p[2])
            for f, p in chunk
        )
        folders = sorted({f for f, _ in chunk})
        body += (
            "\necho '=== analyze ==='\npython -m cli.analyze --checkpoint-folder "
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
    print(f"{len(work)} weak cells, {len(tasks)} missing sweeps -> {len(written)} jobs")
    print(f"submit with: bash {submit}")


if __name__ == "__main__":
    main()
