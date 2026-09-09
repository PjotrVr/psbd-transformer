"""Sweep the deployed configurations on training-seed replicates, to put error bars on them.

This project's own audit records the gap: "Every checkpoint is seed 0 and no configuration has
a replicate, so seed to seed variance is unmeasured." Every confidence interval reported here
is over CELLS, not over training runs, so it describes how much the answer varies across
attacks and datasets and says nothing about how much it would move if the same cell were
trained again.

That is not a hypothetical concern. 26 cells do have replicates, and their attack success rates
move a lot: adaptive_blend at 10% reads 0.604, 0.949 and 0.974 at seeds 0, 1 and 2, and lc at
5% reads 0.545, 0.912 and 0.844. A verdict of "this attack does not implant" taken from seed 0
alone is a statement about one training run.

None of those replicates carries a detection cache, so detection seed-variance cannot be read
from disk at all. This sweeps the deployed single configuration and the published ConvNet
placement it is compared against, on every replicate, so the headline comparison finally has a
seed term.

    PYTHONPATH=. python pbs/generate_seed_replicate_jobs.py
"""

import argparse
import glob
import json
import os
import re

from defences.decision import complete_rates

PROJECT_ROOT = "/lustre/home/pstika/projects/PSBD-ViT"
PROBABILITY = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
POST_RESIDUAL = [
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.07,
    0.09,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]

TARGETS = [
    (
        "before_attention_norm_token_mask",
        "before_attention_norm",
        "token_mask",
        PROBABILITY,
    ),
    ("post_residual", "post_residual", "dropout", POST_RESIDUAL),
]

JOB = """#!/bin/bash
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
echo '=== analyze ==='
python -m cli.analyze --checkpoint-folder {folders}

echo "Finished: $(date)"
exit 0
"""


def replicate_families(asr_bar: float) -> list[str]:
    """Every checkpoint in a family that has at least one seed replicate.

    The base run is included alongside its replicates: a spread needs all of them, and the
    base is the one every existing number was computed from.
    """
    families = {}
    for path in glob.glob("checkpoints/*_seed_*/args.json"):
        folder = os.path.basename(os.path.dirname(path))
        match = re.match(r"(.+)_seed_(\d+)$", folder)
        if not match:
            continue
        families.setdefault(match.group(1), set()).add(folder)
    folders = []
    for base, replicates in families.items():
        members = sorted(replicates | {base})
        # Keep a family only if at least one member implanted; a family that never implants
        # measures the attack's seed sensitivity, not the detector's.
        implanted = False
        for member in members:
            meta_path = os.path.join("checkpoints", member, "args.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as handle:
                if (json.load(handle).get("asr") or 0) >= asr_bar:
                    implanted = True
        if implanted:
            folders += [
                m for m in members if os.path.isdir(os.path.join("checkpoints", m))
            ]
    return sorted(set(folders))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="vit_seedvar2")
    parser.add_argument("--asr-bar", type=float, default=0.85)
    parser.add_argument("--per-job", type=int, default=8)
    args = parser.parse_args()

    folders = replicate_families(args.asr_bar)
    gaps = {}
    for folder in folders:
        psbd = os.path.join("results", folder, "psbd")
        for name, position, operator, rates in TARGETS:
            have = set(complete_rates(psbd, name))
            need = tuple(r for r in rates if r not in have)
            if need:
                gaps.setdefault((position, operator, need), []).append(folder)

    units = []
    for (position, operator, rates), members in gaps.items():
        for start in range(0, len(members), args.per_job):
            units.append(
                (position, operator, rates, members[start : start + args.per_job])
            )

    out_dir = os.path.join("pbs", args.batch)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join("logs", args.batch), exist_ok=True)
    written = []
    for index, (position, operator, rates, members) in enumerate(units, start=1):
        name = f"seedvar_{index}"
        call = [
            "python -m cli.sweep \\",
            f"    --checkpoint-folder {' '.join(members)} \\",
            f"    --position-config {position} \\",
            f"    --perturbation {operator} \\",
            f"    --rates {' '.join(str(r) for r in rates)} \\",
            "    --forward-passes 3 \\",
            "    --skip-existing",
        ]
        hours = min(20, max(4, int(len(members) * len(rates) * 0.5 / 60 * 2.5) + 2))
        path = os.path.join(out_dir, f"{name}.pbs")
        with open(path, "w") as handle:
            handle.write(
                JOB.format(
                    walltime=f"{hours}:00:00",
                    name=name,
                    root=PROJECT_ROOT,
                    batch=args.batch,
                    body="\n".join(call) + "\n",
                    folders=" ".join(members),
                )
            )
        written.append(path)
    with open(os.path.join(out_dir, "submit_all.sh"), "w") as handle:
        handle.write("#!/bin/bash\n")
        for path in written:
            handle.write(f"qsub {os.path.join(PROJECT_ROOT, path)}\n")

    print(f"[ok] pbs/{args.batch}/")
    print(f"     checkpoints in replicate families : {len(folders)}")
    print(
        f"     missing (cell, placement) slots   : {sum(len(v) for v in gaps.values())}"
    )
    print(f"     jobs                              : {len(written)}")
    print(f"     submit  bash pbs/{args.batch}/submit_all.sh")


if __name__ == "__main__":
    main()
