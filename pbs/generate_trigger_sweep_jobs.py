"""Trigger dose-response sweeps: the causal test, and the WaNet fix.

Two sweeps, one per trigger family, both varying trigger STRENGTH while holding attack
family, dataset, target label, poison rate and schedule fixed.

BADNET PATCH SIZE is the additive-trigger arm. The mechanism under test says token masking
beats channel masking because it removes a spatially concentrated trigger, and the evidence
so far is a correlation across attacks whose footprint is constant within an attack, so it is
really 8 points, not 67. Varying patch size inside ONE attack manipulates the cause instead
of observing it.

WANET STRENGTH is the geometric-trigger arm and also fixes an attack that does not implant.
WaNet reads ASR 0.057 to 0.178 at 1% poisoning. The implementation is faithful: the warp
displacement is `strength / 2` pixels at the size the attack is built at, about 0.17 px or
0.54% of width at the paper's s=0.5 on a 32 px image. It is simply too subtle to learn from
few poisoned samples. Note Tiny is built at 64 px, so the SAME strength gives half the
relative warp (0.27%), which is why Tiny needs the highest rates.

Nothing existing is overwritten: every run writes to its own folder tagged with the override.

    PYTHONPATH=. python pbs/generate_trigger_sweep_jobs.py --sweep wanet badnet
"""

import argparse
import os

PROJECT_ROOT = "/lustre/home/pstika/projects/PSBD-ViT"
TRAIN_MINUTES = {"gtsrb": 90, "cifar10": 167, "cifar100": 167, "tiny": 332}

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

TRAIN_CALL = """echo "=== {output} ==="
python train_backdoor.py \\
    --dataset {dataset} \\
    --attack {attack} \\
    --poison-rate {rate} \\
    --architecture vit \\
    --epochs 15 \\
    --seed 0 \\
    --attack-override {override} \\
    --output checkpoints/{output}
"""


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def wanet_runs():
    """Strength sweep where WaNet fails, plus the one 5% cell under the bar."""
    runs = []
    for dataset in ("cifar10", "cifar100", "gtsrb", "tiny"):
        for strength in (1.0, 2.0, 4.0):
            runs.append(
                {
                    "attack": "wanet",
                    "dataset": dataset,
                    "rate": 0.01,
                    "override": f"strength={strength}",
                    "output": f"vit_{dataset}_wanet_0_01_trig_s{tag(strength)}",
                }
            )
    for strength in (2.0, 4.0):
        runs.append(
            {
                "attack": "wanet",
                "dataset": "cifar100",
                "rate": 0.05,
                "override": f"strength={strength}",
                "output": f"vit_cifar100_wanet_0_05_trig_s{tag(strength)}",
            }
        )
    return runs


def badnet_runs():
    """Patch-size dose-response at a rate where BadNet reliably implants.

    Sizes are in pixels at the dataset's native resolution (32 for gtsrb and cifar100), so
    they span roughly 1 to 49 of the 196 tokens once upsampled to 224.
    """
    return [
        {
            "attack": "badnet_a2o",
            "dataset": dataset,
            "rate": 0.05,
            "override": f"patch_size={size}",
            "output": f"vit_{dataset}_badnet_a2o_0_05_trig_p{size}",
        }
        for dataset in ("gtsrb", "cifar100")
        for size in (2, 3, 5, 8, 12, 16)
    ]


def write_jobs(runs, args):
    out_dir = os.path.join("pbs", args.batch)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join("logs", args.batch), exist_ok=True)

    bundles, current, spent = [], [], 0.0
    for run in runs:
        cost = TRAIN_MINUTES.get(run["dataset"], 200)
        if current and spent + cost > args.minutes_per_job:
            bundles.append(current)
            current, spent = [], 0.0
        current.append(run)
        spent += cost
    if current:
        bundles.append(current)

    written = []
    for index, bundle in enumerate(bundles, start=1):
        name = f"trigger_{index}"
        minutes = sum(TRAIN_MINUTES.get(run["dataset"], 200) for run in bundle)
        hours = min(48, max(6, int(minutes / 60 * 2.0) + 2))
        body = "\n".join(TRAIN_CALL.format(**run) for run in bundle)
        path = os.path.join(out_dir, f"{name}.pbs")
        with open(path, "w") as handle:
            handle.write(
                JOB_TEMPLATE.format(
                    walltime=f"{hours}:00:00",
                    name=name,
                    root=PROJECT_ROOT,
                    batch=args.batch,
                    body=body,
                )
            )
        written.append(path)

    with open(os.path.join(out_dir, "submit_all.sh"), "w") as handle:
        handle.write("#!/bin/bash\n")
        for path in written:
            handle.write(f"qsub {os.path.join(PROJECT_ROOT, path)}\n")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep", nargs="+", choices=("wanet", "badnet"), required=True
    )
    parser.add_argument("--batch", default="vit_trigger")
    parser.add_argument("--minutes-per-job", type=float, default=200.0)
    args = parser.parse_args()

    runs = []
    if "wanet" in args.sweep:
        runs += wanet_runs()
    if "badnet" in args.sweep:
        runs += badnet_runs()
    existing = [
        r for r in runs if os.path.isdir(os.path.join("checkpoints", r["output"]))
    ]
    runs = [r for r in runs if r not in existing]

    written = write_jobs(runs, args)
    print(f"[ok] pbs/{args.batch}/")
    print(f"     runs        {len(runs)}  (skipped {len(existing)} already trained)")
    print(f"     jobs        {len(written)}")
    print(
        f"     est GPU     {sum(TRAIN_MINUTES.get(r['dataset'], 200) for r in runs) / 60:.1f} hours"
    )
    for run in runs:
        print(f"       {run['output']:44s} {run['override']}")
    print(f"     submit      bash pbs/{args.batch}/submit_all.sh")


if __name__ == "__main__":
    main()
