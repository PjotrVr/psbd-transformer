"""Emit retraining jobs for the seed replication described in docs/seed-replication-plan.md.

Every checkpoint in the project was trained at seed 0 and no configuration has a
replicate, so seed to seed variance is unmeasured and no error bar is currently
honest. This regenerates a chosen tier at additional seeds.

Arguments are read back from each checkpoint's own args.json rather than restated
here, so a rerun cannot silently differ from the original in a way nobody notices.
The only field that changes is the seed, and the output folder gains a seed tag so
the new run never overwrites the seed 0 run.

Reads every checkpoints/<folder>/args.json under --checkpoints-dir and writes 1
seed_NNN.pbs per wall clock budget into --out-dir (pbs/psbd_seed by default), with
logs under logs/psbd_seed.

Example
    python pbs/generate_seed_jobs.py --tier 1 --seeds 1 2 --dry-run
    python pbs/generate_seed_jobs.py --tier 1 --seeds 1 2
"""

import argparse
import glob
import json
import os

BASE = "/lustre/home/pstika/projects/PSBD-ViT"

PANEL = ("badnet_a2o", "blend", "wanet", "lc", "adaptive_blend")
MAIN_RATES = (0.01, 0.05, 0.1)
PRIMARY_DATASETS = ("cifar100", "tiny")

# Median wall clock per plain Adam training run, from the trained_started_at to
# trained_ended_at spans in args.json with SAM and evasion runs excluded (both
# cost 2 to 4 times a plain run and had doubled an earlier version of this table).
MEDIAN_MINUTES = {
    ("vit", "cifar10"): 84,
    ("vit", "cifar100"): 84,
    ("vit", "gtsrb"): 46,
    ("vit", "tiny"): 167,
    ("vit", "svhn"): 124,
    ("vit", "eurosat"): 37,
    ("swin", "cifar10"): 65,
    ("swin", "cifar100"): 65,
    ("swin", "gtsrb"): 36,
    ("swin", "tiny"): 129,
}
HARD_ATTACKS = ("bpp", "wanet", "tact", "sig", "lc", "adaptive_blend")
DATASET_PRIORITY = ("cifar100", "tiny", "gtsrb", "cifar10", "svhn", "eurosat")
SEEDS_WANTED = 3

TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N psbd_seed_{index:03d}
#PBS -o {base}/logs/psbd_seed/seed_{index:03d}.log
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

{commands}
echo "Finished: $(date)"
exit 0
"""

TRAIN = """python -m cli.train_backdoor \\
    --dataset {dataset} \\
    --attack {attack} \\
    --poison-rate {poison_rate} \\
    --target-label {target_label} \\
    --architecture {architecture} \\
    --epochs {epochs} \\
    --seed {seed} \\
    --output checkpoints/{folder}/attack_result.pt
"""

TRAIN_BENIGN = """python -m cli.train_benign \\
    --datasets {dataset} \\
    --architecture {architecture} \\
    --epochs {epochs} \\
    --seed {seed} \\
    --output checkpoints/{folder}/attack_result.pt
"""


def tier_of(metadata: dict) -> int | None:
    """Which replication tier this checkpoint belongs to, or None if excluded.

    Adaptive attacker and SAM checkpoints are excluded on purpose. The evasion
    claim rests on a 0.952 to 0.322 collapse, which no plausible seed noise
    threatens. The SAM effect is +0.009 AUROC, which no affordable number of
    seeds could establish.
    """
    if metadata.get("evasion") or metadata.get("model_dropout"):
        return None
    if metadata["optimizer"] != "adam":
        return None

    attack = metadata["attack"]
    if attack != "benign":
        if attack not in PANEL or metadata["poison_rate"] not in MAIN_RATES:
            return None
        asr = metadata.get("asr")
        if asr is None or asr < 0.5:
            return None

    architecture, dataset = metadata["architecture"], metadata["dataset"]
    if architecture == "vit" and dataset in PRIMARY_DATASETS:
        return 1
    if architecture == "vit":
        return 2
    return 3


def ledger_runs(coverage_path: str, checkpoints_dir: str) -> list[tuple[dict, int]]:
    """(metadata, seed) for every seed a clearing panel cell still lacks, hardest first.

    Order: hard attacks before easy, lower poison rate first, primary datasets
    first. A cell counts a seed as present when checkpoints/<folder>_seed_<n>/
    args.json exists with a recorded attack success rate.
    """
    with open(coverage_path) as handle:
        cells = [
            c for c in json.load(handle)["cells"] if c.get("asr_class") == "clears"
        ]

    def priority(cell: dict) -> tuple:
        hard = 0 if cell["attack"] in HARD_ATTACKS else 1
        dataset = (
            DATASET_PRIORITY.index(cell["dataset"])
            if cell["dataset"] in DATASET_PRIORITY
            else 9
        )
        return (hard, cell["poison_rate"], dataset, cell["attack"])

    runs = []
    for cell in sorted(cells, key=priority):
        folder = cell["folder_name"]
        args_path = os.path.join(checkpoints_dir, folder, "args.json")
        if not os.path.exists(args_path):
            continue
        with open(args_path) as handle:
            metadata = json.load(handle)
        metadata["folder"] = folder
        present = [0]
        for seed in range(1, 6):
            replicate = os.path.join(
                checkpoints_dir, seeded_folder(folder, seed), "args.json"
            )
            if os.path.exists(replicate):
                with open(replicate) as handle:
                    if json.load(handle).get("asr") is not None:
                        present.append(seed)
        missing = [seed for seed in range(1, 6) if seed not in present][
            : max(0, SEEDS_WANTED - len(present))
        ]
        runs += [(metadata, seed) for seed in missing]
    return runs


def seeded_folder(folder: str, seed: int) -> str:
    """The seed 0 folder name plus a seed tag, so no run overwrites another."""
    tagged = f"{folder}_seed_{seed}"
    return tagged


def discover(tier: int, checkpoints_dir: str) -> list[dict]:
    """The args.json records of every checkpoint in the tier, each tagged with its folder."""
    selected = []
    for path in sorted(glob.glob(os.path.join(checkpoints_dir, "*", "args.json"))):
        with open(path) as handle:
            metadata = json.load(handle)
        if tier_of(metadata) != tier:
            continue
        metadata["folder"] = os.path.basename(os.path.dirname(path))
        selected.append(metadata)
    return selected


def command_for(metadata: dict, seed: int) -> str:
    """The training command that reruns a checkpoint's recorded arguments at a new seed."""
    folder = seeded_folder(metadata["folder"], seed)
    shared = {
        "dataset": metadata["dataset"],
        "architecture": metadata["architecture"],
        "epochs": metadata["epochs"],
        "seed": seed,
        "folder": folder,
    }
    if metadata["attack"] == "benign":
        command = TRAIN_BENIGN.format(**shared)
        return command
    command = TRAIN.format(
        attack=metadata["attack"],
        poison_rate=metadata["poison_rate"],
        target_label=metadata["target_label"],
        **shared,
    )
    return command


def pack(runs: list[tuple[dict, int]], target_minutes: float) -> list[list]:
    """Fill jobs to a wall clock budget, never splitting a single training run."""
    jobs, current, minutes = [], [], 0.0
    for metadata, seed in runs:
        cost = MEDIAN_MINUTES.get((metadata["architecture"], metadata["dataset"]), 300)
        if minutes + cost > target_minutes and current:
            jobs.append(current)
            current, minutes = [], 0.0
        current.append((metadata, seed))
        minutes += cost
    if current:
        jobs.append(current)
    return jobs


def write_jobs(jobs: list[list], out_dir: str, hours: float) -> None:
    """Write 1 seed_NNN.pbs per packed job into out_dir, creating the log directory too."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE, "logs", "psbd_seed"), exist_ok=True)
    for index, job in enumerate(jobs, start=1):
        commands = "\n".join(command_for(metadata, seed) for metadata, seed in job)
        path = os.path.join(out_dir, f"seed_{index:03d}.pbs")
        with open(path, "w") as handle:
            handle.write(
                TEMPLATE.format(
                    base=BASE,
                    index=index,
                    walltime=f"{int(hours):02d}:00:00",
                    commands=commands,
                )
            )


def parse_args() -> argparse.Namespace:
    """The command line: tier, seeds, wall clock budget, directories and dry-run flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument(
        "--from-ledger",
        default=None,
        help="path to results/coverage/coverage.json: replicate every clearing cell to 3 seeds, hardest first, instead of a tier",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="cap the number of training runs emitted",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--checkpoints-dir", default=os.path.join(BASE, "checkpoints"))
    parser.add_argument("--out-dir", default=os.path.join(BASE, "pbs", "psbd_seed"))
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    return arguments


def main() -> None:
    """Select the tier, pack its runs into jobs, report the plan and write the job files."""
    args = parse_args()
    if args.from_ledger:
        runs = ledger_runs(args.from_ledger, args.checkpoints_dir)
        selected = sorted({metadata["folder"] for metadata, _ in runs})
    else:
        selected = discover(args.tier, args.checkpoints_dir)
        runs = [(metadata, seed) for metadata in selected for seed in args.seeds]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    jobs = pack(runs, args.hours * 60 * 0.9)

    total_minutes = sum(
        MEDIAN_MINUTES.get((m["architecture"], m["dataset"]), 300) for m, _ in runs
    )
    source = f"ledger {args.from_ledger}" if args.from_ledger else f"tier {args.tier}"
    print(f"{source}: {len(selected)} checkpoints")
    print(f"seeds {args.seeds}: {len(runs)} training runs")
    print(f"estimated GPU time: {total_minutes / 60:.0f} hours")
    print(f"jobs at {args.hours}h: {len(jobs)}")

    if args.dry_run:
        print("\ndry run, no files written")
        for metadata, seed in runs[:5]:
            print(f"  {seeded_folder(metadata['folder'], seed)}")
        print(f"  ... and {max(len(runs) - 5, 0)} more")
        return

    write_jobs(jobs, args.out_dir, args.hours)
    print(f"\nwrote {len(jobs)} job files to {args.out_dir}")


if __name__ == "__main__":
    main()
