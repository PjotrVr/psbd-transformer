"""Retrain the attacks that do not implant, into NEW folders.

An attack under the ASR bar tells a defence nothing: a detector cannot be credited or
blamed for a backdoor that was never planted. 40 of 108 panel cells sit under 0.85, and the
audit found concrete causes rather than a general weakness. SIG ran at amplitude 0.1 where
Barni et al. use 40/255, so the sinusoid was too faint to learn. Adaptive-Blend planted the
whole pattern during training, dropping the asymmetry that makes the attack work.

Nothing here overwrites a checkpoint. Every run writes to `<canonical>_v2`, so the original
stays evaluable and a comparison between the two is available; args.json records the git
commit, which is what says exactly which fix produced the new number.

Pilot first. A retrain batch is around 3 GPU-hours per cell, so one cell per attack is
trained and read before the rest are committed, on the dataset where the attack is weakest
and cheapest to train.

    PYTHONPATH=. python pbs/generate_retrain_jobs.py --pilot
    PYTHONPATH=. python pbs/generate_retrain_jobs.py --attack sig adaptive_blend
"""

import argparse
import json
import os

PROJECT_ROOT = "/lustre/home/pstika/projects/PSBD-ViT"
# Median observed training minutes at 15 epochs, from the args.json timestamps.
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

TRAIN_CALL = """echo "=== {output} :: {attack} {dataset} rate {rate} ==="
python train_backdoor.py \\
    --dataset {dataset} \\
    --attack {attack} \\
    --poison-rate {rate} \\
    --architecture vit \\
    --epochs 15 \\
    --seed 0 \\
    --output checkpoints/{output}/attack_result.pt
"""


def below_bar_cells(ledger: dict, attacks: set | None) -> list[dict]:
    cells = [
        cell
        for cell in ledger["cells"]
        if cell["asr_class"] in ("below_bar", "unmeasured")
        and not cell["folder_name"].endswith("_v2")
    ]
    if attacks:
        cells = [cell for cell in cells if cell["attack"] in attacks]
    return cells


def pilot_selection(cells: list[dict]) -> list[dict]:
    """One cell per attack: the cheapest dataset, and within it the weakest cell.

    Cost leads because a pilot's only job is a fast go or no-go on whether to commit the
    full batch, and the full batch contains the expensive cells regardless. Weakest within
    that dataset because a fix has to be shown on a cell that actually fails.
    """
    chosen = {}
    for cell in cells:
        key = cell["attack"]
        score = (
            TRAIN_MINUTES.get(cell["dataset"], 999),
            cell["asr"] if cell["asr"] is not None else 0.0,
        )
        if key not in chosen or score < chosen[key][0]:
            chosen[key] = (score, cell)
    return [cell for _, cell in chosen.values()]


def write_jobs(cells: list[dict], args) -> list[str]:
    out_dir = os.path.join("pbs", args.batch)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join("logs", args.batch), exist_ok=True)

    bundles, current, spent = [], [], 0
    for cell in sorted(
        cells, key=lambda c: (c["attack"], c["dataset"], c["poison_rate"])
    ):
        cost = TRAIN_MINUTES.get(cell["dataset"], 200)
        if current and spent + cost > args.minutes_per_job:
            bundles.append(current)
            current, spent = [], 0
        current.append(cell)
        spent += cost
    if current:
        bundles.append(current)

    written = []
    for index, bundle in enumerate(bundles, start=1):
        name = f"retrain_{index}"
        minutes = sum(TRAIN_MINUTES.get(cell["dataset"], 200) for cell in bundle)
        hours = min(48, max(6, int(minutes / 60 * 2.0) + 2))
        body = "\n".join(
            TRAIN_CALL.format(
                output=f"{cell['folder_name']}_v2",
                attack=cell["attack"],
                dataset=cell["dataset"],
                rate=cell["poison_rate"],
            )
            for cell in bundle
        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", default="results/coverage/coverage.json")
    parser.add_argument("--batch", default="vit_retrain")
    parser.add_argument("--attack", nargs="*", default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--minutes-per-job", type=float, default=400.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.coverage) as handle:
        ledger = json.load(handle)

    cells = below_bar_cells(ledger, set(args.attack) if args.attack else None)
    if args.pilot:
        cells = pilot_selection(cells)
        args.batch = f"{args.batch}_pilot"

    written = write_jobs(cells, args)
    print(f"[ok] pbs/{args.batch}/")
    print(f"     cells      {len(cells)}")
    print(f"     jobs       {len(written)}")
    print(
        f"     est GPU    {sum(TRAIN_MINUTES.get(c['dataset'], 200) for c in cells) / 60:.1f} hours"
    )
    for cell in sorted(cells, key=lambda c: c["attack"]):
        asr = "unmeasured" if cell["asr"] is None else f"{cell['asr']:.3f}"
        print(
            f"       {cell['folder_name']:34s} asr {asr:>10s} -> {cell['folder_name']}_v2"
        )
    print(f"     submit     bash pbs/{args.batch}/submit_all.sh")


if __name__ == "__main__":
    main()
