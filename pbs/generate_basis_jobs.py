"""Submit exactly the basis coverage that results/coverage/gaps.json says is missing.

The gap list is the generator's only input, so "what we have not tested" and "what to
submit next" cannot drift apart. Regenerate the ledger and rerun this, and it picks up
wherever the last batch stopped.

Batching, which is where the wall-clock goes. run_one_checkpoint loads a model ONCE and
loops every requested position over it, and --skip-existing short-circuits before the load,
so the efficient unit of work is one invocation per (operator, block range, rate ladder)
fanned across many positions. Baselines are written to results/<cell>/psbd/baseline_*.pt on
first use, so keeping all of a cell's work inside one job means its baseline is built once
instead of once per job.

Two things are deliberately NOT done. Positions from different rate ladders are never
merged into one invocation: gaussian's rate is a sigma and scale_up's is a multiplier, and
already_complete is all-or-nothing per placement, so a superset ladder silently recomputes
every finished rate. And nothing here deletes anything: a cell that needs new placements
added must keep the ones it has.

    PYTHONPATH=. python pbs/generate_basis_jobs.py --asr-class clears
    bash pbs/vit_basis/submit_all.sh
"""

import argparse
import collections
import json
import os

PROJECT_ROOT = "/lustre/home/pstika/projects/PSBD-ViT"
# Measured across 13617 cached placement dirs: 4.2 min for a 9-rate placement over 3 splits
# at k=3, so cost tracks the ladder length rather than the placement count.
MINUTES_PER_RATE = 0.5
# A model load per invocation, plus the one-off baseline build the first invocation pays.
MINUTES_PER_INVOCATION = 0.5
MINUTES_PER_CELL_SETUP = 3.0

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
echo '=== analyze ==='
python -m cli.analyze --checkpoint-folder {folders}

echo "Finished: $(date)"
exit 0
"""


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def selected_gaps(
    gaps: list[dict], ledger: dict, wanted_classes: set[str]
) -> list[dict]:
    """Gap slots whose cell is in one of the requested ASR classes.

    Cells below the ASR bar are excluded by default because raising their attack strength
    retrains them, and a sweep of a checkpoint that is about to be replaced is thrown away.
    """
    asr_class = {cell["folder_name"]: cell["asr_class"] for cell in ledger["cells"]}
    return [gap for gap in gaps if asr_class.get(gap["folder_name"]) in wanted_classes]


def invocation_key(gap: dict) -> tuple:
    """Slots sharing this key can run as one sweep invocation.

    The key carries the MISSING rates rather than the declared ladder. Four ladders were
    extended after the shift-reach audit, so a cell holding the old ladder needs only the new
    rates while a cell holding nothing needs all of them. Merging those two into one call
    would recompute the entire old ladder for the first cell, because already_complete is
    all-or-nothing over the rates it is asked for.
    """
    block_range = tuple(gap["block_range"]) if gap["block_range"] else None
    return (gap["operator"], block_range, tuple(gap["missing_rates"]))


def cell_workload(gaps: list[dict]) -> dict:
    """Per cell, the invocation groups it needs and the positions inside each."""
    workload: dict = collections.defaultdict(lambda: collections.defaultdict(set))
    for gap in gaps:
        workload[gap["folder_name"]][invocation_key(gap)].add(gap["position"])
    return workload


def cell_minutes(groups: dict) -> float:
    total = MINUTES_PER_CELL_SETUP
    for (_, _, rates), positions in groups.items():
        total += len(positions) * (
            len(rates) * MINUTES_PER_RATE + MINUTES_PER_INVOCATION
        )
    return total


def pack_cells(workload: dict, minutes_per_job: float) -> list[list[str]]:
    """Whole cells into jobs, largest first, so a cell's baseline is built once.

    Splitting a cell across jobs would make two jobs each build the same baseline, and if
    they run concurrently they race on the same file.
    """
    ordered = sorted(workload, key=lambda cell: -cell_minutes(workload[cell]))
    bundles: list[list[str]] = []
    budgets: list[float] = []
    for cell in ordered:
        cost = cell_minutes(workload[cell])
        for index, spent in enumerate(budgets):
            if spent + cost <= minutes_per_job:
                bundles[index].append(cell)
                budgets[index] += cost
                break
        else:
            bundles.append([cell])
            budgets.append(cost)
    return bundles


def invocation_lines(cells: list[str], workload: dict) -> str:
    """One sweep call per (operator, block range, ladder, position set).

    Cells are merged into a single call only when they need the SAME positions in that
    group. Merging cells with different needs would run the cross product and pay for
    placements nobody asked for.
    """
    merged: dict = collections.defaultdict(list)
    for cell in cells:
        for key, positions in workload[cell].items():
            merged[(key, tuple(sorted(positions)))].append(cell)

    lines = []
    for ((operator, block_range, rates), positions), folders in sorted(
        merged.items(), key=lambda item: (item[0][0][0], str(item[0][0][1]))
    ):
        band = (
            f" blocks {block_range[0]}-{block_range[1]}"
            if block_range
            else " all_layers"
        )
        lines.append(
            f'echo "=== {operator}{band} :: {len(folders)} cells x {len(positions)} positions ==="'
        )
        call = [
            "python -m cli.sweep \\",
            f"    --checkpoint-folder {' '.join(sorted(folders))} \\",
            f"    --position {' '.join(positions)} \\",
            f"    --operator {operator} \\",
        ]
        if block_range:
            call.append(f"    --block-range {block_range[0]} {block_range[1]} \\")
        call.append(f"    --rates {' '.join(str(rate) for rate in rates)} \\")
        call.append("    --forward-passes 3 \\")
        call.append("    --skip-existing")
        lines.append("\n".join(call))
        lines.append("")
    return "\n".join(lines)


def write_jobs(bundles: list[list[str]], workload: dict, args) -> list[str]:
    out_dir = os.path.join("pbs", args.batch)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join("logs", args.batch), exist_ok=True)
    written = []
    for index, cells in enumerate(bundles, start=1):
        name = f"basis_{index}"
        minutes = sum(cell_minutes(workload[cell]) for cell in cells)
        # Generous headroom: the cost model is a median over past runs, and a job killed at
        # the wall loses every placement it had not yet written.
        hours = min(20, max(4, int(minutes / 60 * 2.5) + 1))
        script = JOB_TEMPLATE.format(
            walltime=f"{hours}:00:00",
            name=name,
            root=PROJECT_ROOT,
            batch=args.batch,
            body=invocation_lines(cells, workload),
            folders=" ".join(sorted(cells)),
        )
        path = os.path.join(out_dir, f"{name}.pbs")
        with open(path, "w") as handle:
            handle.write(script)
        written.append(path)
    submit = os.path.join(out_dir, "submit_all.sh")
    with open(submit, "w") as handle:
        handle.write("#!/bin/bash\n")
        for path in written:
            handle.write(f"qsub {os.path.join(PROJECT_ROOT, path)}\n")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", default="configs/psbd_basis.json")
    parser.add_argument("--coverage-dir", default="results/coverage")
    parser.add_argument("--batch", default="vit_basis")
    parser.add_argument(
        "--asr-class",
        nargs="+",
        default=["clears"],
        choices=["clears", "below_bar", "unmeasured"],
        help="which cells to sweep; below_bar cells are retrained first, so sweeping them now throws the work away",
    )
    parser.add_argument("--minutes-per-job", type=float, default=360.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = load_json(os.path.join(args.coverage_dir, "coverage.json"))
    gaps = load_json(os.path.join(args.coverage_dir, "gaps.json"))["gaps"]

    chosen = selected_gaps(gaps, ledger, set(args.asr_class))
    workload = cell_workload(chosen)
    bundles = pack_cells(workload, args.minutes_per_job)
    written = write_jobs(bundles, workload, args)

    total = sum(cell_minutes(groups) for groups in workload.values())
    invocations = sum(
        len(
            set(
                (key, tuple(sorted(positions)))
                for key, positions in workload[cell].items()
            )
        )
        for cells in bundles
        for cell in cells
    )
    print(f"[ok] pbs/{args.batch}/")
    print(f"     asr classes      {', '.join(args.asr_class)}")
    print(f"     gap slots        {len(chosen)}")
    print(f"     rate-units       {sum(len(gap['missing_rates']) for gap in chosen)}")
    print(f"     cells            {len(workload)}")
    print(f"     jobs             {len(written)}")
    print(f"     sweep calls      {invocations}")
    print(f"     estimated GPU    {total / 60:.1f} hours")
    print(
        f"     longest job      {max(sum(cell_minutes(workload[c]) for c in b) for b in bundles) / 60:.1f} hours"
    )
    print(f"     submit with      bash pbs/{args.batch}/submit_all.sh")


if __name__ == "__main__":
    main()
