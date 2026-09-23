"""Jobs for the competitor detectors over the panel, driven by the coverage ledger.

Cells come from results/coverage/coverage.json (every cell whose attack cleared
the ASR bar) plus the benign references in configs/psbd_basis.json, minus every
(cell, detector) pair whose record already reached the scored status, so "what
is unscored" and "what to submit" are 1 list that cannot drift apart. Detectors
are grouped by cost so a cheap group never waits behind a gradient method, and
each checkpoint gets its own cli.baselines command so every failure stays
attributable to 1 checkpoint in the log.

Nothing here submits. The generator writes pbs/psbd_detectors/<group>_<index>.pbs
and a dry run prints the plan with its estimated hours. Every emitted flag is
checked against cli.baselines' own parser before a file is written.

--queue cpu emits the same work for the CPU queue instead. cli.baselines selects
its device with torch.cuda.is_available(), so a CPU node needs no flag. The job
records the device it used in each detector's provenance either way. The CPU
estimates scale the measured GPU seconds by CPU_SLOWDOWN, so a CPU job asks for a
walltime that matches what it will actually take rather than the GPU figure.

    PYTHONPATH=. python pbs/generate_detector_jobs.py --dry-run
    PYTHONPATH=. python pbs/generate_detector_jobs.py --group cheap
    PYTHONPATH=. python pbs/generate_detector_jobs.py --queue cpu --group cheap
    PYTHONPATH=. python pbs/generate_detector_jobs.py --group cd_l --dataset gtsrb
"""

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cli.baselines import build_parser as baselines_parser  # noqa: E402
from data.splits import BENIGN_PROBE_ATTACK, BENIGN_PROBE_TARGET_LABEL  # noqa: E402
from detectors.records import scored_detectors  # noqa: E402

# Grouped by cost. The cheap group finishes in minutes per checkpoint, the other
# 3 are each 1 method whose cost dwarfs the rest, so a job never mixes them.
DETECTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "cheap": (
        "confidence",
        "strip",
        "scale_up",
        "scale_up_data_limited",
        "ibd_psc",
        "ibd_psc_calibrated",
        "beatrix",
        "ted",
    ),
    "teco": ("teco",),
    "cd_l": ("cd_l",),
    "sentinet": ("sentinet",),
}
# 1 letter per group in the job name, which qstat truncates to 10 characters.
GROUP_LETTER = {"cheap": "c", "teco": "t", "cd_l": "l", "sentinet": "s"}

# validation + clean + backdoor rows per checkpoint, from the split manifests.
INPUTS_PER_CHECKPOINT = {
    "cifar10": 17200,
    "cifar100": 17915,
    "gtsrb": 23208,
    "tiny": 17959,
    "svhn": 2000 + 2 * 24032,
    "eurosat": 2000 + 2 * 3400,
}
# Measured on the login-node A100 in the smoke of 2026-09-10
# (docs/runs/2026-09-10-detector-smoke.md): scoring seconds per input over the
# validation, clean and backdoor rows at batch 64 in bfloat16, teco at batch 256
# where its launch-bound corruption loop runs 40% faster at identical scores.
SECONDS_PER_INPUT = {
    "confidence": 0.001,
    "strip": 0.005,
    "scale_up": 0.004,
    "scale_up_data_limited": 0.002,
    "ibd_psc": 0.002,
    "ibd_psc_calibrated": 0.002,
    "beatrix": 0.001,
    "ted": 0.001,
    "teco": 0.035,
    "cd_l": 0.112,
    "sentinet": 0.057,
}
# Fit seconds per validation image for the methods that fit before scoring, from
# the same smoke. The calibrated IBD-PSC runs Algorithm 1 at up to 5 factors.
FIT_SECONDS_PER_VALIDATION_IMAGE = {
    "scale_up_data_limited": 0.004,
    "ibd_psc": 0.021,
    "ibd_psc_calibrated": 0.105,
    "beatrix": 0.018,
    "ted": 0.004,
    "sentinet": 0.088,
}
VALIDATION_IMAGES = 2000
# How much slower the same scoring is on a CPU node than on the login-node A100,
# measured in the CPU smoke of 2026-09-23 (docs/runs/2026-09-23-cpu-timing.md) at
# 16 threads: confidence scored GTSRB's 23208 rows in 1487s, 15.6 rows per second
# against the A100's 1000, so 64. A factor per detector rather than 1 global one,
# because a method dominated by data movement loses less than a method dominated
# by matmul. The default is the measured figure until a smoke says otherwise.
CPU_SLOWDOWN_DEFAULT = 64.0
CPU_SLOWDOWN: dict[str, float] = {"confidence": 64.0}
# Threads a CPU job asks for, and the value it exports so torch does not spawn
# more than PBS granted it.
CPU_THREADS = 16
# Groups whose commands carry a batch size other than cli.baselines' default.
BATCH_SIZE_BY_GROUP = {"teco": 256}
# Model load and split construction, per checkpoint.
FIXED_MINUTES_PER_CHECKPOINT = 3.0
MIN_WALLTIME_HOURS = 6.0
WALLTIME_MARGIN = 2.0

GPU_TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N det_{letter}_{index:03d}
#PBS -o {base}/logs/psbd_detectors/{group}_{index:03d}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Work:    {group}, {n} checkpoints, est {estimate} min"
nvidia-smi --query-gpu=name --format=csv,noheader

BASE={base}
cd $BASE
source .venv/bin/activate
echo "Commit:  $(git rev-parse HEAD), dirty files: $(git status --porcelain | wc -l)"

{commands}
echo "Finished: $(date)"
exit 0
"""

CPU_TEMPLATE = """#!/bin/bash
#PBS -q cpu
#PBS -l select=1:ncpus={threads}:mem=64gb
#PBS -l walltime={walltime}
#PBS -N det_{letter}_{index:03d}
#PBS -o {base}/logs/psbd_detectors/{group}_cpu_{index:03d}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"
# No GPU on this node, so cli.baselines' torch.cuda.is_available() selects CPU on
# its own. The thread caps stop torch from oversubscribing the cores PBS granted.
export OMP_NUM_THREADS={threads}
export MKL_NUM_THREADS={threads}

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Work:    {group}, {n} checkpoints on CPU, est {estimate} min"

BASE={base}
cd $BASE
source .venv/bin/activate
echo "Commit:  $(git rev-parse HEAD), dirty files: $(git status --porcelain | wc -l)"

{commands}
echo "Finished: $(date)"
exit 0
"""

COMMAND = """python -m cli.baselines \\
    --checkpoint-folder {folder} \\
    --detectors {detectors} \\
    --checkpoints-dir {checkpoints_dir} \\
    --raw-data-dir {raw_data_dir} \\
    --results-dir {results_dir}{probe}{batch} \\
    --skip-existing || echo "[FAILED rc=$?] {folder}"
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(DETECTOR_GROUPS), default=None)
    parser.add_argument("--queue", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--results-dir", default=os.path.join(REPO, "results"))
    parser.add_argument("--checkpoints-dir", default=os.path.join(REPO, "checkpoints"))
    parser.add_argument("--raw-data-dir", default=os.path.join(REPO, "raw_data"))
    parser.add_argument(
        "--coverage", default=None, help="default <results-dir>/coverage/coverage.json"
    )
    parser.add_argument(
        "--declaration", default=os.path.join(REPO, "configs", "psbd_basis.json")
    )
    parser.add_argument("--dataset", nargs="*", default=None)
    parser.add_argument("--attack", nargs="*", default=None)
    parser.add_argument(
        "--folder", nargs="*", default=None, help="explicit checkpoints, for reruns"
    )
    parser.add_argument(
        "--hours", type=float, default=4.0, help="packing target per job"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def panel_folders(coverage: dict, declaration: dict) -> list[dict]:
    """Clearing cells plus the benign references, each with its dataset and attack."""
    folders = [
        {
            "folder": cell["folder_name"],
            "dataset": cell["dataset"],
            "attack": cell["attack"],
        }
        for cell in coverage["cells"]
        if cell.get("asr_class") == "clears"
    ]
    for dataset, folder in declaration["benign_reference"].items():
        if not dataset.startswith("_"):
            folders.append({"folder": folder, "dataset": dataset, "attack": "benign"})
    return folders


def select_folders(folders: list[dict], args: argparse.Namespace) -> list[dict]:
    """The requested subset, by dataset, attack or explicit name."""
    selected = folders
    if args.dataset:
        selected = [f for f in selected if f["dataset"] in args.dataset]
    if args.attack:
        selected = [f for f in selected if f["attack"] in args.attack]
    if args.folder:
        selected = [f for f in selected if f["folder"] in args.folder]
    return selected


def pending_work(
    folders: list[dict], group: str, results_dir: str
) -> list[tuple[dict, list[str]]]:
    """(cell, unscored detectors of the group) for every cell with work left."""
    work = []
    for cell in folders:
        done = scored_detectors(results_dir, cell["folder"])
        pending = [name for name in DETECTOR_GROUPS[group] if name not in done]
        if pending:
            work.append((cell, pending))
    return work


def slowdown(name: str, queue: str) -> float:
    """The factor a detector's measured GPU seconds carry on this queue."""
    if queue == "gpu":
        return 1.0
    factor = CPU_SLOWDOWN.get(name, CPU_SLOWDOWN_DEFAULT)
    return factor


def estimated_minutes(
    cell: dict, detectors: list[str], group: str, queue: str = "gpu"
) -> float:
    """Wall-clock estimate for 1 checkpoint, from the smoke's measured seconds."""
    inputs = INPUTS_PER_CHECKPOINT[cell["dataset"]]
    scoring = inputs * sum(
        SECONDS_PER_INPUT[name] * slowdown(name, queue) for name in detectors
    )
    fitting = VALIDATION_IMAGES * sum(
        FIT_SECONDS_PER_VALIDATION_IMAGE.get(name, 0.0) * slowdown(name, queue)
        for name in detectors
    )
    minutes = (scoring + fitting) / 60.0 + FIXED_MINUTES_PER_CHECKPOINT
    return minutes


def pack_jobs(
    work: list[tuple[dict, list[str]]], group: str, hours: float, queue: str = "gpu"
) -> list[list[tuple[dict, list[str]]]]:
    """Consecutive checkpoints packed into jobs of at most hours estimated minutes."""
    jobs: list[list] = []
    current: list = []
    current_minutes = 0.0
    for item in work:
        minutes = estimated_minutes(item[0], item[1], group, queue)
        if current and current_minutes + minutes > hours * 60.0:
            jobs.append(current)
            current, current_minutes = [], 0.0
        current.append(item)
        current_minutes += minutes
    if current:
        jobs.append(current)
    return jobs


def render_command(
    cell: dict, detectors: list[str], group: str, args: argparse.Namespace
) -> str:
    probe = (
        f" \\\n    --probe-attack {BENIGN_PROBE_ATTACK} --probe-target-label {BENIGN_PROBE_TARGET_LABEL}"
        if cell["attack"] == "benign"
        else ""
    )
    batch = (
        f" \\\n    --batch-size {BATCH_SIZE_BY_GROUP[group]}"
        if group in BATCH_SIZE_BY_GROUP
        else ""
    )
    command = COMMAND.format(
        folder=cell["folder"],
        detectors=" ".join(detectors),
        checkpoints_dir=args.checkpoints_dir,
        raw_data_dir=args.raw_data_dir,
        results_dir=args.results_dir,
        probe=probe,
        batch=batch,
    )
    return command


def walltime_text(estimate_minutes: float) -> str:
    hours = max(MIN_WALLTIME_HOURS, WALLTIME_MARGIN * estimate_minutes / 60.0)
    whole_hours, minutes = divmod(int(round(hours * 60)), 60)
    text = f"{whole_hours:02d}:{minutes:02d}:00"
    return text


def render_job(group: str, index: int, job: list, args: argparse.Namespace) -> str:
    estimate = sum(
        estimated_minutes(cell, detectors, group, args.queue) for cell, detectors in job
    )
    commands = "\n".join(
        render_command(cell, detectors, group, args) for cell, detectors in job
    )
    template = CPU_TEMPLATE if args.queue == "cpu" else GPU_TEMPLATE
    script = template.format(
        walltime=walltime_text(estimate),
        letter=GROUP_LETTER[group],
        index=index,
        group=group,
        base=REPO,
        n=len(job),
        estimate=int(estimate),
        threads=CPU_THREADS,
        commands=commands,
    )
    return script


def emitted_flags(script: str) -> set[str]:
    """Every --flag token of the cli.baselines commands in a script.

    Only the command line and its continuation lines are read, so the flags of
    nvidia-smi or of the shell itself never count against cli.baselines.
    """
    flags = set()
    for line in script.split("\n"):
        stripped = line.strip()
        if not (stripped.startswith("--") or "python -m cli.baselines" in stripped):
            continue
        flags.update(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", stripped))
    return flags


def verify_flags(script: str) -> None:
    """Refuse a script whose commands name a flag cli.baselines does not accept."""
    accepted = set(baselines_parser()._option_string_actions)
    unknown = sorted(emitted_flags(script) - accepted)
    if unknown:
        raise SystemExit(f"emitted flags cli.baselines rejects: {unknown}")


def load_json(path: str) -> dict:
    with open(path) as handle:
        loaded = json.load(handle)
    return loaded


def main() -> None:
    args = build_arg_parser().parse_args()
    coverage_path = args.coverage or os.path.join(
        args.results_dir, "coverage", "coverage.json"
    )
    if not os.path.exists(coverage_path):
        raise SystemExit(
            f"{coverage_path} does not exist. Point --results-dir at the tree that holds "
            "the PSBD caches and the coverage ledger."
        )
    coverage = load_json(coverage_path)
    declaration = load_json(args.declaration)
    folders = select_folders(panel_folders(coverage, declaration), args)
    groups = [args.group] if args.group else sorted(DETECTOR_GROUPS)

    out_dir = os.path.join(REPO, "pbs", "psbd_detectors")
    total_hours = 0.0
    written = []
    for group in groups:
        work = pending_work(folders, group, args.results_dir)
        jobs = pack_jobs(work, group, args.hours, args.queue)
        group_minutes = sum(estimated_minutes(c, d, group, args.queue) for c, d in work)
        total_hours += group_minutes / 60.0
        print(
            f"{group:9s} {len(work):3d} checkpoints with work, {len(jobs):3d} jobs, "
            f"est {group_minutes / 60.0:6.1f} {args.queue.upper()}-hours"
        )
        for index, job in enumerate(jobs, start=1):
            script = render_job(group, index, job, args)
            verify_flags(script)
            if args.dry_run:
                continue
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(os.path.join(REPO, "logs", "psbd_detectors"), exist_ok=True)
            suffix = "_cpu" if args.queue == "cpu" else ""
            path = os.path.join(out_dir, f"{group}{suffix}_{index:03d}.pbs")
            with open(path, "w") as handle:
                handle.write(script)
            written.append(os.path.relpath(path, REPO))
    print(f"total est {total_hours:.1f} {args.queue.upper()}-hours")
    if args.dry_run:
        print("(dry run, nothing written)")
    for path in written:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
