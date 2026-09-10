"""Jobs for the competitor detectors over the panel, driven by the coverage ledger.

Cells come from results/coverage/coverage.json (every cell whose attack cleared
the ASR bar) plus the benign references in configs/psbd_basis.json, minus every
(cell, detector) pair whose record already reached the scored status, so "what
is unscored" and "what to submit" are 1 list that cannot drift apart. Detectors
are grouped by cost so a cheap group never waits behind a gradient method, and
each checkpoint gets its own cli.baselines command so every failure stays
attributable to 1 checkpoint in the log.

Nothing here submits. The generator writes pbs/psbd_detectors/<group>_<index>.pbs
and a dry run prints the plan with its estimated GPU hours. Every emitted flag is
checked against cli.baselines' own parser before a file is written.

    PYTHONPATH=. python pbs/generate_detector_jobs.py --dry-run
    PYTHONPATH=. python pbs/generate_detector_jobs.py --group cheap
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
from detectors import FORWARD_PASSES_PER_INPUT  # noqa: E402
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
# ViT-B/16 forward throughput in bf16 on an A100 at batch 64. The smoke run
# replaces this with the measured figure per group.
IMAGES_PER_SECOND = 2130.0
# TeCo's glass_blur is a launch-bound Python loop that scales with pixels, so its
# wall clock runs well past its forward count. A gradient step counts 2.5.
GROUP_SLOWDOWN = {"cheap": 1.0, "teco": 2.0, "cd_l": 1.0, "sentinet": 1.0}
# Model load, split construction and fitting, per checkpoint.
FIXED_MINUTES_PER_CHECKPOINT = 3.0
MIN_WALLTIME_HOURS = 6.0
WALLTIME_MARGIN = 2.0

TEMPLATE = """#!/bin/bash
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

COMMAND = """python -m cli.baselines \\
    --checkpoint-folder {folder} \\
    --detectors {detectors} \\
    --checkpoints-dir {checkpoints_dir} \\
    --raw-data-dir {raw_data_dir} \\
    --results-dir {results_dir}{probe} \\
    --skip-existing || echo "[FAILED rc=$?] {folder}"
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(DETECTOR_GROUPS), default=None)
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


def estimated_minutes(cell: dict, detectors: list[str], group: str) -> float:
    """Wall-clock estimate for 1 checkpoint, from forward counts and the group's slowdown."""
    passes = sum(FORWARD_PASSES_PER_INPUT[name] for name in detectors)
    inputs = INPUTS_PER_CHECKPOINT[cell["dataset"]]
    seconds = inputs * passes / IMAGES_PER_SECOND * GROUP_SLOWDOWN[group]
    minutes = seconds / 60.0 + FIXED_MINUTES_PER_CHECKPOINT
    return minutes


def pack_jobs(
    work: list[tuple[dict, list[str]]], group: str, hours: float
) -> list[list[tuple[dict, list[str]]]]:
    """Consecutive checkpoints packed into jobs of at most hours estimated minutes."""
    jobs: list[list] = []
    current: list = []
    current_minutes = 0.0
    for item in work:
        minutes = estimated_minutes(item[0], item[1], group)
        if current and current_minutes + minutes > hours * 60.0:
            jobs.append(current)
            current, current_minutes = [], 0.0
        current.append(item)
        current_minutes += minutes
    if current:
        jobs.append(current)
    return jobs


def render_command(cell: dict, detectors: list[str], args: argparse.Namespace) -> str:
    probe = (
        f" \\\n    --probe-attack {BENIGN_PROBE_ATTACK} --probe-target-label {BENIGN_PROBE_TARGET_LABEL}"
        if cell["attack"] == "benign"
        else ""
    )
    command = COMMAND.format(
        folder=cell["folder"],
        detectors=" ".join(detectors),
        checkpoints_dir=args.checkpoints_dir,
        raw_data_dir=args.raw_data_dir,
        results_dir=args.results_dir,
        probe=probe,
    )
    return command


def walltime_text(estimate_minutes: float) -> str:
    hours = max(MIN_WALLTIME_HOURS, WALLTIME_MARGIN * estimate_minutes / 60.0)
    whole_hours, minutes = divmod(int(round(hours * 60)), 60)
    text = f"{whole_hours:02d}:{minutes:02d}:00"
    return text


def render_job(group: str, index: int, job: list, args: argparse.Namespace) -> str:
    estimate = sum(estimated_minutes(cell, detectors, group) for cell, detectors in job)
    commands = "\n".join(
        render_command(cell, detectors, args) for cell, detectors in job
    )
    script = TEMPLATE.format(
        walltime=walltime_text(estimate),
        letter=GROUP_LETTER[group],
        index=index,
        group=group,
        base=REPO,
        n=len(job),
        estimate=int(estimate),
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
        jobs = pack_jobs(work, group, args.hours)
        group_minutes = sum(estimated_minutes(c, d, group) for c, d in work)
        total_hours += group_minutes / 60.0
        print(
            f"{group:9s} {len(work):3d} checkpoints with work, {len(jobs):3d} jobs, "
            f"est {group_minutes / 60.0:6.1f} GPU-hours"
        )
        for index, job in enumerate(jobs, start=1):
            script = render_job(group, index, job, args)
            verify_flags(script)
            if args.dry_run:
                continue
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(os.path.join(REPO, "logs", "psbd_detectors"), exist_ok=True)
            path = os.path.join(out_dir, f"{group}_{index:03d}.pbs")
            with open(path, "w") as handle:
                handle.write(script)
            written.append(os.path.relpath(path, REPO))
    print(f"total est {total_hours:.1f} GPU-hours")
    if args.dry_run:
        print("(dry run, nothing written)")
    for path in written:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
