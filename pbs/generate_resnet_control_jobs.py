"""Emit training jobs for the ResNet-18 control: the paper's own architecture,
placement and adaptive attacker, on the paper's own datasets.

experiments/resnet_control/README.md states the question this control answers:
does our hinge attacker (psu_gap_hinge, --evade-objective's new name for what
used to be called "hinge") beat PSBD on ResNet-18/CIFAR-10 with the paper's
own post_residual dropout placement, or only on ViT. This generator trains a
plain run, the paper's own adaptive attacker (psu_mean, --evade-weight 0.5)
and our attacker (psu_gap_hinge, --evade-weight 1.0) for each of 2 datasets
(cifar10, gtsrb) and 2 attacks (BadNets, Blend) at the paper's own 10%
poisoning ratio: 2 * 2 * 3 = 12 training runs, each followed by a
post_residual dropout sweep over the full rate ladder and cli.analyze.

The 3 variants differ by exactly 1 thing, the evasion flags, so a diff between
2 rendered commands below shows only that:

    python -m cli.train_backdoor --dataset cifar10 --attack badnet_a2o \\
        --poison-rate 0.1 --target-label 0 --architecture resnet18 \\
        --epochs 100 --output checkpoints/<folder>/attack_result.pt

    python -m cli.train_backdoor <the same flags> \\
        --evade-psbd --evade-objective psu_mean --evade-weight 0.5

    python -m cli.train_backdoor <the same flags> \\
        --evade-psbd --evade-objective psu_gap_hinge --evade-weight 1.0

No --evade-position / --evade-operator is passed: cli.train_backdoor's
DEFAULT_EVADE_PROBE_BY_ARCHITECTURE resolves "resnet18" to post_residual:dropout
on its own, which is what makes the 3 commands differ by only the evasion
flags rather than also needing an architecture-specific probe override.

Training runs through `python -m cli.train_backdoor` directly, since
training/loop.py's build_model knows resnet18.

    python pbs/generate_resnet_control_jobs.py --dry-run
    python pbs/generate_resnet_control_jobs.py
    bash pbs/resnet_control/submit_all.sh
"""

import argparse
import os
import sys

BASE = "/lustre/home/pstika/projects/PSBD-ViT"

DATASETS = ("cifar10", "gtsrb")
ATTACKS = ("badnet_a2o", "blend")
POISON_RATE = 0.1
TARGET_LABEL = 0
ARCHITECTURE = "resnet18"

# papers/PSBD/sec/7_appendix.tex's training table: ResNet-18 on CIFAR-10 and
# GTSRB, SGD, 100 epochs. training/loop.py's build_optimizer only offers Adam
# (or SAM-wrapped Adam) and that file is out of this control's scope, so
# these runs keep the project's own Adam recipe instead of the paper's SGD.
# See experiments/resnet_control/README.md's deviations table. The epoch
# count is the 1 paper figure this generator does keep, since nothing blocks
# it and the walltime budget below is sized for it.
EPOCHS = 100

# The paper's own placement, and the only position resnet18's registry has
# (models.positions.RESNET_POSITIONS).
SWEEP_POSITION = "post_residual"
SWEEP_OPERATOR = "dropout"

# The smoke test's own ladder (docs table in the README), reused here so the
# full run's ladder is the same one the smoke result was read against.
SWEEP_RATES: tuple[float, ...] = (
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
)
FORWARD_PASSES = 3

# The 3 evasion variants: (folder tag, evasion flags). "" for the plain run
# keeps its folder name bare, matching the project's checkpoint-naming
# convention of no tag for the unmarked default.
VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("", ()),
    (
        "_evade_paper",
        ("--evade-psbd", "--evade-objective", "psu_mean", "--evade-weight", "0.5"),
    ),
    (
        "_evade_hinge",
        (
            "--evade-psbd",
            "--evade-objective",
            "psu_gap_hinge",
            "--evade-weight",
            "1.0",
        ),
    ),
)

# Measured on the login GPU (experiments/resnet_control/README.md's smoke
# table): 20000 CIFAR-10 samples, 15 epochs, cost 1.0 minute plain and 2.9
# minutes for the hinge attacker (calibration plus 15 evasive epochs).
# Extrapolated per-epoch and per-sample to the full recipe below. GTSRB's
# 39209 training images are close enough to CIFAR-10's 50000 to share this
# estimate rather than carry a 2nd estimate this generator cannot measure
# without the same smoke run on GTSRB.
SMOKE_SAMPLES = 20000
SMOKE_EPOCHS = 15
SMOKE_PLAIN_MINUTES = 1.0
SMOKE_EVADE_MINUTES = 2.9
FULL_SAMPLES = {"cifar10": 50000, "gtsrb": 39209}
# The calibration pass before training is a fixed cost, not a per-epoch cost,
# so it does not scale with EPOCHS. Charged in full here (rather than
# subtracted out and re-added) because the smoke run's 15 epochs are too few
# for the per-epoch rate to be measured cleanly on their own, so the result is
# a deliberately conservative (slightly too large) estimate.
EVASION_CALIBRATION_MINUTES = 0.5

SWEEP_MINUTES_PER_RATE = 0.5
SWEEP_MINUTES_SETUP = 3.0
ANALYZE_MINUTES = 1.0

JOB_TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N resnet_control_{index:03d}
#PBS -o {base}/logs/resnet_control/job_{index:03d}.log
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


def folder_name(dataset: str, attack: str, tag: str) -> str:
    """The canonical checkpoint folder for 1 (dataset, attack, variant) cell."""
    rate_tag = str(POISON_RATE).replace(".", "_")
    return f"resnet18_{dataset}_{attack}_{rate_tag}{tag}"


def train_argv(
    dataset: str, attack: str, folder: str, evade_flags: tuple[str, ...]
) -> list[str]:
    """The training command's flags, as an argv list, for validation and rendering alike."""
    argv = [
        "--dataset",
        dataset,
        "--attack",
        attack,
        "--poison-rate",
        str(POISON_RATE),
        "--target-label",
        str(TARGET_LABEL),
        "--architecture",
        ARCHITECTURE,
        "--epochs",
        str(EPOCHS),
        *evade_flags,
        "--output",
        f"checkpoints/{folder}/attack_result.pt",
    ]
    return argv


def sweep_argv(folder: str) -> list[str]:
    argv = [
        "--checkpoint-folder",
        folder,
        "--position",
        SWEEP_POSITION,
        "--operator",
        SWEEP_OPERATOR,
        "--rates",
        *[str(rate) for rate in SWEEP_RATES],
        "--forward-passes",
        str(FORWARD_PASSES),
        "--skip-existing",
    ]
    return argv


def analyze_argv(folder: str) -> list[str]:
    return ["--checkpoint-folder", folder]


def render_call(module_invocation: str, argv: list[str]) -> str:
    """argv rendered as a backslash-continued invocation, grouping each flag with its values.

    module_invocation is the full `python ...` prefix: a plain `-m cli.sweep`
    style module or the resnet_control wrapper's own module path, matching
    pbs/generate_adaptive_attacker_jobs.py's render_call except for that
    prefix, which is not always `python -m {module}` here.
    """
    lines = [f"{module_invocation} \\"]
    index = 0
    while index < len(argv):
        token = argv[index]
        values = []
        cursor = index + 1
        while cursor < len(argv) and not argv[cursor].startswith("--"):
            values.append(argv[cursor])
            cursor += 1
        piece = " ".join([token] + values)
        index = cursor
        is_last = index >= len(argv)
        lines.append(f"    {piece}" + ("" if is_last else " \\"))
    rendered = "\n".join(lines) + "\n"
    return rendered


def validate_argv(parse_args, argv: list[str], label: str) -> None:
    """Raise with the offending command if argparse itself would reject argv."""
    saved_argv = sys.argv
    sys.argv = ["prog"] + argv
    try:
        parse_args()
    except SystemExit as error:
        raise ValueError(
            f"{label}: argparse rejected `... {' '.join(argv)}` (exit {error.code})"
        ) from error
    finally:
        sys.argv = saved_argv


def discover_cells() -> list[tuple[str, str, str, tuple[str, ...]]]:
    """Every (dataset, attack, folder_tag, evade_flags) cell this generator trains."""
    cells = [
        (dataset, attack, tag, evade_flags)
        for dataset in DATASETS
        for attack in ATTACKS
        for tag, evade_flags in VARIANTS
    ]
    return cells


def run_minutes(dataset: str, evade_flags: tuple[str, ...]) -> float:
    """1 cell's total cost: training plus its own sweep and analyze."""
    scale = (FULL_SAMPLES[dataset] / SMOKE_SAMPLES) * (EPOCHS / SMOKE_EPOCHS)
    if evade_flags:
        training = SMOKE_EVADE_MINUTES * scale + EVASION_CALIBRATION_MINUTES
    else:
        training = SMOKE_PLAIN_MINUTES * scale
    sweep_analyze = (
        SWEEP_MINUTES_SETUP
        + len(SWEEP_RATES) * SWEEP_MINUTES_PER_RATE
        + ANALYZE_MINUTES
    )
    return training + sweep_analyze


def commands_for_cell(
    dataset: str, attack: str, tag: str, evade_flags: tuple[str, ...]
) -> str:
    """The training, sweep and analyze commands for 1 cell, as 1 block."""
    folder = folder_name(dataset, attack, tag)
    train_log = f"checkpoints/{folder}/train.log"
    blocks = [
        f'echo "=== {folder} ==="',
        render_call(
            "python -m cli.train_backdoor",
            train_argv(dataset, attack, folder, evade_flags),
        ).rstrip("\n")
        + f" \\\n    2>&1 | tee {train_log}",
        # Highlighted separately from the full training log so the rate the
        # attacker calibrated against is visible without reading the whole job
        # log. A plain run prints nothing matching this, which is why the
        # fallback line exists.
        f'grep -m1 "calibrated probe rate" {train_log} || '
        f'echo "{folder}: no calibration line (not an evasion run)"',
        render_call("python -m cli.sweep", sweep_argv(folder)),
        render_call("python -m cli.analyze", analyze_argv(folder)),
    ]
    return "\n".join(blocks)


def validate_all(cells: list[tuple[str, str, str, tuple[str, ...]]]) -> None:
    """Every command this generator would emit, checked against its real parser."""
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    from cli import analyze, sweep, train_backdoor

    for dataset, attack, tag, evade_flags in cells:
        folder = folder_name(dataset, attack, tag)
        validate_argv(
            train_backdoor.parse_args,
            train_argv(dataset, attack, folder, evade_flags),
            "train_backdoor",
        )
        validate_argv(sweep.parse_args, sweep_argv(folder), "sweep")
        validate_argv(analyze.parse_args, analyze_argv(folder), "analyze")


def pack(
    cells: list[tuple[str, str, str, tuple[str, ...]]], target_minutes: float
) -> list[list[tuple[str, str, str, tuple[str, ...]]]]:
    """1 job per dataset: the 6 (attack, variant) cells for cifar10, then for gtsrb.

    A dataset's 6 cells always fit comfortably under a 12-hour budget (see the
    dry run's own totals), so first-fit-decreasing packing across datasets
    would only obscure which job holds which dataset for no packing benefit.
    Kept as an explicit per-dataset grouping instead, and the target_minutes
    bound is still checked so a future change to EPOCHS or the rate ladder
    that blows the budget fails here rather than at qsub time.
    """
    jobs = []
    for dataset in DATASETS:
        group = [cell for cell in cells if cell[0] == dataset]
        cost = sum(run_minutes(cell[0], cell[3]) for cell in group)
        if cost > target_minutes:
            raise ValueError(
                f"{dataset}: estimated {cost / 60:.1f}h exceeds the "
                f"{target_minutes / 60:.1f}h budget for 1 job; split it further"
            )
        jobs.append(group)
    return jobs


def write_jobs(
    jobs: list[list[tuple[str, str, str, tuple[str, ...]]]], out_dir: str, hours: float
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE, "logs", "resnet_control"), exist_ok=True)

    written = []
    for index, group in enumerate(jobs, start=1):
        commands = "\n".join(commands_for_cell(*cell) for cell in group)
        path = os.path.join(out_dir, f"job_{index:03d}.pbs")
        with open(path, "w") as handle:
            handle.write(
                JOB_TEMPLATE.format(
                    base=BASE,
                    index=index,
                    walltime=f"{int(hours):02d}:00:00",
                    commands=commands,
                )
            )
        written.append(path)

    submit_path = os.path.join(out_dir, "submit_all.sh")
    with open(submit_path, "w") as handle:
        handle.write("#!/bin/bash\n")
        for path in written:
            handle.write(f"qsub {path}\n")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default=os.path.join(BASE, "pbs", "resnet_control")
    )
    parser.add_argument("--hours", type=float, default=11.0)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    return arguments


def main() -> None:
    args = parse_args()
    cells = discover_cells()
    validate_all(cells)

    jobs = pack(cells, args.hours * 60)
    total_minutes = sum(run_minutes(cell[0], cell[3]) for cell in cells)

    print(f"datasets          {DATASETS}")
    print(f"attacks           {ATTACKS}")
    print(f"variants          {[tag or '(plain)' for tag, _ in VARIANTS]}")
    print(f"training runs     {len(cells)}")
    print(f"probe             {SWEEP_POSITION} {SWEEP_OPERATOR}")
    print(f"rate ladder       {SWEEP_RATES}")
    print(f"jobs at {args.hours}h      {len(jobs)}")
    print(f"estimated GPU time   {total_minutes / 60:.1f} hours")
    for index, group in enumerate(jobs, start=1):
        job_minutes = sum(run_minutes(cell[0], cell[3]) for cell in group)
        print(
            f"  job_{index:03d}: {group[0][0]}, {len(group)} cells, {job_minutes / 60:.1f}h"
        )

    if args.dry_run:
        print("\ndry run, no files written\n")
        for index, group in enumerate(jobs, start=1):
            print(f"job_{index:03d}:")
            for cell in group:
                folder = folder_name(cell[0], cell[1], cell[2])
                print(f"  {folder}")
                print(commands_for_cell(*cell))
        return

    written = write_jobs(jobs, args.out_dir, args.hours)
    print(f"\nwrote {len(written)} job files to {args.out_dir}")
    print(f"submit with          bash {args.out_dir}/submit_all.sh")


if __name__ == "__main__":
    main()
