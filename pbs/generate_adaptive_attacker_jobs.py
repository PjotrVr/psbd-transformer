"""Emit training jobs for the PSBD paper's own adaptive attacker.

The paper's Appendix ("Resistance to Potential Adaptive Attacks") trains BadNets
and WaNet against a loss term that pushes every training sample's PSU down,
weighted by alpha in {0.2, 0.5, 0.9} and reports that PSBD still detects the
result. attacks.evasion's `psbd_paper` objective implements that exact loss
(see its module docstring for the formula and the 2 recorded deviations). This
generator retrains 3 of our own hard cells at that objective, against the
placement we deploy (before_attention_norm token_mask), then sweeps and
analyzes the result at both that deployed placement and the published ConvNet
placement (post_residual dropout), so a defended view and a competing view of
the same adaptive checkpoint land in results/ without a second job.

Reads each cell's training arguments back from its own checkpoints/<folder>/
args.json, exactly as pbs/generate_seed_jobs.py does, so a rerun cannot
silently differ from the base run in a way nobody notices. The only fields
that change are the 4 evasion flags and the output folder.

Every command is validated against the real argparse parsers
(cli.train_backdoor, cli.sweep, cli.analyze) before anything is written, so a
flag typo fails here rather than after a job has queued for hours.

    python pbs/generate_adaptive_attacker_jobs.py --dry-run
    python pbs/generate_adaptive_attacker_jobs.py
    bash pbs/vit_adaptive_attacker/submit_all.sh
"""

import argparse
import json
import os
import sys

BASE = "/lustre/home/pstika/projects/PSBD-ViT"

CELLS = ("vit_cifar100_bpp_0_01", "vit_tiny_bpp_0_01", "vit_gtsrb_wanet_0_1")
ALPHAS = (0.2, 0.5, 0.9)

# The placement we deploy. The attacker trains against exactly this probe,
# which is the strongest, white-box variant of the threat model (see
# docs/plans/adaptive-attacker-and-dropout-stacking.md, E3).
PROBE_POSITION = "before_attention_norm"
PROBE_OPERATOR = "token_mask"

# The 2 placements every trained checkpoint gets swept and analyzed at: the
# deployed recommendation and the published ConvNet placement it is compared
# against. Read from the basis declaration rather than restated, so the rate
# ladder used here can never drift from the one the rest of the panel uses.
SWEEP_PLACEMENT_IDS = ("before_attention_norm_token_mask", "post_residual")

# Median wall clock per plain Adam training run, the same figures
# pbs/generate_seed_jobs.py carries for these 3 datasets (measured from
# trained_started_at to trained_ended_at across the panel, evasion runs
# excluded).
PLAIN_MINUTES = {"cifar100": 84, "tiny": 167, "gtsrb": 46}

# Evasion training reruns the k probe passes every batch, on top of the
# ordinary forward and backward pass, and the paper's own recipe adds the same
# passes for L_ada. Both the hinge and psbd_paper objectives pay this, and
# past hinge runs measured 2 to 4 times a plain run's wall clock, see
# pbs/generate_seed_jobs.py's MEDIAN_MINUTES comment. 4x is the conservative
# end of that range and what the task asks this generator to budget against.
EVASION_MULTIPLIER = 4.0

# generate_basis_jobs.py's own measured figures: 4.2 minutes for a 9-rate
# placement over 3 splits at k=3, so cost tracks ladder length, plus a
# per-invocation model load and a one-off baseline build the first invocation
# pays.
SWEEP_MINUTES_PER_RATE = 0.5
SWEEP_MINUTES_PER_INVOCATION = 0.5
SWEEP_MINUTES_SETUP = 3.0
ANALYZE_MINUTES = 1.0

JOB_TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N vit_adaptive_attacker_{index:03d}
#PBS -o {base}/logs/vit_adaptive_attacker/job_{index:03d}.log
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


def alpha_tag(alpha: float) -> str:
    """0.2 becomes '0_2', matching the checkpoint-naming convention every other tag uses."""
    tag = f"{alpha:.1f}".replace(".", "_")
    return tag


def new_folder_name(base_folder: str, alpha: float) -> str:
    """The output checkpoint folder for a single (cell, alpha) adaptive-attacker run."""
    folder = f"{base_folder}_evade_paper_a{alpha_tag(alpha)}"
    return folder


def load_metadata(checkpoints_dir: str, folder: str) -> dict:
    """A cell's own training arguments, read back from its args.json."""
    path = os.path.join(checkpoints_dir, folder, "args.json")
    with open(path) as handle:
        metadata = json.load(handle)
    metadata["folder"] = folder
    return metadata


def load_sweep_placements(declaration_path: str) -> list[dict]:
    """The 2 headline placements' own (position, operator, rate ladder), from the basis."""
    with open(declaration_path) as handle:
        basis = json.load(handle)["basis"]
    by_id = {entry["id"]: entry for entry in basis}
    placements = [by_id[placement_id] for placement_id in SWEEP_PLACEMENT_IDS]
    return placements


def discover_runs(
    cells: tuple[str, ...], alphas: tuple[float, ...], checkpoints_dir: str
):
    """Every (metadata, alpha) pair this generator trains, cells outer, alphas inner."""
    runs = [
        (load_metadata(checkpoints_dir, folder), alpha)
        for folder in cells
        for alpha in alphas
    ]
    return runs


def train_argv(metadata: dict, alpha: float, folder: str) -> list[str]:
    """The training command's flags, as an argv list, for validation and rendering alike."""
    argv = [
        "--dataset",
        metadata["dataset"],
        "--attack",
        metadata["attack"],
        "--poison-rate",
        str(metadata["poison_rate"]),
        "--target-label",
        str(metadata["target_label"]),
        "--architecture",
        metadata["architecture"],
        "--epochs",
        str(metadata["epochs"]),
        "--seed",
        str(metadata["seed"]),
        "--evade-psbd",
        "--evade-weight",
        str(alpha),
        "--evade-position",
        PROBE_POSITION,
        "--evade-operator",
        PROBE_OPERATOR,
        "--evade-objective",
        "psbd_paper",
        "--output",
        f"checkpoints/{folder}/attack_result.pt",
    ]
    return argv


def sweep_argv(folder: str, placement: dict) -> list[str]:
    """The flags for a single sweep invocation, covering 1 (position, operator) placement."""
    argv = ["--checkpoint-folder", folder, "--position", placement["position"]]
    if placement["block_range"]:
        argv += [
            "--block-range",
            str(placement["block_range"][0]),
            str(placement["block_range"][1]),
        ]
    argv += [
        "--operator",
        placement["operator"],
        "--rates",
        *[str(rate) for rate in placement["rates"]],
        "--forward-passes",
        "3",
        "--skip-existing",
    ]
    return argv


def analyze_argv(folder: str) -> list[str]:
    return ["--checkpoint-folder", folder]


def render_call(module: str, argv: list[str]) -> str:
    """argv rendered as a backslash-continued `python -m {module}` invocation.

    Groups a flag with the values that follow it onto 1 line (so
    `--rates 0.1 0.2 ...` reads as 1 line rather than 1 token per line),
    matching the style of pbs/generate_seed_jobs.py and
    pbs/generate_basis_jobs.py's emitted commands.
    """
    lines = [f"python -m {module} \\"]
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
    """Raise with the offending command if argparse itself would reject argv.

    parse_args is the module's own parse_args function, called with sys.argv
    patched so a --choices typo or a missing required flag is caught before any
    file is written, not after a job has queued for hours.
    """
    saved_argv = sys.argv
    sys.argv = ["prog"] + argv
    try:
        parse_args()
    except SystemExit as error:
        raise ValueError(
            f"{label}: argparse rejected `python -m ... {' '.join(argv)}` "
            f"(exit {error.code})"
        ) from error
    finally:
        sys.argv = saved_argv


def sweep_analyze_minutes(placements: list[dict]) -> float:
    """The wall clock 1 checkpoint's headline sweep and analyze step costs."""
    total = SWEEP_MINUTES_SETUP + ANALYZE_MINUTES
    for placement in placements:
        total += (
            len(placement["rates"]) * SWEEP_MINUTES_PER_RATE
            + SWEEP_MINUTES_PER_INVOCATION
        )
    return total


def run_minutes(metadata: dict, placements: list[dict]) -> float:
    """1 (cell, alpha) run's total cost: evasion training plus its own sweep and analyze."""
    training = PLAIN_MINUTES[metadata["dataset"]] * EVASION_MULTIPLIER
    total = training + sweep_analyze_minutes(placements)
    return total


def pack(
    runs: list[tuple[dict, float]], placements: list[dict], target_minutes: float
) -> list[list[tuple[dict, float]]]:
    """First-fit decreasing: largest runs placed first, into the first job with room.

    A run that alone exceeds target_minutes still gets its own job rather than
    being dropped, exactly as pbs/generate_seed_jobs.py's own pack() falls back
    when a single item is larger than the budget.
    """
    costed = sorted(
        (
            (run_minutes(metadata, placements), metadata, alpha)
            for metadata, alpha in runs
        ),
        key=lambda item: -item[0],
    )
    bins: list[tuple[float, list[tuple[dict, float]]]] = []
    for cost, metadata, alpha in costed:
        for position, (used, group) in enumerate(bins):
            if used + cost <= target_minutes:
                bins[position] = (used + cost, group + [(metadata, alpha)])
                break
        else:
            bins.append((cost, [(metadata, alpha)]))
    jobs = [group for _, group in bins]
    return jobs


def commands_for_run(metadata: dict, alpha: float, placements: list[dict]) -> str:
    """The training, sweep and analyze commands for 1 (cell, alpha) run, as 1 block."""
    folder = new_folder_name(metadata["folder"], alpha)
    blocks = [
        f'echo "=== {folder} ==="',
        render_call("cli.train_backdoor", train_argv(metadata, alpha, folder)),
    ]
    for placement in placements:
        blocks.append(render_call("cli.sweep", sweep_argv(folder, placement)))
    blocks.append(render_call("cli.analyze", analyze_argv(folder)))
    commands = "\n".join(blocks)
    return commands


def validate_all(runs: list[tuple[dict, float]], placements: list[dict]) -> None:
    """Every command this generator would emit, checked against its real parser."""
    # cli.* only resolves from the repository root, and this script is run as a
    # plain file (`python pbs/generate_adaptive_attacker_jobs.py`) rather than
    # `python -m`, so BASE is added to sys.path here rather than assumed.
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    from cli import analyze, sweep, train_backdoor

    for metadata, alpha in runs:
        folder = new_folder_name(metadata["folder"], alpha)
        validate_argv(
            train_backdoor.parse_args,
            train_argv(metadata, alpha, folder),
            "train_backdoor",
        )
        for placement in placements:
            validate_argv(sweep.parse_args, sweep_argv(folder, placement), "sweep")
        validate_argv(analyze.parse_args, analyze_argv(folder), "analyze")


def write_jobs(
    jobs: list[list[tuple[dict, float]]],
    placements: list[dict],
    out_dir: str,
    hours: float,
) -> list[str]:
    """1 vit_adaptive_attacker_NNN.pbs per packed job, plus submit_all.sh."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE, "logs", "vit_adaptive_attacker"), exist_ok=True)

    written = []
    for index, group in enumerate(jobs, start=1):
        commands = "\n".join(
            commands_for_run(metadata, alpha, placements) for metadata, alpha in group
        )
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
    parser.add_argument("--checkpoints-dir", default=os.path.join(BASE, "checkpoints"))
    parser.add_argument(
        "--declaration", default=os.path.join(BASE, "configs/psbd_basis.json")
    )
    parser.add_argument(
        "--out-dir", default=os.path.join(BASE, "pbs", "vit_adaptive_attacker")
    )
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    return arguments


def main() -> None:
    args = parse_args()
    runs = discover_runs(CELLS, ALPHAS, args.checkpoints_dir)
    placements = load_sweep_placements(args.declaration)
    validate_all(runs, placements)

    jobs = pack(runs, placements, args.hours * 60 * 0.9)
    total_minutes = sum(run_minutes(metadata, placements) for metadata, _ in runs)

    print(f"cells             {len(CELLS)}")
    print(f"alphas            {ALPHAS}")
    print(f"training runs     {len(runs)}")
    print(f"probe             {PROBE_POSITION} {PROBE_OPERATOR}")
    print(f"sweep placements  {SWEEP_PLACEMENT_IDS}")
    print(f"jobs at {args.hours}h      {len(jobs)}")
    print(f"estimated GPU time   {total_minutes / 60:.1f} hours")
    print(
        "longest job          "
        f"{max(sum(run_minutes(m, placements) for m, _ in group) for group in jobs) / 60:.1f} hours"
    )

    if args.dry_run:
        print("\ndry run, no files written\n")
        for index, group in enumerate(jobs, start=1):
            job_minutes = sum(
                run_minutes(metadata, placements) for metadata, _ in group
            )
            print(f"job_{index:03d} ({job_minutes / 60:.1f}h):")
            for metadata, alpha in group:
                folder = new_folder_name(metadata["folder"], alpha)
                print(f"  {folder}")
                print(commands_for_run(metadata, alpha, placements))
        return

    written = write_jobs(jobs, placements, args.out_dir, args.hours)
    print(f"\nwrote {len(written)} job files to {args.out_dir}")
    print(f"submit with          bash {args.out_dir}/submit_all.sh")


if __name__ == "__main__":
    main()
