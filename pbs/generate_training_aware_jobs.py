"""Emit jobs for 2 training-time ideas: a defender-side PSU floor penalty (idea A)
and sanitise-then-retrain (idea B), on the 8 base cells where PSBD-TM is weakest
or oddest: TaCT at 5% on cifar10, cifar100 and tiny, WaNet at 5% on cifar10 and
tiny, BPP at 5% on cifar10 and WaNet and SIG at 10% on cifar10 only.

Idea A trains 3 variants of a --evade-psbd --evade-objective psu_floor penalty
per setting: 2 that recalibrate the probe rate every epoch to a sigma=0.8 target
at weight 1.0 and 0.5, and 1 that fixes the probe rate at 0.2 instead of
recalibrating. It never touches is_poisoned, so it is a genuinely defender-side
idea rather than a relabelled adaptive attacker.

Idea B has 2 stages. B1 (--record-sample-loss, suffix _lossrec) trains the base
recipe once more, only to write checkpoints/<folder>_lossrec/sample_loss.npz. A
CPU step (experiments/training_aware/flag_by_loss.py) then ranks that run's
samples by area under their loss curve and flags the 1.5x-poison-count lowest
ones. B2 (--exclude-indices-file, suffix _sanitised) retrains with those samples
dropped. B2 depends on B1 finishing, both because sample_loss.npz has to exist
and because the flagging step is the B2 job's own first command, run on the
compute node so the file it needs is never stale relative to the code that made
it.

Every training command in every stage is followed by the 2 headline sweeps
(before_attention_norm token_mask, post_residual dropout, from
configs/psbd_basis.json) and cli.analyze, so a plain results/<folder>/psbd_metrics.json
lands for every checkpoint this generator trains, panel cells included.

Every command is validated against the real argparse parsers (cli.train_backdoor,
cli.sweep, cli.analyze) before anything is written, exactly as
pbs/generate_adaptive_attacker_jobs.py does, so a flag typo fails here rather than
after a job has queued for hours.

Each base checkpoint's dataset, attack, poison_rate, target_label, architecture,
epochs and seed are read back from its own checkpoints/<folder>/args.json, as
pbs/generate_seed_jobs.py does, so a rerun cannot silently differ from the base
run except in the flags this generator adds. A null seed (every checkpoint
currently on disk, see .claude/skills/git-workflow/SKILL.md section 13) defaults
to 0, the panel's own unmarked seed.

    python pbs/generate_training_aware_jobs.py --stage a --dry-run
    python pbs/generate_training_aware_jobs.py --stage a
    bash pbs/training_aware/a/submit_all.sh
    python pbs/generate_training_aware_jobs.py --stage b1
    bash pbs/training_aware/b1/submit_all.sh
    # after b1 finishes, record its job ids as {"vit_..._tact_0_05": "123456.pbs-server", ...}
    python pbs/generate_training_aware_jobs.py --stage b2 --depends-on pbs/training_aware/b1_ids.json
    bash pbs/training_aware/b2/submit_all.sh
"""

import argparse
import json
import os
import sys

BASE = "/lustre/home/pstika/projects/PSBD-ViT"

# 5% settings first, as the task orders them. TaCT and WaNet at 5% cover cifar10,
# cifar100 and tiny (cifar100 and tiny only where each attack clears at 5%), and
# BPP at 5% and WaNet, SIG at 10% are cifar10 only.
SETTINGS = (
    "vit_cifar10_tact_0_05",
    "vit_cifar100_tact_0_05",
    "vit_tiny_tact_0_05",
    "vit_cifar10_wanet_0_05",
    "vit_tiny_wanet_0_05",
    "vit_cifar10_bpp_0_05",
    "vit_cifar10_wanet_0_1",
    "vit_cifar10_sig_0_1",
)

# The 2 sweep placements every trained checkpoint is read at: the deployed
# recommendation and the published ConvNet placement it is compared against.
# Read from the basis declaration so the rate ladder here can never drift from
# the one the rest of the panel uses.
SWEEP_PLACEMENT_IDS = ("before_attention_norm_token_mask", "post_residual")

# Median wall clock per plain Adam training run, the same figures
# pbs/generate_seed_jobs.py carries, for the 3 datasets this batch touches.
PLAIN_MINUTES = {"cifar10": 84, "cifar100": 84, "tiny": 167}

# Idea A's evasion probe, its own probe pass cost and the batch size that keeps
# it inside a 40 GB card: 3 probe passes per batch on top of the plain forward
# and backward overflow at the panel's default batch, exactly as
# pbs/generate_adaptive_attacker_jobs.py's own EVASION_MULTIPLIER measured.
EVASION_MULTIPLIER = 4.0
EVASION_BATCH_SIZE = 48

IDEA_A_PROBE = "before_attention_norm:token_mask"
IDEA_A_VARIANTS = (
    (
        "_floor_cal_w1",
        [
            "--evade-psbd",
            "--evade-objective",
            "psu_floor",
            "--evade-weight",
            "1.0",
            "--evade-margin",
            "0.8",
            "--evade-recalibrate-every",
            "1",
            "--evade-calibration-target",
            "0.8",
            "--evade-probes",
            IDEA_A_PROBE,
        ],
    ),
    (
        "_floor_cal_w05",
        [
            "--evade-psbd",
            "--evade-objective",
            "psu_floor",
            "--evade-weight",
            "0.5",
            "--evade-margin",
            "0.8",
            "--evade-recalibrate-every",
            "1",
            "--evade-calibration-target",
            "0.8",
            "--evade-probes",
            IDEA_A_PROBE,
        ],
    ),
    (
        "_floor_r02",
        [
            "--evade-psbd",
            "--evade-objective",
            "psu_floor",
            "--evade-weight",
            "1.0",
            "--evade-margin",
            "0.8",
            "--evade-rate",
            "0.2",
            "--evade-probes",
            IDEA_A_PROBE,
        ],
    ),
)

B1_SUFFIX = "_lossrec"
B2_SUFFIX = "_sanitised"

# generate_adaptive_attacker_jobs.py's own measured figures: sweep cost tracks
# ladder length, plus a per-invocation model load and a one-off baseline build
# the first invocation pays.
SWEEP_MINUTES_PER_RATE = 0.5
SWEEP_MINUTES_PER_INVOCATION = 0.5
SWEEP_MINUTES_SETUP = 3.0
ANALYZE_MINUTES = 1.0
# flag_by_loss.py reads a single .npz and writes a JSON list, seconds not minutes,
# but it is not free and B2's cost table should not silently pretend it is.
FLAG_MINUTES = 1.0

JOB_TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N training_aware_{stage}_{index:03d}
#PBS -o {base}/logs/training_aware/{stage}/job_{index:03d}.log
#PBS -j oe
{depend_line}
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


def load_metadata(checkpoints_dir: str, folder: str) -> dict:
    """A base checkpoint's own training arguments, read back from its args.json.

    seed is null on every checkpoint currently on disk (the provenance gap
    .claude/skills/git-workflow/SKILL.md section 13 records), so a null defaults
    to 0, the panel's own unmarked seed, rather than being passed through.
    """
    path = os.path.join(checkpoints_dir, folder, "args.json")
    with open(path) as handle:
        metadata = json.load(handle)
    metadata["folder"] = folder
    if metadata.get("seed") is None:
        metadata["seed"] = 0
    return metadata


def load_sweep_placements(declaration_path: str) -> list[dict]:
    """The 2 headline placements' own (position, operator, rate ladder), from the basis."""
    with open(declaration_path) as handle:
        basis = json.load(handle)["basis"]
    by_id = {entry["id"]: entry for entry in basis}
    placements = [by_id[placement_id] for placement_id in SWEEP_PLACEMENT_IDS]
    return placements


def train_argv(
    metadata: dict, folder: str, extra: list[str], batch_size: int | None = None
) -> list[str]:
    """The training command's flags, as an argv list, shared by every variant.

    extra carries whatever makes 1 variant different from the base recipe
    (--evade-* for idea A, --record-sample-loss or --exclude-indices-file for
    idea B), inserted before --output so it never has to know that flag's value.
    """
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
    ]
    if batch_size is not None:
        argv += ["--batch-size", str(batch_size)]
    argv += extra
    argv += ["--output", f"checkpoints/{folder}/attack_result.pt"]
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

    Groups a flag with the values that follow it onto 1 line, matching the style
    of pbs/generate_seed_jobs.py and pbs/generate_adaptive_attacker_jobs.py's
    emitted commands.
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
    """Raise with the offending command if argparse itself would reject argv."""
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


def a_run_minutes(metadata: dict, placements: list[dict]) -> float:
    """1 idea-A variant's training cost plus its own sweep and analyze."""
    training = PLAIN_MINUTES[metadata["dataset"]] * EVASION_MULTIPLIER
    total = training + sweep_analyze_minutes(placements)
    return total


def b_run_minutes(metadata: dict, placements: list[dict], flagging: bool) -> float:
    """1 idea-B stage's training cost plus its own sweep, analyze and flagging if any."""
    training = PLAIN_MINUTES[metadata["dataset"]]
    total = training + sweep_analyze_minutes(placements)
    if flagging:
        total += FLAG_MINUTES
    return total


def commands_for_a_run(
    metadata: dict, suffix: str, extra: list[str], placements
) -> str:
    """The training, sweep and analyze commands for 1 idea-A variant, as 1 block."""
    folder = f"{metadata['folder']}{suffix}"
    blocks = [
        f'echo "=== {folder} ==="',
        render_call(
            "cli.train_backdoor",
            train_argv(metadata, folder, extra, batch_size=EVASION_BATCH_SIZE),
        ),
    ]
    for placement in placements:
        blocks.append(render_call("cli.sweep", sweep_argv(folder, placement)))
    blocks.append(render_call("cli.analyze", analyze_argv(folder)))
    return "\n".join(blocks)


def commands_for_b1_run(metadata: dict, placements) -> str:
    """B1: the base recipe plus --record-sample-loss, then the headline sweeps."""
    folder = f"{metadata['folder']}{B1_SUFFIX}"
    blocks = [
        f'echo "=== {folder} ==="',
        render_call(
            "cli.train_backdoor",
            train_argv(metadata, folder, ["--record-sample-loss"]),
        ),
    ]
    for placement in placements:
        blocks.append(render_call("cli.sweep", sweep_argv(folder, placement)))
    blocks.append(render_call("cli.analyze", analyze_argv(folder)))
    return "\n".join(blocks)


def commands_for_b2_run(metadata: dict, placements) -> str:
    """B2: flag_by_loss.py on B1's own folder, then a retrain with those samples dropped."""
    base_folder = metadata["folder"]
    folder = f"{base_folder}{B2_SUFFIX}"
    flagged_path = f"checkpoints/{base_folder}{B1_SUFFIX}/flagged_indices.json"
    blocks = [
        f'echo "=== {folder} ==="',
        f"python experiments/training_aware/flag_by_loss.py {base_folder}\n",
        render_call(
            "cli.train_backdoor",
            train_argv(metadata, folder, ["--exclude-indices-file", flagged_path]),
        ),
    ]
    for placement in placements:
        blocks.append(render_call("cli.sweep", sweep_argv(folder, placement)))
    blocks.append(render_call("cli.analyze", analyze_argv(folder)))
    return "\n".join(blocks)


def pack(
    costed_commands: list[tuple[float, str]], target_minutes: float
) -> list[list[str]]:
    """First-fit decreasing bin packing: largest runs placed first, into the first job with room.

    A run that alone exceeds target_minutes still gets its own bin, exactly as
    pbs/generate_adaptive_attacker_jobs.py's own pack() falls back when a single
    item is larger than the budget.
    """
    ordered = sorted(costed_commands, key=lambda item: -item[0])
    bins: list[tuple[float, list[str]]] = []
    for cost, command in ordered:
        for position, (used, group) in enumerate(bins):
            if used + cost <= target_minutes:
                bins[position] = (used + cost, group + [command])
                break
        else:
            bins.append((cost, [command]))
    jobs = [group for _, group in bins]
    return jobs


def b_stage_groups(settings_metadata: list[dict]) -> list[list[dict]]:
    """A deterministic 2-way split (tiny apart from the rest), shared by B1 and B2.

    B2's PBS dependency has to name the exact B1 job a setting's sample_loss.npz
    came from, so B1 and B2 must group settings identically. A fixed split by
    dataset, rather than a generic bin-packer run twice, guarantees that without
    threading state between 2 separate invocations of this script.
    """
    tiny = [metadata for metadata in settings_metadata if metadata["dataset"] == "tiny"]
    other = [
        metadata for metadata in settings_metadata if metadata["dataset"] != "tiny"
    ]
    groups = [group for group in (other, tiny) if group]
    return groups


def build_stage_a(settings_metadata: list[dict], placements, hours: float):
    """Every idea-A run gets its own job: even the cheapest pairing (2 cifar-family
    variants) exceeds a 0.9-margin 12h budget, so packing never actually combines
    2 runs and a dedicated bin per run is simplest and exactly as safe.
    """
    costed = [
        (
            a_run_minutes(metadata, placements),
            commands_for_a_run(metadata, suffix, extra, placements),
        )
        for metadata in settings_metadata
        for suffix, extra in IDEA_A_VARIANTS
    ]
    jobs = pack(costed, hours * 60 * 0.9)
    return jobs, [cost for cost, _ in costed]


def build_stage_b1(settings_metadata: list[dict], placements, hours: float):
    groups = b_stage_groups(settings_metadata)
    jobs = [
        [commands_for_b1_run(metadata, placements) for metadata in group]
        for group in groups
    ]
    costs = [
        b_run_minutes(metadata, placements, flagging=False)
        for metadata in settings_metadata
    ]
    return jobs, costs


def build_stage_b2(
    settings_metadata: list[dict], placements, hours: float, depends_on: dict
):
    groups = b_stage_groups(settings_metadata)
    jobs = []
    depend_lines = []
    for group in groups:
        commands = [commands_for_b2_run(metadata, placements) for metadata in group]
        jobs.append(commands)
        ids = sorted(
            {
                depends_on[metadata["folder"]]
                for metadata in group
                if metadata["folder"] in depends_on
            }
        )
        depend_lines.append(f"#PBS -W depend=afterok:{':'.join(ids)}\n" if ids else "")
    costs = [
        b_run_minutes(metadata, placements, flagging=True)
        for metadata in settings_metadata
    ]
    return jobs, costs, depend_lines


def write_jobs(
    jobs: list[list[str]],
    stage: str,
    out_dir: str,
    hours: float,
    depend_lines: list[str] | None = None,
) -> list[str]:
    """1 job_NNN.pbs per job group, plus submit_all.sh, under out_dir/<stage>/."""
    stage_dir = os.path.join(out_dir, stage)
    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE, "logs", "training_aware", stage), exist_ok=True)

    written = []
    for index, group in enumerate(jobs, start=1):
        depend_line = depend_lines[index - 1] if depend_lines else ""
        commands = "\n".join(group)
        path = os.path.join(stage_dir, f"job_{index:03d}.pbs")
        with open(path, "w") as handle:
            handle.write(
                JOB_TEMPLATE.format(
                    base=BASE,
                    stage=stage,
                    index=index,
                    walltime=f"{int(hours):02d}:00:00",
                    depend_line=depend_line,
                    commands=commands,
                )
            )
        written.append(path)

    submit_path = os.path.join(stage_dir, "submit_all.sh")
    with open(submit_path, "w") as handle:
        handle.write("#!/bin/bash\n")
        for path in written:
            handle.write(f"qsub {path}\n")
    return written


def validate_all(
    settings_metadata: list[dict], placements, stage: str, depends_on: dict
) -> None:
    """Every command this generator would emit for `stage`, checked against its real parser."""
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    from cli import analyze, sweep, train_backdoor

    for metadata in settings_metadata:
        base_folder = metadata["folder"]
        if stage == "a":
            for suffix, extra in IDEA_A_VARIANTS:
                folder = f"{base_folder}{suffix}"
                validate_argv(
                    train_backdoor.parse_args,
                    train_argv(metadata, folder, extra, batch_size=EVASION_BATCH_SIZE),
                    "train_backdoor",
                )
                for placement in placements:
                    validate_argv(
                        sweep.parse_args, sweep_argv(folder, placement), "sweep"
                    )
                validate_argv(analyze.parse_args, analyze_argv(folder), "analyze")
        elif stage == "b1":
            folder = f"{base_folder}{B1_SUFFIX}"
            validate_argv(
                train_backdoor.parse_args,
                train_argv(metadata, folder, ["--record-sample-loss"]),
                "train_backdoor",
            )
            for placement in placements:
                validate_argv(sweep.parse_args, sweep_argv(folder, placement), "sweep")
            validate_argv(analyze.parse_args, analyze_argv(folder), "analyze")
        elif stage == "b2":
            folder = f"{base_folder}{B2_SUFFIX}"
            flagged_path = f"checkpoints/{base_folder}{B1_SUFFIX}/flagged_indices.json"
            validate_argv(
                train_backdoor.parse_args,
                train_argv(metadata, folder, ["--exclude-indices-file", flagged_path]),
                "train_backdoor",
            )
            for placement in placements:
                validate_argv(sweep.parse_args, sweep_argv(folder, placement), "sweep")
            validate_argv(analyze.parse_args, analyze_argv(folder), "analyze")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("a", "b1", "b2"))
    parser.add_argument("--checkpoints-dir", default=os.path.join(BASE, "checkpoints"))
    parser.add_argument(
        "--declaration", default=os.path.join(BASE, "configs/psbd_basis.json")
    )
    parser.add_argument(
        "--out-dir", default=os.path.join(BASE, "pbs", "training_aware")
    )
    parser.add_argument(
        "--depends-on",
        default=None,
        help="stage b2 only: JSON file mapping a base folder (e.g. "
        "vit_cifar10_tact_0_05) to the B1 PBS job id that trained its "
        "_lossrec checkpoint. Required to write real job files, optional for "
        "--dry-run.",
    )
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    return arguments


def main() -> None:
    args = parse_args()
    settings_metadata = [
        load_metadata(args.checkpoints_dir, folder) for folder in SETTINGS
    ]
    placements = load_sweep_placements(args.declaration)

    depends_on = {}
    if args.depends_on:
        with open(args.depends_on) as handle:
            depends_on = json.load(handle)
    if args.stage == "b2" and not depends_on and not args.dry_run:
        raise ValueError(
            "--stage b2 needs --depends-on pointing at a base-folder-to-B1-job-id "
            "map, or --dry-run"
        )

    validate_all(settings_metadata, placements, args.stage, depends_on)

    if args.stage == "a":
        jobs, costs = build_stage_a(settings_metadata, placements, args.hours)
        depend_lines = None
        n_runs = len(settings_metadata) * len(IDEA_A_VARIANTS)
    elif args.stage == "b1":
        jobs, costs = build_stage_b1(settings_metadata, placements, args.hours)
        depend_lines = None
        n_runs = len(settings_metadata)
    else:
        jobs, costs, depend_lines = build_stage_b2(
            settings_metadata, placements, args.hours, depends_on
        )
        n_runs = len(settings_metadata)

    total_minutes = sum(costs)
    print(f"stage                {args.stage}")
    print(f"settings             {len(settings_metadata)}")
    print(f"training runs        {n_runs}")
    print(f"jobs at {args.hours}h       {len(jobs)}")
    print(f"estimated GPU time   {total_minutes / 60:.1f} hours")

    if args.dry_run:
        print("\ndry run, no files written\n")
        for index, group in enumerate(jobs, start=1):
            print(f"job_{index:03d}:")
            for command in group:
                print(command)
        return

    written = write_jobs(jobs, args.stage, args.out_dir, args.hours, depend_lines)
    print(
        f"\nwrote {len(written)} job files to {os.path.join(args.out_dir, args.stage)}"
    )
    print(
        f"submit with          bash {os.path.join(args.out_dir, args.stage, 'submit_all.sh')}"
    )


if __name__ == "__main__":
    main()
