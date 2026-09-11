"""Emit training jobs for the 3-probe union attacker.

H41 (`docs/hypothesis/H41-multi-probe-defence.md`) defeated the single-probe
adaptive attacker from H25 with a min-rank union of 3 probes: token masking at
`before_attention_norm`, dropout at `before_attention_norm` and gain scaling at
`mlp_norm_out`. Every existing evasive checkpoint on disk (the `_evade_l1`
folders) was trained against only the first of those 3, so the union's
apparent win could just be an attacker that never saw the other 2 members. This
generator closes that gap: it retrains against the exact union
(`attacks.evasion`'s multi-probe hinge, `cli.train_backdoor --evade-probes`),
so a reviewer asking "did you train against the defence you're claiming
defeats the attacker" gets a yes.

Retrains the same 14 CIFAR-100 cells the single-probe attacker used (badnet_a2o,
blend, bpp, lf and adaptive_blend at 0.01, 0.05 and 0.1, 15 combinations minus
`adaptive_blend` at 1%, whose single-probe evasive checkpoint never cleared
ASR > 0.9, the bar H41's own evaluation applies), then sweeps and analyzes each
result at the 2 headline placements: the deployed recommendation
(`before_attention_norm` token_mask) and the published ConvNet placement
(`post_residual` dropout), exactly as `pbs/generate_adaptive_attacker_jobs.py`
does for the paper's own single-term adaptive attacker. Whether the union
itself still separates clean from poisoned on these checkpoints is a question
for a follow-up measurement over the resulting caches, not for this generator.

Reads each cell's training arguments back from its own checkpoints/<folder>/
args.json, exactly as pbs/generate_seed_jobs.py and
pbs/generate_adaptive_attacker_jobs.py do, so a rerun cannot silently differ
from the base run in a way nobody notices. The only fields that change are the
evasion flags and the output folder.

Every command is validated against the real argparse parsers
(cli.train_backdoor, cli.sweep, cli.analyze) before anything is written, so a
flag typo fails here rather than after a job has queued for hours.

    python pbs/generate_union_attacker_jobs.py --dry-run
    python pbs/generate_union_attacker_jobs.py
    bash pbs/vit_union_attacker/submit_all.sh
"""

import argparse
import json
import os
import sys

BASE = "/lustre/home/pstika/projects/PSBD-ViT"

# The single-probe attacker's own CIFAR-100 panel (badnet_a2o, blend, bpp, lf,
# adaptive_blend at 1%, 5%, 10%), minus adaptive_blend at 1%: its `_evade_l1`
# checkpoint's ASR is 0.800, below the 0.9 bar H41's evaluation holds every
# evasive checkpoint to, so it was never part of "the models the single-probe
# attacker used" in the sense that matters here. 14 cells.
CELLS = (
    "vit_cifar100_badnet_a2o_0_01",
    "vit_cifar100_badnet_a2o_0_05",
    "vit_cifar100_badnet_a2o_0_1",
    "vit_cifar100_blend_0_01",
    "vit_cifar100_blend_0_05",
    "vit_cifar100_blend_0_1",
    "vit_cifar100_bpp_0_01",
    "vit_cifar100_bpp_0_05",
    "vit_cifar100_bpp_0_1",
    "vit_cifar100_lf_0_01",
    "vit_cifar100_lf_0_05",
    "vit_cifar100_lf_0_1",
    "vit_cifar100_adaptive_blend_0_05",
    "vit_cifar100_adaptive_blend_0_1",
)

# H41's 3-probe pool, minus PSBD-RD and gaussian (`docs/hypothesis/
# H41-multi-probe-defence.md`'s probe table, rows 1-3): the union this
# generator trains against. --evade-weight matches the single-probe
# attacker's own weight (the "_l1" folders' name), so the only variable
# between that attacker and this one is how many probes it was trained
# against.
EVADE_PROBES = (
    "before_attention_norm:token_mask",
    "before_attention_norm:dropout",
    "mlp_norm_out:gain_scale",
)
EVADE_WEIGHT = 1.0
FOLDER_SUFFIX = "_evade_union"

# The 2 placements every trained checkpoint gets swept and analyzed at: the
# deployed recommendation and the published ConvNet placement it is compared
# against. Read from the basis declaration rather than restated, so the rate
# ladder used here can never drift from the one the rest of the panel uses.
SWEEP_PLACEMENT_IDS = ("before_attention_norm_token_mask", "post_residual")

# The union-attacker smoke test (experiments/union_attacker_smoke/README.md)
# found that this 3-probe hinge retains roughly 1 shared base forward plus
# passes=3 stochastic passes PER probe (1 + 3*3 = 10 forward graphs alive at
# once for backward, against 1 + 3 = 4 for a single probe, see
# attacks.evasion's module docstring), which OOM'd a 40GB A100 at batch 64 and
# at batch 32. Batch 16 was the first size that fit, so every training command
# this generator emits pins --batch-size 16 rather than the argparse default.
BATCH_SIZE = 16

# Smoke-measured cost at batch 16: 1 epoch over 8000 CIFAR-100 images (500
# steps) plus 1-time probe calibration (3 probes, 7 candidate rates each) and
# the final ASR/CA eval took 19.6 minutes end to end. Calibration and eval are
# a roughly fixed cost independent of epoch count or dataset size. Splitting
# it out at about 11 of those 19.6 minutes leaves about 8.6 minutes for the
# 500 training steps, or about 1.03 seconds per step. A full run trains 15
# epochs over the full 50000-image CIFAR-100 train set at batch 16 (3125
# steps/epoch, 46875 steps total), so:
#
#     training minutes = CALIBRATION_EVAL_MINUTES + PER_STEP_MINUTES * steps
#
# This lands at roughly 13.6 hours per cell, well above the 6x-a-plain-run
# guess (about 8.4 hours) this generator carried before the smoke test ran:
# a 3-probe union at batch 16 pays far more in step COUNT (8x the steps of a
# batch-128 plain run at the same epoch count) than a single-probe hinge run
# at batch 48 or 64 ever did.
CALIBRATION_EVAL_MINUTES = 11.0
PER_STEP_MINUTES = 1.03 / 60.0
FULL_TRAIN_IMAGES = 50000
EPOCHS_PER_CELL = 15

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
#PBS -N vit_union_attacker_{index:03d}
#PBS -o {base}/logs/vit_union_attacker/job_{index:03d}.log
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


def new_folder_name(base_folder: str) -> str:
    """The output checkpoint folder for a single cell's union-attacker run."""
    folder = f"{base_folder}{FOLDER_SUFFIX}"
    return folder


def load_metadata(checkpoints_dir: str, folder: str) -> dict:
    """A cell's own training arguments, read back from its args.json.

    A handful of the older CIFAR-100 checkpoints in this panel recorded
    `seed: null`, a provenance gap from before the seed was always written,
    not an unset seed: none of these folder names carry a `_seed_{N}` tag, and
    "checkpoints/checkpoint naming and metadata" in CLAUDE.md is explicit that
    an unmarked folder is seed 0. Coalesced here so every downstream command
    gets a real int rather than argparse's `int(None)`.
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


def discover_runs(cells: tuple[str, ...], checkpoints_dir: str) -> list[dict]:
    """Every cell's training metadata, 1 run per cell (the union has no alpha sweep)."""
    runs = [load_metadata(checkpoints_dir, folder) for folder in cells]
    return runs


def train_argv(metadata: dict, folder: str) -> list[str]:
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
        "--batch-size",
        str(BATCH_SIZE),
        "--seed",
        str(metadata["seed"]),
        "--evade-psbd",
        "--evade-probes",
        *EVADE_PROBES,
        "--evade-weight",
        str(EVADE_WEIGHT),
        "--evade-objective",
        "hinge",
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
    `--evade-probes a b c` reads as 1 line rather than 1 token per line),
    matching the style of pbs/generate_seed_jobs.py and
    pbs/generate_adaptive_attacker_jobs.py's emitted commands.
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


def run_minutes(placements: list[dict]) -> float:
    """1 cell's total cost: union-attacker training plus its own sweep and analyze.

    Training minutes are the smoke-measured model (CALIBRATION_EVAL_MINUTES's
    module-level comment): a fixed calibration-and-eval cost plus a per-step
    cost times the full-dataset step count at batch 16.
    """
    steps = EPOCHS_PER_CELL * (FULL_TRAIN_IMAGES / BATCH_SIZE)
    training = CALIBRATION_EVAL_MINUTES + PER_STEP_MINUTES * steps
    total = training + sweep_analyze_minutes(placements)
    return total


def pack(
    runs: list[dict], placements: list[dict], target_minutes: float
) -> list[list[dict]]:
    """First-fit decreasing: largest runs placed first, into the first job with room.

    A run that alone exceeds target_minutes still gets its own job rather than
    being dropped, exactly as pbs/generate_seed_jobs.py's own pack() falls back
    when a single item is larger than the budget. Every cell costs the same
    here (no alpha sweep), so this only matters when target_minutes is set
    below a single run's own cost.
    """
    cost = run_minutes(placements)
    bins: list[tuple[float, list[dict]]] = []
    for metadata in runs:
        for position, (used, group) in enumerate(bins):
            if used + cost <= target_minutes:
                bins[position] = (used + cost, group + [metadata])
                break
        else:
            bins.append((cost, [metadata]))
    jobs = [group for _, group in bins]
    return jobs


def commands_for_run(metadata: dict, placements: list[dict]) -> str:
    """The training, sweep and analyze commands for 1 cell's run, as 1 block."""
    folder = new_folder_name(metadata["folder"])
    blocks = [
        f'echo "=== {folder} ==="',
        render_call("cli.train_backdoor", train_argv(metadata, folder)),
    ]
    for placement in placements:
        blocks.append(render_call("cli.sweep", sweep_argv(folder, placement)))
    blocks.append(render_call("cli.analyze", analyze_argv(folder)))
    commands = "\n".join(blocks)
    return commands


def validate_all(runs: list[dict], placements: list[dict]) -> None:
    """Every command this generator would emit, checked against its real parser."""
    # cli.* only resolves from the repository root, and this script is run as a
    # plain file (`python pbs/generate_union_attacker_jobs.py`) rather than
    # `python -m`, so BASE is added to sys.path here rather than assumed.
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    from cli import analyze, sweep, train_backdoor

    for metadata in runs:
        folder = new_folder_name(metadata["folder"])
        validate_argv(
            train_backdoor.parse_args, train_argv(metadata, folder), "train_backdoor"
        )
        for placement in placements:
            validate_argv(sweep.parse_args, sweep_argv(folder, placement), "sweep")
        validate_argv(analyze.parse_args, analyze_argv(folder), "analyze")


def write_jobs(
    jobs: list[list[dict]], placements: list[dict], out_dir: str, hours: float
) -> list[str]:
    """1 vit_union_attacker_NNN.pbs per packed job, plus submit_all.sh."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE, "logs", "vit_union_attacker"), exist_ok=True)

    written = []
    for index, group in enumerate(jobs, start=1):
        commands = "\n".join(
            commands_for_run(metadata, placements) for metadata in group
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
        "--out-dir", default=os.path.join(BASE, "pbs", "vit_union_attacker")
    )
    # 1 cell alone costs about 13.9 hours (run_minutes), so the default must
    # clear that with the 0.9 packing margin pack() applies, or every job
    # would silently get its own oversized bin instead of the intended budget.
    parser.add_argument("--hours", type=float, default=16.0)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    return arguments


def main() -> None:
    args = parse_args()
    runs = discover_runs(CELLS, args.checkpoints_dir)
    placements = load_sweep_placements(args.declaration)
    validate_all(runs, placements)

    jobs = pack(runs, placements, args.hours * 60 * 0.9)
    total_minutes = sum(run_minutes(placements) for _ in runs)

    print(f"cells             {len(CELLS)}")
    print(f"evade probes      {EVADE_PROBES}")
    print(f"evade weight      {EVADE_WEIGHT}")
    print(f"sweep placements  {SWEEP_PLACEMENT_IDS}")
    print(f"jobs at {args.hours}h      {len(jobs)}")
    print(f"estimated GPU time   {total_minutes / 60:.1f} hours")
    print(
        "longest job          "
        f"{max(sum(run_minutes(placements) for _ in group) for group in jobs) / 60:.1f} hours"
    )

    if args.dry_run:
        print("\ndry run, no files written\n")
        for index, group in enumerate(jobs, start=1):
            job_minutes = sum(run_minutes(placements) for _ in group)
            print(f"job_{index:03d} ({job_minutes / 60:.1f}h):")
            for metadata in group:
                folder = new_folder_name(metadata["folder"])
                print(f"  {folder}")
                print(commands_for_run(metadata, placements))
        return

    written = write_jobs(jobs, placements, args.out_dir, args.hours)
    print(f"\nwrote {len(written)} job files to {args.out_dir}")
    print(f"submit with          bash {args.out_dir}/submit_all.sh")


if __name__ == "__main__":
    main()
