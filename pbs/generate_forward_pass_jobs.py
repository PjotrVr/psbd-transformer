"""A forward pass (k) ladder for the 2 headline placements.

PSU is an expectation over k stochastic forward passes, and every cached
placement on disk so far was swept at the paper's own k=3. This generator adds
k in {1, 5, 10, 20} for the 2 placements the panel headline compares:
RECOMMENDED_PLACEMENT (before_attention_norm, token_mask) and
PUBLISHED_PLACEMENT (post_residual, dropout), both read from
defences.decision so the placement identity cannot drift from the canon. The
checkpoints are the ViT cells in results/coverage/coverage.json that clear the
ASR bar, restricted to cifar100 and tiny at poison rate 0.01 and 0.05, since
those 2 datasets are the project's primary evidence base. Ordering favors the
hard attacks first (bpp, wanet, tact, sig) over the easier family
(badnet_a2o, blend, lf), so a batch cut short by --max-jobs still keeps the
attacks the panel judges detection on.

cli.sweep already names a k != 3 cache with a _k<passes> suffix
(cache_config_name in cli/sweep.py), and already_complete already checks every
rate and split tensor for a placement. Both are imported and reused here
rather than re-derived, so a placement already complete at some k is skipped
before it is ever packed into a job.

Cost model: a k=3 sweep of 1 placement over its full rate ladder and all 3
splits measures 8.5 minutes per checkpoint on 1 A100, and PSU's k forward
passes scale that linearly, so a k=20 sweep costs 8.5 * 20 / 3, about 57
minutes. Invocations are packed sequentially into jobs of --minutes-per-job
(under the fixed 12 hour walltime), preserving the attack-priority order, and
each job ends with a cli.analyze call over the checkpoints it touched so
psbd_metrics.json is refreshed alongside the new cache.

    python pbs/generate_forward_pass_jobs.py --dry-run
    python pbs/generate_forward_pass_jobs.py --max-jobs 4
    bash pbs/vit_forward_passes/submit_all.sh
"""

import argparse
from dataclasses import dataclass
import json
import os
import re
import sys
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import cli.sweep as sweep_cli  # noqa: E402
from defences.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402

PROJECT_ROOT = REPO

# The panel restriction: cifar100 and tiny are the project's primary datasets,
# at the 2 poison rates the hard-baseline comparison is judged on.
DATASETS: tuple[str, ...] = ("cifar100", "tiny")
POISON_RATES: frozenset[float] = frozenset({0.01, 0.05})

# Hard triggers first, the easier family after. A name absent from this tuple
# (there is none in the current panel) sorts last rather than raising.
ATTACK_PRIORITY: tuple[str, ...] = (
    "bpp",
    "wanet",
    "tact",
    "sig",
    "badnet_a2o",
    "blend",
    "lf",
)

K_LADDER: tuple[int, ...] = (1, 5, 10, 20)

# Measured: a k=3 sweep of 1 placement (its full rate ladder, all 3 splits) on
# 1 A100. PSU's k forward passes are the only thing that scales this, so cost
# tracks k linearly from that anchor.
MINUTES_PER_CHECKPOINT_AT_K3: float = 8.5

WALLTIME = "12:00:00"
# Packing budget under the fixed 12 hour walltime, with headroom for a job
# killed at the wall to lose as little finished work as possible.
DEFAULT_MINUTES_PER_JOB = 680.0

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


@dataclass(frozen=True)
class Placement:
    """1 headline placement, read from the basis declaration rather than typed in."""

    id: str
    position: str
    operator: str
    rates: tuple[float, ...]


@dataclass(frozen=True)
class Invocation:
    """1 sweep call: a checkpoint, a placement, a pass count and its cost."""

    folder: str
    placement: Placement
    forward_passes: int
    minutes: float
    cache_name: str


def load_json(path: str) -> dict:
    with open(path) as handle:
        loaded = json.load(handle)
    return loaded


def load_placement(basis: list[dict], placement_id: str) -> Placement:
    """The position, operator and rate ladder the canon declares for placement_id."""
    for entry in basis:
        if entry["id"] == placement_id:
            return Placement(
                id=placement_id,
                position=entry["position"],
                operator=entry["operator"],
                rates=tuple(entry["rates"]),
            )
    raise KeyError(f"{placement_id!r} not found in the basis declaration")


def attack_sort_key(attack: str) -> int:
    rank = (
        ATTACK_PRIORITY.index(attack)
        if attack in ATTACK_PRIORITY
        else len(ATTACK_PRIORITY)
    )
    return rank


def select_checkpoints(coverage: dict) -> list[dict]:
    """The ViT cells the ladder covers, hard attacks first.

    Restricted to cifar100 and tiny at 1% and 5% poisoning, and to cells whose
    asr_class is "clears": a below-bar or diverged cell has no working backdoor
    or no usable model, so a forward pass ladder over it would only measure how
    finely an absent signal can be resolved.
    """
    matching = [
        cell
        for cell in coverage["cells"]
        if cell["dataset"] in DATASETS
        and cell["asr_class"] == "clears"
        and round(cell["poison_rate"], 4) in POISON_RATES
    ]
    ordered = sorted(
        matching,
        key=lambda cell: (
            attack_sort_key(cell["attack"]),
            cell["dataset"],
            cell["poison_rate"],
            cell["folder_name"],
        ),
    )
    return ordered


def invocation_minutes(forward_passes: int) -> float:
    minutes = (
        MINUTES_PER_CHECKPOINT_AT_K3 * forward_passes / sweep_cli.DEFAULT_FORWARD_PASSES
    )
    return minutes


def plan_invocations(
    checkpoints: list[dict],
    placements: list[Placement],
    k_ladder: tuple[int, ...],
    results_dir: str,
) -> tuple[list[Invocation], list[tuple[str, str, int]]]:
    """Every (checkpoint, placement, k) that is not already complete on disk.

    already_complete is cli.sweep's own gate for --skip-existing, so reusing it
    here means a cache this generator calls finished and a cache the sweep
    itself would skip can never disagree.
    """
    planned = []
    already_cached = []
    for checkpoint in checkpoints:
        folder = checkpoint["folder_name"]
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        for placement in placements:
            for forward_passes in k_ladder:
                cache_name = sweep_cli.cache_config_name(
                    placement.position, None, placement.operator, forward_passes
                )
                if sweep_cli.already_complete(psbd_dir, cache_name, placement.rates):
                    already_cached.append((folder, placement.id, forward_passes))
                    continue
                planned.append(
                    Invocation(
                        folder=folder,
                        placement=placement,
                        forward_passes=forward_passes,
                        minutes=invocation_minutes(forward_passes),
                        cache_name=cache_name,
                    )
                )
    return planned, already_cached


def sweep_parser() -> argparse.ArgumentParser:
    """cli.sweep's own ArgumentParser, captured without running its side effects.

    cli.sweep exposes only parse_args, which builds the parser and immediately
    consumes sys.argv. Patching ArgumentParser.parse_args to record self before
    delegating to the original recovers the parser instance without cli/sweep.py
    ever being touched, so the flags checked here are the exact ones cli.sweep
    enforces at run time rather than a hand-copied guess.
    """
    captured: dict[str, argparse.ArgumentParser] = {}
    original_parse_args = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        return original_parse_args(self, *args, **kwargs)

    argparse.ArgumentParser.parse_args = capture
    minimal_argv = [
        "cli.sweep",
        "--checkpoint-folder",
        "probe",
        "--position",
        "post_residual",
    ]
    try:
        with mock.patch.object(sys, "argv", minimal_argv):
            sweep_cli.parse_args()
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args
    return captured["parser"]


def render_sweep_command(inv: Invocation) -> str:
    rates = " ".join(str(rate) for rate in inv.placement.rates)
    lines = [
        f'echo "=== {inv.folder} :: {inv.placement.id} :: k={inv.forward_passes} ==="',
        "python -m cli.sweep \\",
        f"    --checkpoint-folder {inv.folder} \\",
        f"    --position {inv.placement.position} \\",
        f"    --operator {inv.placement.operator} \\",
        f"    --rates {rates} \\",
        f"    --forward-passes {inv.forward_passes} \\",
        "    --skip-existing",
    ]
    command = "\n".join(lines)
    return command


def emitted_sweep_flags(body: str) -> set[str]:
    """Every --flag token of the cli.sweep commands in a job body.

    Only a line starting a sweep invocation or continuing it with a leading
    "--" is read, so the echo lines and the shell itself never count against
    cli.sweep's own parser.
    """
    flags = set()
    for line in body.split("\n"):
        stripped = line.strip()
        if not (stripped.startswith("--") or "python -m cli.sweep" in stripped):
            continue
        flags.update(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", stripped))
    return flags


def verify_sweep_flags(body: str) -> None:
    """Refuse a job body whose commands name a flag cli.sweep does not accept."""
    accepted = set(sweep_parser()._option_string_actions)
    unknown = sorted(emitted_sweep_flags(body) - accepted)
    if unknown:
        raise SystemExit(f"emitted flags cli.sweep rejects: {unknown}")


def pack_invocations(
    invocations: list[Invocation], minutes_per_job: float
) -> list[list[Invocation]]:
    """Consecutive invocations packed into jobs of at most minutes_per_job.

    Sequential rather than a bin-packing search, so the attack-priority order
    of the input list survives into the batch: a job cut by --max-jobs still
    keeps the earliest, hardest attacks.
    """
    jobs: list[list[Invocation]] = []
    current: list[Invocation] = []
    current_minutes = 0.0
    for inv in invocations:
        if current and current_minutes + inv.minutes > minutes_per_job:
            jobs.append(current)
            current, current_minutes = [], 0.0
        current.append(inv)
        current_minutes += inv.minutes
    if current:
        jobs.append(current)
    return jobs


def render_job(
    index: int, batch: str, invocations: list[Invocation]
) -> tuple[str, str]:
    """1 job script and its file name, its commands checked against cli.sweep first."""
    name = f"forward_pass_{index}"
    body = "\n\n".join(render_sweep_command(inv) for inv in invocations)
    verify_sweep_flags(body)
    folders = " ".join(sorted({inv.folder for inv in invocations}))
    script = JOB_TEMPLATE.format(
        walltime=WALLTIME,
        name=name,
        root=PROJECT_ROOT,
        batch=batch,
        body=body,
        folders=folders,
    )
    return name, script


def write_jobs(bundles: list[list[Invocation]], batch: str) -> list[str]:
    out_dir = os.path.join(PROJECT_ROOT, "pbs", batch)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "logs", batch), exist_ok=True)
    written = []
    for index, invocations in enumerate(bundles, start=1):
        name, script = render_job(index, batch, invocations)
        path = os.path.join(out_dir, f"{name}.pbs")
        with open(path, "w") as handle:
            handle.write(script)
        written.append(path)
    submit_path = os.path.join(out_dir, "submit_all.sh")
    with open(submit_path, "w") as handle:
        handle.write("#!/bin/bash\n")
        for path in written:
            handle.write(f"qsub {path}\n")
    return written


def print_plan(
    checkpoints: list[dict],
    placements: list[Placement],
    planned: list[Invocation],
    already_cached: list[tuple[str, str, int]],
) -> None:
    print(
        f"[ok] {len(checkpoints)} checkpoints (cifar100 and tiny, 1% and 5%, asr_class clears)"
    )
    print(f"     placements        {', '.join(p.id for p in placements)}")
    print(f"     k ladder          {', '.join(str(k) for k in K_LADDER)}")
    print(
        f"     invocations       {len(planned)} planned, "
        f"{len(already_cached)} already cached and skipped"
    )
    gpu_hours = sum(inv.minutes for inv in planned) / 60.0
    print(f"     estimated GPU     {gpu_hours:.1f} hours")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage",
        default=os.path.join(PROJECT_ROOT, "results", "coverage", "coverage.json"),
    )
    parser.add_argument(
        "--declaration",
        default=os.path.join(PROJECT_ROOT, "configs", "psbd_basis.json"),
    )
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--batch", default="vit_forward_passes")
    parser.add_argument(
        "--minutes-per-job",
        type=float,
        default=DEFAULT_MINUTES_PER_JOB,
        help="packing budget per job, under the fixed 12 hour walltime",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=None, help="write only the first N packed jobs"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coverage = load_json(args.coverage)
    basis = load_json(args.declaration)["basis"]
    placements = [
        load_placement(basis, RECOMMENDED_PLACEMENT),
        load_placement(basis, PUBLISHED_PLACEMENT),
    ]

    checkpoints = select_checkpoints(coverage)
    planned, already_cached = plan_invocations(
        checkpoints, placements, K_LADDER, args.results_dir
    )
    print_plan(checkpoints, placements, planned, already_cached)

    bundles = pack_invocations(planned, args.minutes_per_job)

    if args.dry_run:
        print("\nplanned invocations:")
        for inv in planned:
            print(
                f"  {inv.folder:34s} {inv.placement.id:34s} k={inv.forward_passes:<3d} "
                f"~{inv.minutes:6.1f} min  -> psbd/{inv.cache_name}/"
            )
        print(f"\n{len(bundles)} jobs at up to {args.minutes_per_job:.0f} minutes each")
        print(f"walltime {WALLTIME} per job")
        print("(dry run, nothing written)")
        return

    total_jobs = len(bundles)
    if args.max_jobs is not None and args.max_jobs < total_jobs:
        bundles = bundles[: args.max_jobs]
        print(f"[max-jobs] writing {len(bundles)} of {total_jobs} packed jobs")

    written = write_jobs(bundles, args.batch)
    for path in written:
        print(f"  wrote {os.path.relpath(path, PROJECT_ROOT)}")
    submit = os.path.join(PROJECT_ROOT, "pbs", args.batch, "submit_all.sh")
    print(f"submit with   bash {os.path.relpath(submit, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
