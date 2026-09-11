"""The full basis declaration, swept on the backdoored Swin-S checkpoints.

The ViT ranking in the paper is read off the panel in configs/psbd_basis.json, every
cell carrying every basis placement. The Swin ranking has no such guarantee: whatever
happens to be cached there today is whatever a one-off script asked for. This generator
gives Swin the same treatment, submitting the declared basis against the Swin checkpoints
that clear the ASR bar rather than a hand-picked subset of placements.

configs/psbd_basis.json names a position, not an architecture: models.positions.
POSITION_REGISTRY carries a separate table per architecture, and a basis position with no
matching key in POSITION_REGISTRY["swin"] (only mlp_neurons, the structured MLP-hidden-unit
position) has no Swin counterpart at all and is skipped, listed by print_plan rather than
silently dropped.

A depth-banded placement needs more than a name match. ViT-B/16 has 12 encoder blocks, so
its bands (blocks 1-4, 5-8, 9-12) are quarters of the stack. Swin-S has 24 blocks (depths
2, 2, 18, 2), so the same fraction of its stack is blocks 1-8, 9-16, 17-24, thirds rather
than quarters. pbs/generate_swin_top3_jobs.py already swept pre_residual at exactly those 2
rescaled bands, so reproducing its rescaling here (rather than resweeping under the literal
ViT numbers) means already_complete recognizes that prior work instead of writing a second,
differently-named cache for the same measurement.

Every other basis field (operator, rate ladder, mask seed 0, forward passes 3) is read
straight from the declaration, so a rate ladder extended there for shift-reach reasons
extends here too without a second edit.

Checkpoints are read directly from checkpoints/, not from a coverage ledger: every
checkpoints/swin_* folder whose args.json names a non-benign, non-badnet_a2a, non-evasive
attack, trained with plain Adam (no _sam_rho folder tag) at seed 0 (no _seed_ folder tag),
whose metrics.json (cli.evaluate's own record) or, failing that, args.json reports an
attack success rate at or above the declared asr_bar.

cache_config_name and already_complete are imported from cli.sweep rather than
re-derived, so a placement this generator calls finished and a placement cli.sweep's
own --skip-existing would skip can never disagree. cli.sweep's own ArgumentParser is
captured the same way, so the flags every emitted command uses are checked against the
exact set cli.sweep accepts before any job is written.

Cost model: a k=3 sweep of 1 placement over its full rate ladder and all 3 splits measures
8.5 minutes per checkpoint on 1 A100, the same anchor pbs/generate_forward_pass_jobs.py
uses. Every placement here runs at the declaration's own k=3 already, so that anchor is
the per-invocation cost directly with no further scaling. Invocations are packed
sequentially into jobs of
--minutes-per-job under the fixed 12 hour walltime, and each job ends with a cli.analyze
call over the checkpoints it touched so psbd_metrics.json is refreshed alongside the cache.

    python pbs/generate_swin_basis_jobs.py --dry-run
    python pbs/generate_swin_basis_jobs.py --max-jobs 4
    bash pbs/swin_basis/submit_all.sh
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
from defences.operators import check_operator_position  # noqa: E402
from models.positions import DROPOUT_CONFIGS, POSITION_REGISTRY  # noqa: E402

PROJECT_ROOT = REPO

# ViT-B/16 has 12 encoder blocks, Swin-S has 24 (depths 2, 2, 18, 2). A ViT depth
# band is a quarter of its stack, so the fraction-matched Swin band is a third of
# 24, not a literal reuse of the ViT block numbers.
VIT_BLOCK_COUNT = 12
SWIN_BLOCK_COUNT = 24
BLOCK_SCALE = SWIN_BLOCK_COUNT // VIT_BLOCK_COUNT

# A checkpoint qualifies only at seed 0, trained with plain Adam (no SAM), on a
# non-benign, non-badnet_a2a, non-evasive attack.
EXCLUDED_ATTACKS: frozenset[str] = frozenset({"benign", "badnet_a2a"})
EXCLUDED_FOLDER_TOKENS: tuple[str, ...] = ("_sam_rho", "_seed_")

# Hard triggers first, so a batch cut short by --max-jobs still keeps the attacks
# the panel judges detection on (docs/hypothesis: BadNet and 10% never headline).
ATTACK_PRIORITY: tuple[str, ...] = (
    "adaptive_blend",
    "wanet",
    "lc",
    "sig",
    "bpp",
    "tact",
    "badnet_a2o",
    "blend",
    "lf",
)

WALLTIME = "12:00:00"
# Packing budget under the fixed 12 hour walltime, with headroom for a job killed
# at the wall to lose as little finished work as possible.
DEFAULT_MINUTES_PER_JOB = 680.0

# Measured: a k=3 sweep of 1 placement (its full rate ladder, all 3 splits) on 1
# A100. Every placement here runs at the declaration's own k=3, so this anchor is
# the per-invocation cost directly, with no k-ladder scaling to apply.
MINUTES_PER_CHECKPOINT_AT_K3: float = 8.5

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
    """1 basis entry, mapped onto Swin: its position, operator and rescaled band.

    cache_name is Swin's own results/<folder>/psbd/<cache_name> directory name.
    The ViT basis id is carried alongside only for the skip report: it names ViT
    block numbers (before_attention_norm_blocks_5_8_token_mask) that would
    misdescribe a Swin sweep whose rescaled band actually runs on blocks 9-16.
    """

    vit_id: str
    position: str
    operator: str
    block_range: tuple[int, int] | None
    rates: tuple[float, ...]
    cache_name: str


@dataclass(frozen=True)
class Invocation:
    """1 sweep call: a checkpoint, a mapped placement and its fixed k=3 cost."""

    folder: str
    placement: Placement
    minutes: float = MINUTES_PER_CHECKPOINT_AT_K3


def load_json(path: str) -> dict:
    with open(path) as handle:
        loaded = json.load(handle)
    return loaded


def declared_asr_bar(declaration: dict) -> float:
    """The ASR bar every panel table uses, read from the basis declaration."""
    asr_bar = float(declaration["asr_bar"])
    return asr_bar


def swin_supports_position(position: str) -> bool:
    """Whether every submodule a ViT basis position names also exists for Swin.

    A multi-position config (pre_residual, post_residual, both_sublayer_inputs)
    has a Swin counterpart only when every position it expands to does, so the
    check runs over DROPOUT_CONFIGS' expansion rather than the bare name.
    """
    expanded = DROPOUT_CONFIGS.get(position, (position,))
    swin_positions = POSITION_REGISTRY["swin"]
    supported = all(name in swin_positions for name in expanded)
    return supported


def rescale_block_range(
    block_range: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """A ViT block range mapped to the same fraction of Swin's larger stack."""
    if block_range is None:
        return None
    first, last = block_range
    rescaled = ((first - 1) * BLOCK_SCALE + 1, last * BLOCK_SCALE)
    return rescaled


def load_placements(basis: list[dict]) -> tuple[list[Placement], list[str]]:
    """Every basis entry mapped onto Swin, and the ids of the ones with no counterpart."""
    mapped = []
    skipped = []
    for entry in basis:
        if not swin_supports_position(entry["position"]):
            skipped.append(entry["id"])
            continue

        block_range = rescale_block_range(
            tuple(entry["block_range"]) if entry["block_range"] else None
        )
        cache_name = sweep_cli.cache_config_name(
            entry["position"],
            block_range,
            entry["operator"],
            sweep_cli.DEFAULT_FORWARD_PASSES,
        )
        mapped.append(
            Placement(
                vit_id=entry["id"],
                position=entry["position"],
                operator=entry["operator"],
                block_range=block_range,
                rates=tuple(entry["rates"]),
                cache_name=cache_name,
            )
        )
    return mapped, skipped


def validate_operator_positions(placements: list[Placement]) -> None:
    """Refuse an (operator, position) pair cli.sweep's own main() would refuse.

    check_operator_position is architecture-agnostic (it reads position and
    operator names, not a loaded model), so this reruns exactly the check
    cli.sweep performs on args.position before any GPU time is spent, over the
    positions a multi-position config expands to.
    """
    for placement in placements:
        for position in DROPOUT_CONFIGS.get(placement.position, (placement.position,)):
            check_operator_position(placement.operator, position)


def attack_sort_key(attack: str) -> int:
    rank = (
        ATTACK_PRIORITY.index(attack)
        if attack in ATTACK_PRIORITY
        else len(ATTACK_PRIORITY)
    )
    return rank


def read_asr(checkpoints_dir: str, folder: str, metadata: dict) -> float | None:
    """The attack success rate, preferring cli.evaluate's own metrics.json record.

    metrics.json is cli.evaluate's dedicated attack-success record.
    args.json's asr field is carried in for checkpoints it has not (yet)
    touched, migrated there from the PSBD baseline cache. Whichever file is on
    disk is read, so a checkpoint missing either file is not excluded for a
    bookkeeping reason.
    """
    metrics_path = os.path.join(checkpoints_dir, folder, "metrics.json")
    if os.path.exists(metrics_path):
        asr = load_json(metrics_path).get("asr")
        if asr is not None:
            return asr
    return metadata.get("asr")


def select_checkpoints(checkpoints_dir: str, asr_bar: float) -> list[dict]:
    """Every swin_* checkpoint the sweep is owed, hard attacks first.

    Restricted to a trained, non-benign, non-badnet_a2a, non-evasive attack, run
    with plain Adam at seed 0, whose measured attack success clears asr_bar. A
    checkpoint whose evasion field is present but falsy (None, the common case
    for a non-evasive run) still counts as non-evasive.
    """
    matching = []
    for folder in sorted(os.listdir(checkpoints_dir)):
        if not folder.startswith("swin_"):
            continue
        if any(token in folder for token in EXCLUDED_FOLDER_TOKENS):
            continue

        args_path = os.path.join(checkpoints_dir, folder, "args.json")
        if not os.path.isfile(args_path):
            continue
        metadata = load_json(args_path)

        if metadata.get("attack") in EXCLUDED_ATTACKS:
            continue
        if metadata.get("evasion"):
            continue

        asr = read_asr(checkpoints_dir, folder, metadata)
        if asr is None or asr < asr_bar:
            continue

        matching.append(
            {
                "folder_name": folder,
                "dataset": metadata.get("dataset"),
                "attack": metadata.get("attack"),
            }
        )

    ordered = sorted(
        matching,
        key=lambda cell: (
            attack_sort_key(cell["attack"]),
            cell["dataset"],
            cell["folder_name"],
        ),
    )
    return ordered


def plan_invocations(
    checkpoints: list[dict],
    placements: list[Placement],
    results_dir: str,
) -> tuple[list[Invocation], list[tuple[str, str]]]:
    """Every (checkpoint, placement) not already complete on disk.

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
            if sweep_cli.already_complete(
                psbd_dir, placement.cache_name, placement.rates
            ):
                already_cached.append((folder, placement.cache_name))
                continue
            planned.append(Invocation(folder=folder, placement=placement))
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
    placement = inv.placement
    rates = " ".join(str(rate) for rate in placement.rates)
    lines = [
        f'echo "=== {inv.folder} :: {placement.cache_name} ==="',
        "python -m cli.sweep \\",
        f"    --checkpoint-folder {inv.folder} \\",
        f"    --position {placement.position} \\",
        f"    --operator {placement.operator} \\",
    ]
    if placement.block_range is not None:
        first, last = placement.block_range
        lines.append(f"    --block-range {first} {last} \\")
    lines.append(f"    --rates {rates} \\")
    lines.append("    --skip-existing")
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

    Sequential rather than a bin-packing search, so the attack-priority order of
    the input list survives into the batch: a job cut by --max-jobs still keeps
    the earliest, hardest attacks.
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
    name = f"swin_basis_{index}"
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
    skipped: list[str],
    planned: list[Invocation],
    already_cached: list[tuple[str, str]],
    bundles: list[list[Invocation]],
) -> None:
    print(
        f"[ok] {len(checkpoints)} swin checkpoints (adam, seed 0, clears the ASR bar)"
    )
    print(
        f"     placements mapped   {len(placements)}: {', '.join(p.vit_id for p in placements)}"
    )
    print(
        f"     placements skipped  {len(skipped)}: {', '.join(skipped) if skipped else '(none)'}"
    )
    print(
        f"     invocations         {len(planned)} planned, "
        f"{len(already_cached)} already cached and skipped"
    )
    gpu_hours = sum(inv.minutes for inv in planned) / 60.0
    print(f"     estimated GPU       {gpu_hours:.1f} hours")
    print(f"     jobs                {len(bundles)} at up to walltime {WALLTIME}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-dir", default=os.path.join(PROJECT_ROOT, "checkpoints")
    )
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument(
        "--declaration",
        default=os.path.join(PROJECT_ROOT, "configs", "psbd_basis.json"),
    )
    parser.add_argument(
        "--asr-bar",
        type=float,
        default=None,
        help="default is asr_bar in configs/psbd_basis.json",
    )
    parser.add_argument("--batch", default="swin_basis")
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
    declaration = load_json(args.declaration)
    asr_bar = (
        args.asr_bar if args.asr_bar is not None else declared_asr_bar(declaration)
    )

    placements, skipped = load_placements(declaration["basis"])
    validate_operator_positions(placements)

    checkpoints = select_checkpoints(args.checkpoints_dir, asr_bar)
    planned, already_cached = plan_invocations(
        checkpoints, placements, args.results_dir
    )
    bundles = pack_invocations(planned, args.minutes_per_job)

    print_plan(checkpoints, placements, skipped, planned, already_cached, bundles)

    if args.dry_run:
        preview = planned[:3]
        print(f"\nfirst {len(preview)} of {len(planned)} planned commands:")
        for inv in preview:
            print()
            print(render_sweep_command(inv))
        print("\n(dry run, nothing written)")
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
