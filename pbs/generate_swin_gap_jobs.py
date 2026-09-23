"""CPU jobs that finish the Swin comparison of the recommended placement.

The recommended placement is already cached on every Swin cell whose attack
implanted. What is missing is the 2 placements the paper compares it against, on
the cells whose sweep never reached them, so the paired gain is currently
measured on fewer models than the panel holds. This generator writes exactly
those (cell, placement) sweeps and nothing else.

Cells come from the checkpoints' own args.json sidecars: panel-shaped, Swin,
attack success at or above the declared bar. Rate ladders come from
configs/psbd_basis.json, so a new cache carries the same rates as the caches it
will be pooled with. A (cell, placement) already on disk is never emitted.

CPU because the GPU queue is oversubscribed and cli.sweep selects its device
with torch.cuda.is_available(), so a CPU node needs no flag and records the
device it used in the placement's run provenance either way.

Nothing here submits. A dry run prints the plan with its estimated CPU hours.

    PYTHONPATH=. python pbs/generate_swin_gap_jobs.py --dry-run
    PYTHONPATH=. python pbs/generate_swin_gap_jobs.py
"""

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cli.sweep import build_parser as sweep_parser  # noqa: E402
from defences.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402

# The placements the Swin argument needs: the recommendation, the placement the
# PSBD paper published, and the twin a pure AUROC selection would have chosen.
TWIN_PLACEMENT = "before_attention_residual_token_mask"
WANTED = (RECOMMENDED_PLACEMENT, PUBLISHED_PLACEMENT, TWIN_PLACEMENT)
ARCHITECTURE = "swin"

# Rows per checkpoint over the validation, clean and backdoor splits, from the
# split manifests, the same table the detector generator reads.
ROWS_PER_CHECKPOINT = {
    "cifar10": 17200,
    "cifar100": 17915,
    "gtsrb": 23208,
    "tiny": 17959,
    "svhn": 2000 + 2 * 24032,
    "eurosat": 2000 + 2 * 3400,
}
# Swin-S forward passes per second on 1 CPU node at CPU_THREADS threads, measured
# in the smoke of 2026-09-23 (docs/runs/2026-09-23-cpu-timing.md): 2400 forwards
# in 92s and 12000 in 311s on the same node, so the marginal rate is 43.8 and the
# fixed cost 37s. Rounded down, because a job that ends at its wall leaves a
# half-written ladder the next run has to redo.
FORWARDS_PER_SECOND_CPU = 40.0
CPU_THREADS = 32
BATCH_SIZE = 32
# Model load, dataset construction and the unperturbed baseline pass.
FIXED_MINUTES = 12.0
MIN_WALLTIME_HOURS = 6.0
WALLTIME_MARGIN = 2.0
MAX_WALLTIME_HOURS = 168.0

TEMPLATE = """#!/bin/bash
#PBS -q cpu
#PBS -l select=1:ncpus={threads}:mem=96gb
#PBS -l walltime={walltime}
#PBS -N sw_{index:03d}
#PBS -o {base}/logs/swin_gap/{index:03d}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"
# No GPU on this node, so cli.sweep's torch.cuda.is_available() selects CPU on
# its own. The thread caps stop torch from oversubscribing the granted cores.
export OMP_NUM_THREADS={threads}
export MKL_NUM_THREADS={threads}

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Work:    {folder} {placement}, {rates_count} rates, est {estimate} min"

BASE={base}
cd $BASE
source .venv/bin/activate
echo "Commit:  $(git rev-parse HEAD), dirty files: $(git status --porcelain | wc -l)"

{command}
echo "Analyzing:"
python -m cli.analyze --checkpoint-folder {folder} \\
    --checkpoints-dir {checkpoints_dir} --results-dir {results_dir} \\
    || echo "[ANALYZE FAILED rc=$?] {folder}"
echo "Finished: $(date)"
exit 0
"""

COMMAND = """python -m cli.sweep \\
    --checkpoint-folder {folder} \\
    --position {position} \\
    --operator {operator} \\
    --rates {rates} \\
    --batch-size {batch_size} \\
    --num-workers 8 \\
    --checkpoints-dir {checkpoints_dir} \\
    --raw-data-dir {raw_data_dir} \\
    --results-dir {results_dir} \\
    --skip-existing || echo "[SWEEP FAILED rc=$?] {folder} {placement}"
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.path.join(REPO, "results"))
    parser.add_argument("--checkpoints-dir", default=os.path.join(REPO, "checkpoints"))
    parser.add_argument("--raw-data-dir", default=os.path.join(REPO, "raw_data"))
    parser.add_argument(
        "--declaration", default=os.path.join(REPO, "configs", "psbd_basis.json")
    )
    parser.add_argument(
        "--placement",
        nargs="*",
        default=None,
        help=f"restrict to some of {WANTED}",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_json(path: str) -> dict:
    with open(path) as handle:
        loaded = json.load(handle)
    return loaded


def basis_entries(declaration: dict) -> dict[str, dict]:
    """The declared basis keyed by its cache directory name."""
    entries = {entry["id"]: entry for entry in declaration["basis"]}
    missing = [name for name in WANTED if name not in entries]
    if missing:
        raise SystemExit(
            f"{missing} are not declared in the basis, so their rate ladders are "
            "unknown and a new cache could not be pooled with the existing ones."
        )
    return entries


def clearing_cells(
    checkpoints_dir: str, declaration: dict, exclude_tokens: tuple[str, ...]
) -> list[dict]:
    """Swin checkpoints that are panel-shaped and whose attack implanted."""
    bar = declaration["asr_bar"]
    cells = []
    for folder in sorted(os.listdir(checkpoints_dir)):
        if not folder.startswith(f"{ARCHITECTURE}_"):
            continue
        if any(token in folder for token in exclude_tokens):
            continue
        sidecar_path = os.path.join(checkpoints_dir, folder, "args.json")
        if not os.path.exists(sidecar_path):
            continue
        sidecar = load_json(sidecar_path)
        asr = sidecar.get("asr")
        if asr is None or asr < bar:
            continue
        if sidecar.get("dataset") not in ROWS_PER_CHECKPOINT:
            continue
        cells.append({"folder": folder, "dataset": sidecar["dataset"], "asr": asr})
    return cells


def pending_sweeps(
    cells: list[dict], results_dir: str, placements: tuple[str, ...]
) -> list[dict]:
    """(cell, placement) pairs with no cache directory on disk."""
    pending = []
    for cell in cells:
        for placement in placements:
            if os.path.isdir(
                os.path.join(results_dir, cell["folder"], "psbd", placement)
            ):
                continue
            pending.append({**cell, "placement": placement})
    return pending


def estimated_minutes(cell: dict, entry: dict) -> float:
    """Wall-clock estimate for 1 sweep, from the measured CPU rows per second."""
    rows = ROWS_PER_CHECKPOINT[cell["dataset"]]
    passes = len(entry["rates"]) * entry.get("forward_passes", 3) + 1
    minutes = rows * passes / FORWARDS_PER_SECOND_CPU / 60.0 + FIXED_MINUTES
    return minutes


def walltime_text(estimate_minutes: float) -> str:
    hours = min(
        MAX_WALLTIME_HOURS,
        max(MIN_WALLTIME_HOURS, WALLTIME_MARGIN * estimate_minutes / 60.0),
    )
    whole_hours, minutes = divmod(int(round(hours * 60)), 60)
    text = f"{whole_hours:02d}:{minutes:02d}:00"
    return text


def render_job(index: int, work: dict, entry: dict, args: argparse.Namespace) -> str:
    """1 job: 1 checkpoint, 1 placement, then its stage-2 analysis."""
    command = COMMAND.format(
        folder=work["folder"],
        position=entry["position"],
        operator=entry["operator"],
        rates=" ".join(f"{rate:g}" for rate in entry["rates"]),
        batch_size=BATCH_SIZE,
        checkpoints_dir=args.checkpoints_dir,
        raw_data_dir=args.raw_data_dir,
        results_dir=args.results_dir,
        placement=work["placement"],
    )
    estimate = estimated_minutes(work, entry)
    script = TEMPLATE.format(
        threads=CPU_THREADS,
        walltime=walltime_text(estimate),
        index=index,
        base=REPO,
        folder=work["folder"],
        placement=work["placement"],
        rates_count=len(entry["rates"]),
        estimate=int(estimate),
        command=command,
        checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir,
    )
    return script


def emitted_flags(script: str) -> set[str]:
    """Every --flag token of the cli.sweep command in a script."""
    flags = set()
    inside = False
    for line in script.split("\n"):
        stripped = line.strip()
        if "python -m cli.sweep" in stripped:
            inside = True
        if inside and stripped.startswith("--"):
            flags.update(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", stripped))
        if inside and not stripped.endswith("\\"):
            inside = False
    return flags


def verify_flags(script: str) -> None:
    """Refuse a script whose command names a flag cli.sweep does not accept."""
    accepted = set(sweep_parser()._option_string_actions)
    unknown = sorted(emitted_flags(script) - accepted)
    if unknown:
        raise SystemExit(f"emitted flags cli.sweep rejects: {unknown}")


def main() -> None:
    args = build_arg_parser().parse_args()
    declaration = load_json(args.declaration)
    entries = basis_entries(declaration)
    exclude = tuple(
        token
        for token in declaration["panel"]["exclude_folder_tokens"]
        if token != "benign"
    ) + ("benign",)
    placements = tuple(args.placement) if args.placement else WANTED

    cells = clearing_cells(args.checkpoints_dir, declaration, exclude)
    work = pending_sweeps(cells, args.results_dir, placements)
    print(f"{len(cells)} Swin cells clear ASR {declaration['asr_bar']:g}")
    for placement in placements:
        have = sum(
            1
            for cell in cells
            if os.path.isdir(
                os.path.join(args.results_dir, cell["folder"], "psbd", placement)
            )
        )
        print(f"  {placement:38s} cached {have:3d}, missing {len(cells) - have:3d}")

    out_dir = os.path.join(REPO, "pbs", "swin_gap")
    total_hours = 0.0
    written = []
    for index, item in enumerate(
        sorted(work, key=lambda w: (w["folder"], w["placement"])), start=1
    ):
        entry = entries[item["placement"]]
        total_hours += estimated_minutes(item, entry) / 60.0
        script = render_job(index, item, entry, args)
        verify_flags(script)
        if args.dry_run:
            continue
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(REPO, "logs", "swin_gap"), exist_ok=True)
        path = os.path.join(out_dir, f"{index:03d}.pbs")
        with open(path, "w") as handle:
            handle.write(script)
        written.append(os.path.relpath(path, REPO))
    print(f"{len(work)} sweeps, est {total_hours:.1f} CPU-hours total")
    if args.dry_run:
        print("(dry run, nothing written)")
    for path in written:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
