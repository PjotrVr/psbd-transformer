"""Re-sweep every viable gaussian cell after the per-sample std fix.

GaussianNoise scaled its noise by x.std() reduced over batch, tokens and channels
together, so a sample's perturbation magnitude depended on its batch neighbours.
scripts/gaussian_batch_coupling/measure.py showed the backdoor split ran 13 to 32
percent hotter than the clean split it is compared against at before_attention_norm,
which is the position whose published AUROC 0.168 inversion the coupling would
produce on its own. The operator now draws per sample, so every gaussian cell has
to be measured again.

The old caches are not overwritten. archive_old_caches() renames each
<position>_gaussian to <position>_gaussian_batchstd first, so the buggy
measurement stays on disk for the before and after comparison while the canonical
name carries the corrected data that every downstream reader picks up.

Rate grids are read off disk per cell rather than assumed, so a re-sweep covers
exactly what was measured before. Cells whose old grid was partial are filled to
the standard grid, since a partial grid was a killed job rather than a choice.

Example
    python pbs/generate_gaussian_rerun_jobs.py --dry-run
    python pbs/generate_gaussian_rerun_jobs.py --archive --hours 4
"""

import argparse
import glob
import json
import os
import shutil
from collections import defaultdict

BASE = "/lustre/home/pstika/projects/PSBD-ViT"

STANDARD_RATES = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
# The E1 calibration extension, added where sigma jumped from 0.072 to 0.749
# between rates 0.2 and 0.3 and the interesting region had no resolution.
EXTENDED_RATES = (
    0.05,
    0.1,
    0.2,
    0.22,
    0.25,
    0.27,
    0.3,
    0.4,
    0.5,
    0.6,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
)

MINUTES_PER_RATE = 0.42

# Positions whose gaussian numbers are actually quoted: before_mlp is the rank 2
# configuration overall, before_attention_norm is the AUROC 0.168 inversion that
# the batch coupling is the leading explanation for. These are swept first so the
# paper-critical half lands in the first few hours rather than after the tail.
PRIORITY_POSITIONS = ("before_attention_norm", "before_mlp")
# Where the superseded batch-coupled caches were moved. They cannot stay under
# results/<folder>/psbd/, because every analysis script discovers position
# configs by listing that directory and a "<position>_gaussian_batchstd" folder
# parses as operator "dropout" at a position that does not exist.
ARCHIVE_ROOT = os.path.join(BASE, "archive", "gaussian_batchstd")

BENIGN_PROBE_ATTACK = "badnet_a2o"
BENIGN_PROBE_TARGET_LABEL = 0

TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N psbd_{prefix}_{index:03d}
#PBS -o {base}/logs/psbd_gauss/{prefix}_{index:03d}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Work:    {n_units} cells, est {estimate} min"
nvidia-smi --query-gpu=name --format=csv,noheader

BASE={base}
cd $BASE
source .venv/bin/activate

{commands}
echo "Finished: $(date)"
exit 0
"""

COMMAND = """python -m cli.sweep \\
    --checkpoint-folder {folders} \\
    --position-config {position} \\
    --perturbation gaussian \\
    --rates {rates}{probe}
"""


def read_metadata(folder):
    path = os.path.join(BASE, "checkpoints", folder, "args.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def is_viable(metadata, min_asr, include_sam):
    """Whether this checkpoint is worth spending a GPU hour re-measuring.

    Two exclusions, both about not buying certainty on a question whose answer
    cannot matter.

    SAM is excluded by default. Its whole effect on detection is +0.009 mean
    AUROC, which needs 10 to 89 seeds per cell to establish and is already
    reported as negligible. It is also 69 percent of the checkpoint set, so
    carrying it costs most of the compute for a claim that is being dropped.

    An attack that never implanted is excluded because PSU on a model with no
    backdoor measures nothing.
    """
    if metadata["optimizer"] == "sam" and not include_sam:
        return False
    if metadata["attack"] == "benign":
        return True
    asr = metadata.get("asr")
    return asr is not None and asr >= min_asr


def rates_for(cell_dir):
    """The grid this cell already carries, promoted to a standard grid if partial."""
    tags = {
        name[len("rate_") :].rsplit("_", 1)[0]
        for name in os.listdir(cell_dir)
        if name.startswith("rate_")
    }
    existing = sorted(float(tag.replace("_", ".")) for tag in tags)
    if len(existing) >= len(EXTENDED_RATES):
        return EXTENDED_RATES
    if set(existing) <= set(STANDARD_RATES):
        return STANDARD_RATES
    # A grid mixing both, so keep the union rather than dropping measured rates.
    return tuple(sorted(set(existing) | set(STANDARD_RATES)))


def discover_cells(min_asr, include_sam):
    """Every (folder, position, rates) whose attack actually implanted."""
    cells = []
    for psbd_dir in sorted(glob.glob(os.path.join(BASE, "results", "*", "psbd"))):
        folder = os.path.basename(os.path.dirname(psbd_dir))
        metadata = read_metadata(folder)
        if metadata is None or not is_viable(metadata, min_asr, include_sam):
            continue
        # Discovery has to survive its own archive step, so a cell that was already
        # moved aside is still found and still generates the job that refills it.
        # The archive keeps the original folder name, so both roots read the same.
        search_dirs = [psbd_dir, os.path.join(ARCHIVE_ROOT, folder)]
        seen_positions = set()
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            for name in sorted(os.listdir(search_dir)):
                cell_dir = os.path.join(search_dir, name)
                if not name.endswith("_gaussian") or not os.path.isdir(cell_dir):
                    continue
                position = name[: -len("_gaussian")]
                if position in seen_positions:
                    continue
                seen_positions.add(position)
                cells.append(
                    {
                        "folder": folder,
                        "position": position,
                        "rates": rates_for(cell_dir),
                        "attack": metadata["attack"],
                        "minutes": len(rates_for(cell_dir)) * MINUTES_PER_RATE,
                    }
                )
    return cells


def archive_old_caches(cells, dry_run):
    """Move each superseded cell out of the live tree, keeping its original name.

    Moving rather than deleting keeps the before and after comparison possible,
    which is the evidence that the fix mattered. Moving OUT of results/ rather
    than renaming in place is what stops the buggy measurement from re-entering
    an analysis under a phantom position name.
    """
    moved = 0
    for cell in cells:
        live = os.path.join(
            BASE, "results", cell["folder"], "psbd", f"{cell['position']}_gaussian"
        )
        archived_dir = os.path.join(ARCHIVE_ROOT, cell["folder"])
        archived = os.path.join(archived_dir, f"{cell['position']}_gaussian")
        if not os.path.isdir(live) or os.path.isdir(archived):
            continue
        if not dry_run:
            os.makedirs(archived_dir, exist_ok=True)
            shutil.move(live, archived)
        moved += 1
    return moved


def pack(cells, target_minutes):
    """Group cells sharing a position and grid, then fill jobs to the time budget.

    Grouped by (position, rates) because one sweep invocation takes a single
    position and a single rate list, so mixing them would need a command each.
    """
    by_shape = defaultdict(list)
    for cell in cells:
        by_shape[(cell["position"], cell["rates"])].append(cell)

    def priority(item):
        (position, _rates), _group = item
        rank = (
            PRIORITY_POSITIONS.index(position)
            if position in PRIORITY_POSITIONS
            else len(PRIORITY_POSITIONS)
        )
        return (rank, position)

    jobs = []
    current, minutes = [], 0.0
    for (position, rates), group in sorted(by_shape.items(), key=priority):
        for cell in group:
            if minutes + cell["minutes"] > target_minutes and current:
                jobs.append(current)
                current, minutes = [], 0.0
            current.append(cell)
            minutes += cell["minutes"]
    if current:
        jobs.append(current)
    return jobs


def build_commands(job):
    """One command per (position, rates) group inside this job."""
    by_shape = defaultdict(list)
    for cell in job:
        by_shape[(cell["position"], cell["rates"])].append(cell)

    lines = []
    for (position, rates), group in sorted(by_shape.items()):
        # A benign model has no attack of its own, so its backdoor split needs an
        # externally named trigger. It is the negative control, so it must run.
        benign = [c for c in group if c["attack"] == "benign"]
        backdoored = [c for c in group if c["attack"] != "benign"]
        rate_text = " ".join(f"{rate:g}" for rate in rates)
        for subset, probe in (
            (backdoored, ""),
            (
                benign,
                f" \\\n    --probe-attack {BENIGN_PROBE_ATTACK}"
                f" --probe-target-label {BENIGN_PROBE_TARGET_LABEL}",
            ),
        ):
            if not subset:
                continue
            lines.append(
                COMMAND.format(
                    folders=" ".join(c["folder"] for c in subset),
                    position=position,
                    rates=rate_text,
                    probe=probe,
                )
            )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-asr", type=float, default=0.5)
    parser.add_argument(
        "--include-sam",
        action="store_true",
        help="sweep SAM checkpoints too. Off by default: SAM is 69 percent of the "
        "checkpoint set and its detection effect is +0.009 AUROC, so including it "
        "spends most of the compute on a claim that cannot be established",
    )
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--prefix", default="gauss")
    parser.add_argument("--out-dir", default=os.path.join(BASE, "pbs", "psbd_gauss"))
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cells = discover_cells(args.min_asr, args.include_sam)
    total_minutes = sum(cell["minutes"] for cell in cells)
    jobs = pack(cells, args.hours * 60 * 0.85)

    print(f"viable gaussian cells: {len(cells)}")
    print(f"estimated GPU time:    {total_minutes / 60:.1f} hours")
    print(f"jobs at {args.hours}h:          {len(jobs)}")

    if args.archive:
        moved = archive_old_caches(cells, args.dry_run)
        print(f"archived old caches:   {moved} (moved under {ARCHIVE_ROOT})")

    if args.dry_run:
        print("\ndry run, no files written")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE, "logs", "psbd_gauss"), exist_ok=True)
    written = []
    for index, job in enumerate(jobs, start=1):
        minutes = sum(cell["minutes"] for cell in job)
        walltime = f"{int(args.hours):02d}:00:00"
        path = os.path.join(args.out_dir, f"{args.prefix}_{index:03d}.pbs")
        with open(path, "w") as handle:
            handle.write(
                TEMPLATE.format(
                    base=BASE,
                    prefix=args.prefix,
                    index=index,
                    walltime=walltime,
                    n_units=len(job),
                    estimate=int(minutes),
                    commands=build_commands(job),
                )
            )
        written.append(path)
    print(f"\nwrote {len(written)} job files to {args.out_dir}")


if __name__ == "__main__":
    main()
