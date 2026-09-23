"""CPU jobs that rerun stage 2 over every cached checkpoint.

cli.analyze is CPU only and reads nothing but the stage-1 tensors, so whenever
detection_report gains a field, every psbd_metrics.json on disk is stale until
it is rerun. Over roughly 1200 cells that is hours in 1 process and minutes
when sharded, and the CPU queue starts jobs within a minute where the GPU queue
does not.

Every folder with a results/<folder>/psbd/ tree is assigned round-robin to 1 of
--shards jobs, and each job passes its whole list to 1 cli.analyze call. Writes
are atomic in cli.analyze, so a sweep job analyzing its own checkpoint at the
same moment cannot leave a torn file.

    PYTHONPATH=. python pbs/generate_analyze_jobs.py --dry-run
    PYTHONPATH=. python pbs/generate_analyze_jobs.py --shards 16
"""

import argparse
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = "analyze_all"
# Measured in the local fan-out of 2026-09-23: 1215 cells over 16 workers in
# about 60 minutes on a shared login node, so 3 hours is generous per shard.
WALLTIME = "03:00:00"
THREADS = 4

TEMPLATE = """#!/bin/bash
#PBS -q cpu
#PBS -l select=1:ncpus={threads}:mem=16gb
#PBS -l walltime={walltime}
#PBS -N an_{index:02d}
#PBS -o {base}/logs/{batch}/{index:02d}.log
#PBS -j oe

export OMP_NUM_THREADS={threads}
export MKL_NUM_THREADS={threads}
cd {base}
source .venv/bin/activate
echo "Node: $(hostname)  Started: $(date)  Commit: $(git rev-parse HEAD)"
echo "Folders: {count}"
python -m cli.analyze --checkpoint-folder {folders} > /dev/null \\
    || echo "[ANALYZE FAILED rc=$?]"
echo "Finished: $(date)"
exit 0
"""


def cached_folders(results_dir: str) -> list[str]:
    """Every checkpoint folder that carries a stage-1 cache."""
    folders = sorted(
        os.path.basename(os.path.dirname(path))
        for path in glob.glob(os.path.join(results_dir, "*", "psbd"))
    )
    return folders


def shard(folders: list[str], count: int) -> list[list[str]]:
    """Round-robin, so large and small cells spread evenly across jobs."""
    shards = [folders[index::count] for index in range(count)]
    return [group for group in shards if group]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.path.join(REPO, "results"))
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    groups = shard(cached_folders(args.results_dir), args.shards)
    print(f"{sum(len(group) for group in groups)} folders in {len(groups)} jobs")
    if args.dry_run:
        print("(dry run, nothing written)")
        return

    out_dir = os.path.join(REPO, "pbs", BATCH)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(REPO, "logs", BATCH), exist_ok=True)
    for index, group in enumerate(groups, start=1):
        path = os.path.join(out_dir, f"{index:02d}.pbs")
        with open(path, "w") as handle:
            handle.write(
                TEMPLATE.format(
                    threads=THREADS,
                    walltime=WALLTIME,
                    index=index,
                    base=REPO,
                    batch=BATCH,
                    count=len(group),
                    folders=" ".join(group),
                )
            )
        print(f"  wrote {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    main()
