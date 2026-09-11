"""Emit PBS jobs for the latent geometry measurement, including LID.

The first geometry run predates `local_intrinsic_dimensionality`, so its CSV has
no LID columns. LID is the quantity the 2 nearest opposite results in the
literature use: Ma et al. (ICLR 2018) report adversarial inputs at LID about 4.36
against about 1.53 for normal, and COLLIDER (ACCV 2022) filters backdoor training
data on the premise that clean samples have LOW LID. Both say corrupted inputs are
locally HIGHER dimensional, the opposite sign to the participation ratio collapse
measured here.

Measuring both on the same features is what turns that from an argument into a
result, so the run has to happen again with LID in the row.

The work splits cleanly by checkpoint, so it is emitted as several jobs rather
than 1 long serial one, matching how every other sweep in this project is
scheduled.

Generated .pbs files are untracked by design; regenerate them from here.

Run:
    python pbs/generate_latent_geometry_jobs.py --jobs 6
"""

import argparse
import os

REPO = "/lustre/home/pstika/projects/PSBD-ViT"
JOB_DIRECTORY = "pbs/psbd_geometry"
LOG_DIRECTORY = "logs/psbd_geometry"

# Measured on the first run: about 15 seconds per checkpoint at 500 samples, with
# Swin's 25 layers dominating. 6 hours per job leaves generous headroom, which is
# the house rule for this project rather than a tight fit.
WALLTIME = "06:00:00"

TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N {name}
#PBS -o {repo}/{log_directory}/{shard}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"

cd {repo}
source .venv/bin/activate
export PYTHONPATH={repo}

python experiments/latent_geometry_predicts_detection/measure.py \\
    --samples {samples} \\
    --shard {shard_index} \\
    --num-shards {num_shards} \\
    --output {repo}/experiments/latent_geometry_predicts_detection/geometry_shard_{shard}.csv

echo "Finished: $(date)"
"""


def write_jobs(num_jobs: int, samples: int) -> list[str]:
    os.makedirs(JOB_DIRECTORY, exist_ok=True)
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

    written = []
    for index in range(num_jobs):
        shard = f"{index + 1:03d}"
        path = os.path.join(JOB_DIRECTORY, f"geom_{shard}.pbs")
        with open(path, "w") as handle:
            handle.write(
                TEMPLATE.format(
                    walltime=WALLTIME,
                    name=f"psbd_geom_{shard}",
                    repo=REPO,
                    log_directory=LOG_DIRECTORY,
                    shard=shard,
                    shard_index=index,
                    num_shards=num_jobs,
                    samples=samples,
                )
            )
        written.append(path)

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--samples", type=int, default=500)
    arguments = parser.parse_args()

    written = write_jobs(arguments.jobs, arguments.samples)
    print(f"wrote {len(written)} job files to {JOB_DIRECTORY}/")
    for path in written:
        print(f"  {path}")
    print("\nsubmit with:")
    print(f"  for f in {JOB_DIRECTORY}/geom_*.pbs; do qsub $f; done")
    print("\nthen merge the shards with:")
    print(
        '  python -c "import glob,pandas as pd; '
        "pd.concat([pd.read_csv(f) for f in sorted(glob.glob("
        "'results/_experiments/latent_geometry_predicts_detection/geometry_shard_*.csv'))])"
        ".to_csv('results/_experiments/latent_geometry_predicts_detection/"
        "geometry_vs_detection.csv', index=False)\""
    )


if __name__ == "__main__":
    main()
