"""Fan a parameter grid out into many small PBS jobs.

A grid search over several parameters can run as one enormous job or as many
small ones. Small ones schedule faster and run in parallel across the cluster,
one GPU each, which is almost always what you want. This module flattens the
grid: you choose which parameters are swept inside each job (run in sequence
within one file, a failure in one continuing to the next) and every other
parameter is flattened, one job file per combination.

Standalone and reusable, tied to no single experiment. Give it a command
template with {name} placeholders, a grid, and the list of in-job parameters.

Example spec (JSON, passed to the CLI):
    {
      "command_template": "python train.py --lr {lr} --seed {seed}",
      "grid": {"lr": [0.1, 0.01], "seed": [0, 1, 2]},
      "keep_in_job": ["seed"],
      "output_dir": "pbs/lr_sweep"
    }
This writes 2 files, one per lr, each running 3 seeds in sequence. If a seed
crashes, the job logs it and moves to the next seed.
"""

import argparse
import itertools
import json
import os
from dataclasses import dataclass


@dataclass
class PbsHeader:
    """PBS directives and environment shared by every generated job.

    Defaults match this repo's cluster convention: a single GPU, the SRCE Supek
    outbound proxy, and a venv activated from the project root.
    """

    queue: str = "gpu"
    select: str = "1:ngpus=1:ncpus=8:mem=64gb"
    walltime: str = "01:00:00"
    job_name_prefix: str = "grid"
    log_dir: str = "logs/grid"
    base_dir: str = "/lustre/home/pstika/projects/PSBD-ViT"
    venv_activate: str = "source .venv/bin/activate"
    env_exports: tuple[tuple[str, str], ...] = (
        ("http_proxy", "http://10.150.1.1:3128"),
        ("https_proxy", "http://10.150.1.1:3128"),
    )


def cartesian_product(grid: dict[str, list]) -> list[dict]:
    """Every combination of the grid as a list of {param: value} dicts.

    An empty grid yields a single empty combination, so a job with no swept
    parameters still runs its one command.
    """
    if not grid:
        return [{}]
    names = list(grid)
    value_lists = [grid[name] for name in names]
    return [dict(zip(names, chosen)) for chosen in itertools.product(*value_lists)]


def _validate_grid(grid: dict[str, list]) -> None:
    if not grid:
        raise ValueError("grid is empty, nothing to flatten")
    for name, values in grid.items():
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(
                f"grid parameter {name} must be a non-empty list, got {values!r}"
            )


def _sanitize(value) -> str:
    """A filesystem and job-name safe token from an arbitrary parameter value."""
    text = str(value)
    for bad in (".", "/", " ", ":", "="):
        text = text.replace(bad, "_")
    return text


def _split_grid(
    grid: dict[str, list], keep_in_job: tuple[str, ...]
) -> tuple[dict[str, list], dict[str, list]]:
    """Partition the grid into the flattened axes and the in-job axes."""
    unknown = [name for name in keep_in_job if name not in grid]
    if unknown:
        raise ValueError(f"keep_in_job names not in grid: {unknown}")
    flatten = {n: v for n, v in grid.items() if n not in keep_in_job}
    kept = {n: v for n, v in grid.items() if n in keep_in_job}
    return flatten, kept


def _job_stem(prefix: str, flatten_combo: dict) -> str:
    """Unique filename stem from the flattened parameter values.

    Each flattened combination is a distinct point, so the ordered tuple of its
    values is unique across jobs. The collision guard in build_pbs_grid catches
    the rare case where two different values sanitize to the same token.
    """
    if not flatten_combo:
        return prefix
    tokens = [_sanitize(flatten_combo[name]) for name in flatten_combo]
    return "_".join([prefix, *tokens])


def _render_commands(
    command_template: str, flatten_combo: dict, kept_grid: dict[str, list]
) -> list[str]:
    """Every command for one job: the flattened point crossed with all in-job points."""
    commands = []
    for kept_combo in cartesian_product(kept_grid):
        params = {**flatten_combo, **kept_combo}
        commands.append(command_template.format(**params))
    return commands


def _render_script(header: PbsHeader, stem: str, commands: list[str]) -> str:
    exports = "\n".join(
        f'export {name}="{value}"' for name, value in header.env_exports
    )
    # json.dumps quotes and escapes each command safely for a bash array element.
    command_lines = "\n".join(f"    {json.dumps(command)}" for command in commands)
    return f"""#!/bin/bash
#PBS -q {header.queue}
#PBS -l select={header.select}
#PBS -l walltime={header.walltime}
#PBS -N {stem}
#PBS -o {header.base_dir}/{header.log_dir}/
#PBS -e {header.base_dir}/{header.log_dir}/

{exports}

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
nvidia-smi

BASE={header.base_dir}
mkdir -p $BASE/{header.log_dir}
cd $BASE
{header.venv_activate}

# One command per (flattened point, in-job point). A failure is logged and the
# loop continues to the next, so one bad combination never aborts the rest.
run_command() {{
    local cmd="$1"
    echo "[start]  $(date -Is) :: $cmd"
    if eval "$cmd"; then
        echo "[ok]     $(date -Is) :: $cmd"
    else
        echo "[FAILED rc=$?] $(date -Is) :: $cmd"
    fi
}}

COMMANDS=(
{command_lines}
)

for cmd in "${{COMMANDS[@]}}"; do
    run_command "$cmd"
done

echo "Finished: $(date)"
"""


def build_pbs_grid(
    command_template: str,
    grid: dict[str, list],
    keep_in_job: tuple[str, ...] = (),
    header: PbsHeader | None = None,
) -> list[tuple[str, str]]:
    """Pure: return (filename, file_content) for every flattened job.

    No disk writes, so this is trivially testable. keep_in_job names the
    parameters swept inside each job. Every other grid parameter is flattened,
    one job per combination.
    """
    header = header or PbsHeader()
    _validate_grid(grid)
    flatten_grid, kept_grid = _split_grid(grid, tuple(keep_in_job))

    jobs = []
    seen_stems: set[str] = set()
    for flatten_combo in cartesian_product(flatten_grid):
        stem = _job_stem(header.job_name_prefix, flatten_combo)
        if stem in seen_stems:
            raise ValueError(
                f"duplicate job stem {stem}, parameter values collide after sanitizing"
            )
        seen_stems.add(stem)
        commands = _render_commands(command_template, flatten_combo, kept_grid)
        jobs.append((f"{stem}.pbs", _render_script(header, stem, commands)))
    return jobs


def write_pbs_grid(
    command_template: str,
    grid: dict[str, list],
    output_dir: str,
    keep_in_job: tuple[str, ...] = (),
    header: PbsHeader | None = None,
) -> list[str]:
    """Side effect: write every job file under output_dir, return the paths.

    Also creates the header's log_dir, since PBS needs the -o and -e directory
    to exist before a submitted job can write into it.
    """
    header = header or PbsHeader()
    jobs = build_pbs_grid(command_template, grid, keep_in_job, header)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(header.log_dir, exist_ok=True)
    paths = []
    for filename, content in jobs:
        path = os.path.join(output_dir, filename)
        with open(path, "w") as handle:
            handle.write(content)
        paths.append(path)
    return paths


def _header_from_spec(spec: dict) -> PbsHeader:
    named = (
        "queue",
        "select",
        "walltime",
        "job_name_prefix",
        "log_dir",
        "base_dir",
        "venv_activate",
    )
    fields = {name: spec[name] for name in named if name in spec}
    if "env_exports" in spec:
        fields["env_exports"] = tuple(tuple(pair) for pair in spec["env_exports"])
    return PbsHeader(**fields)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fan a parameter grid out into many small PBS jobs"
    )
    parser.add_argument(
        "spec",
        help="JSON spec with command_template, grid, output_dir, optional "
        "keep_in_job, and optional PBS header fields",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the jobs that would be written without writing them",
    )
    args = parser.parse_args()

    with open(args.spec) as handle:
        spec = json.load(handle)
    header = _header_from_spec(spec)
    keep_in_job = tuple(spec.get("keep_in_job", ()))

    if args.dry_run:
        jobs = build_pbs_grid(
            spec["command_template"], spec["grid"], keep_in_job, header
        )
        for filename, _ in jobs:
            print(os.path.join(spec["output_dir"], filename))
        print(f"{len(jobs)} jobs (dry run, nothing written)")
        return

    paths = write_pbs_grid(
        spec["command_template"],
        spec["grid"],
        spec["output_dir"],
        keep_in_job,
        header,
    )
    print(f"wrote {len(paths)} jobs to {spec['output_dir']}")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
