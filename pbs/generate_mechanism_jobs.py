"""Run the residual-stream mechanism analyses on GPU, across the trigger taxonomy.

Four measurements per checkpoint, each answering a different part of "why does perturbing
here work":

    residual_decomposition   which sublayer WRITES the backdoor direction, per layer
    logit_attribution        what that writing does to the attacker's class logit
    cls_routing              whether attention ROUTES the trigger to the CLS token
    identity_attention       the causal test: switch cross-token movement off, one layer at a
                             time, and watch the attack die or not

The checkpoint list is chosen to span the three trigger families the placement results split
on, because a mechanism that only explains a patch trigger explains a third of the panel:

    local additive     badnet_a2o, tact, lc      1 to 4 of 196 tokens
    global additive    blend, bpp, lf, sig       196 of 196 tokens
    geometric          wanet                     moves content, adds none

Benign checkpoints are included as controls, not as an afterthought: every per-layer claim
needs one, and a measurement that separates on a model with no backdoor is measuring the
architecture.

    PYTHONPATH=. python pbs/generate_mechanism_jobs.py
"""

import argparse
import json
import os

PROJECT_ROOT = "/lustre/home/pstika/projects/PSBD-ViT"
SCRIPTS = (
    ("residual_decomposition", "--device cuda --controls 8"),
    ("logit_attribution", "--device cuda"),
    ("cls_routing", "--device cuda --limit 512"),
    ("identity_attention", ""),
)
# One representative cell per (attack, dataset) where the attack implanted, plus benign.
FAMILIES = {
    "local_additive": ("badnet_a2o", "tact", "lc"),
    "global_additive": ("blend", "bpp", "lf", "sig"),
    "geometric": ("wanet",),
}


def panel_cells(coverage_path: str) -> list[dict]:
    with open(coverage_path) as handle:
        cells = json.load(handle)["cells"]
    wanted = {name for group in FAMILIES.values() for name in group}
    chosen, seen = [], set()
    # Prefer the highest poison rate available per (attack, dataset): the cleanest signal for
    # a mechanism measurement, where implant strength is a nuisance rather than the variable.
    for cell in sorted(cells, key=lambda c: -(c["poison_rate"] or 0)):
        key = (cell["attack"], cell["dataset"])
        if cell["asr_class"] != "clears" or cell["attack"] not in wanted or key in seen:
            continue
        seen.add(key)
        chosen.append(cell)
    return chosen


def write_jobs(folders, args) -> list[str]:
    out_dir = os.path.join("pbs", args.batch)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join("logs", args.batch), exist_ok=True)
    written = []
    for index in range(0, len(folders), args.per_job):
        bundle = folders[index : index + args.per_job]
        name = f"mech_{index // args.per_job + 1}"
        body = "\n".join(
            f'echo "=== {script} :: {" ".join(bundle)} ==="\n'
            f"python experiments/residual_stream_mechanism/{script}.py \\\n"
            f"    --checkpoint-folder {' '.join(bundle)} {flags}\n"
            for script, flags in SCRIPTS
        )
        script_text = f"""#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime=8:00:00
#PBS -N {name}
#PBS -o {PROJECT_ROOT}/logs/{args.batch}/{name}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"; echo "Node: $(hostname)"; echo "Started: $(date)"
cd {PROJECT_ROOT}
source .venv/bin/activate
export PYTHONPATH=.

{body}
echo "Finished: $(date)"
exit 0
"""
        path = os.path.join(out_dir, f"{name}.pbs")
        with open(path, "w") as handle:
            handle.write(script_text)
        written.append(path)
    with open(os.path.join(out_dir, "submit_all.sh"), "w") as handle:
        handle.write("#!/bin/bash\n")
        for path in written:
            handle.write(f"qsub {os.path.join(PROJECT_ROOT, path)}\n")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", default="results/coverage/coverage.json")
    parser.add_argument("--batch", default="vit_mechanism")
    parser.add_argument("--per-job", type=int, default=2)
    args = parser.parse_args()

    cells = panel_cells(args.coverage)
    folders = [cell["folder_name"] for cell in cells]
    benign = [f"vit_{d}_benign" for d in ("cifar10", "cifar100", "gtsrb", "tiny")]
    folders += [
        name for name in benign if os.path.isdir(os.path.join("checkpoints", name))
    ]

    written = write_jobs(folders, args)
    reverse = {name: family for family, names in FAMILIES.items() for name in names}
    print(f"[ok] pbs/{args.batch}/")
    print(f"     checkpoints {len(folders)}  ({len(benign)} benign controls)")
    print(f"     jobs        {len(written)}   x {len(SCRIPTS)} measurements each")
    for cell in cells:
        print(
            f"       {cell['folder_name']:34s} {reverse.get(cell['attack'], '?'):16s} "
            f"asr {cell['asr']:.3f}"
        )
    print(f"     submit      bash pbs/{args.batch}/submit_all.sh")


if __name__ == "__main__":
    main()
