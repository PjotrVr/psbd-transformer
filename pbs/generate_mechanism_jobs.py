"""Run the residual-stream mechanism analyses on GPU, across the trigger taxonomy.

Four measurements per checkpoint, each answering a different part of "why does perturbing
here work":

    residual_decomposition   which sublayer WRITES the backdoor direction, per layer
    logit_attribution        what that writing does to the attacker's class logit
    cls_routing              whether attention ROUTES the trigger to the CLS token
    artifact_tokens          whether the trigger MANUFACTURES the high-norm token that
                             attention then reads, and whether that is a detector
    register_neurons         whether it recruits the neurons that already drive high-norm
                             tokens, or installs a disjoint set
    sink_anatomy             what the trigger token's value vector carries, and whether a
                             substitute takes over the sink role when it is masked
    sink_hit_confound        whether a masking detector is reading the backdoor or reading
                             which tokens the draw happened to cover, zeroed and mean-filled
    resolution_sweep         whether the attention shares are a statement about the backdoor
                             or about sequence length, by serving the same weights at other
                             resolutions with interpolated positional embeddings
    test_time_registers      whether giving the model spare tokens to sink into weakens the
                             attack, which is both a mechanism test and a candidate defence
    activation_patching      the causal version: overwrite one site with its value on the
                             same image without the trigger, and see how much clean answer
                             returns. Both directions, since sufficiency to restore and
                             sufficiency to induce can disagree
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
    ("artifact_tokens", "--limit 512"),
    ("register_neurons", "--limit 384"),
    ("sink_anatomy", "--limit 384"),
    ("sink_hit_confound", "--limit 256 --draws 12"),
    ("sink_hit_confound_mean", "--limit 256 --draws 12 --substitute mean"),
    ("activation_patching", "--limit 128 --noising"),
    ("test_time_registers", "--limit 1024 --registers 0 1 4 16 64"),
    ("resolution_sweep", "--limit 512 --resolutions 160 192 224 256 320 448"),
)
# One representative cell per (attack, dataset) where the attack implanted, plus benign.
FAMILIES = {
    "local_additive": ("badnet_a2o", "tact", "lc"),
    "global_additive": ("blend", "bpp", "lf", "sig"),
    "geometric": ("wanet",),
}


def swin_cells(checkpoints_dir: str, asr_bar: float) -> list[dict]:
    """Swin checkpoints whose attack implanted, read straight from their provenance.

    The coverage ledger is ViT-only, so Swin cannot be selected through it. Held to the same
    ASR bar, because a detection number on a backdoor that never implanted measures nothing
    on either architecture.
    """
    import glob

    wanted = {name for group in FAMILIES.values() for name in group}
    chosen, seen = [], set()
    for path in sorted(glob.glob(os.path.join(checkpoints_dir, "swin_*", "args.json"))):
        folder = os.path.basename(os.path.dirname(path))
        if any(
            token in folder for token in ("sam_rho", "evade", "a2a", "_ep", "_trig")
        ):
            continue
        with open(path) as handle:
            meta = json.load(handle)
        if meta.get("attack") not in wanted or (meta.get("asr") or 0) < asr_bar:
            continue
        key = (meta["attack"], meta["dataset"])
        if key in seen:
            continue
        seen.add(key)
        chosen.append({**meta, "folder_name": folder})
    return chosen


def panel_cells(coverage_path: str, all_cells: bool = False) -> list[dict]:
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
        if not all_cells:
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
            f"python experiments/residual_stream_mechanism/{script.replace('_mean', '')}.py \\\n"
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
    parser.add_argument(
        "--architecture",
        default="vit",
        choices=("vit", "swin"),
        help="swin is selected from checkpoint provenance, since the coverage ledger is "
        "ViT-only",
    )
    parser.add_argument(
        "--only", nargs="*", default=None, help="run a subset of the measurements"
    )
    parser.add_argument(
        "--all-cells",
        action="store_true",
        help="every clearing cell, not one per attack and dataset",
    )
    args = parser.parse_args()

    global SCRIPTS
    if args.only:
        SCRIPTS = tuple(item for item in SCRIPTS if item[0] in set(args.only))
    if args.architecture == "swin":
        cells = swin_cells("checkpoints", 0.85)
    else:
        cells = panel_cells(args.coverage, args.all_cells)
    folders = [cell["folder_name"] for cell in cells]
    benign = [
        f"{args.architecture}_{d}_benign"
        for d in ("cifar10", "cifar100", "gtsrb", "tiny")
    ]
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
