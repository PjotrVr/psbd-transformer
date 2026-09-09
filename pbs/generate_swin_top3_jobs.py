"""Sweep the ViT top-3 configurations, and what their merges need, on the Swin panel.

The placement ranking was established on ViT. Whether it transfers is a separate question, and
it cannot be answered from cache: of the three configurations that lead on ViT, only
`before_attention_norm_token_mask` exists on Swin at all, on 29 of 53 cells, and the other two
on none. Reading a transfer claim off that coverage would repeat the unequal-coverage defect
the ViT basis was built to remove.

Swept here, on every Swin cell whose attack implanted:

    the three leading ViT configurations, so the comparison is like for like
    pre_residual restricted to blocks 9-16 and 17-24, which the merges need

Swin has 24 blocks against ViT's 12, so its depth bands are thirds of 8 rather than 4, and
`blocks_5_8` has no Swin counterpart; `blocks_9_16` and `blocks_17_24` are the middle and late
bands. Ladders follow what is already on disk so --skip-existing matches and nothing finished
is recomputed.

    PYTHONPATH=. python pbs/generate_swin_top3_jobs.py
"""

import argparse
import glob
import json
import os

from defences.decision import complete_rates

PROJECT_ROOT = "/lustre/home/pstika/projects/PSBD-ViT"
PROBABILITY = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BAND = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# (cache name, position, operator, block range, rate ladder)
TARGETS = [
    (
        "before_attention_norm_token_mask",
        "before_attention_norm",
        "token_mask",
        None,
        PROBABILITY,
    ),
    (
        "before_attention_residual_token_mask",
        "before_attention_residual",
        "token_mask",
        None,
        PROBABILITY,
    ),
    (
        "both_sublayer_inputs_token_mask",
        "both_sublayer_inputs",
        "token_mask",
        None,
        [r for r in PROBABILITY if r != 0.05],
    ),
    ("pre_residual_blocks_9_16", "pre_residual", "dropout", (9, 16), BAND),
    ("pre_residual_blocks_17_24", "pre_residual", "dropout", (17, 24), BAND),
]

JOB = """#!/bin/bash
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


def swin_cells(asr_bar: float) -> list[str]:
    cells = []
    for path in sorted(glob.glob("checkpoints/swin_*/args.json")):
        folder = os.path.basename(os.path.dirname(path))
        if any(
            t in folder for t in ("sam_rho", "evade", "a2a", "_ep", "_trig", "benign")
        ):
            continue
        with open(path) as handle:
            meta = json.load(handle)
        if meta.get("label_mode") not in ("all_to_one", "clean_label"):
            continue
        if meta.get("poison_rate") not in (0.01, 0.05, 0.1):
            continue
        if (meta.get("asr") or 0) < asr_bar:
            continue
        cells.append(folder)
    return cells


def missing(cells, mask_seed: int = 0):
    """(cell, target) pairs whose ladder is not already complete on disk, for one seed.

    A non-zero perturbation seed lives in its own cache directory, suffixed _seedN, so seed 0
    stays addressable by every cache written before the flag existed.
    """
    gaps = []
    suffix = "" if mask_seed == 0 else f"_seed{mask_seed}"
    for cell in cells:
        psbd = os.path.join("results", cell, "psbd")
        for name, position, operator, band, rates in TARGETS:
            have = set(complete_rates(psbd, name + suffix))
            need = [r for r in rates if r not in have]
            if need:
                gaps.append((cell, name, position, operator, band, tuple(need)))
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="swin_top3")
    parser.add_argument("--asr-bar", type=float, default=0.85)
    parser.add_argument("--slots-per-job", type=int, default=16)
    parser.add_argument(
        "--mask-seeds",
        nargs="+",
        type=int,
        default=[0],
        help="perturbation-seed draws. PSU is an expectation over k stochastic passes, so a "
        "single seed reports one draw of the estimator and says nothing about its spread. A "
        "non-zero seed writes to its own cache directory.",
    )
    args = parser.parse_args()

    cells = swin_cells(args.asr_bar)
    gaps = []
    for seed in args.mask_seeds:
        for cell, name, position, operator, band, rates in missing(cells, seed):
            gaps.append((cell, name, position, operator, band, rates, seed))
    # Group so one invocation covers many cells that need the same thing, since
    # run_one_checkpoint loads a model once and loops the positions over it.
    grouped = {}
    for cell, _name, position, operator, band, rates, seed in gaps:
        grouped.setdefault((position, operator, band, rates, seed), []).append(cell)

    units = []
    for (position, operator, band, rates, seed), folders in grouped.items():
        for offset in range(0, len(folders), args.slots_per_job):
            units.append(
                (
                    position,
                    operator,
                    band,
                    rates,
                    seed,
                    folders[offset : offset + args.slots_per_job],
                )
            )

    out_dir = os.path.join("pbs", args.batch)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join("logs", args.batch), exist_ok=True)
    written = []
    for index, (position, operator, band, rates, seed, folders) in enumerate(
        units, start=1
    ):
        name = f"swin_{index}"
        call = [
            "python -m cli.sweep \\",
            f"    --checkpoint-folder {' '.join(folders)} \\",
            f"    --position-config {position} \\",
            f"    --perturbation {operator} \\",
        ]
        if band:
            call.append(f"    --block-range {band[0]} {band[1]} \\")
        if seed:
            call.append(f"    --mask-seed {seed} \\")
        call.append(f"    --rates {' '.join(str(r) for r in rates)} \\")
        call.append("    --forward-passes 3 \\")
        call.append("    --skip-existing")
        hours = min(20, max(4, int(len(folders) * len(rates) * 0.5 / 60 * 2.5) + 2))
        path = os.path.join(out_dir, f"{name}.pbs")
        with open(path, "w") as handle:
            handle.write(
                JOB.format(
                    walltime=f"{hours}:00:00",
                    name=name,
                    root=PROJECT_ROOT,
                    batch=args.batch,
                    body="\n".join(call) + "\n",
                    folders=" ".join(folders),
                )
            )
        written.append(path)
    with open(os.path.join(out_dir, "submit_all.sh"), "w") as handle:
        handle.write("#!/bin/bash\n")
        for path in written:
            handle.write(f"qsub {os.path.join(PROJECT_ROOT, path)}\n")

    print(f"[ok] pbs/{args.batch}/")
    print(f"     swin cells clearing ASR {args.asr_bar}: {len(cells)}")
    print(f"     missing (cell, placement) slots        : {len(gaps)}")
    print(
        f"     rate-units                              : {sum(len(g[5]) for g in gaps)}"
    )
    print(f"     jobs                                    : {len(written)}")
    counts = {}
    for _cell, name, *_rest in gaps:
        counts[name] = counts.get(name, 0) + 1
    print(f"     perturbation seeds                      : {args.mask_seeds}")
    for name, count in sorted(counts.items()):
        print(f"       {name:44s} {count:3d} cells")
    print(f"     submit  bash pbs/{args.batch}/submit_all.sh")


if __name__ == "__main__":
    main()
