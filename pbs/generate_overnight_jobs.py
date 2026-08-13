"""Emit the full overnight PSBD grid: every viable checkpoint, both architectures.

Differs from generate_psbd_jobs.py in three ways, each needed to go wide:

1. **Gated per (attack, poison rate), not per attack.** The earlier generator kept an
   attack only if it worked at every rate, which threw away `wanet` at 10% (ASR 0.96)
   because it fails at 1%. Here each checkpoint is judged on its own measured ASR, so
   a backdoor that works is swept even if the same attack fails elsewhere.

2. **Both architectures.** Swin-S has 24 blocks against ViT-B/16's 12, so a block band
   is expressed as a third of the depth rather than a fixed index range. That is what
   makes "the middle band wins" a testable claim across architectures instead of a
   coincidence of numbering.

3. **A fixed placement shortlist.** Sweeping all 14 placements over the full
   checkpoint set would be tens of thousands of jobs. These five are the ones the
   comparison actually rests on: the published placement, the project's original
   claim, the current best, and the two strongest single positions.

Run from the repo root with PYTHONPATH=. so the flat root imports resolve.
Generation only writes files; qsub is a separate manual step.

Example
    python pbs/generate_overnight_jobs.py --dry-run
    python pbs/generate_overnight_jobs.py --architecture swin --dataset cifar10
"""

import argparse
import json
import os

BASE = "/lustre/home/pstika/projects/PSBD-ViT"

ATTACKS = (
    "badnet_a2o",
    "badnet_a2a",
    "blend",
    "bpp",
    "lf",
    "wanet",
    "adaptive_blend",
    "sig",
    "lc",
    "tact",
)
POISON_TAGS = ("0_005", "0_01", "0_05", "0_1")
SAM_RHOS = ("0_05", "0_1", "0_15", "0_2")

# A checkpoint whose attack does not fire carries no backdoor to detect, and the
# resulting jobs would succeed while measuring nothing. 0.8 is the PSBD paper's own
# failed-case line for TPR, reused here for ASR.
MIN_ASR = 0.8

# The five placements the comparison rests on. Named positions resolve on both
# architectures; the band is added separately because its indices are depth-dependent.
PLACEMENTS = (
    "post_residual",  # the published ConvNet placement
    "pre_residual",  # this project's original claim
    "before_mlp_residual",  # best single position on ViT
    "before_attention_norm",  # runner-up single position
)

# Thirds of the block stack. Expressed as fractions so the same experiment means the
# same thing on a 12-block ViT and a 24-block Swin.
BLOCK_COUNT = {"vit": 12, "swin": 24}
BAND_FRACTIONS = ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0))

# post_residual saturates above p=0.1 on a 12-block ViT and should saturate sooner
# still on a 24-block Swin, so it gets the extended low-rate grid.
FINE_RATES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09)
MAIN_RATES = tuple(i / 10 for i in range(1, 10))

BENIGN_PROBE_ATTACK = "badnet_a2o"
BENIGN_PROBE_TARGET_LABEL = 0
WALLTIME = "00:20:00"

TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N psbd_{checkpoint}_{tag}
#PBS -o {base}/logs/psbd_sweep/{checkpoint}/{tag}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
nvidia-smi

BASE={base}
cd $BASE
source .venv/bin/activate

python psbd_dropout_sweep.py \\
    --checkpoint-folder {checkpoint} \\
    --position-config {position}{extra}{band}{rates}

echo "Finished: $(date)"
exit 0
"""


def bands_for(architecture: str) -> list[tuple[int, int]]:
    """Thirds of the block stack, 1-indexed and inclusive."""
    blocks = BLOCK_COUNT[architecture]
    spans = []
    for low, high in BAND_FRACTIONS:
        first = int(round(low * blocks)) + 1
        last = int(round(high * blocks))
        spans.append((first, last))
    return spans


def read_metrics(checkpoints_dir: str, folder: str) -> dict | None:
    path = os.path.join(checkpoints_dir, folder, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def candidate_checkpoints(architecture: str, datasets: tuple[str, ...], with_sam: bool):
    for dataset in datasets:
        for attack in ATTACKS:
            for tag in POISON_TAGS:
                stem = f"{architecture}_{dataset}_{attack}_{tag}"
                yield stem
                if with_sam:
                    for rho in SAM_RHOS:
                        yield f"{stem}_sam_rho_{rho}"
        yield f"{architecture}_{dataset}_benign"
        if with_sam:
            for rho in SAM_RHOS:
                yield f"{architecture}_{dataset}_benign_sam_rho_{rho}"


def gate(folders, checkpoints_dir: str):
    """Keep checkpoints whose backdoor actually fires; report every exclusion."""
    kept, rejected = [], []
    for folder in folders:
        if not os.path.exists(
            os.path.join(checkpoints_dir, folder, "attack_result.pt")
        ):
            continue  # not trained; silent, since the cartesian product overshoots
        if "benign" in folder:
            kept.append(folder)
            continue
        metrics = read_metrics(checkpoints_dir, folder)
        asr = None if metrics is None else metrics.get("asr")
        if asr is None:
            rejected.append((folder, "no measured ASR"))
        elif asr < MIN_ASR:
            rejected.append((folder, f"ASR {asr:.3f}"))
        else:
            kept.append(folder)
    return kept, rejected


def render(checkpoint: str, position: str, tag: str, band, rates) -> str:
    extra = ""
    if "benign" in checkpoint:
        extra = (
            f" \\\n    --probe-attack {BENIGN_PROBE_ATTACK}"
            f" \\\n    --probe-target-label {BENIGN_PROBE_TARGET_LABEL}"
        )
    return TEMPLATE.format(
        walltime=WALLTIME,
        base=BASE,
        checkpoint=checkpoint,
        position=position,
        tag=tag,
        extra=extra,
        band=f" \\\n    --block-range {band[0]} {band[1]}" if band else "",
        rates=" \\\n    --rates " + " ".join(f"{r:g}" for r in rates) if rates else "",
    )


def write_job(checkpoint: str, position: str, band, rates) -> str:
    tag = position if band is None else f"{position}_blocks_{band[0]}_{band[1]}"
    pbs_dir = os.path.join(BASE, "pbs", "psbd_overnight", checkpoint)
    log_dir = os.path.join(BASE, "logs", "psbd_sweep", checkpoint)
    os.makedirs(pbs_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(pbs_dir, f"{tag}.pbs")
    with open(path, "w") as handle:
        handle.write(render(checkpoint, position, tag, band, rates))
    return path


def jobs_for(checkpoint: str, architecture: str):
    """(position, band, rates) for every job this checkpoint needs."""
    for position in PLACEMENTS:
        # A residual-stream placement needs its own low-rate window as well as the
        # standard grid; everything else only needs the standard grid.
        rates = FINE_RATES + MAIN_RATES if position == "post_residual" else ()
        yield position, None, rates
    for band in bands_for(architecture):
        yield "pre_residual", band, ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", nargs="*", default=["vit", "swin"])
    parser.add_argument("--dataset", nargs="*", default=["cifar10", "cifar100"])
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total, written = 0, 0
    all_rejected = []
    for architecture in args.architecture:
        folders = list(
            candidate_checkpoints(architecture, tuple(args.dataset), not args.no_sam)
        )
        kept, rejected = gate(folders, args.checkpoints_dir)
        all_rejected.extend(rejected)
        per_checkpoint = len(list(jobs_for("x", architecture)))
        print(
            f"{architecture}: {len(kept)} checkpoints x {per_checkpoint} placements "
            f"= {len(kept) * per_checkpoint} jobs   ({len(rejected)} excluded by ASR)"
        )
        total += len(kept) * per_checkpoint
        if args.dry_run:
            continue
        for checkpoint in kept:
            for position, band, rates in jobs_for(checkpoint, architecture):
                write_job(checkpoint, position, band, rates)
                written += 1

    print(f"\ntotal: {total} jobs")
    if all_rejected:
        print(f"excluded {len(all_rejected)} checkpoints, lowest-ASR examples:")
        for folder, reason in sorted(all_rejected, key=lambda r: r[1])[:6]:
            print(f"  {folder}: {reason}")
    if not args.dry_run:
        print(f"wrote {written} files under pbs/psbd_overnight/")


if __name__ == "__main__":
    main()
