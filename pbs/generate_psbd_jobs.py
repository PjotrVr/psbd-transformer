"""Emit PSBD-sweep PBS jobs, one per (checkpoint, position-config).

The checkpoint list is derived, never hand-typed. Every checkpoint already
carries a metrics.json with its measured attack success rate, so this reads that
and drops anything whose backdoor does not actually work. That gate exists
because the previous generated grid targeted vit_cifar100_wanet at 3 poison
rates, and WaNet on CIFAR-100 reaches ASR 0.044 at 1% and 0.649 at 5%: there was
no backdoor there to detect, so those jobs could only ever have produced noise.

Run from the repo root with PYTHONPATH=. so the flat root imports resolve.
Generation only writes files; qsub is a separate manual step.

Example
    python pbs/generate_psbd_jobs.py --phase pre_post
    python pbs/generate_psbd_jobs.py --phase single_positions --dry-run
"""

import argparse
import json
import os

from defences.dropout import DROPOUT_CONFIGS, SINGLE_POSITION_NAMES

BASE = "/lustre/home/pstika/projects/PSBD-ViT"

# CIFAR-10 ViT is the only grid where every candidate attack holds high ASR at
# every poison rate and every SAM rho, so it is the primary. The 5 attacks span
# the trigger taxonomy the backdoor-directions paper separates on: static patch
# (badnet_a2o), global blended (blend), stealthy distributed (bpp, lf), and one
# all-to-all probe (badnet_a2a) whose label geometry PSBD's stated mechanism
# should not survive.
ATTACKS = ("badnet_a2o", "blend", "bpp", "lf", "badnet_a2a")
POISON_RATES = ("0_01", "0_05", "0_1")
SAM_RHOS = ("0_05", "0_1", "0_15", "0_2")

# Below this the attack did not take, and PSU on a model with no backdoor
# measures nothing. 0.8 is the PSBD paper's own "case failed" line for TPR, reused
# here as the viability line for ASR.
MIN_ASR = 0.8

# The benign negative control has no attack of its own, so it is probed with a
# named trigger. If a benign model scores well above chance, the probe is
# responding to the perturbation rather than to a backdoor and the whole sweep is
# measuring an artifact.
BENIGN_PROBE_ATTACK = "badnet_a2o"
BENIGN_PROBE_TARGET_LABEL = 0

# One timed job measured 4:00 on an A100 for a full 10000-image CIFAR split
# (9 rates x 3 splits x 3 passes, first job also builds the baseline). 15 minutes
# is roughly 3.75x that, covering a slower GPU plus I/O variance while staying
# small enough for fast backfill.
WALLTIME = "00:15:00"

TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N psbd_{checkpoint}_{position}{suffix}
#PBS -o {base}/logs/psbd_sweep/{checkpoint}/{position}{suffix}.log
#PBS -j oe

# Compute nodes reach the internet through the proxy, needed on first run for
# torchvision to fetch the dataset and the ImageNet weights.
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


def read_asr(checkpoints_dir: str, folder: str) -> float | None:
    """The already-measured attack success rate, or None for benign and missing."""
    path = os.path.join(checkpoints_dir, folder, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle).get("asr")


def candidate_folders(with_sam: bool) -> list[str]:
    """Every folder name the phase would like to run, before the ASR gate."""
    folders = []
    for attack in ATTACKS:
        for rate in POISON_RATES:
            stem = f"vit_cifar10_{attack}_{rate}"
            folders.append(stem)
            if with_sam:
                folders.extend(f"{stem}_sam_rho_{rho}" for rho in SAM_RHOS)
    folders.append("vit_cifar10_benign")
    if with_sam:
        folders.extend(f"vit_cifar10_benign_sam_rho_{rho}" for rho in SAM_RHOS)
    return folders


def gate_by_asr(
    folders: list[str], checkpoints_dir: str
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split into (kept, [(folder, reason)]) so every exclusion is reported.

    Silent dropping is what produced the stale WaNet grid. A rejected checkpoint
    is printed with its measured ASR, so the gate is auditable from the generator
    output alone.
    """
    kept, rejected = [], []
    for folder in folders:
        if not os.path.exists(
            os.path.join(checkpoints_dir, folder, "attack_result.pt")
        ):
            rejected.append((folder, "no checkpoint on disk"))
            continue
        if "benign" in folder:
            kept.append(folder)
            continue
        asr = read_asr(checkpoints_dir, folder)
        if asr is None:
            rejected.append((folder, "no metrics.json, ASR unknown"))
        elif asr < MIN_ASR:
            rejected.append((folder, f"ASR {asr:.3f} below {MIN_ASR}"))
        else:
            kept.append(folder)
    return kept, rejected


def render(
    checkpoint: str,
    position: str,
    rates: tuple[float, ...] = (),
    suffix: str = "",
    block_range: tuple[int, int] | None = None,
) -> str:
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
        suffix=suffix,
        extra=extra,
        rates=(" \\\n    --rates " + " ".join(f"{r:g}" for r in rates))
        if rates
        else "",
        band=(
            f" \\\n    --block-range {block_range[0]} {block_range[1]}"
            if block_range
            else ""
        ),
    )


def write_job(
    checkpoint: str,
    position: str,
    rates: tuple[float, ...] = (),
    suffix: str = "",
    block_range: tuple[int, int] | None = None,
) -> str:
    """Write one .pbs file and pre-create its log directory, return the path."""
    pbs_dir = os.path.join(BASE, "pbs", "psbd_sweep", checkpoint)
    log_dir = os.path.join(BASE, "logs", "psbd_sweep", checkpoint)
    os.makedirs(pbs_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(pbs_dir, f"{position}{suffix}.pbs")
    with open(path, "w") as handle:
        handle.write(render(checkpoint, position, rates, suffix, block_range))
    return path


PHASES: dict[str, tuple[tuple[str, ...], bool]] = {
    # The two placements the study exists to compare, run and analyzed first.
    "pre_post": (tuple(DROPOUT_CONFIGS), False),
    # Every atomic position, to localize where inside a block the signal lives.
    "single_positions": (SINGLE_POSITION_NAMES, False),
    # SAM, on whichever placements phases 1 and 2 showed to be worth the compute.
    "sam": (tuple(DROPOUT_CONFIGS), True),
    # H10: aim the placement at the depth where an attack's direction is written.
    "depth_bands": (("pre_residual",), False),
}

# ViT-B/16 has 12 blocks. Four contiguous bands, plus the all-blocks reference that
# the other phases already produce. The boundaries bracket the measured direction
# onsets (blend 5, bpp 6, lf 8, badnet_a2o 9), so each attack's onset falls inside a
# different band and the per-attack prediction is separable.
DEPTH_BANDS: tuple[tuple[int, int], ...] = ((1, 4), (5, 8), (9, 12))

# A residual-stream placement masks the whole stream once per block, so its usable
# window sits an order of magnitude below the main grid: measured on
# vit_cifar10_badnet_a2o_0_1, the backdoor direction retains 90% of its separation
# at p=0.01 and 4% at p=0.08, while the main grid starts at 0.1 where it is already
# gone. Comparing placements only on the main grid would therefore compare
# pre_residual inside its window against post_residual entirely outside its own.
FINE_RATES: tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=tuple(PHASES))
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument(
        "--position",
        nargs="*",
        default=None,
        help="override the phase's position list, for a targeted follow-up",
    )
    parser.add_argument(
        "--fine-rates",
        action="store_true",
        help="sweep the sub-0.1 window instead of the 0.1-to-0.9 grid",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positions, with_sam = PHASES[args.phase]
    if args.position:
        positions = tuple(args.position)

    kept, rejected = gate_by_asr(candidate_folders(with_sam), args.checkpoints_dir)

    print(f"phase {args.phase}: {len(positions)} positions x {len(kept)} checkpoints")
    for folder, reason in rejected:
        print(f"  EXCLUDED {folder}: {reason}")
    bands = len(DEPTH_BANDS) if args.phase == "depth_bands" else 1
    if args.dry_run:
        print(f"\n{len(kept) * len(positions) * bands} jobs (dry run, nothing written)")
        return

    rates = FINE_RATES if args.fine_rates else ()
    # A distinct filename, because the fine sweep writes into the same
    # position-config folder as the main grid. That is intentional: the rate tags
    # do not collide, so stage 2 sees one continuous rate axis per placement.
    suffix = "_fine" if args.fine_rates else ""

    if args.phase == "depth_bands":
        # Each band gets its own job and its own cache folder. The bands are swept
        # over the full rate grid rather than a shared rate, because an early band
        # perturbs harder than a late one at the same p (it propagates through more
        # blocks), so they can only be compared after matching on achieved shift
        # ratio, and that needs each band's whole rate curve.
        paths = [
            write_job(
                checkpoint, position, rates, f"_blocks_{first}_{last}", (first, last)
            )
            for checkpoint in kept
            for position in positions
            for first, last in DEPTH_BANDS
        ]
    else:
        paths = [
            write_job(checkpoint, position, rates, suffix)
            for checkpoint in kept
            for position in positions
        ]
    print(f"\nwrote {len(paths)} jobs under pbs/psbd_sweep (walltime {WALLTIME})")


if __name__ == "__main__":
    main()
