"""Generate PBS jobs for the perturbation study, one operator grid at a time.

Separate from generate_batched_jobs.py because that generator hardcodes the
dropout placement study: a fixed placement list, a fixed rate grid, and depth
bands. This one sweeps the operator axis instead, and every operator needs its
own positions and its own rate grid.

Why per-operator rate grids. The rate means something different to each operator.
droppath at 0.5 removes half of all 24 branches, which is certainly saturated,
while gaussian at 0.5 adds noise at half the activation's own standard deviation,
which may be negligible. A shared grid would put most operators outside their
usable window and make the comparison meaningless. The grids below are chosen to
bracket clean-validation shift ratio across roughly [0.05, 0.95]; the first stage
doubles as the calibration that confirms they do.

Cross-operator comparison then happens at matched clean-validation shift ratio,
never at matched rate, using defences.psbd_metrics.select_rate_at_matched_shift.

Example
    python pbs/generate_perturbation_jobs.py --stage pilot --dry-run
    python pbs/generate_perturbation_jobs.py --stage pilot --prefix pert
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_batched_jobs import viable_checkpoints  # noqa: E402
from data.splits import BENIGN_PROBE_ATTACK  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BENIGN_PROBE_TARGET_LABEL = 0

# (positions, rates) per operator, with the reason each grid is shaped as it is.
#
#   channel_mask  masks whole channels of 768, shared across tokens. Comparable in
#                 scale to dropout, so it takes dropout's own window.
#   gaussian      rate is a relative standard deviation, not a removal fraction, so
#                 it needs to run well past 1.0 before it disturbs as much as a
#                 half-strength mask.
#   token_mask    197 tokens with CLS protected. Same window as a channel mask.
#   droppath      only 24 branch outputs exist, so each one is a large unit and the
#                 usable window sits an order of magnitude lower.
#   head_mask     144 heads, and the only position where a head is addressable.
OPERATORS: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {
    "channel_mask": (
        ("before_attention_norm", "pre_residual", "post_residual"),
        (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ),
    "gaussian": (
        ("before_attention_norm", "pre_residual", "post_residual"),
        (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    ),
    "token_mask": (
        ("after_embedding", "before_attention_norm"),
        (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ),
    "droppath": (
        ("pre_residual",),
        (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7),
    ),
    "head_mask": (
        ("attention_heads",),
        (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ),
    # Ports of two published perturbation-consistency detectors. They complete
    # the taxonomy: SCALE-UP perturbs the INPUT, the mask/noise operators perturb
    # ACTIVATIONS, and gain_scale perturbs PARAMETERS (scaling a LayerNorm's gamma
    # and beta is exactly scaling its output).
    #
    # Both rate axes are read as (factor - 1) so rate 0 is the identity, matching
    # every other operator. scale_up rates 0.5 to 10 are SCALE-UP's factors 1.5 to
    # 11; gain_scale rates 0.25 to 9 are omega 1.25 to 10.
    "scale_up": (
        ("input_pixels",),
        (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0),
    ),
    "gain_scale": (
        ("attention_norm_out", "mlp_norm_out", "final_norm_out"),
        (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0),
    ),
    # The paper's operator, present so it can act as the matched baseline and
    # carry the k ablation. Its k=3 caches already exist, so --skip-existing
    # makes re-listing it free.
    "dropout": (
        ("before_attention_norm", "pre_residual", "post_residual"),
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ),
}

# Stage 1 deliberately runs 10% poisoning only. An operator that cannot separate a
# backdoor at the highest poison rate has nothing to offer at 1%, so promoting it
# would spend GPU hours confirming a foregone failure. Only survivors go to
# stage 2's lower rates.
# Every position whose activation is (batch, tokens, channels), which is what the
# channel/token/noise operators need. attention_heads is excluded: it exposes a
# 4-D per-head tensor and only head_mask can read it. mlp_neurons is included and
# is the interesting one, since a channel there is one MLP hidden neuron, so
# channel_mask at that position is literally neuron masking.
FULL_POSITIONS: tuple[str, ...] = (
    "after_embedding",
    "before_attention_norm",
    "before_attention",
    "before_attention_residual",
    "after_attention_residual",
    "before_mlp_norm",
    "before_mlp",
    "mlp_neurons",
    "before_mlp_residual",
    "after_mlp_residual",
)

# droppath zeroes a whole activation for a sample, so it only means "skip this
# computation" at a branch output. On the residual stream it would zero the stream
# itself and destroy the forward pass rather than perturb it.
DROPPATH_POSITIONS: tuple[str, ...] = (
    "before_attention_residual",
    "before_mlp_residual",
    "pre_residual",
)

FULL_POSITION_SETS: dict[str, tuple[str, ...]] = {
    "channel_mask": FULL_POSITIONS,
    "gaussian": FULL_POSITIONS,
    "token_mask": FULL_POSITIONS,
    "droppath": DROPPATH_POSITIONS,
    "head_mask": ("attention_heads",),
    "dropout": FULL_POSITIONS,
    # Both ports are defined by WHERE they act, so their position sets are the
    # same in the full sweep as in the core one.
    "scale_up": ("input_pixels",),
    "gain_scale": ("attention_norm_out", "mlp_norm_out", "final_norm_out"),
}

PILOT_CHECKPOINTS: tuple[str, ...] = (
    "vit_cifar10_badnet_a2o_0_1",
    "vit_cifar10_blend_0_1",
    "vit_cifar10_wanet_0_1",
    "vit_cifar10_adaptive_blend_0_1",
    "vit_cifar10_badnet_a2a_0_1",
    "vit_cifar10_benign",
)

# Measured: 3.8 min per (checkpoint, position, 9 rates) on an A100 at ~2130 img/s.
MINUTES_PER_RATE = 0.42
# head_mask recomputes attention in Python instead of using the fused kernel.
HEAD_MASK_PENALTY = 2.0

TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N psbd_{prefix}_{index:03d}
#PBS -o {base}/logs/psbd_perturb/{prefix}_{index:03d}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Work:    {n_units} units, est {estimate} min"
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
    --position-config {positions} \\
    --perturbation {operator} \\
    --rates {rates}{probe} \\
    --forward-passes {passes}{stack} \\
    --skip-existing
"""


def work_units(folders: list[str], operators: list[str], passes: int) -> list[dict]:
    """One unit is (operator, position, all rates) for one checkpoint.

    Kept at checkpoint granularity so packing never splits a checkpoint across two
    jobs: the 3 no-dropout baseline tensors are computed once per checkpoint and
    shared by every rate, so splitting would recompute them.
    """
    units = []
    for folder in folders:
        for operator in operators:
            positions, rates = OPERATORS[operator]
            penalty = HEAD_MASK_PENALTY if operator == "head_mask" else 1.0
            for position in positions:
                units.append(
                    {
                        "folder": folder,
                        "operator": operator,
                        "position": position,
                        "rates": rates,
                        "minutes": len(rates)
                        * MINUTES_PER_RATE
                        * penalty
                        * (passes / 3.0),
                    }
                )
    return units


def pack(units: list[dict], target_minutes: float) -> list[list[dict]]:
    """Greedy sequential packing, keeping each checkpoint's units adjacent."""
    batches, current, spent = [], [], 0.0
    for unit in sorted(units, key=lambda u: u["folder"]):
        if current and spent + unit["minutes"] > target_minutes:
            batches.append(current)
            current, spent = [], 0.0
        current.append(unit)
        spent += unit["minutes"]
    if current:
        batches.append(current)
    return batches


def build_commands(batch: list[dict], passes: int, model_dropout: float = 0.0) -> str:
    """One command per (operator, position, benign-ness), covering many checkpoints."""
    grouped: dict[tuple, list[str]] = {}
    for unit in batch:
        key = (unit["operator"], unit["position"], "benign" in unit["folder"])
        grouped.setdefault(key, []).append(unit["folder"])

    lines = []
    for (operator, position, is_benign), folders in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
    ):
        _, rates = OPERATORS[operator]
        probe = (
            f" \\\n    --probe-attack {BENIGN_PROBE_ATTACK}"
            f" \\\n    --probe-target-label {BENIGN_PROBE_TARGET_LABEL}"
            if is_benign
            else ""
        )
        lines.append(
            COMMAND.format(
                folders=" ".join(sorted(set(folders))),
                positions=position,
                operator=operator,
                rates=" ".join(f"{r:g}" for r in rates),
                probe=probe,
                passes=passes,
                stack=f" \\\n    --model-dropout {model_dropout:g}"
                if model_dropout
                else "",
            )
        )
    return "\n".join(lines)


def write_jobs(
    batches, prefix: str, walltime: str, passes: int, model_dropout: float = 0.0
) -> list[str]:
    out_dir = os.path.join(REPO, "pbs", "psbd_perturb")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(REPO, "logs", "psbd_perturb"), exist_ok=True)
    written = []
    for index, batch in enumerate(batches, start=1):
        path = os.path.join(out_dir, f"{prefix}_{index:03d}.pbs")
        with open(path, "w") as handle:
            handle.write(
                TEMPLATE.format(
                    walltime=walltime,
                    prefix=prefix,
                    index=index,
                    base=REPO,
                    n_units=len(batch),
                    estimate=int(sum(u["minutes"] for u in batch)),
                    commands=build_commands(batch, passes, model_dropout),
                )
            )
        written.append(path)
    return written


def resolve_folders(args) -> list[str]:
    if args.checkpoint_folder:
        return list(args.checkpoint_folder)
    if args.stage == "pilot":
        return list(PILOT_CHECKPOINTS)
    return viable_checkpoints(
        args.architecture,
        args.dataset,
        with_sam=args.with_sam,
        checkpoints_dir=os.path.join(REPO, "checkpoints"),
        min_asr=args.min_asr,
        only_tags=args.only_tag or None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="pilot", choices=("pilot", "custom"))
    parser.add_argument("--checkpoint-folder", nargs="*", default=[])
    parser.add_argument("--operator", nargs="*", default=sorted(OPERATORS))
    parser.add_argument("--architecture", default="vit")
    parser.add_argument("--dataset", nargs="*", default=["cifar10"])
    parser.add_argument("--only-tag", nargs="*", default=[])
    parser.add_argument(
        "--with-sam",
        action="store_true",
        help=(
            "include SAM checkpoints. Combine with --only-tag sam_rho_0_1 to pick "
            "one rho; the tag match is endswith, so sam_rho_0_1 does not also "
            "catch sam_rho_0_15"
        ),
    )
    parser.add_argument("--min-asr", type=float, default=0.8)
    parser.add_argument("--forward-passes", type=int, default=3)
    parser.add_argument("--hours", type=float, default=4.0, help="packed work per job")
    parser.add_argument("--walltime-hours", type=float, default=6.0)
    parser.add_argument(
        "--position-set",
        default="core",
        choices=("core", "full"),
        help="core is the 3-position subset; full sweeps every valid position",
    )
    parser.add_argument(
        "--position",
        nargs="*",
        default=[],
        help="override the operator's positions (applies to every named operator)",
    )
    parser.add_argument(
        "--rates",
        nargs="*",
        type=float,
        default=[],
        help="override the operator's rate grid",
    )
    parser.add_argument(
        "--model-dropout",
        type=float,
        default=0.0,
        help="activate the model's own dropout at this rate and stack the probe on top",
    )
    parser.add_argument("--prefix", default="pert")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.position_set == "full":
        for name in args.operator:
            OPERATORS[name] = (FULL_POSITION_SETS[name], OPERATORS[name][1])
    if args.position or args.rates:
        for name in args.operator:
            positions, rates = OPERATORS[name]
            OPERATORS[name] = (
                tuple(args.position) or positions,
                tuple(args.rates) or rates,
            )
    folders = resolve_folders(args)
    if not folders:
        raise SystemExit("no checkpoints matched")

    units = work_units(folders, args.operator, args.forward_passes)
    batches = pack(units, args.hours * 60)
    total = sum(u["minutes"] for u in units)

    print(f"{len(folders)} checkpoints, {len(units)} units, est {total:.0f} min total")
    for operator in args.operator:
        subset = [u for u in units if u["operator"] == operator]
        print(
            f"  {operator:14} {len(OPERATORS[operator][0])} positions x "
            f"{len(OPERATORS[operator][1])} rates -> {len(subset)} units, "
            f"{sum(u['minutes'] for u in subset):.0f} min"
        )
    print(f"{len(batches)} jobs, walltime {args.walltime_hours:g}h each")

    if args.dry_run:
        print("(dry run, nothing written)")
        return

    hours, minutes = divmod(int(round(args.walltime_hours * 60)), 60)
    written = write_jobs(
        batches,
        args.prefix,
        f"{hours:02d}:{minutes:02d}:00",
        args.forward_passes,
        args.model_dropout,
    )
    for path in written:
        print(f"  wrote {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    main()
