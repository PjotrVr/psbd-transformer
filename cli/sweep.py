"""PSBD dropout-position sweep for one (checkpoint, position-config), all 9 rates.

One invocation is the atomic unit of work: a single checkpoint, a single
position-config, sweeping the 9 dropout rates internally over one loaded model
and one baseline cache. The outer grid over checkpoints and position-configs is
flattened into separate PBS jobs (pbs/generate_psbd_jobs.py), so many single-GPU
jobs run concurrently rather than one long serial job.

Stage 1 only: this writes the raw per-pass probabilities, the per-pass argmax
classes, and the baseline to disk (defences.cache). Threshold, TPR, FPR, AUROC, and
the shift ratio are a separate cheap CPU step (cli.analyze) that reads those back,
so this script never touches the GPU for anything but forward passes.

A benign checkpoint has no attack of its own, so probing it needs --probe-attack
to name the trigger. That is the sweep's negative control and chance-level
detection is the expected result.
"""

import argparse
import json
import os

import torch

from defences.cache import (
    dropout_pass_path,
    load_or_build_baseline,
    save_dropout_pass_probs,
    write_split_manifest,
)
from data.registry import DATASET_REGISTRY
from defences.inference import compute_dropout_pass_probs
from models.backbones import detect_architecture, load_checkpoint
from defences.operators import (
    PERTURBATIONS,
    build_perturbation,
    check_operator_position,
    effective_forward_passes,
    scale_up,
)
from models.positions import (
    DROPOUT_CONFIGS,
    PORTED_POSITION_NAMES,
    SINGLE_POSITION_NAMES,
    STRUCTURED_POSITION_NAMES,
    plug_dropout,
    unplug_dropout,
)
from data.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from training.loop import current_git_commit

# The 9 dropout rates 0.1 to 0.9, the same grid the archived sweep used.
DROPOUT_RATES: tuple[float, ...] = tuple(i / 10.0 for i in range(1, 10))

# Reseeds the dropout mask sampling (see defences.inference), distinct from the
# data-split seed. Kept fixed so a rerun reproduces the same masks exactly.
PSBD_MASK_SEED = 0
# The PSBD paper's own value (sec/4_method.tex: "We perform forward inference k=3
# times"). Caches at this k keep the bare folder name so every pre-existing cache
# stays addressable.
DEFAULT_FORWARD_PASSES = 3

SPLITS = ("validation", "clean", "backdoor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", required=True, nargs="+")
    parser.add_argument(
        "--position-config",
        required=True,
        nargs="+",
        choices=tuple(DROPOUT_CONFIGS)
        + SINGLE_POSITION_NAMES
        + STRUCTURED_POSITION_NAMES
        + PORTED_POSITION_NAMES,
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "skip a (checkpoint, placement) whose every rate tensor is already on "
            "disk, so a resubmitted batch does not recompute finished work"
        ),
    )
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--forward-passes", type=int, default=DEFAULT_FORWARD_PASSES)
    # A smoke and timing knob only. The real jobs leave it None for the full split.
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--perturbation",
        default="dropout",
        choices=sorted(PERTURBATIONS) + ["scale_up"],
        help="which perturbation operator to inject. dropout is the paper's",
    )
    parser.add_argument(
        "--model-dropout",
        type=float,
        default=0.0,
        help=(
            "activate the model's OWN dropout at this rate and stack the probe on "
            "top. Removal compounds, so the probe's nominal rate stops describing "
            "the disturbance and only measured shift ratio does."
        ),
    )
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument(
        "--probe-attack",
        default=None,
        help="trigger to probe a benign checkpoint with (the negative control)",
    )
    parser.add_argument("--probe-target-label", type=int, default=None)
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=PSBD_MASK_SEED,
        help=(
            "seed for the dropout mask sequence. The estimator is stochastic, so a "
            "single seed reports one draw and says nothing about its spread. A "
            "non-default seed writes to its own cache directory."
        ),
    )
    parser.add_argument(
        "--block-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("FIRST", "LAST"),
        help=(
            "restrict a block-scope position to blocks FIRST..LAST, 1-indexed and "
            "inclusive. Where a trigger's backdoor direction reaches the CLS token "
            "is attack-dependent (blend by layer 5, a static patch not until 9), so "
            "perturbing all 12 blocks cannot separate 'this position matters' from "
            "'this depth matters'."
        ),
    )
    parser.add_argument(
        "--rates",
        nargs="*",
        type=float,
        default=None,
        help=(
            "override the 0.1-to-0.9 grid. A residual-stream position masks the "
            "whole stream once per block, so (1-p)^12 survives the ViT stack and "
            "even p=0.1 is already saturating. Reaching a comparable disturbance "
            "there needs rates an order of magnitude smaller."
        ),
    )
    return parser.parse_args()


def cache_config_name(
    position_config: str,
    block_range: tuple[int, int] | None,
    perturbation: str = "dropout",
    forward_passes: int = DEFAULT_FORWARD_PASSES,
    model_dropout: float = 0.0,
    mask_seed: int = PSBD_MASK_SEED,
) -> str:
    """The results/ subfolder name for one placement.

    A band-restricted run is a different measurement from the same position applied
    to every block, so it needs its own folder. Without the suffix the two would
    write the same rate_<tag>_<split>.pt filenames and the second run would
    silently overwrite the first.

    The same applies to the perturbation operator and to the Monte Carlo pass
    count. PSU is an expectation over k passes, so a k=20 cache is a different
    measurement from a k=3 one at the same (position, operator, rate). Without k
    in the name the two collide, and worse, --skip-existing would find the k=3
    files and skip the k=20 work while reporting success.
    """
    if block_range is None:
        stem = position_config
    else:
        stem = f"{position_config}_blocks_{block_range[0]}_{block_range[1]}"

    # dropout keeps the bare name so every cache written before perturbations
    # existed stays addressable and --skip-existing still finds it.
    if perturbation != "dropout":
        stem = f"{stem}_{perturbation}"
    if forward_passes != DEFAULT_FORWARD_PASSES:
        stem = f"{stem}_k{forward_passes}"
    if model_dropout:
        stem = f"{stem}_pmodel{model_dropout:g}".replace(".", "_")
    # The mask seed works the same way. A different seed is a different draw of
    # the same estimator, so its cache must not collide with seed 0's; seed 0
    # keeps the bare name so every cache written before the flag stays addressable.
    if mask_seed != PSBD_MASK_SEED:
        stem = f"{stem}_seed{mask_seed}"

    return stem


def resolve_architecture(checkpoint_path: str, metadata: dict) -> str:
    """The architecture, cross-checked against the checkpoint's own weights.

    args.json's architecture field was backfilled from folder names by a
    migration, not recorded at training time, so it is inference rather than
    observation. detect_architecture reads the state_dict keys instead. Building
    the wrong architecture would now raise on the strict load anyway, but failing
    here names the actual problem instead of dumping a key mismatch.
    """
    recorded = metadata["architecture"]
    detected = detect_architecture(checkpoint_path)
    if recorded != detected:
        raise ValueError(
            f"{checkpoint_path}: args.json says architecture={recorded!r} but its "
            f"state_dict looks like {detected!r}"
        )
    return detected


def load_model_and_loaders(
    args: argparse.Namespace, folder: str, device: torch.device
) -> tuple[torch.nn.Module, str, dict, dict, dict]:
    """Side-effecting setup: read the checkpoint, build the model and the splits.

    The clean test set behind these loaders is lru_cached per process, so the second
    and later checkpoints of a batch reuse it and pay only the model load.
    """
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    architecture = resolve_architecture(checkpoint_path, metadata)
    model = load_checkpoint(architecture, checkpoint_path, device)

    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        probe_attack=args.probe_attack,
        probe_target_label=args.probe_target_label,
    )
    return model, architecture, loaders, manifest, metadata


def build_operator(perturbation: str, dataset: str | None):
    """The perturbation factory, with SCALE-UP's dataset constants bound in.

    SCALE-UP works in pixel space, so it needs the dataset's normalization
    constants to undo and redo the transform around the clip. They are not
    available at registry level, so the factory is bound here.
    """
    if perturbation == "scale_up":
        spec = DATASET_REGISTRY[dataset]
        return scale_up(spec.mean, spec.std)

    operator = build_perturbation(perturbation)
    return operator


def run_one_rate(
    model: torch.nn.Module,
    loaders: dict,
    baselines: dict,
    psbd_dir: str,
    position_config: str,
    rate: float,
    device: torch.device,
    forward_passes: int,
    use_bfloat16: bool,
    model_dropout: float = 0.0,
    mask_seed: int = PSBD_MASK_SEED,
) -> None:
    """Every split at one rate, with the position already plugged."""
    for split, loader in loaders.items():
        _, baseline_labels, _ = baselines[split]
        per_pass_probs, per_pass_argmax = compute_dropout_pass_probs(
            model,
            loader,
            baseline_labels,
            device,
            forward_passes,
            use_bfloat16,
            mask_seed,
            model_dropout,
        )
        save_dropout_pass_probs(
            dropout_pass_path(psbd_dir, position_config, rate, split),
            per_pass_probs,
            per_pass_argmax,
        )


def sweep_rates(
    model: torch.nn.Module,
    architecture: str,
    position_config: str,
    loaders: dict,
    baselines: dict,
    psbd_dir: str,
    device: torch.device,
    forward_passes: int,
    use_bfloat16: bool,
    rates: tuple[float, ...] = DROPOUT_RATES,
    block_range: tuple[int, int] | None = None,
    cache_name: str | None = None,
    perturbation: str = "dropout",
    model_dropout: float = 0.0,
    mask_seed: int = PSBD_MASK_SEED,
    dataset: str | None = None,
) -> None:
    """For each rate: plug the position, run every split, save, unplug.

    unplug_dropout returns the model to exactly its loaded state, so no reset is
    needed between rates: nothing about the model was ever mutated. The finally
    is not decoration. Without it, a failure mid-rate leaves that rate's dropouts
    attached, and the next rate stacks on top of them for an effective rate of
    1 - (1-p1)(1-p2). Every file would still be written and every number would be
    quietly wrong.
    """
    position_names = DROPOUT_CONFIGS.get(position_config, (position_config,))
    cache_name = cache_name or position_config

    # A deterministic operator returns the same value on every pass, so PSU's
    # expectation over k is exact at 1 pass and a k > 1 sweep would write k
    # identical rows. The cache folder name still carries the REQUESTED k, so
    # naming and --skip-existing keep working against caches written before this.
    passes = effective_forward_passes(perturbation, forward_passes)
    operator = build_operator(perturbation, dataset)
    factory = {name: operator for name in position_names}

    for rate in rates:
        handles = plug_dropout(
            model, architecture, position_names, factory, rate, block_range=block_range
        )
        try:
            run_one_rate(
                model,
                loaders,
                baselines,
                psbd_dir,
                cache_name,
                rate,
                device,
                passes,
                use_bfloat16,
                model_dropout,
                mask_seed,
            )
        finally:
            unplug_dropout(handles)


def write_run_provenance(
    psbd_dir: str, args: argparse.Namespace, position_config: str, device: torch.device
) -> None:
    """The commit, config, and GPU behind this cache, next to the cache itself.

    bfloat16 logits have an 8-bit mantissa, so a near-tie can put the baseline
    argmax on a different class on a different GPU model. Recording which device
    produced a cache makes that reproducible rather than mysterious.
    """
    block_range = tuple(args.block_range) if args.block_range else None
    payload = {
        "git_commit": current_git_commit(),
        "position_config": position_config,
        "perturbation": args.perturbation,
        "block_range": list(args.block_range) if args.block_range else None,
        "dropout_rates": list(args.rates) if args.rates else list(DROPOUT_RATES),
        "forward_passes": args.forward_passes,
        "effective_forward_passes": effective_forward_passes(
            args.perturbation, args.forward_passes
        ),
        "mask_seed": args.mask_seed,
        "split_seed": PSBD_SPLIT_SEED,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "use_bfloat16": not args.no_bfloat16,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
    }

    cache_name = cache_config_name(
        position_config,
        block_range,
        args.perturbation,
        args.forward_passes,
        args.model_dropout,
        args.mask_seed,
    )
    path = os.path.join(psbd_dir, f"run_{cache_name}.json")
    os.makedirs(psbd_dir, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def already_complete(psbd_dir: str, cache_name: str, rates: tuple[float, ...]) -> bool:
    """Whether every rate tensor for this placement is already on disk.

    Checked per split, not just per rate, because a job killed mid-write would
    otherwise look finished and leave a hole that stage 2 discovers much later.
    """
    complete = all(
        os.path.exists(dropout_pass_path(psbd_dir, cache_name, rate, split))
        for rate in rates
        for split in SPLITS
    )
    return complete


def run_one_checkpoint(
    folder: str, args: argparse.Namespace, device: torch.device, use_bfloat16: bool
) -> None:
    """Every requested placement for one checkpoint, over one loaded model."""
    block_range = tuple(args.block_range) if args.block_range else None
    rates = tuple(args.rates) if args.rates else DROPOUT_RATES
    psbd_dir = os.path.join(args.results_dir, folder, "psbd")

    cache_names = {
        position: cache_config_name(
            position,
            block_range,
            args.perturbation,
            args.forward_passes,
            args.model_dropout,
            args.mask_seed,
        )
        for position in args.position_config
    }
    pending = [
        position
        for position, cache_name in cache_names.items()
        if not (args.skip_existing and already_complete(psbd_dir, cache_name, rates))
    ]
    if not pending:
        print(f"[skip] {folder}: every requested placement already complete")
        return

    model, architecture, loaders, manifest, metadata = load_model_and_loaders(
        args, folder, device
    )
    write_split_manifest(psbd_dir, manifest)
    baselines = {
        split: load_or_build_baseline(
            psbd_dir, split, model, loader, device, use_bfloat16
        )
        for split, loader in loaders.items()
    }

    for position in pending:
        write_run_provenance(psbd_dir, args, position, device)
        sweep_rates(
            model,
            architecture,
            position,
            loaders,
            baselines,
            psbd_dir,
            device,
            args.forward_passes,
            use_bfloat16,
            rates=rates,
            block_range=block_range,
            cache_name=cache_names[position],
            perturbation=args.perturbation,
            model_dropout=args.model_dropout,
            mask_seed=args.mask_seed,
            dataset=metadata["dataset"],
        )
        print(f"[ok] {folder} {cache_names[position]}", flush=True)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bfloat16 = not args.no_bfloat16

    # argparse validates the operator and the position separately, so an invalid
    # PAIR passes both checks and produces a complete, plausible sweep answering
    # a different question. Checked once here, before any GPU time is spent.
    for position_config in args.position_config:
        for position in DROPOUT_CONFIGS.get(position_config, (position_config,)):
            check_operator_position(args.perturbation, position)

    for folder in args.checkpoint_folder:
        try:
            run_one_checkpoint(folder, args, device, use_bfloat16)
        except Exception as error:
            # One bad checkpoint must not cost the whole batch its remaining hours.
            print(f"[FAILED] {folder}: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
