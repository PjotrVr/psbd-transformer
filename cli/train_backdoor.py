"""Poison a dataset and train a backdoored ViT-B/16 or Swin-S.

The BackdoorBench-style flow: load clean data, poison a fraction of the training
set, train, evaluate ASR and clean accuracy, then save in the attack_result.pt
format the sweep reads.

Attacks that use cover samples (adaptive_blend, tact) are detected from their
config and routed through the cover-sample dataset. Everything else uses the plain
poisoning path. The training loop itself is the same in both cases.

Example
    python -m cli.train_backdoor --dataset cifar10 --attack badnet_a2a \
        --poison-rate 0.1 --architecture vit --epochs 15 \
        --output checkpoints/vit_cifar10_badnet_a2a_0_1/attack_result.pt
"""

import argparse
import os
import time
from dataclasses import replace

import torch
import torchvision.transforms.v2 as transforms_v2
from lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, Subset

from attacks import (
    ATTACK_NAMES,
    apply_config_overrides,
    build_attack,
    config_overrides,
    default_config,
    adversarial_config_error,
    missing_adversarial_bases,
)
from attacks.generated import GeneratedConfig
from data.registry import DATASET_REGISTRY
from data.loading import (
    base_image_transform,
    extract_labels,
    limit_dataset,
    load_clean_datasets,
)
from evaluation.loaders import build_clean_loader
from evaluation.metrics import clean_accuracy, evaluate_attack
from attacks.evasion import FlaggedPoisonedSet, calibrate_probe_rate
from attacks.poisoning import (
    Attack,
    CoverPoisonedTrainingSet,
    PoisonedTrainingSet,
    choose_indices_with_cover,
    choose_poison_indices,
)
from training.loop import (
    build_model,
    checkpoint_metadata,
    save_checkpoint,
    train_classifier,
    utc_timestamp,
)

# Interpolation is measured on a fixed subsample. The trajectory, not the exact
# value, is what the snapshot sweep reads.
TRAIN_EVAL_SAMPLES = 10000


def parse_attack_overrides(overrides: list[str] | None) -> dict:
    """Turn `key=value` command-line strings into a mapping.

    Casting is left to attacks.apply_config_overrides so training and evaluation agree
    on the type of every field.
    """
    if not overrides:
        return {}
    parsed = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--attack-override needs key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parsed[key] = raw
    return parsed


def resolve_config(attack_name: str, poisoned_dir: str):
    """The attack's config, with the generated adapter pointed at its trigger folder."""
    if attack_name == "generated":
        return GeneratedConfig(poisoned_dir=poisoned_dir)

    config = default_config(attack_name)
    return config


# Cover-sample rates the papers specify, as a multiple of the poisoning rate.
# WaNet's noise mode and Adaptive-Blend's cover both scale with the poison rate, so a
# fixed constant is 10x too small at 10% poisoning and the attack loses the stealth
# the mechanism exists to provide. TaCT's reference selects cover by CLASS rather than
# by rate, so its config constant stands and it is deliberately absent here.
COVER_RATE_MULTIPLES = {
    "wanet": 2.0,  # BackdoorBench cross_ratio 2, and PSBD's twice the poisoning ratio
    "adaptive_blend": 1.0,  # Qi et al. and PSBD: cover ratio equal to the poisoning ratio
    "bpp": 1.0,  # BackdoorBench neg_ratio 0.1 against pratio 0.1
}


def resolve_cover_rate(attack_name: str, poison_rate: float, override):
    """The cover rate to use: an explicit override, else the attack's paper value.

    Returns None when the attack defines no cover mechanism, leaving the config's own
    value in place.
    """
    if override is not None:
        return override
    multiple = COVER_RATE_MULTIPLES.get(attack_name)
    return None if multiple is None else multiple * poison_rate


def build_training_set(
    train_clean: Dataset,
    attack: Attack,
    config,
    poison_rate: float,
    seed: int,
    normalize,
    num_classes: int,
) -> tuple[Dataset, float]:
    """The poisoned training set and the poison rate it actually realized.

    Routes to the cover-sample dataset when the attack config asks for it.
    """
    labels = extract_labels(train_clean)
    cover_rate = getattr(config, "cover_rate", 0.0)
    source_classes = getattr(config, "source_classes", None)

    if cover_rate > 0.0 or source_classes is not None:
        poison_indices, cover_indices = choose_indices_with_cover(
            labels, attack, poison_rate, cover_rate, source_classes, seed
        )
        dataset = CoverPoisonedTrainingSet(
            train_clean, attack, poison_indices, cover_indices, normalize, num_classes
        )
    else:
        poison_indices = choose_poison_indices(labels, attack, poison_rate, seed)
        dataset = PoisonedTrainingSet(
            train_clean, attack, poison_indices, normalize, num_classes
        )

    # The requested rate is capped at the eligible pool, so it is not always what
    # was applied. Measuring it here is the only place both numbers exist at once.
    realized_poison_rate = len(poison_indices) / max(len(labels), 1)
    if abs(realized_poison_rate - poison_rate) > 1e-9:
        print(
            f"poison rate requested {poison_rate:.4f}, realized "
            f"{realized_poison_rate:.4f} ({len(poison_indices)} of {len(labels)} "
            "samples), capped by the eligible pool"
        )

    # A poisoned sample with no perturbed base would silently train the patch-only
    # variant while args.json records the adversarial variant, so refuse before training.
    incoherent = adversarial_config_error(config)
    if incoherent:
        raise ValueError(incoherent)

    absent = missing_adversarial_bases(config, poison_indices)
    if absent:
        raise ValueError(
            f"{len(absent)} of {len(poison_indices)} poisoned indices have no adversarial "
            f"base in {config.adversarial_dir} (first missing: {absent[:5]}). "
            "Generate them with `python -m cli.lc_bases`."
        )

    return dataset, realized_poison_rate


def build_training_loader(
    args: argparse.Namespace, image_size: int
) -> tuple[DataLoader, int, Attack, object, float]:
    """The poisoned training loader, plus the attack record it was built from.

    Evaluation is handled separately by evaluation.metrics and evaluation.loaders, so
    this has no eval-loader concerns at all.
    """
    spec = DATASET_REGISTRY[args.dataset]
    transform = base_image_transform(image_size)
    train_clean, _ = load_clean_datasets(args.dataset, transform, args.raw_data_dir)
    # Subset before poison-index selection so poison_rate is measured against the
    # truncated pool, mirroring the eval-side subset in evaluation.loaders.
    train_clean = limit_dataset(train_clean, args.max_samples, args.seed)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    config = resolve_config(args.attack, args.poisoned_dir)
    cover_rate = resolve_cover_rate(
        args.attack, args.poison_rate, getattr(args, "cover_rate", None)
    )
    if cover_rate is not None:
        config = replace(config, cover_rate=cover_rate)
        print(f"cover rate for {args.attack}: {cover_rate:.4f}")
    config = apply_config_overrides(
        config, parse_attack_overrides(args.attack_override)
    )
    attack = build_attack(args.attack, config, image_size, args.target_label)

    poisoned_train, realized_poison_rate = build_training_set(
        train_clean,
        attack,
        config,
        args.poison_rate,
        args.seed,
        normalize,
        spec.num_classes,
    )
    if getattr(args, "evade_psbd", False):
        # The attacker knows which samples it poisoned. This flag never reaches
        # the defender's side of any evaluation.
        poisoned_train = FlaggedPoisonedSet(poisoned_train)

    train_loader = DataLoader(
        poisoned_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    return train_loader, spec.num_classes, attack, config, realized_poison_rate


def resolve_evasion(
    args: argparse.Namespace,
    num_classes: int,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[dict | None, float]:
    """The adaptive attacker's probe config, and the rate it will run at.

    Returns (None, requested rate) when --evade-psbd is off. A requested rate of
    0.0 means "calibrate", so a throwaway model is built to find the rate whose
    clean-validation shift ratio matches the defender's own sigma target, and the
    RNG is restored afterwards so training starts from the same state either way.
    """
    if not args.evade_psbd:
        return None, args.evade_rate

    probe = {
        "position": args.evade_position,
        "operator": args.evade_operator,
        "architecture": args.architecture,
    }

    evade_rate = args.evade_rate
    if evade_rate == 0.0:
        calibration_model = build_model(args.architecture, num_classes).to(device)
        evade_rate = calibrate_probe_rate(calibration_model, val_loader, probe, device)
        del calibration_model
        torch.cuda.empty_cache()
        seed_everything(args.seed, workers=True)

    evasion = {
        "probe": {**probe, "rate": evade_rate},
        "weight": args.evade_weight,
        "passes": args.evade_passes,
    }
    return evasion, evade_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a backdoored ViT or Swin")
    parser.add_argument("--dataset", choices=tuple(DATASET_REGISTRY), required=True)
    parser.add_argument("--attack", choices=ATTACK_NAMES, required=True)
    parser.add_argument("--poison-rate", type=float, required=True)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--architecture", choices=("vit", "swin"), default="vit")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--evade-psbd",
        action="store_true",
        help=(
            "adaptive attacker: add a hinge penalty that raises poisoned samples' "
            "prediction shift onto the clean distribution, removing the statistic "
            "PSBD reads while keeping the backdoor"
        ),
    )
    parser.add_argument("--evade-weight", type=float, default=1.0)
    parser.add_argument("--evade-position", default="before_attention_norm")
    parser.add_argument("--evade-operator", default="dropout")
    parser.add_argument(
        "--evade-rate",
        type=float,
        default=0.0,
        help="perturbation rate for the evasion probe. 0 (default) auto-calibrates "
        "to the sigma=0.6 matched rate on the validation set before training.",
    )
    parser.add_argument("--evade-passes", type=int, default=3)
    parser.add_argument(
        "--model-dropout-train",
        type=float,
        default=0.0,
        help=(
            "train WITH dropout at this rate. PSBD requires a dropout-free model. "
            "This exists to test that requirement, not to change the default"
        ),
    )
    parser.add_argument("--use-sam", action="store_true")
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument(
        "--poisoned-dir", default="", help="required only for the generated attack"
    )
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--attack-override",
        # extend rather than the default store, so a repeated flag accumulates.
        # With plain nargs a repeated flag replaces the earlier flag, and
        # --attack-override a=1 --attack-override b=2 keeps only b.
        action="extend",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override attack-config fields, e.g. --attack-override patch_size=32 or "
        "strength=2.0. Needed for trigger dose-response sweeps, which otherwise cannot "
        "vary the trigger at all because resolve_config returns default_config().",
    )
    parser.add_argument(
        "--cover-rate",
        type=float,
        default=None,
        help="Cover-sample rate. Default None resolves the value the attack's paper "
        "specifies as a multiple of the poisoning rate: 2x for wanet's noise mode, "
        "1x for adaptive_blend and bpp. TaCT keeps its config constant because its "
        "reference selects cover by class, not by rate.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help="Snapshot every Nth epoch into its own checkpoint folder. 0 (default) "
        "disables snapshotting entirely, so existing runs are unaffected.",
    )
    parser.add_argument(
        "--checkpoint-dense-until",
        type=int,
        default=0,
        help="Snapshot EVERY epoch up to and including this one, then fall back to "
        "--checkpoint-freq. Detection quality moves while the model is still "
        "fitting the training set and stops moving once it interpolates, so a flat "
        "interval spends most of its snapshots in the flat region.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Truncate each dataset to this many samples, reproducibly, for a fast "
        "smoke run (combine with --epochs 1). -1 (default) uses the whole dataset. "
        "This alone does not imply smoke semantics, and --epochs is independent.",
    )
    return parser.parse_args()


def snapshot_epochs(total: int, dense_until: int, freq: int) -> set[int]:
    """Which epochs to snapshot: every one through dense_until, then every freq-th.

    2 phases because the interesting part of the trajectory is the approach to
    training-set interpolation, which is early. Once the model interpolates the
    detection metrics stop moving, so the tail only needs sampling.
    """
    if dense_until <= 0 and freq <= 0:
        return set()
    chosen = set(range(1, min(dense_until, total) + 1))
    if freq > 0:
        chosen |= {e for e in range(1, total + 1) if e % freq == 0 and e > dense_until}
    return chosen


def build_train_eval_loader(train_loader: DataLoader, args) -> DataLoader:
    """The poisoned training set the model actually saw, unshuffled, for train accuracy.

    Measuring interpolation on the clean split is wrong for a clean-label attack. Its
    eligible pool is a single class, so at a high enough rate every target-class
    image is poisoned and the model never saw a clean one. Clean accuracy on that
    class then reads near 0 while the model fits what it was given almost
    perfectly, which looks like a failure to fit and is not.

    Capped at 10000 samples because this runs at every snapshot and the question is
    when the model interpolates, which a fixed subsample tracks as well as the full
    set.
    """
    dataset = train_loader.dataset
    if len(dataset) > TRAIN_EVAL_SAMPLES:
        generator = torch.Generator().manual_seed(args.seed)
        picked = torch.randperm(len(dataset), generator=generator)[:TRAIN_EVAL_SAMPLES]
        dataset = Subset(dataset, picked.tolist())
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def build_snapshot_hook(
    args,
    num_classes,
    attack,
    config,
    realized_poison_rate,
    started_at,
    device,
    train_loader,
):
    """Write each chosen epoch as a full checkpoint folder, or None if disabled.

    Each snapshot is a complete, self-describing checkpoint directory rather than
    a bare state dict, so cli.sweep, cli.analyze and the table generator read it
    with no changes. That is the reason for the naming: <base>_ep07 sits beside
    <base> and looks like any other run.

    ASR is deliberately left None. Evaluating it costs a full poisoned pass and the
    sweep backfills it into args.json from the PSBD baseline cache anyway, so paying
    for it at every snapshot would be waste.
    """
    chosen = snapshot_epochs(
        args.epochs, args.checkpoint_dense_until, args.checkpoint_freq
    )
    if not chosen:
        return None

    train_eval_loader = build_train_eval_loader(train_loader, args)
    base_dir = os.path.dirname(args.output)
    filename = os.path.basename(args.output)
    print(f"snapshotting {len(chosen)} epochs: {sorted(chosen)}")

    def hook(model, epoch: int, validation_accuracy: float) -> None:
        if epoch not in chosen:
            return
        was_training = model.training
        train_accuracy = clean_accuracy(model, train_eval_loader, device, True)
        metadata = checkpoint_metadata(
            dataset=args.dataset,
            attack=args.attack,
            label_mode=attack.label_mode,
            target_label=args.target_label,
            poison_rate=args.poison_rate,
            realized_poison_rate=realized_poison_rate,
            cover_rate=getattr(config, "cover_rate", 0.0),
            architecture=args.architecture,
            use_sam=args.use_sam,
            rho=args.rho,
            epochs=epoch,
            seed=args.seed,
            max_samples=args.max_samples,
            clean_accuracy=validation_accuracy,
            asr=None,
            started_at=started_at,
            ended_at=utc_timestamp(),
            model_dropout=args.model_dropout_train,
        )
        # The trajectory fields the interpolation question turns on. Kept out of
        # checkpoint_metadata so its key set stays identical across entrypoints.
        metadata["epoch"] = epoch
        metadata["train_accuracy"] = train_accuracy
        metadata["snapshot_of"] = os.path.basename(base_dir)
        save_checkpoint(
            model,
            num_classes,
            os.path.join(f"{base_dir}_ep{epoch:02d}", filename),
            metadata=metadata,
        )
        print(
            f"  snapshot epoch {epoch}: train_acc={train_accuracy:.4f} "
            f"val_acc={validation_accuracy:.4f}"
        )
        model.train(was_training)

    return hook


def main() -> None:
    args = parse_args()
    # -1 is a CLI-only sentinel for "no limit". Normalize it to None immediately so
    # no subsetting code ever sees it, since -1 would slice off the last sample.
    args.max_samples = None if args.max_samples == -1 else args.max_samples
    seed_everything(args.seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    started_at = utc_timestamp()
    image_size = DATASET_REGISTRY[args.dataset].image_size

    train_loader, num_classes, attack, config, realized_poison_rate = (
        build_training_loader(args, image_size)
    )
    val_loader = build_clean_loader(
        args.dataset,
        args.raw_data_dir,
        args.batch_size,
        args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # Reseed right before the regular workflow so model init and training start from
    # an identical RNG state whether or not --max-samples triggered any subsetting.
    seed_everything(args.seed, workers=True)

    evasion, evade_rate = resolve_evasion(args, num_classes, val_loader, device)

    model = train_classifier(
        args.architecture,
        num_classes,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        use_sam=args.use_sam,
        rho=args.rho,
        model_dropout=args.model_dropout_train,
        evasion=evasion,
        on_epoch_end=build_snapshot_hook(
            args,
            num_classes,
            attack,
            config,
            realized_poison_rate,
            started_at,
            device,
            train_loader,
        ),
    )
    ended_at = utc_timestamp()

    metrics = evaluate_attack(
        model,
        args.dataset,
        args.attack,
        config,
        args.target_label,
        device,
        args.raw_data_dir,
        args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"final ASR={metrics['asr']:.4f} CA={metrics['clean_accuracy']:.4f}")

    # The cover count, not just the requested rate. A cover mechanism that silently
    # produced zero samples looks identical to a successful run in every other field.
    n_cover = len(getattr(train_loader.dataset, "cover_indices", ()) or ())
    print(f"cover samples: {n_cover}")

    metadata = checkpoint_metadata(
        dataset=args.dataset,
        attack=args.attack,
        label_mode=attack.label_mode,
        target_label=args.target_label,
        poison_rate=args.poison_rate,
        realized_poison_rate=realized_poison_rate,
        cover_rate=getattr(config, "cover_rate", 0.0),
        architecture=args.architecture,
        use_sam=args.use_sam,
        rho=args.rho,
        epochs=args.epochs,
        seed=args.seed,
        max_samples=args.max_samples,
        clean_accuracy=metrics["clean_accuracy"],
        asr=metrics["asr"],
        started_at=started_at,
        ended_at=ended_at,
        evasion={
            "weight": args.evade_weight,
            "position": args.evade_position,
            "operator": args.evade_operator,
            "rate": evade_rate,
            "rate_requested": args.evade_rate,
            "passes": args.evade_passes,
        }
        if args.evade_psbd
        else None,
        model_dropout=args.model_dropout_train,
    )
    metadata["n_cover"] = n_cover
    # Without this, evaluation rebuilds the attack from default_config() and a run
    # trained with a modified trigger is scored against a trigger it never saw.
    metadata["attack_config_overrides"] = config_overrides(config, args.attack)
    save_checkpoint(model, num_classes, args.output, metadata=metadata)
    print(f"saved {args.output}")
    print(
        f"time taken: {args.dataset} {args.attack} rate {args.poison_rate} took "
        f"{(time.time() - started) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
