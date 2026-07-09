"""Poison a dataset and train a backdoored ViT-B/16 or Swin-S.

The BackdoorBench-style flow: load clean data, poison a fraction of the training
set, train, evaluate ASR and clean accuracy, then save in the attack_result.pt
format the sweep reads. Run from inside this directory.

Attacks that use cover samples (adaptive_blend, tact) are detected from their
config and routed through the cover-sample dataset. Everything else uses the plain
poisoning path. The training loop itself is the same in both cases.

Example
    python train_backdoor.py --dataset cifar10 --attack badnet_a2a --poison-rate 0.1 \
        --architecture vit --epochs 10 --output vit_b_16_weights/cifar10_badnet_a2a_0_1/attack_result.pt
"""

import argparse

import torch
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader

from attack_generated import GeneratedConfig
from attacks import ATTACK_NAMES, build_attack, default_config
from config import DATASET_REGISTRY
from datasets import extract_labels, load_clean_datasets
from detection import attack_success_rate, clean_accuracy
from poison import (
    CoverPoisonedTrainingSet,
    FullyPoisonedTestSet,
    PoisonedTrainingSet,
    choose_indices_with_cover,
    choose_poison_indices,
)
from train import save_checkpoint, train_classifier
import time


def working_resolution(dataset_name: str) -> int:
    # Triggers are defined at the native resolution and the model's own Resize
    # upscales to 224, matching how BackdoorBench applies triggers.
    return 64 if dataset_name == "tiny" else 32


def base_transform(image_size: int) -> transforms_v2.Compose:
    # Stop at 0-to-1 pixel values so the trigger can be applied before normalizing.
    return transforms_v2.Compose(
        [transforms_v2.Resize((image_size, image_size)), transforms_v2.ToTensor()]
    )


def resolve_config(attack_name: str, poisoned_dir: str):
    if attack_name == "generated":
        return GeneratedConfig(poisoned_dir=poisoned_dir)
    return default_config(attack_name)


def build_training_set(
    train_clean, attack, config, poison_rate, seed, normalize, num_classes
):
    """Route to the cover-sample dataset when the attack config asks for it."""
    labels = extract_labels(train_clean)
    cover_rate = getattr(config, "cover_rate", 0.0)
    source_classes = getattr(config, "source_classes", None)

    if cover_rate > 0.0 or source_classes is not None:
        poison_indices, cover_indices = choose_indices_with_cover(
            labels, attack, poison_rate, cover_rate, source_classes, seed
        )
        return CoverPoisonedTrainingSet(
            train_clean, attack, poison_indices, cover_indices, normalize, num_classes
        )

    poison_indices = choose_poison_indices(labels, attack, poison_rate, seed)
    return PoisonedTrainingSet(
        train_clean, attack, poison_indices, normalize, num_classes
    )


def build_poisoned_loaders(args, image_size: int):
    spec = DATASET_REGISTRY[args.dataset]
    transform = base_transform(image_size)
    train_clean, test_clean = load_clean_datasets(
        args.dataset, transform, args.raw_data_dir
    )
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    config = resolve_config(args.attack, args.poisoned_dir)
    attack = build_attack(args.attack, config, image_size, args.target_label)

    poisoned_train = build_training_set(
        train_clean,
        attack,
        config,
        args.poison_rate,
        args.seed,
        normalize,
        spec.num_classes,
    )
    clean_test = PoisonedTrainingSet(
        test_clean, attack, set(), normalize, spec.num_classes
    )
    test_labels = extract_labels(test_clean)
    backdoor_test = FullyPoisonedTestSet(
        test_clean, test_labels, attack, normalize, spec.num_classes
    )

    def loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
        )

    return (
        loader(poisoned_train, True),
        loader(clean_test, False),
        loader(backdoor_test, False),
        spec.num_classes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a backdoored ViT or Swin")
    parser.add_argument("--dataset", choices=tuple(DATASET_REGISTRY), required=True)
    parser.add_argument("--attack", choices=ATTACK_NAMES, required=True)
    parser.add_argument("--poison-rate", type=float, required=True)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--architecture", choices=("vit", "swin"), default="vit")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--use-sam", action="store_true")
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument(
        "--poisoned-dir", default="", help="required only for the generated attack"
    )
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    image_size = working_resolution(args.dataset)

    train_loader, clean_loader, backdoor_loader, num_classes = build_poisoned_loaders(
        args, image_size
    )

    model = train_classifier(
        args.architecture,
        num_classes,
        train_loader,
        clean_loader,
        device,
        epochs=args.epochs,
        use_sam=args.use_sam,
        rho=args.rho,
    )

    asr = attack_success_rate(model, backdoor_loader, device, use_bfloat16=True)
    ca = clean_accuracy(model, clean_loader, device, use_bfloat16=True)
    print(f"final ASR={asr:.4f} CA={ca:.4f}")

    save_checkpoint(model, num_classes, args.output)
    print(f"saved {args.output}")
    print(
        f"time taken: {args.dataset} {args.attack} rate {args.poison_rate} took {(time.time() - start) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
