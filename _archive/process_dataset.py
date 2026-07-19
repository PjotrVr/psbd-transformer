from __future__ import annotations

import argparse

from lightning import seed_everything

from attacks import ATTACK_REGISTRY
from utils import (apply_transform_to_all, create_benign_splits,
                   create_poison_base_splits, load_dataset_tensors,
                   save_benign_splits, save_poisoned_splits)


def build_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset preprocessing pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "gtsrb"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./preprocessed_data",
        help="Root folder where processed tensors are written.",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="./raw_data",
        help="Root folder for torchvision raw dataset downloads.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio split from train set.",
    )

    parser.add_argument(
        "--benign", action="store_true", help="Generate benign split only."
    )
    parser.add_argument(
        "--attack",
        type=str,
        choices=sorted(ATTACK_REGISTRY.keys()),
        help="Attack used for poisoning mode.",
    )
    parser.add_argument(
        "--poison_rate",
        type=float,
        default=0.1,
        help="Ratio of train set moved to poisoned train subset.",
    )
    parser.add_argument(
        "--val_heldout_ratio",
        type=float,
        default=0.5,
        help="Ratio of validation set held out as clean-only validation data.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_base_parser()
    pre_args, _ = parser.parse_known_args()

    if pre_args.attack is not None:
        ATTACK_REGISTRY[pre_args.attack].add_parser_arguments(parser)

    return parser.parse_args()


def run_benign(args: argparse.Namespace) -> str:
    train_data, train_labels, test_data, test_labels = load_dataset_tensors(
        dataset_name=args.dataset,
        raw_data_dir=args.raw_data_dir,
    )
    splits = create_benign_splits(
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    return save_benign_splits(args.save_dir, args.dataset, splits)


def run_poisoned(args: argparse.Namespace) -> str:
    attack_module = ATTACK_REGISTRY[args.attack]
    attack_config = attack_module.namespace_to_config(args)

    train_data, train_labels, test_data, test_labels = load_dataset_tensors(
        dataset_name=args.dataset,
        raw_data_dir=args.raw_data_dir,
    )
    benign_splits = create_benign_splits(
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    poison_splits = create_poison_base_splits(
        benign_splits=benign_splits,
        poison_rate=args.poison_rate,
        val_heldout_ratio=args.val_heldout_ratio,
        seed=args.seed,
    )

    image_shape = (
        int(poison_splits.clean_train_data.shape[2]),
        int(poison_splits.clean_train_data.shape[3]),
    )
    transform_obj = attack_module.build_transform(
        attack_config, image_shape=image_shape
    )

    backdoor_train_data, backdoor_train_labels = apply_transform_to_all(
        poison_splits.poison_source_train_data,
        poison_splits.poison_source_train_labels,
        transform_obj,
    )
    backdoor_val_data, backdoor_val_labels = apply_transform_to_all(
        poison_splits.clean_val_data,
        poison_splits.clean_val_labels,
        transform_obj,
    )
    backdoor_test_data, backdoor_test_labels = apply_transform_to_all(
        poison_splits.clean_test_data,
        poison_splits.clean_test_labels,
        transform_obj,
    )

    return save_poisoned_splits(
        save_root=args.save_dir,
        dataset_name=args.dataset,
        attack_name=args.attack,
        poison_rate=args.poison_rate,
        target_label=int(attack_config["target_label"]),
        clean_train_data=poison_splits.clean_train_data,
        clean_train_labels=poison_splits.clean_train_labels,
        backdoor_train_data=backdoor_train_data,
        backdoor_train_labels=backdoor_train_labels,
        clean_val_data=poison_splits.clean_val_data,
        clean_val_labels=poison_splits.clean_val_labels,
        val_heldout_data=poison_splits.val_heldout_data,
        val_heldout_labels=poison_splits.val_heldout_labels,
        backdoor_val_data=backdoor_val_data,
        backdoor_val_labels=backdoor_val_labels,
        clean_test_data=poison_splits.clean_test_data,
        clean_test_labels=poison_splits.clean_test_labels,
        backdoor_test_data=backdoor_test_data,
        backdoor_test_labels=backdoor_test_labels,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.benign or args.attack is None:
        output_dir = run_benign(args)
        print(f"Saved benign data to: {output_dir}")
        return

    output_dir = run_poisoned(args)
    print(f"Saved poisoned data to: {output_dir}")


if __name__ == "__main__":
    main()
