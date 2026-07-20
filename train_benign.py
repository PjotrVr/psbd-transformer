"""Train benign, unpoisoned ViT-B/16 models on each dataset.

These are the negative controls for the detection and latent experiments: a model
that never saw a trigger. Clean accuracy is recorded into the checkpoint metadata,
and the checkpoint is saved under <dataset>_benign/attack_result.pt in the same
format the sweep reads.

A benign model has no attack of its own, so to probe one with the detector later
you supply the trigger explicitly, for example through analyze_latent.py --attack,
rather than reading it from the checkpoint.

Run from inside this directory.

Example
    python train_benign.py --datasets cifar10 cifar100 gtsrb tiny --epochs 15
"""

import argparse

import torch
import torchvision.transforms.v2 as transforms_v2
from lightning import seed_everything
from torch.utils.data import DataLoader

from evaluate import evaluate_benign
from loaders import build_clean_loader
from utils.config import DATASET_REGISTRY
from utils.datasets import limit_dataset, load_clean_datasets
from train import checkpoint_metadata, save_checkpoint, train_classifier, utc_timestamp
import time


def build_benign_train_loader(
    dataset_name: str,
    raw_data_dir: str,
    batch_size: int,
    num_workers: int = 8,
    max_samples: int | None = None,
    seed: int = 0,
) -> tuple[DataLoader, int]:
    """The shuffled training split only. Evaluation reuses loaders.build_clean_loader,
    the same function metrics.py and train_backdoor.py use for their clean loaders.
    """
    spec = DATASET_REGISTRY[dataset_name]
    transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((spec.image_size, spec.image_size)),
            transforms_v2.ToTensor(),
            transforms_v2.Normalize(mean=spec.mean, std=spec.std),
        ]
    )
    train_dataset, _ = load_clean_datasets(dataset_name, transform, raw_data_dir)
    train_dataset = limit_dataset(train_dataset, max_samples, seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return train_loader, spec.num_classes


def train_one_benign(
    dataset_name: str, args: argparse.Namespace, device: torch.device
) -> float:
    # Seeded per dataset, not once before the loop, so each dataset's run is
    # reproducible independent of loop order or an earlier dataset's failure.
    seed_everything(args.seed)
    start = time.time()
    started_at = utc_timestamp()
    train_loader, num_classes = build_benign_train_loader(
        dataset_name,
        args.raw_data_dir,
        args.batch_size,
        args.num_workers,
        args.max_samples,
        args.seed,
    )
    val_loader = build_clean_loader(
        dataset_name,
        args.raw_data_dir,
        args.batch_size,
        args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # Reseed right before the regular workflow so model init and training start from
    # an identical RNG state whether or not --max-samples triggered any subsetting.
    seed_everything(args.seed)
    model = train_classifier(
        args.architecture,
        num_classes,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        use_sam=args.use_sam,
        rho=args.rho,
    )
    ended_at = utc_timestamp()
    accuracy = evaluate_benign(
        model,
        dataset_name,
        device,
        args.raw_data_dir,
        args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )["clean_accuracy"]

    # Canonical name matches normalize_checkpoints.py's template: architecture
    # is always explicit, adam gets no optimizer tag, SAM's rho tag always has
    # an underscore before the digits so a rho sweep keeps each run separate.
    folder_name = f"{args.architecture}_{dataset_name}_benign"
    if args.use_sam:
        folder_name += f"_sam_rho_{str(args.rho).replace('.', '_')}"
    output_path = f"{args.weights_dir}/{folder_name}/attack_result.pt"
    save_checkpoint(
        model,
        num_classes,
        output_path,
        metadata=checkpoint_metadata(
            dataset=dataset_name,
            attack="benign",
            label_mode=None,
            target_label=0,
            poison_rate=0.0,
            cover_rate=0.0,
            architecture=args.architecture,
            use_sam=args.use_sam,
            rho=args.rho,
            epochs=args.epochs,
            seed=args.seed,
            max_samples=args.max_samples,
            clean_accuracy=accuracy,
            asr=None,
            started_at=started_at,
            ended_at=ended_at,
        ),
    )
    print(f"{folder_name} clean accuracy {accuracy:.4f}, saved {output_path}")
    print(f"time taken: {folder_name} took {(time.time() - start) / 60:.1f} min")
    return accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train benign ViT models")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10", "cifar100", "gtsrb", "tiny"],
        choices=tuple(DATASET_REGISTRY),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--architecture", choices=("vit", "swin"), default="vit")
    parser.add_argument("--use-sam", action="store_true")
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--weights-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Truncate each dataset to this many samples, reproducibly, for a fast "
        "smoke run (combine with --epochs 1). -1 (default) uses the whole dataset. "
        "This alone does not imply smoke semantics; --epochs is independent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # -1 is a CLI-only sentinel for "no limit". Normalize once here, before the
    # per-dataset loop, so no subsetting code ever sees it (-1 would slice off one
    # sample instead of meaning "no limit").
    args.max_samples = None if args.max_samples == -1 else args.max_samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracies = {}
    for dataset_name in args.datasets:
        print(f"training {dataset_name}_benign")
        try:
            accuracies[dataset_name] = train_one_benign(dataset_name, args, device)
        except Exception as error:
            # One dataset failing should not waste the datasets after it, so report
            # and continue rather than letting the exception abort the whole run.
            print(f"FAILED {dataset_name}_benign: {error}")

    print("summary of clean accuracy:")
    for dataset_name, accuracy in accuracies.items():
        print(f"  {dataset_name}_benign {accuracy:.4f}")


if __name__ == "__main__":
    main()
