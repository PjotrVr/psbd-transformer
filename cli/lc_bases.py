"""Generate the adversarially perturbed base images the Label-Consistent attack needs.

Reads the clean training split of 1 dataset and the benign surrogate checkpoint
under checkpoints/, and writes 1 cache of perturbed target-class images per
(dataset, target label, epsilon) under results/ through attacks.adversarial.
The surrogate is a model trained on clean data. A single surrogate serves every
victim architecture, because the attacker publishes a poisoned dataset rather
than a model.

    python -m cli.lc_bases --dataset gtsrb --target-label 1 --epsilon 0.0627

The printed target-class accuracy before and after is the check that matters: if
the perturbation does not collapse it, the bases carry no attack and no amount
of training will produce an attack.
"""

import argparse
import time

import torch
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, Subset

from attacks.adversarial import accuracy_on, bases_directory, pgd_perturb, save_bases
from data.registry import DATASET_REGISTRY
from data.loading import base_image_transform, extract_labels, load_clean_datasets
from models.backbones import load_checkpoint
from utils.provenance import current_git_commit


def target_class_indices(dataset, target_label: int) -> list[int]:
    """Dataset indices eligible for clean-label poisoning: the target class itself."""
    indices = [
        i
        for i, label in enumerate(extract_labels(dataset))
        if int(label) == target_label
    ]
    return indices


def perturb_all(model, loader, normalize, epsilon, steps, device, seed):
    """PGD over every batch of the loader, in loader order.

    Every batch yields images of shape (batch, channels, height, width) and labels
    of shape (batch,). Returns the perturbed images concatenated to shape
    (n, channels, height, width) on the CPU, the surrogate's accuracy on the
    unperturbed images and its accuracy on the perturbed ones.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    perturbed, clean_correct, attacked_correct, total = [], 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        clean_correct += accuracy_on(model, images, labels, normalize) * len(labels)
        adversarial = pgd_perturb(
            model, images, labels, normalize, epsilon, steps, generator
        )  # (batch, channels, height, width)
        attacked_correct += accuracy_on(model, adversarial, labels, normalize) * len(
            labels
        )
        total += len(labels)
        perturbed.append(adversarial.cpu())

    images = torch.cat(perturbed)  # (n, channels, height, width)
    clean_accuracy = clean_correct / total
    attacked_accuracy = attacked_correct / total
    return images, clean_accuracy, attacked_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=[8 / 255, 16 / 255, 32 / 255],
        help="L-inf budget in 0-to-1 pixel units; Turner reports 8, 16 and 32 over 255",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--surrogate-architecture", default="vit")
    parser.add_argument(
        "--surrogate-folder",
        default=None,
        help="defaults to checkpoints/{arch}_{dataset}_benign",
    )
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_frozen_surrogate(args: argparse.Namespace, folder: str, device):
    """The benign surrogate in eval mode with every parameter frozen.

    PGD needs gradients with respect to the input only, so freezing the weights
    keeps the backward pass from building parameter gradients it never uses.
    """
    surrogate_path = f"{args.checkpoints_dir}/{folder}/attack_result.pt"
    model = load_checkpoint(args.surrogate_architecture, surrogate_path, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def bases_metadata(
    args: argparse.Namespace,
    spec,
    folder: str,
    epsilon: float,
    clean_accuracy: float,
    attacked_accuracy: float,
) -> dict:
    """The provenance record saved beside 1 epsilon's cache of perturbed images."""
    metadata = {
        "dataset": args.dataset,
        "target_label": args.target_label,
        "epsilon": epsilon,
        "epsilon_over_255": round(epsilon * 255, 3),
        "steps": args.steps,
        "step_size": 2.5 * epsilon / args.steps,
        "seed": args.seed,
        "image_size": spec.image_size,
        "surrogate_folder": folder,
        "surrogate_architecture": args.surrogate_architecture,
        "surrogate_accuracy_on_target_class": clean_accuracy,
        "attacked_accuracy_on_target_class": attacked_accuracy,
        "git_commit": current_git_commit(),
    }
    return metadata


def print_epsilon_result(
    epsilon: float,
    clean_accuracy: float,
    attacked_accuracy: float,
    seconds: float,
    directory: str,
) -> None:
    """The target-class accuracy before and after the perturbation, for 1 epsilon."""
    print(
        f"  eps={epsilon * 255:.0f}/255  target-class accuracy "
        f"{clean_accuracy:.3f} -> {attacked_accuracy:.3f}  "
        f"({seconds:.0f}s)  {directory}"
    )


def main() -> None:
    args = parse_args()
    spec = DATASET_REGISTRY[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std).to(device)

    train_clean, _ = load_clean_datasets(
        args.dataset, base_image_transform(spec.image_size), args.raw_data_dir
    )
    indices = target_class_indices(train_clean, args.target_label)
    loader = DataLoader(
        Subset(train_clean, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    folder = (
        args.surrogate_folder or f"{args.surrogate_architecture}_{args.dataset}_benign"
    )
    model = load_frozen_surrogate(args, folder, device)

    print(
        f"{args.dataset}: {len(indices)} images in class {args.target_label} "
        f"({len(indices) / len(train_clean):.4%} of the training set), "
        f"surrogate {folder}"
    )
    for epsilon in args.epsilon:
        started = time.time()
        images, clean_accuracy, attacked_accuracy = perturb_all(
            model, loader, normalize, epsilon, args.steps, device, args.seed
        )
        directory = bases_directory(
            args.results_dir, args.dataset, args.target_label, epsilon
        )
        metadata = bases_metadata(
            args, spec, folder, epsilon, clean_accuracy, attacked_accuracy
        )
        save_bases(directory, indices, images, metadata)
        print_epsilon_result(
            epsilon, clean_accuracy, attacked_accuracy, time.time() - started, directory
        )


if __name__ == "__main__":
    main()
