"""Generate the adversarially perturbed base images the Label-Consistent attack needs.

1 cache per (dataset, target label, epsilon). The surrogate is a model trained on
clean data. A single surrogate serves every victim architecture, because the
attacker publishes a poisoned dataset rather than a model.

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
    return [
        i
        for i, label in enumerate(extract_labels(dataset))
        if int(label) == target_label
    ]


def perturb_all(model, loader, normalize, epsilon, steps, device, seed):
    """PGD over every batch, returning perturbed images and clean/attacked accuracy."""
    generator = torch.Generator(device=device).manual_seed(seed)
    perturbed, clean_correct, attacked_correct, total = [], 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        clean_correct += accuracy_on(model, images, labels, normalize) * len(labels)
        adversarial = pgd_perturb(
            model, images, labels, normalize, epsilon, steps, generator
        )
        attacked_correct += accuracy_on(model, adversarial, labels, normalize) * len(
            labels
        )
        total += len(labels)
        perturbed.append(adversarial.cpu())
    return torch.cat(perturbed), clean_correct / total, attacked_correct / total


def main() -> None:
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
    args = parser.parse_args()

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
    surrogate_path = f"{args.checkpoints_dir}/{folder}/attack_result.pt"
    model = load_checkpoint(args.surrogate_architecture, surrogate_path, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

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
        save_bases(
            directory,
            indices,
            images,
            {
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
            },
        )
        print(
            f"  eps={epsilon * 255:.0f}/255  target-class accuracy "
            f"{clean_accuracy:.3f} -> {attacked_accuracy:.3f}  "
            f"({time.time() - started:.0f}s)  {directory}"
        )


if __name__ == "__main__":
    main()
