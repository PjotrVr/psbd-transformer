"""Latent-space analysis for a trained checkpoint: TAC, backdoor direction, CKA, PCA.

A worked example of the latent techniques on your own model. It builds paired
clean and triggered versions of the same test images, extracts the per-layer CLS
features, and reports where the trigger lives and how separable clean and backdoor
representations are.

Run it on a benign checkpoint and on a backdoored one and compare. The benign
model should show small TAC everywhere, because it never learned the trigger. The
backdoored model should show TAC and the backdoor direction norm rising at the
layer that carries the backdoor.

Run from inside this directory.

Example
    python analyze_latent.py --dataset cifar100 --attack badnet_a2o \
        --checkpoint vit_b_16_weights/cifar100_badnet_a2o_0_1/attack_result.pt
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, Subset

from attacks import build_attack, default_config
from cka import debiased_linear_cka
from config import DATASET_REGISTRY
from datasets import load_clean_datasets
from direction import backdoor_direction, trigger_activated_change
from dropout import reset_dropout
from embedding import pca_project
from features import extract_layer_features
from models import load_checkpoint
from poison import PoisonedTrainingSet


def working_resolution(dataset_name: str) -> int:
    return 64 if dataset_name == "tiny" else 32


def build_paired_loaders(
    dataset_name, attack, raw_data_dir, batch_size, sample_count, seed
):
    """Return clean and triggered loaders over the same images, index-aligned.

    Both loaders are unshuffled and drawn from the same subset, so position i in
    each refers to the same test image, once clean and once triggered. That
    pairing is what TAC and the backdoor direction require.
    """
    image_size = working_resolution(dataset_name)
    spec = DATASET_REGISTRY[dataset_name]
    transform = transforms_v2.Compose(
        [transforms_v2.Resize((image_size, image_size)), transforms_v2.ToTensor()]
    )
    _, test_clean = load_clean_datasets(dataset_name, transform, raw_data_dir)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        len(test_clean), size=min(sample_count, len(test_clean)), replace=False
    )
    subset = Subset(test_clean, chosen.tolist())

    clean_set = PoisonedTrainingSet(subset, attack, set(), normalize, spec.num_classes)
    triggered_set = PoisonedTrainingSet(
        subset, attack, set(range(len(subset))), normalize, spec.num_classes
    )

    clean_loader = DataLoader(clean_set, batch_size=batch_size, shuffle=False)
    backdoor_loader = DataLoader(triggered_set, batch_size=batch_size, shuffle=False)
    return clean_loader, backdoor_loader


def per_layer_report(clean_features, backdoor_features) -> dict[int, float]:
    """Print per-layer backdoor direction norm, max TAC, and clean-vs-backdoor CKA.

    A rising direction norm and TAC mark where the trigger becomes dominant. A
    falling CKA marks where clean and backdoor representations diverge.
    """
    print(f"{'layer':>5} {'direction_norm':>15} {'max_tac':>10} {'cka':>8}")
    direction_norms: dict[int, float] = {}
    for layer in sorted(clean_features):
        direction = backdoor_direction(clean_features[layer], backdoor_features[layer])
        tac = trigger_activated_change(clean_features[layer], backdoor_features[layer])
        cka_value = debiased_linear_cka(clean_features[layer], backdoor_features[layer])
        direction_norms[layer] = direction.norm().item()
        print(
            f"{layer:>5} {direction_norms[layer]:>15.3f} {tac.max().item():>10.3f} {cka_value:>8.3f}"
        )
    return direction_norms


def save_pca_scatter(
    clean_features, backdoor_features, layer: int, output_path: str
) -> None:
    """PCA of the CLS features at one layer, clean against backdoor.

    PCA is the honest first view, because the backdoor is hypothesized to be a
    linear direction and a linear projection cannot invent structure.
    """
    combined = torch.cat([clean_features[layer], backdoor_features[layer]], dim=0)
    projected = pca_project(combined, num_components=2)
    clean_count = clean_features[layer].shape[0]

    plt.figure(figsize=(6, 6))
    plt.scatter(
        projected[:clean_count, 0],
        projected[:clean_count, 1],
        s=8,
        alpha=0.5,
        label="clean",
    )
    plt.scatter(
        projected[clean_count:, 0],
        projected[clean_count:, 1],
        s=8,
        alpha=0.5,
        label="backdoor",
    )
    plt.title(f"CLS features at layer {layer}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"scatter saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latent-space analysis of a checkpoint"
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_REGISTRY), required=True)
    parser.add_argument(
        "--attack",
        required=True,
        help="the trigger to probe with, for example badnet_a2o",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architecture", choices=("vit", "swin"), default="vit")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--output", default="latent_scatter.png")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matplotlib.use(
        "Agg"
    )  # save figures without a display, since cluster nodes have none
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = working_resolution(args.dataset)
    attack = build_attack(
        args.attack, default_config(args.attack), image_size, args.target_label
    )
    clean_loader, backdoor_loader = build_paired_loaders(
        args.dataset,
        attack,
        args.raw_data_dir,
        args.batch_size,
        args.samples,
        args.seed,
    )

    model = load_checkpoint(args.architecture, args.checkpoint, device)
    reset_dropout(
        model, "pre_residual"
    )  # dropout off, we want the clean baseline features

    clean_features = extract_layer_features(
        model, clean_loader, device, use_bfloat16=True
    )
    backdoor_features = extract_layer_features(
        model, backdoor_loader, device, use_bfloat16=True
    )

    direction_norms = per_layer_report(clean_features, backdoor_features)
    peak_layer = max(direction_norms, key=direction_norms.get)
    print(f"backdoor direction is strongest at layer {peak_layer}")

    save_pca_scatter(clean_features, backdoor_features, peak_layer, args.output)


if __name__ == "__main__":
    main()
