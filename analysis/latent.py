"""Latent-space analysis of a trained checkpoint: TAC, backdoor direction, CKA, PCA.

A worked example of the latent techniques on a model this project trained. It
builds paired clean and triggered versions of the same test images, extracts the
per-layer CLS features, and reports where the trigger lives and how separable clean
and backdoor representations are.

Run it on a benign checkpoint and on a backdoored one and compare. The benign model
should show small TAC everywhere, because it never learned the trigger. The
backdoored model should show TAC and the backdoor direction norm rising at the
layer that carries the backdoor.

This is a library function, not an entrypoint. analyze_latent takes every path and
knob as an argument and returns the per-layer statistics, so a script, a notebook,
or a test can call it without going through a command line.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, Subset

from attacks import build_attack, default_config
from data.registry import DATASET_REGISTRY
from data.loading import base_image_transform, load_clean_datasets
from models.backbones import load_checkpoint
from attacks.poisoning import Attack, PoisonedTrainingSet

from .cka import debiased_linear_cka
from .direction import backdoor_direction, trigger_activated_change
from .embedding import pca_project
from .features import extract_layer_features


def build_paired_loaders(
    dataset_name: str,
    attack: Attack,
    raw_data_dir: str,
    batch_size: int,
    sample_count: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Return clean and triggered loaders over the same images, index-aligned.

    Both loaders are unshuffled and drawn from the same subset, so position i in
    each refers to the same test image, once clean and once triggered. That pairing
    is what TAC and the backdoor direction require.
    """
    spec = DATASET_REGISTRY[dataset_name]
    transform = base_image_transform(spec.image_size)
    _, test_clean = load_clean_datasets(dataset_name, transform, raw_data_dir)

    # Normalization is applied by the dataset wrapper, after the trigger, because
    # every trigger in this project is defined in 0-to-1 pixel space.
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


def layer_statistics(
    clean_features: dict[int, torch.Tensor],
    backdoor_features: dict[int, torch.Tensor],
) -> dict[int, dict[str, float]]:
    """Per-layer backdoor direction norm, max TAC, and clean-vs-backdoor CKA.

    A rising direction norm and TAC mark where the trigger becomes dominant. A
    falling CKA marks where clean and backdoor representations diverge.
    """
    statistics: dict[int, dict[str, float]] = {}
    for layer in sorted(clean_features):
        direction = backdoor_direction(clean_features[layer], backdoor_features[layer])
        tac = trigger_activated_change(clean_features[layer], backdoor_features[layer])
        statistics[layer] = {
            "direction_norm": direction.norm().item(),
            "max_tac": tac.max().item(),
            "cka": debiased_linear_cka(clean_features[layer], backdoor_features[layer]),
        }

    return statistics


def print_layer_report(statistics: dict[int, dict[str, float]]) -> None:
    """Write the per-layer table to stdout, one row per layer."""
    print(f"{'layer':>5} {'direction_norm':>15} {'max_tac':>10} {'cka':>8}")
    for layer in sorted(statistics):
        row = statistics[layer]
        print(
            f"{layer:>5} {row['direction_norm']:>15.3f} "
            f"{row['max_tac']:>10.3f} {row['cka']:>8.3f}"
        )


def save_pca_scatter(
    clean_features: dict[int, torch.Tensor],
    backdoor_features: dict[int, torch.Tensor],
    layer: int,
    output_path: str,
) -> None:
    """PCA of the CLS features at one layer, clean against backdoor.

    PCA is the honest first view, because the backdoor is hypothesized to be a
    linear direction and a linear projection cannot invent structure.
    """
    combined = torch.cat([clean_features[layer], backdoor_features[layer]], dim=0)
    projected = pca_project(combined, num_components=2)  # (2 * num_samples, 2)
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


def analyze_latent(
    dataset: str,
    attack_name: str,
    checkpoint: str,
    architecture: str = "vit",
    target_label: int = 0,
    samples: int = 1000,
    batch_size: int = 64,
    raw_data_dir: str = "raw_data",
    output: str = "latent_scatter.png",
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Run the whole worked example on one checkpoint, returning the per-layer table.

    Prints the table and saves a PCA scatter at the layer whose backdoor direction
    norm is largest, which is the layer the trigger is written into most strongly.
    """
    # Cluster nodes have no display, so the figure backend has to be chosen before
    # the first figure is created.
    matplotlib.use("Agg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = DATASET_REGISTRY[dataset].image_size
    attack = build_attack(
        attack_name, default_config(attack_name), image_size, target_label
    )
    clean_loader, backdoor_loader = build_paired_loaders(
        dataset, attack, raw_data_dir, batch_size, samples, seed
    )

    # load_checkpoint returns the model in eval mode, where every dropout is an
    # identity, which is the clean baseline these features want.
    model = load_checkpoint(architecture, checkpoint, device)

    clean_features = extract_layer_features(
        model, clean_loader, device, use_bfloat16=True
    )
    backdoor_features = extract_layer_features(
        model, backdoor_loader, device, use_bfloat16=True
    )

    statistics = layer_statistics(clean_features, backdoor_features)
    print_layer_report(statistics)

    peak_layer = max(statistics, key=lambda layer: statistics[layer]["direction_norm"])
    print(f"backdoor direction is strongest at layer {peak_layer}")

    save_pca_scatter(clean_features, backdoor_features, peak_layer, output)
    return statistics
