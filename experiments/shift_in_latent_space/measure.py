"""Where do clean samples MOVE under dropout, in latent space rather than in labels?

H7 currently rests on one measurement: a histogram of which class label shifted clean
predictions land on, counted from the cached per-pass argmax. That is entirely
prediction-space. It says the label changed and where it went, and nothing about
whether the representation moved the way the neuron-bias story requires.

PSBD's stated mechanism is a claim about representations: under dropout the model
loses the clean class evidence and falls back on the strongest learned association,
the trigger-to-target path, so a clean sample should drift toward wherever the target
class lives. That is directly testable and has not been tested here.

Four measurements, all at the final block's CLS features, all on the same samples:

  toward_target      cosine between the dropout-induced displacement and the direction
                     from the sample's own class centroid to the TARGET class centroid.
                     Positive means clean samples drift toward the target class.
  toward_landed      the same, but toward the centroid of whichever class the sample
                     actually shifted to. This is the control: if the displacement is
                     simply toward wherever the prediction went, then toward_target
                     carries no extra information.
  along_backdoor     projection of the displacement onto the backdoor direction, the
                     paired clean-versus-triggered difference. This is the sharpest
                     version of the claim: does dropout push a clean sample along the
                     very direction the trigger uses?
  displacement_norm  scale, so the cosines can be read as more than angles.

A benign model probed with the same trigger is measured alongside, because a
pretrained backbone has its own fallback behaviour under heavy perturbation and that
has to be subtracted from any claim about poisoning.

Writes results/<folder>/shift_latent.json and, with --umap, a projection coloured by
where each sample landed.

Example
    PYTHONPATH=. python experiments/shift_in_latent_space/measure.py \
        --checkpoint-folder vit_cifar10_blend_0_1 vit_cifar10_badnet_a2o_0_1
"""

import argparse
import json
import os

import numpy as np
import torch
from lightning import seed_everything

from analysis.direction import backdoor_direction
from analysis.features import extract_layer_features
from attacks import build_attack, default_config
from data.splits import read_checkpoint_metadata, resolve_probe_attack
from models.positions import DROPOUT_CONFIGS, plug_dropout, unplug_dropout
from models.backbones import load_checkpoint
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, Subset

from attacks.poisoning import PoisonedTrainingSet
from experiments.backdoor_direction_layers.measure import build_paired_loaders
from data.registry import DATASET_REGISTRY
from data.loading import extract_labels, load_clean_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--placement", default="pre_residual")
    parser.add_argument("--block-range", nargs=2, type=int, default=[5, 8])
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-attack", default=None)
    parser.add_argument("--probe-target-label", type=int, default=None)
    parser.add_argument("--umap", action="store_true")
    return parser.parse_args()


def build_all_class_loader(dataset_name, attack, raw_data_dir, batch_size, count, seed):
    """A clean loader over EVERY class, for class centroids.

    The paired loaders are filtered by is_eval_poisonable, which for all_to_one drops
    the target class entirely. So the target-class centroid, the one thing this whole
    measurement is about, cannot be computed from them. This unfiltered loader exists
    only to supply centroids and is never used for the displacement itself.
    """
    spec = DATASET_REGISTRY[dataset_name]
    transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((spec.image_size, spec.image_size)),
            transforms_v2.ToTensor(),
        ]
    )
    _, test_clean = load_clean_datasets(dataset_name, transform, raw_data_dir)
    generator = torch.Generator().manual_seed(seed + 1)
    order = torch.randperm(len(test_clean), generator=generator)[:count]
    subset = Subset(test_clean, order.tolist())
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    wrapped = PoisonedTrainingSet(subset, attack, set(), normalize, spec.num_classes)
    return DataLoader(wrapped, batch_size=batch_size, shuffle=False), np.array(
        extract_labels(subset)
    )


def class_centroids(features: torch.Tensor, labels: np.ndarray, num_classes: int):
    """Mean CLS feature per true class, from the unperturbed pass."""
    centroids = {}
    for label in range(num_classes):
        mask = labels == label
        if mask.sum() > 0:
            centroids[label] = features[torch.from_numpy(mask)].mean(dim=0)
    return centroids


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def measure(folder: str, args: argparse.Namespace, device) -> dict | None:
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    attack_name, target_label = resolve_probe_attack(
        metadata, args.probe_attack, args.probe_target_label
    )
    dataset = metadata["dataset"]
    spec = DATASET_REGISTRY[dataset]
    attack = build_attack(
        attack_name, default_config(attack_name), spec.image_size, target_label
    )
    clean_loader, triggered_loader, chosen = build_paired_loaders(
        dataset, attack, args.raw_data_dir, args.batch_size, args.samples, args.seed
    )
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device)

    # True labels for the chosen subset, read straight from the wrapped dataset.
    true_labels = np.array(extract_labels(clean_loader.dataset))

    # Unperturbed clean features, and the backdoor direction from the paired sets.
    seed_everything(args.seed)
    clean = extract_layer_features(
        model, clean_loader, device, use_bfloat16=False, reduction="cls"
    )[args.layer]
    seed_everything(args.seed)
    triggered = extract_layer_features(
        model, triggered_loader, device, use_bfloat16=False, reduction="cls"
    )[args.layer]
    direction = backdoor_direction(clean, triggered)
    unit_direction = direction / direction.norm().clamp_min(1e-8)

    # Clean features again, this time with dropout plugged. Same seed, so the only
    # difference from the pass above is the perturbation.
    names = DROPOUT_CONFIGS.get(args.placement, (args.placement,))
    handles = plug_dropout(
        model,
        metadata["architecture"],
        names,
        {},
        args.rate,
        block_range=tuple(args.block_range) if args.block_range else None,
    )
    try:
        seed_everything(args.seed)
        perturbed = extract_layer_features(
            model, clean_loader, device, use_bfloat16=False, reduction="cls"
        )[args.layer]
        # Which class each sample's prediction landed on after perturbation.
        seed_everything(args.seed)
        landed = []
        with torch.inference_mode():
            for images, _ in clean_loader:
                landed.append(model(images.to(device)).argmax(dim=1).cpu())
        landed = torch.cat(landed).numpy()
    finally:
        unplug_dropout(handles)

    with torch.inference_mode():
        baseline_pred = []
        for images, _ in clean_loader:
            baseline_pred.append(model(images.to(device)).argmax(dim=1).cpu())
        baseline_pred = torch.cat(baseline_pred).numpy()

    displacement = perturbed - clean

    # Centroids come from an unfiltered clean set so the target class is present.
    centroid_loader, centroid_labels = build_all_class_loader(
        dataset, attack, args.raw_data_dir, args.batch_size, args.samples, args.seed
    )
    seed_everything(args.seed)
    centroid_features = extract_layer_features(
        model, centroid_loader, device, use_bfloat16=False, reduction="cls"
    )[args.layer]
    centroids = class_centroids(centroid_features, centroid_labels, spec.num_classes)

    shifted = landed != baseline_pred
    if shifted.sum() < 10:
        return None
    index = torch.from_numpy(np.where(shifted)[0])

    # Direction from each shifted sample's own class centroid to the target centroid.
    to_target, to_landed = [], []
    for position in index.tolist():
        own = centroids.get(int(true_labels[position]))
        target = centroids.get(int(target_label))
        land = centroids.get(int(landed[position]))
        if own is None or target is None or land is None:
            continue
        to_target.append(cosine(displacement[position], target - own).item())
        to_landed.append(cosine(displacement[position], land - own).item())

    projection = (displacement[index] @ unit_direction).mean().item()

    return {
        "folder_name": folder,
        "attack": metadata["attack"],
        "probe_attack": attack_name,
        "target_label": target_label,
        "poison_rate": metadata.get("poison_rate"),
        "placement": args.placement,
        "block_range": list(args.block_range),
        "rate": args.rate,
        "layer": args.layer,
        "n_samples": int(len(clean)),
        "n_shifted": int(shifted.sum()),
        "shift_fraction": float(shifted.mean()),
        "fraction_landed_on_target": float((landed[shifted] == target_label).mean()),
        "cos_toward_target": float(np.mean(to_target)) if to_target else None,
        "cos_toward_landed_class": float(np.mean(to_landed)) if to_landed else None,
        "projection_along_backdoor_direction": projection,
        "displacement_norm": float(displacement[index].norm(dim=1).mean()),
        "clean_feature_norm": float(clean.norm(dim=1).mean()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Latent displacement of CLEAN samples under dropout "
        f"({args.placement} blocks {args.block_range[0]}-{args.block_range[1]}, p={args.rate})\n"
    )
    print(
        f"{'checkpoint':30} {'shifted':>8} {'->y_t':>7} {'cos->target':>12} "
        f"{'cos->landed':>12} {'proj on bd dir':>15}"
    )
    print("-" * 90)
    for folder in args.checkpoint_folder:
        try:
            row = measure(folder, args, device)
        except Exception as error:
            print(f"{folder:30} FAILED {type(error).__name__}: {error}")
            continue
        if row is None:
            print(f"{folder:30} too few shifted samples")
            continue
        path = os.path.join(args.results_dir, folder, "shift_latent.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(row, handle, indent=2)
        print(
            f"{folder:30} {row['shift_fraction']:>8.3f} "
            f"{row['fraction_landed_on_target']:>7.3f} "
            f"{row['cos_toward_target']:>12.4f} {row['cos_toward_landed_class']:>12.4f} "
            f"{row['projection_along_backdoor_direction']:>15.4f}"
        )


if __name__ == "__main__":
    main()
