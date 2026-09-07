"""H29: cross-attack direction universality.

Tests whether different attacks targeting the same class produce parallel
backdoor directions. If the cosine similarity between directions from different
attacks is high (>0.8), the direction is intrinsic to the target class geometry,
not the trigger.

Tests on CIFAR-100 and Tiny at 10%, 5%, and 1% poison rates.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

from analysis.direction import backdoor_direction
from analysis.features import extract_layer_features
from attacks import build_attack, default_config
from models import load_checkpoint
from utils.config import DATASET_REGISTRY
from utils.datasets import extract_labels, load_clean_datasets
from poison import AttackSuccessSet, PoisonedTrainingSet
from torchvision.transforms import v2 as transforms_v2


CHECKPOINTS_DIR = "checkpoints"
OUTPUT_PATH = "results/direction_universality.json"

ATTACKS = [
    "badnet_a2o",
    "blend",
    "wanet",
    "lc",
    "adaptive_blend",
    "sig",
    "lf",
    "bpp",
    "tact",
    "badnet_a2a",
]

SETTINGS = [
    ("cifar100", "0_1"),
    ("cifar100", "0_05"),
    ("cifar100", "0_01"),
    ("tiny", "0_1"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_SAMPLES = 500
TARGET_LAYER = 12


def load_eval_sets(dataset_name, attack_name, target_label):
    spec = DATASET_REGISTRY[dataset_name]
    attack = build_attack(
        attack_name, default_config(attack_name), spec.image_size, target_label
    )
    base_transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((spec.image_size, spec.image_size)),
            transforms_v2.ToTensor(),
        ]
    )
    _, test_base = load_clean_datasets(dataset_name, base_transform, "raw_data")
    test_base = torch.utils.data.Subset(
        test_base, range(min(MAX_SAMPLES, len(test_base)))
    )

    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    labels = extract_labels(test_base)

    clean_set = PoisonedTrainingSet(
        test_base, attack, set(), normalize, spec.num_classes
    )
    backdoor_set = AttackSuccessSet(
        test_base, labels, attack, normalize, spec.num_classes
    )

    clean_loader = torch.utils.data.DataLoader(
        clean_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    backdoor_loader = torch.utils.data.DataLoader(
        backdoor_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    return clean_loader, backdoor_loader


def extract_direction(folder_name, dataset_name, attack_name, target_label):
    ckpt_file = os.path.join(CHECKPOINTS_DIR, folder_name, "attack_result.pt")
    args_path = os.path.join(CHECKPOINTS_DIR, folder_name, "args.json")
    if not os.path.exists(args_path):
        return None

    with open(args_path) as f:
        metadata = json.load(f)

    architecture = metadata.get("architecture", "vit")
    model = load_checkpoint(architecture, ckpt_file, DEVICE)

    clean_loader, backdoor_loader = load_eval_sets(
        dataset_name, attack_name, target_label
    )

    clean_features = extract_layer_features(
        model, clean_loader, DEVICE, use_bfloat16=True
    )
    backdoor_features = extract_layer_features(
        model, backdoor_loader, DEVICE, use_bfloat16=True
    )

    n_min = min(
        clean_features[TARGET_LAYER].shape[0],
        backdoor_features[TARGET_LAYER].shape[0],
    )
    direction = backdoor_direction(
        clean_features[TARGET_LAYER][:n_min],
        backdoor_features[TARGET_LAYER][:n_min],
    )
    norm = float(direction.norm().item())

    del model
    torch.cuda.empty_cache()
    return direction, norm


def cosine_matrix(directions):
    """Pairwise cosine similarity matrix for a dict of {attack: direction}."""
    attacks = sorted(directions.keys())
    n = len(attacks)
    matrix = {}
    for i in range(n):
        for j in range(n):
            a, b = attacks[i], attacks[j]
            cos = float(
                F.cosine_similarity(
                    directions[a].unsqueeze(0), directions[b].unsqueeze(0)
                ).item()
            )
            matrix[f"{a}_vs_{b}"] = cos
    return attacks, matrix


def main():
    all_results = []

    for dataset_name, rate_tag in SETTINGS:
        print(f"\n=== {dataset_name} rate={rate_tag} ===")
        directions = {}
        norms = {}

        for attack_name in ATTACKS:
            folder = f"vit_{dataset_name}_{attack_name}_{rate_tag}"
            if not os.path.exists(os.path.join(CHECKPOINTS_DIR, folder, "args.json")):
                print(f"  skip {folder}: no checkpoint")
                continue

            print(f"  extracting {attack_name}...", end=" ", flush=True)
            result = extract_direction(
                folder, dataset_name, attack_name, target_label=0
            )
            if result is None:
                print("failed")
                continue
            direction, norm = result
            directions[attack_name] = direction
            norms[attack_name] = norm
            print(f"norm={norm:.2f}")

        if len(directions) < 2:
            print("  not enough directions for comparison")
            continue

        attack_order, matrix = cosine_matrix(directions)

        print(f"\n  Pairwise cosine similarity (layer {TARGET_LAYER}):")
        header = "              " + "".join(f"{a[:8]:>10}" for a in attack_order)
        print(header)
        for a in attack_order:
            row = f"  {a:12s}"
            for b in attack_order:
                cos = matrix[f"{a}_vs_{b}"]
                row += f"  {cos:8.3f}"
            print(row)

        off_diagonal = []
        for a in attack_order:
            for b in attack_order:
                if a != b:
                    off_diagonal.append(matrix[f"{a}_vs_{b}"])

        import numpy as np

        print(
            f"\n  Off-diagonal: mean={np.mean(off_diagonal):.3f}, "
            f"min={np.min(off_diagonal):.3f}, max={np.max(off_diagonal):.3f}"
        )

        all_results.append(
            {
                "dataset": dataset_name,
                "rate_tag": rate_tag,
                "poison_rate": float(rate_tag.replace("_", ".")),
                "layer": TARGET_LAYER,
                "attacks": attack_order,
                "direction_norms": norms,
                "cosine_matrix": matrix,
            }
        )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
