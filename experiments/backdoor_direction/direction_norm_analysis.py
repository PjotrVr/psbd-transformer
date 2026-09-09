"""H28 prediction 1: backdoor direction norm scales with poison rate.

Extracts CLS-token features at all 13 layers (embedding + 12 blocks) for clean
and backdoor test sets, computes the backdoor direction (mean paired difference),
and reports its L2 norm at each layer for each poison rate.

Outputs a JSON file for later analysis and a summary to stdout.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch

from analysis.direction import backdoor_direction
from analysis.features import extract_layer_features
from attacks import build_attack, default_config
from models.backbones import load_checkpoint
from data.registry import DATASET_REGISTRY
from data.loading import extract_labels, load_clean_datasets
from attacks.poisoning import AttackSuccessSet, PoisonedTrainingSet
from torchvision.transforms import v2 as transforms_v2


CHECKPOINTS_DIR = "checkpoints"
OUTPUT_PATH = "results/direction_norm_analysis.json"

TARGETS = [
    ("cifar100", "badnet_a2o", ["0_005", "0_01", "0_05", "0_1"]),
    ("cifar100", "blend", ["0_005", "0_01", "0_05", "0_1"]),
    ("tiny", "badnet_a2o", ["0_005", "0_01", "0_05", "0_1"]),
    ("tiny", "blend", ["0_005", "0_01", "0_05", "0_1"]),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_SAMPLES = 500


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


def analyze_checkpoint(folder_name, dataset_name, attack_name):
    ckpt_dir = os.path.join(CHECKPOINTS_DIR, folder_name)
    args_path = os.path.join(ckpt_dir, "args.json")
    if not os.path.exists(args_path):
        print(f"  skip {folder_name}: no args.json")
        return None

    with open(args_path) as f:
        metadata = json.load(f)

    target_label = metadata.get("target_label", 0)
    architecture = metadata.get("architecture", "vit")

    ckpt_file = os.path.join(ckpt_dir, "attack_result.pt")
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

    norms = {}
    for layer_idx in sorted(clean_features.keys()):
        n_min = min(
            clean_features[layer_idx].shape[0],
            backdoor_features[layer_idx].shape[0],
        )
        direction = backdoor_direction(
            clean_features[layer_idx][:n_min],
            backdoor_features[layer_idx][:n_min],
        )
        norms[layer_idx] = float(direction.norm().item())

    del model
    torch.cuda.empty_cache()
    return norms


def main():
    all_results = []

    for dataset_name, attack_name, rate_tags in TARGETS:
        print(f"\n=== {dataset_name} / {attack_name} ===")
        for rate_tag in rate_tags:
            folder = f"vit_{dataset_name}_{attack_name}_{rate_tag}"
            print(f"  {folder}")
            norms = analyze_checkpoint(folder, dataset_name, attack_name)
            if norms is not None:
                rate_float = float(rate_tag.replace("_", "."))
                all_results.append(
                    {
                        "dataset": dataset_name,
                        "attack": attack_name,
                        "poison_rate": rate_float,
                        "folder": folder,
                        "direction_norms": {str(k): v for k, v in norms.items()},
                    }
                )
                last_layer = max(norms.keys())
                print(
                    f"    layer 0: {norms[0]:.4f}  layer {last_layer}: {norms[last_layer]:.4f}"
                )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\n=== Summary (last layer norm by poison rate) ===")
    for dataset_name, attack_name, rate_tags in TARGETS:
        print(f"\n{dataset_name} / {attack_name}:")
        for rate_tag in rate_tags:
            folder = f"vit_{dataset_name}_{attack_name}_{rate_tag}"
            matching = [r for r in all_results if r["folder"] == folder]
            if matching:
                norms = matching[0]["direction_norms"]
                last_key = str(max(int(k) for k in norms.keys()))
                rate_float = float(rate_tag.replace("_", "."))
                print(f"  rate={rate_float:.3f}  norm={norms[last_key]:.4f}")


if __name__ == "__main__":
    main()
