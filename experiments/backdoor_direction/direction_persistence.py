"""H30: cross-layer direction persistence in the residual stream.

Measures cosine similarity of the backdoor direction between consecutive layers
and between each layer and the final layer. Tests whether the direction is
carried through the residual stream with minimal rotation (high cosine = the
direction is persistent) or recomputed at each layer (low cosine = each block
creates a new direction).

Reuses the same 16 checkpoints as direction_norm_analysis.py.
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
OUTPUT_PATH = "results/direction_persistence.json"

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


def cosine(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


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

    layers = sorted(clean_features.keys())
    directions = {}
    norms = {}
    for layer_idx in layers:
        n_min = min(
            clean_features[layer_idx].shape[0],
            backdoor_features[layer_idx].shape[0],
        )
        d = backdoor_direction(
            clean_features[layer_idx][:n_min],
            backdoor_features[layer_idx][:n_min],
        )
        directions[layer_idx] = d
        norms[layer_idx] = float(d.norm().item())

    consecutive_cosine = []
    for i in range(len(layers) - 1):
        l_a, l_b = layers[i], layers[i + 1]
        if norms[l_a] < 1e-6 or norms[l_b] < 1e-6:
            consecutive_cosine.append(None)
        else:
            consecutive_cosine.append(cosine(directions[l_a], directions[l_b]))

    last_layer = layers[-1]
    alignment_to_final = []
    for layer_idx in layers[:-1]:
        if norms[layer_idx] < 1e-6:
            alignment_to_final.append(None)
        else:
            alignment_to_final.append(
                cosine(directions[layer_idx], directions[last_layer])
            )

    del model
    torch.cuda.empty_cache()

    return {
        "consecutive_cosine": consecutive_cosine,
        "alignment_to_final": alignment_to_final,
        "direction_norms": {str(k): v for k, v in norms.items()},
    }


def main():
    all_results = []

    for dataset_name, attack_name, rate_tags in TARGETS:
        print(f"\n=== {dataset_name} / {attack_name} ===")
        for rate_tag in rate_tags:
            folder = f"vit_{dataset_name}_{attack_name}_{rate_tag}"
            print(f"  {folder}")
            result = analyze_checkpoint(folder, dataset_name, attack_name)
            if result is None:
                continue

            rate_float = float(rate_tag.replace("_", "."))
            entry = {
                "dataset": dataset_name,
                "attack": attack_name,
                "poison_rate": rate_float,
                "folder": folder,
                **result,
            }
            all_results.append(entry)

            cc = result["consecutive_cosine"]
            af = result["alignment_to_final"]
            cc_str = [f"{v:.3f}" if v is not None else "n/a" for v in cc]
            print(f"    consecutive: {' '.join(cc_str[-5:])}")
            af_str = [f"{v:.3f}" if v is not None else "n/a" for v in af]
            print(f"    align-final: {' '.join(af_str[-5:])}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\n=== Summary: consecutive cosine by layer pair ===")
    print(f"{'pair':>8}", end="")
    for i in range(12):
        print(f" {i}->{i + 1:>2}", end="")
    print()
    for entry in all_results:
        label = f"{entry['dataset'][:4]}_{entry['attack'][:5]}_{entry['poison_rate']}"
        print(f"{label:>20}", end="")
        for v in entry["consecutive_cosine"]:
            if v is None:
                print("   n/a", end="")
            else:
                print(f"  {v:.3f}", end="")
        print()


if __name__ == "__main__":
    main()
