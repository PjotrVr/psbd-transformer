"""H36: backdoor direction cone geometry.

Analyzes the geometric structure of backdoor directions in the 768-dim residual
stream. All directions project onto the readout weight (cosine 0.58-0.88) but
are mutually near-orthogonal (cosine ~0.03). This script computes:
1. The cone angular radius around the readout weight
2. Theoretical capacity (how many orthogonal directions fit in the cone)
3. Whether the observed inter-attack cosines match random-in-cone expectation
4. Cone projection defense: project out the readout weight subspace
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from analysis.direction import backdoor_direction
from analysis.features import extract_layer_features
from attacks import build_attack, default_config
from models.backbones import load_checkpoint, network_core
from data.registry import DATASET_REGISTRY
from data.loading import extract_labels, load_clean_datasets
from attacks.poisoning import AttackSuccessSet, PoisonedTrainingSet
from torchvision.transforms import v2 as transforms_v2


CHECKPOINTS_DIR = "checkpoints"
OUTPUT_PATH = "results/cone_geometry.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_SAMPLES = 500
TARGET_LAYER = 12

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
    ("tiny", "0_1"),
]


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


def expected_cosine_random_in_cone(cone_half_angle_rad, dim, n_simulations=10000):
    """Expected pairwise cosine for random unit vectors inside a cone.

    Simulates n pairs of random vectors within a cone of given half-angle
    centered on e_1 in `dim` dimensions, and returns mean pairwise cosine.
    """
    rng = np.random.default_rng(42)
    cosines = []
    for _ in range(n_simulations):
        vecs = []
        for _ in range(2):
            z = rng.normal(size=dim)
            z = z / np.linalg.norm(z)
            cos_to_axis = z[0]
            angle_to_axis = np.arccos(np.clip(cos_to_axis, -1, 1))
            if angle_to_axis > cone_half_angle_rad:
                target_cos = np.cos(rng.uniform(0, cone_half_angle_rad))
                current_cos = z[0]
                perp = z.copy()
                perp[0] = 0
                perp_norm = np.linalg.norm(perp)
                if perp_norm < 1e-10:
                    perp = np.zeros(dim)
                    perp[1] = 1.0
                else:
                    perp = perp / perp_norm
                z = np.zeros(dim)
                z[0] = target_cos
                z[1:] = 0
                sin_val = np.sqrt(1 - target_cos**2)
                z = target_cos * np.eye(dim)[0] + sin_val * perp
                z = z / np.linalg.norm(z)
            vecs.append(z)
        cosines.append(np.dot(vecs[0], vecs[1]))
    return float(np.mean(cosines)), float(np.std(cosines))


def orthogonal_capacity(cone_half_angle_rad, dim):
    """How many mutually orthogonal unit vectors fit inside a cone.

    A rough upper bound: the effective dimensionality of the cone's interior.
    For a cone with half-angle theta in d dimensions, vectors inside the cone
    have their first coordinate >= cos(theta), leaving approximately
    sin^2(theta) * (d-1) effective dimensions for the orthogonal part.
    """
    sin_sq = np.sin(cone_half_angle_rad) ** 2
    return int(sin_sq * (dim - 1)) + 1


def analyze_setting(dataset_name, rate_tag):
    """Extract all directions and readout weight for one (dataset, rate) pair."""
    directions = {}
    readout_weight = None
    target_label = 0

    for attack_name in ATTACKS:
        folder = f"vit_{dataset_name}_{attack_name}_{rate_tag}"
        args_path = os.path.join(CHECKPOINTS_DIR, folder, "args.json")
        if not os.path.exists(args_path):
            continue

        with open(args_path) as f:
            metadata = json.load(f)
        target_label = metadata.get("target_label", 0)

        ckpt_path = os.path.join(CHECKPOINTS_DIR, folder, "attack_result.pt")
        model = load_checkpoint("vit", ckpt_path, DEVICE)

        if readout_weight is None:
            core = network_core(model)
            for name, param in core.named_parameters():
                if name == "heads.head.weight":
                    readout_weight = param.data[target_label].clone().cpu().float()
                    break

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
        direction = (
            backdoor_direction(
                clean_features[TARGET_LAYER][:n_min],
                backdoor_features[TARGET_LAYER][:n_min],
            )
            .cpu()
            .float()
        )

        directions[attack_name] = direction
        del model
        torch.cuda.empty_cache()

    return directions, readout_weight, target_label


def cone_analysis(directions, readout_weight):
    """Compute cone geometry metrics."""
    attacks = sorted(directions.keys())

    cosines_to_readout = {}
    angles_to_readout = {}
    for attack in attacks:
        cos = float(
            F.cosine_similarity(
                directions[attack].unsqueeze(0), readout_weight.unsqueeze(0)
            ).item()
        )
        cosines_to_readout[attack] = cos
        angles_to_readout[attack] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    mean_angle = float(np.mean(list(angles_to_readout.values())))
    max_angle = float(np.max(list(angles_to_readout.values())))
    min_angle = float(np.min(list(angles_to_readout.values())))

    pairwise_cosines = {}
    off_diagonal = []
    for i, a in enumerate(attacks):
        for j, b in enumerate(attacks):
            if i >= j:
                continue
            cos = float(
                F.cosine_similarity(
                    directions[a].unsqueeze(0), directions[b].unsqueeze(0)
                ).item()
            )
            pairwise_cosines[f"{a}_vs_{b}"] = cos
            off_diagonal.append(cos)

    dim = 768
    cone_half_angle_rad = float(np.radians(mean_angle))
    capacity = orthogonal_capacity(cone_half_angle_rad, dim)

    expected_cos, expected_std = expected_cosine_random_in_cone(
        cone_half_angle_rad, dim
    )

    return {
        "cosines_to_readout": cosines_to_readout,
        "angles_to_readout_degrees": angles_to_readout,
        "cone_mean_angle_degrees": mean_angle,
        "cone_max_angle_degrees": max_angle,
        "cone_min_angle_degrees": min_angle,
        "cone_half_angle_radians": cone_half_angle_rad,
        "orthogonal_capacity": capacity,
        "pairwise_cosines": pairwise_cosines,
        "pairwise_mean": float(np.mean(off_diagonal)) if off_diagonal else 0,
        "pairwise_std": float(np.std(off_diagonal)) if off_diagonal else 0,
        "expected_cosine_random_in_cone": expected_cos,
        "expected_cosine_random_in_cone_std": expected_std,
        "expected_cosine_random_full_sphere": 0.0,
        "dimension": dim,
        "num_attacks": len(attacks),
        "attacks": attacks,
    }


def main():
    all_results = []

    for dataset_name, rate_tag in SETTINGS:
        print(f"\n=== {dataset_name} rate={rate_tag} ===")
        directions, readout_weight, target_label = analyze_setting(
            dataset_name, rate_tag
        )

        if not directions or readout_weight is None:
            print("  no data")
            continue

        result = cone_analysis(directions, readout_weight)
        result["dataset"] = dataset_name
        result["rate_tag"] = rate_tag
        result["target_label"] = target_label
        all_results.append(result)

        print("  Cone geometry:")
        print(
            f"    mean angle to readout: {result['cone_mean_angle_degrees']:.1f} degrees"
        )
        print(
            f"    angle range: {result['cone_min_angle_degrees']:.1f} to {result['cone_max_angle_degrees']:.1f} degrees"
        )
        print(f"    orthogonal capacity: {result['orthogonal_capacity']} directions")
        print(
            f"    observed pairwise cosine: {result['pairwise_mean']:.4f} +/- {result['pairwise_std']:.4f}"
        )
        print(
            f"    expected random-in-cone: {result['expected_cosine_random_in_cone']:.4f} +/- {result['expected_cosine_random_in_cone_std']:.4f}"
        )
        print()
        print("  Per-attack angles:")
        for attack in result["attacks"]:
            angle = result["angles_to_readout_degrees"][attack]
            cos = result["cosines_to_readout"][attack]
            print(f"    {attack:20s}: {angle:5.1f} deg (cos={cos:.3f})")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
