"""H32: token-level PSU decomposition for spatial trigger localization.

Projects per-token CLS features onto the backdoor direction to reveal which
tokens carry the backdoor signal. For localized attacks (BadNet, TaCT), trigger
patch tokens should dominate. For global attacks (Blend, WaNet), the signal
should be diffuse.

ViT-B/16 on 224x224 produces 196 patch tokens + 1 CLS token = 197 total.
Patch tokens map to a 14x14 spatial grid.
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
OUTPUT_PATH = "results/token_localization.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_SAMPLES = 200
TARGET_LAYER = 12
HIDDEN_DIM = 768
NUM_PATCH_TOKENS = 196
GRID_SIZE = 14


TARGETS = [
    ("cifar100", "badnet_a2o", "0_1"),
    ("cifar100", "blend", "0_1"),
    ("cifar100", "wanet", "0_1"),
    ("cifar100", "adaptive_blend", "0_1"),
    ("cifar100", "sig", "0_1"),
    ("cifar100", "lf", "0_1"),
    ("tiny", "badnet_a2o", "0_1"),
    ("tiny", "blend", "0_1"),
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


def extract_per_token_features(model, loader, device, layer=TARGET_LAYER):
    """Extract per-token features at a given layer.

    Returns [N, 197, 768] tensor (CLS token at index 0, then 196 patches).
    """
    flat_features = extract_layer_features(
        model, loader, device, use_bfloat16=True, reduction="flatten"
    )
    flat = flat_features[layer].float()
    num_tokens = flat.shape[1] // HIDDEN_DIM
    return flat.reshape(flat.shape[0], num_tokens, HIDDEN_DIM)


def per_token_direction_norm(clean_tokens, backdoor_tokens):
    """Compute direction norm per token position.

    For each of the 197 token positions, compute the mean difference between
    backdoor and clean features, then take the L2 norm.

    Returns [197] tensor of direction norms.
    """
    n_min = min(clean_tokens.shape[0], backdoor_tokens.shape[0])
    diff = backdoor_tokens[:n_min] - clean_tokens[:n_min]
    mean_diff = diff.mean(dim=0)
    return mean_diff.norm(dim=-1)


def per_token_projection(clean_tokens, backdoor_tokens, direction):
    """Project per-token difference onto a global direction.

    Returns [197] tensor of mean projection magnitudes.
    """
    n_min = min(clean_tokens.shape[0], backdoor_tokens.shape[0])
    diff = backdoor_tokens[:n_min] - clean_tokens[:n_min]
    direction_unit = direction / direction.norm()
    projections = torch.einsum("ntd,d->nt", diff, direction_unit)
    return projections.mean(dim=0)


def spatial_concentration(patch_norms):
    """How concentrated the signal is: ratio of max patch norm to mean."""
    if patch_norms.max() < 1e-8:
        return 0.0
    return float(patch_norms.max() / patch_norms.mean())


def top_k_fraction(patch_norms, k=4):
    """Fraction of total norm in the top-k patches."""
    total = patch_norms.sum()
    if total < 1e-8:
        return 0.0
    topk = patch_norms.topk(k).values.sum()
    return float(topk / total)


def analyze_checkpoint(folder, dataset_name, attack_name):
    args_path = os.path.join(CHECKPOINTS_DIR, folder, "args.json")
    if not os.path.exists(args_path):
        return None

    with open(args_path) as f:
        metadata = json.load(f)

    target_label = metadata.get("target_label", 0)

    ckpt_path = os.path.join(CHECKPOINTS_DIR, folder, "attack_result.pt")
    model = load_checkpoint("vit", ckpt_path, DEVICE)

    clean_loader, backdoor_loader = load_eval_sets(
        dataset_name, attack_name, target_label
    )

    print("    extracting per-token features...", end=" ", flush=True)
    clean_tokens = extract_per_token_features(model, clean_loader, DEVICE)
    backdoor_tokens = extract_per_token_features(model, backdoor_loader, DEVICE)
    print("done")

    token_norms = per_token_direction_norm(clean_tokens, backdoor_tokens)
    cls_norm = float(token_norms[0])
    patch_norms = token_norms[1:]

    n_min = min(clean_tokens.shape[0], backdoor_tokens.shape[0])
    cls_direction = backdoor_direction(
        clean_tokens[:n_min, 0, :].contiguous(),
        backdoor_tokens[:n_min, 0, :].contiguous(),
    )
    token_projections = per_token_projection(
        clean_tokens, backdoor_tokens, cls_direction
    )
    cls_proj = float(token_projections[0])
    patch_projections = token_projections[1:]

    grid_norms = patch_norms.reshape(GRID_SIZE, GRID_SIZE)
    grid_projs = patch_projections.reshape(GRID_SIZE, GRID_SIZE)

    conc = spatial_concentration(patch_norms)
    top4_frac = top_k_fraction(patch_norms, k=4)
    top16_frac = top_k_fraction(patch_norms, k=16)

    sorted_indices = patch_norms.argsort(descending=True)
    top_patches = []
    for idx in sorted_indices[:10]:
        row = int(idx) // GRID_SIZE
        col = int(idx) % GRID_SIZE
        top_patches.append(
            {
                "index": int(idx),
                "row": row,
                "col": col,
                "norm": float(patch_norms[idx]),
                "projection": float(patch_projections[idx]),
            }
        )

    print(
        f"    CLS norm={cls_norm:.2f}, mean patch norm={float(patch_norms.mean()):.2f}"
    )
    print(
        f"    concentration={conc:.2f}, top4_frac={top4_frac:.3f}, top16_frac={top16_frac:.3f}"
    )
    top5_str = [(p["row"], p["col"], round(p["norm"], 2)) for p in top_patches[:5]]
    print(f"    top 5 patches: {top5_str}")

    del model, clean_tokens, backdoor_tokens
    torch.cuda.empty_cache()

    return {
        "cls_direction_norm": cls_norm,
        "cls_projection": cls_proj,
        "patch_norm_mean": float(patch_norms.mean()),
        "patch_norm_std": float(patch_norms.std()),
        "patch_norm_max": float(patch_norms.max()),
        "concentration_ratio": conc,
        "top4_fraction": top4_frac,
        "top16_fraction": top16_frac,
        "top_10_patches": top_patches,
        "grid_norms": [[float(v) for v in row] for row in grid_norms],
        "grid_projections": [[float(v) for v in row] for row in grid_projs],
    }


def main():
    all_results = []

    for dataset_name, attack_name, rate_tag in TARGETS:
        folder = f"vit_{dataset_name}_{attack_name}_{rate_tag}"
        if not os.path.exists(os.path.join(CHECKPOINTS_DIR, folder, "args.json")):
            print(f"skip {folder}: no checkpoint")
            continue

        print(f"\n=== {folder} ===")
        result = analyze_checkpoint(folder, dataset_name, attack_name)
        if result is not None:
            all_results.append(
                {
                    "dataset": dataset_name,
                    "attack": attack_name,
                    "rate_tag": rate_tag,
                    "folder": folder,
                    **result,
                }
            )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\n=== Summary ===")
    print(
        f"{'attack':20s} {'dataset':8s} {'CLS norm':>10} {'patch mean':>12} {'conc ratio':>12} {'top4 frac':>10}"
    )
    for entry in all_results:
        print(
            f"{entry['attack']:20s} {entry['dataset']:8s} "
            f"{entry['cls_direction_norm']:10.2f} "
            f"{entry['patch_norm_mean']:12.2f} "
            f"{entry['concentration_ratio']:12.2f} "
            f"{entry['top4_fraction']:10.3f}"
        )


if __name__ == "__main__":
    main()
