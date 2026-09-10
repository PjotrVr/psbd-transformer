"""H33: weight-space spectral signature of the backdoor.

Computes the SVD of the weight difference between a backdoored model and its
matched benign model. Tests whether the backdoor perturbation is low-rank (most
variance in the top singular value) and whether the top singular vector aligns
with the backdoor direction.

CPU-only, no GPU needed.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch

from models.backbones import load_checkpoint, network_core


CHECKPOINTS_DIR = "checkpoints"
OUTPUT_PATH = "results/weight_spectral_signature.json"

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

DEVICE = torch.device("cpu")


def weight_pairs(benign_core, backdoor_core):
    """Yield (name, benign_weight, backdoor_weight) for MLP and attention matrices."""
    benign_sd = dict(benign_core.named_parameters())
    backdoor_sd = dict(backdoor_core.named_parameters())

    for name in sorted(benign_sd.keys()):
        if name not in backdoor_sd:
            continue
        w_b = benign_sd[name].data
        w_a = backdoor_sd[name].data
        if w_b.dim() < 2:
            continue
        yield name, w_b, w_a


def spectral_analysis(w_benign, w_backdoor, top_k=5):
    """SVD of weight difference, return spectral concentration and top vectors."""
    diff = (w_backdoor - w_benign).float()
    if diff.dim() > 2:
        diff = diff.reshape(diff.shape[0], -1)

    try:
        U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
    except Exception:
        return None

    total_var = float((S**2).sum().item())
    if total_var < 1e-12:
        return None

    top1_frac = float(S[0] ** 2 / total_var)
    top5_frac = float((S[:top_k] ** 2).sum() / total_var)

    return {
        "top1_sv": float(S[0].item()),
        "top5_sv": [float(s.item()) for s in S[:top_k]],
        "top1_variance_fraction": top1_frac,
        "top5_variance_fraction": top5_frac,
        "total_variance": total_var,
        "rank": int(diff.shape[0]),
        "top_right_singular": Vh[0].clone(),
    }


def main():
    all_results = []

    for dataset_name, rate_tag in SETTINGS:
        benign_folder = f"vit_{dataset_name}_benign"
        benign_path = os.path.join(CHECKPOINTS_DIR, benign_folder, "attack_result.pt")
        if not os.path.exists(benign_path):
            print(
                f"skip {dataset_name} rate={rate_tag}: no benign model at {benign_folder}"
            )
            continue

        benign_model = load_checkpoint("vit", benign_path, DEVICE)
        benign_core = network_core(benign_model)

        print(f"\n=== {dataset_name} rate={rate_tag} ===")

        for attack_name in ATTACKS:
            folder = f"vit_{dataset_name}_{attack_name}_{rate_tag}"
            ckpt_path = os.path.join(CHECKPOINTS_DIR, folder, "attack_result.pt")
            if not os.path.exists(ckpt_path):
                continue

            backdoor_model = load_checkpoint("vit", ckpt_path, DEVICE)
            backdoor_core = network_core(backdoor_model)

            print(f"  {attack_name}:")
            layer_results = {}

            for name, w_b, w_a in weight_pairs(benign_core, backdoor_core):
                result = spectral_analysis(w_b, w_a)
                if result is None:
                    continue

                top_vec = result.pop("top_right_singular")
                layer_results[name] = result

                if result["top1_variance_fraction"] > 0.1:
                    print(
                        f"    {name:50s}: top1={result['top1_variance_fraction']:.3f} "
                        f"sv={result['top1_sv']:.3f}"
                    )

            del backdoor_model
            del backdoor_core

            high_concentration = {
                k: v
                for k, v in layer_results.items()
                if v["top1_variance_fraction"] > 0.1
            }

            all_results.append(
                {
                    "dataset": dataset_name,
                    "attack": attack_name,
                    "rate_tag": rate_tag,
                    "poison_rate": float(rate_tag.replace("_", ".")),
                    "benign_folder": benign_folder,
                    "backdoor_folder": folder,
                    "weight_matrices": layer_results,
                    "high_concentration_count": len(high_concentration),
                    "total_matrices": len(layer_results),
                }
            )

        del benign_model
        del benign_core

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\n=== Summary ===")
    for entry in all_results:
        print(
            f"{entry['dataset']:8s} {entry['attack']:20s} rate={entry['rate_tag']}: "
            f"{entry['high_concentration_count']}/{entry['total_matrices']} matrices with top1 > 10%"
        )

        head_weight = entry["weight_matrices"].get("heads.head.weight", {})
        if head_weight:
            print(
                f"  classifier head: top1={head_weight['top1_variance_fraction']:.3f} "
                f"top5={head_weight['top5_variance_fraction']:.3f}"
            )


if __name__ == "__main__":
    main()
