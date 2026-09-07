"""H40: per-sample attention entropy as detection feature.

Computes attention entropy in the 3 backdoor heads (L5H0, L5H10, L6H3) from
H31 for each sample. Tests whether backdoor samples have measurably different
entropy (more focused attention) than clean samples, producing a per-sample
detection feature.
"""

import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

from attacks import build_attack, default_config
from models import load_checkpoint, network_core
from utils.config import DATASET_REGISTRY
from utils.datasets import extract_labels, load_clean_datasets
from poison import AttackSuccessSet, PoisonedTrainingSet
from torchvision.transforms import v2 as transforms_v2


CHECKPOINTS_DIR = "checkpoints"
OUTPUT_PATH = "results/attention_entropy.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_SAMPLES = 500
NUM_HEADS = 12
NUM_LAYERS = 12
HEAD_DIM = 768 // 12

BACKDOOR_HEADS = [(5, 0), (5, 10), (6, 3)]

TARGETS = [
    ("cifar100", "badnet_a2o", "0_1"),
    ("cifar100", "blend", "0_1"),
    ("cifar100", "wanet", "0_1"),
    ("cifar100", "adaptive_blend", "0_1"),
    ("cifar100", "lc", "0_1"),
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


def extract_attention_entropy(model, loader, device):
    """Extract per-sample attention entropy for the backdoor heads.

    Returns dict: {(layer, head): [N] tensor of entropy values}
    """
    core = network_core(model)
    model.eval()

    head_entropies = {(l, h): [] for l, h in BACKDOOR_HEADS}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input_tuple, output):
            x = input_tuple[0]
            batch_size, seq_len, embed_dim = x.shape

            qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
            q, k, _ = qkv.chunk(3, dim=-1)

            q = q.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            k = k.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)

            scale = math.sqrt(HEAD_DIM)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn_weights = F.softmax(attn_weights, dim=-1)

            for l, h in BACKDOOR_HEADS:
                if l == layer_idx:
                    w = attn_weights[:, h].clamp(min=1e-8)
                    entropy = -(w * w.log()).sum(dim=-1).mean(dim=-1)
                    head_entropies[(l, h)].append(entropy.detach().cpu())

        return hook_fn

    target_layers = set(l for l, h in BACKDOOR_HEADS)
    for layer_idx in target_layers:
        block = core.encoder.layers[layer_idx]
        sa = block.self_attention
        h = sa.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            model(images)

    for h in hooks:
        h.remove()

    result = {}
    for key in head_entropies:
        if head_entropies[key]:
            result[key] = torch.cat(head_entropies[key], dim=0)
    return result


def compute_detection_auroc(clean_entropies, backdoor_entropies):
    """AUROC using attention entropy as detection score.

    Convention: lower entropy = more focused = more likely backdoored.
    So we negate entropy to get a score where higher = more suspicious.
    """
    n_clean = clean_entropies.shape[0]
    n_backdoor = backdoor_entropies.shape[0]

    scores = torch.cat([-clean_entropies, -backdoor_entropies]).numpy()
    labels = np.concatenate([np.zeros(n_clean), np.ones(n_backdoor)])

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5

    return float(auroc)


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

    print("    extracting attention entropy...", end=" ", flush=True)
    clean_entropies = extract_attention_entropy(model, clean_loader, DEVICE)
    backdoor_entropies = extract_attention_entropy(model, backdoor_loader, DEVICE)
    print("done")

    per_head_results = {}
    all_clean_combined = []
    all_backdoor_combined = []

    for layer, head in BACKDOOR_HEADS:
        key = (layer, head)
        if key not in clean_entropies or key not in backdoor_entropies:
            continue

        c = clean_entropies[key]
        b = backdoor_entropies[key]

        auroc = compute_detection_auroc(c, b)

        per_head_results[f"L{layer}H{head}"] = {
            "clean_entropy_mean": float(c.mean()),
            "clean_entropy_std": float(c.std()),
            "backdoor_entropy_mean": float(b.mean()),
            "backdoor_entropy_std": float(b.std()),
            "auroc": auroc,
            "entropy_gap": float(c.mean() - b.mean()),
        }

        all_clean_combined.append(c)
        all_backdoor_combined.append(b)

        print(
            f"      L{layer}H{head}: clean={float(c.mean()):.3f}+/-{float(c.std()):.3f}, "
            f"backdoor={float(b.mean()):.3f}+/-{float(b.std()):.3f}, AUROC={auroc:.3f}"
        )

    if all_clean_combined:
        combined_clean = torch.stack(all_clean_combined, dim=1).mean(dim=1)
        combined_backdoor = torch.stack(all_backdoor_combined, dim=1).mean(dim=1)
        combined_auroc = compute_detection_auroc(combined_clean, combined_backdoor)
        per_head_results["combined_3heads"] = {
            "auroc": combined_auroc,
            "clean_entropy_mean": float(combined_clean.mean()),
            "backdoor_entropy_mean": float(combined_backdoor.mean()),
        }
        print(f"      combined 3 heads: AUROC={combined_auroc:.3f}")

    del model
    torch.cuda.empty_cache()

    return per_head_results


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
                    "per_head": result,
                }
            )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\n=== Summary ===")
    print(f"{'folder':40s} {'L5H0':>8} {'L5H10':>8} {'L6H3':>8} {'combined':>10}")
    for entry in all_results:
        h = entry["per_head"]
        l5h0 = h.get("L5H0", {}).get("auroc", 0)
        l5h10 = h.get("L5H10", {}).get("auroc", 0)
        l6h3 = h.get("L6H3", {}).get("auroc", 0)
        comb = h.get("combined_3heads", {}).get("auroc", 0)
        print(
            f"{entry['folder']:40s} {l5h0:8.3f} {l5h10:8.3f} {l6h3:8.3f} {comb:10.3f}"
        )


if __name__ == "__main__":
    main()
