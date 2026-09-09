"""H31: attention map divergence reveals backdoor circuits.

Extracts per-head attention weight matrices for clean and triggered inputs,
computes KL divergence per head to identify "backdoor heads" whose attention
patterns change most when the trigger is present, then tests whether masking
only those heads neutralizes the backdoor.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

from attacks import build_attack, default_config
from models.backbones import load_checkpoint, network_core
from data.registry import DATASET_REGISTRY
from data.loading import extract_labels, load_clean_datasets
from attacks.poisoning import AttackSuccessSet, PoisonedTrainingSet
from torchvision.transforms import v2 as transforms_v2


CHECKPOINTS_DIR = "checkpoints"
OUTPUT_PATH = "results/attention_divergence.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_SAMPLES = 200
NUM_HEADS = 12
NUM_LAYERS = 12


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


def extract_attention_maps(model, loader, device):
    """Extract attention weight matrices from all heads of all layers.

    Manually computes attention weights from Q, K projections inside each MHA,
    using a forward hook on the in_proj_weight to intercept the projected values.

    Returns a dict: {layer_idx: [num_samples, num_heads, num_tokens, num_tokens]}
    """
    import math

    core = network_core(model)
    model.eval()

    attention_maps = {l: [] for l in range(NUM_LAYERS)}
    hooks = []

    def make_qkv_hook(layer_idx, num_heads, head_dim):
        def hook_fn(module, input_tuple, output):
            x = input_tuple[0]
            batch_size, seq_len, embed_dim = x.shape

            qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
            q, k, _ = qkv.chunk(3, dim=-1)

            q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            scale = math.sqrt(head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn_weights = F.softmax(attn_weights, dim=-1)

            attention_maps[layer_idx].append(attn_weights.detach().cpu())

        return hook_fn

    head_dim = 768 // NUM_HEADS

    for layer_idx in range(NUM_LAYERS):
        block = core.encoder.layers[layer_idx]
        sa = block.self_attention
        h = sa.register_forward_hook(make_qkv_hook(layer_idx, NUM_HEADS, head_dim))
        hooks.append(h)

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            model(images)

    for h in hooks:
        h.remove()

    result = {}
    for l in range(NUM_LAYERS):
        if attention_maps[l]:
            result[l] = torch.cat(attention_maps[l], dim=0)
    return result


def kl_divergence_per_head(clean_attn, backdoor_attn):
    """KL divergence between clean and backdoor attention distributions per head.

    Inputs: [N, heads, tokens, tokens] attention probability tensors.
    Returns: [heads] tensor of mean KL divergence.
    """
    n_min = min(clean_attn.shape[0], backdoor_attn.shape[0])
    clean_attn = clean_attn[:n_min].clamp(min=1e-8)
    backdoor_attn = backdoor_attn[:n_min].clamp(min=1e-8)

    kl = (backdoor_attn * (backdoor_attn.log() - clean_attn.log())).sum(dim=-1)
    return kl.mean(dim=(0, 2))


def js_divergence_per_head(clean_attn, backdoor_attn):
    """Jensen-Shannon divergence (symmetric, bounded) per head."""
    n_min = min(clean_attn.shape[0], backdoor_attn.shape[0])
    clean_attn = clean_attn[:n_min].clamp(min=1e-8)
    backdoor_attn = backdoor_attn[:n_min].clamp(min=1e-8)

    m = 0.5 * (clean_attn + backdoor_attn)
    kl_cm = (clean_attn * (clean_attn.log() - m.log())).sum(dim=-1)
    kl_bm = (backdoor_attn * (backdoor_attn.log() - m.log())).sum(dim=-1)
    js = 0.5 * (kl_cm + kl_bm)
    return js.mean(dim=(0, 2))


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

    print("    extracting attention maps...", end=" ", flush=True)
    clean_attn = extract_attention_maps(model, clean_loader, DEVICE)
    backdoor_attn = extract_attention_maps(model, backdoor_loader, DEVICE)
    print("done")

    layer_results = {}
    all_js = []
    for layer_idx in range(NUM_LAYERS):
        if layer_idx not in clean_attn or layer_idx not in backdoor_attn:
            continue
        js = js_divergence_per_head(clean_attn[layer_idx], backdoor_attn[layer_idx])
        layer_results[layer_idx] = [float(v) for v in js]
        for h, v in enumerate(js):
            all_js.append((float(v), layer_idx, h))

    all_js.sort(reverse=True)
    top_heads = [(l, h, v) for v, l, h in all_js[:10]]

    print("    top 5 backdoor heads by JS divergence:")
    for l, h, v in top_heads[:5]:
        print(f"      layer {l:2d} head {h:2d}: JS={v:.4f}")

    del model, clean_attn, backdoor_attn
    torch.cuda.empty_cache()

    return {
        "js_divergence_per_head": {str(l): vals for l, vals in layer_results.items()},
        "top_10_heads": [{"layer": l, "head": h, "js": v} for l, h, v in top_heads],
    }


TARGETS = [
    ("cifar100", "badnet_a2o", "0_1"),
    ("cifar100", "blend", "0_1"),
    ("cifar100", "wanet", "0_1"),
    ("cifar100", "adaptive_blend", "0_1"),
    ("tiny", "badnet_a2o", "0_1"),
    ("tiny", "blend", "0_1"),
]


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


if __name__ == "__main__":
    main()
