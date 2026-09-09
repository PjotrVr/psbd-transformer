"""H35: targeted head perturbation as PSBD operator.

Tests whether masking the 3 backdoor heads from H31 (L5H0, L5H10, L6H3) as a
deterministic PSBD operator outperforms random head masking (H22: 0.539 AUROC).

Uses the existing PSBD evaluation infrastructure: PSU fractional = (base - dropped) / base.
Detection: low PSU = poisoned (one-sided only).
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np

from attacks import build_attack, default_config
from defences.inference import forward_probs
from defences.operators import masked_attention_forward
from models.backbones import load_checkpoint, network_core
from data.registry import DATASET_REGISTRY
from data.loading import extract_labels, limit_dataset, load_clean_datasets
from attacks.poisoning import AttackSuccessSet, PoisonedTrainingSet
from torchvision.transforms import v2 as transforms_v2


CHECKPOINTS_DIR = "checkpoints"
OUTPUT_PATH = "results/targeted_head_psbd.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_SAMPLES = 500

BACKDOOR_HEADS = [(5, 0), (5, 10), (6, 3)]
BADNET_EXTRA_HEADS = [(9, 7), (10, 9)]

TARGETS = [
    ("cifar100", "badnet_a2o", "0_1"),
    ("cifar100", "blend", "0_1"),
    ("cifar100", "wanet", "0_1"),
    ("cifar100", "lc", "0_1"),
    ("cifar100", "adaptive_blend", "0_1"),
    ("cifar100", "badnet_a2o", "0_05"),
    ("cifar100", "blend", "0_05"),
    ("tiny", "badnet_a2o", "0_1"),
    ("tiny", "blend", "0_1"),
]


class MultiFixedHeadMask(nn.Module):
    """Zero multiple specific attention heads deterministically."""

    def __init__(self, head_indices):
        super().__init__()
        self.head_indices = list(head_indices)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f"expected (batch, heads, tokens, dim), got {tuple(x.shape)}"
            )
        out = x.clone()
        for h in self.head_indices:
            if 0 <= h < x.shape[1]:
                out[:, h] = 0.0
        return out


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
    # A random subset, never a first-N slice. The ImageFolder-backed loaders list
    # samples sorted by class, so on Tiny a first-500 slice covers 10 of 200 classes
    # and on GTSRB it is similarly degenerate. Measured, not assumed.
    test_base = limit_dataset(test_base, MAX_SAMPLES, seed=0)

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


def plug_targeted_heads(model, head_set):
    """Install targeted head masking at specified (block, head) pairs.

    Groups heads by block and installs one MultiFixedHeadMask per block.
    Uses masked_attention_forward to intercept the attention computation.
    """
    core = network_core(model)
    hooks = []
    originals = []

    heads_by_block = {}
    for block_idx, head_idx in head_set:
        heads_by_block.setdefault(block_idx, []).append(head_idx)

    for block_idx, head_indices in heads_by_block.items():
        block = core.encoder.layers[block_idx]
        sa = block.self_attention
        mask = MultiFixedHeadMask(head_indices).to(DEVICE)

        original_forward = sa.forward

        def make_patched(orig_sa, m):
            def patched(query, key, value, **kwargs):
                return masked_attention_forward(orig_sa, m, query, key, value, **kwargs)

            return patched

        sa.forward = make_patched(sa, mask)
        originals.append((sa, original_forward))

    return originals


def unplug_targeted_heads(originals):
    for sa, original_forward in originals:
        sa.forward = original_forward


def compute_psu(model, loader, device, use_bfloat16=True):
    """Compute PSU fractional = (base_prob - perturbed_prob) / base_prob."""
    model.eval()

    base_probs_list = []
    with torch.inference_mode():
        for images, _ in loader:
            probs = forward_probs(model, images, device, use_bfloat16)
            max_probs, labels = probs.max(dim=1)
            base_probs_list.append((max_probs.cpu(), labels.cpu()))

    base_probs = torch.cat([p for p, _ in base_probs_list])
    base_labels = torch.cat([l for _, l in base_probs_list])
    return base_probs, base_labels


def compute_perturbed_psu(model, loader, base_labels, device, use_bfloat16=True):
    """Compute perturbed probabilities for the base-predicted class."""
    model.eval()
    perturbed_probs_list = []
    offset = 0

    with torch.inference_mode():
        for images, _ in loader:
            probs = forward_probs(model, images, device, use_bfloat16)
            batch_size = probs.shape[0]
            batch_labels = base_labels[offset : offset + batch_size]
            perturbed = probs[range(batch_size), batch_labels]
            perturbed_probs_list.append(perturbed.cpu())
            offset += batch_size

    return torch.cat(perturbed_probs_list)


def evaluate_targeted_heads(model, dataset_name, attack_name, target_label, head_set):
    """Full PSBD evaluation with targeted head masking."""
    clean_loader, backdoor_loader = load_eval_sets(
        dataset_name, attack_name, target_label
    )

    clean_base, clean_labels = compute_psu(model, clean_loader, DEVICE)
    bd_base, bd_labels = compute_psu(model, backdoor_loader, DEVICE)

    originals = plug_targeted_heads(model, head_set)

    clean_perturbed = compute_perturbed_psu(model, clean_loader, clean_labels, DEVICE)
    bd_perturbed = compute_perturbed_psu(model, backdoor_loader, bd_labels, DEVICE)

    unplug_targeted_heads(originals)

    clean_psu = ((clean_base - clean_perturbed) / clean_base.clamp(min=1e-8)).numpy()
    bd_psu = ((bd_base - bd_perturbed) / bd_base.clamp(min=1e-8)).numpy()

    # AttackSuccessSet drops the ineligible rows, so the 2 loaders serve different
    # populations and comparing them as served measures which classes were dropped as
    # well as the defence. indices are positions into the same base, so they pair the
    # clean rows back onto their own counterparts.
    clean_psu = clean_psu[np.asarray(backdoor_loader.dataset.indices, dtype=int)]

    scores = np.concatenate([clean_psu, bd_psu])
    labels = np.concatenate([np.zeros(len(clean_psu)), np.ones(len(bd_psu))])

    auroc = roc_auc_score(labels, -scores)

    return {
        "auroc": float(auroc),
        "clean_psu_mean": float(clean_psu.mean()),
        "clean_psu_std": float(clean_psu.std()),
        "backdoor_psu_mean": float(bd_psu.mean()),
        "backdoor_psu_std": float(bd_psu.std()),
    }


def main():
    all_results = []

    for dataset_name, attack_name, rate_tag in TARGETS:
        folder = f"vit_{dataset_name}_{attack_name}_{rate_tag}"
        args_path = os.path.join(CHECKPOINTS_DIR, folder, "args.json")
        if not os.path.exists(args_path):
            print(f"skip {folder}: no checkpoint")
            continue

        with open(args_path) as f:
            metadata = json.load(f)
        target_label = metadata.get("target_label", 0)

        print(f"\n=== {folder} ===")
        ckpt_path = os.path.join(CHECKPOINTS_DIR, folder, "attack_result.pt")

        entry = {
            "dataset": dataset_name,
            "attack": attack_name,
            "rate_tag": rate_tag,
            "folder": folder,
        }

        model = load_checkpoint("vit", ckpt_path, DEVICE)
        result_3head = evaluate_targeted_heads(
            model, dataset_name, attack_name, target_label, BACKDOOR_HEADS
        )
        entry["3_backdoor_heads"] = result_3head
        print(f"  3 backdoor heads: AUROC={result_3head['auroc']:.3f}")
        del model
        torch.cuda.empty_cache()

        model = load_checkpoint("vit", ckpt_path, DEVICE)
        all_heads = BACKDOOR_HEADS + BADNET_EXTRA_HEADS
        result_5head = evaluate_targeted_heads(
            model, dataset_name, attack_name, target_label, all_heads
        )
        entry["5_heads_with_badnet_extra"] = result_5head
        print(f"  5 heads (+badnet): AUROC={result_5head['auroc']:.3f}")
        del model
        torch.cuda.empty_cache()

        all_results.append(entry)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\n=== Summary ===")
    print(f"{'folder':40s} {'3-head':>8} {'5-head':>8}")
    for entry in all_results:
        a3 = entry["3_backdoor_heads"]["auroc"]
        a5 = entry["5_heads_with_badnet_extra"]["auroc"]
        print(f"{entry['folder']:40s} {a3:8.3f} {a5:8.3f}")


if __name__ == "__main__":
    main()
