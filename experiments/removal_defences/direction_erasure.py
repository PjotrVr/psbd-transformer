"""H34: backdoor erasure via direction orthogonalization.

Tests whether the backdoor can be removed by orthogonalizing the MLP output
weights against the backdoor direction. Two modes:
  Stage 1 (known direction): uses the true direction from paired features.
  Stage 2 (blind direction): uses the target class's readout weight as a proxy.

Reports baseline ASR/CA, erased ASR/CA, and random-direction control.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch

from analysis.direction import backdoor_direction, orthogonalize_weight
from analysis.features import extract_layer_features
from attacks import build_attack, default_config
from evaluate import evaluate_attack
from models import load_checkpoint, network_core
from utils.config import DATASET_REGISTRY
from utils.datasets import extract_labels, load_clean_datasets
from poison import AttackSuccessSet, PoisonedTrainingSet
from torchvision.transforms import v2 as transforms_v2


CHECKPOINTS_DIR = "checkpoints"
OUTPUT_PATH = "results/direction_erasure.json"

ATTACKS = [
    "badnet_a2o",
    "blend",
    "wanet",
    "lc",
    "adaptive_blend",
]

SETTINGS = [
    ("cifar100", "0_1"),
    ("cifar100", "0_05"),
    ("tiny", "0_1"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_SAMPLES = 500
ERASE_LAYERS = [10, 11]


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


def eval_model(model, dataset_name, attack_name, target_label):
    config = default_config(attack_name)
    metrics = evaluate_attack(
        model, dataset_name, attack_name, config, target_label, DEVICE
    )
    return metrics["asr"], metrics["clean_accuracy"]


def erase_direction(model, direction, layers):
    """Remove the direction from MLP output weights at specified layers."""
    core = network_core(model)
    direction = direction.to(next(core.parameters()).device)
    for layer_idx in layers:
        weight_name = f"encoder.layers.encoder_layer_{layer_idx}.mlp.3.weight"
        for name, param in core.named_parameters():
            if name == weight_name:
                with torch.no_grad():
                    param.copy_(orthogonalize_weight(param.data, direction))
                break

        attn_name = (
            f"encoder.layers.encoder_layer_{layer_idx}.self_attention.out_proj.weight"
        )
        for name, param in core.named_parameters():
            if name == attn_name:
                with torch.no_grad():
                    param.copy_(orthogonalize_weight(param.data, direction))
                break


def analyze_checkpoint(folder, dataset_name, attack_name):
    ckpt_path = os.path.join(CHECKPOINTS_DIR, folder, "attack_result.pt")
    args_path = os.path.join(CHECKPOINTS_DIR, folder, "args.json")
    if not os.path.exists(args_path):
        return None

    with open(args_path) as f:
        metadata = json.load(f)

    target_label = metadata.get("target_label", 0)

    model = load_checkpoint("vit", ckpt_path, DEVICE)

    asr_base, ca_base = eval_model(model, dataset_name, attack_name, target_label)
    print(f"    baseline: ASR={asr_base:.3f} CA={ca_base:.3f}")

    if asr_base < 0.5:
        print(f"    skip: attack did not implant (ASR={asr_base:.3f})")
        del model
        torch.cuda.empty_cache()
        return {
            "baseline": {"asr": asr_base, "ca": ca_base},
            "skip_reason": "attack_did_not_implant",
        }

    clean_loader, backdoor_loader = load_eval_sets(
        dataset_name, attack_name, target_label
    )

    layer_for_direction = max(ERASE_LAYERS)
    clean_features = extract_layer_features(
        model, clean_loader, DEVICE, use_bfloat16=True
    )
    backdoor_features = extract_layer_features(
        model, backdoor_loader, DEVICE, use_bfloat16=True
    )
    n_min = min(
        clean_features[layer_for_direction].shape[0],
        backdoor_features[layer_for_direction].shape[0],
    )
    true_direction = backdoor_direction(
        clean_features[layer_for_direction][:n_min],
        backdoor_features[layer_for_direction][:n_min],
    )
    true_norm = float(true_direction.norm().item())

    del clean_features, backdoor_features

    core = network_core(model)
    readout_weight = None
    for name, param in core.named_parameters():
        if name == "heads.head.weight":
            readout_weight = param.data[target_label].clone()
            break

    true_direction = true_direction.to(DEVICE)
    cosine_readout = float(
        torch.nn.functional.cosine_similarity(
            true_direction.unsqueeze(0), readout_weight.unsqueeze(0)
        ).item()
    )
    print(f"    direction norm={true_norm:.2f}, cosine to readout={cosine_readout:.3f}")

    results = {
        "baseline": {"asr": asr_base, "ca": ca_base},
        "direction_norm": true_norm,
        "cosine_direction_to_readout": cosine_readout,
        "erase_layers": ERASE_LAYERS,
    }

    # Stage 1: known direction
    model_known = load_checkpoint("vit", ckpt_path, DEVICE)
    erase_direction(model_known, true_direction, ERASE_LAYERS)
    asr_known, ca_known = eval_model(
        model_known, dataset_name, attack_name, target_label
    )
    results["known_direction"] = {"asr": asr_known, "ca": ca_known}
    print(f"    known direction: ASR={asr_known:.3f} CA={ca_known:.3f}")
    del model_known

    # Stage 2: blind direction (readout weight as proxy)
    model_blind = load_checkpoint("vit", ckpt_path, DEVICE)
    erase_direction(model_blind, readout_weight, ERASE_LAYERS)
    asr_blind, ca_blind = eval_model(
        model_blind, dataset_name, attack_name, target_label
    )
    results["blind_direction"] = {"asr": asr_blind, "ca": ca_blind}
    print(f"    blind (readout): ASR={asr_blind:.3f} CA={ca_blind:.3f}")
    del model_blind

    # Control: random direction
    torch.manual_seed(42)
    random_dir = torch.randn(768)
    model_random = load_checkpoint("vit", ckpt_path, DEVICE)
    erase_direction(model_random, random_dir, ERASE_LAYERS)
    asr_random, ca_random = eval_model(
        model_random, dataset_name, attack_name, target_label
    )
    results["random_direction"] = {"asr": asr_random, "ca": ca_random}
    print(f"    random control: ASR={asr_random:.3f} CA={ca_random:.3f}")
    del model_random

    del model
    torch.cuda.empty_cache()
    return results


def main():
    all_results = []

    for dataset_name, rate_tag in SETTINGS:
        print(f"\n=== {dataset_name} rate={rate_tag} ===")
        for attack_name in ATTACKS:
            folder = f"vit_{dataset_name}_{attack_name}_{rate_tag}"
            if not os.path.exists(os.path.join(CHECKPOINTS_DIR, folder, "args.json")):
                continue

            print(f"  {attack_name}:")
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
        f"{'folder':40s} {'base ASR':>10} {'known':>10} {'blind':>10} {'random':>10} {'base CA':>10} {'known CA':>10}"
    )
    for entry in all_results:
        if "skip_reason" in entry:
            continue
        print(
            f"{entry['folder']:40s} "
            f"{entry['baseline']['asr']:10.3f} "
            f"{entry['known_direction']['asr']:10.3f} "
            f"{entry['blind_direction']['asr']:10.3f} "
            f"{entry['random_direction']['asr']:10.3f} "
            f"{entry['baseline']['ca']:10.3f} "
            f"{entry['known_direction']['ca']:10.3f}"
        )


if __name__ == "__main__":
    main()
