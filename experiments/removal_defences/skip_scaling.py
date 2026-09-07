"""H39: skip connection scaling as zero-cost defense.

Scales down the residual (skip) connection at layers 10-11 by a factor alpha,
testing whether this removes the backdoor without knowing the direction.
The idea: H34 showed the direction rides the skip path, so scaling it down
should reduce the accumulated backdoor signal.

Modifies the forward pass at inference time using hooks.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch

from attacks import default_config
from evaluate import evaluate_attack
from models import load_checkpoint, network_core


CHECKPOINTS_DIR = "checkpoints"
OUTPUT_PATH = "results/skip_scaling.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

ALPHAS = [1.0, 0.5, 0.3, 0.1, 0.0]
SCALE_LAYERS = [10, 11]


def make_skip_scaling_hook(alpha):
    """Hook that scales the residual input before adding the branch output.

    ViT EncoderBlock forward is:
        x = x + self.self_attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

    We intercept the block's forward to scale the skip path. The hook captures
    the input x, and modifies the output to:
        output = alpha * input + (output - input)
    which is equivalent to alpha * x + branch.
    """

    def hook_fn(module, input_tuple, output):
        x_input = input_tuple[0]
        branch = output - x_input
        return alpha * x_input + branch

    return hook_fn


def eval_with_skip_scaling(model, dataset_name, attack_name, target_label, alpha):
    core = network_core(model)
    hooks = []

    for layer_idx in SCALE_LAYERS:
        block = core.encoder.layers[layer_idx]
        h = block.register_forward_hook(make_skip_scaling_hook(alpha))
        hooks.append(h)

    config = default_config(attack_name)
    metrics = evaluate_attack(
        model, dataset_name, attack_name, config, target_label, DEVICE
    )

    for h in hooks:
        h.remove()

    return metrics["asr"], metrics["clean_accuracy"]


def main():
    all_results = []

    for dataset_name, rate_tag in SETTINGS:
        print(f"\n=== {dataset_name} rate={rate_tag} ===")
        for attack_name in ATTACKS:
            folder = f"vit_{dataset_name}_{attack_name}_{rate_tag}"
            args_path = os.path.join(CHECKPOINTS_DIR, folder, "args.json")
            if not os.path.exists(args_path):
                continue

            with open(args_path) as f:
                metadata = json.load(f)
            target_label = metadata.get("target_label", 0)

            print(f"  {attack_name}:")
            alpha_results = {}

            for alpha in ALPHAS:
                ckpt_path = os.path.join(CHECKPOINTS_DIR, folder, "attack_result.pt")
                model = load_checkpoint("vit", ckpt_path, DEVICE)

                asr, ca = eval_with_skip_scaling(
                    model, dataset_name, attack_name, target_label, alpha
                )
                alpha_results[str(alpha)] = {"asr": asr, "ca": ca}
                print(f"    alpha={alpha:.1f}: ASR={asr:.3f} CA={ca:.3f}")

                del model
                torch.cuda.empty_cache()

            all_results.append(
                {
                    "dataset": dataset_name,
                    "attack": attack_name,
                    "rate_tag": rate_tag,
                    "folder": folder,
                    "scale_layers": SCALE_LAYERS,
                    "results_by_alpha": alpha_results,
                }
            )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\n=== Summary ===")
    print(f"{'folder':40s}", end="")
    for alpha in ALPHAS:
        print(f" a={alpha:.1f} ASR", end="")
    print(f" {'a=1 CA':>8} {'a=0.3 CA':>8}")

    for entry in all_results:
        print(f"{entry['folder']:40s}", end="")
        for alpha in ALPHAS:
            r = entry["results_by_alpha"][str(alpha)]
            print(f"    {r['asr']:.3f}", end="")
        ca_1 = entry["results_by_alpha"]["1.0"]["ca"]
        ca_03 = entry["results_by_alpha"]["0.3"]["ca"]
        print(f" {ca_1:8.3f} {ca_03:8.3f}")


if __name__ == "__main__":
    main()
