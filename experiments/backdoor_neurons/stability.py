"""Is the top-k TAC set a property of the MODEL, or of the 600 samples used?

The SAM conclusion is a Jaccard comparison: the top-20 TAC dimensions of an
Adam-trained backdoor barely overlap those of the same attack trained with SAM. That
reading is only valid if the top-20 is reproducible in the first place. If TAC's
ranking were dominated by sampling noise, every cross-model Jaccard would sit at
chance for trivial reasons and the SAM claim would be vacuous.

So measure the ceiling directly: split the same paired samples into 2 disjoint halves,
rank dimensions independently in each, and take the Jaccard. This is the same model,
the same trigger, the same layer, differing only in which images were used.

    split_half_jaccard  how much of the ranking survives a change of sample
    chance              two independent k-of-768 draws, measured not assumed

A cross-model Jaccard is only interpretable BETWEEN those 2 numbers. Near the
split-half value means the dimensions were preserved; near chance means they moved.

Example
    PYTHONPATH=. python experiments/backdoor_neurons/stability.py --attack badnet_a2o blend
"""

import argparse
import json
import os
import random

import torch
from lightning import seed_everything

from analysis.direction import trigger_activated_change
from analysis.features import extract_layer_features
from attacks import build_attack, default_config
from defences.checkpoint_eval import read_checkpoint_metadata, resolve_probe_attack
from models import load_checkpoint
from experiments.backdoor_direction_layers.measure import build_paired_loaders
from utils.config import DATASET_REGISTRY

TOP_K = 20
CHANCE_TRIALS = 20000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack",
        nargs="*",
        default=["badnet_a2o", "blend", "bpp", "lf", "badnet_a2a"],
    )
    parser.add_argument("--rho", nargs="*", default=["", "0_1", "0_2"])
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--architecture", default="vit")
    parser.add_argument("--poison-tag", default="0_1")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-attack", default="badnet_a2o")
    parser.add_argument("--probe-target-label", type=int, default=0)
    return parser.parse_args()


def jaccard(a, b) -> float:
    left, right = set(a), set(b)
    return len(left & right) / max(len(left | right), 1)


def chance_jaccard(k: int, dim: int, seed: int) -> tuple[float, float]:
    """The null, measured rather than assumed, as mean and 95th percentile."""
    generator = random.Random(seed)
    values = [
        jaccard(generator.sample(range(dim), k), generator.sample(range(dim), k))
        for _ in range(CHANCE_TRIALS)
    ]
    values.sort()
    return sum(values) / len(values), values[int(0.95 * len(values))]


def split_half(folder: str, args: argparse.Namespace, device) -> dict | None:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    if not os.path.exists(path):
        return None
    metadata = read_checkpoint_metadata(path)
    spec = DATASET_REGISTRY[metadata["dataset"]]
    benign = metadata["attack"] == "benign"
    attack_name, target_label = resolve_probe_attack(
        metadata,
        args.probe_attack if benign else None,
        args.probe_target_label if benign else None,
    )
    attack = build_attack(
        attack_name, default_config(attack_name), spec.image_size, target_label
    )
    clean_loader, triggered_loader, _ = build_paired_loaders(
        metadata["dataset"],
        attack,
        args.raw_data_dir,
        args.batch_size,
        args.samples,
        args.seed,
    )
    model = load_checkpoint(metadata["architecture"], path, device)

    seed_everything(args.seed)
    clean = extract_layer_features(model, clean_loader, device, False, "cls")
    seed_everything(args.seed)
    triggered = extract_layer_features(model, triggered_loader, device, False, "cls")

    report_path = os.path.join(args.results_dir, folder, "backdoor_neurons.json")
    with open(report_path) as handle:
        peak = json.load(handle)["peak_layer"]

    c, t = clean[peak].cpu(), triggered[peak].cpu()
    # The halves are an interleave, not a cut, so any drift over the loader's order
    # lands in both halves rather than separating them.
    first, second = torch.arange(0, len(c), 2), torch.arange(1, len(c), 2)
    top_a = trigger_activated_change(c[first], t[first]).topk(args.top_k).indices
    top_b = trigger_activated_change(c[second], t[second]).topk(args.top_k).indices
    return {
        "folder_name": folder,
        "peak_layer": peak,
        "n_per_half": int(len(first)),
        "split_half_jaccard": jaccard(top_a.tolist(), top_b.tolist()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, p95 = chance_jaccard(args.top_k, 768, args.seed)
    print(f"Split-half stability of the top-{args.top_k} TAC set at each peak layer.")
    print(f"chance Jaccard: mean {mean:.4f}, p95 {p95:.4f}\n")
    print(f"{'checkpoint':40} {'peak':>5} {'n/half':>7} {'split-half J':>13}")
    print("-" * 68)

    rows = []
    for attack in args.attack:
        for rho in args.rho:
            stem = f"{args.architecture}_{args.dataset}_{attack}"
            if attack != "benign":
                stem += f"_{args.poison_tag}"
            folder = stem if rho == "" else f"{stem}_sam_rho_{rho}"
            try:
                row = split_half(folder, args, device)
            except Exception as error:
                print(f"{folder:40} FAILED {type(error).__name__}: {error}", flush=True)
                continue
            if row is None:
                continue
            rows.append(row)
            print(
                f"{folder:40} {row['peak_layer']:>5} {row['n_per_half']:>7} "
                f"{row['split_half_jaccard']:>13.2f}",
                flush=True,
            )

    if rows:
        average = sum(r["split_half_jaccard"] for r in rows) / len(rows)
        print(
            f"\nmean split-half Jaccard {average:.2f} against a chance floor of {mean:.3f}"
        )
        out = os.path.join(args.results_dir, "backdoor_neuron_stability.json")
        with open(out, "w") as handle:
            json.dump(
                {"chance_mean": mean, "chance_p95": p95, "rows": rows}, handle, indent=2
            )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
