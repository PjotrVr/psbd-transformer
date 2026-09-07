"""Does SAM amplify the backdoor in ViT, as it does in the paper's ResNet18?

This is the positive control for H6. If PSBD gets worse on SAM-trained models, there
are two very different explanations:

  (a) SAM does amplify the backdoor here, exactly as the SAM paper claims, but the
      benefit does not reach a prediction-space detector. That is a finding about
      detector families and is the interesting outcome.

  (b) SAM does not amplify the backdoor in ViT at all, so there is nothing for PSBD
      to have missed. Then the finding is about transformers or AdamW, and says
      nothing about detector families.

Only a measurement on our own checkpoints separates them, and it uses the SAM paper's
own metrics so the comparison is on their terms:

  top2_tac      mean of the two largest per-dimension trigger-activated changes.
                Their "backdoor effect", the quantity they show correlates with
                detector AUC at r = 0.71.
  silhouette    how separated clean and triggered features are at the penultimate
                layer. They report 0.19 -> 0.32 for BadNets under SAM.
  rel_direction ||mean(x_trigger - x_clean)|| / mean||x_clean||, this project's
                scale-free direction magnitude, included so the result ties back to
                the rest of the ledger.

All three are computed at the final block output, matching their "last convolutional
layer" choice, in fp32, over eligible paired samples only.

Example
    PYTHONPATH=. python experiments/sam_backdoor_effect/measure.py \
        --attack badnet_a2o blend bpp lf --poison-tag 0_1
"""

import argparse
import json
import os

import numpy as np
import torch
from lightning import seed_everything

from analysis.direction import backdoor_direction, trigger_activated_change
from analysis.features import extract_layer_features
from attacks import build_attack, default_config
from defences.checkpoint_eval import read_checkpoint_metadata
from models import load_checkpoint
from experiments.backdoor_direction_layers.measure import build_paired_loaders
from utils.config import DATASET_REGISTRY

RHOS = ("", "0_05", "0_1", "0_15", "0_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack", nargs="*", default=["badnet_a2o", "blend", "bpp", "lf"]
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--architecture", default="vit")
    parser.add_argument("--poison-tag", default="0_1")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def silhouette(clean: torch.Tensor, triggered: torch.Tensor) -> float:
    """Silhouette of the two-cluster clean/triggered split, the paper's Fig-5 metric.

    Computed on a subsample because a full pairwise distance matrix over both sets is
    quadratic and the statistic is stable well below the full split.
    """
    from sklearn.metrics import silhouette_score

    features = torch.cat([clean, triggered]).numpy()
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(triggered))])
    return float(silhouette_score(features, labels))


def measure_one(folder: str, args: argparse.Namespace, device) -> dict | None:
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    if not os.path.exists(checkpoint_path):
        return None
    metadata = read_checkpoint_metadata(checkpoint_path)
    spec = DATASET_REGISTRY[metadata["dataset"]]
    attack = build_attack(
        metadata["attack"],
        default_config(metadata["attack"]),
        spec.image_size,
        metadata["target_label"],
    )
    clean_loader, backdoor_loader, _ = build_paired_loaders(
        metadata["dataset"],
        attack,
        args.raw_data_dir,
        args.batch_size,
        args.samples,
        args.seed,
    )
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device)

    clean = extract_layer_features(
        model, clean_loader, device, use_bfloat16=False, reduction="cls"
    )[args.layer]
    triggered = extract_layer_features(
        model, backdoor_loader, device, use_bfloat16=False, reduction="cls"
    )[args.layer]

    tac = trigger_activated_change(clean, triggered)
    direction = backdoor_direction(clean, triggered)
    scale = clean.norm(dim=1).mean().clamp_min(1e-8)

    metrics = json.load(
        open(os.path.join(args.checkpoints_dir, folder, "metrics.json"))
    )
    return {
        "folder": folder,
        "rho": metadata.get("rho"),
        "asr": metrics.get("asr"),
        "clean_accuracy": metrics.get("clean_accuracy"),
        "top2_tac": float(tac.topk(2).values.mean()),
        "mean_tac": float(tac.mean()),
        "rel_direction": float(direction.norm() / scale),
        "silhouette": silhouette(clean, triggered),
        # Their stage-2 exists to undo this, so it is the quantity most likely to
        # explain a prediction-space detector degrading while a feature-space one
        # improves.
        "clean_intra_class_variance": float(clean.var(dim=0).mean()),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"SAM's effect on the backdoor, {args.architecture}/{args.dataset}, "
        f"poison {args.poison_tag}, layer {args.layer}, {args.samples} paired samples\n"
    )
    print(
        f"{'attack':14} {'rho':>6} {'ASR':>5} {'top2 TAC':>9} {'silhouette':>11} "
        f"{'rel dir':>8} {'clean var':>10}"
    )
    print("-" * 74)

    rows = []
    for attack in args.attack:
        stem = f"{args.architecture}_{args.dataset}_{attack}_{args.poison_tag}"
        for rho in RHOS:
            folder = stem if rho == "" else f"{stem}_sam_rho_{rho}"
            row = measure_one(folder, args, device)
            if row is None:
                continue
            row["attack"] = attack
            rows.append(row)
            label = "adam" if rho == "" else rho.replace("_", ".")
            print(
                f"{attack:14} {label:>6} {row['asr']:>5.2f} {row['top2_tac']:>9.4f} "
                f"{row['silhouette']:>11.4f} {row['rel_direction']:>8.4f} "
                f"{row['clean_intra_class_variance']:>10.4f}"
            )
        print()

    # The paper's claim, restated as a per-attack comparison against the Adam baseline.
    print("SAM against Adam, per attack (positive means SAM amplified it):")
    print(f"{'attack':14} {'d top2 TAC':>11} {'d silhouette':>13} {'d clean var':>12}")
    for attack in args.attack:
        adam = next(
            (r for r in rows if r["attack"] == attack and r["rho"] is None), None
        )
        sam = [r for r in rows if r["attack"] == attack and r["rho"] is not None]
        if adam is None or not sam:
            continue
        mean = lambda key: sum(r[key] for r in sam) / len(sam)  # noqa: E731
        print(
            f"{attack:14} {mean('top2_tac') - adam['top2_tac']:>+11.4f} "
            f"{mean('silhouette') - adam['silhouette']:>+13.4f} "
            f"{mean('clean_intra_class_variance') - adam['clean_intra_class_variance']:>+12.4f}"
        )

    path = os.path.join(args.results_dir, "sam_backdoor_effect.json")
    with open(path, "w") as handle:
        json.dump(
            {"layer": args.layer, "samples": args.samples, "rows": rows},
            handle,
            indent=2,
        )
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
