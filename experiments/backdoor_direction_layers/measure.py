"""Where in the ViT stack does each attack write its backdoor direction?

For one checkpoint, extracts the residual stream at every block for paired clean
and triggered versions of the same images, then reports per layer:

  rel_direction  ||mean(x_trigger - x_clean)|| / mean||x_clean||
  max_tac        the largest per-dimension trigger-activated change
  cka            debiased linear CKA between the clean and triggered features

Three deliberate choices, each of which changes the answer:

**fp32, not bfloat16.** The clean and triggered passes are separate forwards, so
their rounding errors are independent. The direction is a signed mean and averages
that noise down by sqrt(N); TAC is a mean of absolute differences and does not.
bf16 carries ~0.4% relative error, and ViT residual norms grow with depth, so a
bf16 TAC has a noise floor that rises with depth. That is exactly the shape a real
finding would have, which makes it the worst possible artifact to leave in.

**Relative direction norm, not raw.** The residual stream's norm grows
monotonically with depth in a ViT, so an argmax over the raw norm selects a late
layer for essentially any model, backdoored or benign. Dividing by the mean clean
feature norm makes the quantity scale-free and comparable across layers.

**Eligible images only.** Triggering an image whose true class is already the
target contributes a near-zero difference and dilutes the direction by roughly the
class prior. poison.is_eval_poisonable is the same eligibility rule the ASR set
uses, and for the same reason: the question is whether the trigger moves a
non-target image.

Run on the login node; one checkpoint at 1000 samples takes about a minute.

Example
    python experiments/backdoor_direction_layers/measure.py \
        --checkpoint-folder vit_cifar10_badnet_a2o_0_1
"""

import argparse
import json
import os

import torch
import torchvision.transforms.v2 as transforms_v2
from lightning import seed_everything
from torch.utils.data import DataLoader, Subset

from analysis.cka import debiased_linear_cka
from analysis.direction import backdoor_direction, trigger_activated_change
from analysis.features import extract_layer_features
from attacks import build_attack, default_config
from data.splits import read_checkpoint_metadata, resolve_probe_attack
from models.backbones import load_checkpoint
from attacks.poisoning import PoisonedTrainingSet, is_eval_poisonable
from data.registry import DATASET_REGISTRY
from data.loading import extract_labels, load_clean_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reduction", default="cls", choices=("cls", "mean"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-attack", default=None)
    parser.add_argument("--probe-target-label", type=int, default=None)
    return parser.parse_args()


def build_paired_loaders(
    dataset_name: str,
    attack,
    raw_data_dir: str,
    batch_size: int,
    sample_count: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, list[int]]:
    """Clean and triggered loaders over the same eligible images, row-aligned.

    Both are shuffle=False over one Subset, so row i of each is the same test
    image once clean and once triggered. That pairing is what the direction and
    TAC require, and it is asserted downstream rather than assumed.
    """
    spec = DATASET_REGISTRY[dataset_name]
    transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((spec.image_size, spec.image_size)),
            transforms_v2.ToTensor(),
        ]
    )
    _, test_clean = load_clean_datasets(dataset_name, transform, raw_data_dir)
    labels = extract_labels(test_clean)

    eligible = [
        index
        for index, label in enumerate(labels)
        if is_eval_poisonable(attack.label_mode, int(label), attack.target_label)
    ]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(eligible), generator=generator)[:sample_count]
    chosen = [eligible[position] for position in order.tolist()]

    subset = Subset(test_clean, chosen)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    clean_set = PoisonedTrainingSet(subset, attack, set(), normalize, spec.num_classes)
    triggered_set = PoisonedTrainingSet(
        subset, attack, set(range(len(subset))), normalize, spec.num_classes
    )

    def loader(dataset):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader(clean_set), loader(triggered_set), chosen


def per_layer_table(clean: dict, backdoor: dict) -> list[dict]:
    """One row per layer, all quantities scale-free or explicitly normalized."""
    rows = []
    for layer in sorted(clean):
        clean_features, backdoor_features = clean[layer], backdoor[layer]
        assert clean_features.shape == backdoor_features.shape, (
            f"layer {layer}: clean and triggered features are not row-aligned "
            f"({clean_features.shape} vs {backdoor_features.shape})"
        )
        direction = backdoor_direction(clean_features, backdoor_features)
        scale = clean_features.norm(dim=1).mean().clamp_min(1e-8)
        tac = trigger_activated_change(clean_features, backdoor_features)
        rows.append(
            {
                "layer": layer,
                "rel_direction_norm": float(direction.norm() / scale),
                "direction_norm": float(direction.norm()),
                "feature_norm": float(scale),
                "max_tac": float(tac.max()),
                "mean_tac": float(tac.mean()),
                "cka": float(debiased_linear_cka(clean_features, backdoor_features)),
            }
        )
    return rows


def onset_layer(rows: list[dict]) -> int:
    """The layer with the largest jump in relative direction norm.

    The earliest sharp gain, not the global maximum. The companion paper picks the
    earliest layer deliberately: a late layer can look strongest simply because the
    representation is closest to the logits there, which says nothing about where
    the trigger information first became linearly available.
    """
    best_layer, best_gain = rows[0]["layer"], float("-inf")
    for previous, current in zip(rows, rows[1:]):
        gain = current["rel_direction_norm"] - previous["rel_direction_norm"]
        if gain > best_gain:
            best_layer, best_gain = current["layer"], gain
    return best_layer


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(
        args.checkpoints_dir, args.checkpoint_folder, "attack_result.pt"
    )
    metadata = read_checkpoint_metadata(checkpoint_path)
    attack_name, target_label = resolve_probe_attack(
        metadata, args.probe_attack, args.probe_target_label
    )
    dataset_name = metadata["dataset"]
    attack = build_attack(
        attack_name,
        default_config(attack_name),
        DATASET_REGISTRY[dataset_name].image_size,
        target_label,
    )

    clean_loader, backdoor_loader, chosen = build_paired_loaders(
        dataset_name,
        attack,
        args.raw_data_dir,
        args.batch_size,
        args.samples,
        args.seed,
    )
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device)

    # fp32 on both passes: see the module docstring on the bf16 TAC noise floor.
    clean_features = extract_layer_features(
        model, clean_loader, device, use_bfloat16=False, reduction=args.reduction
    )
    backdoor_features = extract_layer_features(
        model, backdoor_loader, device, use_bfloat16=False, reduction=args.reduction
    )

    rows = per_layer_table(clean_features, backdoor_features)
    onset = onset_layer(rows)

    print(f"{args.checkpoint_folder}  attack={attack_name}  reduction={args.reduction}")
    print(f"{'layer':>5} {'rel_dir':>9} {'max_tac':>9} {'cka':>7} {'feat_norm':>10}")
    for row in rows:
        print(
            f"{row['layer']:>5} {row['rel_direction_norm']:>9.4f} "
            f"{row['max_tac']:>9.4f} {row['cka']:>7.4f} {row['feature_norm']:>10.2f}"
        )
    print(f"\nsharpest relative gain enters at layer {onset}")

    output = {
        "folder_name": args.checkpoint_folder,
        "attack": attack_name,
        "dataset": dataset_name,
        "reduction": args.reduction,
        "samples": len(chosen),
        "seed": args.seed,
        "dtype": "float32",
        "onset_layer": onset,
        "layers": rows,
    }
    directory = os.path.join(args.results_dir, args.checkpoint_folder)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"direction_layers_{args.reduction}.json")
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"written to {path}")


if __name__ == "__main__":
    main()
