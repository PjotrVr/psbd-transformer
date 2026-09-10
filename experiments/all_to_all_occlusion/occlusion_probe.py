"""A spatially selective probe for all-to-all backdoors.

Dropout is the wrong probe for all-to-all. It degrades the trigger channel and the
natural-image channel together, so a poisoned sample loses its poison prediction and
its source-class evidence at the same time and simply drains to whatever class the
network collapses onto. Measured on ViT: poisoned samples revert to their source
class at 0.027, four times BELOW chance, while 0.90 of their shifts land on the same
sink clean data uses.

Occlusion is selective. The trigger occupies specific pixels; the object does not.
Blank the right patch and the trigger channel dies while the natural channel survives,
so the prediction should fall back from (y + 1) to y. That off-by-one transition is the
attacker's own permutation showing itself, and clean data has no reason to produce it.

The permutation is not assumed. It is estimated from the aggregate transition counts
with the Hungarian algorithm and calibrated against clean validation data, so the
score needs no knowledge of the attack.

    python experiments/all_to_all_occlusion/occlusion_probe.py \
        --checkpoint checkpoints/vit_cifar10_badnet_a2a_0_1/attack_result.pt
"""

import argparse
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

from attacks import build_attack, default_config
from evaluation.loaders import build_balanced_eval_loaders
from data.splits import (
    read_checkpoint_metadata,
    resolve_probe_attack,
)
from data.registry import DATASET_REGISTRY
from models.backbones import load_checkpoint
from data.registry import RunConfig


def occlusion_grid(image_size: int, patch: int, stride: int) -> list[tuple[int, int]]:
    """Top-left corners of every patch position, including the far edges."""
    stops = list(range(0, image_size - patch + 1, stride))
    if stops[-1] != image_size - patch:
        stops.append(image_size - patch)
    return [(r, c) for r in stops for c in stops]


@torch.no_grad()
def transitions(model, loader, device, positions, patch, limit):
    """Baseline predictions and, per occlusion position, the occluded predictions."""
    base, occluded, seen = [], [], 0
    for images, _ in loader:
        images = images.to(device)
        base.append(model(images).argmax(1).cpu())
        per_position = []
        for row, col in positions:
            probe = images.clone()
            probe[:, :, row : row + patch, col : col + patch] = 0.0
            per_position.append(model(probe).argmax(1).cpu())
        occluded.append(torch.stack(per_position, 1))  # (batch, n_positions)
        seen += len(images)
        if limit and seen >= limit:
            break
    return torch.cat(base), torch.cat(occluded)


def estimate_permutation(base, occluded, num_classes):
    """The bijection that best explains the observed occlusion transitions.

    Diagonal excluded: staying put is not a transition, and it dominates the counts.
    """
    counts = np.zeros((num_classes, num_classes))
    flat_from = base.repeat_interleave(occluded.shape[1]).numpy()
    flat_to = occluded.reshape(-1).numpy()
    moved = flat_from != flat_to
    np.add.at(counts, (flat_from[moved], flat_to[moved]), 1)
    row_sums = counts.sum(1, keepdims=True)
    rates = counts / np.clip(row_sums, 1, None)
    np.fill_diagonal(rates, 0.0)
    rows, cols = linear_sum_assignment(-rates)
    return cols, float(rates[rows, cols].mean())


def score(base, occluded, permutation):
    """Fraction of occlusions that send a sample to its permuted class."""
    target = torch.as_tensor(permutation)[base].unsqueeze(1)
    return (occluded == target).float().mean(1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    # Defaults are derived from the image size at run time: the loader yields the
    # dataset's native resolution and the model resizes internally, so a fixed
    # pixel patch would be meaningless across datasets.
    parser.add_argument("--patch", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0, help="0 uses whatever the loaders provide"
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=150,
        help="RunConfig cap on the eval splits; this, not --limit, "
        "sets how much data the permutation is fitted on",
    )
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--probe-attack",
        default=None,
        help="trigger to probe a benign checkpoint with; the negative control",
    )
    parser.add_argument("--probe-target-label", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = load_checkpoint("vit", args.checkpoint, device).eval()
    config = RunConfig(batch_size=64, examples_per_class=args.examples_per_class)
    metadata = read_checkpoint_metadata(args.checkpoint)
    attack_name, target_label = resolve_probe_attack(
        metadata, args.probe_attack, args.probe_target_label
    )
    dataset_name = metadata["dataset"]
    size = DATASET_REGISTRY[dataset_name].image_size
    attack = build_attack(attack_name, default_config(attack_name), size, target_label)
    clean_val, clean_eval, backdoor_eval = build_balanced_eval_loaders(
        dataset_name,
        attack,
        image_size=size,
        clean_val_size=config.clean_val_size,
        examples_per_class=config.examples_per_class,
        raw_data_dir=config.raw_data_dir,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    print(
        f"model {metadata['attack']}, probed with {attack_name} (target {target_label})"
    )
    image_size = next(iter(clean_val))[0].shape[-1]
    patch = args.patch or max(4, image_size // 4)
    stride = args.stride or max(2, patch // 2)
    positions = occlusion_grid(image_size, patch, stride)
    print(
        f"{len(positions)} occlusion positions, {patch}px patch, stride {stride}, {image_size}px image"
    )

    splits = {}
    for name, loader in (
        ("validation", clean_val),
        ("clean", clean_eval),
        ("backdoor", backdoor_eval),
    ):
        splits[name] = transitions(model, loader, device, positions, patch, args.limit)
        print(f"  {name}: {len(splits[name][0])} samples")

    # Estimate the permutation on the suspect data, calibrate the score it produces
    # against clean validation, which by assumption contains no poison.
    permutation, strength = estimate_permutation(*splits["backdoor"], args.num_classes)
    null_perm, null_strength = estimate_permutation(
        *splits["validation"], args.num_classes
    )
    print(
        f"\nHungarian match strength: suspect {strength:.4f} vs clean-validation null "
        f"{null_strength:.4f}  ({strength / max(null_strength, 1e-9):.2f}x)"
    )
    print(f"estimated permutation: {list(permutation)}")
    print(f"identity would be    : {list(range(args.num_classes))}")

    # Ablation. If "the prediction moved at all" separates as well as "the prediction
    # moved to its permuted class", the permutation readout is doing no work and this is
    # just occlusion sensitivity.
    def moved(base, occluded):
        return (occluded != base.unsqueeze(1)).float().mean(1).numpy()

    labels_ab = np.r_[
        np.zeros(len(splits["clean"][0])), np.ones(len(splits["backdoor"][0]))
    ]
    ab = roc_auc_score(
        labels_ab, np.r_[moved(*splits["clean"]), moved(*splits["backdoor"])]
    )
    print(
        f"\nABLATION, any movement (no permutation readout): AUROC {max(ab, 1 - ab):.4f}"
    )
    null_permutation, _ = estimate_permutation(*splits["validation"], args.num_classes)
    print(f"permutation estimated from CLEAN VALIDATION: {list(null_permutation)}")
    ident = list(range(args.num_classes))
    print(
        f"  suspect perm is a single {args.num_classes}-cycle: "
        f"{sorted(permutation.tolist()) == ident and len(set(permutation.tolist())) == args.num_classes}"
    )

    half = len(splits["backdoor"][0]) // 2
    fit = tuple(t[:half] for t in splits["backdoor"])
    held = tuple(t[half:] for t in splits["backdoor"])
    held_clean = tuple(t[half:] for t in splits["clean"])
    perm_a, _ = estimate_permutation(*fit, args.num_classes)
    y_h = np.r_[np.zeros(len(held_clean[0])), np.ones(len(held[0]))]
    transfer = roc_auc_score(
        y_h, np.r_[score(*held_clean, perm_a), score(*held, perm_a)]
    )
    print(
        f"\nSPLIT-HALF: permutation fitted on half the suspect data, scored on the "
        f"held-out half -> AUROC {transfer:.4f}"
    )
    print(f"  fitted permutation {list(perm_a)}")

    clean_scores = score(*splits["clean"], permutation)
    backdoor_scores = score(*splits["backdoor"], permutation)
    labels = np.r_[np.zeros(len(clean_scores)), np.ones(len(backdoor_scores))]
    auroc = roc_auc_score(labels, np.r_[clean_scores, backdoor_scores])
    print(
        f"\nmean score  clean {clean_scores.mean():.4f}   backdoor {backdoor_scores.mean():.4f}"
    )
    print(f"AUROC (occlusion + permutation readout): {auroc:.4f}")
    print("  for reference, PSU on this cell: 0.440 (0.560 two-sided)")


if __name__ == "__main__":
    main()
