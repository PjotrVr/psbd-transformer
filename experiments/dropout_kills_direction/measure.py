"""Does dropout at each placement destroy the backdoor direction, or spare it?

This is the causal link between the placement result and the mechanism. PSBD works
when dropout destroys the evidence a clean prediction rests on while leaving the
trigger-to-target path intact. That claim is stated in probability space (PSU), but
it is really a claim about the residual stream, and this measures it there.

Procedure, for one checkpoint and one placement:

  1. With no dropout, compute the backdoor direction r at the final layer from
     paired clean and triggered features. This is the reference direction; it is
     never recomputed under perturbation, because the question is how much of
     *this* direction survives.
  2. For each rate, plug the placement, and project the perturbed final-layer CLS
     features onto r for clean and for triggered inputs.
  3. Report the separation between the two projections, normalized by the
     unperturbed separation.

The prediction that distinguishes the placements:

  a placement PSBD works at    keeps the triggered projection high while the
                               clean one collapses, so separation is preserved
  a placement PSBD fails at    collapses both, so separation goes to zero and
                               there is nothing left for a threshold to find

Note this measures the same phenomenon as PSU but one step upstream, so agreement
between the two is evidence the mechanism story is right rather than a restatement
of it.

**Checked against the LayerNorm artifact that invalidated H16 section 9, and clean.**
The final LayerNorm sits between this measurement and PSU, and rescaling to unit
variance can restore what looks destroyed at a block output. Run with --post-ln on
`vit_cifar10_badnet_a2o_0_1`, the normalized separations are unchanged:
pre_residual 0.696 / 0.253 / 0.065 becomes 0.778 / 0.297 / 0.064, and post_residual
stays 0.015 / 0.000 / 0.000, at rates 0.1 / 0.3 / 0.5.

The reason is the choice of statistic. The RAW projections move enormously under
LayerNorm (post_residual at rate 0.5 reads 26.5 before it and 0.104 after), but
`separation` divides by the pooled standard deviation, and that quotient already
absorbs the rescaling LayerNorm applies. A raw mean difference here would have been
as fragile as the ablation was.

Example
    PYTHONPATH=. python experiments/dropout_kills_direction/measure.py \
        --checkpoint-folder vit_cifar10_badnet_a2o_0_1 \
        --placement pre_residual post_residual
"""

import argparse
import json
import os

import torch
from lightning import seed_everything

from analysis.direction import backdoor_direction, project_onto_direction
from analysis.features import extract_layer_features
from attacks import build_attack, default_config
from data.splits import read_checkpoint_metadata, resolve_probe_attack
from models.positions import DROPOUT_CONFIGS, plug_dropout, unplug_dropout
from models.backbones import load_checkpoint, network_core
from experiments.backdoor_direction_layers.measure import build_paired_loaders
from data.registry import DATASET_REGISTRY

# The default grid. A residual-stream placement saturates below its first entry,
# so --rates exists to reach the 0.01-to-0.09 window where it might still have a
# non-degenerate regime. See docs/hypothesis/H9-strength-not-position.md.
RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", required=True)
    parser.add_argument("--placement", nargs="*", default=list(DROPOUT_CONFIGS))
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--post-ln", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-attack", default=None)
    parser.add_argument("--probe-target-label", type=int, default=None)
    parser.add_argument("--rates", nargs="*", type=float, default=None)
    return parser.parse_args()


def final_layer_features(
    model, loader, device, layer: int, seed: int, post_ln: bool = False
) -> torch.Tensor:
    """CLS features at one layer, with the mask sequence pinned by seed.

    Reseeding immediately before extraction is what makes the clean and triggered
    passes comparable: without it they draw different masks, and the difference
    between them would be mask noise plus trigger rather than trigger. At the
    rates in play here the mask noise is the larger term, so this is not a
    refinement, it decides whether the measurement means anything.

    post_ln applies the final LayerNorm, and at layer 12 that is the difference
    between what the block emits and what the head reads. It matters here because
    this script's whole purpose is to explain PSU, which is measured at the output,
    and LayerNorm sits in between. Renormalizing to unit variance can restore a
    separation that looks collapsed at the block output, which is exactly the
    artifact that produced a false result in H16 section 9.
    """
    seed_everything(seed)
    features = extract_layer_features(
        model, loader, device, use_bfloat16=False, reduction="cls"
    )[layer]
    if post_ln:
        with torch.inference_mode():
            features = network_core(model).encoder.ln(features.to(device)).float().cpu()
    return features


def predictions(model, loader, device, seed: int) -> torch.Tensor:
    """Argmax class per sample, with the mask sequence pinned by seed."""
    seed_everything(seed)
    out = []
    with torch.inference_mode():
        for images, _ in loader:
            out.append(model(images.to(device)).argmax(dim=1).cpu())
    return torch.cat(out)


def separation(
    clean_projection: torch.Tensor, backdoor_projection: torch.Tensor
) -> float:
    """Standardized mean difference along the direction (Cohen's d).

    A plain mean difference would confound "the two groups moved apart" with "the
    perturbation inflated everything's variance", and dropout does inflate
    variance. Pooled standard deviation in the denominator separates them.
    """
    difference = backdoor_projection.mean() - clean_projection.mean()
    pooled = torch.sqrt(
        (clean_projection.var() + backdoor_projection.var()) / 2
    ).clamp_min(1e-8)
    return float(difference / pooled)


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
    architecture = metadata["architecture"]
    attack = build_attack(
        attack_name,
        default_config(attack_name),
        DATASET_REGISTRY[dataset_name].image_size,
        target_label,
    )
    clean_loader, backdoor_loader, _ = build_paired_loaders(
        dataset_name,
        attack,
        args.raw_data_dir,
        args.batch_size,
        args.samples,
        args.seed,
    )
    model = load_checkpoint(architecture, checkpoint_path, device)

    clean_reference = final_layer_features(
        model, clean_loader, device, args.layer, args.seed, args.post_ln
    )
    backdoor_reference = final_layer_features(
        model, backdoor_loader, device, args.layer, args.seed, args.post_ln
    )
    direction = backdoor_direction(clean_reference, backdoor_reference)
    baseline_separation = separation(
        project_onto_direction(clean_reference, direction),
        project_onto_direction(backdoor_reference, direction),
    )

    print(f"{args.checkpoint_folder}  attack={attack_name}  layer={args.layer}")
    print(
        f"unperturbed separation along the backdoor direction: {baseline_separation:.3f}\n"
    )
    print(
        f"{'placement':26} {'rate':>5} {'sigma':>7} {'proj_bd':>9} {'sep':>7} {'sep/base':>9}"
    )

    results = {
        "folder_name": args.checkpoint_folder,
        "attack": attack_name,
        "layer": args.layer,
        "samples": args.samples,
        "seed": args.seed,
        "baseline_separation": baseline_separation,
        "placements": {},
    }

    baseline_prediction = predictions(model, clean_loader, device, args.seed)

    for placement in args.placement:
        names = DROPOUT_CONFIGS.get(placement, (placement,))
        rows = []
        for rate in tuple(args.rates) if args.rates else RATES:
            handles = plug_dropout(model, architecture, names, {}, rate)
            try:
                clean_features = final_layer_features(
                    model, clean_loader, device, args.layer, args.seed, args.post_ln
                )
                backdoor_features = final_layer_features(
                    model, backdoor_loader, device, args.layer, args.seed, args.post_ln
                )
                # Comparing placements at a shared RATE is the standing error this
                # project keeps rediscovering (H9): the same rate is a different
                # intervention strength at different positions. Recording the clean
                # shift ratio here gives the common axis to compare on instead.
                shifted = predictions(model, clean_loader, device, args.seed)
            finally:
                unplug_dropout(handles)
            shift_ratio = float((shifted != baseline_prediction).float().mean())

            clean_projection = project_onto_direction(clean_features, direction)
            backdoor_projection = project_onto_direction(backdoor_features, direction)
            value = separation(clean_projection, backdoor_projection)
            rows.append(
                {
                    "rate": rate,
                    "clean_shift_ratio": shift_ratio,
                    "projection_clean": float(clean_projection.mean()),
                    "projection_backdoor": float(backdoor_projection.mean()),
                    "separation": value,
                    "separation_ratio": value / baseline_separation
                    if baseline_separation
                    else None,
                }
            )
            print(
                f"{placement:26} {rate:>5.2f} {shift_ratio:>7.3f} "
                f"{rows[-1]['projection_backdoor']:>9.3f} {value:>7.3f} "
                f"{rows[-1]['separation_ratio']:>9.3f}"
            )
        results["placements"][placement] = rows

    directory = os.path.join(args.results_dir, args.checkpoint_folder)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "dropout_direction_survival.json")
    with open(path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
