"""Are the located dimensions actually load-bearing, or only correlated?

H16 names a peak layer and a handful of dimensions per attack, all from TAC, which
is a *difference* measurement: it says those dimensions move most when the trigger
appears. That is correlational. It does not establish that the model reads them, and
a large activation change in a dimension the classifier ignores would look identical.

The causal test is to delete them. Zero the named dimensions in the residual stream
at the peak layer, on every input, and re-measure:

    asr           should collapse, if the backdoor is routed through them
    clean_accuracy should barely move, since k is at most 20 of 768

Two controls, both necessary:

  random_k      the same NUMBER of dimensions, drawn at random from the same layer.
                Separates "these dimensions" from "removing any k dimensions at this
                depth breaks the model".
  bottom_k      the k dimensions with the LOWEST TAC at the same layer. A stricter
                control than random, because it holds the selection procedure fixed
                and flips only which end of the ranking is taken.
  random_dir    a random rank-1 direction, removed the same way. The direction
                ablation's own control: removing SOME direction is not the claim.

Ablation is applied as a forward hook on the block output, so no weight is modified
and the model is restored exactly by removing the handle.

Example
    PYTHONPATH=. python scripts/backdoor_neurons/ablate.py --attack badnet_a2o blend
"""

import argparse
import json
import os

import torch
from lightning import seed_everything

from analysis.direction import backdoor_direction, trigger_activated_change
from analysis.features import extract_layer_features
from attacks import build_attack, default_config
from defences.checkpoint_eval import (
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
    resolve_probe_attack,
)
from defences.detection import attack_success_rate, clean_accuracy
from models import load_checkpoint, vit_core
from scripts.backdoor_direction_layers.measure import build_paired_loaders
from utils.config import DATASET_REGISTRY

PSBD_SPLIT_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack",
        nargs="*",
        default=["badnet_a2o", "blend", "bpp", "lf", "badnet_a2a"],
    )
    parser.add_argument("--rho", nargs="*", default=[""])
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--architecture", default="vit")
    parser.add_argument("--poison-tag", default="0_1")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 300])
    parser.add_argument("--random-directions", type=int, default=2)
    parser.add_argument("--rank", nargs="*", type=int, default=[2, 4, 8, 16])
    # Ablating at the peak layer leaves every later block free to rewrite the
    # direction, which is a competing explanation for SAM's resistance. Forcing the
    # layer to 12 removes that freedom and separates the 2 readings.
    parser.add_argument("--layer", type=int, default=None)
    # Remove the direction AFTER the final LayerNorm instead of at a block output.
    # LN renormalizes to unit variance, so a component deleted before it is partly
    # restored by the rescaling of whatever survives; deleting after it is what the
    # head actually sees.
    parser.add_argument("--post-ln", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--tac-samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_ablation_hook(dimensions: torch.Tensor):
    """Zero the named residual-stream dimensions on a block's output.

    Applied to every token, because the CLS token reads from the others through
    attention in later blocks; zeroing CLS alone would leave the information intact
    for a downstream block to re-read.
    """

    def hook(module, inputs, output):
        output = output.clone()
        output[..., dimensions] = 0.0
        return output

    return hook


def make_subspace_hook(basis: torch.Tensor):
    """Remove the span of several directions at once.

    Section 9 of H16 shows a rank-1 removal stops working as SAM's rho rises, which
    says the backdoor becomes spread over a subspace rather than a line. The rank at
    which ASR finally collapses turns that into a number instead of a binary.

        x_ablated = x - Q (Q^T x),  Q an orthonormal basis of the subspace

    The basis is orthonormalized first, because the leading difference directions
    are not orthogonal and subtracting them one by one would over-subtract their
    shared component.
    """
    orthonormal, _ = torch.linalg.qr(basis.T)

    def hook(module, inputs, output):
        coefficients = output @ orthonormal
        return output - coefficients @ orthonormal.T

    return hook


def difference_basis(
    clean: torch.Tensor, triggered: torch.Tensor, rank: int
) -> torch.Tensor:
    """A rank-r basis for the clean-to-triggered difference, mean direction first.

    Row 0 is the mean difference, so rank 1 reproduces the rank-1 ablation exactly
    and the sweep is a strict extension of it. Rows 1 onward are the leading
    principal directions of the *centered* difference, which is the variation the
    mean does not capture. If a backdoor is a pure common shift, those rows carry
    nothing and rank r behaves like rank 1.
    """
    difference = triggered - clean
    mean_direction = difference.mean(dim=0, keepdim=True)
    if rank == 1:
        return mean_direction
    centered = difference - mean_direction
    _, _, components = torch.linalg.svd(centered, full_matrices=False)
    return torch.cat([mean_direction, components[: rank - 1]])


def make_direction_hook(direction: torch.Tensor):
    """Remove the component along the backdoor direction, keeping everything else.

    This is the ablation the H16 claim actually implies. TAC ranks dimensions in the
    standard basis, but a linear direction in the residual stream need not be
    axis-aligned, and its energy can be spread thinly over hundreds of coordinates.
    Deleting the k largest coordinates then leaves most of the direction intact,
    which is exactly what deleting the top 20 was observed to do.

        x_ablated = x - (x dot unit_direction) * unit_direction
    """
    unit = direction / direction.norm().clamp_min(1e-8)

    def hook(module, inputs, output):
        component = (output * unit).sum(dim=-1, keepdim=True)
        return output - component * unit

    return hook


def peak_layer_signals(model, metadata, attack, peak, args, device):
    """Per-dimension TAC and the backdoor direction at the peak layer, paired inputs."""
    clean_loader, triggered_loader, _ = build_paired_loaders(
        metadata["dataset"],
        attack,
        args.raw_data_dir,
        args.batch_size,
        args.tac_samples,
        args.seed,
    )
    seed_everything(args.seed)
    clean = extract_layer_features(model, clean_loader, device, False, "cls")[peak]
    seed_everything(args.seed)
    triggered = extract_layer_features(model, triggered_loader, device, False, "cls")[
        peak
    ]
    # In post-LN mode the direction has to live in the same space as the hook, so
    # apply the final LayerNorm here too. It is per-token, so running it on the CLS
    # row alone matches running it on the full sequence.
    if args.post_ln:
        layer_norm = vit_core(model).encoder.ln
        with torch.inference_mode():
            clean = layer_norm(clean.to(device)).float()
            triggered = layer_norm(triggered.to(device)).float()

    clean, triggered = clean.cpu(), triggered.cpu()
    return (
        trigger_activated_change(clean, triggered),
        backdoor_direction(clean, triggered),
        clean,
        triggered,
    )


def evaluate(model, loaders, device) -> tuple[float, float]:
    # fp32 throughout: the ablation zeroes 20 of 768 dimensions and the effect on ASR
    # is the measurement, so it must not compete with a bfloat16 noise floor.
    return (
        attack_success_rate(model, loaders["backdoor"], device, False),
        clean_accuracy(model, loaders["clean"], device, False),
    )


def ablate(folder: str, args: argparse.Namespace, device) -> dict | None:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    report_path = os.path.join(args.results_dir, folder, "backdoor_neurons.json")
    if not (os.path.exists(path) and os.path.exists(report_path)):
        return None
    with open(report_path) as handle:
        peak = args.layer or json.load(handle)["peak_layer"]

    metadata = read_checkpoint_metadata(path)
    benign = metadata["attack"] == "benign"
    attack_name, target_label = resolve_probe_attack(
        metadata, "badnet_a2o" if benign else None, 0 if benign else None
    )
    spec = DATASET_REGISTRY[metadata["dataset"]]
    attack = build_attack(
        attack_name, default_config(attack_name), spec.image_size, target_label
    )

    loaders, _ = build_psbd_loaders_from_checkpoint(
        path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        probe_attack="badnet_a2o" if benign else None,
        probe_target_label=0 if benign else None,
    )
    model = load_checkpoint(metadata["architecture"], path, device)
    core = vit_core(model)
    block = core.encoder.ln if args.post_ln else list(core.encoder.layers)[peak - 1]

    seed_everything(args.seed)
    base_asr, base_accuracy = evaluate(model, loaders, device)

    # TAC is recomputed here rather than read from backdoor_neurons.json, because the
    # bottom-of-ranking control needs the full vector and the report stores only the
    # top-k. Same layer, same estimator, so "top" reproduces the report's set.
    tac, direction, clean, triggered = peak_layer_signals(
        model, metadata, attack, peak, args, device
    )
    width = tac.numel()
    order = tac.argsort(descending=True)
    generator = torch.Generator().manual_seed(args.seed)

    variants = {"direction": make_direction_hook(direction.to(device))}
    # The control the direction ablation needs: removing SOME rank-1 direction is
    # not the claim, removing THIS one is. Random directions and the mean clean
    # feature direction are both rank-1 removals of comparable form.
    for index in range(args.random_directions):
        random_direction = torch.randn(width, generator=generator)
        variants[f"random_dir_{index}"] = make_direction_hook(
            random_direction.to(device)
        )
    for rank in args.rank:
        variants[f"rank_{rank}"] = make_subspace_hook(
            difference_basis(clean, triggered, rank).to(device)
        )
    # A random subspace of the LARGEST rank, because that is where "you removed 16
    # dimensions of variance, of course it broke" is the most plausible alternative
    # explanation. The rank-1 case already has its random_dir controls.
    if args.rank:
        largest = max(args.rank)
        variants[f"rank_{largest}_random"] = make_subspace_hook(
            torch.randn(largest, width, generator=generator).to(device)
        )
    for k in args.top_k:
        variants[f"top_{k}"] = make_ablation_hook(order[:k].to(device))
        variants[f"bottom_{k}"] = make_ablation_hook(order[-k:].to(device))
        variants[f"random_{k}"] = make_ablation_hook(
            torch.randperm(width, generator=generator)[:k].to(device)
        )

    results = {}
    for name, hook in variants.items():
        handle = block.register_forward_hook(hook)
        try:
            seed_everything(args.seed)
            asr, accuracy = evaluate(model, loaders, device)
        finally:
            handle.remove()
        results[name] = {"asr": asr, "clean_accuracy": accuracy}

    return {
        "folder_name": folder,
        "attack": metadata["attack"],
        "peak_layer": peak,
        "top_k": args.top_k,
        "baseline": {"asr": base_asr, "clean_accuracy": base_accuracy},
        "ablated": results,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Ablating at each attack's peak layer. ASR must collapse and clean accuracy")
    print("hold, and the top-k must beat its bottom-k and random-k controls.\n")
    columns = (
        ["direction"]
        + [f"random_dir_{i}" for i in range(args.random_directions)]
        + [f"rank_{r}" for r in args.rank]
        + ([f"rank_{max(args.rank)}_random"] if args.rank else [])
        + [f"{side}_{k}" for k in args.top_k for side in ("top", "bottom", "random")]
    )
    header = f"{'checkpoint':30} {'peak':>5} {'base':>12} " + " ".join(
        f"{name:>13}" for name in columns
    )
    print(header)
    print("-" * len(header))

    rows = []
    for attack in args.attack:
        for rho in args.rho:
            stem = f"{args.architecture}_{args.dataset}_{attack}"
            if attack != "benign":
                stem += f"_{args.poison_tag}"
            folder = stem if rho == "" else f"{stem}_sam_rho_{rho}"
            try:
                row = ablate(folder, args, device)
            except Exception as error:
                print(f"{folder:30} FAILED {type(error).__name__}: {error}", flush=True)
                continue
            if row is None:
                continue
            rows.append(row)
            base = row["baseline"]
            cells = " ".join(
                f"{row['ablated'][name]['asr']:>6.3f}/"
                f"{row['ablated'][name]['clean_accuracy']:<6.3f}"
                for name in columns
            )
            print(
                f"{folder:30} {row['peak_layer']:>5} "
                f"{base['asr']:>5.3f}/{base['clean_accuracy']:<6.3f} {cells}",
                flush=True,
            )

    if rows:
        out = os.path.join(args.results_dir, "backdoor_neuron_ablation.json")
        with open(out, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
