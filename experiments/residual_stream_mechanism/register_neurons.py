"""Does the trigger RECRUIT the neurons that already drive high-norm tokens, or build its own?

A sparse set of MLP neurons drives the high-norm outlier tokens that ViTs produce
spontaneously (Jiang, Dravid, Efros, Gandelsman, arXiv:2506.08010, who find 10 such neurons
of ~770,000 in OpenCLIP ViT-B/16, with outliers emerging right after the layer-6 MLP). If the
backdoor's trigger tokens are driven by the SAME neurons, the attack is hijacking machinery
the network already had. If by a DISJOINT set, it installed a parallel mechanism.

The two answers imply different defences, which is why this is worth the measurement: a hijack
is mitigable by the register interventions already published, a parallel mechanism is not.

Method, following their Algorithm 1 but with a rank-based definition of "outlier position"
rather than a norm threshold. This ViT has no natural outlier population to threshold, its
patch-norm distribution is unimodal with max only 1.4x the median, so a fixed cutoff selects
nothing and the ranking is what carries the information.

    clean pass       score each of the 12 x 3072 neurons by mean post-GELU activation at the
                     top-`k` highest-norm patch tokens minus its mean at the rest. The top
                     scorers are this model's register neurons.
    triggered pass   score the same neurons at the TRIGGER's tokens the same way.
    overlap          how much the two top-`n` sets share, against the hypergeometric
                     expectation for two random sets of that size.

    PYTHONPATH=. python experiments/residual_stream_mechanism/register_neurons.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from attacks import apply_config_overrides, build_attack, default_config
from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from utils.numerics import safe_ratio_positive
from models import load_checkpoint, network_core
from utils.config import DATASET_REGISTRY

MODEL_INPUT, PATCH = 224, 16
GRID = MODEL_INPUT // PATCH


def trigger_tokens(metadata) -> torch.Tensor:
    size = DATASET_REGISTRY[metadata["dataset"]].image_size
    config = apply_config_overrides(
        default_config(metadata["attack"]), metadata.get("attack_config_overrides")
    )
    attack = build_attack(metadata["attack"], config, size, metadata["target_label"])
    plant = attack.apply_trigger_eval or attack.apply_trigger
    base = torch.rand(3, size, size, generator=torch.Generator().manual_seed(0))
    delta = (plant(base.clone(), 0) - base).abs().sum(dim=0, keepdim=True)
    up = F.interpolate(
        delta.unsqueeze(0), size=(MODEL_INPUT, MODEL_INPUT), mode="bilinear"
    )[0, 0]
    per_token = (
        up.reshape(GRID, PATCH, GRID, PATCH)
        .permute(0, 2, 1, 3)
        .reshape(-1, PATCH * PATCH)
    )
    return (per_token.sum(dim=1) > 1e-6).nonzero(as_tuple=True)[0]


@torch.inference_mode()
def neuron_scores(model, core, loader, device, limit, top_k, fixed_tokens=None):
    """Mean post-GELU activation at selected patch positions minus at the others.

    `fixed_tokens` selects the trigger's positions; when None the positions are chosen per
    image as the `top_k` highest-norm patch tokens, which is the register definition.
    """
    n_layers = len(core.encoder.layers)
    hidden, stream = {}, {}
    handles = []
    for index, block in enumerate(core.encoder.layers):
        handles.append(
            block.mlp[3].register_forward_pre_hook(
                lambda _m, inputs, index=index: hidden.__setitem__(
                    index, inputs[0].detach()
                )
            )
        )
        handles.append(
            block.register_forward_hook(
                lambda _m, _i, out, index=index: stream.__setitem__(index, out.detach())
            )
        )
    totals = {index: None for index in range(n_layers)}
    seen = 0
    for images, _ in loader:
        model(images.to(device))
        for index in range(n_layers):
            activation = hidden[index][:, 1:, :].float()  # patch tokens, (B, 196, 3072)
            if fixed_tokens is not None:
                mask = torch.zeros(
                    activation.shape[:2], dtype=torch.bool, device=activation.device
                )
                mask[:, fixed_tokens] = True
            else:
                norms = stream[index][:, 1:, :].float().norm(dim=-1)
                chosen = norms.topk(top_k, dim=-1).indices
                mask = torch.zeros_like(norms, dtype=torch.bool)
                mask.scatter_(1, chosen, True)
            # Both counts can be zero: a trigger set can be empty, and top_k could in
            # principle cover every patch. safe_ratio_positive yields NaN there rather than
            # an inf that would poison the running total.
            selected = safe_ratio_positive(
                (activation * mask.unsqueeze(-1)).sum(dim=1),
                mask.sum(dim=1, keepdim=True),
            )
            other = safe_ratio_positive(
                (activation * (~mask).unsqueeze(-1)).sum(dim=1),
                (~mask).sum(dim=1, keepdim=True),
            )
            delta = torch.nan_to_num(selected - other, nan=0.0).sum(dim=0)
            totals[index] = delta if totals[index] is None else totals[index] + delta
        seen += images.shape[0]
        if seen >= limit:
            break
    for handle in handles:
        handle.remove()
    return torch.stack(
        [totals[index] / seen for index in range(n_layers)]
    )  # (layers, 3072)


def overlap_report(clean_scores, trigger_scores, n_top):
    """Top-n neuron sets and their overlap, against the random expectation."""
    total = clean_scores.numel()
    clean_top = set(clean_scores.flatten().topk(n_top).indices.tolist())
    trigger_top = set(trigger_scores.flatten().topk(n_top).indices.tolist())
    shared = clean_top & trigger_top
    expected = n_top * n_top / total
    return {
        "n_top": n_top,
        "n_neurons": total,
        "overlap": len(shared),
        "expected_by_chance": expected,
        "enrichment": len(shared) / expected if expected else float("nan"),
        "jaccard": len(shared) / len(clean_top | trigger_top),
        "shared_layers": sorted(
            {index // clean_scores.shape[1] + 1 for index in shared}
        ),
    }


def analyse(folder, args) -> dict:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    probe = "badnet_a2o" if metadata["attack"] == "benign" else None
    resolved = {**metadata, "attack": probe, "target_label": 0} if probe else metadata
    tokens = trigger_tokens(resolved)
    loaders, _ = build_psbd_loaders_from_checkpoint(
        path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        probe_attack=probe,
        probe_target_label=0 if probe else None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(metadata["architecture"], path, device).eval()
    core = network_core(model)

    clean = neuron_scores(model, core, loaders["clean"], device, args.limit, args.top_k)
    triggered = neuron_scores(
        model,
        core,
        loaders["backdoor"],
        device,
        args.limit,
        args.top_k,
        tokens.to(device),
    )
    report = {
        "folder_name": folder,
        "attack": metadata["attack"],
        "dataset": metadata["dataset"],
        "n_trigger_tokens": int(len(tokens)),
        "top_k_positions": args.top_k,
        "overlap": [overlap_report(clean, triggered, n) for n in args.n_top],
        "clean_top_layers": (clean.max(dim=1).values).tolist(),
        "trigger_top_layers": (triggered.max(dim=1).values).tolist(),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument(
        "--top-k", type=int, default=4, help="patch positions defining an outlier"
    )
    parser.add_argument("--n-top", nargs="+", type=int, default=[10, 50, 200])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for folder in args.checkpoint_folder:
        try:
            report = analyse(folder, args)
        except Exception as error:  # noqa: BLE001
            print(f"[FAILED] {folder}: {type(error).__name__}: {error}", flush=True)
            continue
        out_dir = os.path.join(args.results_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "register_neurons.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n[ok] {folder}  ({report['attack']})")
        print(
            f"{'top-n':>7s} {'overlap':>8s} {'by chance':>10s} {'enrichment':>11s} {'jaccard':>8s}  layers"
        )
        for row in report["overlap"]:
            print(
                f"{row['n_top']:7d} {row['overlap']:8d} {row['expected_by_chance']:10.2f} "
                f"{row['enrichment']:11.1f} {row['jaccard']:8.3f}  {row['shared_layers']}"
            )
        peak_clean = max(range(12), key=lambda i: report["clean_top_layers"][i]) + 1
        peak_trig = max(range(12), key=lambda i: report["trigger_top_layers"][i]) + 1
        print(
            f"     strongest register-neuron layer: clean {peak_clean}, triggered {peak_trig}"
        )


if __name__ == "__main__":
    main()
