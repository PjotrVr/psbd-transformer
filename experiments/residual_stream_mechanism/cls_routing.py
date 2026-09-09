"""Does attention ROUTE the trigger to the CLS token, and at what depth?

The residual decomposition shows the MLP writes more of the backdoor direction than attention
does, but the MLP is token-wise: it can amplify what a token already holds and can never move
anything between tokens. Only attention can carry a corner patch's content to the CLS token
the classifier reads. So the routing step is the one with no substitute, and it should be
visible as CLS attention mass landing on the trigger's tokens.

That is also the prediction that separates the trigger families. A distributed trigger is
already present in every token, so it needs no routing and reaches CLS early; a patch trigger
must be routed and arrives late. This project's own onset measurements agree (blend 5, bpp 6,
lf 8, badnet 9), and the token-mask advantage tracks that ordering at r = +0.94.

Two things measured here, per layer:

    cls_to_trigger    CLS attention mass on the trigger's tokens, triggered minus the same
                      clean images. Attention mass is a WEIGHT, and a weight alone is not an
                      explanation (Jain and Wallace 2019), so the value-weighted version is
                      reported beside it: the weight times the norm of the value vector it
                      pulls (Kobayashi et al. 2020), which is what actually reaches CLS.
    cls_entropy       entropy of the CLS attention row. A trigger acting as an attention sink
                      should lower it.

The trigger's token set is computed from the attack itself, at the resolution the attack is
applied and then upsampled the way the model upsamples, so it is measured rather than assumed.

    PYTHONPATH=. python experiments/residual_stream_mechanism/cls_routing.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from attacks import apply_config_overrides, build_attack, default_config
from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from models import load_checkpoint, network_core
from utils.config import DATASET_REGISTRY

MODEL_INPUT = 224
PATCH = 16
GRID = MODEL_INPUT // PATCH


def trigger_tokens(metadata, threshold: float = 1e-6) -> torch.Tensor:
    """Which of the 196 patch tokens the trigger actually changes.

    The attack is applied at the dataset's native resolution and the model upsamples to 224,
    so the same upsample is applied here before mapping pixels onto the token grid.
    """
    size = DATASET_REGISTRY[metadata["dataset"]].image_size
    config = apply_config_overrides(
        default_config(metadata["attack"]), metadata.get("attack_config_overrides")
    )
    attack = build_attack(metadata["attack"], config, size, metadata["target_label"])
    plant = attack.apply_trigger_eval or attack.apply_trigger
    base = torch.rand(3, size, size, generator=torch.Generator().manual_seed(0))
    difference = (plant(base.clone(), 0) - base).abs().sum(dim=0, keepdim=True)
    upsampled = F.interpolate(
        difference.unsqueeze(0), size=(MODEL_INPUT, MODEL_INPUT), mode="bilinear"
    )[0, 0]
    per_token = (
        upsampled.reshape(GRID, PATCH, GRID, PATCH)
        .permute(0, 2, 1, 3)
        .reshape(-1, PATCH * PATCH)
    )
    return (per_token.sum(dim=1) > threshold).nonzero(as_tuple=True)[0]


def attach(core, store):
    handles = []
    for index, block in enumerate(core.encoder.layers):
        attention = block.self_attention
        heads = attention.num_heads
        dim = attention.embed_dim // heads

        def hook(module, inputs, _output, index=index, heads=heads, dim=dim):
            x = inputs[0]
            batch, tokens, _ = x.shape
            qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.reshape(batch, tokens, heads, dim).transpose(1, 2)
            k = k.reshape(batch, tokens, heads, dim).transpose(1, 2)
            v = v.reshape(batch, tokens, heads, dim).transpose(1, 2)
            # CLS query only: row 0 of the attention matrix.
            weights = F.softmax(
                (q[:, :, :1] @ k.transpose(-2, -1)) / math.sqrt(dim), dim=-1
            )[:, :, 0]
            store[index] = (weights.detach(), v.norm(dim=-1).detach())

        handles.append(attention.register_forward_hook(hook))
    return handles


@torch.inference_mode()
def run_split(model, core, loader, device, tokens, limit, keep_rows=None):
    totals, count, row = {}, 0, 0
    for images, _ in loader:
        store = {}
        handles = attach(core, store)
        model(images.to(device))
        for handle in handles:
            handle.remove()
        batch = images.shape[0]
        wanted = [
            i for i in range(batch) if keep_rows is None or (row + i) in keep_rows
        ]
        row += batch
        if wanted:
            index = torch.tensor(wanted, dtype=torch.long)
            for layer, (weights, value_norm) in store.items():
                w = weights[index]
                vn = value_norm[index]
                weighted = w * vn
                weighted = weighted / weighted.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                # token 0 is CLS; the trigger set indexes patches, so shift by one
                keys = tokens + 1
                entry = totals.setdefault(layer, {"w": 0.0, "vw": 0.0, "ent": 0.0})
                entry["w"] += float(w[..., keys].sum(dim=-1).mean(dim=-1).sum())
                entry["vw"] += float(weighted[..., keys].sum(dim=-1).mean(dim=-1).sum())
                entry["ent"] += float(
                    (-(w.clamp(min=1e-12).log() * w).sum(dim=-1)).mean(dim=-1).sum()
                )
            count += len(wanted)
        if count >= limit:
            break
    return {
        layer: {key: value / count for key, value in entry.items()}
        for layer, entry in totals.items()
    }, count


def paired_clean_rows(manifest) -> set:
    clean_indices = manifest["analysis_clean_indices"]
    backdoor_indices = set(manifest["analysis_backdoor_indices"])
    return {
        row
        for row, original in enumerate(clean_indices)
        if original in backdoor_indices
    }


def analyse(folder, args) -> dict:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    probe = "badnet_a2o" if metadata["attack"] == "benign" else None
    if probe:
        metadata = {**metadata, "attack": probe, "target_label": 0}
    tokens = trigger_tokens(metadata)
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        max_samples=args.max_samples,
        probe_attack=probe,
        probe_target_label=0 if probe else None,
    )
    device = torch.device(args.device)
    model = load_checkpoint(metadata["architecture"], path, device).eval()
    core = network_core(model)

    backdoor, n_bd = run_split(
        model, core, loaders["backdoor"], device, tokens, args.limit
    )
    clean, _ = run_split(
        model,
        core,
        loaders["clean"],
        device,
        tokens,
        args.limit,
        paired_clean_rows(manifest),
    )
    layers = [
        {
            "layer": layer + 1,
            "weight_backdoor": backdoor[layer]["w"],
            "weight_clean": clean[layer]["w"],
            "weight_delta": backdoor[layer]["w"] - clean[layer]["w"],
            "value_weighted_delta": backdoor[layer]["vw"] - clean[layer]["vw"],
            "entropy_delta": backdoor[layer]["ent"] - clean[layer]["ent"],
        }
        for layer in sorted(backdoor)
    ]
    return {
        "folder_name": folder,
        "attack": metadata["attack"],
        "n_trigger_tokens": int(len(tokens)),
        "trigger_token_share": float(len(tokens) / (GRID * GRID)),
        "n_backdoor": n_bd,
        "layers": layers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=2400)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--device", default="cpu")
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
        with open(os.path.join(out_dir, "cls_routing.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print(
            f"\n[ok] {folder}  trigger occupies {report['n_trigger_tokens']}/196 tokens "
            f"({report['trigger_token_share']:.1%})  n={report['n_backdoor']}"
        )
        print(
            f"{'layer':>5s} {'CLS->trigger (bd)':>18s} {'(clean)':>9s} {'delta':>8s} "
            f"{'value-weighted':>15s} {'entropy delta':>14s}"
        )
        for row in report["layers"]:
            print(
                f"{row['layer']:5d} {row['weight_backdoor']:18.4f} {row['weight_clean']:9.4f} "
                f"{row['weight_delta']:+8.4f} {row['value_weighted_delta']:+15.4f} "
                f"{row['entropy_delta']:+14.4f}"
            )


if __name__ == "__main__":
    main()
