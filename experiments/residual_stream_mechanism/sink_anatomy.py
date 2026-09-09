"""What is the trigger's sink MADE of, and can the network do without it?

Two published tests, run together because they share the machinery and answer halves of one
question.

VALUE NORM (Fesser, Jacobs, Fel, Keller, Kakade, arXiv:2606.08105). Two mechanisms look
identical in an attention heatmap and are not the same thing. An "adaptive NOP" sink has
near-zero value norm: attending to it suppresses the head's contribution, so it is a way of
doing nothing. A "broadcast" sink has a meaningful value norm and redistributes information.
A backdoor trigger should be neither: a targeted broadcast carrying class-specific content,
so its value norm should be LARGE. If instead it sits in the low band with NOP sinks, the
attack is pure attention routing with no payload, which would be a surprise worth reporting.

DORMANT TAKEOVER (Lu, Liao, Wang, Yang, Shi, arXiv:2507.16018). Natural sinks are replaceable:
mask the current one and a dormant token is promoted in its place, because the sink role is a
competition with a queue of candidates. A payload-carrying trigger should NOT be replaceable,
because no other token holds its content. Masking the trigger and watching whether the
attention concentration recovers therefore separates "the trigger won a competition for an
existing role" from "the trigger installed something new".

Reported per layer:

    value_norm_*        mean ||v|| over heads for trigger, top-attention and ordinary tokens
    cls_mass_before     the single most-attended patch token's share of CLS attention
    cls_mass_after      the same after the trigger's tokens are removed and the row
                        renormalised, giving a substitute the chance to take the role
    takeover_fraction   after / before. Near 1 means the concentration was restored by a
                        substitute and the role is replaceable, as it is for a natural sink;
                        well below 1 means it was not

    PYTHONPATH=. python experiments/residual_stream_mechanism/sink_anatomy.py \
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
from utils.numerics import safe_ratio, safe_ratio_positive
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


def attach(core, store):
    handles = []
    for index, block in enumerate(core.encoder.layers):
        attention = block.self_attention
        heads, dim = attention.num_heads, attention.embed_dim // attention.num_heads

        def hook(module, inputs, _output, index=index, heads=heads, dim=dim):
            x = inputs[0]
            batch, tokens, _ = x.shape
            qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            shape = (batch, tokens, heads, dim)
            q = q.reshape(shape).transpose(1, 2)
            k = k.reshape(shape).transpose(1, 2)
            v = v.reshape(shape).transpose(1, 2)
            weights = F.softmax(
                (q[:, :, :1] @ k.transpose(-2, -1)) / math.sqrt(dim), dim=-1
            )[:, :, 0]
            store[index] = (weights.detach(), v.norm(dim=-1).detach())

        handles.append(attention.register_forward_hook(hook))
    return handles


@torch.inference_mode()
def measure(model, core, loader, device, tokens, limit):
    n_layers = len(core.encoder.layers)
    totals = {
        index: {
            k: 0.0
            for k in ("v_trigger", "v_topattn", "v_other", "mass_before", "mass_after")
        }
        for index in range(n_layers)
    }
    seen = 0
    keys = (tokens + 1).to(device)  # token 0 is CLS
    for images, _ in loader:
        store = {}
        handles = attach(core, store)
        model(images.to(device))
        for handle in handles:
            handle.remove()
        for index in range(n_layers):
            weights, value_norm = store[index]  # (B, heads, tokens), (B, heads, tokens)
            entry = totals[index]
            other = torch.ones(
                weights.shape[-1], dtype=torch.bool, device=weights.device
            )
            other[keys] = False
            other[0] = False
            entry["v_trigger"] += float(value_norm[..., keys].mean(dim=(1, 2)).sum())
            entry["v_other"] += float(value_norm[..., other].mean(dim=(1, 2)).sum())
            # the most-attended non-trigger token, per head: the natural sink candidate
            masked_weights = weights.clone()
            masked_weights[..., keys] = -1.0
            top = masked_weights.argmax(dim=-1, keepdim=True)
            entry["v_topattn"] += float(
                value_norm.gather(-1, top).mean(dim=(1, 2)).sum()
            )
            # Concentration, not raw mass: the single most-attended PATCH token's share,
            # before and after the trigger is removed and the row renormalised. Comparing a
            # 4-token sum against a 1-token share would not be a takeover measurement, and
            # averaging over heads on one side while summing on the other is how the first
            # version of this produced ratios of 176.
            patch = weights[..., 1:]
            entry["mass_before"] += float(patch.max(dim=-1).values.mean(dim=1).sum())
            without = weights.clone()
            without[..., keys] = 0.0
            without = without / without.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            entry["mass_after"] += float(
                without[..., 1:].max(dim=-1).values.mean(dim=1).sum()
            )
        seen += images.shape[0]
        if seen >= limit:
            break
    return {
        index: {key: value / seen for key, value in entry.items()}
        for index, entry in totals.items()
    }, seen


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
    triggered, n = measure(model, core, loaders["backdoor"], device, tokens, args.limit)
    clean, _ = measure(model, core, loaders["clean"], device, tokens, args.limit)
    return {
        "folder_name": folder,
        "attack": metadata["attack"],
        "dataset": metadata["dataset"],
        "n_trigger_tokens": int(len(tokens)),
        "n_samples": n,
        "layers": [
            {
                "layer": index + 1,
                "value_norm_trigger": triggered[index]["v_trigger"],
                "value_norm_topattn": triggered[index]["v_topattn"],
                "value_norm_other": triggered[index]["v_other"],
                "value_ratio": safe_ratio_positive(
                    triggered[index]["v_trigger"], triggered[index]["v_other"]
                ),
                "cls_mass_before": triggered[index]["mass_before"],
                "cls_mass_after": triggered[index]["mass_after"],
                "takeover_fraction": safe_ratio_positive(
                    triggered[index]["mass_after"], triggered[index]["mass_before"]
                ),
                "cls_mass_clean": clean[index]["mass_before"],
            }
            for index in sorted(triggered)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=256)
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
        with open(os.path.join(out_dir, "sink_anatomy.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n[ok] {folder} ({report['attack']})  n={report['n_samples']}")
        print(
            f"{'layer':>5s} {'|v| trig':>9s} {'|v| sink':>9s} {'|v| other':>10s} {'ratio':>6s} "
            f"{'CLS mass':>9s} {'after mask':>11s} {'takeover':>9s}"
        )
        for row in report["layers"]:
            print(
                f"{row['layer']:5d} {row['value_norm_trigger']:9.3f} {row['value_norm_topattn']:9.3f} "
                f"{row['value_norm_other']:10.3f} {row['value_ratio']:6.2f} "
                f"{row['cls_mass_before']:9.4f} {row['cls_mass_after']:11.4f} "
                f"{row['takeover_fraction']:9.3f}"
            )


if __name__ == "__main__":
    main()
