"""Does the trigger's attention share depend on how many tokens there are?

Every attention number reported for this model is at 224 pixels, so 196 patch tokens. That is
the low end of the range where sinks have been studied: CLIP ViT-L/14 runs at 257 tokens,
DINOv2 at 257 or 1370, LLMs at thousands. The softmax-normalisation account of why sinks form
says a query must place its probability mass somewhere, which makes sink strength a function of
how many candidates there are. If so, "the CLS token spends 93% of its attention on 4 tokens"
is partly a statement about sequence length and not only about the backdoor.

No published study sweeps input resolution at fixed weights and reports outlier statistics, so
this is measured rather than assumed. Positional embeddings are interpolated with torchvision's
own `interpolate_embeddings`, the same routine used for fine-tuning at a new resolution, so no
weight is retrained and the only thing that changes is the token count.

Reported per resolution:

    trigger_share       CLS attention mass on the trigger's tokens, per layer
    trigger_share_norm  the same divided by the share a uniform row would give the same
                        tokens, which is what makes counts comparable across sequence lengths
    max_norm_ratio      max over median patch-token norm, the outlier statistic
    asr, clean_accuracy the model still has to work at the new resolution for any of it to mean
                        anything

    PYTHONPATH=. python experiments/residual_stream_mechanism/resolution_sweep.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from torchvision.models.vision_transformer import interpolate_embeddings
from torchvision.transforms import v2 as transforms_v2

from attacks import apply_config_overrides, build_attack, default_config
from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from models import build_vit, load_checkpoint, network_core
from utils.config import DATASET_REGISTRY

PATCH = 16


def trigger_tokens(metadata, model_input: int) -> torch.Tensor:
    """Patch tokens the trigger changes, at a given model input resolution."""
    grid = model_input // PATCH
    size = DATASET_REGISTRY[metadata["dataset"]].image_size
    config = apply_config_overrides(
        default_config(metadata["attack"]), metadata.get("attack_config_overrides")
    )
    attack = build_attack(metadata["attack"], config, size, metadata["target_label"])
    plant = attack.apply_trigger_eval or attack.apply_trigger
    base = torch.rand(3, size, size, generator=torch.Generator().manual_seed(0))
    delta = (plant(base.clone(), 0) - base).abs().sum(dim=0, keepdim=True)
    up = F.interpolate(
        delta.unsqueeze(0), size=(model_input, model_input), mode="bilinear"
    )[0, 0]
    per_token = (
        up.reshape(grid, PATCH, grid, PATCH)
        .permute(0, 2, 1, 3)
        .reshape(-1, PATCH * PATCH)
    )
    return (per_token.sum(dim=1) > 1e-6).nonzero(as_tuple=True)[0]


def rebuild_at(model, num_classes, model_input, device):
    """The same weights, serving a different input resolution.

    Only the positional embedding changes, by torchvision's own interpolation, and the Resize
    in front of the network is retargeted. Everything else is the trained checkpoint.
    """
    core = network_core(model)
    state = interpolate_embeddings(model_input, PATCH, core.state_dict())
    fresh = build_vit(num_classes)
    inner = network_core(fresh)
    # build_vit fixes image_size at 224, so the positional embedding parameter has to be
    # reshaped to the interpolated one before the state dict will load.
    inner.image_size = model_input
    inner.encoder.pos_embedding = torch.nn.Parameter(
        torch.empty_like(state["encoder.pos_embedding"])
    )
    inner.load_state_dict(state)
    wrapped = torch.nn.Sequential(
        transforms_v2.Resize((model_input, model_input), antialias=True), inner
    ).to(device)
    return wrapped.eval()


@torch.inference_mode()
def measure(model, core, loader, device, tokens, limit):
    store = {}
    handles = []
    for index, block in enumerate(core.encoder.layers):
        attention = block.self_attention
        heads, dim = attention.num_heads, attention.embed_dim // attention.num_heads

        def hook(module, inputs, _out, index=index, heads=heads, dim=dim):
            x = inputs[0]
            batch, count, _ = x.shape
            qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
            q, k, _ = qkv.chunk(3, dim=-1)
            q = q.reshape(batch, count, heads, dim).transpose(1, 2)
            k = k.reshape(batch, count, heads, dim).transpose(1, 2)
            store[index] = F.softmax(
                (q[:, :, :1] @ k.transpose(-2, -1)) / math.sqrt(dim), dim=-1
            )[:, :, 0].detach()

        handles.append(attention.register_forward_hook(hook))
    norms = {}
    handles += [
        core.encoder.layers[i].register_forward_hook(
            lambda _m, _i, out, i=i: norms.__setitem__(
                i, out[:, 1:, :].detach().norm(dim=-1)
            )
        )
        for i in range(len(core.encoder.layers))
    ]
    share, ratio, seen = {}, {}, 0
    keys = (tokens + 1).to(device)
    for images, _ in loader:
        model(images.to(device))
        for index, weights in store.items():
            share[index] = float(
                weights[..., keys].sum(dim=-1).mean(dim=-1).sum()
            ) + share.get(index, 0.0)
            n = norms[index]
            ratio[index] = float(
                (n.max(dim=-1).values / n.median(dim=-1).values).sum()
            ) + ratio.get(index, 0.0)
        seen += images.shape[0]
        if seen >= limit:
            break
    for handle in handles:
        handle.remove()
    return (
        {i: v / seen for i, v in share.items()},
        {i: v / seen for i, v in ratio.items()},
        seen,
    )


@torch.inference_mode()
def accuracy(model, loader, device, limit):
    correct = total = 0
    for images, labels in loader:
        correct += int((model(images.to(device)).argmax(dim=1).cpu() == labels).sum())
        total += len(labels)
        if total >= limit:
            break
    return correct / total if total else float("nan")


def analyse(folder, args) -> dict:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    if metadata["attack"] == "benign":
        raise ValueError("needs a real trigger")
    loaders, _ = build_psbd_loaders_from_checkpoint(
        path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_checkpoint(metadata["architecture"], path, device).eval()
    num_classes = DATASET_REGISTRY[metadata["dataset"]].num_classes

    rows = []
    for model_input in args.resolutions:
        model = (
            base
            if model_input == 224
            else rebuild_at(base, num_classes, model_input, device)
        )
        core = network_core(model)
        tokens = trigger_tokens(metadata, model_input)
        n_patches = (model_input // PATCH) ** 2
        share, ratio, n = measure(
            model, core, loaders["backdoor"], device, tokens, args.limit
        )
        uniform = len(tokens) / (n_patches + 1)
        rows.append(
            {
                "resolution": model_input,
                "n_patch_tokens": n_patches,
                "n_trigger_tokens": int(len(tokens)),
                "uniform_share": uniform,
                "asr": accuracy(model, loaders["backdoor"], device, args.limit),
                "clean_accuracy": accuracy(model, loaders["clean"], device, args.limit),
                "trigger_share": {k + 1: v for k, v in share.items()},
                "trigger_share_normalised": {
                    k + 1: v / uniform for k, v in share.items()
                },
                "max_norm_ratio": {k + 1: v for k, v in ratio.items()},
                "n_samples": n,
            }
        )
    return {
        "folder_name": folder,
        "attack": metadata["attack"],
        "dataset": metadata["dataset"],
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument(
        "--resolutions", nargs="+", type=int, default=[160, 224, 320, 448]
    )
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
        with open(os.path.join(out_dir, "resolution_sweep.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n[ok] {folder} ({report['attack']})")
        print(
            f"{'res':>5s} {'tokens':>7s} {'trig':>5s} {'ASR':>6s} {'CA':>6s} "
            f"{'share L6':>9s} {'L12':>7s} {'norm x uniform L12':>19s} {'max/med L6':>11s}"
        )
        for row in report["rows"]:
            s = row["trigger_share"]
            sn = row["trigger_share_normalised"]
            r = row["max_norm_ratio"]
            print(
                f"{row['resolution']:5d} {row['n_patch_tokens']:7d} {row['n_trigger_tokens']:5d} "
                f"{row['asr']:6.3f} {row['clean_accuracy']:6.3f} "
                f"{s.get(6, float('nan')):9.4f} {s.get(12, float('nan')):7.4f} "
                f"{sn.get(12, float('nan')):19.1f} {r.get(6, float('nan')):11.2f}"
            )


if __name__ == "__main__":
    main()
