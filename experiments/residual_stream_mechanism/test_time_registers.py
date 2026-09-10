"""If the model is given somewhere else to put a sink, does it stop using the trigger?

The trigger wins an attention competition it should not win: on triggered inputs the CLS token
spends up to 93% of its attention on 4 of 196 patches. Jiang, Dravid, Efros and Gandelsman
(arXiv:2506.08010) showed that a ViT's high-norm outliers can be moved off the patch grid at
TEST time by giving the network a spare token to put them in, with no retraining. If the
backdoor is exploiting the absence of such a token, spare capacity should weaken it.

That makes this simultaneously a mechanism test and a candidate defence, which is why it is
worth running even though the prior is against it: this ViT has no native register population
(patch norms are unimodal, 0.000% above twice the median), and the trigger's neurons are
disjoint from the ones driving natural high-norm tokens, so there may be no competition to
redirect. A null result here is informative and cheap.

Spare tokens are appended to the sequence at the embedding output, initialised to the mean
patch embedding so they carry no content of their own, and they participate in attention from
then on. Nothing is trained and no weight changes.

Reported against the unmodified model:

    asr, clean_accuracy   the defence's actual cost and benefit
    trigger_attention     CLS attention mass on the trigger's tokens, per layer, which is the
                          mechanism claim: if spare capacity absorbs the sink, this falls

    PYTHONPATH=. python experiments/residual_stream_mechanism/test_time_registers.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from attacks import apply_config_overrides, build_attack, default_config
from data.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from models.backbones import load_checkpoint, network_core
from data.registry import DATASET_REGISTRY

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


def attach_registers(core, count):
    """Append `count` spare tokens after the embedding, before the block stack.

    They are initialised to the mean patch embedding so they introduce no content, and are
    dropped again before the readout, which reads token 0 only, so the classifier never sees
    them directly and any effect is through attention alone.
    """
    if count == 0:
        return lambda: None

    def hook(_module, inputs):
        x = inputs[0]
        spare = x[:, 1:, :].mean(dim=1, keepdim=True).expand(-1, count, -1)
        return (torch.cat([x, spare], dim=1),) + inputs[1:]

    handle = core.encoder.layers.register_forward_pre_hook(hook)
    return handle.remove


@torch.inference_mode()
def evaluate(model, loader, device, limit):
    correct = total = 0
    for images, labels in loader:
        predicted = model(images.to(device)).argmax(dim=1).cpu()
        correct += int((predicted == labels).sum())
        total += len(labels)
        if total >= limit:
            break
    return correct / total if total else float("nan")


@torch.inference_mode()
def trigger_attention(model, core, loader, device, tokens, limit):
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
    totals, seen = {}, 0
    keys = (tokens + 1).to(device)
    for images, _ in loader:
        model(images.to(device))
        for index, weights in store.items():
            totals[index] = float(
                weights[..., keys].sum(dim=-1).mean(dim=-1).sum()
            ) + totals.get(index, 0.0)
        seen += images.shape[0]
        if seen >= limit:
            break
    for handle in handles:
        handle.remove()
    return {index: value / seen for index, value in totals.items()}


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

    rows = []
    for count in args.registers:
        remove = attach_registers(core, count)
        try:
            rows.append(
                {
                    "registers": count,
                    "asr": evaluate(model, loaders["backdoor"], device, args.limit),
                    "clean_accuracy": evaluate(
                        model, loaders["clean"], device, args.limit
                    ),
                    "trigger_attention": trigger_attention(
                        model, core, loaders["backdoor"], device, tokens, args.limit
                    ),
                }
            )
        finally:
            remove()
    base = rows[0]
    for row in rows:
        row["asr_drop"] = base["asr"] - row["asr"]
        row["clean_drop"] = base["clean_accuracy"] - row["clean_accuracy"]
        row["selectivity"] = row["asr_drop"] - row["clean_drop"]
    return {
        "folder_name": folder,
        "attack": metadata["attack"],
        "dataset": metadata["dataset"],
        "n_trigger_tokens": int(len(tokens)),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--registers", nargs="+", type=int, default=[0, 1, 4, 16])
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
        with open(os.path.join(out_dir, "test_time_registers.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n[ok] {folder} ({report['attack']})")
        print(
            f"{'registers':>10s} {'ASR':>7s} {'drop':>7s} {'CA':>7s} {'drop':>7s} "
            f"{'select':>7s} {'CLS->trig L6':>13s} {'L12':>7s}"
        )
        for row in report["rows"]:
            attn = row["trigger_attention"]
            print(
                f"{row['registers']:10d} {row['asr']:7.3f} {row['asr_drop']:+7.3f} "
                f"{row['clean_accuracy']:7.3f} {row['clean_drop']:+7.3f} "
                f"{row['selectivity']:+7.3f} {attn.get(5, float('nan')):13.4f} "
                f"{attn.get(11, float('nan')):7.4f}"
            )


if __name__ == "__main__":
    main()
