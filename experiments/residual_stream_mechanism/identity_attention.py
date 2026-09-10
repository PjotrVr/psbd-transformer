"""The causal test of token mixing: switch attention off, one layer at a time.

The descriptive decomposition says the MLP writes more of the backdoor direction than
attention does, yet the MLP is token-wise and cannot move anything between tokens. If a
corner-patch trigger has to be ROUTED from its own patches to the CLS token the classifier
reads, then attention at the routing depth is the step with no substitute, and removing it
should destroy the backdoor while leaving clean accuracy comparatively intact.

The intervention is surgical. Attention is replaced by the IDENTITY pattern at one layer:
every token attends only to itself, so the value and output projections, every MLP, the
residual stream and every other layer's attention are untouched, and the ONLY thing removed
is cross-token movement at that layer. Formally the sublayer becomes

    a^l_j = W_O W_V LN_1(h^l)_j + b        instead of      a^l_j = W_O sum_k alpha_jk W_V LN_1(h^l)_k

Sweeping the layer gives a causal depth profile that no correlational measure can give. A
perturbation-based placement study is exactly a coarse, stochastic version of this
intervention, so where this collapses is where a perturbation should be injected.

Both metrics are reported at every layer, because an intervention that destroys the model
destroys the backdoor too and proves nothing:

    asr             attack success rate with the layer's attention identity-ablated
    clean_accuracy  clean accuracy under the SAME intervention, the control

    PYTHONPATH=. python experiments/residual_stream_mechanism/identity_attention.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from data.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from models.backbones import load_checkpoint, network_core


class IdentityAttention(torch.nn.Module):
    """`self_attention` with the attention pattern forced to the identity.

    Wraps rather than edits: the original module is kept and restored by `unpatch`, so no
    weight is touched and the model is bit-identical afterwards.
    """

    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, query, key, value, need_weights=False, **kwargs):
        weight = self.attention.in_proj_weight
        bias = self.attention.in_proj_bias
        dim = self.attention.embed_dim
        # Only the value projection is needed: with an identity pattern each token's output
        # is its own value vector, so queries and keys never enter.
        values = F.linear(query, weight[2 * dim :], bias[2 * dim :])
        return self.attention.out_proj(values), None


def patch_layer(core, index):
    block = core.encoder.layers[index]
    original = block.self_attention
    block.self_attention = IdentityAttention(original)
    return lambda: setattr(block, "self_attention", original)


@torch.inference_mode()
def accuracy(model, loader, device, limit):
    correct = total = 0
    for images, labels in loader:
        predicted = model(images.to(device)).argmax(dim=1).cpu()
        correct += int((predicted == labels).sum())
        total += len(labels)
        if total >= limit:
            break
    return correct / total if total else float("nan")


def analyse(folder, args) -> dict:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    probe = "badnet_a2o" if metadata["attack"] == "benign" else None
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

    baseline = {
        "asr": accuracy(model, loaders["backdoor"], device, args.limit),
        "clean_accuracy": accuracy(model, loaders["clean"], device, args.limit),
    }
    rows = []
    for index in range(len(core.encoder.layers)):
        restore = patch_layer(core, index)
        try:
            rows.append(
                {
                    "layer": index + 1,
                    "asr": accuracy(model, loaders["backdoor"], device, args.limit),
                    "clean_accuracy": accuracy(
                        model, loaders["clean"], device, args.limit
                    ),
                }
            )
        finally:
            restore()
    for row in rows:
        row["asr_drop"] = baseline["asr"] - row["asr"]
        row["clean_drop"] = baseline["clean_accuracy"] - row["clean_accuracy"]
        # An intervention that wrecks the model wrecks the backdoor with it. Only a drop
        # that exceeds the clean cost is evidence about the backdoor specifically.
        row["selectivity"] = row["asr_drop"] - row["clean_drop"]
    return {
        "folder_name": folder,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "poison_rate": metadata.get("poison_rate"),
        "baseline": baseline,
        "layers": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=2000)
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
        with open(os.path.join(out_dir, "identity_attention.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        base = report["baseline"]
        print(
            f"\n[ok] {folder}  baseline ASR {base['asr']:.3f}  CA {base['clean_accuracy']:.3f}"
        )
        print(
            f"{'layer':>5s} {'ASR':>7s} {'drop':>7s} {'CA':>7s} {'drop':>7s} {'selectivity':>12s}"
        )
        for row in report["layers"]:
            print(
                f"{row['layer']:5d} {row['asr']:7.3f} {row['asr_drop']:+7.3f} "
                f"{row['clean_accuracy']:7.3f} {row['clean_drop']:+7.3f} {row['selectivity']:+12.3f}"
            )


if __name__ == "__main__":
    main()
