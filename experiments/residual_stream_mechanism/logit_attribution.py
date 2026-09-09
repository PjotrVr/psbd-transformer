"""Direct logit attribution: which sublayer pushes the prediction to the attacker's class?

The residual decomposition says where the backdoor DIRECTION is written. This says what that
writing does to the ANSWER, which is the quantity a detector ultimately reads.

ViT's readout is `logits = head(ln(h_final)[CLS])`, and h_final is the sum of the embedding
and every sublayer write. LayerNorm is not linear, so the sum does not pass through it. The
standard fix (Elhage et al., A Mathematical Framework for Transformer Circuits, 2021) is to
FREEZE the normalisation: compute the per-token scale on the ACTUAL final residual, then treat
it as a constant. LayerNorm is then affine and attribution is exact and additive:

    logit_c - b_c - w_c . beta = sum over components of  w_c . ( (r - mean(r)) / s * gamma )

Freezing matters here beyond bookkeeping. This project has already produced one clean,
monotone, benign-controlled and entirely false result by measuring a scale-dependent statistic
across a LayerNorm, so an attribution that let the scale float would be the same mistake.

Reported per layer, for the attention and MLP writes separately:

    target_push        contribution to the attacker's target-class logit
    delta_push         that contribution on triggered inputs MINUS on the same clean images.
                       This is the backdoor's own doing, with the clean computation removed.

A benign checkpoint is the control: with no backdoor, delta_push must be flat and near zero.

    PYTHONPATH=. python experiments/residual_stream_mechanism/logit_attribution.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import os

import torch

from data.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from models.backbones import load_checkpoint, network_core


def collect_components(model, core, images):
    """Every additive contribution to the final residual stream, plus the stream itself."""
    store = {}
    handles = []
    embedding = {}
    handles.append(
        core.encoder.layers.register_forward_pre_hook(
            lambda _m, inputs: embedding.__setitem__("x", inputs[0].detach().float())
        )
    )
    for index, block in enumerate(core.encoder.layers):
        handles.append(
            block.dropout.register_forward_hook(
                lambda _m, _i, out, index=index: store.__setitem__(
                    ("attention_write", index), out.detach().float()
                )
            )
        )
        handles.append(
            block.mlp.register_forward_hook(
                lambda _m, _i, out, index=index: store.__setitem__(
                    ("mlp_write", index), out.detach().float()
                )
            )
        )
    final = {}
    handles.append(
        core.encoder.ln.register_forward_pre_hook(
            lambda _m, inputs: final.__setitem__("x", inputs[0].detach().float())
        )
    )
    with torch.inference_mode():
        model(images)
    for handle in handles:
        handle.remove()
    return embedding["x"], store, final["x"]


def attribute(component, final, layer_norm, weight_row):
    """Contribution of one residual component to one class logit, at the CLS token.

    The scale is taken from `final`, the actual residual the readout sees, and held fixed, so
    the components sum to the true logit up to the readout bias.
    """
    cls_component = component[:, 0, :]
    cls_final = final[:, 0, :]
    variance = cls_final.var(dim=-1, keepdim=True, unbiased=False)
    scale = torch.sqrt(variance + layer_norm.eps)
    centred = cls_component - cls_component.mean(dim=-1, keepdim=True)
    normalised = centred / scale * layer_norm.weight
    return normalised @ weight_row


def run_split(model, core, loader, device, target_label, limit, keep_rows=None):
    layer_norm = core.encoder.ln
    weight_row = core.heads.head.weight[target_label].detach().float()
    totals, count, row = {}, 0, 0
    for images, _ in loader:
        embedding, store, final = collect_components(model, core, images.to(device))
        batch = images.shape[0]
        wanted = [
            i for i in range(batch) if keep_rows is None or (row + i) in keep_rows
        ]
        row += batch
        if wanted:
            index = torch.tensor(wanted, dtype=torch.long)
            pieces = {("embedding", -1): embedding, **store}
            for key, tensor in pieces.items():
                value = attribute(tensor[index], final[index], layer_norm, weight_row)
                totals[key] = value.sum() + totals.get(key, 0.0)
            count += len(wanted)
        if count >= limit:
            break
    return {key: float(value / count) for key, value in totals.items()}, count


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
    target = metadata["target_label"] if probe is None else 0

    backdoor, n_bd = run_split(
        model, core, loaders["backdoor"], device, target, args.limit
    )
    clean, n_cl = run_split(
        model,
        core,
        loaders["clean"],
        device,
        target,
        args.limit,
        paired_clean_rows(manifest),
    )
    layers = []
    for index in range(len(core.encoder.layers)):
        entry = {"layer": index + 1}
        for tap in ("attention_write", "mlp_write"):
            entry[f"{tap}_backdoor"] = backdoor[(tap, index)]
            entry[f"{tap}_clean"] = clean[(tap, index)]
            entry[f"{tap}_delta"] = backdoor[(tap, index)] - clean[(tap, index)]
        layers.append(entry)
    return {
        "folder_name": folder,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "target_label": target,
        "n_backdoor": n_bd,
        "n_clean_paired": n_cl,
        "embedding_delta": backdoor[("embedding", -1)] - clean[("embedding", -1)],
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
    parser.add_argument("--limit", type=int, default=256)
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
        with open(os.path.join(out_dir, "logit_attribution.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        total = sum(
            row["attention_write_delta"] + row["mlp_write_delta"]
            for row in report["layers"]
        )
        print(
            f"\n[ok] {folder}  n={report['n_backdoor']}  target class {report['target_label']}"
        )
        print(
            "     push toward the target class, triggered MINUS clean, at the CLS token"
        )
        print(f"{'layer':>5s} {'attn':>9s} {'mlp':>9s} {'cumulative':>11s}")
        running = report["embedding_delta"]
        print(f"{'emb':>5s} {'':>9s} {'':>9s} {running:11.3f}")
        for row in report["layers"]:
            running += row["attention_write_delta"] + row["mlp_write_delta"]
            print(
                f"{row['layer']:5d} {row['attention_write_delta']:9.3f} "
                f"{row['mlp_write_delta']:9.3f} {running:11.3f}"
            )
        print(f"     total sublayer push: {total:.3f}")


if __name__ == "__main__":
    main()
