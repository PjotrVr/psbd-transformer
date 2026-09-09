"""Which sublayer writes the backdoor into the residual stream, and at what depth?

The residual stream is linear:

    h^(l+1) = h^l + a^l + m^l

where a^l is what the attention branch writes and m^l what the MLP branch writes. Define the
backdoor direction at any point as the paired mean difference, d = mean(backdoor) -
mean(clean). Because the mean is linear, the decomposition is EXACT:

    d_h^(l+1) = d_h^l + d_a^l + d_m^l

so the growth of the backdoor direction can be attributed, layer by layer, to the attention
branch or the MLP branch. That is the question a placement study needs answered: a
perturbation should only disrupt the backdoor if it lands where the direction is still being
written, and it should target the sublayer doing the writing.

The three taps are already named in the position registry, so this measures the same tensors
the perturbation study perturbs:

    stream in         block pre-hook                 h^l
    attention write   `dropout` post-hook            a^l   (= before_attention_residual)
    mlp write         `mlp` post-hook                m^l   (= before_mlp_residual)

Reported per layer:

    rel_write         ||d|| of the write, over ||d_h^12||, so contributions are comparable
    stream_share      ||d_h^l|| / ||h^l||, the direction as a fraction of the stream it rides
    cos_to_final      alignment of the write with the FINAL direction. A large write that is
                      orthogonal to the final direction does not build the backdoor
    cls_share         how much of the direction sits on the CLS token rather than the patches
    top8_share        how concentrated the direction is over tokens

A rising per-layer norm is not evidence of anything on its own: residual-stream norms grow
roughly exponentially with depth regardless of what is in them (Heimersheim and Turner 2023),
which is why the primary quantity here is the direction's SHARE of the stream and not its norm.

And a difference of means over finitely many samples is never exactly zero, so every number
needs its sampling floor. The control is a label SHUFFLE: pool the paired triggered and clean
samples, split them at random, and take the same difference of means. That null has no
backdoor in it by construction, so whatever it reads is what noise alone produces. Reported as
a mean and a 2-sigma band over several draws, the convention used for difference-of-means
directions elsewhere (Arditi et al., NeurIPS 2024).

Means are accumulated incrementally, so memory does not grow with the sample count.

    PYTHONPATH=. python experiments/residual_stream_mechanism/residual_decomposition.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import os

import torch

from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from utils.numerics import safe_ratio, safe_ratio_positive
from models import load_checkpoint, network_core

TAPS = ("stream_in", "attention_write", "mlp_write")


def attach_taps(core, store):
    """Hook the stream and both branch writes on every block."""
    handles = []
    for index, block in enumerate(core.encoder.layers):

        def keep(name, index):
            def hook(_module, inputs, output=None):
                tensor = inputs[0] if output is None else output
                store.setdefault((name, index), []).append(tensor.detach().float())

            return hook

        handles.append(block.register_forward_pre_hook(keep("stream_in", index)))
        handles.append(
            block.dropout.register_forward_hook(
                lambda m, i, o, index=index: store.setdefault(
                    ("attention_write", index), []
                ).append(o.detach().float())
            )
        )
        handles.append(
            block.mlp.register_forward_hook(
                lambda m, i, o, index=index: store.setdefault(
                    ("mlp_write", index), []
                ).append(o.detach().float())
            )
        )
    return handles


@torch.inference_mode()
def accumulate(model, core, loader, device, keep_rows, n_layers):
    """Per-token sums of every tap, over the rows selected by `keep_rows`."""
    totals, counts = {}, 0
    row = 0
    for images, _ in loader:
        store = {}
        handles = attach_taps(core, store)
        model(images.to(device))
        for handle in handles:
            handle.remove()

        batch = images.shape[0]
        wanted = [
            i for i in range(batch) if keep_rows is None or (row + i) in keep_rows
        ]
        row += batch
        if not wanted:
            continue
        index = torch.tensor(wanted, dtype=torch.long)
        counts += len(wanted)
        for key, tensors in store.items():
            selected = tensors[0][index].sum(dim=0)
            totals[key] = selected if key not in totals else totals[key] + selected
    return {key: value / counts for key, value in totals.items()}, counts


def paired_clean_rows(manifest) -> set:
    clean_indices = manifest["analysis_clean_indices"]
    backdoor_indices = set(manifest["analysis_backdoor_indices"])
    return {
        row
        for row, original in enumerate(clean_indices)
        if original in backdoor_indices
    }


def summarise(clean_means, backdoor_means, n_layers) -> list[dict]:
    direction = {key: backdoor_means[key] - clean_means[key] for key in backdoor_means}
    final = (
        direction[("stream_in", n_layers - 1)]
        + direction[("attention_write", n_layers - 1)]
        + direction[("mlp_write", n_layers - 1)]
    )
    final_flat = final.reshape(-1)
    final_norm = final_flat.norm().clamp(min=1e-9)

    rows = []
    for layer in range(n_layers):
        entry = {"layer": layer + 1}
        stream = direction[("stream_in", layer)]
        stream_activation = clean_means[("stream_in", layer)]
        entry["stream_share"] = float(
            safe_ratio_positive(stream.norm(), stream_activation.norm())
        )
        entry["direction_norm_rel"] = float(
            safe_ratio_positive(stream.norm(), final_norm)
        )
        for tap in ("attention_write", "mlp_write"):
            write = direction[(tap, layer)]
            flat = write.reshape(-1)
            entry[f"{tap}_rel"] = float(safe_ratio_positive(flat.norm(), final_norm))
            entry[f"{tap}_cos_final"] = float(
                torch.dot(flat, final_flat) / (flat.norm().clamp(min=1e-9) * final_norm)
            )
            per_token = write.norm(dim=-1)
            mass = per_token / per_token.sum().clamp(min=1e-9)
            entry[f"{tap}_cls_share"] = float(mass[0])
            entry[f"{tap}_top8_share"] = float(
                mass[1:].sort(descending=True).values[:8].sum()
            )
        per_token = stream.norm(dim=-1)
        mass = per_token / per_token.sum().clamp(min=1e-9)
        entry["stream_cls_share"] = float(mass[0])
        entry["stream_top8_share"] = float(
            mass[1:].sort(descending=True).values[:8].sum()
        )
        rows.append(entry)
    return rows


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
    n_layers = len(core.encoder.layers)

    backdoor_means, n_bd = accumulate(
        model, core, loaders["backdoor"], device, None, n_layers
    )
    clean_means, n_cl = accumulate(
        model, core, loaders["clean"], device, paired_clean_rows(manifest), n_layers
    )
    report = {
        "folder_name": folder,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "poison_rate": metadata.get("poison_rate"),
        "n_backdoor": n_bd,
        "n_clean_paired": n_cl,
        "layers": summarise(clean_means, backdoor_means, n_layers),
    }
    # The shuffle null: the same statistic with the backdoor/clean split replaced by a random
    # one, so the real numbers can be read against what sampling noise alone produces.
    if args.controls:
        floors = []
        for draw in range(args.controls):
            shuffled = {key: backdoor_controls[(key, draw)] for key in backdoor_means}
            floors.append(
                [
                    row["stream_share"]
                    for row in summarise(
                        {k: torch.zeros_like(v) for k, v in clean_means.items()},
                        shuffled,
                        n_layers,
                    )
                ]
            )
        stacked = torch.tensor(floors)
        report["control_stream_share_mean"] = stacked.mean(dim=0).tolist()
        report["control_stream_share_sd"] = stacked.std(dim=0).tolist()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=2400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--controls",
        type=int,
        default=8,
        help="label-shuffle draws giving the sampling floor for a difference of means",
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
        with open(os.path.join(out_dir, "residual_decomposition.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n[ok] {folder}  n={report['n_backdoor']} paired")
        print(
            f"{'layer':>5s} {'dir/final':>9s} {'share of stream':>15s} "
            f"{'attn write':>10s} {'cos':>6s} {'mlp write':>10s} {'cos':>6s} {'CLS share':>9s}"
        )
        for row in report["layers"]:
            print(
                f"{row['layer']:5d} {row['direction_norm_rel']:9.3f} {row['stream_share']:15.3f} "
                f"{row['attention_write_rel']:10.3f} {row['attention_write_cos_final']:6.2f} "
                f"{row['mlp_write_rel']:10.3f} {row['mlp_write_cos_final']:6.2f} "
                f"{row['stream_cls_share']:9.3f}"
            )


if __name__ == "__main__":
    main()
