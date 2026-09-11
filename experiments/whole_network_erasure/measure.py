"""Whole-network direction removal, done the way Karayalcin et al. do it.

"Backdoor Directions in Vision Transformers" (Karayalcin, Krcek, Chen and Picek,
arXiv 2603.10806, Section 4) estimates the backdoor direction at every layer as
the mean activation difference between triggered and clean copies of the same
images, picks 1 layer by its Eq. 1, and following Arditi et al. (Section 4.1)
projects the unit direction out of every matrix that writes to the residual
stream: the embedding and every attention and MLP output projection. They report
ASR falling from 97.7 to 6.7 overall. Our earlier erasure (H34) edited only
blocks 10 and 11 and left BadNet and Blend at ASR 1.000, so this script settles
whether the difference is the scope of the edit.

    original form
        r^l   = (1 / |X_pair|) sum_{(x, x_t)} (x_t^l - x^l)                       Sec. 4.1
        r_hat = argmax_{r^l} {(ASR_{+r^l} + RA_{-r^l}) - (ASR_{+r^{l-1}} + RA_{-r^{l-1}})}   Eq. 1
        W_new = W - r_hat r_hat^T W                                               Sec. 4.2
    symbols
        x^l, x_t^l     class-token residual stream at layer l of a clean image and
                       its triggered copy, (dim,)
        X_pair         the paired images the direction is estimated on
        ASR_{+r^l}     share of clean images predicted as the target after adding
                       r^l at layer l
        RA_{-r^l}      share of triggered images predicted as their true class
                       after subtracting r^l at layer l
        r_hat          the unit direction of the selected layer
        W              a residual-stream write matrix, (dim, input_dim)

Layer 0 is the embedding output entering block 1 and layer l the output of block
l, the indexing analysis.features.captured_layers uses. The direction is estimated
on the first half of the paired analysis rows and every rate is read on the
second half, so no number is read on the images the direction came from.

Reported per checkpoint: the baseline, class-token and all-token steering at every
layer, the layer Eq. 1 picks for each steering variant, whole-network removal with
the direction of every layer (weights only as the formula states, and weights and
biases), removal restricted to blocks 10 and 11 as H34 did, and a random unit
direction removed from the whole network as the control.

    PYTHONPATH=. python experiments/whole_network_erasure/measure.py \
        --checkpoints-dir /lustre/home/pstika/projects/PSBD-ViT/checkpoints \
        --raw-data-dir /lustre/home/pstika/projects/PSBD-ViT/raw_data \
        --folders vit_cifar10_badnet_a2o_0_1 vit_cifar10_blend_0_1
"""

import argparse
import copy
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from analysis.features import captured_layers, transformer_blocks  # noqa: E402
from data.splits import (  # noqa: E402
    BENIGN_PROBE_ATTACK,
    BENIGN_PROBE_TARGET_LABEL,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from experiments._paths import experiment_result_path  # noqa: E402
from models.backbones import load_checkpoint, network_core  # noqa: E402

SLUG = "whole_network_erasure"
NUM_LAYERS = 12
ESTIMATE_PAIRS = 500
H34_BLOCKS = (10, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--max-samples", type=int, default=2000, help="analysis rows read per split"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def stack_loader(loader) -> tuple[torch.Tensor, torch.Tensor]:
    """Every image and label a shuffle=False loader serves, on the CPU."""
    images, labels = [], []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
    stacked = torch.cat(images), torch.cat(labels).long()  # (n, C, H, W), (n,)
    return stacked


def paired_rows(loaders, manifest) -> dict[str, torch.Tensor]:
    """Triggered images with the clean copy and true label of the same source image."""
    clean_images, clean_labels = stack_loader(loaders["clean"])
    backdoor_images, target_labels = stack_loader(loaders["backdoor"])
    row_of = {
        original: row for row, original in enumerate(manifest["analysis_clean_indices"])
    }
    rows = torch.tensor(
        [row_of[original] for original in manifest["analysis_backdoor_indices"]]
    )  # (n_backdoor,)
    pairs = {
        "clean": clean_images[rows],  # (n_backdoor, C, H, W)
        "backdoor": backdoor_images,  # (n_backdoor, C, H, W)
        "true_label": clean_labels[rows],  # (n_backdoor,)
        "target": target_labels,  # (n_backdoor,)
        "all_clean": clean_images,  # (n_clean, C, H, W)
        "all_clean_label": clean_labels,  # (n_clean,)
    }
    return pairs


@torch.inference_mode()
def residual_stream(model, images, device, batch_size) -> torch.Tensor:
    """Every token at layers 0 to 12, float32, (n, 13, tokens, dim), on the CPU."""
    chunks = []
    layers = tuple(range(NUM_LAYERS + 1))
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)  # (batch, C, H, W)
        with captured_layers(model, layers, "vit") as captured:
            model(batch)
            stacked = torch.stack(
                [captured[layer].float() for layer in layers], dim=1
            )  # (batch, 13, tokens, dim)
        chunks.append(stacked.cpu())
    stream = torch.cat(chunks)  # (n, 13, tokens, dim)
    return stream


def directions(model, pairs, device, batch_size) -> dict[str, torch.Tensor]:
    """The class-token and all-token directions of every layer from the estimation pairs."""
    clean = residual_stream(
        model, pairs["clean"][:ESTIMATE_PAIRS], device, batch_size
    )  # (n, 13, tokens, dim)
    triggered = residual_stream(
        model, pairs["backdoor"][:ESTIMATE_PAIRS], device, batch_size
    )  # (n, 13, tokens, dim)
    difference = (triggered - clean).mean(dim=0)  # (13, tokens, dim)
    found = {
        "cls": difference[:, 0, :],
        "all_tokens": difference,
    }  # (13, dim), (13, tokens, dim)
    return found


def steering_hook(vector: torch.Tensor, sign: float, cls_only: bool):
    """A hook adding sign * vector to the residual stream, at token 0 or at every token."""

    def add(activation: torch.Tensor) -> torch.Tensor:
        steered = activation.clone()  # (batch, tokens, dim)
        if cls_only:
            steered[:, 0, :] += sign * vector.to(activation.device, activation.dtype)
        else:
            steered += sign * vector.to(activation.device, activation.dtype)
        return steered

    return add


@torch.inference_mode()
def predictions(
    model, images, device, batch_size, layer=None, edit=None
) -> torch.Tensor:
    """Argmax predictions, with an optional residual-stream edit at 1 layer, (n,)."""
    core = network_core(model)
    blocks = transformer_blocks(core, "vit")
    handle = None
    if edit is not None and layer == 0:
        handle = blocks[0].register_forward_pre_hook(
            lambda _m, inputs: (edit(inputs[0]),)
        )
    elif edit is not None:
        handle = blocks[layer - 1].register_forward_hook(
            lambda _m, _i, output: edit(output)
        )
    try:
        predicted = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)  # (batch, C, H, W)
            predicted.append(model(batch).argmax(dim=1).cpu())  # (batch,)
    finally:
        if handle is not None:
            handle.remove()
    stacked = torch.cat(predicted)  # (n,)
    return stacked


def rates(
    model, pairs, device, batch_size, layer=None, edit_clean=None, edit_backdoor=None
) -> dict:
    """ASR on triggered images, RA on triggered images, positive-steering ASR on clean ones and CA."""
    evaluation = slice(ESTIMATE_PAIRS, None)
    target = pairs["target"][evaluation]
    true_label = pairs["true_label"][evaluation]
    on_backdoor = predictions(
        model, pairs["backdoor"][evaluation], device, batch_size, layer, edit_backdoor
    )  # (n,)
    on_clean_pairs = predictions(
        model, pairs["clean"][evaluation], device, batch_size, layer, edit_clean
    )  # (n,)
    measured = {
        "asr": float((on_backdoor == target).float().mean()),
        "ra": float((on_backdoor == true_label).float().mean()),
        "asr_clean": float((on_clean_pairs == target).float().mean()),
        "ca_pairs": float((on_clean_pairs == true_label).float().mean()),
        "n_eval": int(len(target)),
    }
    return measured


def clean_accuracy(model, pairs, device, batch_size) -> float:
    """Accuracy on the clean rows not used to estimate the direction, every class included."""
    images = pairs["all_clean"][ESTIMATE_PAIRS:]
    labels = pairs["all_clean_label"][ESTIMATE_PAIRS:]
    predicted = predictions(model, images, device, batch_size)  # (n,)
    accuracy = float((predicted == labels).float().mean())
    return accuracy


def residual_writers(
    core: nn.Module, blocks: tuple[int, ...] | None
) -> list[tuple[str, nn.Parameter, bool]]:
    """(name, parameter, is_bias) for every tensor that writes to the residual stream.

    blocks None is the whole network, embedding included, as Karayalcin et al. and
    Arditi et al. edit it. A tuple of 1-based block indices restricts the edit to
    those blocks' attention and MLP output projections, the H34 scope.
    """
    writers = []
    if blocks is None:
        writers += [
            ("conv_proj.weight", core.conv_proj.weight, False),
            ("conv_proj.bias", core.conv_proj.bias, True),
            ("class_token", core.class_token, True),
            ("encoder.pos_embedding", core.encoder.pos_embedding, True),
        ]
    for index, layer in enumerate(core.encoder.layers, start=1):
        if blocks is not None and index not in blocks:
            continue
        writers += [
            (
                f"block{index}.out_proj.weight",
                layer.self_attention.out_proj.weight,
                False,
            ),
            (f"block{index}.out_proj.bias", layer.self_attention.out_proj.bias, True),
            (f"block{index}.fc2.weight", layer.mlp[3].weight, False),
            (f"block{index}.fc2.bias", layer.mlp[3].bias, True),
        ]
    return writers


@torch.no_grad()
def orthogonalised(
    model: nn.Module, direction: torch.Tensor, blocks=None, include_bias=False
) -> nn.Module:
    """A copy of the model with the unit direction projected out of its residual writes.

        W_new = W - r_hat r_hat^T W   on the output (residual) dimension
        b_new = b - r_hat r_hat^T b   when include_bias is set

    The class token and the position embedding are vectors the embedding adds to
    the stream, so they count as biases. The conv patch projection is read as a
    (dim, 3 * 16 * 16) matrix.
    """
    edited = copy.deepcopy(model)
    core = network_core(edited)
    unit = direction / direction.norm()  # (dim,)
    for _name, parameter, is_bias in residual_writers(core, blocks):
        if is_bias and not include_bias:
            continue
        unit_cast = unit.to(parameter.device, parameter.dtype)  # (dim,)
        flat = (
            parameter.data.reshape(-1, unit.numel())
            if is_bias
            else parameter.data.reshape(unit.numel(), -1)
        )
        if is_bias:
            flat -= (flat @ unit_cast).unsqueeze(1) * unit_cast.unsqueeze(
                0
            )  # (rows, dim)
        else:
            flat -= torch.outer(unit_cast, unit_cast @ flat)  # (dim, input_dim)
    return edited


def eq1_layer(steering: list[dict]) -> int:
    """Eq. 1: the layer with the largest jump in ASR_+ plus RA_- over the layer before it."""
    scores = [row["asr_clean"] + row["ra"] for row in steering]
    jumps = [scores[layer] - scores[layer - 1] for layer in range(1, len(scores))]
    selected = 1 + max(range(len(jumps)), key=jumps.__getitem__)
    return selected


def analyse(folder: str, args: argparse.Namespace, device: torch.device) -> dict:
    checkpoint = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint)
    benign = metadata["attack"] == "benign"
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        max_samples=args.max_samples,
        probe_attack=BENIGN_PROBE_ATTACK if benign else None,
        probe_target_label=BENIGN_PROBE_TARGET_LABEL if benign else None,
    )
    model = load_checkpoint(metadata["architecture"], checkpoint, device)
    pairs = paired_rows(loaders, manifest)
    found = directions(model, pairs, device, args.batch_size)

    report = {
        "folder": folder,
        "attack": metadata["attack"],
        "dataset": metadata["dataset"],
        "poison_rate": metadata.get("poison_rate"),
        "n_estimate": ESTIMATE_PAIRS,
        "baseline": {
            **rates(model, pairs, device, args.batch_size),
            "ca": clean_accuracy(model, pairs, device, args.batch_size),
        },
        "direction_norm": [
            float(found["cls"][layer].norm()) for layer in range(NUM_LAYERS + 1)
        ],
    }

    for variant, cls_only in (("steer_cls", True), ("steer_all_tokens", False)):
        rows = []
        for layer in range(NUM_LAYERS + 1):
            vector = found["cls"][layer] if cls_only else found["all_tokens"][layer]
            rows.append(
                rates(
                    model,
                    pairs,
                    device,
                    args.batch_size,
                    layer,
                    edit_clean=steering_hook(vector, 1.0, cls_only),
                    edit_backdoor=steering_hook(vector, -1.0, cls_only),
                )
            )
        report[variant] = rows
        report[f"{variant}_eq1_layer"] = eq1_layer(rows)

    for variant, blocks, include_bias in (
        ("whole_weights", None, False),
        ("whole_weights_and_biases", None, True),
        ("h34_blocks_10_11", H34_BLOCKS, False),
    ):
        rows = []
        for layer in range(NUM_LAYERS + 1):
            edited = orthogonalised(model, found["cls"][layer], blocks, include_bias)
            measured = rates(edited, pairs, device, args.batch_size)
            measured["ca"] = clean_accuracy(edited, pairs, device, args.batch_size)
            rows.append(measured)
            del edited
        report[variant] = rows

    generator = torch.Generator().manual_seed(args.seed)
    random_direction = torch.randn(found["cls"].shape[1], generator=generator)  # (dim,)
    edited = orthogonalised(model, random_direction.to(device), None, True)
    report["random_direction_whole"] = {
        **rates(edited, pairs, device, args.batch_size),
        "ca": clean_accuracy(edited, pairs, device, args.batch_size),
    }
    return report


def summary_line(report: dict) -> str:
    """1 line: baseline, the Eq. 1 layer and removal at that layer under each scope."""
    layer = report["steer_cls_eq1_layer"]
    best_whole = min(report["whole_weights_and_biases"], key=lambda row: row["asr"])
    best_layer = report["whole_weights_and_biases"].index(best_whole)
    line = (
        f"{report['folder']:36s} base ASR {report['baseline']['asr']:.3f} CA {report['baseline']['ca']:.3f} | "
        f"eq1 layer {layer:2d}: whole W ASR {report['whole_weights'][layer]['asr']:.3f} "
        f"W+b ASR {report['whole_weights_and_biases'][layer]['asr']:.3f} RA {report['whole_weights_and_biases'][layer]['ra']:.3f} "
        f"CA {report['whole_weights_and_biases'][layer]['ca']:.3f} | blocks 10-11 ASR {report['h34_blocks_10_11'][layer]['asr']:.3f} | "
        f"best layer {best_layer:2d} ASR {best_whole['asr']:.3f} CA {best_whole['ca']:.3f} | random ASR {report['random_direction_whole']['asr']:.3f}"
    )
    return line


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for folder in args.folders:
        report = analyse(folder, args, device)
        path = experiment_result_path(SLUG, f"{folder}.json", args.results_dir)
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2)
        print(summary_line(report), flush=True)


if __name__ == "__main__":
    main()
