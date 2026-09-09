"""Does the trigger manufacture an attention sink the model does not natively have?

Large self-supervised ViTs spontaneously produce a small population of high-norm "artifact"
or register tokens that attract CLS attention and carry global rather than local information
(Darcet et al., ICLR 2024; Sun et al. arXiv:2402.17762 for the LLM form). That matters here
for two opposite reasons.

If such a population exists NATIVELY, it is a confound: a detector that randomly masks tokens
and measures the prediction change may be measuring whether the mask happened to hit an
artifact token, not whether the input carries a backdoor. That would make PSBD on ViT an
artifact-token detector, and it has to be ruled out before any positive claim.

If it does NOT exist natively but the TRIGGER creates one, that is a mechanism and a
signature: the backdoor is manufacturing the routing machinery it needs.

Measured per layer, over patch tokens only (CLS excluded, since CLS is not a spatial token):

If the trigger does manufacture one, the same quantity is a DETECTOR that needs no
perturbation, no trigger knowledge and a single forward pass, so it is scored as one here.
The clean split is paired down to the backdoor split's own images first, or the comparison
comes partly from which classes each split happens to contain.

    ratio            max patch-token norm over the median, the outlier criterion
    outlier_share    fraction of patch tokens above `outlier_multiple` times the median
    trigger_is_top   how often the highest-norm patch token is one the trigger changed
    trigger_rank     mean percentile rank of the trigger's tokens by norm, 1.0 = highest

The benign checkpoint is the control for the native population, and clean inputs through the
BACKDOORED model are the control for the weights.

    PYTHONPATH=. python experiments/residual_stream_mechanism/artifact_tokens.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from attacks import apply_config_overrides, build_attack, default_config
import numpy as np
from sklearn.metrics import roc_auc_score

from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.psbd_metrics import pair_clean_to_backdoor
from models import load_checkpoint, network_core
from utils.config import DATASET_REGISTRY

MODEL_INPUT = 224
PATCH = 16
GRID = MODEL_INPUT // PATCH


def trigger_tokens(metadata) -> torch.Tensor:
    """Patch tokens the trigger actually changes, at the resolution the model sees."""
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
    return (per_token.sum(dim=1) > 1e-6).nonzero(as_tuple=True)[0]


@torch.inference_mode()
def token_norm_stats(model, core, loader, device, tokens, limit, multiple):
    acts = {}
    handles = [
        core.encoder.layers[index].register_forward_hook(
            lambda _m, _i, out, index=index: acts.__setitem__(
                index, out.detach().float()
            )
        )
        for index in range(len(core.encoder.layers))
    ]
    n_layers = len(core.encoder.layers)
    totals = {
        layer: {"ratio": 0.0, "share": 0.0, "top": 0.0, "rank": 0.0}
        for layer in range(n_layers)
    }
    per_sample = {layer: [] for layer in range(n_layers)}
    seen = 0
    trigger_set = set(tokens.tolist())
    for images, _ in loader:
        model(images.to(device))
        batch = images.shape[0]
        for layer in range(n_layers):
            norms = acts[layer][:, 1:, :].norm(dim=-1)  # patch tokens only
            median = norms.median(dim=-1, keepdim=True).values
            entry = totals[layer]
            sample_ratio = norms.max(dim=-1).values / median.squeeze(-1)
            per_sample[layer].append(sample_ratio.cpu())
            entry["ratio"] += float(sample_ratio.sum())
            entry["share"] += float(
                (norms > multiple * median).float().mean(dim=-1).sum()
            )
            top = norms.argmax(dim=-1)
            entry["top"] += float(sum(1 for t in top.tolist() if t in trigger_set))
            if len(trigger_set):
                order = norms.argsort(dim=-1).argsort(dim=-1).float() / (
                    norms.shape[1] - 1
                )
                entry["rank"] += float(order[:, tokens].mean(dim=-1).sum())
        seen += batch
        if seen >= limit:
            break
    for handle in handles:
        handle.remove()
    return (
        {
            layer: {key: value / seen for key, value in entry.items()}
            for layer, entry in totals.items()
        },
        {layer: torch.cat(values) for layer, values in per_sample.items()},
        seen,
    )


def analyse(folder, args) -> dict:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    probe = "badnet_a2o" if metadata["attack"] == "benign" else None
    resolved = {**metadata, "attack": probe, "target_label": 0} if probe else metadata
    tokens = trigger_tokens(resolved)
    loaders, manifest = build_psbd_loaders_from_checkpoint(
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

    triggered, triggered_scores, n = token_norm_stats(
        model,
        core,
        loaders["backdoor"],
        device,
        tokens,
        args.limit,
        args.outlier_multiple,
    )
    # The clean split covers the whole analysis pool and the backdoor split is a subset of
    # it, so the clean pass must run to the end before it can be paired down.
    clean, clean_scores, _ = token_norm_stats(
        model,
        core,
        loaders["clean"],
        device,
        tokens,
        len(loaders["clean"].dataset),
        args.outlier_multiple,
    )
    detection = {}
    for layer in triggered:
        paired = pair_clean_to_backdoor(clean_scores[layer], manifest)
        length = min(len(paired), len(triggered_scores[layer]))
        labels = np.concatenate([np.zeros(length), np.ones(length)])
        scores = np.concatenate(
            [paired[:length].numpy(), triggered_scores[layer][:length].numpy()]
        )
        detection[layer + 1] = (
            float(roc_auc_score(labels, scores)) if length else float("nan")
        )
    return {
        "folder_name": folder,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "poison_rate": metadata.get("poison_rate"),
        "n_trigger_tokens": int(len(tokens)),
        "outlier_multiple": args.outlier_multiple,
        "n_samples": n,
        "detection_auroc": detection,
        "layers": [
            {
                "layer": layer + 1,
                "ratio_triggered": triggered[layer]["ratio"],
                "ratio_clean": clean[layer]["ratio"],
                "outlier_share_triggered": triggered[layer]["share"],
                "outlier_share_clean": clean[layer]["share"],
                "trigger_is_top_triggered": triggered[layer]["top"],
                "trigger_is_top_clean": clean[layer]["top"],
                "trigger_rank_triggered": triggered[layer]["rank"],
                "trigger_rank_clean": clean[layer]["rank"],
            }
            for layer in sorted(triggered)
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
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--outlier-multiple", type=float, default=2.0)
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
        with open(os.path.join(out_dir, "artifact_tokens.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print(
            f"\n[ok] {folder}  trigger covers {report['n_trigger_tokens']}/196 tokens  "
            f"n={report['n_samples']}"
        )
        print(
            f"{'layer':>5s} {'ratio trig':>11s} {'ratio clean':>12s} "
            f"{'outlier% trig':>14s} {'top is trigger':>15s} {'trigger rank':>13s} {'AUROC':>7s}"
        )
        for row in report["layers"]:
            print(
                f"{row['layer']:5d} {row['ratio_triggered']:11.2f} {row['ratio_clean']:12.2f} "
                f"{row['outlier_share_triggered']:13.2%} {row['trigger_is_top_triggered']:14.2%} "
                f"{row['trigger_rank_triggered']:13.3f} "
                f"{report['detection_auroc'][row['layer']]:7.3f}"
            )


if __name__ == "__main__":
    main()
