"""Does the trigger manufacture an attention sink the model does not natively have?

Large self-supervised ViTs spontaneously produce a small population of high-norm "artifact" or
register tokens that attract attention and carry global rather than local information (Darcet
et al., ICLR 2024; Sun et al., arXiv:2402.17762 for the LLM form). That matters here for two
opposite reasons.

If such a population exists NATIVELY it is a confound: a detector that masks tokens at random
may be measuring whether the mask happened to hit an artifact token rather than whether the
input carries a backdoor. If it does NOT exist natively but the TRIGGER creates one, that is a
mechanism and a signature.

Runs on both architectures. Swin has no class token and its grid is halved at every stage, 56x56
in stage 1 down to 7x7 in stage 4, so the trigger is pooled onto each block's OWN grid rather
than onto a fixed 14x14 layout, and the class-token exclusion is applied only where there is one.

Measured per layer over patch tokens:

    ratio            max patch-token norm over the median, the outlier criterion
    outlier_share    fraction of patch tokens above `outlier_multiple` times the median
    trigger_is_top   how often the highest-norm patch token is one the trigger changed
    trigger_rank     mean percentile rank of the trigger's tokens by norm, 1.0 = highest

The same quantity is also scored as a DETECTOR, since it needs no perturbation, no trigger
knowledge and one forward pass. The clean split is paired down to the backdoor split's own
images first, or the comparison partly measures which images each split contains.

    PYTHONPATH=. python experiments/residual_stream_mechanism/artifact_tokens.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1 swin_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from attacks import apply_config_overrides, build_attack, default_config
from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.psbd_metrics import pair_clean_to_backdoor
from experiments.residual_stream_mechanism.architecture import (
    as_tokens,
    block_grid,
    detect_model_architecture,
    patch_token_slice,
    transformer_blocks,
    trigger_token_mask,
)
from models import load_checkpoint, network_core
from utils.config import DATASET_REGISTRY
from utils.numerics import safe_ratio_positive

MODEL_INPUT = 224


def trigger_pixel_delta(metadata) -> torch.Tensor:
    """The trigger's absolute per-pixel change, at the resolution the model sees.

    Kept in pixels rather than token indices so it can be pooled onto whichever grid a block
    uses. A fixed token mapping is a ViT assumption, not a general one.
    """
    size = DATASET_REGISTRY[metadata["dataset"]].image_size
    config = apply_config_overrides(
        default_config(metadata["attack"]), metadata.get("attack_config_overrides")
    )
    attack = build_attack(metadata["attack"], config, size, metadata["target_label"])
    plant = attack.apply_trigger_eval or attack.apply_trigger
    base = torch.rand(3, size, size, generator=torch.Generator().manual_seed(0))
    delta = (plant(base.clone(), 0) - base).abs().sum(dim=0, keepdim=True)
    return F.interpolate(
        delta.unsqueeze(0), size=(MODEL_INPUT, MODEL_INPUT), mode="bilinear"
    )[0, 0]


@torch.inference_mode()
def token_norm_stats(
    model, core, loader, device, pixel_delta, limit, multiple, architecture
):
    blocks = transformer_blocks(core, architecture)
    n_layers = len(blocks)
    acts = {}
    handles = [
        block.register_forward_hook(
            lambda _m, _i, out, index=index: acts.__setitem__(
                index, out.detach().float()
            )
        )
        for index, block in enumerate(blocks)
    ]
    patches = patch_token_slice(architecture)
    totals = {
        layer: {"ratio": 0.0, "share": 0.0, "top": 0.0, "rank": 0.0}
        for layer in range(n_layers)
    }
    per_sample = {layer: [] for layer in range(n_layers)}
    masks, seen = {}, 0
    for images, _ in loader:
        model(images.to(device))
        for layer in range(n_layers):
            tokens = as_tokens(acts[layer])[:, patches, :]
            if layer not in masks:
                masks[layer] = trigger_token_mask(
                    pixel_delta, block_grid(acts[layer], architecture)
                ).to(tokens.device)
            trigger = masks[layer]
            norms = tokens.norm(dim=-1)
            median = norms.median(dim=-1, keepdim=True).values
            entry = totals[layer]
            sample_ratio = safe_ratio_positive(
                norms.max(dim=-1).values, median.squeeze(-1)
            )
            per_sample[layer].append(sample_ratio.cpu())
            entry["ratio"] += float(sample_ratio.nansum())
            entry["share"] += float(
                (norms > multiple * median).float().mean(dim=-1).sum()
            )
            entry["top"] += float(trigger[norms.argmax(dim=-1)].float().sum())
            if bool(trigger.any()):
                order = norms.argsort(dim=-1).argsort(dim=-1).float() / max(
                    norms.shape[1] - 1, 1
                )
                entry["rank"] += float(order[:, trigger].mean(dim=-1).sum())
        seen += images.shape[0]
        if seen >= limit:
            break
    for handle in handles:
        handle.remove()
    return (
        {layer: {k: v / seen for k, v in e.items()} for layer, e in totals.items()},
        {layer: torch.cat(v) for layer, v in per_sample.items()},
        seen,
        {layer: int(m.sum()) for layer, m in masks.items()},
    )


def analyse(folder, args) -> dict:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    probe = "badnet_a2o" if metadata["attack"] == "benign" else None
    resolved = {**metadata, "attack": probe, "target_label": 0} if probe else metadata
    pixel_delta = trigger_pixel_delta(resolved)
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
    architecture = detect_model_architecture(core)

    triggered, triggered_scores, n, counts = token_norm_stats(
        model,
        core,
        loaders["backdoor"],
        device,
        pixel_delta,
        args.limit,
        args.outlier_multiple,
        architecture,
    )
    clean, clean_scores, _, _ = token_norm_stats(
        model,
        core,
        loaders["clean"],
        device,
        pixel_delta,
        len(loaders["clean"].dataset),
        args.outlier_multiple,
        architecture,
    )
    detection = {}
    for layer in triggered:
        paired = pair_clean_to_backdoor(clean_scores[layer], manifest)
        length = min(len(paired), len(triggered_scores[layer]))
        labels = np.concatenate([np.zeros(length), np.ones(length)])
        scores = np.concatenate(
            [paired[:length].numpy(), triggered_scores[layer][:length].numpy()]
        )
        finite = np.isfinite(scores)
        detection[layer + 1] = (
            float(roc_auc_score(labels[finite], scores[finite]))
            if finite.sum() > 10 and len(set(labels[finite])) > 1
            else float("nan")
        )
    return {
        "folder_name": folder,
        "architecture": architecture,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "poison_rate": metadata.get("poison_rate"),
        "outlier_multiple": args.outlier_multiple,
        "n_samples": n,
        "detection_auroc": detection,
        "layers": [
            {
                "layer": layer + 1,
                "n_trigger_tokens": counts[layer],
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
        print(f"\n[ok] {folder} [{report['architecture']}]  n={report['n_samples']}")
        print(
            f"{'layer':>5s} {'trig tok':>8s} {'ratio trig':>11s} {'ratio clean':>12s} "
            f"{'outlier%':>9s} {'top is trig':>12s} {'trig rank':>10s} {'AUROC':>7s}"
        )
        for row in report["layers"]:
            print(
                f"{row['layer']:5d} {row['n_trigger_tokens']:8d} {row['ratio_triggered']:11.2f} "
                f"{row['ratio_clean']:12.2f} {row['outlier_share_triggered']:8.2%} "
                f"{row['trigger_is_top_triggered']:11.2%} {row['trigger_rank_triggered']:10.3f} "
                f"{report['detection_auroc'][row['layer']]:7.3f}"
            )


if __name__ == "__main__":
    main()
