"""Where are the backdoor neurons: which layers, which dimensions, and does SAM move them?

One report per checkpoint, combining every latent tool in `analysis/` so the same
question is asked several independent ways:

  per layer      TAC (trigger-activated change), relative backdoor-direction norm,
                 and debiased linear CKA between clean and triggered features. These
                 say WHICH LAYERS carry the trigger.
  per dimension  the top-k TAC dimensions at the peak layer, and the CLP outlier rule
                 (mean + 3 std). These say WHICH NEURONS.
  weight space   per-channel Lipschitz of each block's MLP output projection, and its
                 rank correlation with the measured TAC. If a data-free weight
                 quantity tracks the data-driven one, a defender can find backdoor
                 channels without any data at all.
  head alignment the Karayalcin data-free detector with its Z > 3 rule, which names a
                 suspected target class from weights alone.
  PCA and UMAP   how separable clean and triggered features are in 2D. PCA is the
                 honest first look because the backdoor is hypothesised to be a linear
                 direction and a linear projection cannot invent structure. UMAP is
                 reported with k-NN purity rather than by eye, so it is a number.

Everything is fp32. bfloat16 gives TAC a noise floor that grows with depth, because
TAC is a mean of absolute differences and does not average that noise away, and ViT
residual norms grow with depth.

Example
    PYTHONPATH=. python scripts/backdoor_neurons/measure.py \
        --attack badnet_a2o blend bpp lf --rho "" 0_1 0_2
"""

import argparse
import json
import os

import numpy as np
import torch
from lightning import seed_everything

from analysis.cka import debiased_linear_cka
from analysis.direction import (
    backdoor_direction,
    outlier_dimensions,
    trigger_activated_change,
)
from analysis.embedding import pca_project, umap_project
from analysis.features import extract_layer_features
from analysis.lipschitz import (
    alignment_outlier_score,
    head_weight_alignment,
    mlp_output_channel_lipschitz,
)
from attacks import build_attack, default_config
from defences.checkpoint_eval import read_checkpoint_metadata, resolve_probe_attack
from models import load_checkpoint
from scripts.backdoor_direction_layers.measure import build_paired_loaders
from utils.config import DATASET_REGISTRY

TOP_K = 20
ALIGNMENT_QUANTILE = 0.999
ALIGNMENT_LAYERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack", nargs="*", default=["badnet_a2o", "blend", "bpp", "lf"]
    )
    parser.add_argument("--rho", nargs="*", default=["", "0_1", "0_2"])
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--architecture", default="vit")
    parser.add_argument("--poison-tag", default="0_1")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--probe-attack", default="badnet_a2o")
    parser.add_argument("--probe-target-label", type=int, default=0)
    return parser.parse_args()


def separability_2d(clean: torch.Tensor, triggered: torch.Tensor, project) -> dict:
    """Silhouette and k-NN purity of the clean/triggered split in a 2D projection.

    k-NN purity is the honest way to report a UMAP: the fraction of each point's 10
    nearest neighbours sharing its label. It turns a picture into a number and does
    not depend on how the eye groups blobs.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    combined = torch.cat([clean, triggered])
    embedded = project(combined)
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(triggered))])

    neighbours = NearestNeighbors(n_neighbors=11).fit(embedded)
    _, indices = neighbours.kneighbors(embedded)
    purity = float((labels[indices[:, 1:]] == labels[:, None]).mean())
    return {
        "silhouette": float(silhouette_score(embedded, labels)),
        "knn_purity": purity,
    }


def analyse(folder: str, args: argparse.Namespace, device) -> dict | None:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    if not os.path.exists(path):
        return None
    metadata = read_checkpoint_metadata(path)
    spec = DATASET_REGISTRY[metadata["dataset"]]
    # A benign checkpoint carries no attack, so it is probed with one. Every number
    # here then reads as the null: whatever TAC, separability and Z a clean model
    # produces when the same trigger is painted on its inputs.
    # The override is only legal on a benign checkpoint; resolve_probe_attack
    # rejects it on a backdoored one rather than measuring a trigger the model
    # never saw, so it is passed only where it applies.
    benign = metadata["attack"] == "benign"
    attack_name, target_label = resolve_probe_attack(
        metadata,
        args.probe_attack if benign else None,
        args.probe_target_label if benign else None,
    )
    attack = build_attack(
        attack_name, default_config(attack_name), spec.image_size, target_label
    )
    clean_loader, triggered_loader, _ = build_paired_loaders(
        metadata["dataset"],
        attack,
        args.raw_data_dir,
        args.batch_size,
        args.samples,
        args.seed,
    )
    model = load_checkpoint(metadata["architecture"], path, device)

    # Everything downstream is CPU work (scipy, sklearn, umap), and the per-layer
    # feature stack is the only thing large enough for the move to matter.
    seed_everything(args.seed)
    clean = {
        k: v.cpu()
        for k, v in extract_layer_features(
            model, clean_loader, device, False, "cls"
        ).items()
    }
    seed_everything(args.seed)
    triggered = {
        k: v.cpu()
        for k, v in extract_layer_features(
            model, triggered_loader, device, False, "cls"
        ).items()
    }

    lipschitz = mlp_output_channel_lipschitz(model)

    layers = []
    for layer in sorted(clean):
        c, t = clean[layer], triggered[layer]
        tac = trigger_activated_change(c, t)
        direction = backdoor_direction(c, t)
        scale = c.norm(dim=1).mean().clamp_min(1e-8)
        row = {
            "layer": layer,
            "tac_mean": float(tac.mean()),
            "tac_max": float(tac.max()),
            "rel_direction_norm": float(direction.norm() / scale),
            "cka": float(debiased_linear_cka(c, t)),
            "n_outlier_dims": int(len(outlier_dimensions(tac, 3.0))),
            "top_dims": tac.topk(TOP_K).indices.tolist(),
        }
        # Does a data-free weight quantity track the data-driven TAC at this layer?
        if layer in lipschitz:
            from scipy.stats import spearmanr

            rho, _ = spearmanr(lipschitz[layer].numpy(), tac.numpy())
            row["lipschitz_tac_spearman"] = float(rho)
            row["lipschitz_top_dims"] = lipschitz[layer].topk(TOP_K).indices.tolist()
        layers.append(row)

    peak = max(layers, key=lambda r: r["rel_direction_norm"])

    alignment = head_weight_alignment(model, ALIGNMENT_LAYERS, ALIGNMENT_QUANTILE)
    z, suspected = alignment_outlier_score(alignment)

    final = 12 if 12 in clean else max(clean)
    projections = {
        "pca": separability_2d(
            clean[final], triggered[final], lambda x: pca_project(x, 2)
        )
    }
    if not args.skip_umap:
        projections["umap"] = separability_2d(
            clean[final],
            triggered[final],
            lambda x: umap_project(
                x, num_neighbors=15, min_distance=0.1, seed=args.seed
            ),
        )

    return {
        "folder_name": folder,
        "attack": metadata["attack"],
        "probe_attack": attack_name,
        "rho": metadata.get("rho"),
        "poison_rate": metadata.get("poison_rate"),
        "target_label": metadata["target_label"],
        "samples": args.samples,
        "peak_layer": peak["layer"],
        "layers": layers,
        "head_alignment": {
            "z": z,
            "suspected_target": suspected,
            "true_target": target_label,
            "correct": suspected == target_label,
            "flagged": z > 3.0,
            "scores": alignment.tolist(),
        },
        "projections": projections,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports = []
    for attack in args.attack:
        for rho in args.rho:
            stem = f"{args.architecture}_{args.dataset}_{attack}"
            # A benign checkpoint has no poison rate, so it carries no rate tag.
            if attack != "benign":
                stem += f"_{args.poison_tag}"
            folder = stem if rho == "" else f"{stem}_sam_rho_{rho}"
            try:
                report = analyse(folder, args, device)
            except Exception as error:
                print(f"{folder}: FAILED {type(error).__name__}: {error}", flush=True)
                continue
            if report is None:
                continue
            out = os.path.join(args.results_dir, folder, "backdoor_neurons.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as handle:
                json.dump(report, handle, indent=2)
            reports.append(report)
            print(
                f"{folder:38} peak_layer={report['peak_layer']:>2} "
                f"Z={report['head_alignment']['z']:>6.2f} "
                f"target={'hit' if report['head_alignment']['correct'] else 'miss'} "
                f"pca_purity={report['projections']['pca']['knn_purity']:.3f}",
                flush=True,
            )
    print(f"\nwrote {len(reports)} reports")


if __name__ == "__main__":
    main()
