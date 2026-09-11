"""Whether SAM training amplifies the trigger's footprint or widens PSBD's margin.

The SAM paper (Zhang et al., arXiv 2411.11525, papers/reliable_poisoned_sample_
detection_against_backdoor_attacks_enhanced_by_sharpness_aware_minimization/,
Section 3) claims sharpness-aware minimisation amplifies backdoor neurons, which
is why poisoned samples separate more cleanly downstream. On our ViT-B/16
checkpoints PSBD-TM (token_mask at before_attention_norm) gains only +0.02 to
+0.04 AUROC under SAM and PSBD-RD gets worse. PSBD reads the margin of the
predicted class under a token-masking perturbation, not a neuron activation, so
the amplification claim and the detection result are 2 separate questions:

    1. does SAM amplify the trigger's footprint in the residual stream (its
       direction norm, its TAC), the quantity the SAM paper's amplification
       metrics track,
    2. does SAM widen the gap between how a triggered and a clean input respond
       to PSBD's own perturbation (the psu_ratio separation), the quantity that
       would actually move PSBD-TM's AUROC.

6 matched (Adam, SAM rho=0.1) checkpoint pairs at 1% and 5% poisoning, BadNet,
Blend, BPP, LF and WaNet on CIFAR-10 and CIFAR-100, 500 paired clean and
triggered images per checkpoint. Per layer: the class-token backdoor-direction
norm relative to the clean residual-stream norm (Karayalcin et al.'s form, see
experiments/whole_network_erasure/measure.py) and its onset layer, and the mean
TAC (analysis.direction.trigger_activated_change). At the final layer with no
perturbation: the logit margin of the predicted class over the runner-up, on
clean and on triggered inputs. Under token_mask at before_attention_norm at the
rate the 0.8 adaptive rule selected for that checkpoint (results/<folder>/
psbd_metrics.json, falling back to 0.5 when the record is absent): the mean
psu_ratio on clean and on triggered inputs and the AUROC separating them, run
through defences.operators, models.positions.plug_dropout and defences.scores
exactly as cli/sweep.py and cli/analyze.py do.

paired_rows and residual_stream are imported from experiments.whole_network_
erasure.measure rather than reimplemented, since both experiments read the same
paired clean and backdoor images and the same per-layer residual stream.

    PYTHONPATH=. .venv/bin/python experiments/sam_mechanism/measure.py \
        --checkpoints-dir checkpoints --raw-data-dir raw_data
"""

import argparse
import importlib.util
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from analysis.direction import backdoor_direction, trigger_activated_change  # noqa: E402
from data.splits import (  # noqa: E402
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.inference import (  # noqa: E402
    build_baseline_cache,
    compute_dropout_pass_probs,
    forward_logits,
)
from defences.operators import build_operator, check_operator_position  # noqa: E402
from defences.scores import psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from models.backbones import load_checkpoint  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402

# whole_network_erasure/measure.py is not a package (no __init__.py), so its
# paired_rows and residual_stream are loaded by path rather than by a normal
# import, to reuse the exact construction rather than a second copy of it.
_WNE_MODULE_PATH = os.path.join(
    REPO_ROOT, "experiments", "whole_network_erasure", "measure.py"
)
_wne_spec = importlib.util.spec_from_file_location(
    "whole_network_erasure_measure", _WNE_MODULE_PATH
)
_wne = importlib.util.module_from_spec(_wne_spec)
_wne_spec.loader.exec_module(_wne)

SLUG = "sam_mechanism"
NUM_LAYERS = _wne.NUM_LAYERS
PAIR_COUNT = 500
POSITION = "before_attention_norm"
OPERATOR = "token_mask"
FALLBACK_RATE = 0.5
FORWARD_PASSES = 3
PSBD_MASK_SEED = 0
RESIDUAL_STREAM_BATCH = 64
DIRECTION_NORM_FLOOR = 1e-8

# 6 matched (Adam, SAM rho=0.1) pairs at 1% and 5% poisoning. CIFAR-100 carries
# BadNet, Blend, BPP and LF. CIFAR-10 fills the WaNet and the second BadNet
# cell, since these are the pairs whose SAM checkpoint also has a
# psbd_metrics.json record on disk.
PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "badnet_a2o_1pct_cifar100",
        "vit_cifar100_badnet_a2o_0_01",
        "vit_cifar100_badnet_a2o_0_01_sam_rho_0_1",
    ),
    (
        "blend_5pct_cifar100",
        "vit_cifar100_blend_0_05",
        "vit_cifar100_blend_0_05_sam_rho_0_1",
    ),
    ("bpp_1pct_cifar100", "vit_cifar100_bpp_0_01", "vit_cifar100_bpp_0_01_sam_rho_0_1"),
    ("lf_5pct_cifar100", "vit_cifar100_lf_0_05", "vit_cifar100_lf_0_05_sam_rho_0_1"),
    (
        "wanet_5pct_cifar10",
        "vit_cifar10_wanet_0_05",
        "vit_cifar10_wanet_0_05_sam_rho_0_1",
    ),
    (
        "badnet_a2o_5pct_cifar10",
        "vit_cifar10_badnet_a2o_0_05",
        "vit_cifar10_badnet_a2o_0_05_sam_rho_0_1",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def class_token_stream(
    model: torch.nn.Module, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Class-token residual stream at layers 0 to 12, (n, 13, dim), float32."""
    stream = _wne.residual_stream(
        model, images, device, RESIDUAL_STREAM_BATCH
    )  # (n, 13, tokens, dim)
    cls_stream = stream[:, :, 0, :]  # (n, 13, dim)
    return cls_stream


def onset_layer(rel_norm: list[float]) -> int | None:
    """The first layer > 0 whose relative direction norm reaches half the final layer's.

    The same rule scripts/paper/mech_tac_layers.py reads from the per-layer TAC
    pass, so an onset layer here means the same thing it means in the paper.
    """
    final = rel_norm[-1]
    for layer, value in enumerate(rel_norm):
        if layer > 0 and value >= 0.5 * final:
            return layer
    return None


def direction_profile(cls_clean: torch.Tensor, cls_triggered: torch.Tensor) -> dict:
    """Per-layer relative direction norm and TAC, plus the peak and onset layer.

    cls_clean and cls_triggered are (n, 13, dim), index-aligned pairs.
    """
    rel_norm, tac_mean = [], []
    for layer in range(NUM_LAYERS + 1):
        clean_layer = cls_clean[:, layer, :]  # (n, dim)
        triggered_layer = cls_triggered[:, layer, :]  # (n, dim)
        direction = backdoor_direction(clean_layer, triggered_layer)  # (dim,)
        tac = trigger_activated_change(clean_layer, triggered_layer)  # (dim,)
        # Divided by the mean clean norm so growth is not the residual stream
        # itself growing, the same normalisation run_tac_layers.py uses.
        scale = clean_layer.norm(dim=1).mean().clamp_min(DIRECTION_NORM_FLOOR)
        rel_norm.append(float(direction.norm() / scale))
        tac_mean.append(float(tac.mean()))

    peak_layer = max(range(len(rel_norm)), key=rel_norm.__getitem__)
    profile = {
        "rel_direction_norm": rel_norm,
        "tac_mean": tac_mean,
        "peak_layer": peak_layer,
        "peak_rel_direction_norm": rel_norm[peak_layer],
        "onset_layer": onset_layer(rel_norm),
        "final_tac_mean": tac_mean[-1],
    }
    return profile


@torch.inference_mode()
def predicted_class_margin(
    model: torch.nn.Module, images: torch.Tensor, device: torch.device, batch_size: int
) -> float:
    """Mean logit margin of the predicted class over the runner-up, no perturbation.

    images is (n, C, H, W). Autocast off, the same fp32 pass direction_profile
    reads its features from, since a margin compared across bf16 and fp32 passes
    would conflate the amplification question with rounding.
    """
    margins = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        logits = forward_logits(
            model, batch, device, use_bfloat16=False
        )  # (batch, classes)
        top2 = logits.topk(2, dim=1).values  # (batch, 2)
        margins.append((top2[:, 0] - top2[:, 1]).cpu())
    margin = float(torch.cat(margins).mean())
    return margin


def read_adaptive_rate(results_dir: str, folder: str) -> float:
    """The 0.8-rule rate for token_mask at before_attention_norm, or the fallback."""
    path = os.path.join(results_dir, folder, "psbd_metrics.json")
    if not os.path.exists(path):
        return FALLBACK_RATE
    with open(path) as handle:
        metrics = json.load(handle)
    placement = metrics["placements"].get(f"{POSITION}_{OPERATOR}")
    if placement is None or placement.get("adaptive_rate") is None:
        return FALLBACK_RATE
    rate = float(placement["adaptive_rate"])
    return rate


def image_loader(images: torch.Tensor, batch_size: int) -> DataLoader:
    """A shuffle=False loader over a plain image tensor, dummy labels unused by PSU."""
    dummy_labels = torch.zeros(len(images), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(images, dummy_labels), batch_size=batch_size, shuffle=False
    )
    return loader


def concatenated_baseline(cache: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-batch baseline cache stitched into 1 (n, classes) and 1 (n,) tensor."""
    probs = torch.cat([batch["probs"] for batch in cache])  # (n, classes)
    labels = torch.cat([batch["labels"] for batch in cache])  # (n,)
    return probs, labels


def psu_ratio_pass(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
    rate: float,
) -> torch.Tensor:
    """psu_ratio for a plain image tensor under token_mask at before_attention_norm.

    The same 3 calls cli/sweep.py and cli/analyze.py make: build the no-perturbation
    baseline, plug the probe, run the perturbed passes, unplug, score.
    """
    loader = image_loader(images, batch_size)
    baseline_cache = build_baseline_cache(model, loader, device, use_bfloat16=True)
    baseline_probs, baseline_labels = concatenated_baseline(baseline_cache)

    check_operator_position(OPERATOR, POSITION)
    handles = plug_dropout(
        model, "vit", (POSITION,), {POSITION: build_operator(OPERATOR)}, rate
    )
    try:
        per_pass_probs, _ = compute_dropout_pass_probs(
            model,
            loader,
            baseline_labels,
            device,
            FORWARD_PASSES,
            use_bfloat16=True,
            seed=PSBD_MASK_SEED,
        )  # (passes, n)
    finally:
        unplug_dropout(handles)

    ratio = psu_ratio_from_cache(
        baseline_probs, baseline_labels, per_pass_probs
    )  # (n,)
    return ratio


def separation_auroc(clean_ratio: torch.Tensor, triggered_ratio: torch.Tensor) -> float:
    """AUROC separating clean from triggered psu_ratio, low score the poisoned side.

    Mirrors defences.decision.detection_report's own convention: both score sets
    are negated because low psu_ratio is the positive (poisoned) evidence and
    roc_auc_score expects higher to mean more positive.
    """
    scores = torch.cat([-clean_ratio, -triggered_ratio]).numpy()
    labels = torch.cat(
        [torch.zeros(len(clean_ratio)), torch.ones(len(triggered_ratio))]
    ).numpy()
    auroc = float(roc_auc_score(labels, scores))
    return auroc


def analyse_checkpoint(
    folder: str, args: argparse.Namespace, device: torch.device
) -> dict:
    """Every number for 1 checkpoint: direction profile, margins and PSU separation."""
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        max_samples=args.max_samples,
    )
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device)

    pairs = _wne.paired_rows(loaders, manifest)
    clean_images = pairs["clean"][:PAIR_COUNT]  # (PAIR_COUNT, C, H, W)
    triggered_images = pairs["backdoor"][:PAIR_COUNT]  # (PAIR_COUNT, C, H, W)

    cls_clean = class_token_stream(model, clean_images, device)  # (PAIR_COUNT, 13, dim)
    cls_triggered = class_token_stream(model, triggered_images, device)
    profile = direction_profile(cls_clean, cls_triggered)

    clean_margin = predicted_class_margin(model, clean_images, device, args.batch_size)
    triggered_margin = predicted_class_margin(
        model, triggered_images, device, args.batch_size
    )

    rate = read_adaptive_rate(args.results_dir, folder)
    clean_ratio = psu_ratio_pass(model, clean_images, device, args.batch_size, rate)
    triggered_ratio = psu_ratio_pass(
        model, triggered_images, device, args.batch_size, rate
    )
    auroc = separation_auroc(clean_ratio, triggered_ratio)

    report = {
        "folder": folder,
        "attack": metadata["attack"],
        "dataset": metadata["dataset"],
        "poison_rate": metadata.get("poison_rate"),
        "optimizer": metadata.get("optimizer"),
        "rho": metadata.get("rho"),
        "n_pairs": len(clean_images),
        "psbd_tm_rate": rate,
        **profile,
        "clean_margin": clean_margin,
        "triggered_margin": triggered_margin,
        "psu_ratio_mean_clean": float(clean_ratio.mean()),
        "psu_ratio_mean_triggered": float(triggered_ratio.mean()),
        "psu_ratio_auroc": auroc,
    }
    return report


def pair_summary_line(label: str, adam_report: dict, sam_report: dict) -> str:
    """1 line: peak and onset layer, margins and PSU separation, Adam vs SAM."""
    line = (
        f"{label:28s} Adam peak L{adam_report['peak_layer']:2d} "
        f"({adam_report['peak_rel_direction_norm']:.2f}) onset L{adam_report['onset_layer']} "
        f"margin {adam_report['clean_margin']:.2f}/{adam_report['triggered_margin']:.2f} "
        f"AUROC {adam_report['psu_ratio_auroc']:.3f} | "
        f"SAM peak L{sam_report['peak_layer']:2d} "
        f"({sam_report['peak_rel_direction_norm']:.2f}) onset L{sam_report['onset_layer']} "
        f"margin {sam_report['clean_margin']:.2f}/{sam_report['triggered_margin']:.2f} "
        f"AUROC {sam_report['psu_ratio_auroc']:.3f}"
    )
    return line


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for label, adam_folder, sam_folder in PAIRS:
        adam_report = analyse_checkpoint(adam_folder, args, device)
        sam_report = analyse_checkpoint(sam_folder, args, device)
        pair_report = {"pair": label, "adam": adam_report, "sam": sam_report}

        path = experiment_result_path(SLUG, f"{label}.json", args.results_dir)
        with open(path, "w") as handle:
            json.dump(pair_report, handle, indent=2)
        print(pair_summary_line(label, adam_report, sam_report), flush=True)


if __name__ == "__main__":
    main()
