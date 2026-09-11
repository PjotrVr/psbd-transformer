"""Why token masking at the attention input (site A) fails on 3 ViT models.

CLAUDE.md's headline placement, `before_attention_norm_token_mask` (site A), reads
CIFAR-10 WaNet 10% at AUROC 0.459, CIFAR-10 SIG 10% at 0.418 and Tiny TaCT 1% at
0.575, while `before_attention_residual_token_mask` (site B, masking the attention
branch's output rather than its input) and the residual-stream dropout placements
(`pre_residual`, `post_residual`) read the same 3 models at 0.85 to 0.93. The same
attacks read site A well elsewhere: CIFAR-10 WaNet 5% (0.936), GTSRB WaNet 10%
(0.948), CIFAR-100 TaCT 1% (0.909). This script measures why, on the 6 models
(3 failing, 3 controls, `MODELS` below), in 4 parts.

1. From results/<folder>/psbd_metrics.json, the rate ladder of site A, site B,
   `pre_residual` and `post_residual`: AUROC and TPR at the quantile-0.10 rule
   against the achieved clean-validation shift ratio at every swept rate, plus
   the rate the 0.8 adaptive rule selects. Pure CPU, no cache recompute.
2. From the caches, the per-image response to site A and to `post_residual` at
   each site's own adaptively selected rate: PSU and psu_ratio of clean and
   triggered inputs, and the share of triggered inputs (attack-success captured
   only) whose per-pass argmax ever leaves the baseline class under the
   perturbation, against the same share for clean inputs. Also CPU only,
   generalising panel_data in scripts/paper/fig_psu_histograms.py to any
   placement.
3. On the GPU, 500 paired clean and triggered images per model
   (experiments.whole_network_erasure.measure.paired_rows): the logit margin of
   the predicted class over the runner-up with no perturbation
   (experiments.sam_mechanism.measure.predicted_class_margin), since a weak
   shortcut with high ASR is the case where PSBD's own statistic reads a coin.
4. Activation patching restricted to the trigger's own token group
   (experiments.residual_stream_mechanism.activation_patching), against the CLS
   token and a random group of the same size, at blocks 4, 8 and 12, at
   site="resid" and site="attn". The trigger token group is attack-specific: for
   WaNet the 20% of patches the warp field displaces most (read from
   attacks.wanet's own grid builder at the checkpoint's parameters), for SIG the
   20% of patches the column sinusoid peaks in, for TaCT the 4 corner patches
   the checkerboard patch sits in one of.

    PYTHONPATH=. .venv/bin/python experiments/failure_modes/measure.py \
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
import torch.nn.functional as F  # noqa: E402

from attacks import apply_config_overrides, default_config  # noqa: E402
import attacks.sig as sig_attack  # noqa: E402
import attacks.wanet as wanet_attack  # noqa: E402
from data.registry import DATASET_REGISTRY  # noqa: E402
from data.splits import SPLITS, read_checkpoint_metadata  # noqa: E402
from defences.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import attack_success_mask, pair_clean_to_backdoor  # noqa: E402
from defences.scores import psu_from_cache, psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from models.backbones import load_checkpoint, network_core  # noqa: E402
from utils.numerics import safe_ratio  # noqa: E402


# whole_network_erasure, sam_mechanism and the patching experiment are not
# packages (no __init__.py), so their reusable pieces are loaded by path rather
# than reimplemented: paired_rows and residual_stream build the exact clean and
# triggered image pairs the rest of the PSBD-ViT analysis already uses,
# predicted_class_margin is the GPU margin reader, cache_stream and patch_hook
# are the causal-tracing primitives.
def _load_module(relative_path: str, name: str):
    path = os.path.join(REPO_ROOT, "experiments", relative_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_wne = _load_module("whole_network_erasure/measure.py", "wne_measure")
_sam = _load_module("sam_mechanism/measure.py", "sam_measure")
_patch = _load_module(
    "residual_stream_mechanism/activation_patching.py", "patch_measure"
)

SLUG = "failure_modes"
SITE_A = "before_attention_norm_token_mask"
SITE_B = "before_attention_residual_token_mask"
PLACEMENTS = (SITE_A, SITE_B, "pre_residual", "post_residual")
LADDER_QUANTILE = "q0.10"

# The 3 failing models and their 3 controls, 1 per attack family, matching the
# plan's own reading of the failures and the CLAUDE.md headline.
MODELS: tuple[dict, ...] = (
    {"folder": "vit_cifar10_wanet_0_1", "role": "failing"},
    {"folder": "vit_cifar10_sig_0_1", "role": "failing"},
    {"folder": "vit_tiny_tact_0_01", "role": "failing"},
    {"folder": "vit_cifar10_wanet_0_05", "role": "control"},
    {"folder": "vit_gtsrb_wanet_0_1", "role": "control"},
    {"folder": "vit_cifar100_tact_0_01", "role": "control"},
)

PAIR_COUNT = 500
PATCH_BATCH_SIZE = 32
PATCH_LAYERS = (4, 8, 12)
PATCH_SITES = ("resid", "attn")
TOKEN_TOP_FRACTION = 0.2
# 2 shares within this of each other read as "nothing separates" rather than a
# directional gap, since the caches carry sampling noise at 500 to 2000 images.
NOTHING_SEPARATES_MARGIN = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for spec in MODELS:
        folder = spec["folder"]
        metrics = load_psbd_metrics(args.results_dir, folder)

        rate_ladder = {
            placement: ladder_rows(metrics, placement) for placement in PLACEMENTS
        }
        per_image = {
            placement: per_image_summary(args.results_dir, folder, metrics, placement)
            for placement in (SITE_A, "post_residual")
        }
        margin, patching = gpu_measurements(folder, args, device)

        report = {
            "folder": folder,
            "role": spec["role"],
            "dataset": metrics["dataset"],
            "attack": metrics["attack"],
            "poison_rate": metrics["poison_rate"],
            "rate_ladder": rate_ladder,
            "per_image": per_image,
            "margin": margin,
            "activation_patching": patching,
        }

        path = experiment_result_path(SLUG, f"{folder}.json", args.results_dir)
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2)

        print(summary_line(report), flush=True)


def load_psbd_metrics(results_dir: str, folder: str) -> dict:
    """A checkpoint's psbd_metrics.json, the stage-2 record cli.analyze wrote."""
    path = os.path.join(results_dir, folder, "psbd_metrics.json")
    with open(path) as handle:
        metrics = json.load(handle)
    return metrics


def ladder_rows(metrics: dict, placement: str) -> dict:
    """1 placement's full rate ladder: shift ratio against AUROC and TPR at q0.10.

    Every rate cli.sweep wrote for this checkpoint and placement, in ascending
    order, so a reader can see where the placement's clean-validation shift
    saturates against where its AUROC keeps moving. Reads detection_psu_ratio,
    the fractional-PSU field CLAUDE.md's canon names the headline statistic,
    never the sibling "detection" field, which is the absolute-PSU form kept
    beside it.
    """
    block = metrics["placements"][placement]
    rows = [
        {
            "rate": row["rate"],
            "shift_ratio_validation": row["shift_ratio"]["validation"],
            "auroc_q10": row["detection_psu_ratio"][LADDER_QUANTILE]["auroc"],
            "tpr_q10": row["detection_psu_ratio"][LADDER_QUANTILE]["tpr"],
            "fpr_q10": row["detection_psu_ratio"][LADDER_QUANTILE]["fpr"],
        }
        for row in sorted(block["rates"], key=lambda row: row["rate"])
    ]
    ladder = {"rows": rows, "adaptive_rate": block["adaptive_rate"]}
    return ladder


def per_image_summary(
    results_dir: str, folder: str, metrics: dict, placement: str
) -> dict:
    """1 placement's per-image PSU and shift behaviour, at its own adaptive rate.

    Generalises panel_data in scripts/paper/fig_psu_histograms.py to any
    placement rather than only RECOMMENDED_PLACEMENT, and adds the per-image
    shift share panel_data does not compute.
    """
    block = metrics["placements"][placement]
    rate = block["adaptive_rate"]
    if rate is None:
        return {"rate": None, "note": "never reaches the adaptive shift target"}

    psbd_dir = os.path.join(results_dir, folder, "psbd")
    manifest = read_split_manifest(psbd_dir)

    baseline = {
        split: load_baseline(baseline_path(psbd_dir, split)) for split in SPLITS
    }
    passes = {
        split: load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        for split in SPLITS
    }

    clean_probs, clean_labels, _ = baseline["clean"]
    backdoor_probs, backdoor_labels, backdoor_loader_labels = baseline["backdoor"]
    clean_pass_probs, clean_pass_argmax = passes["clean"]
    backdoor_pass_probs, backdoor_pass_argmax = passes["backdoor"]

    psu_clean = psu_from_cache(
        clean_probs, clean_labels, clean_pass_probs
    )  # (n_clean,)
    psu_backdoor = psu_from_cache(
        backdoor_probs, backdoor_labels, backdoor_pass_probs
    )  # (n_backdoor,)
    psu_ratio_clean = psu_ratio_from_cache(
        clean_probs, clean_labels, clean_pass_probs
    )  # (n_clean,)
    psu_ratio_backdoor = psu_ratio_from_cache(
        backdoor_probs, backdoor_labels, backdoor_pass_probs
    )  # (n_backdoor,)

    # Whether ANY of the k perturbed passes moved the prediction away from the
    # no-perturbation label, the per-image version of defences.scores.shift_ratio
    # (which only reports the (sample, pass)-averaged fraction).
    clean_shifted = (clean_pass_argmax.long() != clean_labels.view(1, -1).long()).any(
        dim=0
    )  # (n_clean,)
    backdoor_shifted = (
        backdoor_pass_argmax.long() != backdoor_labels.view(1, -1).long()
    ).any(dim=0)  # (n_backdoor,)

    paired_clean_psu = pair_clean_to_backdoor(psu_clean, manifest)  # (n_backdoor,)
    paired_clean_psu_ratio = pair_clean_to_backdoor(psu_ratio_clean, manifest)
    paired_clean_shifted = pair_clean_to_backdoor(
        clean_shifted.float(), manifest
    ).bool()

    # A triggered image the model still classifies correctly was never captured
    # by the backdoor, so scoring its shift as the attack's evidence charges the
    # detector for a failure that is not its own (same rule cli.analyze applies
    # to detection_captured_only).
    captured = attack_success_mask(backdoor_labels, backdoor_loader_labels)
    if captured is not None and bool(captured.any()):
        clean_share = float(paired_clean_shifted[captured].float().mean())
        triggered_share = float(backdoor_shifted[captured].float().mean())
        psu_clean_mean = float(paired_clean_psu[captured].mean())
        psu_backdoor_mean = float(psu_backdoor[captured].mean())
        psu_ratio_clean_mean = float(paired_clean_psu_ratio[captured].mean())
        psu_ratio_backdoor_mean = float(psu_ratio_backdoor[captured].mean())
        n_captured = int(captured.sum())
    else:
        clean_share = float(paired_clean_shifted.float().mean())
        triggered_share = float(backdoor_shifted.float().mean())
        psu_clean_mean = float(paired_clean_psu.mean())
        psu_backdoor_mean = float(psu_backdoor.mean())
        psu_ratio_clean_mean = float(paired_clean_psu_ratio.mean())
        psu_ratio_backdoor_mean = float(psu_ratio_backdoor.mean())
        n_captured = 0

    gap = triggered_share - clean_share
    if abs(gap) < NOTHING_SEPARATES_MARGIN:
        classification = "nothing_separates"
    elif gap > 0:
        classification = "wrong_direction"
    else:
        classification = "as_expected"

    summary = {
        "rate": rate,
        "n_captured": n_captured,
        "psu_clean_mean": psu_clean_mean,
        "psu_backdoor_mean": psu_backdoor_mean,
        "psu_ratio_clean_mean": psu_ratio_clean_mean,
        "psu_ratio_backdoor_mean": psu_ratio_backdoor_mean,
        "clean_shift_share": clean_share,
        "triggered_shift_share": triggered_share,
        "shift_share_gap": gap,
        "classification": classification,
    }
    return summary


def gpu_measurements(
    folder: str, args: argparse.Namespace, device: torch.device
) -> tuple[dict, dict]:
    """The margin reading and the activation patching for 1 checkpoint.

    Loads the checkpoint and builds its 500 paired clean/triggered images once,
    reused by both measurements.
    """
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    loaders, manifest = build_loaders(checkpoint_path, args)
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device).eval()

    pairs = _wne.paired_rows(loaders, manifest)
    clean_images = pairs["clean"][:PAIR_COUNT]  # (PAIR_COUNT, C, H, W)
    triggered_images = pairs["backdoor"][:PAIR_COUNT]  # (PAIR_COUNT, C, H, W)

    clean_margin = _sam.predicted_class_margin(
        model, clean_images, device, args.batch_size
    )
    triggered_margin = _sam.predicted_class_margin(
        model, triggered_images, device, args.batch_size
    )
    margin = {
        "clean_margin": clean_margin,
        "triggered_margin": triggered_margin,
        "n_pairs": int(len(clean_images)),
    }

    patching = patching_summary(metadata, model, clean_images, triggered_images, args)
    return margin, patching


def build_loaders(checkpoint_path: str, args: argparse.Namespace):
    """The PSBD loaders and split manifest for 1 checkpoint, shuffle=False."""
    loaders, manifest = _wne.build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        max_samples=args.max_samples,
    )
    return loaders, manifest


def patch_scores(pixel_map: torch.Tensor) -> torch.Tensor:
    """Per-patch summed magnitude of a native-resolution (H, W) map, (196,).

    Upsamples to the model's 224x224 input the same way the trigger detects its
    own footprint in activation_patching.trigger_tokens, then sums each 16x16
    patch, so any per-pixel signal (a warp displacement, a sinusoid amplitude)
    can be ranked patch by patch.
    """
    upsampled = F.interpolate(
        pixel_map.unsqueeze(0).unsqueeze(0).float(),
        size=(_patch.MODEL_INPUT, _patch.MODEL_INPUT),
        mode="bilinear",
    )[0, 0]  # (224, 224)
    per_patch = (
        upsampled.reshape(_patch.GRID, _patch.PATCH, _patch.GRID, _patch.PATCH)
        .permute(0, 2, 1, 3)
        .reshape(-1, _patch.PATCH * _patch.PATCH)
    )  # (196, 256)
    scores = per_patch.sum(dim=1)  # (196,)
    return scores


def top_fraction_patches(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """The (k,) 0-indexed patch positions with the highest score, k = round(fraction * 196)."""
    k = max(1, round(fraction * len(scores)))
    top = scores.topk(k).indices
    return top


def wanet_trigger_tokens(metadata: dict) -> torch.Tensor:
    """The 20% of patches WaNet's warp field displaces most.

    Rebuilds the exact grid attacks.wanet.build uses for this checkpoint (its
    own defaults, since none of the 3 WaNet models carry a config override) and
    ranks patches by the norm of grid_sample offset from the identity grid,
    rather than by a pixel diff, since a smooth warp with padding_mode="border"
    can leave large flat regions pixel-identical while still resampling from a
    displaced source location.
    """
    image_size = DATASET_REGISTRY[metadata["dataset"]].image_size
    config = apply_config_overrides(
        default_config("wanet"), metadata.get("attack_config_overrides")
    )
    identity = wanet_attack._identity_grid(image_size)  # (1, H, W, 2)
    warped = wanet_attack._warping_grid(
        image_size, config.control_grid_size, config.strength, config.field_seed
    )  # (1, H, W, 2)
    displacement = (warped - identity)[0].norm(dim=-1)  # (H, W)
    scores = patch_scores(displacement)  # (196,)
    tokens = top_fraction_patches(scores, TOKEN_TOP_FRACTION)
    return tokens


def sig_trigger_tokens(metadata: dict) -> torch.Tensor:
    """The 20% of patches SIG's column sinusoid is closest to its peak amplitude.

    The signal is constant down every column and varies with column position
    only (attacks.sig._column_signal), so its patch ranking is periodic in the
    column axis and picks out whichever columns of the 16x16 grid sit nearest
    the sinusoid's crests and troughs.
    """
    image_size = DATASET_REGISTRY[metadata["dataset"]].image_size
    config = apply_config_overrides(
        default_config("sig"), metadata.get("attack_config_overrides")
    )
    signal = sig_attack._column_signal(
        image_size, config.amplitude, config.frequency
    )  # (1, 1, image_size)
    pixel_map = signal[0, 0].abs().unsqueeze(0).expand(image_size, image_size)  # (H, W)
    scores = patch_scores(pixel_map)  # (196,)
    tokens = top_fraction_patches(scores, TOKEN_TOP_FRACTION)
    return tokens


def tact_trigger_tokens() -> torch.Tensor:
    """The 4 corner patches of the 14x14 grid.

    TaCT's checkerboard patch sits in the bottom-right corner of the native
    image (attacks.tact.build), which after resizing to 224 falls inside 1 or 2
    patches depending on the dataset's native resolution. All 4 grid corners are
    patched together so the group's size does not depend on that resizing
    detail and stays comparable across the failing and the control TaCT model.
    """
    grid = _patch.GRID
    corners = torch.tensor(
        [0, grid - 1, grid * (grid - 1), grid * grid - 1], dtype=torch.long
    )
    return corners


TOKEN_GROUP_BUILDERS = {
    "wanet": wanet_trigger_tokens,
    "sig": sig_trigger_tokens,
}


def trigger_token_group(metadata: dict) -> torch.Tensor:
    """The attack-specific trigger token group, 0-indexed into the 196 patches."""
    attack = metadata["attack"]
    if attack == "tact":
        return tact_trigger_tokens()
    builder = TOKEN_GROUP_BUILDERS.get(attack)
    if builder is None:
        raise ValueError(f"no trigger-token definition for attack {attack!r}")
    return builder(metadata)


def patch_sweep(
    model,
    core,
    images_from: torch.Tensor,
    donor_cache: dict,
    target: torch.Tensor,
    true_label: torch.Tensor,
    baseline: torch.Tensor,
    reference: torch.Tensor,
    groups: dict[str, torch.Tensor],
) -> list[dict]:
    """Normalised recovery for every (site, layer, group) in PATCH_SITES x PATCH_LAYERS.

    A restricted rerun of activation_patching.sweep, which sweeps every layer.
    Blocks 4, 8 and 12 are the early, middle and late reference points the plan
    asks for, so the other 9 layers are skipped rather than computed and
    discarded.
    """
    rows = []
    with torch.inference_mode():
        for site in PATCH_SITES:
            for layer in PATCH_LAYERS:
                block = core.encoder.layers[layer - 1]
                donor = donor_cache[(site, layer - 1)]
                for name, positions in groups.items():
                    handle = _patch.patch_hook(block, site, donor, positions)
                    logits = model(images_from)
                    handle.remove()
                    metric = (
                        logits.gather(1, target[:, None])
                        - logits.gather(1, true_label[:, None])
                    ).squeeze(1)  # (batch,)
                    rows.append(
                        {
                            "site": site,
                            "layer": layer,
                            "group": name,
                            "numerator": float((metric - baseline).sum()),
                            "denominator": float((reference - baseline).sum()),
                            "n": int(len(metric)),
                        }
                    )
    return rows


def patching_summary(
    metadata: dict,
    model,
    clean_images: torch.Tensor,
    triggered_images: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    """Recovery of the clean answer at the trigger group, the CLS token and a random group."""
    core = network_core(model)
    target_label = metadata["target_label"]

    tokens = trigger_token_group(metadata)  # (k,) 0-indexed patches
    generator = torch.Generator().manual_seed(args.seed)
    random_group = torch.randperm(196, generator=generator)[: len(tokens)] + 1
    groups = {
        "cls": torch.tensor([0]),
        "trigger": tokens + 1,
        "random_same_size": random_group,
    }

    collected = []
    for start in range(0, len(triggered_images), PATCH_BATCH_SIZE):
        end = start + PATCH_BATCH_SIZE
        clean_batch = clean_images[start:end].to(next(model.parameters()).device)
        triggered_batch = triggered_images[start:end].to(
            next(model.parameters()).device
        )
        if len(clean_batch) == 0 or len(triggered_batch) != len(clean_batch):
            break

        with torch.inference_mode():
            clean_cache, clean_logits = _patch.cache_stream(model, core, clean_batch)
            triggered_cache, triggered_logits = _patch.cache_stream(
                model, core, triggered_batch
            )
            true_label = clean_logits.argmax(dim=1)
            target = torch.full_like(true_label, target_label)
            m_clean = (
                clean_logits.gather(1, target[:, None])
                - clean_logits.gather(1, true_label[:, None])
            ).squeeze(1)
            m_triggered = (
                triggered_logits.gather(1, target[:, None])
                - triggered_logits.gather(1, true_label[:, None])
            ).squeeze(1)

        collected += patch_sweep(
            model,
            core,
            triggered_batch,
            clean_cache,
            target,
            true_label,
            m_triggered,
            m_clean,
            groups,
        )

    merged: dict[tuple[str, int, str], dict] = {}
    for row in collected:
        key = (row["site"], row["layer"], row["group"])
        entry = merged.setdefault(key, {"num": 0.0, "den": 0.0, "n": 0})
        entry["num"] += row["numerator"]
        entry["den"] += row["denominator"]
        entry["n"] += row["n"]

    rows = [
        {
            "site": site,
            "layer": layer,
            "group": group,
            "recovery": safe_ratio(entry["num"], entry["den"], floor=1e-6),
            "n": entry["n"],
        }
        for (site, layer, group), entry in sorted(merged.items())
    ]
    summary = {"n_trigger_tokens": int(len(tokens)), "rows": rows}
    return summary


def summary_line(report: dict) -> str:
    """1 line: role, site A and post_residual's adaptive AUROC, margin, patching at block 8."""
    a = report["rate_ladder"][SITE_A]
    post = report["rate_ladder"]["post_residual"]
    a_auroc = next(
        (row["auroc_q10"] for row in a["rows"] if row["rate"] == a["adaptive_rate"]),
        None,
    )
    post_auroc = next(
        (
            row["auroc_q10"]
            for row in post["rows"]
            if row["rate"] == post["adaptive_rate"]
        ),
        None,
    )
    trigger_recovery_block8_resid = next(
        (
            row["recovery"]
            for row in report["activation_patching"]["rows"]
            if row["site"] == "resid"
            and row["layer"] == 8
            and row["group"] == "trigger"
        ),
        float("nan"),
    )
    line = (
        f"{report['folder']:28s} [{report['role']:7s}] "
        f"A={a_auroc if a_auroc is not None else float('nan'):.3f} "
        f"post_residual={post_auroc if post_auroc is not None else float('nan'):.3f} "
        f"margin clean/triggered="
        f"{report['margin']['clean_margin']:.2f}/{report['margin']['triggered_margin']:.2f} "
        f"trigger-recovery(resid, block8)={trigger_recovery_block8_resid:.3f}"
    )
    return line


if __name__ == "__main__":
    main()
