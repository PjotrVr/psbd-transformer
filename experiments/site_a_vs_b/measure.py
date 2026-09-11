"""Site A (attention input) against site B (attention branch output), same operator, 2 places.

`before_attention_norm_token_mask` (site A, PSBD-TM, the recommended placement) and
`before_attention_residual_token_mask` (site B) both score mean AUROC around 0.935 on the
65 clearing cells, which reads as the same placement seen twice. Per model the picture
disagrees: B beats A by 0.52 on CIFAR-10 SIG 10% and 0.39 on CIFAR-10 WaNet 10%, A beats B
by 0.39 on Tiny WaNet 5% and 0.24 on GTSRB WaNet 10%. This script asks whether that is 2
probes reading the same signal at different sensitivity, or 2 probes that catch different
poisoned images.

Method, in 4 parts.

(1) Per model, per-image fractional PSU (`defences.scores.psu_ratio_from_cache`) is read
    from the stage-1 caches at each site's own `adaptive_rate`
    (`results/<folder>/psbd_metrics.json`, `select_rate_adaptively` at
    `ADAPTIVE_SHIFT_TARGET`), generalising `panel_data` in
    `scripts/paper/fig_psu_histograms.py` to any placement id (`read_placement_psu` below
    imports the same helpers from `defences.cache`, `defences.decision` and
    `defences.scores`, it does not recompute the statistic). Per model: AUROC of site A,
    of site B, of their min-rank union (`defences.decision.multi_probe_auroc`, the rule
    `experiments/probe_union/measure.py` uses), the best of the 2 single sites, the
    Spearman correlation of the 2 per-image PSU vectors on the triggered images and on the
    clean images paired to them, the share of triggered images caught by exactly 1 site at
    the 25% quantile threshold (`HEADLINE_QUANTILE`), and each site's achieved rate and
    clean-validation shift ratio. Grouped by attack and by trigger locality (local:
    badnet_a2o, badnet_a2a, tact; global: blend, sig, wanet, lf, bpp).

(2) The same 2 sites read at matched achieved clean shift ratio (0.6, 0.7, 0.8, 0.9),
    interpolating each site's own rate ladder onto that shift with
    `defences.decision.interpolate_at_target_shift`, the way
    `scripts/paper/fig_shift_ladder.py` reads the recommended placement's ladder, so a
    per-model gap caused by the adaptive rule landing on different rates is separated from
    a gap in the site itself.

(3) The mechanism, on 4 models (CIFAR-10 WaNet 10%, CIFAR-10 SIG 10%, CIFAR-100 BadNets 1%,
    Tiny WaNet 5%), 200 paired clean/triggered images, at each site's own selected rate.
    Site A is a pre-hook on the block's first LayerNorm (`ln_1`): the mask lands before the
    norm, the norm re-normalises every surviving token and turns a zeroed token into the
    LayerNorm of a zero vector, a constant embedding that still enters attention as key and
    value. Site B is a post-hook on the block's attention-output dropout module, right
    before the residual add: it zeroes only what attention wrote to the token in this
    block and keeps the token's residual content. `experiments.whole_network_erasure
    .measure.directions` (imported, not copied) gives the backdoor direction at each
    block's output. Measured with the project's own perturbation machinery
    (`models.positions.plug_dropout` restricted to 1 block with `defences.operators
    .build_operator("token_mask")`), never a re-implementation of the operator: the
    relative-norm change of the block's output and of its class token, on clean and on
    triggered inputs, and the share of that change lying along the backdoor direction.

(4) A re-read of `results/<folder>/activation_patching.json` for those 4 models: recovery
    of the clean answer under trigger-token patching at `site="resid"` (A) against
    `site="attn"` (B), per layer, the layout `scripts/paper/mech_activation_patching.py`
    reads.

`compare_sites` takes any 2 placement ids, so the same measurement runs for whichever
operator already exists at both sites. 4 pairs currently qualify (at least 10 models on
each side, before intersecting): token_mask (the pair above), dropout
(`before_attention_norm` against `before_attention_residual`, the operator's unsuffixed
placement id, `cli.sweep.cache_config_name`), channel_mask and gaussian.

    PYTHONPATH=. .venv/bin/python experiments/site_a_vs_b/measure.py
    PYTHONPATH=. .venv/bin/python experiments/site_a_vs_b/measure.py --skip-mechanism
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from data.splits import SPLITS  # noqa: E402
from defences.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import (  # noqa: E402
    HEADLINE_QUANTILE,
    RECOMMENDED_PLACEMENT,
    detection_report,
    interpolate_at_target_shift,
    multi_probe_auroc,
    pair_clean_to_backdoor,
    threshold_at_quantile,
)
from defences.operators import build_operator  # noqa: E402
from defences.scores import psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.whole_network_erasure.measure import (  # noqa: E402
    directions,
    paired_rows,
    residual_stream,
)
from models.backbones import load_checkpoint  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402
from data.splits import (  # noqa: E402
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from scripts.paper._common import (  # noqa: E402
    bootstrap_ci,
    clearing_cells,
    load_coverage,
    load_json,
    load_psbd_metrics,
    mean_or_none,
    rate_row,
)

SLUG = "site_a_vs_b"

SITE_A = RECOMMENDED_PLACEMENT  # before_attention_norm_token_mask
SITE_B = "before_attention_residual_token_mask"
# models.positions.POSITION_REGISTRY keys off the bare position, not the placement id
# (which folds in the operator suffix for cache naming), so the mechanism part plugs
# these directly.
SITE_A_POSITION = "before_attention_norm"
SITE_B_POSITION = "before_attention_residual"

# Every (operator, site A id, site B id) that already has a sweep on both sites. dropout's
# placement id carries no operator suffix (cli.sweep.cache_config_name: the suffix is
# dropped exactly when the operator is "dropout", the unmarked default).
OPERATOR_PAIRS = {
    "token_mask": (SITE_A, SITE_B),
    "dropout": ("before_attention_norm", "before_attention_residual"),
    "channel_mask": (
        "before_attention_norm_channel_mask",
        "before_attention_residual_channel_mask",
    ),
    "gaussian": (
        "before_attention_norm_gaussian",
        "before_attention_residual_gaussian",
    ),
}
MIN_MODELS_PER_PAIR = 10

LOCAL_ATTACKS = ("badnet_a2o", "badnet_a2a", "tact")
GLOBAL_ATTACKS = ("blend", "sig", "wanet", "lf", "bpp")

SHIFT_TARGETS = (0.6, 0.7, 0.8, 0.9)
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 0

# Part 3, the mechanism models named in the question, 1 per hard corner of the panel.
MECHANISM_MODELS = (
    "vit_cifar10_wanet_0_1",
    "vit_cifar10_sig_0_1",
    "vit_cifar100_badnet_a2o_0_01",
    "vit_tiny_wanet_0_05",
)
MECHANISM_BLOCKS = (
    1,
    4,
    6,
    9,
    12,
)  # 1-indexed, the layernorm_absorption.py depth sample
MECHANISM_PAIRS = 200  # split 100 for direction estimation, 100 for the measured change
MECHANISM_SEED = 0
MECHANISM_BATCH = 32

# Part 4, the activation-patching groups and sites this re-read names by hand: the trigger
# tokens, at the residual-stream site (A) and the attention-branch site (B).
PATCHING_GROUP = "trigger"
PATCHING_SITE_A = "resid"
PATCHING_SITE_B = "attn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--skip-mechanism",
        action="store_true",
        help="skip parts 3 and 4 (GPU forward passes), for a CPU-only rerun of parts 1-2",
    )
    parser.add_argument(
        "--output", default=experiment_result_path(SLUG, "site_a_vs_b.json")
    )
    return parser.parse_args()


def locality(attack: str) -> str | None:
    """'local', 'global' or None for an attack the question's grouping leaves out."""
    if attack in LOCAL_ATTACKS:
        return "local"
    if attack in GLOBAL_ATTACKS:
        return "global"
    return None


def select_models(results_dir: str, site_a: str, site_b: str) -> list[dict]:
    """Clearing cells whose psbd_metrics.json reached an adaptive rate at both sites.

    Generalises experiments.probe_union.measure.select_models, which hardcodes the 2
    headline placements, to any pair.
    """
    coverage = load_coverage(results_dir)
    cells = clearing_cells(coverage)

    selected = []
    for cell in cells:
        report = load_psbd_metrics(results_dir, cell["folder_name"])
        if report is None:
            continue
        placements = report.get("placements", {})
        block_a = placements.get(site_a)
        block_b = placements.get(site_b)
        if not block_a or block_a.get("adaptive_rate") is None:
            continue
        if not block_b or block_b.get("adaptive_rate") is None:
            continue
        cell["report"] = report
        selected.append(cell)
    return selected


def read_placement_psu(
    results_dir: str, folder: str, report: dict, placement: str
) -> dict:
    """1 placement's per-image fractional PSU at its own adaptive rate, read from the cache.

    Generalises panel_data in scripts/paper/fig_psu_histograms.py (hardcoded to
    RECOMMENDED_PLACEMENT) to any placement id, importing the same cache and decision
    helpers rather than recomputing the statistic.
    """
    block = report["placements"][placement]
    rate = block["adaptive_rate"]

    psbd_dir = os.path.join(results_dir, folder, "psbd")
    manifest = read_split_manifest(psbd_dir)

    psu_by_split = {}
    for split in SPLITS:
        baseline_probs, baseline_labels, _ = load_baseline(
            baseline_path(psbd_dir, split)
        )
        per_pass_probs, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        psu_by_split[split] = psu_ratio_from_cache(
            baseline_probs, baseline_labels, per_pass_probs
        )  # (n_split,)

    clean_paired = pair_clean_to_backdoor(
        psu_by_split["clean"], manifest
    )  # (n_backdoor,)
    row = rate_row(block, rate)
    data = {
        "placement": placement,
        "rate": rate,
        "shift_ratio": row["shift_ratio"]["validation"] if row else None,
        "validation": psu_by_split["validation"],  # (n_validation,)
        "clean": clean_paired,  # (n_backdoor,)
        "backdoor": psu_by_split["backdoor"],  # (n_backdoor,)
    }
    return data


def best_of(*values: float) -> float:
    """The largest defined value, nan when every value is nan."""
    defined = [v for v in values if v == v]
    best = max(defined) if defined else float("nan")
    return best


def per_model_row(results_dir: str, cell: dict, site_a: str, site_b: str) -> dict:
    """1 model's AUROC of each site, their union, correlation and exclusive-catch shares."""
    folder = cell["folder_name"]
    report = cell["report"]
    data_a = read_placement_psu(results_dir, folder, report, site_a)
    data_b = read_placement_psu(results_dir, folder, report, site_b)

    report_a = detection_report(
        data_a["validation"], data_a["clean"], data_a["backdoor"], HEADLINE_QUANTILE
    )
    report_b = detection_report(
        data_b["validation"], data_b["clean"], data_b["backdoor"], HEADLINE_QUANTILE
    )
    auroc_union = multi_probe_auroc(
        [data_a["clean"], data_b["clean"]],
        [data_a["backdoor"], data_b["backdoor"]],
        [data_a["validation"], data_b["validation"]],
    )

    # Spearman on the images each site actually scored: the backdoor split, and the
    # clean split restricted to those same images (data_a/data_b["clean"] are both
    # already paired to the backdoor row order by read_placement_psu).
    triggered_corr, _ = spearmanr(
        data_a["backdoor"].numpy(), data_b["backdoor"].numpy()
    )
    clean_corr, _ = spearmanr(data_a["clean"].numpy(), data_b["clean"].numpy())

    threshold_a = threshold_at_quantile(data_a["validation"], HEADLINE_QUANTILE)
    threshold_b = threshold_at_quantile(data_b["validation"], HEADLINE_QUANTILE)
    caught_a = data_a["backdoor"] < threshold_a  # (n_backdoor,) bool
    caught_b = data_b["backdoor"] < threshold_b
    n_triggered = caught_a.numel()
    exclusive_a = float((caught_a & ~caught_b).float().mean()) if n_triggered else None
    exclusive_b = float((~caught_a & caught_b).float().mean()) if n_triggered else None
    exclusive_either = (
        float((caught_a ^ caught_b).float().mean()) if n_triggered else None
    )
    both = float((caught_a & caught_b).float().mean()) if n_triggered else None
    neither = float((~caught_a & ~caught_b).float().mean()) if n_triggered else None

    row = {
        "folder": folder,
        "dataset": report["dataset"],
        "attack": report["attack"],
        "locality": locality(report["attack"]),
        "poison_rate": report["poison_rate"],
        "n_triggered": n_triggered,
        "rate_a": data_a["rate"],
        "shift_a": data_a["shift_ratio"],
        "auroc_a": report_a["auroc"],
        "rate_b": data_b["rate"],
        "shift_b": data_b["shift_ratio"],
        "auroc_b": report_b["auroc"],
        "auroc_union": auroc_union,
        "auroc_best_of_2": best_of(report_a["auroc"], report_b["auroc"]),
        "spearman_triggered": float(triggered_corr),
        "spearman_clean": float(clean_corr),
        "exclusive_a_share": exclusive_a,
        "exclusive_b_share": exclusive_b,
        "exclusive_either_share": exclusive_either,
        "both_share": both,
        "neither_share": neither,
    }
    return row


def auroc_ladder(report: dict, placement: str) -> tuple[dict, dict] | None:
    """shift_by_rate and value_by_rate(auroc) for 1 placement, for shift-matched interpolation."""
    block = report.get("placements", {}).get(placement)
    if block is None:
        return None
    shift_by_rate, value_by_rate = {}, {}
    for row in block.get("rates", []):
        rate = row["rate"]
        shift_by_rate[rate] = row["shift_ratio"]["validation"]
        value_by_rate[rate] = row["detection_psu_ratio"][f"q{HEADLINE_QUANTILE:.2f}"][
            "auroc"
        ]
    if not shift_by_rate:
        return None
    return shift_by_rate, value_by_rate


def matched_shift_row(
    report: dict, site_a: str, site_b: str, target: float
) -> dict | None:
    """AUROC of each site interpolated onto 1 shared clean-validation shift ratio."""
    ladder_a = auroc_ladder(report, site_a)
    ladder_b = auroc_ladder(report, site_b)
    if ladder_a is None or ladder_b is None:
        return None
    auroc_a = interpolate_at_target_shift(ladder_a[0], ladder_a[1], target)
    auroc_b = interpolate_at_target_shift(ladder_b[0], ladder_b[1], target)
    if auroc_a is None or auroc_b is None:
        return None
    return {
        "auroc_a": auroc_a,
        "auroc_b": auroc_b,
        "delta_b_minus_a": auroc_b - auroc_a,
    }


def matched_shift_summary(models: list[dict], site_a: str, site_b: str) -> dict:
    """Mean AUROC of each site and the paired B-minus-A gap, at every shift target."""
    summary = {}
    for target in SHIFT_TARGETS:
        rows = [
            matched_shift_row(model["report"], site_a, site_b, target)
            for model in models
        ]
        rows = [r for r in rows if r is not None]
        summary[str(target)] = {
            "n_models": len(rows),
            "auroc_a_mean": mean_or_none([r["auroc_a"] for r in rows]),
            "auroc_b_mean": mean_or_none([r["auroc_b"] for r in rows]),
            "delta_b_minus_a_mean": mean_or_none([r["delta_b_minus_a"] for r in rows]),
        }
    return summary


def group_summary(
    rows: list[dict], key_fn, order: tuple[str, ...] | None = None
) -> dict:
    """Mean AUROC of each site, the union and the correlations, grouped by key_fn(row)."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = key_fn(row)
        if key is None:
            continue
        groups.setdefault(key, []).append(row)

    names = order if order is not None else sorted(groups)
    summary = {}
    for name in names:
        subset = groups.get(name, [])
        summary[name] = {
            "n_models": len(subset),
            "auroc_a_mean": mean_or_none([r["auroc_a"] for r in subset]),
            "auroc_b_mean": mean_or_none([r["auroc_b"] for r in subset]),
            "auroc_union_mean": mean_or_none([r["auroc_union"] for r in subset]),
            "auroc_best_of_2_mean": mean_or_none(
                [
                    r["auroc_best_of_2"]
                    for r in subset
                    if r["auroc_best_of_2"] == r["auroc_best_of_2"]
                ]
            ),
            "spearman_triggered_mean": mean_or_none(
                [r["spearman_triggered"] for r in subset]
            ),
            "spearman_clean_mean": mean_or_none([r["spearman_clean"] for r in subset]),
            "exclusive_either_share_mean": mean_or_none(
                [r["exclusive_either_share"] for r in subset]
            ),
        }
    return summary


def paired_gain_ci(
    rows: list[dict], key_a: str, key_b: str, resamples: int, seed: int
) -> dict:
    """Bootstrap CI on the mean of rows[key_b] - rows[key_a], paired within model."""
    deltas = [row[key_b] - row[key_a] for row in rows]
    low, high = bootstrap_ci(deltas, resamples, seed)
    return {
        "mean": mean_or_none(deltas),
        "ci_low": low,
        "ci_high": high,
        "n": len(deltas),
    }


def compare_sites(
    results_dir: str, site_a: str, site_b: str, resamples: int, seed: int
) -> dict | None:
    """The full part-1/part-2 comparison of 2 placement ids: any pair, not just the headline 2.

    Returns None when fewer than MIN_MODELS_PER_PAIR clearing cells carry an adaptive rate
    at both sites, so a thin pair is skipped rather than reported on noise.
    """
    models = select_models(results_dir, site_a, site_b)
    if len(models) < MIN_MODELS_PER_PAIR:
        return None

    rows = [per_model_row(results_dir, model, site_a, site_b) for model in models]

    result = {
        "site_a": site_a,
        "site_b": site_b,
        "n_models": len(rows),
        "overall": group_summary(rows, lambda r: "all")["all"],
        "auroc_b_minus_a": paired_gain_ci(rows, "auroc_a", "auroc_b", resamples, seed),
        "by_attack": group_summary(rows, lambda r: r["attack"]),
        "by_locality": group_summary(
            rows, lambda r: r["locality"], ("local", "global")
        ),
        "matched_shift": matched_shift_summary(models, site_a, site_b),
        "rows": rows,
    }
    return result


def load_mechanism_images(
    folder: str, checkpoints_dir: str, raw_data_dir: str, seed: int
):
    """Up to MECHANISM_PAIRS clean/triggered paired images for 1 checkpoint, and its metadata."""
    checkpoint = os.path.join(checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint)
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint,
        seed=seed,
        raw_data_dir=raw_data_dir,
        batch_size=MECHANISM_BATCH,
        num_workers=4,
        max_samples=MECHANISM_PAIRS,
    )
    pairs = paired_rows(loaders, manifest)
    return metadata, pairs


def block_output_change(
    model,
    architecture: str,
    images: torch.Tensor,
    device: torch.device,
    position: str,
    rate: float,
    block_index: int,
    seed: int,
) -> torch.Tensor:
    """The block's own residual-stream output with 1 token-mask probe plugged at 1 block.

    Restricted to a single block with block_range so the perturbation matches exactly what
    the real detection sweep injects at that site, and no operator logic is reimplemented
    here: models.positions.plug_dropout and defences.operators.build_operator do the
    masking, this only captures the result. The global RNG is seeded before the call so
    the mask realisation is reproducible across a rerun of this function.
    """
    torch.manual_seed(seed)
    factory = {position: build_operator("token_mask")}
    handles = plug_dropout(
        model,
        architecture,
        (position,),
        factory,
        rate,
        block_range=(block_index, block_index),
    )
    try:
        stream = residual_stream(
            model, images, device, MECHANISM_BATCH
        )  # (n, 13, tok, dim)
    finally:
        unplug_dropout(handles)
    return stream[:, block_index, :, :]  # (n, tokens, dim), layer index = block index


def relative_change(
    baseline: torch.Tensor, perturbed: torch.Tensor
) -> tuple[float, float]:
    """Mean relative norm of the change, over the whole block output and over the class token.

    baseline and perturbed are both (n, tokens, dim).
    """
    delta = perturbed - baseline  # (n, tokens, dim)
    full_relative = delta.flatten(1).norm(dim=1) / baseline.flatten(1).norm(
        dim=1
    ).clamp_min(1e-8)  # (n,)
    cls_relative = (delta[:, 0, :].norm(dim=1)) / baseline[:, 0, :].norm(
        dim=1
    ).clamp_min(1e-8)  # (n,)
    return float(full_relative.mean()), float(cls_relative.mean())


def direction_share(
    baseline: torch.Tensor, perturbed: torch.Tensor, direction: torch.Tensor
) -> tuple[float, float]:
    """Mean |cosine| of the change against the backdoor direction, full block and class token.

    direction is (tokens, dim) for the full share and direction[0] (dim,) for the class
    token share, both from experiments.whole_network_erasure.measure.directions'
    "all_tokens" and "cls" fields at this block's layer.
    """
    delta = perturbed - baseline  # (n, tokens, dim)
    unit_full = direction.flatten() / direction.flatten().norm().clamp_min(
        1e-8
    )  # (tok*dim,)
    proj_full = delta.flatten(1) @ unit_full  # (n,)
    share_full = proj_full.abs() / delta.flatten(1).norm(dim=1).clamp_min(1e-8)  # (n,)

    unit_cls = direction[0] / direction[0].norm().clamp_min(1e-8)  # (dim,)
    proj_cls = delta[:, 0, :] @ unit_cls  # (n,)
    share_cls = proj_cls.abs() / delta[:, 0, :].norm(dim=1).clamp_min(1e-8)  # (n,)
    return float(share_full.mean()), float(share_cls.mean())


def measure_mechanism_model(
    folder: str,
    results_dir: str,
    checkpoints_dir: str,
    raw_data_dir: str,
    device: torch.device,
) -> dict:
    """Block-output and class-token relative change under site A and site B, per block.

    200 paired clean/triggered images, split as experiments.whole_network_erasure.measure
    splits its own estimation pairs: the first 100 estimate the backdoor direction, the
    second 100 are where the change is measured, so no number is read on the images the
    direction came from.
    """
    report = load_psbd_metrics(results_dir, folder)
    rate_a = report["placements"][SITE_A]["adaptive_rate"]
    rate_b = report["placements"][SITE_B]["adaptive_rate"]

    metadata, pairs = load_mechanism_images(
        folder, checkpoints_dir, raw_data_dir, MECHANISM_SEED
    )
    model = load_checkpoint(
        metadata["architecture"],
        os.path.join(checkpoints_dir, folder, "attack_result.pt"),
        device,
    )

    half = len(pairs["clean"]) // 2
    estimation_pairs = {
        "clean": pairs["clean"][:half],
        "backdoor": pairs["backdoor"][:half],
    }
    found = directions(
        model, estimation_pairs, device, MECHANISM_BATCH
    )  # cls (13,dim), all_tokens (13,tok,dim)

    clean_images = pairs["clean"][half:]
    triggered_images = pairs["backdoor"][half:]

    # 1 unperturbed pass per image group, read at every block instead of 1 pass per block,
    # since no probe is plugged for the baseline and every layer comes free from the same
    # forward pass.
    baseline_clean_stream = residual_stream(
        model, clean_images, device, MECHANISM_BATCH
    )  # (n, 13, tok, dim)
    baseline_triggered_stream = residual_stream(
        model, triggered_images, device, MECHANISM_BATCH
    )

    blocks_out = {}
    for block_index in MECHANISM_BLOCKS:
        block_row = {}
        baseline_clean = baseline_clean_stream[:, block_index, :, :]  # (n, tok, dim)
        baseline_triggered = baseline_triggered_stream[:, block_index, :, :]
        direction_here = found["all_tokens"][block_index]  # (tok, dim)

        for site_name, position, rate in (
            ("a", SITE_A_POSITION, rate_a),
            ("b", SITE_B_POSITION, rate_b),
        ):
            for group_name, images, baseline in (
                ("clean", clean_images, baseline_clean),
                ("triggered", triggered_images, baseline_triggered),
            ):
                perturbed = block_output_change(
                    model,
                    "vit",
                    images,
                    device,
                    position,
                    rate,
                    block_index,
                    MECHANISM_SEED,
                )
                full_rel, cls_rel = relative_change(baseline, perturbed)
                full_share, cls_share = direction_share(
                    baseline, perturbed, direction_here
                )
                block_row[f"{site_name}_{group_name}"] = {
                    "relative_change_full": full_rel,
                    "relative_change_cls": cls_rel,
                    "direction_share_full": full_share,
                    "direction_share_cls": cls_share,
                }
        blocks_out[str(block_index)] = block_row

    result = {
        "folder": folder,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "rate_a": rate_a,
        "rate_b": rate_b,
        "n_direction_pairs": half,
        "n_measured_pairs": len(clean_images),
        "blocks": blocks_out,
    }
    return result


def read_activation_patching(results_dir: str, folder: str) -> dict | None:
    """Recovery per layer at site A (resid) and site B (attn), group='trigger', for 1 model."""
    record = load_json(os.path.join(results_dir, folder, "activation_patching.json"))
    if record is None:
        return None
    recovery = {"resid": {}, "attn": {}}
    for row in record["rows"]:
        if row["group"] != PATCHING_GROUP:
            continue
        if row["site"] == PATCHING_SITE_A:
            recovery["resid"][row["layer"]] = row["recovery"]
        elif row["site"] == PATCHING_SITE_B:
            recovery["attn"][row["layer"]] = row["recovery"]
    return recovery


def build_report(args: argparse.Namespace) -> dict:
    pairs_out = {}
    for operator, (site_a, site_b) in OPERATOR_PAIRS.items():
        compared = compare_sites(
            args.results_dir, site_a, site_b, args.bootstrap, args.seed
        )
        if compared is not None:
            pairs_out[operator] = compared
            print(
                f"{operator}: {compared['n_models']} models, "
                f"A={compared['overall']['auroc_a_mean']:.3f} "
                f"B={compared['overall']['auroc_b_mean']:.3f} "
                f"union={compared['overall']['auroc_union_mean']:.3f} "
                f"delta(B-A)={compared['auroc_b_minus_a']['mean']:+.3f} "
                f"CI [{compared['auroc_b_minus_a']['ci_low']:+.3f}, "
                f"{compared['auroc_b_minus_a']['ci_high']:+.3f}]"
            )
        else:
            print(
                f"{operator}: fewer than {MIN_MODELS_PER_PAIR} models with both sites, skipped"
            )

    mechanism_out, patching_out = {}, {}
    if not args.skip_mechanism:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for folder in MECHANISM_MODELS:
            mechanism_out[folder] = measure_mechanism_model(
                folder,
                args.results_dir,
                args.checkpoints_dir,
                args.raw_data_dir,
                device,
            )
            print(f"mechanism: {folder} done")
        for folder in MECHANISM_MODELS:
            recovery = read_activation_patching(args.results_dir, folder)
            if recovery is not None:
                patching_out[folder] = recovery

    report = {
        "question": (
            "are site A (before_attention_norm_token_mask) and site B "
            "(before_attention_residual_token_mask) the same placement seen twice, or 2 "
            "complementary probes"
        ),
        "site_a": SITE_A,
        "site_b": SITE_B,
        "min_models_per_pair": MIN_MODELS_PER_PAIR,
        "shift_targets": list(SHIFT_TARGETS),
        "mechanism_models": list(MECHANISM_MODELS),
        "mechanism_blocks": list(MECHANISM_BLOCKS),
        "mechanism_pairs": MECHANISM_PAIRS,
        "pairs": pairs_out,
        "mechanism": mechanism_out,
        "activation_patching": patching_out,
    }
    return report


def main() -> None:
    args = parse_args()
    report = build_report(args)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
