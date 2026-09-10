"""Stage 2 of PSBD: read a checkpoint's cache and write its psbd_metrics.json.

CPU only, seconds per checkpoint, no model and no dataset loaded. Everything it
needs was written by cli.sweep into results/<folder>/psbd/.

Output goes to results/<folder>/psbd_metrics.json, deliberately not metrics.json,
which cli.evaluate owns for baseline attack-success and clean-accuracy numbers.

Example
    python -m cli.analyze --checkpoint-folder vit_cifar10_badnet_a2o_0_1
    python -m cli.analyze --all
"""

import argparse
import json
import os

import torch

from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from data.registry import DATASET_REGISTRY
from defences.decision import (
    shift_key,
    HEADLINE_QUANTILE,
    PSBD_QUANTILES,
    SHIFT_MATCH_TARGETS,
    attack_success_mask,
    complete_rates,
    detection_report,
    pair_clean_to_backdoor,
    select_rate_adaptively,
    select_rate_at_matched_shift,
    select_rate_by_oracle,
)
from defences.scores import (
    psu_from_cache,
    psu_ratio_from_cache,
    shift_ratio,
    shift_target_histogram,
)
from data.splits import read_checkpoint_metadata

SPLITS = ("validation", "clean", "backdoor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="*", default=[])
    parser.add_argument(
        "--all",
        action="store_true",
        help="analyze every folder under results/ that has a psbd/ subtree",
    )
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def discover_folders(results_dir: str) -> list[str]:
    """Every results/<folder> carrying a stage-1 cache, in sorted order."""
    if not os.path.isdir(results_dir):
        return []

    folders = sorted(
        name
        for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name, "psbd"))
    )
    return folders


# Subfolders of results/<cell>/psbd/ that hold a cache rather than a placement.
# cli.head_profile writes head_profile/ beside the placement folders, and it must
# not be read as a placement, where it would parse as an unknown operator.
NON_PLACEMENT_CACHES = ("head_profile",)


def discover_position_configs(psbd_dir: str) -> list[str]:
    """The position-config subfolders stage 1 actually wrote, in sorted order."""
    configs = sorted(
        name
        for name in os.listdir(psbd_dir)
        if os.path.isdir(os.path.join(psbd_dir, name))
        and name not in NON_PLACEMENT_CACHES
    )
    return configs


def load_baselines(
    psbd_dir: str,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The no-perturbation probs, argmax and loader labels for all 3 splits."""
    baselines = {
        split: load_baseline(baseline_path(psbd_dir, split)) for split in SPLITS
    }
    return baselines


def shift_to_target_fraction(
    histogram: list[int] | None, target_label: int | None
) -> float | None:
    """Which fraction of shifted predictions landed on the attacker's target class."""
    if histogram is None or target_label is None:
        return None

    total = sum(histogram)
    if total == 0:
        return None
    return float(histogram[target_label] / total)


def analyze_one_rate(
    psbd_dir: str,
    position_config: str,
    rate: float,
    baselines: dict,
    manifest: dict,
    num_classes: int,
    target_label: int | None,
) -> dict:
    """Every number for a (position_config, rate): sigma, shifts and detection.

    The clean split is paired down to the backdoor split's images before the
    comparison, so TPR and FPR are measured over the same population. The
    unpaired FPR is reported alongside because it is what a defender screening a
    whole unfiltered pool would actually see.
    """
    psu: dict[str, torch.Tensor] = {}
    psu_ratio: dict[str, torch.Tensor] = {}
    sigma: dict[str, float | None] = {}
    histograms: dict[str, list[int] | None] = {}

    for split in SPLITS:
        probs, labels, _ = baselines[split]
        per_pass_probs, per_pass_argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, position_config, rate, split)
        )
        psu[split] = psu_from_cache(probs, labels, per_pass_probs)
        psu_ratio[split] = psu_ratio_from_cache(probs, labels, per_pass_probs)
        sigma[split] = shift_ratio(labels, per_pass_argmax)
        histograms[split] = shift_target_histogram(labels, per_pass_argmax, num_classes)

    paired_clean = pair_clean_to_backdoor(psu["clean"], manifest)

    detection = {
        f"q{quantile:.2f}": detection_report(
            psu["validation"], paired_clean, psu["backdoor"], quantile
        )
        for quantile in PSBD_QUANTILES
    }
    unpaired = detection_report(
        psu["validation"], psu["clean"], psu["backdoor"], HEADLINE_QUANTILE
    )

    # A triggered image the model still classifies correctly was never captured
    # by the backdoor, so its PSU is a clean sample's PSU. Scoring it as a
    # detection positive charges the detector for the attack's failure. Reported
    # as a second view rather than replacing the first, because a defender
    # screening real inputs does not know which ones the backdoor captured.
    #
    # The clean side must be subset by the same mask. Captured images are
    # systematically the low-confidence ones, since a trigger flips an uncertain
    # image more easily than a confident image, so restricting only the backdoor
    # side compares hard images against easy images rather than measuring
    # detection. Left unrestricted, the benign control reads far above chance.
    # Subset both sides and it returns to chance.
    _, backdoor_labels, backdoor_targets = baselines["backdoor"]
    captured = attack_success_mask(backdoor_labels, backdoor_targets)
    captured_detection = None
    if captured is not None and bool(captured.any()):
        captured_detection = {
            f"q{quantile:.2f}": detection_report(
                psu["validation"],
                paired_clean[captured],
                psu["backdoor"][captured],
                quantile,
            )
            for quantile in PSBD_QUANTILES
        }

    # The same detection, scored on the confidence-normalised PSU. Reported beside
    # the paper's absolute form rather than replacing it, so the headline numbers
    # stay comparable to the published method while the improvement is visible.
    ratio_detection = {
        f"q{quantile:.2f}": detection_report(
            psu_ratio["validation"],
            pair_clean_to_backdoor(psu_ratio["clean"], manifest),
            psu_ratio["backdoor"],
            quantile,
        )
        for quantile in PSBD_QUANTILES
    }

    row = {
        "rate": rate,
        "shift_ratio": sigma,
        "detection_psu_ratio": ratio_detection,
        "shift_target_histogram": histograms,
        # The fraction of all shifted clean predictions that landed on the
        # attacker's target class: PSBD's mechanism claim as a single number.
        "clean_shift_to_target_fraction": shift_to_target_fraction(
            histograms["clean"], target_label
        ),
        "psu_mean": {split: float(values.mean()) for split, values in psu.items()},
        "psu_std": {split: float(values.std()) for split, values in psu.items()},
        "n_samples": {split: int(values.numel()) for split, values in psu.items()},
        "n_clean_paired": int(paired_clean.numel()),
        "n_backdoor_captured": int(captured.sum()) if captured is not None else None,
        "detection": detection,
        "detection_captured_only": captured_detection,
        "detection_unpaired_clean": unpaired,
    }
    return row


def analyze_position_config(
    psbd_dir: str,
    position_config: str,
    baselines: dict,
    manifest: dict,
    num_classes: int,
    target_label: int | None,
) -> dict:
    """All rates for a placement, plus the rate-selection verdicts.

    Rates come from complete_rates, which only returns a rate whose 3 splits are
    all on disk, so this stays safe to run against a results tree a sweep is
    still writing into.
    """
    rates = complete_rates(psbd_dir, position_config)
    by_rate = {
        rate: analyze_one_rate(
            psbd_dir,
            position_config,
            rate,
            baselines,
            manifest,
            num_classes,
            target_label,
        )
        for rate in rates
    }

    validation_sigma = {
        rate: row["shift_ratio"]["validation"] for rate, row in by_rate.items()
    }
    headline = f"q{HEADLINE_QUANTILE:.2f}"
    auroc_by_rate = {
        rate: row["detection"][headline]["auroc"] for rate, row in by_rate.items()
    }

    adaptive_rate = select_rate_adaptively(validation_sigma)
    oracle_rate = select_rate_by_oracle(auroc_by_rate)

    # The comparison that answers "which placement is better". Reading 2
    # placements off the same p compares a branch perturbation against a
    # perturbation that masks the whole residual stream 12 times over, so the
    # winner is decided by strength rather than position. Matching on
    # clean-validation sigma puts both at the same measured disturbance first.
    matched = {}
    for target in SHIFT_MATCH_TARGETS:
        rate = select_rate_at_matched_shift(validation_sigma, target)
        if rate is None:
            matched[shift_key(target)] = None
            continue
        matched[shift_key(target)] = {
            "rate": rate,
            "achieved_shift_ratio": validation_sigma[rate],
            **by_rate[rate]["detection"][headline],
        }

    block = {
        "rates": [by_rate[rate] for rate in rates],
        "adaptive_rate": adaptive_rate,
        "adaptive": by_rate[adaptive_rate]["detection"][headline]
        if adaptive_rate is not None
        else None,
        "oracle_rate": oracle_rate,
        "oracle": by_rate[oracle_rate]["detection"][headline]
        if oracle_rate is not None
        else None,
        "matched_shift": matched,
    }
    return block


def analyze_checkpoint(folder: str, checkpoints_dir: str, results_dir: str) -> dict:
    """The full stage-2 record for a checkpoint, every placement on disk."""
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    manifest = read_split_manifest(psbd_dir)
    metadata = read_checkpoint_metadata(
        os.path.join(checkpoints_dir, folder, "attack_result.pt")
    )
    num_classes = DATASET_REGISTRY[metadata["dataset"]].num_classes
    # A benign model has no target class, so the shift-to-target question does
    # not apply to it and is recorded as null rather than as class 0.
    target_label = None if metadata["attack"] == "benign" else metadata["target_label"]

    baselines = load_baselines(psbd_dir)
    report = {
        "folder_name": folder,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "label_mode": metadata.get("label_mode"),
        "poison_rate": metadata.get("poison_rate"),
        "optimizer": metadata.get("optimizer"),
        "rho": metadata.get("rho"),
        "target_label": target_label,
        "headline_quantile": HEADLINE_QUANTILE,
        "split_sizes": {
            "validation": manifest["n_heldout"],
            "clean": len(manifest["analysis_clean_indices"]),
            "backdoor": len(manifest["analysis_backdoor_indices"]),
        },
        "placements": {
            config: analyze_position_config(
                psbd_dir, config, baselines, manifest, num_classes, target_label
            )
            for config in discover_position_configs(psbd_dir)
        },
    }
    return report


def save_report(results_dir: str, folder: str, report: dict) -> None:
    """Write a checkpoint's stage-2 record next to its cache."""
    path = os.path.join(results_dir, folder, "psbd_metrics.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)


def summarize(report: dict) -> str:
    """A line per checkpoint, with both rate-selection verdicts per placement.

    Both numbers, always. Printing the oracle alone reads as the result and it is
    an upper bound, since it picks the rate by reading the labels. The adaptive
    value is what a defender gets, and on some placements the 2 differ widely.
    """
    cells = []
    for config, block in sorted(report["placements"].items()):
        adaptive, oracle = block["adaptive"], block["oracle"]
        if not (adaptive and oracle):
            cells.append(f"{config} auroc=n/a")
            continue
        cells.append(
            f"{config} adaptive={adaptive['auroc']:.3f}@p={block['adaptive_rate']}"
            f" oracle={oracle['auroc']:.3f}@p={block['oracle_rate']}"
        )

    line = "  ".join(cells)
    return line


def main() -> None:
    args = parse_args()
    folders = discover_folders(args.results_dir) if args.all else args.checkpoint_folder
    if not folders:
        raise SystemExit("nothing to analyze: pass --checkpoint-folder or --all")

    headline = f"q{HEADLINE_QUANTILE:.2f}"
    for folder in folders:
        try:
            report = analyze_checkpoint(folder, args.checkpoints_dir, args.results_dir)
        except Exception as error:
            print(f"FAILED {folder}: {type(error).__name__}: {error}")
            continue

        save_report(args.results_dir, folder, report)
        print(f"{folder} [{headline}] {summarize(report)}")


if __name__ == "__main__":
    main()
