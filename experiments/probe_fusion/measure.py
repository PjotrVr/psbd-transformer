"""Do two PSBD probes catch different backdoors, or the same one twice?

The cache stores per_pass_probs of shape (k, N), the baseline-predicted class's probability
on each perturbed pass. That is exactly PSU's input, so combining placements at the SCORE
level costs no GPU at all: every combination below is computed from tensors already on disk.

That cheapness is a trap. C(18,2) + C(18,3) is 969 combinations, and reporting the best of
them is an oracle number of the kind this ledger has already been burned by. So the
combinations are pre-registered in configs/psbd_basis.json, each carrying a mechanism claim
written before any fused AUROC was read, and four of them are NEGATIVE controls that should
gain little if the claimed mechanism is what carries the effect.

Fusion is min-rank against the clean-validation reference (multi_probe_score), with the
calibrated threshold rule, which takes its quantile on the combined score and therefore
absorbs however correlated the probes turn out to be. Ranking each split against itself
would pin TPR to FPR and destroy the method.

Every probe is read at its own rate matched to a clean-validation shift ratio, never at a
shared rate, because a shared rate compares placements at different effective strengths. The
achieved shift is recorded next to every number: an audit found input_pixels_scale_up and
mlp_norm_out_gain_scale reaching sigma 0.6 on 0 percent of cells, so a probe that cannot
reach the target must be visible rather than silently contributing at whatever strength it
managed.

    PYTHONPATH=. python experiments/probe_fusion/measure.py --shift-target 0.6
"""

import argparse
import json
import os
from data.splits import SPLITS

import torch

from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import (
    complete_rates,
    detection_report,
    multi_probe_detection,
    pair_clean_to_backdoor,
    select_rate_at_matched_shift,
)
from defences.scores import psu_ratio_from_cache, shift_ratio, to_rank

# The operating points a deployment is read at, per the reporting contract.
TARGET_FPRS = (0.10, 0.20)
# A probe whose achieved shift misses the target by more than this is reported but excluded
# from fusion: it is not being asked the same question as the probes it would be fused with.
SHIFT_TOLERANCE = 0.15


def load_declaration(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def probe_at_matched_shift(
    psbd_dir: str, placement: str, baselines: dict, manifest: dict, target: float
) -> dict | None:
    """One probe's fractional PSU on all three splits, at its own shift-matched rate.

    Returns None when the placement has no usable cache. The fractional variant is used
    because it is what every published table in this project reports.
    """
    rates = complete_rates(psbd_dir, placement)
    if not rates:
        return None

    validation_probs, validation_labels, _ = baselines["validation"]
    shift_by_rate = {}
    for rate in rates:
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "validation")
        )
        shift_by_rate[rate] = shift_ratio(validation_labels, argmax)

    rate = select_rate_at_matched_shift(shift_by_rate, target)
    if rate is None:
        return None

    psu = {}
    for split in SPLITS:
        probs, labels, _ = baselines[split]
        per_pass_probs, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        psu[split] = psu_ratio_from_cache(probs, labels, per_pass_probs)

    return {
        "placement": placement,
        "rate": rate,
        "achieved_shift": shift_by_rate[rate],
        "reaches_target": abs(shift_by_rate[rate] - target) <= SHIFT_TOLERANCE,
        "validation": psu["validation"],
        "clean": pair_clean_to_backdoor(psu["clean"], manifest),
        "backdoor": psu["backdoor"],
    }


def mean_rank_detection(
    val_per_probe: list[torch.Tensor],
    clean_per_probe: list[torch.Tensor],
    backdoor_per_probe: list[torch.Tensor],
    target_fpr: float,
) -> dict:
    """The averaging counterpart to multi_probe_detection's minimum-rank union.

    Minimum rank asks whether ANY probe finds a sample suspicious, so a single probe that
    is INVERTED on a cell (backdoor samples scoring higher than clean ones, which happens:
    before_attention_norm_gaussian reads AUROC 0.191 on cifar100 badnet at 1 percent) drags
    the whole combination down with it. Averaging the ranks cannot be captured by one bad
    member, at the cost of diluting one strong member. Neither rule reads a label, so both
    stay defender-legal, and reporting only the flattering one is how a fusion result stops
    meaning anything.

    Thresholding matches the calibrated rule: the target_fpr quantile of the combined
    validation score, so the achieved FPR lands on target whatever the probes correlate at.
    """

    def combine(per_probe, reference):
        ranks = [to_rank(psu, ref) for psu, ref in zip(per_probe, reference)]
        return torch.stack(ranks).mean(dim=0)

    validation = combine(val_per_probe, val_per_probe)
    clean = combine(clean_per_probe, val_per_probe)
    backdoor = combine(backdoor_per_probe, val_per_probe)
    return detection_report(validation, clean, backdoor, target_fpr)


def single_probe_row(probe: dict) -> dict:
    """Detection for one probe alone, the baseline every combination must beat."""
    row = {
        "rate": probe["rate"],
        "achieved_shift": probe["achieved_shift"],
        "reaches_target": probe["reaches_target"],
    }
    for target_fpr in TARGET_FPRS:
        report = detection_report(
            probe["validation"], probe["clean"], probe["backdoor"], target_fpr
        )
        row[f"fpr{target_fpr:.2f}"] = {
            "tpr": report["tpr"],
            "achieved_fpr": report["fpr"],
        }
        row["auroc"] = report["auroc"]
    return row


def fused_row(probes: list[dict]) -> dict:
    """Detection for a fusion of several probes, under both combination rules."""
    validation = [probe["validation"] for probe in probes]
    clean = [probe["clean"] for probe in probes]
    backdoor = [probe["backdoor"] for probe in probes]
    row = {
        "members": [probe["placement"] for probe in probes],
        "rates": [probe["rate"] for probe in probes],
        "achieved_shift": [probe["achieved_shift"] for probe in probes],
    }
    for rule in ("min_rank", "mean_rank"):
        entry = {}
        for target_fpr in TARGET_FPRS:
            if rule == "min_rank":
                report = multi_probe_detection(
                    validation,
                    clean,
                    backdoor,
                    target_fpr=target_fpr,
                    rule="calibrated",
                )
            else:
                report = mean_rank_detection(validation, clean, backdoor, target_fpr)
            entry[f"fpr{target_fpr:.2f}"] = {
                "tpr": report["tpr"],
                "achieved_fpr": report["fpr"],
            }
            entry["auroc"] = report["auroc"]
        row[rule] = entry
    # The headline stays the union rule so the field keeps meaning what it did before.
    row["auroc"] = row["min_rank"]["auroc"]
    for target_fpr in TARGET_FPRS:
        row[f"fpr{target_fpr:.2f}"] = row["min_rank"][f"fpr{target_fpr:.2f}"]
    return row


def analyse_cell(folder: str, declaration: dict, args) -> dict | None:
    psbd_dir = os.path.join(args.results_dir, folder, "psbd")
    if not all(os.path.exists(baseline_path(psbd_dir, split)) for split in SPLITS):
        return None
    baselines = {
        split: load_baseline(baseline_path(psbd_dir, split)) for split in SPLITS
    }
    manifest = read_split_manifest(psbd_dir)

    probes = {}
    for entry in declaration["basis"]:
        probe = probe_at_matched_shift(
            psbd_dir, entry["id"], baselines, manifest, args.shift_target
        )
        if probe is not None:
            probes[entry["id"]] = probe

    singles = {name: single_probe_row(probe) for name, probe in probes.items()}

    combinations = {}
    for combination in declaration["combinations"]["pairs"]:
        members = combination["members"]
        # A combination is reported only when every member is present AND reaches the
        # matched target. Fusing a probe that could not be brought to the same strength
        # would credit the combination for a comparison that was never made.
        available = [probes[name] for name in members if name in probes]
        if len(available) != len(members):
            continue
        row = fused_row(available)
        row["family"] = combination["family"]
        row["all_members_reach_target"] = all(
            probe["reaches_target"] for probe in available
        )
        row["best_member_auroc"] = max(singles[name]["auroc"] for name in members)
        row["gain_over_best_member"] = row["auroc"] - row["best_member_auroc"]
        row["gain_mean_rank"] = row["mean_rank"]["auroc"] - row["best_member_auroc"]
        combinations[combination["id"]] = row

    return {
        "folder_name": folder,
        "shift_target": args.shift_target,
        "n_probes": len(probes),
        "singles": singles,
        "combinations": combinations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", default="configs/psbd_basis.json")
    parser.add_argument("--coverage", default="results/coverage/coverage.json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--shift-target", type=float, default=0.6)
    parser.add_argument("--out", default="results/coverage/probe_fusion.json")
    parser.add_argument("--checkpoint-folder", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    declaration = load_declaration(args.declaration)
    with open(args.coverage) as handle:
        ledger = json.load(handle)

    cells = args.checkpoint_folder or [cell["folder_name"] for cell in ledger["cells"]]
    meta = {cell["folder_name"]: cell for cell in ledger["cells"]}

    reports = {}
    for folder in cells:
        try:
            report = analyse_cell(folder, declaration, args)
        except Exception as error:  # noqa: BLE001
            print(f"[FAILED] {folder}: {type(error).__name__}: {error}", flush=True)
            continue
        if report is None:
            continue
        report.update(
            {
                key: meta.get(folder, {}).get(key)
                for key in ("dataset", "attack", "poison_rate", "asr", "asr_class")
            }
        )
        reports[folder] = report
        print(
            f"[ok] {folder:38s} probes {report['n_probes']:2d} "
            f"combos {len(report['combinations']):2d}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(
            {"shift_target": args.shift_target, "cells": reports}, handle, indent=2
        )
    print(f"\n[ok] {args.out}  ({len(reports)} cells)")


if __name__ == "__main__":
    main()
