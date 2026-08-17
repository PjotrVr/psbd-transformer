"""Per-dataset detection tables, with the coverage bar enforced in code.

The reporting standard (docs/hypothesis/README.md) is that no conclusion is
reported unless it spans all three axes at once: 4 datasets, 3 poison rates, and
the full attack panel. This script refuses to emit a row that does not, and
prints what is missing instead.

That refusal is the point. Every earlier round of this project produced a
CIFAR-10 number, generalised it, and had to walk it back; enforcing the bar
mechanically is more reliable than remembering to.

A cell is REQUIRED only if its attack actually implants (ASR >= --min-asr). An
attack that failed to implant is reported as such and does not block coverage,
because there is no backdoor there to detect. wanet at 1% is the standing case:
it does not implant on any dataset (peak ASR 0.379).

One-sided throughout: low score means poisoned, and a value below 0.5 is printed
as the failure it is, never re-signed.

Example
    python defence_tables.py --operator dropout --position before_attention_norm
    python defence_tables.py --coverage-only
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from defences.psbd_cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.psbd_metrics import (
    complete_rates,
    pair_clean_to_backdoor,
    psu_from_cache,
    psu_ratio_from_cache,
    select_rate_at_matched_shift,
    shift_ratio,
)

DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
POISON_TAGS = (("0_01", "1%"), ("0_05", "5%"), ("0_1", "10%"))
PANEL = ("badnet_a2o", "blend", "wanet", "lc", "adaptive_blend")


def read_asr(checkpoints_dir: str, folder: str):
    path = os.path.join(checkpoints_dir, folder, "metrics.json")
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path)).get("asr")
    except Exception:
        return None


def required_cells(checkpoints_dir: str, architecture: str, min_asr: float):
    """Every (dataset, rate, attack) the bar demands, split into required and dead.

    Dead cells are carried rather than dropped so the tables can say "the attack
    failed to implant" instead of leaving a silent blank that reads like a
    detection failure.
    """
    required, dead = [], []
    for dataset in DATASETS:
        for tag, label in POISON_TAGS:
            for attack in PANEL:
                folder = f"{architecture}_{dataset}_{attack}_{tag}"
                asr = read_asr(checkpoints_dir, folder)
                if asr is None:
                    continue
                entry = (dataset, label, attack, folder, asr)
                (required if asr >= min_asr else dead).append(entry)
    return required, dead


def cell_score(psbd_dir, name, kind, sigma_target):
    """One-sided AUROC and TPR at the target FPR, at the sigma-matched rate."""
    manifest = read_split_manifest(psbd_dir)
    shift_by_rate = {}
    for rate in complete_rates(psbd_dir, name):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, name, rate, "validation")
        )
        shift_by_rate[rate] = shift_ratio(labels, argmax)
    chosen = select_rate_at_matched_shift(shift_by_rate, sigma_target)
    if chosen is None:
        return None

    build = psu_ratio_from_cache if kind == "fractional" else psu_from_cache
    scored = {}
    for split in ("clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, name, chosen, split)
        )
        scored[split] = build(probs, labels, per_pass)
    clean = pair_clean_to_backdoor(scored["clean"], manifest).float().numpy()
    backdoor = scored["backdoor"].float().numpy()

    y = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    raw = np.concatenate([-clean, -backdoor])
    fpr, tpr, _ = roc_curve(y, raw)
    return {
        "rate": chosen,
        "auroc": float(roc_auc_score(y, raw)),
        "tpr": float(np.interp(0.05, fpr, tpr)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--architecture", default="vit")
    parser.add_argument("--operator", default="dropout")
    parser.add_argument("--position", default="before_attention_norm")
    parser.add_argument(
        "--score", default="fractional", choices=("fractional", "absolute")
    )
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--min-asr", type=float, default=0.5)
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="print the table even when the bar is not met. Anything produced "
        "this way is PROVISIONAL and must be labelled so wherever it is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = (
        args.position
        if args.operator == "dropout"
        else f"{args.position}_{args.operator}"
    )
    required, dead = required_cells(
        args.checkpoints_dir, args.architecture, args.min_asr
    )

    have, missing = {}, []
    for dataset, label, attack, folder, asr in required:
        psbd_dir = os.path.join(args.results_dir, folder, "psbd")
        result = (
            cell_score(psbd_dir, name, args.score, args.sigma)
            if os.path.isdir(os.path.join(psbd_dir, name))
            else None
        )
        if result is None:
            missing.append((dataset, label, attack))
        else:
            have[(dataset, label, attack)] = (result, asr)

    total = len(required)
    print(
        f"configuration: {args.operator} @ {args.position} ({args.score} PSU, sigma>={args.sigma})"
    )
    print(f"coverage: {len(have)}/{total} required cells")
    print(
        f"  {len(dead)} cells excluded: attack did not implant at ASR>={args.min_asr}"
    )

    by_dataset = {d: [0, 0] for d in DATASETS}
    for dataset, label, attack, folder, asr in required:
        by_dataset[dataset][1] += 1
        if (dataset, label, attack) in have:
            by_dataset[dataset][0] += 1
    print(
        "  per dataset: "
        + "  ".join(f"{d}={a}/{b}" for d, (a, b) in by_dataset.items())
    )

    if missing and not args.coverage_only:
        shown = missing[:12]
        print(f"\n  MISSING ({len(missing)}):")
        for dataset, label, attack in shown:
            print(f"    {dataset:9} {label:>4} {attack}")
        if len(missing) > len(shown):
            print(f"    ... and {len(missing) - len(shown)} more")

    if args.coverage_only:
        return
    if missing and not args.allow_partial:
        print(
            "\nREFUSED: the coverage bar is not met, so no table is emitted.\n"
            "Rerun once the missing cells land, or pass --allow-partial and label\n"
            "every number it produces PROVISIONAL."
        )
        raise SystemExit(1)

    for dataset in DATASETS:
        rows = [
            (label, attack, have[(dataset, label, attack)])
            for d, label, attack, _, _ in required
            if d == dataset and (dataset, label, attack) in have
        ]
        if not rows:
            continue
        print(f"\n### {dataset}   (AUROC / TPR@5%FPR (oracle), one-sided)\n")
        print("| poison | attack | ASR | AUROC | TPR@5%FPR (oracle) | rate |")
        print("|---|---|---|---|---|---|")
        for label, attack, (result, asr) in rows:
            flag = " (weak)" if asr < 0.8 else ""
            print(
                f"| {label} | `{attack}`{flag} | {asr:.3f} | {result['auroc']:.3f} "
                f"| {result['tpr']:.3f} | {result['rate']:g} |"
            )
        for d, label, attack, _, asr in dead:
            if d == dataset:
                print(
                    f"| {label} | `{attack}` | {asr:.3f} | attack failed to implant | -- | -- |"
                )


if __name__ == "__main__":
    main()
