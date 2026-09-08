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

Every target FPR is reported at two thresholds, and the gap between them is the
part that matters (the same distinction psbd_operating_points.py draws):

  deployable  the threshold is the q-quantile of CLEAN VALIDATION score, with q
              set to the target FPR, exactly the rule the PSBD paper prescribes.
              This is what a defender can actually build, since it needs no
              poisoned data. The FPR it achieves on the paired clean analysis
              pool is printed beside it, because nothing guarantees the
              validation quantile transfers to the analysis pool.

  oracle      the target FPR read off the labelled ROC curve of the analysis
              pool. Not deployable: it places the threshold using the very
              clean/poison split the detector is supposed to find. Reported as an
              upper bound only.

The dropout rate is chosen by matching the clean-validation shift ratio to
--sigma, but the rate grid is not guaranteed to contain a rate that reaches the
target. The achieved sigma is therefore printed per cell and any cell further
than --sigma-tolerance from the target is marked, so a strength-matched claim can
never be made from a cell that was never matched.

Example
    python defence_tables.py --operator dropout --position before_attention_norm
    python defence_tables.py --operator head_mask --position attention_heads --fpr 0.05
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
    threshold_at_quantile,
)

DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
POISON_TAGS = (("0_01", "1%"), ("0_05", "5%"), ("0_1", "10%"))
PANEL = ("badnet_a2o", "blend", "wanet", "lc", "adaptive_blend")

DEFAULT_FPRS = (0.01, 0.05, 0.10, 0.25)

# Appended to the achieved sigma of any cell the rate grid could not match. A
# marked cell is still printed, because dropping it would hide that the grid is
# too coarse for this operator, but it must never be read as strength-matched.
SIGMA_MISMATCH_MARKER = "*"


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


def cell_score(
    psbd_dir: str,
    name: str,
    kind: str,
    sigma_target: float,
    target_fprs: list[float],
) -> dict | None:
    """One-sided AUROC and both TPRs per target FPR, at the sigma-matched rate.

    Returns None when the position config holds no complete rate on disk.
    Otherwise returns the chosen rate, the clean-validation shift ratio that rate
    actually achieved, the one-sided AUROC, and one operating point per entry of
    target_fprs in the order given.

    achieved_sigma is part of the contract because the rate selector returns the
    nearest rate unconditionally: it never fails, so the caller, not this
    function, has to decide whether the match was close enough to report as one.
    """
    manifest = read_split_manifest(psbd_dir)

    # The validation baseline depends on the checkpoint alone, never on the rate,
    # so it is loaded once for the whole rate scan rather than inside it.
    validation_probs, validation_labels, _ = load_baseline(
        baseline_path(psbd_dir, "validation")
    )

    shift_by_rate = {}
    for rate in complete_rates(psbd_dir, name):
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, name, rate, "validation")
        )
        shift_by_rate[rate] = shift_ratio(validation_labels, argmax)

    chosen = select_rate_at_matched_shift(shift_by_rate, sigma_target)
    if chosen is None:
        return None
    achieved_sigma = shift_by_rate[chosen]

    build = psu_ratio_from_cache if kind == "fractional" else psu_from_cache

    # Validation has to be scored with the same rule as clean and backdoor, or the
    # deployable threshold is read off a distribution the detector never produces.
    validation_pass_probs, _ = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, name, chosen, "validation")
    )
    validation_score = build(
        validation_probs, validation_labels, validation_pass_probs
    )  # (n_validation,)

    scored = {}
    for split in ("clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, name, chosen, split)
        )
        scored[split] = build(probs, labels, per_pass)
    clean = pair_clean_to_backdoor(scored["clean"], manifest).float().numpy()
    backdoor = scored["backdoor"].float().numpy()
    assert len(clean) == len(backdoor), (
        "pairing must leave one clean row per backdoor row"
    )

    truth = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])

    # Negated once, because low score is the poisoned evidence and roc_auc_score
    # wants higher to mean more positive. Never negated a second time: an AUROC
    # below 0.5 is the detector failing, and re-signing it would hide that.
    ranking = np.concatenate([-clean, -backdoor])
    auroc = float(roc_auc_score(truth, ranking))
    curve_fpr, curve_tpr, _ = roc_curve(truth, ranking)

    operating_points = []
    for target in target_fprs:
        deployable_threshold = threshold_at_quantile(validation_score, target)
        operating_points.append(
            {
                "target_fpr": float(target),
                "threshold": deployable_threshold,
                "tpr_deployable": float((backdoor < deployable_threshold).mean()),
                "fpr_achieved": float((clean < deployable_threshold).mean()),
                "tpr_oracle": float(np.interp(target, curve_fpr, curve_tpr)),
            }
        )

    result = {
        "rate": chosen,
        "achieved_sigma": float(achieved_sigma),
        "auroc": auroc,
        "operating_points": operating_points,
    }
    return result


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
    parser.add_argument(
        "--sigma-tolerance",
        type=float,
        default=0.1,
        help="how far the achieved clean-validation shift ratio may sit from "
        "--sigma before the cell is marked as not strength-matched.",
    )
    parser.add_argument(
        "--fpr",
        nargs="+",
        type=float,
        default=list(DEFAULT_FPRS),
        help="target false-positive rates; one column pair per value.",
    )
    parser.add_argument("--min-asr", type=float, default=0.5)
    parser.add_argument(
        "--include-sam",
        action="store_true",
        help="keep SAM checkpoints. Off by default, see scripts/detection_summary.py",
    )
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="print the table even when the bar is not met. Anything produced "
        "this way is PROVISIONAL and must be labelled so wherever it is used.",
    )
    args = parser.parse_args()

    # A target outside (0, 1) is not a false-positive rate, and np.quantile would
    # raise deep inside the scoring loop instead of here at the boundary.
    for target in args.fpr:
        if not 0.0 < target < 1.0:
            parser.error(f"--fpr values must lie in (0, 1), got {target}")
    if args.sigma_tolerance < 0.0:
        parser.error("--sigma-tolerance must not be negative")
    return args


def table_columns(target_fprs: list[float]) -> list[str]:
    """Column headings, in the order every row must fill them.

    %g rather than a percent format so a sub-1% target keeps its digits instead
    of collapsing to "0%".
    """
    columns = ["poison", "attack", "ASR", "sigma", "rate", "AUROC"]
    for target in target_fprs:
        columns.append(f"TPR@{target * 100:g}% depl (FPR)")
        columns.append(f"TPR@{target * 100:g}% oracle")
    return columns


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
            cell_score(psbd_dir, name, args.score, args.sigma, args.fpr)
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

    unmatched = [
        key
        for key, (result, _) in have.items()
        if abs(result["achieved_sigma"] - args.sigma) > args.sigma_tolerance
    ]
    if unmatched:
        print(
            f"  {len(unmatched)} cells NOT strength-matched: achieved sigma is "
            f"further than {args.sigma_tolerance} from {args.sigma}, marked "
            f"'{SIGMA_MISMATCH_MARKER}' in the tables"
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

    columns = table_columns(args.fpr)
    print(
        "\ndepl = deployable: threshold at the target-FPR quantile of clean "
        "validation score, needing no poisoned data, with the FPR it actually "
        "achieved on the paired clean pool in brackets. oracle = the same target "
        "read off the labelled ROC curve, an upper bound no defender can reach. "
        "Deployable can exceed oracle only by overspending the clean budget, which "
        "the bracketed FPR makes visible."
    )

    for dataset in DATASETS:
        rows = [
            (label, attack, have[(dataset, label, attack)])
            for d, label, attack, _, _ in required
            if d == dataset and (dataset, label, attack) in have
        ]
        if not rows:
            continue
        print(f"\n### {dataset}   (one-sided, {args.score} PSU, sigma~{args.sigma})\n")
        print("| " + " | ".join(columns) + " |")
        print("|" + "|".join("---" for _ in columns) + "|")

        marked_any = False
        for label, attack, (result, asr) in rows:
            weak = " (weak)" if asr < 0.8 else ""
            mismatched = (
                abs(result["achieved_sigma"] - args.sigma) > args.sigma_tolerance
            )
            marked_any = marked_any or mismatched
            marker = SIGMA_MISMATCH_MARKER if mismatched else ""
            cells = [
                label,
                f"`{attack}`{weak}",
                f"{asr:.3f}",
                f"{result['achieved_sigma']:.3f}{marker}",
                f"{result['rate']:g}",
                f"{result['auroc']:.3f}",
            ]
            for point in result["operating_points"]:
                cells.append(
                    f"{point['tpr_deployable']:.3f} ({point['fpr_achieved']:.3f})"
                )
                cells.append(f"{point['tpr_oracle']:.3f}")
            print("| " + " | ".join(cells) + " |")

        for d, label, attack, _, asr in dead:
            if d != dataset:
                continue
            cells = [label, f"`{attack}`", f"{asr:.3f}", "--", "--", "did not implant"]
            cells += ["--"] * (2 * len(args.fpr))
            print("| " + " | ".join(cells) + " |")

        if marked_any:
            print(
                f"\n{SIGMA_MISMATCH_MARKER} the rate grid holds no rate landing "
                f"within {args.sigma_tolerance} of sigma {args.sigma} for this "
                "operator, so the nearest achieved sigma is shown instead (it may "
                "under or overshoot). These cells are NOT strength-matched and "
                "must not be compared against matched cells."
            )


if __name__ == "__main__":
    main()
