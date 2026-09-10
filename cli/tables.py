"""Per-dataset detection tables, with the coverage bar enforced in code.

The reporting standard (docs/hypothesis/README.md) is that no conclusion is
reported unless it spans all 3 axes at once: 4 datasets, 3 poison rates and the
full attack panel. This script refuses to emit a row that does not, and prints
what is missing instead. A number from 1 dataset does not generalise, and
enforcing the bar mechanically is more reliable than remembering to.

A cell is required only if its attack actually implants (ASR >= --min-asr). An
attack that failed to implant is reported as such and does not block coverage,
because there is no backdoor there to detect.

One-sided throughout: low score means poisoned, and a value below 0.5 is printed
as the failure it is, never re-signed.

Every target FPR is reported at 2 thresholds, and the gap between them is the
part that matters (the same distinction cli.operating_points draws):

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
    python -m cli.tables --operator dropout --position before_attention_norm
    python -m cli.tables --operator head_mask --position attention_heads --fpr 0.05
    python -m cli.tables --coverage-only
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import (
    ADAPTIVE_SHIFT_TARGET,
    PLACEMENT_MATCH_TARGET,
    complete_rates,
    pair_clean_to_backdoor,
    select_rate_adaptively,
    select_rate_at_matched_shift,
    threshold_at_quantile,
)
from defences.scores import psu_from_cache, psu_ratio_from_cache, shift_ratio

DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
POISON_TAGS = (("0_01", "1%"), ("0_05", "5%"), ("0_1", "10%"))
PANEL = ("badnet_a2o", "blend", "wanet", "lc", "adaptive_blend")

DEFAULT_FPRS = (0.01, 0.05, 0.10, 0.25)

# Which rate a cell is read at. These are 2 different protocols and the choice
# moves the numbers a long way, because the nearest rate to a mid-ladder sigma can
# sit well below the rate that actually separates.
#
#   deployable  PSBD's own rule: the smallest rate reaching the target. This is
#               what a defender executes and what a detection number must be read
#               at. Returns None when the grid never reaches the target, which is
#               a real answer and not a gap to be filled by the nearest rate.
#
#   matched     the nearest rate to the target, for comparing a placement
#               against another at equal disturbance. It always returns something,
#               so it can silently report a cell that was never matched, which is
#               what --sigma-tolerance marks.
#
# This script emits DETECTION tables, so it defaults to deployable. Pass
# --rate-rule matched only when the table's question is about placement.
RATE_RULES = {
    "deployable": select_rate_adaptively,
    "matched": select_rate_at_matched_shift,
}
DEFAULT_RATE_RULE = "deployable"

# Each rule has its own target, and pairing a rule with the other's target is the
# specific mistake this mapping exists to prevent.
DEFAULT_SIGMA = {
    "deployable": ADAPTIVE_SHIFT_TARGET,
    "matched": PLACEMENT_MATCH_TARGET,
}

# Appended to the achieved sigma of any cell the rate grid could not match. A
# marked cell is still printed, because dropping it would hide that the grid is
# too coarse for this operator, but it must never be read as strength-matched.
SIGMA_MISMATCH_MARKER = "*"

# Below this ASR the backdoor is present but unreliable, so the row is labelled
# rather than silently averaged in with the ones that implanted cleanly.
WEAK_ASR = 0.8

MAX_MISSING_SHOWN = 12


def is_mismatched(result: dict, args) -> bool:
    """Whether a cell's achieved sigma disqualifies it from a matched comparison.

    Only the "matched" rule can produce such a cell. The "deployable" rule selects
    on sigma >= target and returns None when nothing qualifies, so every cell it
    does return satisfies the constraint by construction and overshoot is the rule
    working as specified. Marking those would flag every correct cell.
    """
    if args.rate_rule != "matched":
        return False
    return abs(result["achieved_sigma"] - args.sigma) > args.sigma_tolerance


def read_cell_metadata(checkpoints_dir: str, folder: str) -> dict:
    """ASR and poison-rate facts for a checkpoint, preferring args.json.

    args.json is authoritative and metrics.json is not. Every metrics.json on disk
    predates the source-restricted eval set, so its TaCT and Adaptive-Blend rows
    disagree with args.json by enough to decide whether a cell clears the ASR bar.
    args.json's value is corroborated independently by asr_from_cache, read back
    from the PSBD baseline cache.

    The requested rate is also a request. A clean-label attack is eligible only on
    the target class, so it saturates and 3 folder names can name 1 run. The
    realized rate says which.
    """
    args_path = os.path.join(checkpoints_dir, folder, "args.json")
    metrics_path = os.path.join(checkpoints_dir, folder, "metrics.json")

    metadata = {}
    for path in (metrics_path, args_path):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as handle:
                metadata.update(json.load(handle))
        except Exception:
            continue

    if not metadata:
        return {}
    return {
        "asr": metadata.get("asr"),
        "realized_poison_rate": metadata.get("realized_poison_rate"),
        "poison_rate_capped": metadata.get("poison_rate_capped"),
    }


def read_asr(checkpoints_dir: str, folder: str) -> float | None:
    """The measured ASR, or None when unavailable."""
    return read_cell_metadata(checkpoints_dir, folder).get("asr")


def required_cells(
    checkpoints_dir: str, architecture: str, min_asr: float
) -> tuple[list[tuple], list[tuple]]:
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
    rate_rule: str = DEFAULT_RATE_RULE,
) -> dict | None:
    """One-sided AUROC and both TPRs per target FPR, at the sigma-matched rate.

    Returns None when the position config holds no complete rate on disk.
    Otherwise returns the chosen rate, the clean-validation shift ratio that rate
    actually achieved, the one-sided AUROC and an operating point per entry of
    target_fprs in the order given.

    achieved_sigma is part of the contract because the "matched" rule returns the
    nearest rate unconditionally: it never fails, so the caller, not this
    function, has to decide whether the match was close enough to report as matched.
    The "deployable" rule returns None instead when the grid never reaches the
    target, so a None here means 2 different things depending on the rule and the
    caller has to know which one it asked for.
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

    chosen = RATE_RULES[rate_rule](shift_by_rate, sigma_target)
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
        "pairing must leave a clean row per backdoor row"
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
    parser.add_argument(
        "--rate-rule",
        default=DEFAULT_RATE_RULE,
        choices=tuple(RATE_RULES),
        help="deployable (PSBD's own: smallest rate reaching --sigma) or matched "
        "(nearest rate to --sigma, for placement comparison only).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="target clean-validation shift ratio. Defaults to the target that "
        f"belongs to --rate-rule: {DEFAULT_SIGMA}.",
    )
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
        help="target false-positive rates, one column pair per value.",
    )
    parser.add_argument("--min-asr", type=float, default=0.5)
    parser.add_argument(
        "--include-sam",
        action="store_true",
        help="keep SAM checkpoints. Off by default, see cli/summary.py",
    )
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="print the table even when the bar is not met. Anything produced "
        "this way is PROVISIONAL and must be labelled so wherever it is used.",
    )
    args = parser.parse_args()

    # A target outside (0, 1) is not a false-positive rate. np.quantile would
    # raise deep inside the scoring loop instead of here at the boundary.
    for target in args.fpr:
        if not 0.0 < target < 1.0:
            parser.error(f"--fpr values must lie in (0, 1), got {target}")
    if args.sigma_tolerance < 0.0:
        parser.error("--sigma-tolerance must not be negative")
    if args.sigma is None:
        args.sigma = DEFAULT_SIGMA[args.rate_rule]
    return args


def placement_name(position: str, operator: str) -> str:
    """The results/ subfolder a (position, operator) pair was cached under.

    The paper's dropout keeps the bare position name, which is what cli.sweep
    writes, so it is the one case with no operator suffix.
    """
    if operator == "dropout":
        return position
    return f"{position}_{operator}"


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


def score_every_cell(
    args: argparse.Namespace, name: str, required: list[tuple]
) -> tuple[dict, list[tuple]]:
    """Score every required cell that has a cache, and list the ones that do not."""
    have, missing = {}, []
    for dataset, label, attack, folder, asr in required:
        psbd_dir = os.path.join(args.results_dir, folder, "psbd")
        result = (
            cell_score(psbd_dir, name, args.score, args.sigma, args.fpr, args.rate_rule)
            if os.path.isdir(os.path.join(psbd_dir, name))
            else None
        )
        if result is None:
            missing.append((dataset, label, attack))
        else:
            have[(dataset, label, attack)] = (result, asr)

    return have, missing


def print_coverage(
    args: argparse.Namespace,
    required: list[tuple],
    dead: list[tuple],
    have: dict,
    missing: list[tuple],
) -> None:
    """How much of the bar this configuration actually meets, and where it falls short."""
    print(
        f"configuration: {args.operator} @ {args.position} "
        f"({args.score} PSU, sigma>={args.sigma})"
    )
    print(f"coverage: {len(have)}/{len(required)} required cells")
    print(
        f"  {len(dead)} cells excluded: attack did not implant at ASR>={args.min_asr}"
    )

    by_dataset = {dataset: [0, 0] for dataset in DATASETS}
    for dataset, label, attack, _folder, _asr in required:
        by_dataset[dataset][1] += 1
        if (dataset, label, attack) in have:
            by_dataset[dataset][0] += 1
    print(
        "  per dataset: "
        + "  ".join(f"{d}={a}/{b}" for d, (a, b) in by_dataset.items())
    )

    unmatched = [
        key for key, (result, _asr) in have.items() if is_mismatched(result, args)
    ]
    if unmatched:
        print(
            f"  {len(unmatched)} cells NOT strength-matched: achieved sigma is "
            f"further than {args.sigma_tolerance} from {args.sigma}, marked "
            f"'{SIGMA_MISMATCH_MARKER}' in the tables"
        )

    if missing and not args.coverage_only:
        shown = missing[:MAX_MISSING_SHOWN]
        print(f"\n  MISSING ({len(missing)}):")
        for dataset, label, attack in shown:
            print(f"    {dataset:9} {label:>4} {attack}")
        if len(missing) > len(shown):
            print(f"    ... and {len(missing) - len(shown)} more")


def render_row(label: str, attack: str, result: dict, asr: float, marker: str) -> str:
    """A table row: the cell's provenance, its AUROC and both TPRs per target FPR."""
    weak = " (weak)" if asr < WEAK_ASR else ""
    cells = [
        label,
        f"`{attack}`{weak}",
        f"{asr:.3f}",
        f"{result['achieved_sigma']:.3f}{marker}",
        f"{result['rate']:g}",
        f"{result['auroc']:.3f}",
    ]
    for point in result["operating_points"]:
        cells.append(f"{point['tpr_deployable']:.3f} ({point['fpr_achieved']:.3f})")
        cells.append(f"{point['tpr_oracle']:.3f}")

    row = "| " + " | ".join(cells) + " |"
    return row


def print_dataset_table(
    dataset: str,
    args: argparse.Namespace,
    required: list[tuple],
    dead: list[tuple],
    have: dict,
    columns: list[str],
) -> None:
    """A dataset's table, plus the did-not-implant rows underneath it."""
    rows = [
        (label, attack, have[(dataset, label, attack)])
        for d, label, attack, _folder, _asr in required
        if d == dataset and (dataset, label, attack) in have
    ]
    if not rows:
        return

    rule_note = (
        f"smallest rate reaching sigma {args.sigma}"
        if args.rate_rule == "deployable"
        else f"nearest rate to sigma {args.sigma}"
    )
    print(f"\n### {dataset}   (one-sided, {args.score} PSU, {rule_note})\n")
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")

    marked_any = False
    for label, attack, (result, asr) in rows:
        mismatched = is_mismatched(result, args)
        marked_any = marked_any or mismatched
        marker = SIGMA_MISMATCH_MARKER if mismatched else ""
        print(render_row(label, attack, result, asr, marker))

    for d, label, attack, _folder, asr in dead:
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


def main() -> None:
    args = parse_args()
    name = placement_name(args.position, args.operator)

    required, dead = required_cells(
        args.checkpoints_dir, args.architecture, args.min_asr
    )
    have, missing = score_every_cell(args, name, required)
    print_coverage(args, required, dead, have, missing)

    if args.coverage_only:
        return
    if missing and not args.allow_partial:
        print(
            "\nREFUSED: the coverage bar is not met, so no table is emitted.\n"
            "Rerun once the missing cells land, or pass --allow-partial and label\n"
            "every number it produces PROVISIONAL."
        )
        raise SystemExit(1)

    print(
        "\ndepl = deployable: threshold at the target-FPR quantile of clean "
        "validation score, needing no poisoned data, with the FPR it actually "
        "achieved on the paired clean pool in brackets. oracle = the same target "
        "read off the labelled ROC curve, an upper bound no defender can reach. "
        "Deployable can exceed oracle only by overspending the clean budget, which "
        "the bracketed FPR makes visible."
    )

    columns = table_columns(args.fpr)
    for dataset in DATASETS:
        print_dataset_table(dataset, args, required, dead, have, columns)


if __name__ == "__main__":
    main()
