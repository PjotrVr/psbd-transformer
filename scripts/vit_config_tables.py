"""One markdown per deployed configuration: every dataset, poison rate and attack.

A configuration is one or more placements. Given several, they are fused at the score level
with the min-rank rule against the clean-validation reference, which is the same combination
a defender would run at inference: no extra training, no extra checkpoint, one extra
perturbation sweep per member.

Ordering is fixed and deliberate. Poison rate descends 10, 5, 1 percent because the easiest
regime comes first and the hardest last. Datasets run cifar10, cifar100, gtsrb, tiny. Attacks
are grouped EASY then HARD, because a number on BadNet says nothing about whether a detector
works and grouping keeps it from being read as if it did.

Every placement is read at ITS OWN perturbation rate, chosen so the clean-validation shift
ratio lands nearest the target, so members of a fused configuration are combined at matched
disturbance rather than at a shared nominal rate.

    PYTHONPATH=. python scripts/vit_config_tables.py \
        --placement pre_residual_blocks_9_12 --out docs/psbd-vit-config-pre-residual-9-12.md
"""

import argparse
import json
import os
import statistics

from defences.psbd_cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.psbd_metrics import (
    complete_rates,
    detection_report,
    multi_probe_detection,
    pair_clean_to_backdoor,
    psu_ratio_from_cache,
    select_rate_at_matched_shift,
    shift_ratio,
)

SPLITS = ("validation", "clean", "backdoor")
RATE_ORDER = (0.1, 0.05, 0.01)
DATASET_ORDER = ("cifar10", "cifar100", "gtsrb", "tiny")
# Grouped easy first, then hard, each ordered by how hard this panel found them to detect.
EASY_ORDER = ("badnet_a2o", "blend", "lf")
HARD_ORDER = ("bpp", "wanet", "tact", "sig", "lc", "adaptive_blend")
ATTACK_ORDER = EASY_ORDER + HARD_ORDER
SHIFT_TARGET = 0.6
TARGET_FPRS = (0.10, 0.20)


def probe_scores(psbd_dir, placement, baselines, manifest, target):
    """One placement's fractional PSU on all three splits, at its shift-matched rate."""
    rates = complete_rates(psbd_dir, placement)
    if not rates:
        return None
    _, validation_labels, _ = baselines["validation"]
    shift_by_rate = {}
    for rate in rates:
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "validation")
        )
        shift_by_rate[rate] = shift_ratio(validation_labels, argmax)
    rate = select_rate_at_matched_shift(shift_by_rate, target)
    if rate is None:
        return None
    scores = {}
    for split in SPLITS:
        probs, labels, _ = baselines[split]
        per_pass, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        scores[split] = psu_ratio_from_cache(probs, labels, per_pass)
    scores["clean"] = pair_clean_to_backdoor(scores["clean"], manifest)
    scores["rate"] = rate
    scores["achieved_shift"] = shift_by_rate[rate]
    return scores


def detect(members, target_fpr):
    """Detection for one configuration, single placement or fused."""
    if len(members) == 1:
        only = members[0]
        return detection_report(
            only["validation"], only["clean"], only["backdoor"], target_fpr
        )
    return multi_probe_detection(
        [m["validation"] for m in members],
        [m["clean"] for m in members],
        [m["backdoor"] for m in members],
        target_fpr=target_fpr,
        rule="calibrated",
    )


def measure_cell(folder, placements, results_dir, target):
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    if not all(os.path.exists(baseline_path(psbd_dir, s)) for s in SPLITS):
        return None
    baselines = {s: load_baseline(baseline_path(psbd_dir, s)) for s in SPLITS}
    manifest = read_split_manifest(psbd_dir)
    members = [
        probe_scores(psbd_dir, p, baselines, manifest, target) for p in placements
    ]
    if any(m is None for m in members):
        return None
    row = {
        "rates": [m["rate"] for m in members],
        "achieved_shift": [m["achieved_shift"] for m in members],
    }
    for target_fpr in TARGET_FPRS:
        report = detect(members, target_fpr)
        row[f"tpr{int(target_fpr * 100)}"] = report["tpr"]
        row[f"fpr{int(target_fpr * 100)}"] = report["fpr"]
        row["auroc"] = report["auroc"]
    return row


def fmt(value, places=3):
    if value is None or value != value:
        return "--"
    return f"{value:.{places}f}"


def build(args):
    with open(args.coverage) as handle:
        ledger = json.load(handle)
    cells = {c["folder_name"]: c for c in ledger["cells"]}
    benign = ledger["benign_reference_accuracy"]

    measured = {}
    for folder, cell in cells.items():
        if cell["asr_class"] != "clears":
            continue
        row = measure_cell(folder, args.placement, args.results_dir, args.shift_target)
        if row:
            measured[folder] = row

    label = args.title or " + ".join(f"`{p}`" for p in args.placement)
    out = []
    w = out.append
    w(f"# PSBD-ViT: {label}")
    w("")
    if len(args.placement) > 1:
        w(
            "A **fused** configuration. Its members are swept separately and combined at the score"
        )
        w(
            "level with the min-rank rule against the clean-validation reference, so a sample is"
        )
        w(
            "flagged when ANY member finds it suspicious. This needs no extra training and no extra"
        )
        w("checkpoint, only one more perturbation sweep per member at inference time.")
        w("")
        w("Members:")
        w("")
        for p in args.placement:
            w(f"- `{p}`")
    else:
        w(f"The single placement `{args.placement[0]}`.")
    w("")
    w(
        "Each member is read at **its own** perturbation rate, chosen so the clean-validation"
    )
    w(
        f"shift ratio lands nearest **{args.shift_target}**, so everything below is measured at"
    )
    w("matched disturbance rather than at a shared nominal rate.")
    w("")
    w(
        "Columns: **ASR** attack success rate and **CA** clean accuracy of the poisoned model,"
    )
    w(
        "both from the checkpoint's own provenance; **CA benign** the same-dataset benign ViT and"
    )
    w(
        "**dCA** the difference, so an attack that buys success by wrecking the model is visible;"
    )
    w("**AUROC**; and **TPR** at the 10% and 20% false-positive operating points.")
    w("")
    w(
        "Only cells whose attack actually implanted (ASR >= 0.85) appear. A `--` is an unmeasured"
    )
    w("cell, never a failure.")
    w("")

    for rate in RATE_ORDER:
        w(f"# poison rate {rate:.0%}")
        w("")
        for dataset in DATASET_ORDER:
            group = [
                (folder, cell)
                for folder, cell in cells.items()
                if cell["dataset"] == dataset
                and cell["poison_rate"] == rate
                and cell["asr_class"] == "clears"
            ]
            w(f"## {dataset}, poison rate {rate:.0%}")
            w("")
            reference = benign.get(dataset)
            if not group:
                w(f"No attack implanted at ASR >= 0.85 on {dataset} at {rate:.0%}.")
                w("")
                continue
            w(
                "| attack | class | ASR | CA | CA benign | dCA | AUROC | TPR@10%FPR | TPR@20%FPR |"
            )
            w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
            order = {name: index for index, name in enumerate(ATTACK_ORDER)}
            for folder, cell in sorted(
                group, key=lambda item: order.get(item[1]["attack"], 99)
            ):
                row = measured.get(folder)
                kind = "easy" if cell["attack"] in EASY_ORDER else "**hard**"
                drop = cell.get("clean_accuracy_drop")
                w(
                    f"| {cell['attack']} | {kind} | {fmt(cell.get('asr'))} | "
                    f"{fmt(cell.get('clean_accuracy'))} | {fmt(reference)} | "
                    f"{'--' if drop is None else f'{drop:+.3f}'} | "
                    f"{fmt(row['auroc']) if row else '--'} | "
                    f"{fmt(row['tpr10']) if row else '--'} | "
                    f"{fmt(row['tpr20']) if row else '--'} |"
                )
            w("")

    w("# Summary")
    w("")
    rows = [(cells[f], measured[f]) for f in measured]
    hard = [(c, r) for c, r in rows if c["attack"] in HARD_ORDER]
    easy = [(c, r) for c, r in rows if c["attack"] in EASY_ORDER]
    primary = [(c, r) for c, r in rows if c["dataset"] in ("cifar100", "tiny")]
    w("| subset | n | AUROC | TPR@10%FPR | TPR@20%FPR | achieved FPR at 10% |")
    w("|---|---:|---:|---:|---:|---:|")
    for name, subset in (
        ("all cells", rows),
        ("**hard** attacks", hard),
        ("easy attacks", easy),
        ("primary datasets (cifar100 + tiny)", primary),
    ):
        if not subset:
            continue
        w(
            f"| {name} | {len(subset)} | "
            f"{fmt(statistics.mean([r['auroc'] for _, r in subset]))} | "
            f"{fmt(statistics.mean([r['tpr10'] for _, r in subset]))} | "
            f"{fmt(statistics.mean([r['tpr20'] for _, r in subset]))} | "
            f"{fmt(statistics.mean([r['fpr10'] for _, r in subset]), 4)} |"
        )
    w("")
    w("| by poison rate | n | AUROC | TPR@10%FPR | TPR@20%FPR |")
    w("|---|---:|---:|---:|---:|")
    for rate in RATE_ORDER:
        subset = [(c, r) for c, r in rows if c["poison_rate"] == rate]
        if not subset:
            continue
        w(
            f"| {rate:.0%} | {len(subset)} | "
            f"{fmt(statistics.mean([r['auroc'] for _, r in subset]))} | "
            f"{fmt(statistics.mean([r['tpr10'] for _, r in subset]))} | "
            f"{fmt(statistics.mean([r['tpr20'] for _, r in subset]))} |"
        )
    w("")
    floor = min((r["auroc"] for _, r in rows), default=float("nan"))
    inverted = sum(1 for _, r in rows if r["auroc"] < 0.5)
    w(
        f"Worst-case cell AUROC **{fmt(floor)}**, inverted cells (AUROC < 0.5) **{inverted}** "
        f"of {len(rows)}."
    )
    w("")
    return "\n".join(out) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--coverage", default="results/coverage/coverage.json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--shift-target", type=float, default=SHIFT_TARGET)
    return parser.parse_args()


def main():
    args = parse_args()
    text = build(args)
    with open(args.out, "w") as handle:
        handle.write(text)
    print(f"[ok] {args.out}  ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
