"""Every attention-side probe we measured, inside the declared basis and outside it.

Attention is the only operation in a ViT block that moves information between
tokens, so it is where the mechanism of \\cref{sec:mechanism} says a token-level
perturbation should work. The declared basis holds 9 attention-side placements
and the sweeps cover more than that, including head masking inside
multi-head attention, which the basis never declared.

Those extra probes cannot join the basis ranking. The declaration is
pre-registered precisely so that a placement cannot be added to the ranking after
its score is known, and adding them would be that. They are reported here
instead, marked as outside the basis, with the models behind each shown rather
than pooled, so a reader can see that the search was wide and that the
pre-registered winner is still the winner.

Depth bands and mask-seed replicates are left out, because they are the same
placement measured again rather than a different probe.

    PYTHONPATH=. python scripts/paper/tab_attention.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import collections
import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defences.decision import RECOMMENDED_PLACEMENT  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    build_parser,
    fmt,
    load_coverage,
    load_declaration,
    load_psbd_metrics,
    mean_or_none,
    placement_words,
    split_placement,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_attention.py"
RULE = "adaptive"
# Every named tensor boundary of the attention sublayer, from its input to the
# stream just after its residual add.
ATTENTION_POSITIONS = (
    "before_attention_norm",
    "before_attention",
    "attention_heads",
    "before_attention_residual",
    "after_attention_residual",
    "attention_norm_out",
)
# Below this a mean says more about which models happened to be swept than about
# the probe.
MIN_CELLS = 5
TPR_KEYS = ("q0.10", "q0.20")


def is_plain_attention_probe(placement: str) -> bool:
    """Whether a cache directory is an attention-side probe at full depth."""
    parsed = split_placement(placement)
    if parsed["position"] not in ATTENTION_POSITIONS:
        return False
    if parsed["variant"] is not None or parsed["block_range"] is not None:
        return False
    return True


def readings(results_dir: str, folders: list[str]) -> dict[str, dict[str, list[float]]]:
    """Per attention probe, every clearing model's AUROC and TPR at both budgets."""
    found: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: {"auroc": [], TPR_KEYS[0]: [], TPR_KEYS[1]: []}
    )
    for folder in folders:
        report = load_psbd_metrics(results_dir, folder)
        if report is None:
            continue
        for placement in report.get("placements", {}):
            if not is_plain_attention_probe(placement):
                continue
            block = psbd_values(report, placement, RULE)
            if block is None:
                continue
            quantile = block.get(HEADLINE_KEY)
            if quantile is None or quantile.get("auroc") is None:
                continue
            found[placement]["auroc"].append(quantile["auroc"])
            for key in TPR_KEYS:
                cell = block.get(key)
                if cell is not None and cell.get("tpr") is not None:
                    found[placement][key].append(cell["tpr"])
    return found


def main() -> None:
    args = build_parser(__doc__).parse_args()
    declaration = load_declaration(args.declaration)
    declared = {entry["id"] for entry in declaration["basis"]}
    coverage = load_coverage(args.results_dir)
    folders = [
        cell["folder_name"]
        for cell in coverage["cells"]
        if cell.get("asr_class") == "clears"
    ]
    found = readings(args.results_dir, folders)

    ordered = sorted(
        (
            (name, values)
            for name, values in found.items()
            if len(values["auroc"]) >= MIN_CELLS
        ),
        key=lambda item: -(mean_or_none(item[1]["auroc"]) or 0.0),
    )
    rows = []
    for placement, values in ordered:
        rows.append(
            [
                placement_words(placement),
                "yes" if placement in declared else "no",
                str(len(values["auroc"])),
                fmt(mean_or_none(values["auroc"])),
                fmt(mean_or_none(values[TPR_KEYS[0]]), places=2),
                fmt(mean_or_none(values[TPR_KEYS[1]]), places=2),
            ]
        )

    inputs = [
        f"{args.results_dir}/coverage/coverage.json ({len(folders)} clearing cells)",
        f"{args.results_dir}/*/psbd_metrics.json",
    ]
    write_table(
        path=os.path.join(args.paper_dir, "tables", "attention_probes.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Every attention-side probe with at least "
            f"{MIN_CELLS} backdoored ViT-B/16 models, at the adaptive rule and the "
            "headline quantile. The second column says whether the pre-registered "
            "basis declares the placement. The probes it does not declare are not "
            "part of the ranking in \\cref{tab:basis-ranking} and were not eligible "
            "to be, since adding a placement to a ranking after reading its score is "
            "what the declaration exists to prevent. Coverage is uneven, so rows are "
            "not paired."
        ),
        label="tab:attention-probes",
        header=["probe", "in basis", "n", "AUROC", "TPR@10", "TPR@20"],
        rows=rows,
        align="llrrrr",
    )

    outside = [(name, values) for name, values in ordered if name not in declared]
    inside = [(name, values) for name, values in ordered if name in declared]
    best_outside_name, best_outside = outside[0]
    recommended = mean_or_none(found[RECOMMENDED_PLACEMENT]["auroc"])
    macros = {
        "attention_probes_total": (
            str(len(ordered)),
            f"attention-side probes measured on at least {MIN_CELLS} clearing models",
        ),
        "attention_probes_in_basis": (
            str(len(inside)),
            "of those the pre-registered basis declares",
        ),
        "attention_probes_outside_basis": (
            str(len(outside)),
            "of those the basis does not declare, so they sit outside the ranking",
        ),
        "attention_probes_best_outside": (
            fmt(mean_or_none(best_outside["auroc"])),
            f"best mean AUROC among the probes outside the basis, {best_outside_name}",
        ),
        "attention_probes_best_outside_name": (
            placement_words(best_outside_name),
            "the probe outside the basis with the highest mean AUROC",
        ),
        "attention_probes_best_outside_n": (
            str(len(best_outside["auroc"])),
            "models behind attention_probes_best_outside",
        ),
        "attention_probes_beating_recommended": (
            str(
                sum(
                    1
                    for _, values in outside
                    if (mean_or_none(values["auroc"]) or 0.0) > (recommended or 0.0)
                )
            ),
            "probes outside the basis whose mean AUROC exceeds the recommended placement",
        ),
    }
    head = found.get("attention_heads_head_mask")
    if head and len(head["auroc"]) >= MIN_CELLS:
        macros["attention_head_mask_auroc"] = (
            fmt(mean_or_none(head["auroc"])),
            "mean AUROC of masking whole attention heads at the adaptive rule",
        )
        macros["attention_head_mask_n"] = (
            str(len(head["auroc"])),
            "clearing models carrying the head-masking probe",
        )
        macros["attention_head_mask_tpr_ten"] = (
            fmt(mean_or_none(head[TPR_KEYS[0]]), places=2),
            "mean TPR of head masking at a 10% false-positive budget",
        )
    write_macros(
        os.path.join(args.paper_dir, "tables", "attention_probes.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"attention probes: {len(ordered)} measured, {len(inside)} declared, "
        f"{len(outside)} outside, best outside {macros['attention_probes_best_outside'][0]}"
    )


if __name__ == "__main__":
    main()
