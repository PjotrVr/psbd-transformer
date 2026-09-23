"""The headline gain of the recommended placement, split by poison rate and dataset.

This table was hand-typed in the results section, which is how it came to say 65
models when the panel holds 69 and to quote the matched-rule gain beside
adaptive-rule readings. It is the paper's summary of its own result, so it is the
last table that should have been typed.

Rows are the whole panel, then 1 per poison rate, then 1 per dataset. Every row
is a paired difference on the models carrying both placements, with a bootstrap
interval over models. The leave-one-attack-out row is the smallest gain that
survives dropping any single attack, which is the check that no 1 attack carries
the result.

    PYTHONPATH=. python scripts/paper/tab_gains.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import collections
import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defences.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    attack_label,
    bootstrap_ci,
    build_parser,
    ci_text,
    dataset_label,
    fmt,
    load_coverage,
    load_psbd_metrics,
    mean_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_gains.py"
# The deployable rule. Every number in this table is read at it, so the table
# cannot mix rules the way the typed version did.
RULE = "adaptive"


def paired_cells(results_dir: str, cells: list[dict]) -> list[dict]:
    """1 record per model carrying both placements, with its gain and its facets."""
    paired = []
    for cell in cells:
        report = load_psbd_metrics(results_dir, cell["folder_name"])
        if report is None:
            continue
        readings = {}
        for name, placement in (
            ("ours", RECOMMENDED_PLACEMENT),
            ("published", PUBLISHED_PLACEMENT),
        ):
            block = psbd_values(report, placement, RULE)
            quantile = (block or {}).get(HEADLINE_KEY)
            readings[name] = None if quantile is None else quantile.get("auroc")
        if readings["ours"] is None or readings["published"] is None:
            continue
        paired.append(
            {
                "folder": cell["folder_name"],
                "dataset": cell["dataset"],
                "attack": cell["attack"],
                "poison_rate": cell["poison_rate"],
                "gain": readings["ours"] - readings["published"],
                "ours": readings["ours"],
                "published": readings["published"],
            }
        )
    return paired


def gain_row(label: str, group: list[dict], resamples: int, seed: int) -> list[str]:
    """1 row: the label, the models behind it, the mean gain and its interval."""
    deltas = [record["gain"] for record in group]
    low, high = bootstrap_ci(deltas, resamples, seed)
    row = [
        label,
        str(len(deltas)),
        fmt(mean_or_none(deltas), signed=True),
        ci_text(low, high),
    ]
    return row


def leave_one_attack_out(
    paired: list[dict], resamples: int, seed: int
) -> tuple[str, float, float]:
    """The weakest gain left after dropping any single attack, and its interval.

    Returns the attack whose removal costs the most, so the sentence in the prose
    can name it rather than assert that no attack carries the result.
    """
    attacks = sorted({record["attack"] for record in paired})
    worst = None
    for attack in attacks:
        kept = [record for record in paired if record["attack"] != attack]
        if not kept:
            continue
        deltas = [record["gain"] for record in kept]
        mean = mean_or_none(deltas)
        low, high = bootstrap_ci(deltas, resamples, seed)
        if worst is None or mean < worst[1]:
            worst = (attack, mean, low)
    return worst


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage = load_coverage(args.results_dir)
    cells = [cell for cell in coverage["cells"] if cell.get("asr_class") == "clears"]
    paired = paired_cells(args.results_dir, cells)
    inputs = [
        f"{args.results_dir}/coverage/coverage.json ({len(cells)} clearing cells)",
        f"{args.results_dir}/*/psbd_metrics.json",
    ]

    by_rate = collections.defaultdict(list)
    by_dataset = collections.defaultdict(list)
    for record in paired:
        by_rate[record["poison_rate"]].append(record)
        by_dataset[record["dataset"]].append(record)

    rows = [gain_row("all models", paired, args.bootstrap, args.seed)]
    rows.append(["\\multicolumn{4}{l}{\\emph{by poison rate}}"])
    for rate in sorted(by_rate):
        rows.append(
            gain_row(
                f"{rate * 100:g}% poisoning", by_rate[rate], args.bootstrap, args.seed
            )
        )
    rows.append(["\\multicolumn{4}{l}{\\emph{by dataset}}"])
    for dataset in sorted(by_dataset, key=lambda name: -len(by_dataset[name])):
        rows.append(
            gain_row(
                dataset_label(dataset),
                by_dataset[dataset],
                args.bootstrap,
                args.seed,
            )
        )

    write_table(
        path=os.path.join(args.paper_dir, "tables", "gains.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Paired AUROC gain of token masking at the attention input over dropout "
            "after both residual adds, at the adaptive rule and the headline "
            f"quantile, with 95\\% bootstrap intervals over models from "
            f"{args.bootstrap} resamples. Every row is a paired difference on the "
            "models carrying both placements."
        ),
        label="tab:gains",
        header=["subset", "models", "gain", "95% interval"],
        rows=rows,
        align="lrrl",
    )

    lowest_rate = min(by_rate)
    highest_rate = max(by_rate)
    attack, worst_mean, worst_low = leave_one_attack_out(
        paired, args.bootstrap, args.seed
    )
    macros = {
        "gains_models": (str(len(paired)), "models carrying both compared placements"),
        "gains_all": (
            fmt(mean_or_none([r["gain"] for r in paired]), signed=True),
            "mean paired AUROC gain over every compared model at the adaptive rule",
        ),
        "gains_lowest_rate": (
            fmt(mean_or_none([r["gain"] for r in by_rate[lowest_rate]]), signed=True),
            f"mean paired gain at {lowest_rate * 100:g}% poisoning",
        ),
        "gains_lowest_rate_n": (
            str(len(by_rate[lowest_rate])),
            f"models at {lowest_rate * 100:g}% poisoning",
        ),
        "gains_highest_rate": (
            fmt(mean_or_none([r["gain"] for r in by_rate[highest_rate]]), signed=True),
            f"mean paired gain at {highest_rate * 100:g}% poisoning",
        ),
        "gains_highest_rate_n": (
            str(len(by_rate[highest_rate])),
            f"models at {highest_rate * 100:g}% poisoning",
        ),
        "gains_leave_one_attack_out": (
            fmt(worst_mean, signed=True),
            "smallest mean gain left after dropping any single attack",
        ),
        "gains_leave_one_attack_out_low": (
            fmt(worst_low, signed=True),
            "lower bootstrap bound of gains_leave_one_attack_out",
        ),
        "gains_leave_one_attack_out_attack": (
            attack_label(attack),
            "the attack whose removal costs the gain the most",
        ),
        "gains_positive_datasets": (
            str(
                sum(
                    1
                    for dataset in by_dataset
                    if (mean_or_none([r["gain"] for r in by_dataset[dataset]]) or 0) > 0
                )
            ),
            "datasets whose mean paired gain is positive",
        ),
        "gains_datasets": (str(len(by_dataset)), "datasets in the gain table"),
    }
    for dataset, group in by_dataset.items():
        macros[f"gains_{dataset}"] = (
            fmt(mean_or_none([r["gain"] for r in group]), signed=True),
            f"mean paired gain on {dataset_label(dataset)} over {len(group)} models",
        )
    write_macros(
        os.path.join(args.paper_dir, "tables", "gains.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"gains: {len(paired)} paired models, all {macros['gains_all'][0]}, "
        f"leave-one-out worst {attack} {macros['gains_leave_one_attack_out'][0]}"
    )


if __name__ == "__main__":
    main()
