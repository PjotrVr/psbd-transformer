"""H20x: the family split, input-side against residual-adjacent, on the basis panel.

H20 read the split with 1 operator, dropout, moved across positions on CIFAR-10.
The basis panel carries no dropout at any input-side position, so the
operator-matched reading here holds token_mask fixed: 3 input-side placements
(before_attention_norm, before_mlp_norm, both_sublayer_inputs) against 2
residual-adjacent ones (before_attention_residual, after_attention_residual).
The basis file's family tags mix operators (gaussian, scale_up and dropout sit
inside the same 2 families), and that confounded reading is kept as a second
table so the 2 definitions can be compared side by side. The dropout-matched row
waits on the operator-matched sweep and prints \\pending until it runs.

A cell's family value is the mean AUROC at q0.25 over whichever of the family's
placements are cached on it. The gap is input-side minus residual-adjacent
within the cell, read at the matched 0.6 rate, the cross-placement comparison
device, and separately at the adaptive 0.8 rate.

    PYTHONPATH=. python scripts/paper/tab_family_split.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defences.decision import EASY_ATTACKS, HARD_ATTACKS  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    bootstrap_ci,
    build_parser,
    ci_text,
    clearing_cells,
    fmt,
    load_coverage,
    load_declaration,
    load_psbd_metrics,
    mean_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_family_split.py"
PENDING = r"\pending"
DATASET_ORDER = ("cifar10", "cifar100", "gtsrb", "tiny", "svhn", "eurosat")
RATE_ORDER = (0.01, 0.05, 0.1)
RULES = ("matched", "adaptive")
FAMILIES = ("input_side", "residual_adjacent")

# The operator-matched definition: token_mask at every input-side and
# residual-adjacent position the basis carries, no other operator.
TOKEN_MASK_INPUT_SIDE = (
    "before_attention_norm_token_mask",
    "before_mlp_norm_token_mask",
    "both_sublayer_inputs_token_mask",
)
TOKEN_MASK_RESIDUAL_ADJACENT = (
    "before_attention_residual_token_mask",
    "after_attention_residual_token_mask",
)
OPERATOR_MATCHED = "token_mask"
CONFOUNDED = "basis_tags"

# Local triggers occupy a fixed spatial patch, global triggers cover the whole
# image, the split H27 found token_mask separates cleanly (patch 0.985 against
# warp 0.747).
LOCAL_ATTACKS = ("badnet_a2o", "lc", "tact")
GLOBAL_ATTACKS = ("blend", "bpp", "lf", "sig", "wanet", "adaptive_blend")

GAP_HEADER = [
    "subset",
    "n match",
    "gap matched06",
    "95% CI",
    "n adapt",
    "gap adaptive08",
    "95% CI",
]


def family_definitions(declaration: dict) -> dict[str, dict[str, list[str]]]:
    """The 2 definitions, each a family name to its placement ids.

    The confounded definition reads the basis file's family tags as written, the
    operator-matched one keeps only the token_mask members of each family.
    """
    basis = declaration["basis"]
    confounded = {
        family: [entry["id"] for entry in basis if entry["family"] == family]
        for family in FAMILIES
    }
    operator_matched = {
        "input_side": list(TOKEN_MASK_INPUT_SIDE),
        "residual_adjacent": list(TOKEN_MASK_RESIDUAL_ADJACENT),
    }
    declared = {entry["id"] for entry in basis}
    missing = [
        placement
        for family in operator_matched.values()
        for placement in family
        if placement not in declared
    ]
    if missing:
        raise SystemExit(f"basis file lacks operator-matched placements: {missing}")
    definitions = {OPERATOR_MATCHED: operator_matched, CONFOUNDED: confounded}
    return definitions


def family_mean_auroc(
    report: dict | None, placement_ids: list[str], rule: str
) -> float | None:
    """1 cell's family value: the mean over whichever family members are cached."""
    values = []
    for placement in placement_ids:
        block = psbd_values(report, placement, rule)
        if block is not None and HEADLINE_KEY in block:
            values.append(block[HEADLINE_KEY]["auroc"])
    mean = mean_or_none(values)
    return mean


def measure_cell(
    results_dir: str, folder: str, definitions: dict[str, dict[str, list[str]]]
) -> dict[str, dict[str, dict[str, float | None]]]:
    """definition, then rule, then family, to this cell's family value."""
    report = load_psbd_metrics(results_dir, folder)
    measured = {
        definition: {
            rule: {
                family: family_mean_auroc(report, placement_ids, rule)
                for family, placement_ids in families.items()
            }
            for rule in RULES
        }
        for definition, families in definitions.items()
    }
    return measured


def family_gap(cells: list[dict], definition: str, rule: str) -> list[float]:
    """input_side minus residual_adjacent, paired, over cells where both have a value."""
    deltas = []
    for cell in cells:
        values = cell["families"][definition][rule]
        input_side = values["input_side"]
        residual_adjacent = values["residual_adjacent"]
        if input_side is not None and residual_adjacent is not None:
            deltas.append(input_side - residual_adjacent)
    return deltas


def subsets(cells: list[dict]) -> list[tuple[str, list[dict]]]:
    """The rows every gap table shares: all, per dataset, per rate, hard and easy."""
    groups = [("all", cells)]
    for dataset in DATASET_ORDER:
        groups.append((dataset, [c for c in cells if c["dataset"] == dataset]))
    for rate in RATE_ORDER:
        groups.append(
            (f"rate {rate:g}", [c for c in cells if c["poison_rate"] == rate])
        )
    groups.append(("hard attacks", [c for c in cells if c["attack"] in HARD_ATTACKS]))
    groups.append(("easy attacks", [c for c in cells if c["attack"] in EASY_ATTACKS]))
    return groups


def gap_row(
    name: str, cells: list[dict], definition: str, resamples: int, seed: int
) -> list[str]:
    """1 table row: the gap at both rules with its interval and coverage count."""
    row = [name]
    for rule in RULES:
        deltas = family_gap(cells, definition, rule)
        low, high = bootstrap_ci(deltas, resamples, seed)
        row += [
            str(len(deltas)),
            fmt(mean_or_none(deltas), signed=True),
            ci_text(low, high),
        ]
    return row


def placement_list(placement_ids: list[str]) -> str:
    text = ", ".join(f"`{placement}`" for placement in placement_ids)
    return text


def write_gap_table(
    args, cells: list[dict], definition: str, families: dict[str, list[str]], inputs
) -> None:
    """The per-subset gap table for 1 definition, the dropout row pending on the sweep."""
    rows = [
        gap_row(name, subset, definition, args.bootstrap, args.seed)
        for name, subset in subsets(cells)
    ]
    if definition == OPERATOR_MATCHED:
        rows.append(["all, dropout (pending sweep)"] + [PENDING] * 6)
        path = os.path.join(args.paper_dir, "tables", "family_split.tex")
        label = "tab:family-split"
        caption = (
            "The within-cell family gap with the operator held fixed at token_mask: "
            f"input-side ({placement_list(families['input_side'])}) minus "
            f"residual-adjacent ({placement_list(families['residual_adjacent'])}), "
            "each family's per-cell value the mean AUROC at q0.25 over whichever of "
            "its members are cached. Read at the matched 0.6 rate and separately at "
            f"the adaptive 0.8 rate, {args.bootstrap}-resample bootstrap 95\\% "
            "intervals. The dropout row is H20's original operator, which the basis "
            r"panel carries at no input-side position, so it prints \pending until "
            "the operator-matched sweep runs."
        )
    else:
        path = os.path.join(args.paper_dir, "tables", "family_split_confounded.tex")
        label = "tab:family-split-confounded"
        caption = (
            "The same gap read from the basis file's family tags as written, "
            f"input-side ({placement_list(families['input_side'])}) minus "
            f"residual-adjacent ({placement_list(families['residual_adjacent'])}). "
            "The 2 families mix operators (gaussian, scale_up, channel_mask and "
            "dropout beside token_mask), so this reading confounds position with "
            "operator and is kept only for comparison with the operator-matched "
            "table."
        )
    write_table(
        path=path,
        generator=GENERATOR,
        inputs=inputs,
        caption=caption,
        label=label,
        header=GAP_HEADER,
        rows=rows,
        align="lrrlrrl",
    )


def leave_one_out_means(
    cells: list[dict], attacks: list[str], resamples: int, seed: int
) -> tuple[list[list[str]], dict[str, float | None]]:
    """The matched gap refit with 1 attack dropped at a time, rows and means."""
    rows = []
    means = {}
    for attack in attacks:
        rest = [c for c in cells if c["attack"] != attack]
        deltas = family_gap(rest, OPERATOR_MATCHED, "matched")
        low, high = bootstrap_ci(deltas, resamples, seed)
        means[attack] = mean_or_none(deltas)
        rows.append(
            [
                f"without {attack}",
                str(len(deltas)),
                fmt(means[attack], signed=True),
                ci_text(low, high),
            ]
        )
    return rows, means


def write_leave_one_out_table(
    args, cells: list[dict], attacks: list[str], inputs
) -> dict:
    rows, means = leave_one_out_means(cells, attacks, args.bootstrap, args.seed)
    write_table(
        path=os.path.join(args.paper_dir, "tables", "family_split_leave_one_out.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "The operator-matched gap at the matched 0.6 rate, refit with 1 attack "
            f"dropped at a time over the {len(attacks)} attacks present in the "
            "65-cell panel, so a gap driven by a single attack would show as a "
            "refit crossing 0."
        ),
        label="tab:family-split-loo",
        header=["dropped attack", "n", "gap matched06", "95% CI"],
        rows=rows,
        align="lrrl",
    )
    return means


def write_local_global_table(
    args, cells: list[dict], attacks_present: list[str], inputs
) -> tuple[list[dict], list[dict]]:
    """The gap by trigger locality, for both definitions, absent attacks named."""
    local_cells = [c for c in cells if c["attack"] in LOCAL_ATTACKS]
    global_cells = [c for c in cells if c["attack"] in GLOBAL_ATTACKS]
    rows = []
    for definition, tag in (
        (OPERATOR_MATCHED, "token_mask"),
        (CONFOUNDED, "basis tags"),
    ):
        rows.append(
            gap_row(
                f"local triggers, {tag}",
                local_cells,
                definition,
                args.bootstrap,
                args.seed,
            )
        )
        rows.append(
            gap_row(
                f"global triggers, {tag}",
                global_cells,
                definition,
                args.bootstrap,
                args.seed,
            )
        )

    absent = sorted(set(LOCAL_ATTACKS + GLOBAL_ATTACKS) - set(attacks_present))
    absent_text = (
        " "
        + ", ".join(absent)
        + " clear no cell on this panel and contribute to neither row."
        if absent
        else ""
    )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "family_split_local_global.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "The family gap split by trigger locality, under the operator-matched "
            "token_mask definition and the confounded basis-tag definition. Local "
            "triggers occupy a fixed spatial patch (" + ", ".join(LOCAL_ATTACKS) + "), "
            "global triggers cover the whole image ("
            + ", ".join(GLOBAL_ATTACKS)
            + ")."
            + absent_text
        ),
        label="tab:family-split-local-global",
        header=[
            "trigger locality",
            "n match",
            "gap matched06",
            "95% CI",
            "n adapt",
            "gap adaptive08",
            "95% CI",
        ],
        rows=rows,
        align="lrrlrrl",
    )
    return local_cells, global_cells


def gap_macros(
    stem: str,
    cells: list[dict],
    definition: str,
    rule: str,
    meaning: str,
    resamples: int,
    seed: int,
) -> dict[str, tuple[str, str]]:
    """A gap's mean and its 2 interval bounds under 1 macro stem."""
    deltas = family_gap(cells, definition, rule)
    low, high = bootstrap_ci(deltas, resamples, seed)
    macros = {
        stem: (
            fmt(mean_or_none(deltas), signed=True),
            f"{meaning}, over the {len(deltas)} cells it covers",
        ),
        f"{stem}_low": (
            fmt(low, signed=True),
            f"lower bound of the 95% bootstrap interval on {stem}",
        ),
        f"{stem}_high": (
            fmt(high, signed=True),
            f"upper bound of the 95% bootstrap interval on {stem}",
        ),
    }
    return macros


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    coverage = load_coverage(args.results_dir)
    declaration = load_declaration(args.declaration)
    cells = clearing_cells(coverage)
    definitions = family_definitions(declaration)

    for cell in cells:
        cell["families"] = measure_cell(
            args.results_dir, cell["folder_name"], definitions
        )

    inputs = [
        coverage_path,
        args.declaration,
        f"{args.results_dir}/<folder>/psbd_metrics.json (65 cells)",
    ]
    attacks_present = sorted({c["attack"] for c in cells})

    for definition, families in definitions.items():
        write_gap_table(args, cells, definition, families, inputs)
    leave_one_out = write_leave_one_out_table(args, cells, attacks_present, inputs)
    local_cells, global_cells = write_local_global_table(
        args, cells, attacks_present, inputs
    )

    macros = {}
    macros.update(
        gap_macros(
            "family_gap_matched",
            cells,
            OPERATOR_MATCHED,
            "matched",
            "mean paired AUROC gap, input-side minus residual-adjacent, token_mask at "
            "every position, at the matched 0.6 rate",
            args.bootstrap,
            args.seed,
        )
    )
    macros.update(
        gap_macros(
            "family_gap_adaptive",
            cells,
            OPERATOR_MATCHED,
            "adaptive",
            "mean paired AUROC gap, input-side minus residual-adjacent, token_mask at "
            "every position, at the adaptive 0.8 rate",
            args.bootstrap,
            args.seed,
        )
    )
    macros.update(
        gap_macros(
            "family_gap_confounded_matched",
            cells,
            CONFOUNDED,
            "matched",
            "mean paired AUROC gap under the basis file's operator-mixing family tags, "
            "at the matched 0.6 rate",
            args.bootstrap,
            args.seed,
        )
    )
    macros.update(
        gap_macros(
            "family_gap_local",
            local_cells,
            OPERATOR_MATCHED,
            "matched",
            "mean paired token_mask family gap at the matched 0.6 rate, local-trigger cells only",
            args.bootstrap,
            args.seed,
        )
    )
    macros.update(
        gap_macros(
            "family_gap_global",
            global_cells,
            OPERATOR_MATCHED,
            "matched",
            "mean paired token_mask family gap at the matched 0.6 rate, global-trigger cells only",
            args.bootstrap,
            args.seed,
        )
    )
    refit_means = [mean for mean in leave_one_out.values() if mean is not None]
    macros["family_gap_min_leave_one_attack_out"] = (
        fmt(min(refit_means), signed=True) if refit_means else "--",
        "smallest mean matched-06 token_mask family gap over the leave-one-attack-out "
        "refits, the weakest the gap gets when any single attack is removed",
    )
    write_macros(
        os.path.join(args.paper_dir, "tables", "family_split.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"family_split: token_mask gap matched {macros['family_gap_matched'][0]} "
        f"{ci_text(*bootstrap_ci(family_gap(cells, OPERATOR_MATCHED, 'matched'), args.bootstrap, args.seed))}, "
        f"basis-tag gap matched {macros['family_gap_confounded_matched'][0]}"
    )


if __name__ == "__main__":
    main()
