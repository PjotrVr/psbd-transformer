"""How much of each perturbation a LayerNorm passes through, measured per block.

This is the measurement the operator claim rests on. A LayerNorm rescales each
token to unit variance, so a perturbation that raises a token's variance is
partly divided away before attention ever sees it, while one that sets whole
tokens to 0 moves the variance very little. The claim was stated in the paper
with hand-typed percentages and no population, so this reads the stored records
and names both.

Survival is the perturbation's relative norm after the LayerNorm divided by its
relative norm before it. 1 means the norm passed it through untouched. The
positions whose name lacks the norm suffix sit AFTER the LayerNorm, so nothing
normalizes them, and they are the control: every operator has to survive at 1
there or the measurement is not measuring a norm.

    PYTHONPATH=. python scripts/paper/mech_layernorm.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import collections
import glob
import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    POSITION_WORDS,
    build_parser,
    fmt,
    is_panel_folder,
    load_json,
    mean_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/mech_layernorm.py"
RECORD = "layernorm_absorption.json"
# Positions in reading order: the 2 that a LayerNorm follows, then the 2 that
# sit after one, which is the control.
POSITIONS = (
    ("before_attention_norm", True),
    ("before_mlp_norm", True),
    ("before_attention", False),
    ("before_mlp", False),
)
OPERATORS = ("token_mask", "channel_mask", "gaussian")


def records(results_dir: str) -> dict[str, list[dict]]:
    """The absorption rows of every panel-shaped model that carries them."""
    found = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", RECORD))):
        folder = os.path.basename(os.path.dirname(path))
        if not is_panel_folder(folder) or "benign" in folder:
            continue
        payload = load_json(path)
        if payload is None or not payload.get("rows"):
            continue
        found[folder] = payload["rows"]
    return found


def survival(rows: list[dict], position: str, operator: str) -> list[float]:
    """Every block's survival of 1 operator at 1 position."""
    values = [
        row["survival"]
        for row in rows
        if row["position"] == position and row["operator"] == operator
    ]
    return values


def main() -> None:
    args = build_parser(__doc__).parse_args()
    found = records(args.results_dir)
    if not found:
        raise SystemExit(f"no {RECORD} under {args.results_dir}")

    pooled: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for rows in found.values():
        for position, _ in POSITIONS:
            for operator in OPERATORS:
                pooled[(position, operator)].extend(survival(rows, position, operator))

    table_rows = []
    for position, normalized in POSITIONS:
        row = [
            POSITION_WORDS.get(position, position),
            "yes" if normalized else "no",
        ]
        for operator in OPERATORS:
            values = pooled[(position, operator)]
            row.append(fmt(mean_or_none(values)))
        table_rows.append(row)

    inputs = [f"{args.results_dir}/*/{RECORD} ({len(found)} models)"]
    write_table(
        path=os.path.join(args.paper_dir, "tables", "layernorm_absorption.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "What a LayerNorm passes through. Survival is the perturbation's "
            "relative norm after the LayerNorm over its relative norm before it, "
            f"averaged over blocks and over {len(found)} models. The last 2 rows sit "
            "after the LayerNorm, where nothing normalizes them, and are the control."
        ),
        label="tab:layernorm-absorption",
        header=["position", "a norm follows"]
        + [operator.replace("_", " ") for operator in OPERATORS],
        rows=table_rows,
        align="ll" + "r" * len(OPERATORS),
    )

    macros = {
        "absorption_models": (
            str(len(found)),
            "models carrying the LayerNorm absorption record",
        ),
    }
    for stem, position in (
        ("attention_input", "before_attention_norm"),
        ("mlp_input", "before_mlp_norm"),
    ):
        for operator in OPERATORS:
            values = pooled[(position, operator)]
            mean = mean_or_none(values)
            macros[f"absorption_{stem}_{operator}_survival"] = (
                fmt(mean),
                f"share of a {operator.replace('_', ' ')} disturbance the {stem} LayerNorm passes through",
            )
            macros[f"absorption_{stem}_{operator}_removed"] = (
                fmt(None if mean is None else 1.0 - mean),
                f"share of a {operator.replace('_', ' ')} disturbance the {stem} LayerNorm absorbs",
            )
    control = [
        value
        for position, normalized in POSITIONS
        if not normalized
        for operator in OPERATORS
        for value in pooled[(position, operator)]
    ]
    macros["absorption_control_survival"] = (
        fmt(mean_or_none(control)),
        "survival at the positions no LayerNorm follows, the control",
    )
    write_macros(
        os.path.join(args.paper_dir, "tables", "layernorm_absorption.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"layernorm absorption: {len(found)} models, control {macros['absorption_control_survival'][0]}"
    )


if __name__ == "__main__":
    main()
