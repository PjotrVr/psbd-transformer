"""PSBD's wall-clock cost against a plain forward pass, at k = 1, 3, 5, 10, 20.

Reads results/_experiments/psbd_cost/cost.json, written by
experiments/psbd_cost/measure.py. That script times a plain unperturbed
forward pass and PSBD at token_mask, before_attention_norm (the recommended
placement) at each k, batch 128, fp32, on the login-node A100, for a
backdoored ViT-B/16 checkpoint and, when a Swin-S checkpoint was measured too,
that checkpoint. dropout at post_residual is measured too and read here only to
confirm the cost tracks k rather than the placement, since a table row per
placement would repeat the same k axis for numbers that agree to within noise.

    PYTHONPATH=. python scripts/paper/tab_cost.py --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    build_parser,
    fmt,
    load_json,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_cost.py"
SLUG = "psbd_cost"
# The placement the paper's headline table asks the cost of. The 2nd cached
# placement, dropout_post_residual, is read only to confirm the cost tracks k
# rather than the placement, not to carry its own table row.
HEADLINE_PLACEMENT = "token_mask_before_attention_norm"
CHECK_PLACEMENT = "dropout_post_residual"
ARCHITECTURE_LABELS = {"vit": "ViT-B/16", "swin": "Swin-S"}


def load_cost(results_dir: str) -> dict:
    """The cost record, or a SystemExit naming the measurement that has to run first."""
    path = os.path.join(results_dir, "_experiments", SLUG, "cost.json")
    cost = load_json(path)
    if cost is None:
        raise SystemExit(f"{path} does not exist, run experiments/psbd_cost/measure.py")
    return cost


def ms_per_input(seconds_per_input: float) -> str:
    """Seconds per input as milliseconds, 3 decimals, so a k=1 pass reads as more than 0.000."""
    text = f"{seconds_per_input * 1000:.3f}"
    return text


def cost_row(architecture_record: dict, forward_passes: int, placement: str) -> dict:
    """The by_k row at forward_passes for placement, keyed as time_config wrote it."""
    row = architecture_record["placements"][placement]["by_k"][str(forward_passes)]
    return row


def main() -> None:
    args = build_parser(__doc__).parse_args()
    cost = load_cost(args.results_dir)
    inputs = [os.path.join(args.results_dir, "_experiments", SLUG, "cost.json")]

    architectures = list(cost["architectures"])
    forward_pass_counts = cost["forward_pass_counts"]

    header = ["k"]
    for architecture in architectures:
        label = ARCHITECTURE_LABELS.get(architecture, architecture)
        header += [
            f"{label} ms",
            "slowdown",
        ]

    rows = []
    for forward_passes in forward_pass_counts:
        row = [str(forward_passes)]
        for architecture in architectures:
            record = cost["architectures"][architecture]
            plain_seconds = record["plain"]["seconds_per_input"]
            timing = cost_row(record, forward_passes, HEADLINE_PLACEMENT)
            slowdown = timing["seconds_per_input"] / plain_seconds
            row += [
                ms_per_input(timing["seconds_per_input"]),
                fmt(slowdown, places=1) + "x",
            ]
        rows.append(row)

    architecture_names = " and ".join(
        ARCHITECTURE_LABELS.get(architecture, architecture)
        for architecture in architectures
    )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "cost.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            f"Wall-clock cost of PSBD-TM against the number of forward passes k, "
            f"batch 128, fp32, on 1 {cost['gpu_name']}."
        ),
        label="tab:cost",
        header=header,
        rows=rows,
        align="l" + "rrr" * len(architectures),
    )

    macros = {}
    for architecture in architectures:
        record = cost["architectures"][architecture]
        plain_seconds = record["plain"]["seconds_per_input"]
        macros[f"cost_plain_ms_{architecture}"] = (
            ms_per_input(plain_seconds),
            f"{ARCHITECTURE_LABELS.get(architecture, architecture)} milliseconds per "
            "input, 1 plain unperturbed forward pass",
        )
        for forward_passes in (3, 20):
            headline_timing = cost_row(record, forward_passes, HEADLINE_PLACEMENT)
            slowdown = headline_timing["seconds_per_input"] / plain_seconds
            macros[f"cost_slowdown_k{forward_passes}_{architecture}"] = (
                fmt(slowdown, places=2) + "x",
                f"{ARCHITECTURE_LABELS.get(architecture, architecture)} slowdown of "
                f"PSBD at k={forward_passes} against the plain pass, token_mask at "
                "before_attention_norm",
            )

        # The placement check: dropout at post_residual against token_mask at
        # before_attention_norm, both at k=3, should agree closely if the cost
        # is driven by k rather than by which position or operator is probed.
        headline_k3 = cost_row(record, 3, HEADLINE_PLACEMENT)
        check_k3 = cost_row(record, 3, CHECK_PLACEMENT)
        placement_gap = (
            check_k3["seconds_per_input"] / headline_k3["seconds_per_input"] - 1.0
        )
        macros[f"cost_placement_gap_k3_{architecture}"] = (
            fmt(placement_gap, places=3, signed=True),
            f"{ARCHITECTURE_LABELS.get(architecture, architecture)} fractional "
            "difference in seconds per input at k=3 between dropout at "
            "post_residual and token_mask at before_attention_norm",
        )

    write_macros(
        os.path.join(args.paper_dir, "tables", "cost.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(f"cost: {architectures}, k={forward_pass_counts}")


if __name__ == "__main__":
    main()
