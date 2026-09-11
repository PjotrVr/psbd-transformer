"""Activation patching across the panel: where in depth the backdoor is causally located.

Every clearing cell carries an activation_patching.json with the normalised
recovery of the clean answer when 1 token group at 1 layer is patched from the
clean run into the triggered run. Recovery near 1 at a (layer, group) means the
backdoor was carried there and nowhere else at that depth. This averages the
residual-stream site over cells per attack and plots recovery against depth for
the trigger's own tokens, the class token and a random group of the trigger's
size, which is the null. The crossover layer, where the class token first
recovers more than the trigger tokens, is the depth at which attention has
finished routing the trigger.

    PYTHONPATH=. python scripts/paper/mech_activation_patching.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from defences.decision import EASY_ATTACKS, HARD_ATTACKS  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    OKABE_ITO,
    build_parser,
    clearing_cells,
    figure_sidecar,
    fmt,
    load_coverage,
    load_json,
    mean_or_none,
    save_figure,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/mech_activation_patching.py"
SITE = "resid"
GROUPS = ("trigger", "cls", "random_same_size")
GROUP_LABELS = {"trigger": "trigger tokens", "cls": "class token", "random_same_size": "random tokens, same count"}
LOCAL_ATTACKS = ("badnet_a2o", "lc", "tact")
RECOVERED = 0.5
# The early band the trigger-token recovery is averaged over, the first half of the stack.
EARLY_LAST_LAYER = 6


def recovery_by_layer(record: dict) -> dict[str, dict[int, float]]:
    """group to layer to recovery, at the residual-stream site."""
    curves: dict[str, dict[int, float]] = {group: {} for group in GROUPS}
    for row in record["rows"]:
        if row["site"] == SITE and row["group"] in curves:
            curves[row["group"]][row["layer"]] = row["recovery"]
    return curves


def mean_curves(records: list[dict]) -> dict[str, list[float]]:
    """Per group, the mean recovery over records at each layer, layers ascending."""
    layers = sorted({layer for record in records for layer in recovery_by_layer(record)["trigger"]})
    averaged = {}
    for group in GROUPS:
        averaged[group] = [
            mean_or_none([recovery_by_layer(record)[group][layer] for record in records if layer in recovery_by_layer(record)[group]])
            for layer in layers
        ]
    averaged["layers"] = layers
    return averaged


def crossover_layer(curves: dict[str, list[float]]) -> int | None:
    """The first layer where the class token recovers more than the trigger tokens."""
    for layer, trigger, cls in zip(curves["layers"], curves["trigger"], curves["cls"]):
        if cls is not None and trigger is not None and cls > trigger:
            return layer
    return None


def last_full_layer(curves: dict[str, list[float]]) -> int | None:
    """The last layer at which the trigger tokens alone still recover more than half."""
    kept = [layer for layer, value in zip(curves["layers"], curves["trigger"]) if value is not None and value >= RECOVERED]
    last = max(kept) if kept else None
    return last


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    cells = clearing_cells(load_coverage(args.results_dir))
    by_attack: dict[str, list[dict]] = {}
    for cell in cells:
        record = load_json(os.path.join(args.results_dir, cell["folder_name"], "activation_patching.json"))
        if record is not None:
            by_attack.setdefault(cell["attack"], []).append(record)
    inputs = [coverage_path, f"{args.results_dir}/<folder>/activation_patching.json ({sum(map(len, by_attack.values()))} cells)"]

    attacks = [attack for attack in list(HARD_ATTACKS) + list(EASY_ATTACKS) if attack in by_attack]
    curves = {attack: mean_curves(by_attack[attack]) for attack in attacks}

    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True, sharey=True)
    for axis, attack in zip(axes.flat, attacks):
        for index, group in enumerate(GROUPS):
            axis.plot(curves[attack]["layers"], curves[attack][group], marker="o", markersize=3, linewidth=1.3, color=OKABE_ITO[index], label=GROUP_LABELS[group])
        axis.set_title(f"{attack} (n={len(by_attack[attack])})", fontsize=9)
        axis.axhline(0.0, color="0.6", linewidth=0.8)
        axis.set_ylim(-0.1, 1.1)
    for axis in axes.flat[len(attacks):]:
        axis.axis("off")
    for axis in axes[1]:
        axis.set_xlabel("layer")
    for axis in axes[:, 0]:
        axis.set_ylabel("recovery of the clean answer")
    axes.flat[0].legend(fontsize=7, loc="lower left")
    figure_path = os.path.join(args.paper_dir, "figures", "mech_activation_patching.pdf")
    save_figure(fig, figure_path)
    figure_sidecar(figure_path.replace(".pdf", ".json"), GENERATOR, inputs, {"curves": curves, "cells_per_attack": {a: len(r) for a, r in by_attack.items()}})

    rows = []
    stats = {}
    for attack in attacks:
        curve = curves[attack]
        random_max = max(value for value in curve["random_same_size"] if value is not None)
        early_trigger = mean_or_none([value for layer, value in zip(curve["layers"], curve["trigger"]) if layer <= EARLY_LAST_LAYER and value is not None])
        final_cls = curve["cls"][-1]
        stats[attack] = {"crossover": crossover_layer(curve), "last_full": last_full_layer(curve), "random_max": random_max, "early_trigger": early_trigger, "final_cls": final_cls}
        rows.append(
            [
                attack,
                "local" if attack in LOCAL_ATTACKS else "global",
                str(len(by_attack[attack])),
                fmt(early_trigger),
                str(stats[attack]["last_full"]) if stats[attack]["last_full"] is not None else "--",
                str(stats[attack]["crossover"]) if stats[attack]["crossover"] is not None else "--",
                fmt(final_cls),
                fmt(random_max),
            ]
        )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "activation_patching.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Activation patching at the residual stream, averaged over clearing cells "
            "per attack: mean recovery when the trigger's own tokens are patched over "
            "the first 6 layers, the last layer at which those tokens alone recover "
            "at least half the clean answer, the first layer at which the class token "
            "recovers more than the trigger tokens, the class token's recovery at the "
            "last layer, and the largest recovery a random token group of the same "
            "count reaches at any layer, the null."
        ),
        label="tab:activation-patching",
        header=["attack", "trigger", "n", "trigger tokens, layers 1 to 6", "last layer trigger tokens suffice", "crossover layer", "class token, last layer", "random null, max"],
        rows=rows,
        align="llrrrrrr",
    )

    local = [stats[a] for a in attacks if a in LOCAL_ATTACKS]
    global_ = [stats[a] for a in attacks if a not in LOCAL_ATTACKS]
    macros = {
        "patching_cells": (str(sum(map(len, by_attack.values()))), "clearing cells with activation patching"),
        "patching_early_layers": (f"1 to {EARLY_LAST_LAYER}", "the layer band the early trigger-token recovery is averaged over"),
        "patching_badnet_last_full": (str(stats.get("badnet_a2o", {}).get("last_full")), "badnet_a2o last layer at which the trigger tokens alone recover at least half the clean answer"),
        "patching_random_null_max": (fmt(max(s["random_max"] for s in stats.values())), "largest recovery any random token group reaches, over every attack and layer"),
        "patching_badnet_early_trigger": (fmt(stats.get("badnet_a2o", {}).get("early_trigger")), "badnet_a2o mean recovery from the trigger tokens over layers 1 to 6"),
        "patching_badnet_crossover": (str(stats.get("badnet_a2o", {}).get("crossover")), "badnet_a2o layer at which the class token first recovers more than the trigger tokens"),
        "patching_badnet_final_cls": (fmt(stats.get("badnet_a2o", {}).get("final_cls")), "badnet_a2o class-token recovery at the last layer"),
        "patching_local_early_trigger": (fmt(mean_or_none([s["early_trigger"] for s in local])), "mean early trigger-token recovery over the local-trigger attacks"),
        "patching_global_early_trigger": (fmt(mean_or_none([s["early_trigger"] for s in global_])), "mean early trigger-token recovery over the global-trigger attacks"),
        "patching_local_crossover": (fmt(mean_or_none([s["crossover"] for s in local if s["crossover"] is not None]), places=1), "mean crossover layer over the local-trigger attacks"),
        "patching_blend_early_trigger": (fmt(stats.get("blend", {}).get("early_trigger")), "blend mean recovery from the trigger tokens over layers 1 to 6"),
    }
    write_macros(os.path.join(args.paper_dir, "tables", "activation_patching.macros.json"), GENERATOR, inputs, macros)
    print("patching: " + ", ".join(f"{a}: cross {s['crossover']} last {s['last_full']} rand {fmt(s['random_max'])}" for a, s in stats.items()))


if __name__ == "__main__":
    main()
