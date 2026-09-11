"""The backdoor direction: its depth profile, its causal ablation and its crystallisation.

3 records. backdoor_neurons.json on the CIFAR-10 checkpoints gives the relative
backdoor-direction norm, the mean trigger-activated change and the clean-versus-
triggered CKA at every layer, with the benign reference probed by the same
trigger as the control. backdoor_neuron_ablation.json gives attack success after
removing the rank-1 direction, a random direction, the top coordinates or random
coordinates at the peak layer. direction_persistence.json on the CIFAR-100 and
Tiny checkpoints gives each layer's direction's alignment to the final one,
from which the crystallisation layer is read.

    PYTHONPATH=. python scripts/paper/mech_direction.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import glob
import os
import sys

sys.path.insert(0, os.getcwd())

import scripts.paper._style  # noqa: E402,F401  the shared figure style
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper._common import (  # noqa: E402
    OKABE_ITO,
    attack_label,
    build_parser,
    figure_sidecar,
    fmt,
    is_panel_folder,
    load_json,
    mean_or_none,
    save_figure,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/mech_direction.py"
ABLATIONS = (
    ("direction", "rank-1 direction"),
    ("random_dir_0", "random direction"),
    ("top_20", "top-20 coordinates"),
    ("random_20", "random-20 coordinates"),
)
CRYSTALLISED = 0.5


def neuron_records(results_dir: str) -> dict[str, dict]:
    """attack to record, ViT checkpoints outside the SAM set only."""
    records = {}
    for path in sorted(
        glob.glob(os.path.join(results_dir, "vit_*", "backdoor_neurons.json"))
    ):
        folder = os.path.basename(os.path.dirname(path))
        if not is_panel_folder(folder):
            continue
        record = load_json(path)
        records[record["attack"]] = record
    return records


def main() -> None:
    args = build_parser(__doc__).parse_args()
    ablation_path = os.path.join(args.results_dir, "backdoor_neuron_ablation.json")
    persistence_path = os.path.join(args.results_dir, "direction_persistence.json")
    records = neuron_records(args.results_dir)
    ablation = [
        row
        for row in load_json(ablation_path) or []
        if is_panel_folder(row["folder_name"])
    ]
    persistence = load_json(persistence_path) or []
    inputs = [
        f"{args.results_dir}/vit_cifar10_*/backdoor_neurons.json ({len(records)})",
        ablation_path,
        persistence_path,
    ]

    # Figure: 3 depth profiles per attack with the benign reference dashed.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    plotted = {}
    keys = (
        ("rel_direction_norm", "relative backdoor-direction norm"),
        ("tac_mean", "mean trigger-activated change"),
        ("cka", "clean-versus-triggered CKA"),
    )
    for index, (attack, record) in enumerate(sorted(records.items())):
        layers = [row["layer"] for row in record["layers"]]
        plotted[attack] = {
            key: [row[key] for row in record["layers"]] for key, _ in keys
        }
        for axis, (key, label) in zip(axes, keys):
            style = (
                {"color": "0.5", "linestyle": "--"}
                if attack == "benign"
                else {"color": OKABE_ITO[index % len(OKABE_ITO)]}
            )
            axis.plot(
                layers,
                plotted[attack][key],
                marker="o",
                markersize=3,
                linewidth=1.3,
                label=attack_label(attack),
                **style,
            )
            axis.set_xlabel("layer")
            axis.set_ylabel(label)
    axes[0].legend(fontsize=7)
    figure_path = os.path.join(args.paper_dir, "figures", "mech_direction.pdf")
    save_figure(fig, figure_path)
    figure_sidecar(figure_path.replace(".pdf", ".json"), GENERATOR, inputs, plotted)

    # Table: the causal ablation at each checkpoint's peak layer.
    rows = []
    for row in sorted(ablation, key=lambda r: r["attack"]):
        row_cells = [
            attack_label(row["attack"]),
            str(row["peak_layer"]),
            fmt(row["baseline"]["asr"]),
            fmt(row["baseline"]["clean_accuracy"]),
        ]
        for key, _ in ABLATIONS:
            block = row["ablated"].get(key)
            row_cells.append(
                f"{fmt(block['asr'])} / {fmt(block['clean_accuracy'])}"
                if block
                else "--"
            )
        rows.append(row_cells)
    write_table(
        path=os.path.join(args.paper_dir, "tables", "direction_ablation.tex"),
        generator=GENERATOR,
        inputs=[ablation_path],
        caption=(
            "The causal ablation at the peak layer, CIFAR-10 at the highest panel "
            "rate: attack success over clean accuracy after projecting out the rank-1 "
            "backdoor direction, a random direction, the top-20 trigger-activated "
            "coordinates and 20 random coordinates, as a forward hook after the "
            "final LayerNorm."
        ),
        label="tab:direction-ablation",
        header=[
            "attack",
            "peak layer",
            "ASR",
            "CA",
            *[label for _, label in ABLATIONS],
        ],
        rows=rows,
        align="lrrrrrrr",
    )

    # Crystallisation: the first layer whose direction aligns at least half with the final one.
    crystal_rows = []
    crystal: dict[str, list[int]] = {}
    for record in persistence:
        alignment = record.get("alignment_to_final") or []
        layers_aligned = [
            index + 1
            for index, value in enumerate(alignment)
            if value is not None and value >= CRYSTALLISED
        ]
        if not layers_aligned:
            continue
        crystal.setdefault(record["attack"], []).append(min(layers_aligned))
    for attack, layers in sorted(crystal.items()):
        crystal_rows.append(
            [
                attack_label(attack),
                str(len(layers)),
                fmt(mean_or_none(layers), places=1),
                str(min(layers)),
                str(max(layers)),
            ]
        )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "direction_crystallisation.tex"),
        generator=GENERATOR,
        inputs=[persistence_path],
        caption=(
            "Crystallisation depth on CIFAR-100 and Tiny across poison rates: the first "
            "layer whose backdoor direction aligns at cosine at least "
            f"{CRYSTALLISED:g} with the final layer's, per checkpoint, summarised per attack."
        ),
        label="tab:direction-crystallisation",
        header=["attack", "checkpoints", "mean layer", "earliest", "latest"],
        rows=crystal_rows,
        align="lrrrr",
    )

    benign = records.get("benign", {}).get("layers", [])
    direction_rows = [row for row in ablation if row["ablated"].get("direction")]
    macros = {
        "direction_checkpoints": (
            str(len(records)),
            "CIFAR-10 checkpoints with a backdoor_neurons record, benign included",
        ),
        "direction_benign_peak_rel_norm": (
            fmt(
                max((row["rel_direction_norm"] for row in benign), default=None),
                places=2,
            ),
            "peak relative backdoor-direction norm on the benign reference under the same trigger",
        ),
        "direction_backdoored_min_final_rel_norm": (
            fmt(
                min(
                    (
                        record["layers"][-1]["rel_direction_norm"]
                        for attack, record in records.items()
                        if attack != "benign"
                    ),
                    default=None,
                ),
                places=2,
            ),
            "smallest last-layer relative direction norm over the backdoored CIFAR-10 checkpoints",
        ),
        "ablation_checkpoints": (
            str(len(direction_rows)),
            "checkpoints in the causal ablation outside the SAM set",
        ),
        "ablation_direction_max_asr": (
            fmt(
                max(
                    row["ablated"]["direction"]["asr"]
                    for row in direction_rows
                    if row["attack"] != "badnet_a2a"
                ),
                places=2,
            ),
            "highest attack success left after removing the rank-1 direction, all-to-one attacks",
        ),
        "ablation_top_coordinates_min_asr": (
            fmt(
                min(
                    row["ablated"]["top_20"]["asr"]
                    for row in direction_rows
                    if row["attack"] != "badnet_a2a"
                ),
                places=2,
            ),
            "lowest attack success left after zeroing the top-20 coordinates, all-to-one attacks",
        ),
        "ablation_random_direction_min_asr": (
            fmt(
                min(
                    row["ablated"]["random_dir_0"]["asr"]
                    for row in direction_rows
                    if row["attack"] != "badnet_a2a"
                ),
                places=2,
            ),
            "lowest attack success left after removing a random direction, all-to-one attacks",
        ),
        "crystallisation_badnet_layer": (
            fmt(mean_or_none(crystal.get("badnet_a2o", [])), places=1),
            "mean crystallisation layer of badnet_a2o on CIFAR-100 and Tiny",
        ),
        "crystallisation_blend_layer": (
            fmt(mean_or_none(crystal.get("blend", [])), places=1),
            "mean crystallisation layer of blend on CIFAR-100 and Tiny",
        ),
        "crystallisation_checkpoints": (
            str(sum(map(len, crystal.values()))),
            "checkpoints in the crystallisation record",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "direction.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"direction: {len(records)} records, ablation {len(direction_rows)}, crystal { {k: len(v) for k, v in crystal.items()} }"
    )


if __name__ == "__main__":
    main()
