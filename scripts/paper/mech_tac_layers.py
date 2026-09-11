"""Per-layer TAC and direction norm on GTSRB, CIFAR-100 and Tiny, from the GPU pass record.

Reads results/_experiments/tac_layers/tac_layers.json, written by
run_tac_layers.py, and draws the relative backdoor-direction norm and the mean
trigger-activated change against depth, 1 panel per dataset, the benign
reference dashed. The onset layer is the first at which the relative direction
norm reaches half its final value, the number the placement account keys on.
When the record is absent the table prints pending and no macro is written, so
a build never fails on a pass that has not run.

    PYTHONPATH=. python scripts/paper/mech_tac_layers.py --paper-dir paper
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
    figure_sidecar,
    fmt,
    load_json,
    mean_or_none,
    save_figure,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/mech_tac_layers.py"
RECORD = os.path.join("results", "_experiments", "tac_layers", "tac_layers.json")
DATASET_ORDER = ("gtsrb", "cifar100", "tiny")
HALF = 0.5


def onset_layer(layers: list[dict]) -> int | None:
    """The first layer whose relative direction norm reaches half the final layer's."""
    final = layers[-1]["rel_direction_norm"]
    for row in layers:
        if row["layer"] > 0 and row["rel_direction_norm"] >= HALF * final:
            return row["layer"]
    return None


def write_pending(paper_dir: str) -> None:
    write_table(
        path=os.path.join(paper_dir, "tables", "tac_layers.tex"),
        generator=GENERATOR,
        inputs=[RECORD],
        caption="Per-layer TAC pass pending: run scripts/paper/run\\_tac\\_layers.py on the login-node GPU.",
        label="tab:tac-layers",
        header=["dataset", "attack", "status"],
        rows=[["--", "--", r"\pending"]],
        align="lll",
    )


def main() -> None:
    args = build_parser(__doc__).parse_args()
    record = load_json(RECORD)
    if record is None:
        write_pending(args.paper_dir)
        print("tac layers: record absent, pending table written")
        return
    records = record["records"]
    inputs = [RECORD]

    fig, axes = plt.subplots(2, len(DATASET_ORDER), figsize=(11, 6), sharex=True)
    plotted = {}
    order = ["benign"] + list(HARD_ATTACKS) + list(EASY_ATTACKS)
    for column, dataset in enumerate(DATASET_ORDER):
        rows = sorted((r for r in records if r["dataset"] == dataset), key=lambda r: (order.index(r["attack"]) if r["attack"] in order else 99, r["folder"]))
        for index, item in enumerate(rows):
            layers = [row["layer"] for row in item["layers"]]
            norm = [row["rel_direction_norm"] for row in item["layers"]]
            tac = [row["tac_mean"] for row in item["layers"]]
            plotted[item["folder"]] = {"layers": layers, "rel_direction_norm": norm, "tac_mean": tac, "cka": [row["cka"] for row in item["layers"]]}
            label = item["folder"].replace(f"vit_{dataset}_", "")
            style = {"color": "0.5", "linestyle": "--"} if item["attack"] == "benign" else {"color": OKABE_ITO[index % len(OKABE_ITO)]}
            axes[0, column].plot(layers, norm, marker="o", markersize=3, linewidth=1.2, label=label, **style)
            axes[1, column].plot(layers, tac, marker="o", markersize=3, linewidth=1.2, label=label, **style)
        axes[0, column].set_title(dataset)
        axes[1, column].set_xlabel("layer")
        axes[0, column].legend(fontsize=6)
    axes[0, 0].set_ylabel("relative backdoor-direction norm")
    axes[1, 0].set_ylabel("mean trigger-activated change")
    figure_path = os.path.join(args.paper_dir, "figures", "mech_tac_layers.pdf")
    save_figure(fig, figure_path)
    figure_sidecar(figure_path.replace(".pdf", ".json"), GENERATOR, inputs, plotted)

    table_rows = []
    onsets: dict[str, list[int]] = {}
    benign_peaks = []
    for item in sorted(records, key=lambda r: (DATASET_ORDER.index(r["dataset"]) if r["dataset"] in DATASET_ORDER else 9, r["folder"])):
        peak = max(item["layers"], key=lambda row: row["rel_direction_norm"])
        onset = onset_layer(item["layers"])
        final = item["layers"][-1]
        if item["attack"] == "benign":
            benign_peaks.append(peak["rel_direction_norm"])
        else:
            onsets.setdefault(item["attack"], []).append(onset)
        table_rows.append(
            [
                item["dataset"],
                item["folder"].replace(f"vit_{item['dataset']}_", ""),
                str(peak["layer"]),
                fmt(peak["rel_direction_norm"], places=2),
                str(onset) if onset is not None else "--",
                fmt(final["cka"], places=2),
                fmt(final["tac_mean"], places=2),
            ]
        )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "tac_layers.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            f"Per-layer trigger-activated change on {len(records)} checkpoints, "
            f"{record['samples']} paired images each, class-token features in single "
            "precision: the layer where the relative backdoor-direction norm peaks and "
            "its value, the onset layer at which it first reaches half its final value, "
            "and the last layer's clean-versus-triggered CKA and mean TAC. The benign "
            "reference of each dataset is probed with the panel's patch trigger."
        ),
        label="tab:tac-layers",
        header=["dataset", "checkpoint", "peak layer", "peak rel. norm", "onset layer", "final CKA", "final TAC"],
        rows=table_rows,
        align="llrrrrr",
    )

    macros = {
        "tac_layers_checkpoints": (str(len(records)), "checkpoints in the per-layer TAC pass"),
        "tac_layers_samples": (str(record["samples"]), "paired images per checkpoint in the per-layer TAC pass"),
        "tac_layers_benign_peak_max": (fmt(max(benign_peaks), places=2) if benign_peaks else "--", "largest peak relative direction norm over the benign references"),
        "tac_layers_badnet_onset": (fmt(mean_or_none([o for o in onsets.get("badnet_a2o", []) if o is not None]), places=1), "mean onset layer of badnet_a2o in the per-layer TAC pass"),
        "tac_layers_blend_onset": (fmt(mean_or_none([o for o in onsets.get("blend", []) if o is not None]), places=1), "mean onset layer of blend in the per-layer TAC pass"),
        "tac_layers_earliest_onset_attack": (
            min(((mean_or_none([o for o in v if o is not None]) or 99), k) for k, v in onsets.items())[1] if onsets else "--",
            "the attack with the earliest mean onset layer in the per-layer TAC pass",
        ),
        "tac_layers_latest_onset_attack": (
            max(((mean_or_none([o for o in v if o is not None]) or 0), k) for k, v in onsets.items())[1] if onsets else "--",
            "the attack with the latest mean onset layer in the per-layer TAC pass",
        ),
    }
    write_macros(os.path.join(args.paper_dir, "tables", "tac_layers.macros.json"), GENERATOR, inputs, macros)
    print("tac layers: " + ", ".join(f"{k}: onset {v}" for k, v in onsets.items()) + f", benign peaks {benign_peaks}")


if __name__ == "__main__":
    main()
