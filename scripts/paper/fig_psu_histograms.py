"""Histograms of the fractional PSU statistic, where PSBD-TM separates and where it does not.

3 cells where before_attention_norm_token_mask (PSBD-TM, RECOMMENDED_PLACEMENT)
clears cleanly sit beside 3 cells where a backdoor implanted but the detector
reads near chance, so the same figure carries both regimes rather than only the
flattering regime. Every panel reads the cache cli.sweep wrote under
results/<folder>/psbd/ at the rate cli.analyze's select_rate_adaptively chose
for this placement (psbd_metrics.json's adaptive_rate, target
ADAPTIVE_SHIFT_TARGET 0.8), and the fractional PSU per sample comes from
defences.scores.psu_ratio_from_cache, the threshold and AUROC from
defences.decision.threshold_at_quantile and detection_report at
HEADLINE_QUANTILE. Nothing here recomputes either statistic.

    PYTHONPATH=. .venv/bin/python scripts/paper/fig_psu_histograms.py \\
        --results-dir results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import scripts.paper._style as style  # noqa: E402  the shared figure style
import matplotlib.pyplot as plt  # noqa: E402

from data.splits import SPLITS  # noqa: E402
from defences.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import (  # noqa: E402
    HEADLINE_QUANTILE,
    RECOMMENDED_PLACEMENT,
    detection_report,
    pair_clean_to_backdoor,
    threshold_at_quantile,
)
from defences.scores import psu_ratio_from_cache  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    attack_label,
    build_parser,
    dataset_label,
    figure_sidecar,
    load_psbd_metrics,
)

GENERATOR = "scripts/paper/fig_psu_histograms.py"
NUM_BINS = 40
CLEAN_COLOUR = style.PALETTE[0]
BACKDOOR_COLOUR = style.PALETTE[1]
THRESHOLD_COLOUR = "black"

# (folder, row) with row 0 the 3 cells PSBD-TM clears and row 1 the 3 cells
# where a backdoor implanted but the detector reads near chance (docs/hypothesis
# and CLAUDE.md's headline: cifar10 wanet and cifar10 sig at 10% both invert).
PANELS = (
    ("vit_cifar100_badnet_a2o_0_01", 0),
    ("vit_gtsrb_bpp_0_05", 0),
    ("vit_tiny_blend_0_1", 0),
    ("vit_cifar10_wanet_0_1", 1),
    ("vit_cifar10_sig_0_1", 1),
    ("vit_tiny_tact_0_01", 1),
)


def panel_data(results_dir: str, folder: str) -> dict:
    """1 panel's per-sample PSU ratio, its threshold and its AUROC, all read from the cache.

    The adaptive rate comes straight from psbd_metrics.json's adaptive_rate field
    (cli.analyze's select_rate_adaptively output), never recomputed here, since
    that field is already the placement's canonical rate choice.
    """
    report = load_psbd_metrics(results_dir, folder)
    if report is None:
        raise SystemExit(f"{folder} has no psbd_metrics.json, run cli.analyze first")

    block = report["placements"][RECOMMENDED_PLACEMENT]
    rate = block["adaptive_rate"]
    if rate is None:
        raise SystemExit(
            f"{folder} never reaches the adaptive shift target at {RECOMMENDED_PLACEMENT}"
        )

    psbd_dir = os.path.join(results_dir, folder, "psbd")
    manifest = read_split_manifest(psbd_dir)

    psu_ratio_by_split = {}
    for split in SPLITS:
        baseline_probs, baseline_labels, _ = load_baseline(
            baseline_path(psbd_dir, split)
        )
        per_pass_probs, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, RECOMMENDED_PLACEMENT, rate, split)
        )
        psu_ratio_by_split[split] = psu_ratio_from_cache(
            baseline_probs, baseline_labels, per_pass_probs
        )  # (n_split,)

    # The clean split covers the whole analysis pool while the backdoor split
    # covers only the attack's eligible subset, so the clean side is paired down
    # to the same images before the 2 are compared or plotted side by side.
    clean_psu = pair_clean_to_backdoor(
        psu_ratio_by_split["clean"], manifest
    )  # (n_backdoor,)
    backdoor_psu = psu_ratio_by_split["backdoor"]  # (n_backdoor,)
    validation_psu = psu_ratio_by_split["validation"]  # (n_validation,)

    threshold = threshold_at_quantile(validation_psu, HEADLINE_QUANTILE)
    report_at_headline = detection_report(
        validation_psu, clean_psu, backdoor_psu, HEADLINE_QUANTILE
    )

    data = {
        "folder": folder,
        "dataset": report["dataset"],
        "attack": report["attack"],
        "poison_rate": report["poison_rate"],
        "rate": rate,
        "clean_psu": clean_psu.numpy(),
        "backdoor_psu": backdoor_psu.numpy(),
        "threshold": threshold,
        "auroc": report_at_headline["auroc"],
    }
    return data


def panel_title(data: dict) -> str:
    """Dataset, attack, poison rate and AUROC, the 4 things every panel must show."""
    rate_percent = f"{data['poison_rate'] * 100:g}%"
    title = (
        f"{dataset_label(data['dataset'])}, {attack_label(data['attack'])} "
        f"{rate_percent} (AUROC {data['auroc']:.3f})"
    )
    return title


def draw_panel(ax, data: dict) -> None:
    """1 histogram: clean and triggered PSU ratio, both densities, the threshold marked."""
    combined = np.concatenate([data["clean_psu"], data["backdoor_psu"]])
    bin_edges = np.histogram_bin_edges(combined, bins=NUM_BINS)

    ax.hist(
        data["clean_psu"],
        bins=bin_edges,
        density=True,
        color=CLEAN_COLOUR,
        alpha=0.6,
        label="clean",
    )
    ax.hist(
        data["backdoor_psu"],
        bins=bin_edges,
        density=True,
        color=BACKDOOR_COLOUR,
        alpha=0.6,
        label="triggered",
    )
    ax.axvline(data["threshold"], color=THRESHOLD_COLOUR, linestyle="--", linewidth=1.0)
    ax.set_title(panel_title(data), fontsize=8)


def write_figure(paper_dir: str, panels: list[dict]) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for data, ax in zip(panels, axes.flat):
        draw_panel(ax, data)

    for ax in axes[1, :]:
        ax.set_xlabel("prediction shift uncertainty (PSU)")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = os.path.join(paper_dir, "figures", "fig_psu_histograms.pdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    return path


def sidecar_plotted(panels: list[dict]) -> dict:
    """Every panel's numbers: folder, rate, threshold, AUROC and the histogram itself."""
    plotted = {}
    for data in panels:
        combined = np.concatenate([data["clean_psu"], data["backdoor_psu"]])
        bin_edges = np.histogram_bin_edges(combined, bins=NUM_BINS)
        clean_counts, _ = np.histogram(data["clean_psu"], bins=bin_edges, density=True)
        backdoor_counts, _ = np.histogram(
            data["backdoor_psu"], bins=bin_edges, density=True
        )
        plotted[data["folder"]] = {
            "dataset": data["dataset"],
            "attack": data["attack"],
            "poison_rate": data["poison_rate"],
            "adaptive_rate": data["rate"],
            "threshold": data["threshold"],
            "auroc": data["auroc"],
            "n_clean": int(data["clean_psu"].size),
            "n_backdoor": int(data["backdoor_psu"].size),
            "bin_edges": bin_edges.tolist(),
            "clean_density": clean_counts.tolist(),
            "backdoor_density": backdoor_counts.tolist(),
        }
    return plotted


def main() -> None:
    args = build_parser(__doc__).parse_args()

    panels = [panel_data(args.results_dir, folder) for folder, _row in PANELS]
    figure_path = write_figure(args.paper_dir, panels)

    figure_sidecar(
        path=figure_path.replace(".pdf", ".json"),
        generator=GENERATOR,
        inputs=[
            f"{args.results_dir}/<folder>/psbd/ (raw baseline and per-pass caches)",
            f"{args.results_dir}/<folder>/psbd_metrics.json (adaptive_rate)",
        ],
        plotted=sidecar_plotted(panels),
    )

    for data in panels:
        print(
            f"{data['folder']}: rate={data['rate']} threshold={data['threshold']:.4f} "
            f"auroc={data['auroc']:.4f}"
        )
    print(f"wrote {figure_path}, {figure_path.replace('.pdf', '.png')} and its sidecar")


if __name__ == "__main__":
    main()
