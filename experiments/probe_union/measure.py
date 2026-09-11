"""Does the H41 probe union help on ordinary, non-adaptive backdoored ViT-B/16 models?

H41 (`docs/hypothesis/H41-multi-probe-defence.md`) built the min-rank probe
union to defeat an attacker trained against a single probed operator. Nobody
trains against a probe on the 65 clearing cells the paper's headline reads, so
this script asks the separate question: does the union still help, hurt or do
nothing there. It also asks whether the union specifically rescues the 2 cells
the headline names as inverted (`docs/hypothesis/README.md`, cifar10 wanet and
cifar10 sig at 10%).

The union rule (H41's mechanism section) is unchanged. For probe j, rank_j(x)
is the percentile of x's fractional PSU within probe j's own clean-validation
distribution. The combined score is min_j rank_j(x). The calibrated threshold
is the target-FPR quantile of that combined score on clean validation.
`defences.decision.multi_probe_auroc` and `multi_probe_detection` compute both.
Nothing here reimplements them.

6 probe sets, each read on whichever of the 65 models hold every one of its
placements at a rate the 0.8 adaptive rule reached
(`results/<folder>/psbd_metrics.json`'s `placements[<id>]["adaptive_rate"]`):

    psbd_tm             before_attention_norm_token_mask alone, the reference
    psbd_tm_rd          + post_residual (PSBD-RD, the published placement)
    psbd_tm_attn_branch + before_attention_residual_token_mask (attention branch output)
    adaptive_3probe     + before_attention_norm dropout, + mlp_norm_out_gain_scale
                          (H41's adaptive-attacker pool minus PSBD-RD and gaussian)
    adaptive_4probe     adaptive_3probe + post_residual
    all_65_basis        every basis placement present on all 65 models

Every per-sample PSU comes from the stage-1 cache under
results/<folder>/psbd/<placement>/, read exactly as cli.analyze and
scripts/paper/fig_psu_histograms.py read it: `defences.cache` for the raw
tensors, `defences.scores.psu_ratio_from_cache` for the fractional PSU (the
canon headline statistic), `defences.decision.pair_clean_to_backdoor` to
restrict the clean split to the backdoor split's eligible images.

    PYTHONPATH=. .venv/bin/python experiments/probe_union/measure.py
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from data.splits import SPLITS  # noqa: E402
from defences.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import (  # noqa: E402
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    multi_probe_auroc,
    multi_probe_detection,
    pair_clean_to_backdoor,
)
from defences.scores import psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    bootstrap_ci,
    clearing_cells,
    load_coverage,
    load_psbd_metrics,
    mean_or_none,
)

PSBD_TM = RECOMMENDED_PLACEMENT  # before_attention_norm_token_mask
PSBD_RD = PUBLISHED_PLACEMENT  # post_residual
ATTN_BRANCH_TM = "before_attention_residual_token_mask"
ATTN_INPUT_DROPOUT = "before_attention_norm"
MLP_NORM_GAIN = "mlp_norm_out_gain_scale"

PROBE_SETS = {
    "psbd_tm": (PSBD_TM,),
    "psbd_tm_rd": (PSBD_TM, PSBD_RD),
    "psbd_tm_attn_branch": (PSBD_TM, ATTN_BRANCH_TM),
    "adaptive_3probe": (PSBD_TM, ATTN_INPUT_DROPOUT, MLP_NORM_GAIN),
    "adaptive_4probe": (PSBD_TM, ATTN_INPUT_DROPOUT, MLP_NORM_GAIN, PSBD_RD),
    # all_65_basis is filled in by basis_ids_present_on_all_models, since which
    # placements clear that bar is a fact about the cache on disk, not a
    # decision this script should hardcode and let drift from it.
    "all_65_basis": None,
}

# The pairing every gain in the table is read against.
REFERENCE_SET = "psbd_tm"
TARGET_FPRS = (0.10, 0.20)
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 0

# Attacks and the 1 rate the CLAUDE.md headline names for the rescue question.
WANET_CIFAR10_FOLDER = "vit_cifar10_wanet_0_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--declaration", default="configs/psbd_basis.json")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--output", default=experiment_result_path("probe_union", "probe_union.json")
    )
    return parser.parse_args()


def select_models(results_dir: str) -> list[dict]:
    """The 65 clearing cells that carry both headline placements, tab_headline's selection.

    A cell qualifies when it cleared the ASR bar (`clearing_cells`) and its
    psbd_metrics.json reached the adaptive rate for both PSBD-TM and PSBD-RD,
    the same 2 configurations `scripts/paper/tab_headline.py` requires of every
    row it reports.
    """
    coverage = load_coverage(results_dir)
    cells = clearing_cells(coverage)

    selected = []
    for cell in cells:
        report = load_psbd_metrics(results_dir, cell["folder_name"])
        if report is None:
            continue
        placements = report.get("placements", {})
        tm_block = placements.get(PSBD_TM)
        rd_block = placements.get(PSBD_RD)
        if not tm_block or tm_block.get("adaptive_rate") is None:
            continue
        if not rd_block or rd_block.get("adaptive_rate") is None:
            continue
        cell["report"] = report
        selected.append(cell)
    return selected


def basis_ids_present_on_all_models(
    models: list[dict], declaration_path: str
) -> list[str]:
    """Every basis placement id whose adaptive_rate is set on every 1 of `models`."""
    with open(declaration_path) as handle:
        basis_ids = [entry["id"] for entry in json.load(handle)["basis"]]

    present = []
    for placement in basis_ids:
        holds_everywhere = all(
            (model["report"]["placements"].get(placement) or {}).get("adaptive_rate")
            is not None
            for model in models
        )
        if holds_everywhere:
            present.append(placement)
    return present


def model_has_probes(model: dict, placements: tuple[str, ...]) -> bool:
    """Whether every probe in this set reached an adaptive rate on this model."""
    blocks = model["report"]["placements"]
    return all(
        (blocks.get(p) or {}).get("adaptive_rate") is not None for p in placements
    )


def load_probe_psu(psbd_dir: str, placement: str, rate: float, manifest: dict) -> tuple:
    """1 probe's fractional PSU on (validation, clean paired to backdoor, backdoor).

    validation is (n_validation,), clean_paired and backdoor are both
    (n_backdoor,). Fractional PSU (`psu_ratio_from_cache`) is the canon headline
    statistic (CLAUDE.md), read exactly as cli.analyze reads it for every rate row
    of `detection_psu_ratio`.
    """
    psu = {}
    for split in SPLITS:
        baseline_probs, baseline_labels, _ = load_baseline(
            baseline_path(psbd_dir, split)
        )
        per_pass_probs, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        psu[split] = psu_ratio_from_cache(
            baseline_probs, baseline_labels, per_pass_probs
        )  # (n_split,)

    clean_paired = pair_clean_to_backdoor(psu["clean"], manifest)  # (n_backdoor,)
    return psu["validation"], clean_paired, psu["backdoor"]


def measure_model(results_dir: str, model: dict, placements: tuple[str, ...]) -> dict:
    """AUROC and TPR at both target FPRs of the min-rank union of `placements`, on 1 model."""
    psbd_dir = os.path.join(results_dir, model["folder_name"], "psbd")
    manifest = read_split_manifest(psbd_dir)
    blocks = model["report"]["placements"]

    val_per_probe, clean_per_probe, backdoor_per_probe = [], [], []
    for placement in placements:
        rate = blocks[placement]["adaptive_rate"]
        val, clean, backdoor = load_probe_psu(psbd_dir, placement, rate, manifest)
        val_per_probe.append(val)
        clean_per_probe.append(clean)
        backdoor_per_probe.append(backdoor)

    auroc = multi_probe_auroc(clean_per_probe, backdoor_per_probe, val_per_probe)
    row = {
        "folder": model["folder_name"],
        "dataset": model["report"]["dataset"],
        "attack": model["report"]["attack"],
        "poison_rate": model["report"]["poison_rate"],
        "auroc": auroc,
    }
    for target_fpr in TARGET_FPRS:
        detection = multi_probe_detection(
            val_per_probe, clean_per_probe, backdoor_per_probe, target_fpr, "calibrated"
        )
        row[f"tpr_at_{target_fpr:.2f}"] = detection["tpr"]
        row[f"fpr_at_{target_fpr:.2f}"] = detection["fpr"]
    return row


def measure_set(
    results_dir: str, models: list[dict], placements: tuple[str, ...]
) -> list[dict]:
    """1 probe set's per-model rows, over the subset of `models` that hold every probe."""
    rows = [
        measure_model(results_dir, model, placements)
        for model in models
        if model_has_probes(model, placements)
    ]
    return rows


def set_summary(rows: list[dict], placements: tuple[str, ...]) -> dict:
    """A probe set's aggregate: how many models, its placements, mean AUROC and TPR."""
    summary = {
        "placements": list(placements),
        "n_models": len(rows),
        "auroc_mean": mean_or_none([row["auroc"] for row in rows]),
    }
    for target_fpr in TARGET_FPRS:
        key = f"tpr_at_{target_fpr:.2f}"
        summary[f"{key}_mean"] = mean_or_none([row[key] for row in rows])
    return summary


def paired_gain(
    reference_rows: list[dict], set_rows: list[dict], resamples: int, seed: int
) -> dict:
    """The paired AUROC gain of a set over the reference, on the models both cover.

    Paired within model (matched by folder name), never averaged separately and
    subtracted, so the gain is measured on exactly the population the smaller
    set restricts it to.
    """
    reference_by_folder = {row["folder"]: row["auroc"] for row in reference_rows}
    deltas = [
        row["auroc"] - reference_by_folder[row["folder"]]
        for row in set_rows
        if row["folder"] in reference_by_folder
    ]
    low, high = bootstrap_ci(deltas, resamples, seed)
    gain = {
        "n": len(deltas),
        "mean_gain": mean_or_none(deltas),
        "ci_low": low,
        "ci_high": high,
    }
    return gain


def per_attack_means(rows: list[dict]) -> dict:
    """Mean AUROC per attack token, over whichever models a set's rows cover."""
    by_attack: dict[str, list[float]] = {}
    for row in rows:
        by_attack.setdefault(row["attack"], []).append(row["auroc"])
    means = {attack: mean_or_none(values) for attack, values in by_attack.items()}
    return means


def wanet_cifar10_cell(rows: list[dict]) -> dict | None:
    """The 1 row for vit_cifar10_wanet_0_1, the cell CLAUDE.md's headline calls inverted."""
    for row in rows:
        if row["folder"] == WANET_CIFAR10_FOLDER:
            return row
    return None


def build_report(args: argparse.Namespace) -> dict:
    models = select_models(args.results_dir)
    all_65_ids = basis_ids_present_on_all_models(models, args.declaration)
    probe_sets = dict(PROBE_SETS)
    probe_sets["all_65_basis"] = tuple(all_65_ids)

    sets_out = {}
    for name, placements in probe_sets.items():
        rows = measure_set(args.results_dir, models, placements)
        sets_out[name] = {
            "summary": set_summary(rows, placements),
            "per_model": rows,
            "per_attack_mean_auroc": per_attack_means(rows),
        }

    reference_rows = sets_out[REFERENCE_SET]["per_model"]
    for name, block in sets_out.items():
        if name == REFERENCE_SET:
            continue
        block["gain_over_psbd_tm"] = paired_gain(
            reference_rows, block["per_model"], args.bootstrap, args.seed
        )

    report = {
        "n_models_selected": len(models),
        "reference_set": REFERENCE_SET,
        "target_fprs": list(TARGET_FPRS),
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "probe_sets": sets_out,
        "wanet_cifar10": {
            name: wanet_cifar10_cell(block["per_model"])
            for name, block in sets_out.items()
        },
    }
    return report


def print_summary(report: dict) -> None:
    print(f"{report['n_models_selected']} models selected")
    for name, block in report["probe_sets"].items():
        summary = block["summary"]
        print(
            f"{name}: n={summary['n_models']} placements={summary['placements']} "
            f"auroc={summary['auroc_mean']:.3f}"
        )
    print("wanet on cifar10 at 10%:")
    for name, row in report["wanet_cifar10"].items():
        if row is not None:
            print(f"  {name}: auroc={row['auroc']:.3f}")
        else:
            print(f"  {name}: not covered")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    print_summary(report)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
