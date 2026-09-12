"""Builds the defender_floor_smoke README table from the psbd_metrics.json files.

Run after pbs/defender_floor_smoke/smoke.pbs has finished:
    python -m experiments.defender_floor_smoke.read_results

Reads AUROC and TPR from detection_psu_ratio at the rate row named by
adaptive_rate (never the "adaptive" block, per defences.decision's canon)
for both headline placements and ASR / clean accuracy from the checkpoint's
own args.json. Prints a markdown table to stdout, meant to be pasted into
experiments/defender_floor_smoke/README.md by hand next to the verdict.
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# (base model, variant, checkpoint folder), in table order.
MODELS = ("bpp_0_05", "tact_0_05", "bpp_0_01", "tact_0_01")
VARIANTS = ("control", "calibrated", "rate0_2")
VARIANT_LABELS = {
    "control": "plain",
    "calibrated": "psu_floor calibrated (0.8)",
    "rate0_2": "psu_floor fixed rate 0.2",
}
PLACEMENTS = {
    "PSBD-TM": "before_attention_norm_token_mask",
    "PSBD-RD": "post_residual",
}
CLEAN_QUANTILE = "q0.10"


def read_args_json(folder):
    with open(os.path.join(CHECKPOINTS_DIR, folder, "args.json")) as handle:
        return json.load(handle)


def read_psbd_metrics(folder):
    with open(os.path.join(RESULTS_DIR, folder, "psbd_metrics.json")) as handle:
        return json.load(handle)


def rate_row_at_adaptive_rate(placement):
    """The 1 entry of placement["rates"] whose "rate" matches placement["adaptive_rate"]."""
    adaptive_rate = placement["adaptive_rate"]
    for row in placement["rates"]:
        if row["rate"] == adaptive_rate:
            return adaptive_rate, row
    raise ValueError(f"no rate row matches adaptive_rate {adaptive_rate}")


def placement_cell(metrics, placement_key):
    """A (rate, auroc, tpr@10, tpr@20) tuple for 1 placement, all None if never swept."""
    placement = metrics["placements"].get(placement_key)
    if placement is None:
        return None, None, None, None
    rate, row = rate_row_at_adaptive_rate(placement)
    detection = row["detection_psu_ratio"]
    auroc = detection[CLEAN_QUANTILE]["auroc"]
    tpr_10 = detection["q0.10"]["tpr"]
    tpr_20 = detection["q0.20"]["tpr"]
    return rate, auroc, tpr_10, tpr_20


def format_cell(rate, auroc, tpr_10, tpr_20):
    if rate is None:
        return "n/a", "n/a", "n/a", "n/a"
    return f"{rate:.3f}", f"{auroc:.3f}", f"{tpr_10:.3f}", f"{tpr_20:.3f}"


def build_table():
    header = (
        "| model | variant | ASR | clean acc | PSBD-TM rate | PSBD-TM AUROC | "
        "PSBD-TM TPR@10 | PSBD-TM TPR@20 | PSBD-RD rate | PSBD-RD AUROC | "
        "PSBD-RD TPR@10 | PSBD-RD TPR@20 |"
    )
    separator = "|" + "---|" * 12
    rows = [header, separator]

    for model in MODELS:
        for variant in VARIANTS:
            folder = f"vit_cifar100_{model}_floor_smoke_{variant}"
            args = read_args_json(folder)
            metrics = read_psbd_metrics(folder)

            tm_rate, tm_auroc, tm_tpr10, tm_tpr20 = placement_cell(
                metrics, PLACEMENTS["PSBD-TM"]
            )
            rd_rate, rd_auroc, rd_tpr10, rd_tpr20 = placement_cell(
                metrics, PLACEMENTS["PSBD-RD"]
            )
            tm_cells = format_cell(tm_rate, tm_auroc, tm_tpr10, tm_tpr20)
            rd_cells = format_cell(rd_rate, rd_auroc, rd_tpr10, rd_tpr20)

            row = (
                f"| {model} | {VARIANT_LABELS[variant]} | {args['asr']:.3f} | "
                f"{args['clean_accuracy']:.3f} | "
                + " | ".join(tm_cells)
                + " | "
                + " | ".join(rd_cells)
                + " |"
            )
            rows.append(row)

    return "\n".join(rows)


if __name__ == "__main__":
    print(build_table())
