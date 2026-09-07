"""Head-to-head: PSBD as published against the ViT-adapted variant.

Every result in this project so far is a single knob measured in isolation. This
assembles them into two complete, runnable defences and scores them the same way,
which is the comparison a reader actually needs.

**Neither configuration is allowed to see a poison label.** Both pick their dropout
rate by the paper's adaptive rule on clean validation data, never by best AUROC. The
"oracle" numbers reported elsewhere are upper bounds and deliberately do not appear
here.

  psbd_paper      the method as published, ported to ViT
                    placement  post_residual, every block (the ConvNet placement)
                    score      absolute PSU, P_c - mean_k(P_c_dropout)
                    rate       smallest p with clean-validation shift ratio >= 0.8
                    rule       one-sided, flag low PSU
                    threshold  25th percentile of clean-validation score

  psbd_vit        the same method with four changes, each justified by its own
                  hypothesis file
                    placement  pre_residual restricted to blocks 5-8   (H10)
                    score      fractional PSU, 1 - mean_k(P_c_dropout)/P_c  (H12)
                    rate       shift-ratio target 0.7 rather than 0.8   (H11)
                    rule       two-sided, flag whichever tail separates (H5)
                    threshold  unchanged, 25th percentile

Two of psbd_vit's four choices are fitted: the band and the 0.7 target were both
selected after looking at results on the six attacks in the derivation set. So its
numbers on those six are optimistic and are labelled as such. The held-out column is
the one to read.

Example
    python -m cli.variants
    python -m cli.variants --held-out vit_cifar10_sig_0_1 vit_cifar10_lc_0_1
"""

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from psbd.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from psbd.decision import (
    complete_rates,
    pair_clean_to_backdoor,
    threshold_at_quantile,
)
from psbd.scores import psu_from_cache, psu_ratio_from_cache, shift_ratio

SPLITS = ("validation", "clean", "backdoor")
QUANTILE = 0.25

VARIANTS = {
    "psbd_paper": {
        "placement": "post_residual",
        "score": "absolute",
        "shift_target": 0.8,
        "two_sided": False,
    },
    "psbd_vit": {
        "placement": "pre_residual_blocks_5_8",
        "score": "fractional",
        "shift_target": 0.7,
        "two_sided": True,
    },
}

TABLE_WIDTH = 118


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    # H20 puts pre_residual_blocks_5_8 in the losing placement family, so the
    # variant's placement is worth swapping. Exposed rather than edited in place so
    # the published configuration stays the default and the swap is explicit.
    parser.add_argument(
        "--placement",
        default=None,
        help="override psbd_vit's placement. The default keeps the H13 configuration",
    )
    parser.add_argument(
        "--held-out",
        nargs="*",
        default=[],
        help="checkpoints not used to choose any hyperparameter, reported separately",
    )
    return parser.parse_args()


def score_split(psbd_dir: str, placement: str, rate: float, split: str, kind: str):
    """One split's per-sample score and its shift ratio, at one rate."""
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    per_pass, argmax = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, placement, rate, split)
    )
    build = psu_ratio_from_cache if kind == "fractional" else psu_from_cache

    return build(probs, labels, per_pass), shift_ratio(labels, argmax)


def select_rate(psbd_dir: str, placement: str, kind: str, shift_target: float):
    """The smallest rate whose clean-validation shift ratio reaches the target.

    Chosen on clean validation data alone, exactly as the paper's rule prescribes.
    Only the target constant differs between the two variants.
    """
    for rate in complete_rates(psbd_dir, placement):
        _, sigma = score_split(psbd_dir, placement, rate, "validation", kind)
        if sigma is not None and sigma >= shift_target:
            return rate
    return None


def evaluate(psbd_dir: str, config: dict) -> dict | None:
    """Run one complete defence on one checkpoint. Returns None if it cannot run."""
    placement, kind = config["placement"], config["score"]
    if not complete_rates(psbd_dir, placement):
        return None
    manifest = read_split_manifest(psbd_dir)

    chosen = select_rate(psbd_dir, placement, kind, config["shift_target"])
    if chosen is None:
        return {"rate": None, "reason": "no swept rate reaches the shift-ratio target"}

    scores = {}
    for split in SPLITS:
        scores[split], _ = score_split(psbd_dir, placement, chosen, split, kind)
    clean = pair_clean_to_backdoor(scores["clean"], manifest)
    backdoor = scores["backdoor"]

    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    ranking = np.concatenate([-clean.float().numpy(), -backdoor.float().numpy()])
    auroc = float(roc_auc_score(labels, ranking))

    if config["two_sided"] and auroc < 0.5:
        # The separation is real but the tail is reversed, so the decision rule
        # flips with it: flag HIGH score instead of low.
        #
        # The THRESHOLD has to flip too, and forgetting that is not a cosmetic
        # error. The quantile is what sets the tolerated clean loss rate: flagging
        # below the 25th percentile costs 25% of clean data. Flagging ABOVE that
        # same 25th percentile costs 75%, so an unflipped threshold reports a TPR
        # near 0.8 on data with no signal at all. It did exactly that on the benign
        # control (TPR 0.813 at AUROC 0.507) before this was corrected. Using the
        # complementary quantile keeps the clean cost at 25% either way, so the two
        # directions are comparable and the control reads honestly.
        threshold = threshold_at_quantile(scores["validation"], 1.0 - QUANTILE)
        tpr = float((backdoor > threshold).float().mean())
        fpr = float((clean > threshold).float().mean())
        auroc = 1.0 - auroc
        direction = "inverted"
    else:
        threshold = threshold_at_quantile(scores["validation"], QUANTILE)
        tpr = float((backdoor < threshold).float().mean())
        fpr = float((clean < threshold).float().mean())
        direction = "as_expected"

    result = {
        "rate": chosen,
        "auroc": auroc,
        "tpr": tpr,
        "fpr": fpr,
        "direction": direction,
    }
    return result


def collect_rows(results_dir: str, checkpoints_dir: str) -> list[tuple]:
    """Both variants evaluated on every checkpoint with a stage-1 cache.

    SAM checkpoints are skipped: SAM is a training-time change under a separate
    question, and carrying it here would double every row of this comparison.
    """
    folders = sorted(
        os.path.basename(os.path.dirname(path))
        for path in glob.glob(os.path.join(results_dir, "*", "psbd"))
    )

    rows = []
    for folder in folders:
        if "sam_rho" in folder:
            continue
        meta_path = os.path.join(checkpoints_dir, folder, "metrics.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        result = {name: evaluate(psbd_dir, cfg) for name, cfg in VARIANTS.items()}
        if not any(result.values()):
            continue
        rows.append((folder, meta, result))

    return rows


def variant_cells(result: dict | None, with_direction: bool) -> str:
    """One variant's 4 numbers, or placeholders when it could not run."""
    if result and result.get("auroc") is not None:
        cells = (
            f" {result['rate']:>5g} {result['auroc']:>7.3f} "
            f"{result['tpr']:>6.3f} {result['fpr']:>6.3f}"
        )
        if with_direction:
            cells += f" {result['direction'][:9]:>9}"
        return cells

    cells = f" {'--':>5} {'--':>7} {'--':>6} {'--':>6}"
    if with_direction:
        cells += f" {'--':>9}"
    return cells


def show(title: str, subset: list[tuple]) -> None:
    """One block of the comparison, plus the mean delta over its backdoored rows."""
    if not subset:
        return

    print(f"\n## {title}\n")
    print(
        f"{'checkpoint':32} {'ASR':>5} | {'p':>5} {'AUROC':>7} {'TPR':>6} {'FPR':>6} "
        f"| {'p':>5} {'AUROC':>7} {'TPR':>6} {'FPR':>6} {'dir':>9} {'dAUROC':>7}"
    )
    print("-" * TABLE_WIDTH)

    deltas = []
    for folder, meta, result in subset:
        paper, adapted = result.get("psbd_paper"), result.get("psbd_vit")
        asr = meta.get("asr")
        asr_cell = "--" if asr is None else f"{asr:.2f}"

        line = f"{folder:32} {asr_cell:>5} |"
        line += variant_cells(paper, with_direction=False) + " |"
        line += variant_cells(adapted, with_direction=True)
        if (
            paper
            and adapted
            and paper.get("auroc") is not None
            and adapted.get("auroc") is not None
        ):
            delta = adapted["auroc"] - paper["auroc"]
            deltas.append((folder, delta))
            line += f" {delta:>+7.3f}"
        print(line)

    backdoored = [delta for folder, delta in deltas if "benign" not in folder]
    if backdoored:
        wins = sum(1 for delta in backdoored if delta > 0)
        print(
            f"\n  mean delta over {len(backdoored)} backdoored: "
            f"{sum(backdoored) / len(backdoored):+.3f}"
            f"   wins {wins}/{len(backdoored)}"
        )


def main() -> None:
    args = parse_args()
    held_out = set(args.held_out)
    if args.placement:
        VARIANTS["psbd_vit"] = {**VARIANTS["psbd_vit"], "placement": args.placement}
        print(f"psbd_vit placement overridden to {args.placement}\n")

    rows = collect_rows(args.results_dir, args.checkpoints_dir)

    print("PSBD as published, against the ViT-adapted variant.")
    print("Both choose their dropout rate on clean validation data only. No oracle.")
    show(
        "Derivation set (two of psbd_vit's four choices were fitted here, so these "
        "numbers are optimistic)",
        [row for row in rows if row[0] not in held_out and "benign" not in row[0]],
    )
    show(
        "HELD OUT (nothing was fitted on these)",
        [row for row in rows if row[0] in held_out],
    )
    show(
        "Negative control (both must sit near 0.5)",
        [row for row in rows if "benign" in row[0]],
    )


if __name__ == "__main__":
    main()
