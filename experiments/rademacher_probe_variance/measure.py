"""Does the minimum variance trace probe actually reduce variance, and does it help?

docs/theory-perturbation-consistency.md, prediction 5. PSU under an isotropic
perturbation is Hutchinson's estimator of trace(H), and Hutchinson's variance is

    gaussian     2 * frobenius(H)^2
    rademacher   2 * (frobenius(H)^2 - sum of squared diagonal entries)

so a Rademacher probe estimates the same expectation with variance smaller by
exactly the diagonal energy of H. Both operators are zero mean with the same
per entry scale, so at a matched rate they differ only as estimators.

Two things are measured, because they can disagree and the distinction matters.

1. The estimator claim, directly. Run many independent single pass estimates of
   the same quantity and compare their spread. This tests the theory itself, and
   its answer also reports how diagonally dominant H is, which nothing else here
   can observe.
2. The downstream claim. Compare detection AUROC at several pass counts. Lower
   estimator variance should help most at small k and vanish as k grows.

A tie on 1 means the Hessian of the predicted class probability is not
diagonally dominant, which is a finding about the network. A loss on 1 would
refute the trace estimator reading.

Run:
    python experiments/rademacher_probe_variance/measure.py
"""

import argparse
import os

import pandas as pd
import torch

from defences.decision import HEADLINE_QUANTILE, detection_report
from defences.inference import build_baseline_cache, compute_dropout_pass_probs
from models.backbones import detect_architecture, load_checkpoint
from defences.operators import build_perturbation
from models.positions import plug_dropout, unplug_dropout
from defences.scores import psu_from_cache, shift_ratio
from data.splits import build_psbd_loaders_from_checkpoint

COMPARED_OPERATORS = ("gaussian", "rademacher")

# Both operators read rate as a relative standard deviation and apply it per
# sample, so a shared rate is already a matched Sigma and no sigma matching step
# is needed between them. The sweep is over rates only to see the whole curve.
RATES = (0.1, 0.2, 0.3, 0.5, 0.8)

# Pass counts for the downstream comparison. The predicted advantage shrinks as
# 1 / k, so it should be visible at 2 and gone by 16.
PASS_COUNTS = (2, 4, 8, 16)

# Independent single pass estimates used to measure the estimator's own spread.
VARIANCE_REPEATS = 24


def probe_context(model, architecture, position, operator, rate):
    """Attach one probe and return its handles, for use in a try block."""
    handles = plug_dropout(
        model, architecture, (position,), {position: build_perturbation(operator)}, rate
    )
    return handles


def single_pass_estimates(
    model, loader, baseline_labels, baseline_probs, device, repeats
):
    """One PSU vector per repeat, each from an independent single pass, (repeats, N).

    Each row is an independent 1 probe Hutchinson estimate of the same quantity,
    so the spread across rows is the estimator's variance and nothing else. The
    seed is varied per repeat, which is the only thing that differs between them.
    """
    estimates = []
    for repeat in range(repeats):
        probs, _argmax = compute_dropout_pass_probs(
            model, loader, baseline_labels, device, 1, True, seed=repeat
        )
        estimates.append(psu_from_cache(baseline_probs, baseline_labels, probs))

    stacked = torch.stack(estimates)  # (repeats, N)
    return stacked


def measure_checkpoint(folder, position, samples, checkpoints_dir):
    """Both comparisons for one checkpoint, as a list of flat rows."""
    checkpoint = os.path.join(checkpoints_dir, folder, "attack_result.pt")
    loaders, _manifest = build_psbd_loaders_from_checkpoint(
        checkpoint, max_samples=samples
    )
    architecture = detect_architecture(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(architecture, checkpoint, device)

    baselines = {
        name: build_baseline_cache(model, loader, device, True)
        for name, loader in loaders.items()
    }
    probs = {n: torch.cat([b["probs"] for b in c]) for n, c in baselines.items()}
    labels = {n: torch.cat([b["labels"] for b in c]) for n, c in baselines.items()}

    rows = []
    for rate in RATES:
        for operator in COMPARED_OPERATORS:
            handles = probe_context(model, architecture, position, operator, rate)
            try:
                model.train()

                # 1. The estimator claim, on the clean split.
                repeated = single_pass_estimates(
                    model,
                    loaders["clean"],
                    labels["clean"],
                    probs["clean"],
                    device,
                    VARIANCE_REPEATS,
                )
                # Spread across repeats for a fixed sample, then averaged over
                # samples. Averaging the variances rather than varying everything
                # at once keeps per-sample differences out of the number.
                estimator_variance = repeated.var(dim=0, unbiased=True).mean().item()
                estimator_mean = repeated.mean().item()

                # 2. The downstream claim, at each pass count.
                for passes in PASS_COUNTS:
                    scored = {}
                    for name, loader in loaders.items():
                        pass_probs, argmax = compute_dropout_pass_probs(
                            model, loader, labels[name], device, passes, True, seed=0
                        )
                        scored[name] = {
                            "psu": psu_from_cache(
                                probs[name], labels[name], pass_probs
                            ),
                            "sigma": shift_ratio(labels[name], argmax),
                        }
                    report = detection_report(
                        scored["validation"]["psu"],
                        scored["clean"]["psu"],
                        scored["backdoor"]["psu"],
                        HEADLINE_QUANTILE,
                    )
                    rows.append(
                        {
                            "folder": folder,
                            "architecture": architecture,
                            "position": position,
                            "operator": operator,
                            "rate": rate,
                            "passes": passes,
                            "sigma_validation": scored["validation"]["sigma"],
                            "auroc": report["auroc"],
                            "tpr": report["tpr"],
                            "fpr": report["fpr"],
                            "estimator_variance": estimator_variance,
                            "estimator_mean": estimator_mean,
                        }
                    )
            finally:
                model.eval()
                unplug_dropout(handles)
        print(f"  {folder} rate {rate} done")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folders",
        nargs="+",
        default=[
            "vit_cifar100_badnet_a2o_0_01",
            "vit_cifar100_blend_0_01",
            "swin_cifar100_badnet_a2o_0_01",
        ],
    )
    parser.add_argument("--position", default="before_attention_norm")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument(
        "--output", default="experiments/rademacher_probe_variance/probe_variance.csv"
    )
    arguments = parser.parse_args()

    rows = []
    for folder in arguments.folders:
        print(f"measuring {folder}")
        rows.extend(
            measure_checkpoint(
                folder, arguments.position, arguments.samples, arguments.checkpoints_dir
            )
        )

    measured = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    measured.to_csv(arguments.output, index=False)
    print(f"wrote {arguments.output} with {len(measured)} rows")


if __name__ == "__main__":
    main()
