"""One table: PSBD against the baselines, at deployable false-positive rates.

Reads results/<folder>/psbd_metrics.json and results/<folder>/baseline_metrics.json,
which were produced on the identical split, threshold rule and quantile grid, so the
only difference between the columns is the score being thresholded.

The 25th-percentile operating point every PSBD paper reports discards a quarter of
clean data. This defaults to 1% and 5% instead, because that is the range a defender
would actually run at, and because a method can look strong at 25% FPR and collapse
at 1%.

PSBD is reported at its best placement per checkpoint, chosen by TPR at the tightest
FPR shown. That is generous to PSBD and is stated as such: the baselines have no
placement to choose, so this comparison flatters the method under study rather than
the alternatives.

Example
    python -m cli.compare_detectors
    python -m cli.compare_detectors --fpr 0.01 --poison-rate 0.01
"""

import argparse
import glob
import json
import os

DETECTORS = ("psbd", "strip", "confidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--fpr", nargs="*", type=float, default=[0.01, 0.05])
    parser.add_argument("--poison-rate", type=float, default=None)
    return parser.parse_args()


def read_json(path: str) -> dict | None:
    """A JSON file, or None when it does not exist."""
    if not os.path.exists(path):
        return None

    with open(path) as handle:
        return json.load(handle)


def best_psbd(report: dict, key: str) -> tuple[str, dict] | None:
    """The placement and rate with the highest TPR at the given quantile.

    Searched over placements and rates, which is a selection the baselines do not get
    to make. Reported anyway because the alternative, fixing one placement, would
    understate PSBD. The asymmetry is called out in the header instead of hidden.
    """
    best = None
    for placement, block in report.get("placements", {}).items():
        for row in block.get("rates", []):
            detection = row.get("detection", {}).get(key)
            if detection is None:
                continue
            if best is None or detection["tpr"] > best[1]["tpr"]:
                best = (f"{placement}@{row['rate']:g}", detection)

    return best


def cell(value: float | None) -> str:
    """A table cell, with a placeholder for a detector that produced no number."""
    return "--" if value is None else f"{value:.3f}"


def build_header(target_fprs: list[float]) -> str:
    """The fixed-width header, one column triple per target FPR."""
    header = f"{'attack':16} {'pr':>5} {'ASR':>5}"
    for value in target_fprs:
        header += (
            f" | {f'PSBD@{value:.0%}':>10} {f'STRIP@{value:.0%}':>11} "
            f"{f'conf@{value:.0%}':>10}"
        )
    return header


def tpr_by_detector(psbd: dict, baseline: dict, key: str) -> dict[str, float | None]:
    """Each detector's TPR at one quantile, with PSBD at its best placement."""
    chosen = best_psbd(psbd, key)
    strip = baseline["detectors"].get("strip", {}).get(key)
    confidence = baseline["detectors"].get("confidence", {}).get(key)

    values = {
        "psbd": None if chosen is None else chosen[1]["tpr"],
        "strip": None if strip is None else strip["tpr"],
        "confidence": None if confidence is None else confidence["tpr"],
    }
    return values


def main() -> None:
    args = parse_args()
    keys = [f"q{value:.2f}" for value in args.fpr]

    print("Detector comparison at deployable false-positive rates.")
    print("Identical splits, identical clean-validation threshold rule, identical")
    print("quantile grid. PSBD is shown at its best placement and rate per")
    print("checkpoint. The baselines have no such choice to make.\n")

    header = build_header(args.fpr)
    print(header)
    print("-" * len(header))

    totals = {name: {key: [] for key in keys} for name in DETECTORS}

    for path in sorted(
        glob.glob(os.path.join(args.results_dir, "*", "psbd_metrics.json"))
    ):
        folder = os.path.basename(os.path.dirname(path))
        if "sam_rho" in folder:
            continue

        psbd = read_json(path)
        baseline = read_json(
            os.path.join(args.results_dir, folder, "baseline_metrics.json")
        )
        if psbd is None or baseline is None:
            continue

        meta = (
            read_json(os.path.join(args.checkpoints_dir, folder, "metrics.json")) or {}
        )
        if args.poison_rate is not None and meta.get("poison_rate") != args.poison_rate:
            continue

        attack = psbd.get("attack") or "benign"
        asr = meta.get("asr")
        line = (
            f"{attack:16} {psbd.get('poison_rate') or 0:>5.3f} "
            f"{'--' if asr is None else f'{asr:.2f}':>5}"
        )
        for key in keys:
            values = tpr_by_detector(psbd, baseline, key)
            if attack != "benign":
                for name, value in values.items():
                    if value is not None:
                        totals[name][key].append(value)
            line += (
                f" | {cell(values['psbd']):>10} {cell(values['strip']):>11} "
                f"{cell(values['confidence']):>10}"
            )
        print(line)

    print("\nmean TPR over backdoored checkpoints:")
    for name in DETECTORS:
        cells = []
        for key, target in zip(keys, args.fpr):
            values = totals[name][key]
            cells.append(
                f"{target:.0%}: {sum(values) / len(values):.3f}"
                if values
                else f"{target:.0%}: --"
            )
        print(f"  {name:12} " + "   ".join(cells))


if __name__ == "__main__":
    main()
