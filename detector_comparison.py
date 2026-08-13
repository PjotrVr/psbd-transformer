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
    python detector_comparison.py
    python detector_comparison.py --fpr 0.01 --poison-rate 0.01
"""

import argparse
import glob
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--fpr", nargs="*", type=float, default=[0.01, 0.05])
    parser.add_argument("--poison-rate", type=float, default=None)
    return parser.parse_args()


def read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def best_psbd(report: dict, key: str) -> tuple[str, dict] | None:
    """The placement and rate with the highest TPR at the given quantile.

    Searched over placements and rates, which is a selection the baselines do not get
    to make. Reported anyway because the alternative, fixing one placement, would
    understate PSBD; the asymmetry is called out in the header instead of hidden.
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


def main() -> None:
    args = parse_args()
    keys = [f"q{value:.2f}" for value in args.fpr]

    print("Detector comparison at deployable false-positive rates.")
    print("Identical splits, identical clean-validation threshold rule, identical")
    print("quantile grid. PSBD is shown at its best placement and rate per")
    print("checkpoint; the baselines have no such choice to make.\n")

    header = f"{'attack':16} {'pr':>5} {'ASR':>5}"
    for value in args.fpr:
        header += (
            f" | {f'PSBD@{value:.0%}':>10} {f'STRIP@{value:.0%}':>11} "
            f"{f'conf@{value:.0%}':>10}"
        )
    print(header)
    print("-" * len(header))

    totals = {
        name: {key: [] for key in keys} for name in ("psbd", "strip", "confidence")
    }

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
            chosen = best_psbd(psbd, key)
            strip = baseline["detectors"].get("strip", {}).get(key)
            conf = baseline["detectors"].get("confidence", {}).get(key)
            values = {
                "psbd": None if chosen is None else chosen[1]["tpr"],
                "strip": None if strip is None else strip["tpr"],
                "confidence": None if conf is None else conf["tpr"],
            }
            if attack != "benign":
                for name, value in values.items():
                    if value is not None:
                        totals[name][key].append(value)
            line += (
                f" | {_cell(values['psbd']):>10} {_cell(values['strip']):>11} "
                f"{_cell(values['confidence']):>10}"
            )
        print(line)

    print("\nmean TPR over backdoored checkpoints:")
    for name in ("psbd", "strip", "confidence"):
        cells = []
        for key, value in args.fpr and zip(keys, args.fpr) or []:
            values = totals[name][key]
            cells.append(
                f"{value:.0%}: {sum(values) / len(values):.3f}"
                if values
                else f"{value:.0%}: --"
            )
        print(f"  {name:12} " + "   ".join(cells))


def _cell(value) -> str:
    return "--" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
